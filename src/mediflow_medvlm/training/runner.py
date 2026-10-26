"""Finite single-GPU LoRA/QLoRA training with atomic, strict-resumable state."""
from __future__ import annotations

from contextlib import ExitStack, nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import signal
import subprocess
import sys
import time
import uuid

from ..runtime import (BudgetExceeded, BudgetLedger, GPULease, ResourceBusy, append_event,
                       atomic_json, canonical_hash, directory_digest, file_hash, utcnow, resolve_resource_policy)
from .config import assert_resume_identity, load_config, load_frozen_data, verified_checkpoint, training_limits
from .models import VISION_MARKERS, load_model, versions
from .rendering import ALLOWED_INPUT_FIELDS, RENDERER_VERSION, encode_sample


def parameter_digest(model):
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        if value.requires_grad:
            digest.update(name.encode())
            digest.update(value.detach().cpu().float().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _move(batch, model):
    return {key: value.to(model.device) if hasattr(value, "to") else value for key, value in batch.items()}


def validation_loss(model, processor, rows, config, stop_check=None):
    import torch
    model.eval()
    total, tokens, attempted, completed, overflow = 0.0, 0, 0, 0, 0
    with torch.no_grad():
        for row in balanced_validation_rows(rows, config.get("validation_samples", 32), config.get("seed", 20261023)):
            if stop_check and stop_check():
                model.train()
                return {"loss": None, "target_tokens": tokens, "attempted": attempted,
                        "completed": completed, "context_overflow": overflow, "interrupted": True}
            attempted += 1
            try:
                batch, metadata = encode_sample(row, processor, model.config, max_length=config.get("max_length", 2048), image_root=config.get("image_root"))
            except ValueError as exc:
                if str(exc) != "CONTEXT_OVERFLOW":
                    raise
                overflow += 1
                continue
            loss = float(model(**_move(batch, model)).loss.detach().float())
            if not math.isfinite(loss):
                raise ValueError("NONFINITE_VALIDATION_LOSS")
            count = metadata["supervised_tokens_after_shift"]
            total += loss * count
            tokens += count
            completed += 1
    model.train()
    if not tokens:
        raise ValueError("validation has no supervised tokens within context budget")
    return {"loss": total / tokens, "target_tokens": tokens, "attempted": attempted, "completed": completed, "context_overflow": overflow}


def balanced_validation_rows(rows, count=32, seed=20261023):
    """Source-balanced, frozen deterministic selection; never reads test."""
    groups = {}
    for row in rows:
        groups.setdefault(row["source_id"], []).append(row)
    for source in groups:
        groups[source].sort(key=lambda row: canonical_hash([seed, row["sample_id"]]))
    result = []
    position = 0
    while len(result) < count:
        candidates = [groups[source][position] for source in sorted(groups) if position < len(groups[source])]
        if not candidates:
            break
        result.extend(candidates[:count - len(result)])
        position += 1
    return result


def save_checkpoint(output, model, processor, optimizer, scheduler, scaler, state, identity):
    import torch
    root = Path(output) / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    destination = root / (f"step-{state['optimizer_steps']:06d}-" + uuid.uuid4().hex[:8])
    if state.get("best_checkpoint") == "__this__":
        state["best_checkpoint"] = str(destination)
    pending = root / (".pending-" + uuid.uuid4().hex)
    pending.mkdir()
    model.save_pretrained(pending, safe_serialization=True)
    processor.save_pretrained(pending)
    torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "python_rng": random.getstate(), "state": state}, pending / "training_state.pt")
    info = {"complete": True, "created_at": utcnow(), "identity": identity, **state,
            "artifacts": directory_digest(pending)}
    atomic_json(pending / "checkpoint.json", info)
    os.replace(pending, destination)
    atomic_json(Path(output) / "latest.json", {"checkpoint": str(destination), "optimizer_steps": state["optimizer_steps"]})
    return destination


def _git_identity():
    root = Path(__file__).resolve().parents[3]
    files = {str(path.relative_to(root)): file_hash(path)
             for directory in (root / "src", root / "configs")
             for path in sorted(directory.rglob("*"))
             if path.is_file() and path.suffix in {".py", ".json", ".yaml", ".yml"}}
    source = {"source_tree_sha256": canonical_hash(files), "source_files": files,
              "training_code_sha256": canonical_hash({name: digest for name, digest in files.items()
                                                       if "/training/" in name or name.endswith("/runtime.py")})}
    try:
        return {**source, "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip(),
                "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL).strip())}
    except (subprocess.SubprocessError, OSError):
        return {**source, "commit": None, "dirty": None}


def train(config_path, resume=None, *, legacy_source_snapshot=None):
    config = load_config(config_path)
    resource_policy = resolve_resource_policy(config)
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / "config.json"
    digest = canonical_hash(config)
    if snapshot.exists() and canonical_hash(json.loads(snapshot.read_text())) != digest:
        raise ValueError("existing run config is immutable; create a new run")
    manifest_path = output / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if previous and not resume and (previous.get("optimizer_steps", 0) or previous.get("status") == "SUCCEEDED"):
        raise ValueError("run has prior training; use strict resume or a new run_id")
    attempt_id = uuid.uuid4().hex
    attempt_path = output / "attempts" / (attempt_id + ".json")
    if previous:
        # Preserve prior outcomes even when a later strict resume is rejected.
        atomic_json(output / "attempts" / ("previous-" + canonical_hash(previous) + ".json"), previous)
    atomic_json(snapshot, config)
    state = {"schema_version": "medvlm.run.v1", "run_id": config["run_id"], "pair_id": config["pair_id"],
             "experiment_id": config["experiment_id"], "started_at": utcnow(), "finished_at": None,
             "config_hash": digest, "model_id": config["model_id"], "revision": config["revision"],
             "code": _git_identity(), "versions": versions(), "host": {"os": platform.system(), "architecture": platform.machine()},
             "status": "RUNNING", "evidence_level": "IMPLEMENTED_ONLY", "reason_code": None,
             "optimizer_steps": 0, "microsteps": 0, "samples_processed": 0, "supervised_tokens": 0,
             "renderer_version": RENDERER_VERSION, "allowed_input_fields": sorted(ALLOWED_INPUT_FIELDS),
             "test_used_for_selection": False, "adapter": None, "best_checkpoint": None,
             "latest_checkpoint": None, "resume_from": str(resume) if resume else None,
             "effective_batch": config.get("gradient_accumulation_steps", 16), "training_pid": os.getpid(),
             "energy_joules": None, "gpu_seconds": 0.0,
             "attempt_id": attempt_id, "resume_identity_verified": False,
             "resource_policy": resource_policy.as_dict()}
    atomic_json(attempt_path, state)
    if not resume or previous is None:
        atomic_json(manifest_path, state)
    ledger = None
    started = time.monotonic()
    prior_wall = prior_attempt_wall_seconds(output, attempt_id, previous)
    state["started_at"] = previous.get("started_at", state["started_at"]) if previous else state["started_at"]
    state["attempt_started_at"] = utcnow()
    stop_control = {"requested": False, "deadline": resource_policy.deadline(started, prior_wall)}
    original_handlers = {sig: signal.signal(sig, lambda signum, frame: stop_control.update(requested=True))
                         for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        rows, data_hashes = load_frozen_data(config)
        state["data"] = data_hashes
        resume_info = verified_checkpoint(resume) if resume else None
        if resume_info:
            old_identity = resume_info["identity"]
            for key, value in {"config_sha256": digest, "model_id": config["model_id"],
                               "revision": config["revision"], "data": data_hashes}.items():
                if old_identity.get(key) != value:
                    raise ValueError("resume preflight identity mismatch: " + key)
            prior_wall = max(prior_wall, resume_info.get("timing_snapshot", {}).get("cumulative_wall_seconds", 0.0))
            stop_control["deadline"] = resource_policy.deadline(started, prior_wall)
        import torch
        if config.get("device", "cuda") == "cuda" and not torch.cuda.is_available():
            state.update(status="BLOCKED_HARDWARE", reason_code="GPU_NOT_AVAILABLE", evidence_level="CPU_TESTED")
            return state
        torch.set_num_threads(config.get("cpu_threads", 4))
        seed = config.get("seed", 20261023)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.cuda.reset_peak_memory_stats()
            properties = torch.cuda.get_device_properties(0)
            state["gpu"] = {"name": properties.name, "total_memory_bytes": properties.total_memory,
                            "cuda_runtime": torch.version.cuda, "bf16_supported": torch.cuda.is_bf16_supported()}
        runtime = Path(config["data_root"]) / "runtime"
        is_gpu = config.get("device", "cuda") == "cuda"
        with ExitStack() as stack:
            if is_gpu:
                stack.enter_context(GPULease(runtime / "medvlm_gpu.lock", config["run_id"]))
            if is_gpu:
                ledger = BudgetLedger(runtime / "gpu_budget.json", policy=resource_policy)
                if resume_info:
                    saved_budget = resume_info.get("budget_snapshot")
                    previous_budget = (previous or {}).get("budget_snapshot")
                    if saved_budget or previous_budget:
                        if saved_budget:
                            state["budget_recovery"] = ledger.recover(saved_budget)
                        if previous_budget:
                            state["budget_recovery_previous_manifest"] = ledger.recover(previous_budget)
                    elif ledger.snapshot()["models"].get(config["model_id"], 0.0) <= 0:
                        raise ValueError("LEGACY_BUDGET_LEDGER_REQUIRED: preserve the original shared ledger")
                    else:
                        state["budget_recovery"] = "legacy checkpoint; existing nonzero shared ledger retained"
                ledger.start(config["model_id"], config["run_id"], "train")
                stack.callback(ledger.finish)
            state = _train_locked(config, output, state, rows, data_hashes, ledger, resume,
                                  stop_control=stop_control, attempt_started=started, prior_wall=prior_wall,
                                  legacy_source_snapshot=legacy_source_snapshot, previous_manifest=previous)
    except StopBeforeTraining as exc:
        state.update(status="SKIPPED_BUDGET", reason_code=str(exc))
    except ResourceBusy:
        state.update(status="BLOCKED_HARDWARE", reason_code="RESOURCE_BUSY")
    except BudgetExceeded:
        state.update(status="SKIPPED_BUDGET", reason_code="GPU_BUDGET_EXHAUSTED")
    except Exception as exc:
        if "out of memory" in str(exc).lower():
            status, reason = "BLOCKED_HARDWARE", "OOM"
        elif "GPU_NOT_AVAILABLE" in str(exc):
            status, reason = "BLOCKED_HARDWARE", "GPU_NOT_AVAILABLE"
        elif "401" in str(exc) or "403" in str(exc) or "gated" in str(exc).lower():
            status, reason = "BLOCKED_ACCESS", "AUTH_REQUIRED"
        else:
            status, reason = "FAILED_REPRODUCIBLE", type(exc).__name__
        state.update(status=status, reason_code=reason, error=str(exc)[:1000])
        append_event(output / "events.jsonl", {"event": "failure", "reason_code": reason, "error": str(exc)[:1000]})
    finally:
        for sig, handler in original_handlers.items():
            signal.signal(sig, handler)
        state["finished_at"] = utcnow()
        state["attempt_wall_seconds"] = time.monotonic() - started
        state["wall_seconds"] = prior_wall + state["attempt_wall_seconds"]
        state["time_scope"] = "cumulative completed attempts; checkpoint lower bound after crashes; GPU occupancy from shared ledger"
        state["attempt_gpu_seconds"] = ledger.session_seconds if ledger else 0.0
        if ledger:
            usage = ledger.snapshot()
            phases = usage["run_phases"].get(config["run_id"], {})
            state["gpu_seconds"] = phases.get("train", 0.0)
            state["run_gpu_seconds_all_phases"] = sum(phases.values())
            state["model_gpu_seconds_all_runs"] = usage["models"].get(config["model_id"], 0.0)
            state["budget_snapshot"] = usage
        atomic_json(attempt_path, state)
        if not resume or previous is None or state["resume_identity_verified"]:
            atomic_json(manifest_path, state)
    return state


class StopBeforeTraining(RuntimeError):
    pass


def prior_attempt_wall_seconds(output, current_id, previous):
    unique = {}
    for path in (Path(output) / "attempts").glob("*.json"):
        item = json.loads(path.read_text())
        if item.get("attempt_id") == current_id or not item.get("finished_at"):
            continue
        key = item.get("attempt_id") or "legacy-" + canonical_hash(item)
        unique[key] = item.get("attempt_wall_seconds", item.get("wall_seconds", 0.0))
    return max(sum(unique.values()), (previous or {}).get("wall_seconds", 0.0))


def capture_rng():
    import torch
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(snapshot):
    import torch
    random.setstate(snapshot["python"])
    torch.set_rng_state(snapshot["torch"])
    if snapshot["cuda"]:
        torch.cuda.set_rng_state_all(snapshot["cuda"])


def _train_locked(config, output, state, rows, data_hashes, ledger, resume, *, stop_control,
                  attempt_started, prior_wall, legacy_source_snapshot=None, previous_manifest=None):
    import torch
    from transformers import get_linear_schedule_with_warmup
    def should_stop():
        return (stop_control["requested"] or
                (stop_control["deadline"] is not None and time.monotonic() >= stop_control["deadline"]) or
                (ledger is not None and not ledger.can_continue(config["model_id"], config.get("checkpoint_reserve_seconds", 180))))
    def mark_checkpoint(train_state):
        train_state["timing_snapshot"] = {"cumulative_wall_seconds": prior_wall + time.monotonic() - attempt_started}
        if ledger:
            train_state["budget_snapshot"] = ledger.snapshot()
    if should_stop():
        raise StopBeforeTraining("INTERRUPTED" if stop_control["requested"] else "BUDGET_OR_WALL_LIMIT")
    model, processor, model_metadata = load_model(config, training=True, adapter_path=resume)
    state["model"] = model_metadata
    identity = {"identity_schema_version": "medvlm.training-identity.v2",
                "config_sha256": state["config_hash"], "data": data_hashes,
                "training_code_sha256": state["code"]["training_code_sha256"],
                "model_id": config["model_id"], "revision": config["revision"],
                "base_files_sha256": model_metadata["native_files"]["sha256"],
                "template_sha256": model_metadata["chat_template_sha256"], "renderer_version": RENDERER_VERSION,
                "target_modules": model_metadata["exact_target_modules"], "quantization": model_metadata["quantization"]}
    state["identity"] = identity
    accumulation = int(config.get("gradient_accumulation_steps", 16))
    limits = training_limits(config, len(rows["train"]), any(row["task_type"] == "medical_vqa" for row in rows["train"]))
    maximum = limits["max_optimizer_steps"]
    state["declared_training_limits"] = limits
    order = list(range(len(rows["train"])))
    random.Random(config.get("seed", 20261023)).shuffle(order)
    order = order[:limits["train_samples_cap"]]
    usable, rejected = [], []
    offered = 0
    # Validate actual template boundaries before any optimizer update; overflows
    # are excluded intact and their sample IDs are recorded without clinical text.
    with (output / "mask_audit.jsonl").open("w", encoding="utf-8") as mask_log:
        for index in order:
            if should_stop():
                raise StopBeforeTraining("INTERRUPTED" if stop_control["requested"] else "BUDGET_OR_WALL_LIMIT")
            offered += 1
            try:
                _, metadata = encode_sample(rows["train"][index], processor, model.config,
                                             max_length=config.get("max_length", 2048), image_root=config.get("image_root"))
                usable.append(index)
                mask_log.write(json.dumps(metadata) + "\n")
            except ValueError as exc:
                if str(exc) != "CONTEXT_OVERFLOW":
                    raise
                rejected.append({"sample_id": rows["train"][index]["sample_id"], "reason": "CONTEXT_OVERFLOW"})
            if ledger and not ledger.can_continue(config["model_id"], 120):
                raise BudgetExceeded("insufficient GPU budget after preflight")
            if len(usable) >= maximum * accumulation:
                break
    atomic_json(output / "excluded.json", rejected)
    if not usable:
        raise ValueError("NO_TRAIN_SAMPLES_WITHIN_CONTEXT_BUDGET")
    state["training_selection"] = {"offered": offered, "usable": len(usable), "excluded": len(rejected),
                                   "sample_order_sha256": canonical_hash([rows["train"][i]["sample_id"] for i in usable])}
    validation_selection = balanced_validation_rows(rows["validation"], config.get("validation_samples", 32), config.get("seed", 20261023))
    state["validation_selection"] = {"policy": "source_balanced_round_robin_seeded_hash_v1", "sample_ids": [row["sample_id"] for row in validation_selection],
                                     "source_counts": {source: sum(row["source_id"] == source for row in validation_selection) for source in sorted({row["source_id"] for row in validation_selection})}}
    updates = min(maximum, math.ceil(len(usable) / accumulation))
    if config.get("mode") == "smoke" and updates < min(10, maximum) and not config.get("synthetic_fixture"):
        raise ValueError("insufficient valid samples for 10 smoke optimizer updates")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config.get("learning_rate", 2e-4), weight_decay=config.get("weight_decay", 0.0))
    scheduler = get_linear_schedule_with_warmup(optimizer, min(config.get("warmup_steps", 0), updates), updates)
    use_scaler = model_metadata["dtype"] == "float16" and config.get("device", "cuda") == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    train_state = {"optimizer_steps": 0, "microsteps": 0, "samples_processed": 0, "supervised_tokens": 0,
                   "next_sample_position": 0, "sample_order_sha256": state["training_selection"]["sample_order_sha256"],
                   "best_validation_loss": float("inf"), "best_checkpoint": None}
    initial_digest = parameter_digest(model)
    state["adapter_parameters_before_sha256"] = initial_digest
    if resume:
        previous_provenance = (previous_manifest or {}).get("legacy_identity_provenance", previous_manifest)
        checkpoint = assert_resume_identity(identity, resume, legacy_source_snapshot=legacy_source_snapshot,
                                            legacy_manifest=previous_provenance)
        if checkpoint.get("identity_migration"):
            state["resume_identity_migration"] = checkpoint["identity_migration"]
            state["legacy_identity_provenance"] = {"identity": previous_provenance["identity"],
                                                   "model": {"native_files": previous_provenance["model"]["native_files"]}}
            state["legacy_source_snapshot"] = str(legacy_source_snapshot)
            append_event(output / "events.jsonl", {"event": "legacy_identity_migration", **checkpoint["identity_migration"]})
        elif previous_manifest and previous_manifest.get("legacy_identity_provenance"):
            # Retain verified provenance when an unchanged v2 latest checkpoint
            # still points at a selected best checkpoint from the legacy run.
            state["legacy_identity_provenance"] = previous_manifest["legacy_identity_provenance"]
            state["legacy_source_snapshot"] = previous_manifest["legacy_source_snapshot"]
        for relative, digest in checkpoint["artifacts"]["files"].items():
            if file_hash(Path(resume) / relative) != digest:
                raise ValueError("checkpoint artifact hash mismatch: " + relative)
        # Only locally generated, digest-verified state files are accepted here.
        restored = torch.load(Path(resume) / "training_state.pt", map_location="cpu", weights_only=False)
        train_state = restored["state"]
        if train_state["sample_order_sha256"] != state["training_selection"]["sample_order_sha256"]:
            raise ValueError("resume sampler ordering mismatch")
        optimizer.load_state_dict(restored["optimizer"])
        scheduler.load_state_dict(restored["scheduler"])
        scaler.load_state_dict(restored["scaler"])
        torch.set_rng_state(restored["torch_rng"])
        random.setstate(restored["python_rng"])
        if restored["cuda_rng"]:
            torch.cuda.set_rng_state_all(restored["cuda_rng"])
        state["resume_optimizer_restored"] = True
        state["resume_identity_verified"] = True
        state["best_checkpoint"] = train_state.get("best_checkpoint")
        state["latest_checkpoint"] = str(resume)
        state.update({key: train_state[key] for key in ("optimizer_steps", "microsteps", "samples_processed", "supervised_tokens")})
    baseline = validation_loss(model, processor, rows["validation"], config, stop_check=should_stop)
    if baseline.get("interrupted"):
        raise StopBeforeTraining("INTERRUPTED" if stop_control["requested"] else "BUDGET_OR_WALL_LIMIT")
    state["initial_validation"] = baseline
    append_event(output / "events.jsonl", {"event": "validation_before_updates", **baseline})
    stopped = False
    try:
        while train_state["optimizer_steps"] < updates and train_state["next_sample_position"] < len(usable):
            if should_stop():
                stopped = True
                break
            start_pos = train_state["next_sample_position"]
            group = usable[start_pos:start_pos + accumulation]
            optimizer.zero_grad(set_to_none=True)
            loss_sum, target_count = 0.0, 0
            before = time.monotonic()
            group_rng = capture_rng()
            group_scaler = scaler.state_dict()
            partial_microsteps = 0
            interrupted_group = False
            for index in group:
                if should_stop():
                    interrupted_group = True
                    break
                batch, metadata = encode_sample(rows["train"][index], processor, model.config,
                                                 max_length=config.get("max_length", 2048), image_root=config.get("image_root"))
                batch = _move(batch, model)
                use_autocast = config.get("device", "cuda") == "cuda"
                with torch.autocast("cuda", dtype=getattr(torch, model_metadata["dtype"]), enabled=use_autocast):
                    loss = model(**batch).loss
                if not torch.isfinite(loss):
                    raise ValueError("NONFINITE_TRAINING_LOSS")
                scaler.scale(loss / len(group)).backward()
                loss_sum += float(loss.detach().float())
                target_count += metadata["supervised_tokens_after_shift"]
                partial_microsteps += 1
                if should_stop():
                    interrupted_group = True
                    break
            if interrupted_group:
                optimizer.zero_grad(set_to_none=True)
                scaler.load_state_dict(group_scaler)
                restore_rng(group_rng)
                stopped = True
                append_event(output / "events.jsonl", {"event": "discarded_incomplete_optimizer_group",
                                                        "microsteps": partial_microsteps,
                                                        "next_sample_position": train_state["next_sample_position"]})
                break
            scaler.unscale_(optimizer)
            trainable = [p for p in model.parameters() if p.requires_grad]
            if not any(p.grad is not None and bool(torch.any(p.grad != 0)) for p in trainable):
                raise ValueError("ZERO_ADAPTER_GRADIENT")
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in trainable):
                raise ValueError("NONFINITE_ADAPTER_GRADIENT")
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad and parameter.grad is not None:
                    raise ValueError("FROZEN_BASE_HAS_GRADIENT: " + name)
            norm = torch.nn.utils.clip_grad_norm_(trainable, config.get("max_grad_norm", 1.0))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_state["optimizer_steps"] += 1
            train_state["microsteps"] += len(group)
            train_state["samples_processed"] += len(group)
            train_state["supervised_tokens"] += target_count
            train_state["next_sample_position"] += len(group)
            state.update({key: train_state[key] for key in ("optimizer_steps", "microsteps", "samples_processed", "supervised_tokens")})
            event = {"event": "optimizer_update", "step": train_state["optimizer_steps"], "loss": loss_sum / len(group),
                     "gradient_norm": float(norm), "target_tokens": target_count, "samples": len(group),
                     "seconds": time.monotonic() - before, "lr": scheduler.get_last_lr()[0]}
            append_event(output / "events.jsonl", event)
            print(json.dumps(event), flush=True)
            interval = config.get("checkpoint_every_steps", 100)
            if train_state["optimizer_steps"] % interval == 0 or train_state["optimizer_steps"] == updates:
                checked = validation_loss(model, processor, rows["validation"], config, stop_check=should_stop)
                is_best = not checked.get("interrupted") and checked["loss"] < train_state["best_validation_loss"]
                if is_best:
                    train_state["best_validation_loss"] = checked["loss"]
                    train_state["best_checkpoint"] = "__this__"
                mark_checkpoint(train_state)
                checkpoint_path = save_checkpoint(output, model, processor, optimizer, scheduler, scaler, train_state, identity)
                state["latest_checkpoint"] = str(checkpoint_path)
                if is_best:
                    state["best_checkpoint"] = str(checkpoint_path)
                    train_state["best_checkpoint"] = str(checkpoint_path)
                    atomic_json(output / "best.json", {"checkpoint": str(checkpoint_path), "validation": checked})
                append_event(output / "events.jsonl", {"event": "checkpoint", "step": train_state["optimizer_steps"], "validation": checked})
                atomic_json(output / "manifest.json", state)
                # Keep every previous run untouched; prune only this run's own
                # fully completed, unselected recovery checkpoints.
                keep = {state.get("latest_checkpoint"), state.get("best_checkpoint")}
                for old in (output / "checkpoints").glob("step-*"):
                    if str(old) not in keep and (old / "checkpoint.json").exists():
                        shutil.rmtree(old)
                if checked.get("interrupted"):
                    stopped = True
                    break
        if stopped:
            mark_checkpoint(train_state)
            checkpoint_path = save_checkpoint(output, model, processor, optimizer, scheduler, scaler, train_state, identity)
            state["latest_checkpoint"] = str(checkpoint_path)
            state.update(status="SKIPPED_BUDGET", reason_code="INTERRUPTED" if stop_control["requested"] else "BUDGET_OR_WALL_LIMIT")
            keep = {state.get("latest_checkpoint"), state.get("best_checkpoint")}
            for old in (output / "checkpoints").glob("step-*"):
                if str(old) not in keep and (old / "checkpoint.json").exists():
                    shutil.rmtree(old)
        else:
            state.update(status="SUCCEEDED", reason_code="ADAPTER_TRAINED_RELOAD_PENDING")
    finally:
        # This function only publishes complete optimizer groups. Signal handlers
        # remain owned by train(), including model load and preflight phases.
        pass
    state["adapter_parameters_after_sha256"] = parameter_digest(model)
    state["adapter_changed"] = initial_digest != state["adapter_parameters_after_sha256"]
    if train_state["optimizer_steps"] and not state["adapter_changed"] and not resume:
        raise ValueError("ADAPTER_PARAMETERS_UNCHANGED")
    state["evidence_level"] = "SYNTHETIC_PIPELINE_ONLY" if config.get("synthetic_fixture") else "REAL_GPU_SMOKE"
    state["actual_epochs"] = train_state["samples_processed"] / len(rows["train"])
    state["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
    state["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None
    selected = state.get("best_checkpoint") or state.get("latest_checkpoint")
    if selected:
        state["adapter"] = {"path": selected, "digest": directory_digest(selected)["sha256"], "base_revision": config["revision"]}
        atomic_json(output / "selection.json", {"run_id": config["run_id"], "pair_id": config["pair_id"],
                                                "checkpoint": selected, "selection_policy": "minimum_validation_loss",
                                                "identity": identity, "adapter": state["adapter"], "test_used": False,
                                                "frozen_at": utcnow(), "reload_verified": False})
    return state


def reload_checkpoint(config_path, checkpoint_path):
    """Invoke from a NEW process; load disk base+adapter+processor and generate."""
    import torch
    config = load_config(config_path)
    resource_policy = resolve_resource_policy(config)
    output = Path(config["output_dir"])
    manifest = json.loads((output / "manifest.json").read_text())
    if canonical_hash(config) != manifest["config_hash"]:
        raise ValueError("reload config hash differs from immutable training config")
    if manifest.get("training_pid") == os.getpid():
        raise ValueError("reload must execute in a new process after training")
    checkpoint = assert_resume_identity(manifest["identity"], checkpoint_path,
                                        legacy_source_snapshot=manifest.get("legacy_source_snapshot"),
                                        legacy_manifest=manifest.get("legacy_identity_provenance"))
    for relative, digest in checkpoint["artifacts"]["files"].items():
        if file_hash(Path(checkpoint_path) / relative) != digest:
            raise ValueError("checkpoint artifact hash mismatch")
    result = {"run_id": config["run_id"], "checkpoint": str(checkpoint_path), "pid": os.getpid(),
              "status": "RUNNING", "started_at": utcnow(), "fresh_process_required": True,
              "resource_policy": resource_policy.as_dict()}
    runtime = Path(config["data_root"]) / "runtime"
    ledger = None
    is_gpu = config.get("device", "cuda") == "cuda"
    try:
        with ExitStack() as stack:
            if is_gpu:
                stack.enter_context(GPULease(runtime / "medvlm_gpu.lock", config["run_id"] + "-reload"))
            if is_gpu:
                ledger = BudgetLedger(runtime / "gpu_budget.json", policy=resource_policy).start(config["model_id"], config["run_id"], "reload")
                stack.callback(ledger.finish)
            model, processor, metadata = load_model(config, adapter_path=checkpoint_path)
            expected_base = checkpoint["identity"].get("base_files_sha256") or manifest["model"]["native_files"]["sha256"]
            if metadata["native_files"]["sha256"] != expected_base:
                raise ValueError("BASE_FILES_DIGEST_MISMATCH: cached base differs from training")
            # The processor saved with the adapter must reproduce its parent template.
            from .models import load_processor
            saved_config = {**config, "model_id": str(checkpoint_path), "local_files_only": True}
            processor = load_processor(saved_config, model.config)
            saved_tokenizer = getattr(processor, "tokenizer", processor)
            if canonical_hash(processor.chat_template or saved_tokenizer.chat_template) != checkpoint["identity"]["template_sha256"]:
                raise ValueError("saved processor chat template differs from checkpoint identity")
            rows, data_hashes = load_frozen_data(config)
            if data_hashes != manifest["identity"]["data"]:
                raise ValueError("reload data identity changed")
            row = rows["validation"][0]
            inputs, _ = encode_sample(row, processor, model.config, max_length=config.get("max_length", 2048),
                                      image_root=config.get("image_root"), include_labels=False)
            with torch.no_grad():
                generated = model.generate(**_move(inputs, model), max_new_tokens=config.get("reload_max_new_tokens", 32),
                                           do_sample=False, use_cache=True)
            count = generated.shape[-1] - inputs["input_ids"].shape[-1]
            if count <= 0:
                raise ValueError("reload produced no generated tokens")
            tokenizer = getattr(processor, "tokenizer", processor)
            raw = tokenizer.decode(generated[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            # Restricted artifact: never print medical text to normal logs.
            atomic_json(output / "reload_output.restricted.json", {"sample_id": row["sample_id"], "generated_text": raw})
            result.update(status="SUCCEEDED", generated_tokens=count, model=metadata,
                          evidence_level="SYNTHETIC_PIPELINE_ONLY" if config.get("synthetic_fixture") else "REAL_GPU_SMOKE")
            selection = json.loads((output / "selection.json").read_text())
            if str(checkpoint_path) == selection["checkpoint"]:
                selection["reload_verified"] = True
                atomic_json(output / "selection.json", selection)
            manifest["reload_verified"] = bool(selection.get("reload_verified", False))
            if manifest["reload_verified"]:
                manifest["reason_code"] = "ADAPTER_TRAINED_AND_RELOADED"
            atomic_json(output / "manifest.json", manifest)
    except Exception as exc:
        result.update(status="FAILED_REPRODUCIBLE", reason_code=type(exc).__name__, error=str(exc)[:500])
    finally:
        result["finished_at"] = utcnow()
        atomic_json(output / "reload.json", result)
    return result
