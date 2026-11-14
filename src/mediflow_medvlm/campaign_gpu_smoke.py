"""One supervisor-owned V6 native GPU smoke, never common-freeze or main training.

Only an explicit CLI launch performs GPU work. No queue, lease, recovery counter,
model download, development/test read, or historical runner mutation lives here.
Interrupted training has no resumable optimizer/RNG checkpoint. Each authorized
retry starts deterministically from base and preserves the previous attempt.
"""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
import uuid
import traceback
import fcntl

from . import campaign_preflight as cpu
from .campaign_supervisor import atomic_json, canonical_hash, file_hash
from .campaign_v6 import cell_spec
from .runtime import utcnow, ResourcePolicy, BudgetLedger

SCHEMA = "medvlm.v6.native-gpu-smoke-config.v1"
SCOPE = "NATIVE_GPU_SMOKE_ONLY"
IDENTITY_KEYS = cpu.IDENTITY_KEYS
SOURCE_FILES = tuple(dict.fromkeys((*cpu.SOURCE_FILES, "campaign_gpu_smoke.py", "training/v6_gpu.py")))
RECIPE = {"seed": 20261023, "optimizer_steps": 20, "examples": 320,
          "microbatch": 1, "gradient_accumulation": 16, "optimizer": "AdamW",
          "weight_decay": 0.0, "betas": [0.9, 0.999], "eps": 1e-8,
          "gradient_clip_norm": 1.0, "warmup": 0, "scheduler": "linear",
          "loss": "mean_example_target_token_mean", "quantization": "nf4_double_quant",
          "lora_rank": 8, "lora_alpha": 16, "lora_dropout": 0.05}


class SmokeError(ValueError):
    """A bounded reason code safe for public stderr; native errors stay private."""


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _absolute(path):
    result = Path(path)
    if not result.is_absolute() or ".." in result.parts:
        raise SmokeError("ABSOLUTE_NORMALIZED_PATH_REQUIRED")
    return result


def source_identity(repo=None):
    package = Path(__file__).resolve().parent if repo is None else Path(repo) / "src/mediflow_medvlm"
    files = {"src/mediflow_medvlm/" + name: file_hash(package / name) for name in SOURCE_FILES}
    return files, canonical_hash(files)


def runtime_versions():
    result = cpu.runtime_versions()
    for name in ("accelerate", "peft", "bitsandbytes", "safetensors", "numpy"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _preflight(path):
    path = _absolute(path)
    receipt = _read(path)
    if receipt.get("status") != "SUCCEEDED" or receipt.get("scope") != "TRAIN_PREFLIGHT_ELIGIBILITY_ONLY":
        raise SmokeError("CPU_PREFLIGHT_NOT_SUCCEEDED")
    artifacts = receipt.get("artifacts", [])
    configs = [a for a in artifacts if Path(a["path"]).name == "config.json"]
    if len(configs) != 1:
        raise SmokeError("CPU_PREFLIGHT_CONFIG_MISSING")
    config = _read(_absolute(configs[0]["path"]))
    if config.get("schema_version") != cpu.SCHEMA or config.get("config_hash") != canonical_hash({k: v for k, v in config.items() if k != "config_hash"}):
        raise SmokeError("CPU_PREFLIGHT_CONFIG_INVALID")
    # Verify the immutable completion marker and every artifact, without claiming
    # that the old CPU snapshot is the currently imported GPU worker snapshot.
    if cpu._completed_output(config) != receipt:
        raise SmokeError("CPU_PREFLIGHT_RECEIPT_MARKER_MISMATCH")
    summary = _read(Path(config["output_dir"]) / "summary.json")
    records = [json.loads(line) for line in (Path(config["output_dir"]) / "eligibility.jsonl").read_text().splitlines()]
    if file_hash(config["data"]["train_file"]) != config["data"]["train_sha256"]:
        raise SmokeError("TRAIN_BYTES_CHANGED")
    rows = cpu._train_rows(config["data"]["train_file"], config["data"]["expected_count"], config["task"])
    if len(records) != len(rows) or summary.get("original_count") != len(rows):
        raise SmokeError("CPU_PREFLIGHT_ROW_COUNT_MISMATCH")
    eligible = []
    for index, (row, record) in enumerate(zip(rows, records)):
        if (record.get("position") != index or record.get("sample_id") != row["sample_id"]
                or record.get("input_sha256") != canonical_hash(row["input"])
                or record.get("target_sha256") != canonical_hash(row["supervision"])):
            raise SmokeError("CPU_PREFLIGHT_ROW_IDENTITY_MISMATCH")
        if record.get("eligible") is True:
            eligible.append(record)
    ids = [r["sample_id"] for r in eligible]
    if (summary.get("eligible_ids") != ids or summary.get("eligible_ids_sha256") != canonical_hash(ids)
            or summary.get("full_targets_preserved") is not True or summary.get("dev_test_read") is not False
            or summary.get("eligible_count") != len(ids) or receipt.get("eligible_count") != len(ids)):
        raise SmokeError("CPU_PREFLIGHT_ELIGIBILITY_MISMATCH")
    return config, records, rows


def _selection(preconfig, records):
    eligible = [r for r in records if r.get("eligible") is True]
    eligible.sort(key=lambda r: (canonical_hash([RECIPE["seed"], r["sample_id"]]), r["sample_id"]))
    if len(eligible) < 321:
        raise SmokeError("NEED_320_TRAIN_AND_ONE_HELD_TRAIN_EXAMPLE")
    chosen = eligible[:321]
    return {"rule": "sha256_seed_sample_id_ascending", "seed": RECIPE["seed"],
            "train_records": chosen[:320], "held_train_record": chosen[320],
            "selected_ids_sha256": canonical_hash([r["sample_id"] for r in chosen[:320]]),
            "preflight_data_hash": preconfig["data_hash"], "partition": "train"}


def build_config(cell_id, preflight_receipt, output_dir, *, repo=None, run_id=None):
    """Pure planning API: freeze a single smoke from a verified CPU receipt."""
    cell = cell_spec(cell_id)
    preconfig, records, _ = _preflight(preflight_receipt)
    if any(preconfig.get(key) != value for key, value in cell.items()):
        raise SmokeError("CPU_PREFLIGHT_CELL_MISMATCH")
    model = cpu._prepared(preconfig["model"]["preparation_receipt"])
    if model != preconfig["model"]:
        raise SmokeError("PREPARED_MODEL_CHANGED")
    output = _absolute(output_dir)
    if any(part in {"reports", "public_summary", ".git"} for part in output.parts):
        raise SmokeError("PRIVATE_OUTPUT_REQUIRED")
    files, source_hash = source_identity(repo)
    data = {**preconfig["data"], "preflight_receipt": str(_absolute(preflight_receipt)),
            "preflight_receipt_sha256": file_hash(preflight_receipt),
            "preflight_config_hash": preconfig["config_hash"], "selection": _selection(preconfig, records)}
    run_id = run_id or "v6-native-smoke-" + cell_id
    config = {"schema_version": SCHEMA, **cell, "job_id": run_id, "run_id": run_id,
              "scope": SCOPE, "max_length": 2048, "fixed_template_date": "2026-11-14",
              "device": "cuda", "local_files_only": True, "trust_remote_code": False,
              "recipe": {**RECIPE, "learning_rate": 2e-4 if cell["task"] == "text" else 1e-4},
              "model": model, "data": data, "output_dir": str(output),
              "runtime_versions": runtime_versions(), "resource_policy": ResourcePolicy().as_dict(),
              "gpu_ledger_path": str(Path(model["cache_dir"]).parent.parent / "runtime/gpu_budget.json"),
              "supervisor_lock_paths": [str(Path(model["cache_dir"]).parent.parent / "v6_v7/worker.lock"),
                                        str(Path(model["cache_dir"]).parent.parent / "runtime/medvlm_gpu.lock")],
              "source_files": files, "source_hash": source_hash,
              "model_hash": canonical_hash(model), "data_hash": canonical_hash(data)}
    config["config_hash"] = canonical_hash(config)
    return config


def _validate(config):
    if config.get("schema_version") != SCHEMA or config.get("config_hash") != canonical_hash({k: v for k, v in config.items() if k != "config_hash"}):
        raise SmokeError("CONFIG_HASH_OR_SCHEMA_MISMATCH")
    spec = cell_spec(config["cell_id"])
    if any(config.get(k) != v for k, v in spec.items()):
        raise SmokeError("CELL_IDENTITY_MISMATCH")
    expected = {**RECIPE, "learning_rate": 2e-4 if spec["task"] == "text" else 1e-4}
    if (config.get("recipe") != expected or config.get("max_length") != 2048
            or config.get("fixed_template_date") != "2026-11-14" or config.get("device") != "cuda"
            or config.get("local_files_only") is not True or config.get("trust_remote_code") is not False
            or config.get("scope") != SCOPE):
        raise SmokeError("FINITE_NATIVE_SMOKE_RECIPE_REQUIRED")
    if config.get("resource_policy") != ResourcePolicy().as_dict():
        raise SmokeError("UNLIMITED_RESOURCE_POLICY_REQUIRED")
    if config.get("gpu_ledger_path") != str(Path(config["model"]["cache_dir"]).parent.parent / "runtime/gpu_budget.json"):
        raise SmokeError("SHARED_GPU_LEDGER_REQUIRED")
    data_root = Path(config["model"]["cache_dir"]).parent.parent
    if config.get("supervisor_lock_paths") != [str(data_root / "v6_v7/worker.lock"), str(data_root / "runtime/medvlm_gpu.lock")]:
        raise SmokeError("SUPERVISOR_LOCK_PATHS_CHANGED")
    if config["runtime_versions"] != runtime_versions():
        raise SmokeError("GPU_RUNTIME_VERSIONS_CHANGED")
    if (config["source_files"], config["source_hash"]) != source_identity():
        raise SmokeError("SOURCE_IDENTITY_MISMATCH")
    if canonical_hash(config["model"]) != config["model_hash"] or cpu._prepared(config["model"]["preparation_receipt"]) != config["model"]:
        raise SmokeError("MODEL_IDENTITY_MISMATCH")
    data = config["data"]
    if canonical_hash(data) != config["data_hash"] or file_hash(data["preflight_receipt"]) != data["preflight_receipt_sha256"]:
        raise SmokeError("PREFLIGHT_OR_DATA_IDENTITY_MISMATCH")
    preconfig, records, rows = _preflight(data["preflight_receipt"])
    if (preconfig["config_hash"] != data["preflight_config_hash"] or preconfig["model"] != config["model"]
            or any(preconfig["data"].get(k) != data.get(k) for k in preconfig["data"])
            or data["selection"] != _selection(preconfig, records)):
        raise SmokeError("PREFLIGHT_SELECTION_CHANGED")
    for key in IDENTITY_KEYS:
        supplied = os.environ.get("MEDVLM_SUPERVISOR_" + key.upper())
        if supplied is not None and config[key] != supplied:
            raise SmokeError("SUPERVISOR_IDENTITY_MISMATCH_" + key.upper())
    _absolute(config["output_dir"])
    return {row["sample_id"]: row for row in rows}


def _inherited_supervisor_fds(config):
    """Keep the supervisor's exact two open-file descriptions alive in reload.

    No reacquisition and no LOCK_UN: closing an ancestor must not make a surviving
    CUDA reload orphan invisible to the single-owner supervisor.
    """
    if any(os.environ.get("MEDVLM_SUPERVISOR_" + key.upper()) != config[key] for key in IDENTITY_KEYS):
        raise SmokeError("SUPERVISOR_EXECUTION_IDENTITY_REQUIRED")
    expected = [_absolute(p) for p in config["supervisor_lock_paths"]]
    found = {}
    for entry in Path("/proc/self/fd").iterdir():
        try:
            fd = int(entry.name)
            link = Path(os.readlink(entry))
            for path in expected:
                if link == path and os.fstat(fd).st_ino == path.stat().st_ino and os.fstat(fd).st_dev == path.stat().st_dev:
                    found[str(path)] = fd
        except (OSError, ValueError):
            continue
    if len(found) != len(expected):
        raise SmokeError("SUPERVISOR_GPU_AND_WORKER_LEASES_REQUIRED")
    for path in expected:
        fd = os.open(path, os.O_RDWR)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            fcntl.flock(fd, fcntl.LOCK_UN)
            raise SmokeError("SUPERVISOR_INHERITED_LEASE_NOT_HELD")
        finally:
            os.close(fd)
    return tuple(found[str(path)] for path in expected)


def _verify_native_contract(config, contract):
    preconfig, _, _ = _preflight(config["data"]["preflight_receipt"])
    previous = _read(Path(preconfig["output_dir"]) / "architecture.json")
    if any(contract.get(key) != previous.get(key) for key in ("architecture", "template", "thinking")):
        raise SmokeError("CPU_GPU_NATIVE_CONTRACT_CHANGED")


def _encode(row, record, bundle, config):
    try:
        batch, metadata = cpu._checked_encode(row, bundle.processor, bundle.architecture,
                                              config["task"], config["data"]["image_root"])
    except Exception as exc:
        raise SmokeError("SELECTED_TRAIN_ENCODING_FAILED_NO_DROP_ALLOWED") from exc
    # This compares image SHA/order/dimensions, sequence, full target span and all
    # processor tensor shapes/dtypes with the exact CPU native eligibility proof.
    if metadata != record["native"]:
        raise SmokeError("SELECTED_NATIVE_ENCODING_CHANGED")
    return batch, metadata


def _target_loss(model, batch):
    import torch
    labels = batch["labels"]
    moved = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in batch.items()}
    with torch.autocast(device_type=model.device.type, dtype=torch.bfloat16, enabled=model.device.type == "cuda"):
        output = model(**{k: v for k, v in moved.items() if k != "labels"})
    shifted = moved["labels"][:, 1:]
    supervised = shifted != -100
    logits = output.logits[:, :-1, :][supervised].float()
    loss = torch.nn.functional.cross_entropy(logits, shifted[supervised], reduction="mean")
    if not bool(torch.isfinite(loss)):
        raise SmokeError("NONFINITE_TRAIN_LOSS")
    return loss, int((labels[:, 1:] != -100).sum())


def _progress(config, step, stage):
    path = os.environ.get("MEDVLM_SUPERVISOR_PROGRESS_PATH")
    if path:
        atomic_json(path, {**{k: config[k] for k in IDENTITY_KEYS}, "status": "RUNNING", "stage": stage,
            "optimizer_step": step, "total_updates": 20, "samples_processed": step * 16,
            "last_progress_at": utcnow(), "heartbeat_at": utcnow(), "pid": os.getpid(),
            "last_complete_checkpoint": None})


def _train(config, rows, staging):
    _inherited_supervisor_fds(config)
    import torch
    from .training.v6_gpu import (load_v6_gpu_bundle, adapter_digest,
                                  projection_gradient_report, projection_adapter_digests, assert_frozen_base)
    if not torch.cuda.is_available():
        raise SmokeError("GPU_NOT_AVAILABLE")
    random.seed(RECIPE["seed"])
    torch.manual_seed(RECIPE["seed"])
    torch.cuda.manual_seed_all(RECIPE["seed"])
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    bundle = load_v6_gpu_bundle(config["model"], config["cell_id"], fixed_template_date=config["fixed_template_date"])
    _verify_native_contract(config, bundle.contract)
    model = bundle.model
    assert_frozen_base(model)
    initial = adapter_digest(model)
    initial_projections = projection_adapter_digests(model, bundle.contract["architecture"])
    nonzero_seen = {path: {"A": False, "B": False} for path in initial_projections}
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=config["recipe"]["learning_rate"], betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(0.0, 1.0 - step / 20))
    optimizer.zero_grad(set_to_none=True)
    losses, tokens, gradients, norms, learning_rates = [], [], [], [], []
    selected = config["data"]["selection"]["train_records"]
    sequence_tokens = 0
    for step in range(20):
        _progress(config, step, "native_gpu_training")
        learning_rates.append(optimizer.param_groups[0]["lr"])
        for record in selected[step * 16:(step + 1) * 16]:
            batch, metadata = _encode(rows[record["sample_id"]], record, bundle, config)
            loss, count = _target_loss(model, batch)
            (loss / 16).backward()
            losses.append(float(loss.detach()))
            tokens.append(count)
            sequence_tokens += metadata["sequence_tokens"]
        report = projection_gradient_report(model, bundle.contract["architecture"])
        gradients.append(report)
        for projection in report["projections"]:
            for label in ("A", "B"):
                nonzero_seen[projection["path"]][label] |= projection["tensors"][label]["nonzero"]
        norm = float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0))
        if not math.isfinite(norm):
            raise SmokeError("NONFINITE_GRADIENT_NORM")
        norms.append(norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    if not all(value for tensors in nonzero_seen.values() for value in tensors.values()):
        raise SmokeError("PER_PROJECTION_NONZERO_GRADIENT_NOT_OBSERVED")
    final = adapter_digest(model)
    final_projections = projection_adapter_digests(model, bundle.contract["architecture"])
    if (not initial_projections or initial_projections.keys() != final_projections.keys()
            or any(initial_projections[path][label] == final_projections[path][label]
                   for path in initial_projections for label in ("A", "B"))):
        raise SmokeError("PER_PROJECTION_ADAPTER_UPDATE_MISSING")
    if initial == final:
        raise SmokeError("ADAPTER_UNCHANGED")
    checkpoint = staging / "adapter"
    checkpoint.mkdir(mode=0o700)
    model.save_pretrained(checkpoint, safe_serialization=True)
    processor_path = staging / "processor"
    processor_path.mkdir(mode=0o700)
    bundle.processor.save_pretrained(processor_path)
    if not any(p.name.startswith("adapter_model") and p.suffix == ".safetensors" for p in checkpoint.iterdir()):
        raise SmokeError("ADAPTER_WEIGHTS_MISSING")
    torch.cuda.synchronize()
    properties = torch.cuda.get_device_properties(0)
    metrics = {"optimizer_steps": 20, "samples_processed": len(losses), "excluded_count": 0,
        "adapter_changed": initial != final, "initial_adapter_sha256": initial, "adapter_sha256": final,
        "initial_projection_sha256": initial_projections, "final_projection_sha256": final_projections,
        "all_projection_tensors_changed": True, "nonzero_gradients_observed": nonzero_seen, "training_pid": os.getpid(), "loss_mean": sum(losses) / len(losses), "example_losses": losses,
        "supervised_tokens": sum(tokens), "example_supervised_tokens": tokens, "sequence_tokens": sequence_tokens,
        "gradient_reports": gradients, "gradient_norms_before_clip": norms, "learning_rates": learning_rates,
        "final_learning_rate": optimizer.param_groups[0]["lr"], "native_contract": bundle.contract,
        "resource": {"wall_seconds": time.monotonic() - started, "cuda_runtime": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(), "device_name": properties.name,
            "device_total_memory_bytes": properties.total_memory, "device_capability": list(torch.cuda.get_device_capability()),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "runtime_versions": runtime_versions(), "energy_joules": None}, "finished_at": utcnow()}
    atomic_json(staging / "metrics.json", metrics)
    del loss, batch, optimizer, scheduler, model, bundle
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


def _tree_files(root):
    result = {}
    for path in sorted(Path(root).rglob("*")):
        if path.is_symlink():
            raise SmokeError("OUTPUT_SYMLINK_FORBIDDEN")
        if path.is_file() and path.relative_to(root).as_posix() != "completion.json":
            result[path.relative_to(root).as_posix()] = file_hash(path)
    return result


def _checkpoint(config, staging, metrics):
    files = {p.relative_to(staging / "adapter").as_posix(): file_hash(p)
             for p in sorted((staging / "adapter").rglob("*")) if p.is_file()}
    if not files or not any(n.startswith("adapter_model") and n.endswith(".safetensors") for n in files):
        raise SmokeError("ADAPTER_WEIGHTS_MISSING")
    processor_files = {p.relative_to(staging / "processor").as_posix(): file_hash(p)
                       for p in sorted((staging / "processor").rglob("*")) if p.is_file()}
    if not processor_files:
        raise SmokeError("SAVED_PROCESSOR_MISSING")
    value = {"complete": True, **{k: config[k] for k in IDENTITY_KEYS},
             "optimizer_steps": 20, "samples_processed": 320, "adapter_sha256": metrics["adapter_sha256"],
             "adapter_files": files, "processor_files": processor_files, "resume_supported": False, "optimizer_rng_saved": False}
    atomic_json(staging / "checkpoint.json", value)
    return value


def _verify_checkpoint(config, directory):
    checkpoint = _read(directory / "checkpoint.json")
    if (checkpoint.get("complete") is not True or checkpoint.get("optimizer_steps") != 20
            or checkpoint.get("samples_processed") != 320
            or any(checkpoint.get(k) != config[k] for k in IDENTITY_KEYS)):
        raise SmokeError("CHECKPOINT_IDENTITY_MISMATCH")
    actual = {p.relative_to(directory / "adapter").as_posix(): file_hash(p)
              for p in sorted((directory / "adapter").rglob("*")) if p.is_file()}
    if actual != checkpoint.get("adapter_files") or any(p.is_symlink() for p in (directory / "adapter").rglob("*")):
        raise SmokeError("CHECKPOINT_ARTIFACT_CHANGED")
    actual_processor = {p.relative_to(directory / "processor").as_posix(): file_hash(p)
                        for p in sorted((directory / "processor").rglob("*")) if p.is_file()}
    if (not actual_processor or actual_processor != checkpoint.get("processor_files")
            or any(p.is_symlink() for p in (directory / "processor").rglob("*"))):
        raise SmokeError("SAVED_PROCESSOR_ARTIFACT_CHANGED")
    return checkpoint


def reload_checkpoint(config_path, directory, receipt_path):
    """Fresh CLI process verifies exact saved adapter and one held TRAIN forward."""
    config = _read(config_path)
    _inherited_supervisor_fds(config)
    import torch
    from .training.v6_gpu import load_v6_gpu_bundle, adapter_digest
    rows = _validate(config)
    directory = _absolute(directory)
    checkpoint = _verify_checkpoint(config, directory)
    _inherited_supervisor_fds(config)
    bundle = load_v6_gpu_bundle(config["model"], config["cell_id"], fixed_template_date=config["fixed_template_date"],
                                adapter_path=directory / "adapter", processor_path=directory / "processor", training=False)
    _verify_native_contract(config, bundle.contract)
    training_contract = _read(directory / "metrics.json")["native_contract"]
    for key in ("loader", "revision", "quantization", "snapshot_path", "processor_class"):
        if bundle.contract.get(key) != training_contract.get(key):
            raise SmokeError("RELOAD_NATIVE_RUNTIME_CONTRACT_CHANGED")
    signature = bundle.contract["parameter_storage"]["base_dtype_signature"]
    if signature != training_contract["parameter_storage"]["base_dtype_signature"]:
        raise SmokeError("RELOAD_BASE_DTYPE_SIGNATURE_CHANGED")
    if adapter_digest(bundle.model) != checkpoint["adapter_sha256"]:
        raise SmokeError("RELOADED_ADAPTER_WEIGHTS_DIFFER")
    record = config["data"]["selection"]["held_train_record"]
    batch, _ = _encode(rows[record["sample_id"]], record, bundle, config)
    bundle.model.eval()
    with torch.no_grad():
        loss, tokens = _target_loss(bundle.model, batch)
    from .training.rendering import encode_sample
    from .training.v6_native import verify_native_no_thinking_output
    inputs, _ = encode_sample(rows[record["sample_id"]], bundle.processor, bundle.architecture,
        max_length=2048, image_root=config["data"]["image_root"], include_labels=False)
    moved = {k: v.to(bundle.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    prompt_length = inputs["input_ids"].shape[-1]
    max_new_tokens = min(32, 2048 - prompt_length)
    if max_new_tokens < 1:
        raise SmokeError("HELD_TRAIN_GENERATION_CONTEXT_EXHAUSTED")
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        generated = bundle.model.generate(**moved, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
    continuation = generated[0, prompt_length:]
    count = continuation.shape[-1]
    if count < 1:
        raise SmokeError("RELOAD_GENERATED_NO_TOKENS")
    tokenizer = getattr(bundle.processor, "tokenizer", bundle.processor)
    raw = tokenizer.decode(continuation, skip_special_tokens=False)
    thinking = verify_native_no_thinking_output(raw)
    atomic_json(directory / "reload.generated.private.json", {"held_train_sample_id": record["sample_id"],
        "generated_ids": continuation.tolist(), "generated_text": raw})
    _verify_checkpoint(config, directory)
    _validate(config)
    result = {"status": "SUCCEEDED", **{k: config[k] for k in IDENTITY_KEYS}, "pid": os.getpid(),
        "fresh_process_required": True, "adapter_weights_verified": True, "adapter_sha256": checkpoint["adapter_sha256"],
        "held_train_sample_id": record["sample_id"], "held_train_loss": float(loss), "target_tokens": tokens,
        "finite_forward": True, "base_dtype_signature": signature, "generated_tokens": count, "generation_max_new_tokens": max_new_tokens,
        "generated_ids_sha256": canonical_hash(continuation.tolist()), "no_thinking_output": thinking,
        "dev_test_read": False, "runtime_versions": runtime_versions(), "finished_at": utcnow()}
    atomic_json(receipt_path, result)
    return result


def _reload(config_path, staging, config, metrics, invoke=None):
    path = staging / "reload.json"
    args = [sys.executable, "-m", "mediflow_medvlm.campaign_gpu_smoke", "--config", str(config_path),
            "--reload-checkpoint", str(staging), "--receipt", str(path)]
    # Never forward native stdout/stderr, which may contain a private input.
    result = (invoke or subprocess.run)(args, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        pass_fds=_inherited_supervisor_fds(config))
    for name in ("stdout", "stderr"):
        captured = getattr(result, name, b"") or b""
        if isinstance(captured, str):
            captured = captured.encode("utf-8")
        (staging / ("reload." + name + ".private.log")).write_bytes(captured)
    if result.returncode != 0 or not path.is_file():
        raise SmokeError("FRESH_PROCESS_RELOAD_FAILED")
    value = _read(path)
    if (value.get("status") != "SUCCEEDED" or any(value.get(k) != config[k] for k in IDENTITY_KEYS)
            or not value.get("pid") or value["pid"] in {os.getpid(), metrics["training_pid"]}
            or value.get("adapter_sha256") != metrics["adapter_sha256"]
            or value.get("held_train_sample_id") != config["data"]["selection"]["held_train_record"]["sample_id"]
            or value.get("finite_forward") is not True or value.get("adapter_weights_verified") is not True
            or value.get("fresh_process_required") is not True or value.get("dev_test_read") is not False
            or value.get("base_dtype_signature") != metrics["native_contract"]["parameter_storage"]["base_dtype_signature"]
            or value.get("generated_tokens", 0) < 1
            or value.get("runtime_versions") != config["runtime_versions"]
            or not math.isfinite(value.get("held_train_loss", math.nan)) or value.get("target_tokens", 0) < 1):
        raise SmokeError("FRESH_PROCESS_RELOAD_PROOF_INVALID")
    return value


def _completed_output(config):
    output = Path(config["output_dir"])
    if not (output / "completion.json").is_file():
        raise SmokeError("INCOMPLETE_OUTPUT_NO_FULL_STATE_RESUME")
    marker = _read(output / "completion.json")
    if marker.get("status") != "SUCCEEDED" or any(marker.get(k) != config[k] for k in IDENTITY_KEYS):
        raise SmokeError("EXISTING_RECEIPT_IDENTITY_MISMATCH")
    actual = _tree_files(output)
    expected = {str(Path(a["path"]).relative_to(output)): a["sha256"] for a in marker.get("artifacts", [])}
    if not {"config.json", "metrics.json", "checkpoint.json", "reload.json"} <= actual.keys() or actual != expected:
        raise SmokeError("EXISTING_OUTPUT_ARTIFACT_CHANGED")
    if _read(output / "config.json") != config:
        raise SmokeError("EXISTING_CONFIG_MISMATCH")
    _verify_checkpoint(config, output)
    return {**marker, "artifacts": marker["artifacts"] + [{"path": str(output / "completion.json"), "sha256": file_hash(output / "completion.json")}]}


def run_smoke(config_path, receipt_path, *, invoke=None):
    config_path, receipt_path = _absolute(config_path), _absolute(receipt_path)
    config = _read(config_path)
    initial_config_bytes = file_hash(config_path)
    output = _absolute(config["output_dir"])
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with output.with_name(output.name + ".worker.lock").open("a") as lock:
        os.chmod(lock.name, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = _validate(config)
        if output.exists():
            result = _completed_output(config)
            if receipt_path.exists() and _read(receipt_path) != result:
                raise SmokeError("EXISTING_RECEIPT_MARKER_MISMATCH")
            atomic_json(receipt_path, result)
            return result
        if receipt_path.exists():
            raise SmokeError("RECEIPT_WITHOUT_COMPLETE_OUTPUT")
        # A retry is a new preserved attempt under the same config identity.
        # No partial optimizer/adapter state can silently initialize this attempt.
        staging = Path(tempfile.mkdtemp(prefix=output.name + ".attempt-", dir=output.parent))
        old_umask = os.umask(0o077)
        try:
            atomic_json(staging / "attempt.json", {"status": "STARTED",
                **{k: config[k] for k in IDENTITY_KEYS}, "attempt_id": uuid.uuid4().hex,
                "resume_supported": False, "retry_start": "original_base", "started_at": utcnow()})
            atomic_json(staging / "config.json", config)
            _progress(config, 0, "load_native_gpu")
            _inherited_supervisor_fds(config)
            ledger = BudgetLedger(config["gpu_ledger_path"], policy=ResourcePolicy())
            ledger.start(config["model_id"], config["run_id"], "native_smoke_train")
            try:
                metrics = _train(config, rows, staging)
            finally:
                ledger.finish()
            metrics["gpu_accounting"] = {"train_seconds": ledger.session_seconds,
                "ledger_path": config["gpu_ledger_path"], "snapshot_after_train": ledger.snapshot()}
            atomic_json(staging / "metrics.json", metrics)
            if (metrics.get("optimizer_steps") != 20 or metrics.get("samples_processed") != 320
                    or metrics.get("excluded_count") != 0 or metrics.get("adapter_changed") is not True):
                raise SmokeError("INCOMPLETE_SMOKE_TRAINING")
            _validate(config)
            _checkpoint(config, staging, metrics)
            _progress(config, 20, "fresh_process_reload")
            reload_ledger = BudgetLedger(config["gpu_ledger_path"], policy=ResourcePolicy())
            reload_ledger.start(config["model_id"], config["run_id"], "native_smoke_reload")
            try:
                _reload(config_path, staging, config, metrics, invoke)
            finally:
                reload_ledger.finish()
            metrics["gpu_accounting"].update(reload_seconds=reload_ledger.session_seconds,
                snapshot_after_reload=reload_ledger.snapshot())
            atomic_json(staging / "metrics.json", metrics)
            _validate(config)
            if file_hash(config_path) != initial_config_bytes:
                raise SmokeError("CONFIG_FILE_CHANGED_DURING_SMOKE")
            _verify_checkpoint(config, staging)
            # fsync model/processor files too: save_pretrained alone is not a
            # durable checkpoint and cannot close the rename/receipt crash gap.
            for path in staging.rglob("*"):
                if path.is_file():
                    os.chmod(path, 0o600)
                    with path.open("rb") as stream:
                        os.fsync(stream.fileno())
                elif path.is_dir():
                    os.chmod(path, 0o700)
            artifacts = [{"path": str(output / name), "sha256": digest} for name, digest in _tree_files(staging).items()]
            result = {"schema_version": "medvlm.v6.native-gpu-smoke-receipt.v1", "status": "SUCCEEDED",
                **{k: config[k] for k in IDENTITY_KEYS}, "cell_id": config["cell_id"], "scope": SCOPE,
                "optimizer_steps": 20, "samples_processed": 320, "excluded_count": 0,
                "fresh_process_reload_verified": True, "gpu_smoke_verified": True, "adapter_changed": True,
                "common_train_freeze_decided": False, "v6_main_training_completed": False,
                "usable_as_main_initialization": False, "campaign_terminal": False, "dev_test_read": False,
                "artifacts": artifacts, "finished_at": utcnow()}
            atomic_json(staging / "completion.json", result)
            for directory in [p for p in staging.rglob("*") if p.is_dir()] + [staging]:
                fd = os.open(directory, os.O_DIRECTORY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            os.rename(staging, output)
            fd = os.open(output.parent, os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            result = _completed_output(config)
            atomic_json(receipt_path, result)
            return result
        except BaseException as exc:
            if staging.exists():
                atomic_json(staging / "failure.private.json", {"exception_type": type(exc).__name__,
                    "exception_message": str(exc), "traceback": traceback.format_exc()})
                atomic_json(staging / "failure.json", {"status": "FAILED", "exception_type": type(exc).__name__,
                    "reason_code": str(exc) if isinstance(exc, SmokeError) else "NATIVE_FAILURE",
                    "resume_supported": False, "retry_start": "original_base", "finished_at": utcnow()})
            raise
        finally:
            os.umask(old_umask)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--reload-checkpoint")
    args = parser.parse_args(argv)
    try:
        if args.reload_checkpoint:
            result = reload_checkpoint(_absolute(args.config), _absolute(args.reload_checkpoint), _absolute(args.receipt))
        else:
            result = run_smoke(args.config, args.receipt)
    except Exception as exc:
        failure = {"status": "FAILED", "scope": SCOPE, "exception_type": type(exc).__name__, "finished_at": utcnow()}
        if isinstance(exc, SmokeError):
            failure["reason_code"] = str(exc)
        if args.reload_checkpoint:
            atomic_json(Path(args.reload_checkpoint) / "reload.failure.private.json",
                {"exception_type": type(exc).__name__, "exception_message": str(exc), "traceback": traceback.format_exc()})
        print(json.dumps(failure), file=sys.stderr)
        return 1
    print(json.dumps({"status": result["status"], "scope": SCOPE}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
