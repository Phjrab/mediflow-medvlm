"""Finite R2 V6/V7 plan and a CPU-only, independently launchable Hub preparation job.

Only ``plan`` and ``prepare-model`` are implemented here. Future GPU/data/evaluation
jobs are explicitly pending implementation; preparing weights is never a smoke,
training success, architecture verification, or a campaign terminal result.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import threading

from .runtime import atomic_json, canonical_hash, file_hash, utcnow

CELLS = (
    ("T1", "Qwen/Qwen3-1.7B", "text", "qwen"),
    ("T2", "Qwen/Qwen3-4B", "text", "qwen"),
    ("T3", "google/gemma-3-4b-it", "text", "gemma"),
    ("T4", "google/medgemma-1.5-4b-it", "text", "gemma"),
    ("T5", "microsoft/Phi-4-mini-instruct", "text", "phi"),
    ("T6", "HuggingFaceTB/SmolLM3-3B", "text", "smollm"),
    ("V1", "Qwen/Qwen3.5-2B", "vqa", "qwen_vlm"),
    ("V2", "google/gemma-3-4b-it", "vqa", "gemma_vlm"),
    ("V3", "google/medgemma-1.5-4b-it", "vqa", "gemma_vlm"),
    ("V4", "HuggingFaceTB/SmolVLM2-2.2B-Instruct", "vqa", "smolvlm"),
)
SEEDS = (20261023, 20261114, 20261115)
STAGES = ("preflight_smoke", "common_train_freeze", "first_seed_final",
          "validation_caps", "full_dev", "family_selection", "additional_seeds",
          "test", "report")
RECIPE = {"passes": 1, "train_context": 2048, "microbatch": 1,
          "gradient_accumulation": 16, "quantization": "nf4_double_quant",
          "lora_rank": 8, "lora_alpha": 16, "lora_dropout": 0.05,
          "text_lr": 2e-4, "vqa_lr": 1e-4, "loss": "mean_example_target_token_mean",
          "optimizer": "AdamW", "weight_decay": 0, "warmup": 0,
          "scheduler": "linear", "gradient_clip_norm": 1.0,
          "optimizer_betas_eps": "PENDING_EXISTING_ENVIRONMENT_VERIFICATION",
          "checkpoint": "one_pass_final", "smoke_optimizer_updates": 20,
          "max_technical_recoveries_after_initial": 2}
SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def cell_spec(cell_id):
    for key, model, task, family in CELLS:
        if key == cell_id:
            return {"cell_id": key, "model_id": model, "task": task, "family": family}
    raise ValueError("undeclared cell")


def model_key(model_id):
    return model_id.replace("/", "--")


def source_identity(repo):
    files = {name: file_hash(Path(repo) / name) for name in
             ("src/mediflow_medvlm/campaign_v6.py", "src/mediflow_medvlm/campaign_supervisor.py",
              "src/mediflow_medvlm/runtime.py",
              "src/mediflow_medvlm/__init__.py", "src/mediflow_medvlm/domain.py",
              "src/mediflow_medvlm/validation.py")}
    # Match the supervisor's byte-for-byte source mapping convention.
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return files, digest


def build_plan(data_root, campaign_root, *, repo=None, python=None):
    repo = Path(repo or Path(__file__).resolve().parents[2]).resolve()
    data_root, campaign_root = Path(data_root).resolve(), Path(campaign_root).resolve()
    files, source_hash = source_identity(repo)
    cells = [dict(cell_spec(c[0]), seed=SEEDS[0], status="PENDING") for c in CELLS]
    jobs, seen = [], set()
    for cell in cells:
        if cell["model_id"] in seen:
            continue
        seen.add(cell["model_id"])
        job_id = "prepare-" + cell["cell_id"]
        receipt = campaign_root / "receipts" / (job_id + ".json")
        config_hash = canonical_hash({"operation": "prepare-model", "model_id": cell["model_id"],
                                      "trust_remote_code": False, "full_model_load": False})
        jobs.append({"job_id": job_id, "phase": "model_preparation", "cell_id": cell["cell_id"],
                     "run_id": "v6-" + job_id, "dependencies": [], "implementation_pending": False,
                     "argv": [str(python or sys.executable), "-m", "mediflow_medvlm.campaign_v6",
                              "prepare-model", "--cell", cell["cell_id"], "--data-root", str(data_root),
                              "--campaign-root", str(campaign_root), "--receipt", str(receipt)],
                     "cwd": str(repo), "source_files": files, "source_hash": source_hash,
                     "config_hash": config_hash,
                     "data_hash": canonical_hash({"data_root": str(data_root), "training_data": "NOT_READ_PREPARATION"}),
                     "model_hash": canonical_hash({"model_id": cell["model_id"], "revision": "RESOLVE_AND_PIN_ONCE"}),
                     "success_receipt": str(receipt), "requires_gpu": False, "uses_gpu": False})
    dependencies = [j["job_id"] for j in jobs]
    for stage in STAGES:
        job_id = "v6-" + stage
        jobs.append({"job_id": job_id, "phase": stage, "cell_id": "ALL_V6", "run_id": job_id,
                     "dependencies": dependencies, "dependency_policy": "terminal", "implementation_pending": True,
                     "status": "PENDING_IMPLEMENTATION", "reason": "Scheduled Codex must implement and verify this gate before enabling it."})
        dependencies = [job_id]
    for condition in ("P0", "P1", "R0", "R1"):
        jobs.append({"job_id": "v7-" + condition, "phase": "v7", "cell_id": condition,
                     "run_id": "v7-" + condition + "-20261023", "dependencies": ["v6-report"], "dependency_policy": "terminal",
                     "implementation_pending": True, "status": "PENDING_IMPLEMENTATION",
                     "requires_v6_terminal_and_t1_protocol": True})
    jobs.append({"job_id": "v7-report", "phase": "report", "cell_id": "ALL_V7", "run_id": "v7-report",
                 "dependencies": ["v7-" + c for c in ("P0", "P1", "R0", "R1")], "dependency_policy": "terminal",
                 "implementation_pending": True, "status": "PENDING_IMPLEMENTATION"})
    return {"schema_version": "medvlm.v6-v7.manifest.v1", "campaign_id": "medvlm-v6-v7-r2-20261114",
            "implementation_status": "INITIAL_PREPARATION_ONLY", "data_root": str(data_root),
            "campaign_root": str(campaign_root), "cells": cells, "jobs": jobs,
            "counts": {"unique_models": 8, "v6_first_seed_conditions": 10,
                       "v6_additional_seed_conditions_max": 8, "v6_total_conditions_max": 18,
                       "v7_conditions_max": 4, "v6_smoke_paths_max": 10},
            "seeds": list(SEEDS), "recipe": RECIPE,
            "first_seed_conditions": [{"condition_id": c[0] + "-20261023", "cell_id": c[0],
                                       "seed": SEEDS[0], "start_from": "original_base"} for c in CELLS],
            "additional_seed_slots": [{"slot_id": f"{axis}-family-{rank}-{seed}", "axis": axis,
                                       "family_rank": rank, "seed": seed,
                                       "cell_id": "RESOLVED_ONLY_BY_FROZEN_VALIDATION_SELECTION"}
                                      for axis in ("text", "vqa") for rank in (1, 2) for seed in SEEDS[1:]],
            "resource_policy": {"budget_mode": "unlimited", "total_gpu_hours": None,
                                "per_model_gpu_hours": None, "run_wall_seconds": None,
                                "job_timeout_seconds": None, "campaign_wall_seconds": None},
            "stage_order": list(STAGES),
            "selection": {"only_full_common_validation": True, "axis_representatives_max": 2,
                          "different_families": True, "training_seed": SEEDS[0],
                          "additional_seeds": list(SEEDS[1:]), "test_before_selection": False,
                          "text_score": "0.5*mean(71875_mcq_strict,71875_short_f1,71875_long_f1)+0.5*mean(pubmedqa_strict,medquad_f1)",
                          "vqa_score": "0.5*(slake_dev_open_em+slake_dev_closed_em)",
                          "tie_break": "unrounded_classification_mean_or_overall_em_then_model_id"},
            "v7": {"model_id": CELLS[0][1], "conditions": ["P0", "P1", "R0", "R1"],
                   "seed": SEEDS[0], "n_default": 2000, "n_min": 800, "n_multiple": 80,
                   "replay_fraction": 0.2, "start_from": "original_base_independently",
                   "after": "V6_TERMINAL_AND_T1_PATH_AND_FROZEN_PROTOCOL",
                   "missing_prerequisite_terminal_status": "BLOCKED_PREREQUISITE"},
            "completion_requires": ["all_declared_cells_terminal", "final_report_verified", "actual_ci_verified"],
            "pending_implementation_is_terminal": False, "jetson": "USER_DEFERRED_NOT_RUN"}


def select_families(dev_rows, *, caps_frozen=False):
    """Pure validation-only selection; all ten cells must have explicit outcomes."""
    expected = {c[0] for c in CELLS}
    if not caps_frozen or set(dev_rows) != expected:
        raise ValueError("all first-seed full-dev outcomes and frozen caps are required")
    terminal = {"SUCCEEDED", "FAILED", "BLOCKED_ACCESS", "BLOCKED_HARDWARE", "BLOCKED_PREREQUISITE"}
    if any(row.get("status") not in terminal for row in dev_rows.values()):
        raise ValueError("first-seed full-dev barrier incomplete")
    chosen = {}
    for axis in ("text", "vqa"):
        representatives = {}
        for cid, row in dev_rows.items():
            spec = cell_spec(cid)
            if spec["task"] != axis or row["status"] != "SUCCEEDED":
                continue
            if row.get("split") != "validation" or not row.get("full_common_dev"):
                raise ValueError("selection accepts only full common validation")
            score, tie = float(row["score"]), float(row["tie_score"])
            if not math.isfinite(score) or not math.isfinite(tie):
                raise ValueError("non-finite selection metric")
            key = (-score, -tie, spec["model_id"])
            family = spec["family"]
            if family not in representatives or key < representatives[family][0]:
                representatives[family] = (key, cid)
        chosen[axis] = [cid for _, cid in sorted(representatives.values())[:2]]
    return chosen


def _safe_name(name):
    p = Path(name)
    if p.is_absolute() or ".." in p.parts or "\\" in name:
        raise ValueError("unsafe upstream filename")
    return name


def preparation_files(names):
    """Exact allowlist: native assets, card/license, and one safetensors format."""
    names = {_safe_name(n) for n in names}
    small = {n for n in names if "/" not in n and
             (n.endswith(".json") and not n.startswith(("training", "trainer", "optimizer"))
              or n in {"README.md", "LICENSE", "LICENSE.txt", "LICENSE.md", "NOTICE", "NOTICE.txt",
                       "tokenizer.model", "spiece.model", "sentencepiece.bpe.model", "merges.txt",
                       "vocab.txt", "chat_template.jinja", "added_tokens.json"})}
    small.update(n for n in names if n.startswith("chat_templates/") and n.endswith(".jinja"))
    # Avoid optional duplicated original/pytorch/onnx/quantized checkpoint formats.
    if "model.safetensors.index.json" in names:
        weights = {n for n in names if re.fullmatch(r"model-\d+-of-\d+\.safetensors", n)}
    elif "model.safetensors" in names:
        weights = {"model.safetensors"}
    else:
        raise ValueError("no supported native safetensors checkpoint")
    if not weights or len(small | weights) > 4096 or "config.json" not in small:
        raise ValueError("unsupported or unbounded upstream file layout")
    return sorted(small), sorted(weights)


def _cached_revision(model_id, cache):
    base = cache / ("models--" + model_id.replace("/", "--"))
    ref = base / "refs" / "main"
    revisions = [ref.read_text().strip()] if ref.is_file() else []
    snapshots = base / "snapshots"
    if snapshots.is_dir():
        revisions.extend(sorted(p.name for p in snapshots.iterdir() if p.is_dir()))
    for revision in revisions:
        if SHA_RE.fullmatch(revision) and (snapshots / revision / "config.json").is_file():
            return revision
    return None


def resolve_pin(model_id, directory, cache, api):
    """Persist the immutable revision before any file transfer, retain on retries."""
    pin_path = directory / "pin.json"
    prior = json.loads(pin_path.read_text()) if pin_path.exists() else None
    if prior and (prior.get("model_id") != model_id or not SHA_RE.fullmatch(prior.get("revision", ""))):
        raise ValueError("persisted model pin identity mismatch")
    revision = prior["revision"] if prior else _cached_revision(model_id, cache)
    info = api.model_info(model_id, revision=revision, files_metadata=True)
    if not SHA_RE.fullmatch(info.sha) or (revision and info.sha != revision):
        raise ValueError("Hub did not verify requested immutable revision")
    pin = prior or {"model_id": model_id, "revision": info.sha, "pinned_at": utcnow(),
                    "selection": "existing_cached_revision_verified_upstream" if revision else "hub_revision_resolved_once"}
    if not prior:
        atomic_json(pin_path, pin)
    return pin, info


@contextmanager
def _model_lock(directory):
    import fcntl
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "prepare.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _heartbeats(path, progress):
    stop = threading.Event()
    def write():
        atomic_json(path, {**progress, "heartbeat_at": utcnow(), "pid": os.getpid()})
    def loop():
        while not stop.wait(30):
            write()
    write()
    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    return stop, thread


def _identity(cell, data_root):
    defaults = {"run_id": "v6-prepare-" + cell["cell_id"],
                "source_hash": source_identity(Path(__file__).resolve().parents[2])[1],
                "config_hash": canonical_hash({"operation": "prepare-model", "model_id": cell["model_id"],
                                               "trust_remote_code": False, "full_model_load": False}),
                "data_hash": canonical_hash({"data_root": str(data_root), "training_data": "NOT_READ_PREPARATION"}),
                "model_hash": canonical_hash({"model_id": cell["model_id"], "revision": "RESOLVE_AND_PIN_ONCE"})}
    return {k: os.environ.get("MEDVLM_SUPERVISOR_" + k.upper(), value) for k, value in defaults.items()}


def prepare_model(cell_id, data_root, campaign_root, receipt):
    """Fetch one unique base, construct native config/tokenizer/processor, no GPU."""
    from huggingface_hub import HfApi, snapshot_download, try_to_load_from_cache
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer
    import transformers

    data_root, campaign_root, receipt = Path(data_root).resolve(), Path(campaign_root).resolve(), Path(receipt).resolve()
    cell = cell_spec(cell_id)
    cache = data_root / "models" / "hub"
    cache.mkdir(parents=True, exist_ok=True)
    directory = campaign_root / "models" / model_key(cell["model_id"])
    identity = _identity(cell, data_root)
    progress = {"status": "RUNNING", "phase": "model_preparation", "cell_id": cell_id,
                "model_id": cell["model_id"], **identity, "last_progress_at": utcnow(),
                "optimizer_step": 0, "total_updates": 0, "stage": "resolve_pin"}
    progress_path = Path(os.environ.get("MEDVLM_SUPERVISOR_PROGRESS_PATH", str(receipt.with_suffix(".progress.json"))))
    stop, heartbeat = _heartbeats(progress_path, progress)
    try:
        with _model_lock(directory):
            # HfApi follows normal environment/on-disk HF credentials. Never read or print tokens.
            pin, info = resolve_pin(cell["model_id"], directory, cache, HfApi())
            revision = pin["revision"]
            card = info.card_data.to_dict() if info.card_data and hasattr(info.card_data, "to_dict") else dict(info.card_data or {})
            provenance = {"model_id": cell["model_id"], "revision": revision,
                          "card_url": f"https://huggingface.co/{cell['model_id']}/blob/{revision}/README.md",
                          "config_url": f"https://huggingface.co/{cell['model_id']}/resolve/{revision}/config.json",
                          "card_metadata": card, "declared_license": card.get("license"),
                          "gated": info.gated, "checked_at": utcnow(),
                          "credential_source": "normal_local_huggingface_credentials_not_recorded"}
            atomic_json(directory / "provenance.json", provenance)
            siblings = {s.rfilename: s for s in info.siblings}
            small, weights = preparation_files(siblings)
            def cached(name):
                found = try_to_load_from_cache(cell["model_id"], name, cache_dir=str(cache), revision=revision)
                return Path(found) if isinstance(found, str) and Path(found).is_file() else None
            def required_bytes(names):
                total = 0
                for name in names:
                    size = siblings[name].size
                    if not isinstance(size, int) or size < 0:
                        raise ValueError("upstream required file size unavailable")
                    local = cached(name)
                    if not local or local.stat().st_size != size:
                        total += size
                return total
            # Check full missing footprint BEFORE any base weights are downloaded.
            missing = required_bytes(small + weights)
            free = shutil.disk_usage(cache).free
            reserve = max(1024**3, math.ceil(missing * 0.05))
            disk = {"checked_at": utcnow(), "cache_dir": str(cache), "free_bytes": free,
                    "missing_bytes": missing, "reserve_bytes": reserve, "required_bytes": missing + reserve}
            atomic_json(directory / "disk_check.json", disk)
            if free < missing + reserve:
                raise OSError("INSUFFICIENT_DISK_BEFORE_WEIGHT_DOWNLOAD")
            progress.update(stage="native_assets", last_progress_at=utcnow())
            snapshot = Path(snapshot_download(cell["model_id"], revision=revision, cache_dir=str(cache),
                                             allow_patterns=small, max_workers=2))
            config = AutoConfig.from_pretrained(str(snapshot), trust_remote_code=False, local_files_only=True)
            architectures = config.architectures or []
            available = [name for name in architectures if hasattr(transformers, name)]
            if not available:
                raise ValueError("NATIVE_ARCHITECTURE_UNAVAILABLE")
            tokenizer = AutoTokenizer.from_pretrained(str(snapshot), trust_remote_code=False, local_files_only=True)
            processor = None
            if any(c[1] == cell["model_id"] and c[2] == "vqa" for c in CELLS):
                processor = AutoProcessor.from_pretrained(str(snapshot), trust_remote_code=False, local_files_only=True)
            template = getattr(processor, "chat_template", None) or getattr(tokenizer, "chat_template", None)
            if not template:
                raise ValueError("NATIVE_CHAT_TEMPLATE_MISSING")
            index = snapshot / "model.safetensors.index.json"
            if index.exists():
                indexed = {_safe_name(n) for n in json.loads(index.read_text())["weight_map"].values()}
                if not indexed or not indexed.issubset(set(weights)):
                    raise ValueError("native weight index does not match bounded file list")
                weights = sorted(indexed)
            progress.update(stage="weights_download", last_progress_at=utcnow())
            snapshot_download(cell["model_id"], revision=revision, cache_dir=str(cache),
                              allow_patterns=weights, max_workers=2)
            progress.update(stage="verify_files", last_progress_at=utcnow())
            verified = {}
            for name in small + weights:
                path = snapshot / name
                if not path.is_file() or path.stat().st_size != siblings[name].size:
                    raise ValueError("cached/downloaded model file size mismatch")
                digest = file_hash(path)
                lfs = getattr(siblings[name], "lfs", None)
                upstream_hash = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs else None
                if upstream_hash and digest != upstream_hash:
                    raise ValueError("cached/downloaded model LFS digest mismatch")
                verified[name] = {"sha256": digest, "bytes": path.stat().st_size,
                                  "upstream_lfs_sha256": upstream_hash}
            provenance.update(files=verified, verified_at=utcnow())
            atomic_json(directory / "provenance.json", provenance)
            native = {"model_id": cell["model_id"], "revision": revision, "snapshot_path": str(snapshot),
                      "cache_dir": str(cache), "config": config.to_dict(), "architectures": architectures,
                      "native_classes_available": available, "tokenizer_class": type(tokenizer).__name__,
                      "processor_class": type(processor).__name__ if processor else None,
                      "chat_template_sha256": canonical_hash(template), "trust_remote_code": False,
                      "full_model_loaded": False, "gpu_smoke_verified": False,
                      "architecture_lora_mask_reload_verification": "PENDING_IMPLEMENTATION",
                      "shared_cells": [c[0] for c in CELLS if c[1] == cell["model_id"]]}
            atomic_json(directory / "native_assets.json", native)
            artifacts = [{"path": str(directory / n), "sha256": file_hash(directory / n)}
                         for n in ("pin.json", "disk_check.json", "provenance.json", "native_assets.json")]
            result = {"schema_version": "medvlm.v6.preparation-receipt.v1", "status": "SUCCEEDED",
                      "scope": "MODEL_PREPARATION_ONLY", "cell_id": cell_id, **identity,
                      "job_id": os.environ.get("MEDVLM_SUPERVISOR_JOB_ID", "prepare-" + cell_id),
                      "job_hash": os.environ.get("MEDVLM_SUPERVISOR_JOB_HASH"),
                      "model_id": cell["model_id"], "model_revision": revision,
                      "model_files_hash": canonical_hash(verified), "artifacts": artifacts,
                      "finished_at": utcnow(), "training_performed": False, "campaign_terminal": False}
            atomic_json(receipt, result)
            progress.update(status="SUCCEEDED", stage="prepared", last_progress_at=utcnow())
            return result
    except Exception as exc:
        code = getattr(getattr(exc, "response", None), "status_code", None)
        # Exception strings/HTTP request headers may contain credentials; retain class/status only.
        failure = {"status": "FAILED", "scope": "MODEL_PREPARATION_ONLY", **identity,
                   "cell_id": cell_id, "exception_type": type(exc).__name__, "http_status": code,
                   "phase": progress["stage"], "finished_at": utcnow(), "campaign_terminal": False}
        if code in (401, 403):
            failure["reason_code"] = "ACTUAL_HUB_ACCESS_DENIED"
        elif str(exc) in {"INSUFFICIENT_DISK_BEFORE_WEIGHT_DOWNLOAD", "NATIVE_ARCHITECTURE_UNAVAILABLE",
                          "NATIVE_CHAT_TEMPLATE_MISSING", "upstream required file size unavailable",
                          "cached/downloaded model LFS digest mismatch"}:
            failure["reason_code"] = str(exc)
        atomic_json(receipt.with_suffix(".failure.json"), failure)
        progress.update(status="FAILED", exception_type=type(exc).__name__, last_progress_at=utcnow())
        raise
    finally:
        stop.set()
        heartbeat.join(timeout=2)
        atomic_json(progress_path, {**progress, "heartbeat_at": utcnow(), "pid": os.getpid()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "prepare-model"):
        item = sub.add_parser(name)
        item.add_argument("--data-root", required=True)
        item.add_argument("--campaign-root", required=True)
        if name == "plan":
            item.add_argument("--output")
            item.add_argument("--python", default=sys.executable)
        else:
            item.add_argument("--cell", choices=[c[0] for c in CELLS], required=True)
            item.add_argument("--receipt", required=True)
    args = parser.parse_args(argv)
    if args.command == "plan":
        plan = build_plan(args.data_root, args.campaign_root, python=args.python)
        if args.output:
            atomic_json(args.output, plan)
        else:
            print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0
    try:
        result = prepare_model(args.cell, args.data_root, args.campaign_root, args.receipt)
        print(json.dumps({k: result[k] for k in ("status", "scope", "model_id", "model_revision")}))
        return 0
    except Exception as exc:
        # Do not print raw third-party errors or a traceback containing request secrets.
        print(json.dumps({"status": "FAILED", "exception_type": type(exc).__name__, "cell_id": args.cell}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
