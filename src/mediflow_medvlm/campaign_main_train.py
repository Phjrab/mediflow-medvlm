"""One immutable V6 first-seed cell: original Base, full pass, final reload.

CPU planning/verification never initializes CUDA. Only explicit child CLI stages
load the unchanged native GPU runtime. Interrupted attempts are preserved; their
adapter is not optimizer/RNG state and cannot resume or initialize another run.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
import traceback
import uuid

from . import telemetry as telemetry_module
from . import campaign_common_train_freeze as common
from . import campaign_gpu_smoke as smoke
from .campaign_supervisor import atomic_json, canonical_hash, file_hash, verify_source
from .campaign_v6 import cell_spec
from .runtime import BudgetLedger, ResourcePolicy, utcnow

SCHEMA = "medvlm.v6.main-train-config.v1"
SCOPE = "NATIVE_FIRST_SEED_ONE_PASS_FINAL_ONLY"
SOURCE_FILES = tuple(dict.fromkeys((*common.SOURCE_FILES, "campaign_main_train.py", "telemetry.py", "telemetry_probe.py")))
IDENTITY_KEYS = smoke.IDENTITY_KEYS
MainTrainError = smoke.SmokeError
_read, _absolute = smoke._read, smoke._absolute
_inherited_supervisor_fds = smoke._inherited_supervisor_fds
_require_parent_without_cuda = smoke._require_parent_without_cuda
_encode, _target_loss = smoke._encode, smoke._target_loss
_verify_native_contract = smoke._verify_native_contract
_tree_files = smoke._tree_files
runtime_versions = smoke.runtime_versions


def require(condition, code):
    if not condition:
        raise MainTrainError(code)


def source_identity(repo=None):
    package = Path(__file__).resolve().parent if repo is None else Path(repo) / "src/mediflow_medvlm"
    files = {"src/mediflow_medvlm/" + name: file_hash(package / name) for name in SOURCE_FILES}
    return files, canonical_hash(files)


def _freeze(path):
    path = _absolute(path)
    receipt = _read(path)
    require(receipt.get("status") == "SUCCEEDED" and receipt.get("scope") == common.SCOPE
        and receipt.get("common_train_freeze_decided") is True
        and receipt.get("v6_main_training_completed") is False, "SUCCESSFUL_COMMON_FREEZE_REQUIRED")
    matches = [a for a in receipt["artifacts"] if Path(a["path"]).name == "config.json"]
    require(len(matches) == 1, "COMMON_FREEZE_CONFIG_REQUIRED")
    config = _read(_absolute(matches[0]["path"]))
    require(config.get("schema_version") == common.SCHEMA and config.get("scope") == common.SCOPE
        and config.get("config_hash") == canonical_hash({k: v for k, v in config.items() if k != "config_hash"})
        and common._completed(config) == receipt and config.get("success_receipt") == str(path),
        "COMMON_FREEZE_RECEIPT_IDENTITY_MISMATCH")
    manifest, state = common.gate._active(config["state_path"])
    job = next(j for j in manifest["jobs"] if j["job_id"] == common.JOB_ID)
    require(all(job.get(k) == config.get(k) for k in (*IDENTITY_KEYS, "source_files")), "COMMON_FREEZE_JOB_MISMATCH")
    require(state["jobs"][common.JOB_ID].get("status") == "SUCCEEDED"
        and config["source_hash"] == canonical_hash(config["source_files"])
        and job.get("phase") == "common_train_freeze" and job.get("cell_id") == "ALL_V6", "COMMON_FREEZE_TERMINAL_SOURCE_IDENTITY_MISMATCH")
    verify_source(job)  # Its archived cwd, never the current imported supervisor hash.
    require(common.gate._success(job, state["jobs"][common.JOB_ID]) == receipt, "COMMON_FREEZE_STATE_PROOF_MISMATCH")
    plan = _read(Path(config["output_dir"]) / "freeze.json")
    require(plan.get("freeze_id") == config.get("freeze_id") == receipt.get("freeze_id")
        and plan["freeze_id"] == canonical_hash({k: v for k, v in plan.items() if k != "freeze_id"})
        and config["data_hash"] == canonical_hash({"snapshot": config["prerequisites"],
            "protection": config["protection_references"], "freeze_id": plan["freeze_id"]})
        and config["model_hash"] == canonical_hash(plan["per_cell_run_plan"])
        and plan.get("comparison_cohort_frozen") is True and plan.get("later_failure_may_widen_cohort") is False
        and plan.get("eval_answers_used_for_selection") is False and plan.get("eval_scores_read") is False,
        "COMMON_FREEZE_PLAN_IDENTITY_MISMATCH")
    return config, plan, {"path": str(path), "sha256": file_hash(path),
        "job_sha256": canonical_hash(job), "terminal_state_sha256": canonical_hash(state["jobs"][common.JOB_ID])}


def _inputs(freeze_config, plan, cell_id):
    spec = cell_spec(cell_id)
    require(cell_id in plan["supported_cells"] and cell_id in plan["per_cell_run_plan"], "CELL_NOT_IN_FROZEN_SUPPORTED_SET")
    axis = plan["axes"][spec["task"]]
    require(cell_id in axis["cells"] and axis["common_count"] == (7997 if spec["task"] == "text" else 1979),
        "EXACT_FROZEN_COMMON_COHORT_REQUIRED")
    run = plan["per_cell_run_plan"][cell_id]
    require(all(run.get(k) == v for k, v in spec.items()) and run.get("adapter_initialization") is None
        and run.get("initialization_source") == "original_pinned_base" and run.get("smoke_adapter_reuse") is False
        and run.get("historical_training_reused") is False, "ORIGINAL_INDEPENDENT_BASE_REQUIRED")
    native_ref = run["native_preflight_config"]
    require(file_hash(native_ref["path"]) == native_ref["sha256"], "NATIVE_PREFLIGHT_CONFIG_CHANGED")
    native = _read(native_ref["path"])
    smoke_ref = run["gpu_smoke_config"]
    require(file_hash(smoke_ref["path"]) == smoke_ref["sha256"], "FROZEN_GPU_CONFIG_CHANGED")
    proven_gpu = _read(smoke_ref["path"])
    require(proven_gpu.get("runtime_versions") == runtime_versions()
        and proven_gpu.get("model") == run["model"], "PROVEN_GPU_RUNTIME_OR_MODEL_CHANGED")
    old_job = freeze_config["prerequisites"]["jobs"]["v6-native-preflight-" + cell_id]
    verify_source(old_job)
    native_config, records, parent_rows = smoke._preflight(old_job["success_receipt"])
    require(native_config == native and native["model"] == run["model"]
        and native["model"] == smoke.cpu._prepared(native["model"]["preparation_receipt"]), "FROZEN_NATIVE_MODEL_MISMATCH")
    path = Path(freeze_config["output_dir"]) / (spec["task"] + ".train.jsonl")
    require(file_hash(path) == axis["train_sha256"], "COMMON_TRAIN_BYTES_CHANGED")
    rows = smoke.cpu._train_rows(path, axis["common_count"], spec["task"])
    ids = [r["sample_id"] for r in rows]
    require(ids == axis["sample_ids"] and canonical_hash(ids) == axis["sample_order_sha256"]
        and len(set(ids)) == len(ids), "COMMON_TRAIN_ID_ORDER_MISMATCH")
    positions = axis["source_positions"]
    require(len(positions) == len(rows) and positions == sorted(set(positions))
        and all(type(i) is int and 0 <= i < len(parent_rows) for i in positions)
        and [parent_rows[i] for i in positions] == rows, "COMMON_PARENT_CONTENT_MISMATCH")
    selected = [records[i] for i in positions]
    proof = axis["per_model"][cell_id]
    for row, record in zip(rows, selected):
        metadata = record.get("native", {})
        require(record.get("eligible") is True and record["sample_id"] == row["sample_id"]
            and record["input_sha256"] == canonical_hash(row["input"])
            and record["target_sha256"] == canonical_hash(row["supervision"])
            and all(metadata.get(k) is True for k in ("full_target_preserved", "prefix_verified", "prompt_image_pad_mask_verified", "all_tensors_cpu"))
            and type(metadata.get("sequence_tokens")) is int and 1 < metadata["sequence_tokens"] <= 2048
            and type(metadata.get("supervised_tokens_after_shift")) is int
            and 0 < metadata["supervised_tokens_after_shift"] < metadata["sequence_tokens"], "FULL_TARGET_ELIGIBILITY_PROOF_REQUIRED")
    require(sum(r["native"]["sequence_tokens"] for r in selected) == proof["common_sequence_tokens"]
        and sum(r["native"]["supervised_tokens_after_shift"] for r in selected) == proof["common_target_tokens"],
        "FROZEN_MODEL_TOKEN_TOTAL_MISMATCH")
    order = axis["training_orders"]["20261023"]
    expected = common._orders(ids)["20261023"]
    require(order == expected and order["optimizer_steps"] == (500 if spec["task"] == "text" else 124)
        and order["final_update_examples"] == (13 if spec["task"] == "text" else 11), "FROZEN_EXPOSURE_ORDER_MISMATCH")
    return native, {r["sample_id"]: r for r in rows}, [selected[i] for i in order["indices"]], axis, order


def _recipe(plan, task):
    frozen = plan["recipe"]
    expected = {"passes": 1, "max_length": 2048, "microbatch": 1, "effective_batch_size": 16,
        "partial_update_divisor": "actual_example_count", "repeat_or_drop_examples": False,
        "loss": "mean_examples_of_mean_full_target_token_cross_entropy", "quantization": "nf4_double_quant",
        "compute_dtype": "bfloat16", "lora_rank": 8, "lora_alpha": 16, "lora_dropout": .05,
        "optimizer": "AdamW", "betas": [.9, .999], "eps": 1e-8, "weight_decay": 0., "scheduler": "linear_decay",
        "warmup_steps": 0, "gradient_clip_norm": 1., "text_learning_rate": 2e-4, "vqa_learning_rate": 1e-4,
        "base_vision_projector_embeddings_head_frozen": True, "base_load_dtype": "float32_before_nf4",
        "gradient_checkpointing": True, "full_input_and_answer_preserved": True, "assistant_target_only": True,
        "prompt_image_pad_labels": -100, "fresh_process_final_reload_required": True,
        "checkpoint": "one_pass_final_only", "first_seed": 20261023}
    require(all(canonical_hash(frozen.get(k)) == canonical_hash(v) for k, v in expected.items()), "FROZEN_MAIN_RECIPE_MISMATCH")
    return {**frozen, "seed": 20261023, "learning_rate": frozen[task + "_learning_rate"]}


def build_config(common_freeze_receipt, cell_id, output_dir, *, repo=None, success_receipt=None, job_id=None):
    """Read-only immutable single-cell planner; no queue or GPU work."""
    freeze, plan, reference = _freeze(common_freeze_receipt)
    spec = cell_spec(cell_id)
    native, _, records, axis, order = _inputs(freeze, plan, cell_id)
    recipe = _recipe(plan, spec["task"])
    recipe.update(optimizer_steps=order["optimizer_steps"], examples=axis["common_count"])
    root = _absolute(freeze["prerequisites"]["declaration"]["campaign_root"])
    output = _absolute(output_dir)
    job_id = job_id or "v6-first-seed-" + cell_id
    require(job_id == "v6-first-seed-" + cell_id, "FIRST_SEED_JOB_ID_REQUIRED")
    receipt = _absolute(success_receipt or root / "receipts" / (job_id + ".json"))
    for path in (output, receipt): common.gate._private(path, root)
    files, source = source_identity(repo)
    model = native["model"]
    data_root = Path(model["cache_dir"]).parent.parent
    data = {"common_freeze": reference, "freeze_id": plan["freeze_id"], "partition": "train",
        "train_file": str(Path(freeze["output_dir"]) / (spec["task"] + ".train.jsonl")),
        "train_sha256": axis["train_sha256"], "count": len(records), "image_root": native["data"]["image_root"],
        "preflight_receipt": freeze["prerequisites"]["jobs"]["v6-native-preflight-" + cell_id]["success_receipt"],
        "preflight_config": plan["per_cell_run_plan"][cell_id]["native_preflight_config"],
        "source_order_sha256": axis["sample_order_sha256"], "sample_order_sha256": order["sample_order_sha256"],
        "indices_sha256": order["indices_sha256"], "update_example_counts": order["update_example_counts"],
        "sequence_tokens": axis["per_model"][cell_id]["common_sequence_tokens"],
        "target_tokens": axis["per_model"][cell_id]["common_target_tokens"],
        "probe_sample_id": records[-1]["sample_id"], "probe_scope": "TRAINING_EXAMPLE_SANITY_CHECK"}
    config = {"schema_version": SCHEMA, "scope": SCOPE, "phase": "native_main_train", **spec,
        "job_id": job_id, "run_id": job_id, "model": model, "data": data, "recipe": recipe,
        "output_dir": str(output), "success_receipt": str(receipt), "device": "cuda", "max_length": 2048,
        "fixed_template_date": native["fixed_template_date"], "local_files_only": True, "trust_remote_code": False,
        "adapter_initialization": None, "runtime_versions": runtime_versions(), "resource_policy": ResourcePolicy().as_dict(),
        "gpu_ledger_path": str(data_root / "runtime/gpu_budget.json"),
        "supervisor_lock_paths": [str(data_root / "v6_v7/worker.lock"), str(data_root / "runtime/medvlm_gpu.lock")],
        "source_files": files, "source_hash": source, "data_hash": canonical_hash(data), "model_hash": canonical_hash(model)}
    config["config_hash"] = canonical_hash(config)
    return config


def validate_config_identity(config, *, repo=None):
    """Static failed-cell audit only; does not certify input availability or execution.

    A terminal aggregate must additionally compare this config with its captured
    registration science identity. Successful results require validate_config.
    """
    require(config.get("schema_version") == SCHEMA and config.get("config_hash") == canonical_hash(
        {k: v for k, v in config.items() if k != "config_hash"}), "CONFIG_HASH_OR_SCHEMA_MISMATCH")
    spec = cell_spec(config["cell_id"])
    require(all(config.get(k) == v for k, v in spec.items()) and config.get("phase") == "native_main_train"
        and config.get("scope") == SCOPE and config.get("job_id") == config.get("run_id") == "v6-first-seed-" + config["cell_id"]
        and config.get("adapter_initialization") is None and config.get("device") == "cuda"
        and config.get("max_length") == 2048 and config.get("fixed_template_date") == "2026-11-14"
        and config.get("local_files_only") is True and config.get("trust_remote_code") is False,
        "STATIC_MAIN_CONFIG_POLICY_MISMATCH")
    recipe = _recipe({"recipe": config["recipe"]}, spec["task"])
    require(canonical_hash(config["recipe"]) == canonical_hash(recipe) and type(recipe["seed"]) is int and recipe["seed"] == 20261023
        and recipe["examples"] == config["data"]["count"] == (7997 if spec["task"] == "text" else 1979)
        and recipe["optimizer_steps"] == (500 if spec["task"] == "text" else 124)
        and config["data"].get("partition") == "train"
        and config["data"].get("probe_scope") == "TRAINING_EXAMPLE_SANITY_CHECK"
        and config["data"].get("update_example_counts") == [16] * (recipe["optimizer_steps"] - 1) + [13 if spec["task"] == "text" else 11],
        "STATIC_ONE_PASS_IDENTITY_MISMATCH")
    require(config.get("source_hash") == canonical_hash(config["source_files"])
        and (config["source_files"], config["source_hash"]) == source_identity(repo)
        and config.get("model_hash") == canonical_hash(config["model"])
        and config.get("data_hash") == canonical_hash(config["data"])
        and config.get("resource_policy") == ResourcePolicy().as_dict(), "STATIC_CONFIG_IDENTITY_MISMATCH")
    return config


def validate_config(config, *, repo=None):
    require(config.get("schema_version") == SCHEMA and config.get("config_hash") == canonical_hash(
        {k: v for k, v in config.items() if k != "config_hash"}), "CONFIG_HASH_OR_SCHEMA_MISMATCH")
    expected = build_config(config["data"]["common_freeze"]["path"], config["cell_id"], config["output_dir"],
        repo=repo, success_receipt=config["success_receipt"], job_id=config["job_id"])
    require(canonical_hash(config) == canonical_hash(expected), "FROZEN_CONFIG_IDENTITY_MISMATCH")
    return config


def _validate(config):
    validate_config(config)
    for key in IDENTITY_KEYS:
        supplied = os.environ.get("MEDVLM_SUPERVISOR_" + key.upper())
        require(supplied is None or supplied == config[key], "SUPERVISOR_IDENTITY_MISMATCH")
    freeze, plan, _ = _freeze(config["data"]["common_freeze"]["path"])
    _, rows, records, _, _ = _inputs(freeze, plan, config["cell_id"])
    return rows, records


def _progress(config, step, samples, stage, checkpoint=None):
    path = os.environ.get("MEDVLM_SUPERVISOR_PROGRESS_PATH")
    if path:
        atomic_json(path, {**{k: config[k] for k in IDENTITY_KEYS}, "status": "RUNNING", "stage": stage,
            "optimizer_step": step, "total_updates": config["recipe"]["optimizer_steps"], "samples_processed": samples,
            "total_examples": config["recipe"]["examples"], "pid": os.getpid(), "last_progress_at": utcnow(),
            "heartbeat_at": utcnow(), "last_complete_checkpoint": checkpoint, "optimizer_rng_resume_supported": False})


def _updates(model, optimizer, scheduler, records, config, example, gradient_report, progress=None, telemetry=None):
    """One pass; every target-token-mean example has equal weight within its update."""
    import torch
    losses, tokens, sequences, reports, norms, rates, seen, counts = [], [], [], [], [], [], [], []
    for step, start in enumerate(range(0, len(records), 16)):
        step_started = time.monotonic()
        batch_records = records[start:start + 16]
        if progress: progress(step, len(seen))
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.zero_grad(set_to_none=True)
        for record in batch_records:
            loss, count, sequence = example(record)
            require(bool(torch.isfinite(loss)), "NONFINITE_TRAIN_LOSS")
            (loss / len(batch_records)).backward()
            losses.append(float(loss.detach()))
            tokens.append(count); sequences.append(sequence); seen.append(record["sample_id"])
        reports.append(gradient_report())
        norm = float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0))
        require(math.isfinite(norm), "NONFINITE_GRADIENT_NORM")
        norms.append(norm)
        optimizer.step(); scheduler.step()
        counts.append(len(batch_records))
        if telemetry is not None:
            payload = telemetry_module.update_payload(step=step + 1, total=(len(records) + 15) // 16,
                losses=losses[start:], tokens=tokens[start:], sequences=sequences[start:], samples=len(seen),
                samples_total=len(records), lr_used=rates[-1], lr_next=optimizer.param_groups[0]["lr"],
                norm=norm, wall=time.monotonic() - step_started, allocator=telemetry_module.allocator_metrics(torch))
            payload.update(supervised_tokens_cumulative=sum(tokens), sequence_tokens_cumulative=sum(sequences))
            telemetry.step(payload)
        if progress: progress(step + 1, len(seen))
    optimizer.zero_grad(set_to_none=True)
    return {"optimizer_steps": len(counts), "samples_processed": len(seen), "excluded_count": 0,
        "sample_ids": seen, "sample_order_sha256": canonical_hash(seen), "update_example_counts": counts,
        "micro_loss_divisor_per_update": counts, "example_losses": losses, "loss_mean": sum(losses) / len(losses),
        "example_supervised_tokens": tokens, "supervised_tokens": sum(tokens), "sequence_tokens": sum(sequences),
        "gradient_reports": reports, "gradient_norms_before_clip": norms, "learning_rates": rates,
        "final_learning_rate": optimizer.param_groups[0]["lr"]}


def _train(config, rows, records, staging, telemetry=None):
    _inherited_supervisor_fds(config)
    import torch
    from .training.v6_gpu import (load_v6_gpu_bundle, adapter_digest, projection_gradient_report,
        projection_adapter_digests, assert_frozen_base)
    require(torch.cuda.is_available(), "GPU_NOT_AVAILABLE")
    seed = config["recipe"]["seed"]
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    bundle = load_v6_gpu_bundle(config["model"], config["cell_id"], fixed_template_date=config["fixed_template_date"])
    _verify_native_contract(config, bundle.contract)
    model = bundle.model
    assert_frozen_base(model)
    initial = adapter_digest(model)
    before = projection_adapter_digests(model, bundle.contract["architecture"])
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=config["recipe"]["learning_rate"], betas=tuple(config["recipe"]["betas"]),
        eps=config["recipe"]["eps"], weight_decay=0.)
    total = config["recipe"]["optimizer_steps"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(0., 1. - step / total))
    def example(record):
        batch, metadata = _encode(rows[record["sample_id"]], record, bundle, config)
        loss, count = _target_loss(model, batch)
        require(count == record["native"]["supervised_tokens_after_shift"], "ACTUAL_TARGET_TOKEN_COUNT_CHANGED")
        return loss, count, metadata["sequence_tokens"]
    metrics = _updates(model, optimizer, scheduler, records, config, example,
        lambda: projection_gradient_report(model, bundle.contract["architecture"]),
        lambda step, count: _progress(config, step, count, "one_pass_native_main_training"), telemetry=telemetry)
    assert_frozen_base(model)
    final = adapter_digest(model)
    after = projection_adapter_digests(model, bundle.contract["architecture"])
    metrics.update(initial_adapter_sha256=initial, adapter_sha256=final, adapter_changed=initial != final,
        initial_projection_sha256=before, final_projection_sha256=after,
        all_projection_tensors_changed=bool(before) and before.keys() == after.keys() and all(
            before[path][label] != after[path][label] for path in before for label in ("A", "B")),
        training_pid=os.getpid(), native_contract=bundle.contract, base_frozen_verified=True,
        retry_start="original_base", optimizer_rng_resume_supported=False)
    _metrics_proof(config, metrics, records)
    adapter, processor = staging / "adapter", staging / "processor"
    adapter.mkdir(mode=0o700); processor.mkdir(mode=0o700)
    model.save_pretrained(adapter, safe_serialization=True)
    bundle.processor.save_pretrained(processor)
    require(any(p.name.startswith("adapter_model") and p.suffix == ".safetensors" for p in adapter.iterdir()), "ADAPTER_WEIGHTS_MISSING")
    torch.cuda.synchronize()
    properties = torch.cuda.get_device_properties(0)
    metrics["resource"] = {"wall_seconds": time.monotonic() - started, "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(), "device_name": properties.name,
        "device_total_memory_bytes": properties.total_memory, "device_capability": list(torch.cuda.get_device_capability()),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "runtime_versions": runtime_versions(), "energy_joules": None}
    metrics["finished_at"] = utcnow()
    return metrics


def _finite(values, length):
    return isinstance(values, list) and len(values) == length and all(type(v) in (int, float) and math.isfinite(v) for v in values)


def _metrics_proof(config, metrics, records):
    recipe = config["recipe"]
    steps, count = recipe["optimizer_steps"], recipe["examples"]
    ids = [r["sample_id"] for r in records]
    require(len(ids) == len(set(ids)) == count and metrics.get("optimizer_steps") == steps
        and metrics.get("samples_processed") == count and metrics.get("excluded_count") == 0
        and metrics.get("sample_ids") == ids and metrics.get("sample_order_sha256") == canonical_hash(ids) == config["data"]["sample_order_sha256"]
        and metrics.get("update_example_counts") == config["data"]["update_example_counts"]
        and metrics.get("micro_loss_divisor_per_update") == config["data"]["update_example_counts"], "INCOMPLETE_OR_REORDERED_ONE_PASS")
    expected_tokens = [r["native"]["supervised_tokens_after_shift"] for r in records]
    require(_finite(metrics.get("example_losses"), count) and metrics.get("example_supervised_tokens") == expected_tokens
        and metrics.get("supervised_tokens") == sum(expected_tokens) == config["data"]["target_tokens"]
        and metrics.get("sequence_tokens") == sum(r["native"]["sequence_tokens"] for r in records) == config["data"]["sequence_tokens"]
        and math.isfinite(metrics.get("loss_mean", math.nan))
        and math.isclose(metrics["loss_mean"], sum(metrics["example_losses"]) / count, rel_tol=1e-9, abs_tol=1e-9), "FULL_TARGET_ONE_PASS_PROOF_MISMATCH")
    require(_finite(metrics.get("learning_rates"), steps) and all(math.isclose(value, recipe["learning_rate"] * (1 - i / steps),
        rel_tol=1e-9, abs_tol=1e-12) for i, value in enumerate(metrics["learning_rates"]))
        and metrics.get("final_learning_rate") == 0., "ONE_PASS_SCHEDULER_PROOF_MISMATCH")
    before, after = metrics.get("initial_projection_sha256", {}), metrics.get("final_projection_sha256", {})
    require(before and before.keys() == after.keys() and metrics.get("adapter_changed") is True
        and metrics.get("initial_adapter_sha256") != metrics.get("adapter_sha256")
        and metrics.get("all_projection_tensors_changed") is True and all(before[p][k] != after[p][k] for p in before for k in ("A", "B"))
        and metrics.get("base_frozen_verified") is True and metrics.get("retry_start") == "original_base"
        and metrics.get("optimizer_rng_resume_supported") is False, "INDEPENDENT_BASE_OR_ADAPTER_UPDATE_PROOF_MISMATCH")
    require(_finite(metrics.get("gradient_norms_before_clip"), steps) and len(metrics.get("gradient_reports", [])) == steps,
        "FULL_GRADIENT_AUDIT_REQUIRED")
    seen = {p: {"A": False, "B": False} for p in before}
    for report in metrics["gradient_reports"]:
        projections = report.get("projections", [])
        require(report.get("all_finite") is True and len(projections) == len(before) and {p["path"] for p in projections} == set(before), "GRADIENT_PROJECTION_COVERAGE_MISMATCH")
        for projection in projections:
            for label in ("A", "B"):
                tensor = projection["tensors"][label]
                require(tensor.get("finite") is True and type(tensor.get("nonzero")) is bool
                    and math.isfinite(tensor.get("l2_norm", math.nan)), "NONFINITE_PROJECTION_GRADIENT")
                seen[projection["path"]][label] |= tensor.get("nonzero") is True
    require(all(v for tensors in seen.values() for v in tensors.values()), "PER_PROJECTION_NONZERO_GRADIENT_NOT_OBSERVED")
    coverage = metrics.get("native_contract", {}).get("coverage", {})
    require(coverage.get("base_frozen") is True and coverage.get("vision_projector_embeddings_head_frozen") is True,
        "FROZEN_NATIVE_COVERAGE_PROOF_REQUIRED")
    contract = metrics["native_contract"]
    preconfig = _read(config["data"]["preflight_config"]["path"])
    architecture = _read(Path(preconfig["output_dir"]) / "architecture.json")
    require(all(contract.get(k) == architecture.get(k) for k in ("architecture", "template", "thinking"))
        and contract.get("weights_loaded") is True and contract.get("cuda_residency_verified") is True
        and contract.get("files", {}).get("all_prepared_files_rehashed") is True
        and contract["files"].get("model_files_hash") == config["model"]["model_files_hash"]
        and contract.get("revision") == config["model"]["revision"]
        and contract.get("snapshot_path") == config["model"]["snapshot_path"]
        and contract.get("quantization", {}).get("method") == "nf4"
        and contract["quantization"].get("double_quant") is True
        and contract["quantization"].get("compute_dtype") == "bfloat16"
        and contract.get("gradient_checkpointing") is True, "NATIVE_GPU_CONTRACT_MISMATCH")
    paths = contract["architecture"]["exact_target_modules"]
    require(paths and len(set(paths)) == len(paths) and set(paths) == set(before)
        and coverage.get("exact_target_modules") == paths and coverage.get("target_count") == len(paths)
        and contract["architecture"].get("target_count") == len(paths)
        and coverage.get("target_modules_sha256") == canonical_hash(paths)
        and coverage.get("quantized_targets_verified") is True, "NATIVE_PROJECTION_INVENTORY_MISMATCH")
    return metrics


def _checkpoint(config, directory, metrics):
    files = {p.relative_to(directory / "adapter").as_posix(): file_hash(p)
             for p in sorted((directory / "adapter").rglob("*")) if p.is_file()}
    processors = {p.relative_to(directory / "processor").as_posix(): file_hash(p)
                  for p in sorted((directory / "processor").rglob("*")) if p.is_file()}
    require(files and any(n.startswith("adapter_model") and n.endswith(".safetensors") for n in files), "ADAPTER_WEIGHTS_MISSING")
    require(bool(processors), "SAVED_PROCESSOR_MISSING")
    require(not any(p.is_symlink() for root in (directory / "adapter", directory / "processor") for p in root.rglob("*")), "CHECKPOINT_SYMLINK_FORBIDDEN")
    value = {"complete": True, "final_checkpoint": True, "selection_rule": "ONE_PASS_FINAL_NOT_MINIMUM_LOSS",
        **{k: config[k] for k in IDENTITY_KEYS}, "optimizer_steps": config["recipe"]["optimizer_steps"],
        "samples_processed": config["recipe"]["examples"], "sample_order_sha256": config["data"]["sample_order_sha256"],
        "adapter_sha256": metrics["adapter_sha256"], "adapter_files": files, "processor_files": processors,
        "resume_supported": False, "optimizer_rng_saved": False}
    atomic_json(directory / "checkpoint.json", value)
    return value


def _verify_checkpoint(config, directory):
    value = _read(directory / "checkpoint.json")
    require(value.get("complete") is True and value.get("final_checkpoint") is True
        and value.get("selection_rule") == "ONE_PASS_FINAL_NOT_MINIMUM_LOSS"
        and all(value.get(k) == config[k] for k in IDENTITY_KEYS)
        and value.get("optimizer_steps") == config["recipe"]["optimizer_steps"]
        and value.get("samples_processed") == config["recipe"]["examples"]
        and value.get("sample_order_sha256") == config["data"]["sample_order_sha256"]
        and value.get("resume_supported") is False and value.get("optimizer_rng_saved") is False, "FINAL_CHECKPOINT_IDENTITY_MISMATCH")
    for root, key in ((directory / "adapter", "adapter_files"), (directory / "processor", "processor_files")):
        require(root.is_dir() and not root.is_symlink() and not any(p.is_symlink() for p in root.rglob("*")), "CHECKPOINT_SYMLINK_FORBIDDEN")
        actual = {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(root.rglob("*")) if p.is_file()}
        require(bool(actual) and actual == value[key], "FINAL_CHECKPOINT_ARTIFACT_CHANGED")
    require(any(n.startswith("adapter_model") and n.endswith(".safetensors") for n in value["adapter_files"]), "ADAPTER_WEIGHTS_MISSING")
    return value


def _final_selection(config, directory):
    checkpoint = _verify_checkpoint(config, directory)
    return {"schema_version": "medvlm.v6.final-checkpoint-selection.v1", **{k: config[k] for k in IDENTITY_KEYS},
        "selection_rule": "ONE_PASS_FINAL_NOT_MINIMUM_LOSS", "checkpoint_sha256": file_hash(directory / "checkpoint.json"),
        "adapter_sha256": checkpoint["adapter_sha256"], "optimizer_steps": checkpoint["optimizer_steps"],
        "samples_processed": checkpoint["samples_processed"], "dev_test_read": False,
        "historical_selection_files_modified": False, "optimizer_rng_resume_supported": False}


def _child_identity():
    from .campaign_supervisor import process_identity
    value = process_identity()
    require(value is not None, "CHILD_PROCESS_IDENTITY_UNAVAILABLE")
    return value


def train_checkpoint(config_path, directory, receipt_path):
    config = _read(config_path)
    directory, receipt_path = _absolute(directory), _absolute(receipt_path)
    _inherited_supervisor_fds(config)
    rows, records = _validate(config)
    attempt = _read(directory / "attempt.json")
    require(_read(directory / "config.json") == config and attempt.get("status") == "STARTED"
        and all(attempt.get(k) == config[k] for k in IDENTITY_KEYS)
        and not receipt_path.exists() and not any((directory / n).exists() for n in
            ("adapter", "processor", "train.metrics.json", "checkpoint.json", "final_selection.json")),
        "FRESH_ORIGINAL_BASE_TRAIN_STAGE_REQUIRED")
    writer = telemetry_module.attempt_writer(config, directory, role="training")
    writer.start_sampler()
    try:
        metrics = _train(config, rows, records, directory, telemetry=writer) if writer.enabled else _train(config, rows, records, directory)
    finally:
        writer.close()
    _metrics_proof(config, metrics, records)
    require(metrics.get("training_pid") == os.getpid(), "TRAINING_PROCESS_IDENTITY_MISMATCH")
    atomic_json(directory / "train.metrics.json", metrics)
    value = {"status": "SUCCEEDED", "scope": "NATIVE_MAIN_ONE_PASS_TRAIN_ONLY",
        **{k: config[k] for k in IDENTITY_KEYS}, "training_pid": os.getpid(), "process_identity": _child_identity(),
        "optimizer_steps": config["recipe"]["optimizer_steps"], "samples_processed": config["recipe"]["examples"],
        "metrics_sha256": file_hash(directory / "train.metrics.json"), "retry_start": "original_base", "finished_at": utcnow()}
    atomic_json(receipt_path, value)
    return value


def _invoke(args, config, invoke=None):
    """Capture actual child identity before waiting; no timeout or second owner."""
    from types import SimpleNamespace
    from .campaign_supervisor import process_identity
    fds = _inherited_supervisor_fds(config)
    if invoke is not None:
        return invoke(args, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, pass_fds=fds)
    child = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, pass_fds=fds)
    identity = process_identity(child.pid)
    stdout, stderr = child.communicate()
    return SimpleNamespace(returncode=child.returncode, stdout=stdout, stderr=stderr,
                           pid=child.pid, process_identity=identity)


def _process(stage, config_path, directory, config, invoke=None):
    _require_parent_without_cuda()
    path = directory / (stage + ".json")
    args = [sys.executable, "-m", "mediflow_medvlm.campaign_main_train", "--config", str(config_path),
        "--" + stage + "-checkpoint", str(directory), "--receipt", str(path)]
    result = _invoke(args, config, invoke)
    for name in ("stdout", "stderr"):
        data = getattr(result, name, b"") or b""
        (directory / (stage + "." + name + ".private.log")).write_bytes(data.encode() if isinstance(data, str) else data)
    require(result.returncode == 0 and path.is_file(), stage.upper() + "_PROCESS_FAILED")
    value = _read(path)
    pid = value.get("training_pid") if stage == "train" else value.get("pid")
    require(value.get("status") == "SUCCEEDED" and all(value.get(k) == config[k] for k in IDENTITY_KEYS)
        and type(pid) is int and pid > 0 and pid != os.getpid(), "CHILD_PROCESS_PROOF_INVALID")
    identity = value.get("process_identity", {})
    require(identity.get("worker_pid") == pid and identity.get("boot_identity")
        and identity.get("process_start_time"), "CHILD_PROCESS_IDENTITY_INVALID")
    if getattr(result, "pid", None) is not None:
        require(result.pid == pid and result.process_identity == identity, "ACTUAL_CHILD_IDENTITY_MISMATCH")
    _require_parent_without_cuda()
    return value, {"pid": pid, "process_identity": identity, "exit_code": result.returncode, "exit_verified": True}


def reload_checkpoint(config_path, directory, receipt_path):
    """Fresh final reload and trained-example sanity probe, never evaluation."""
    config = _read(config_path)
    _inherited_supervisor_fds(config)
    import torch
    from .training.v6_gpu import load_v6_gpu_bundle, adapter_digest
    rows, records = _validate(config)
    directory = _absolute(directory)
    checkpoint = _verify_checkpoint(config, directory)
    _inherited_supervisor_fds(config)
    bundle = load_v6_gpu_bundle(config["model"], config["cell_id"], fixed_template_date=config["fixed_template_date"],
                                adapter_path=directory / "adapter", processor_path=directory / "processor", training=False)
    _verify_native_contract(config, bundle.contract)
    training_contract = _read(directory / "metrics.json")["native_contract"]
    for key in ("loader", "revision", "quantization", "snapshot_path", "processor_class", "native_runtime_compatibility"):
        if bundle.contract.get(key) != training_contract.get(key):
            raise MainTrainError("RELOAD_NATIVE_RUNTIME_CONTRACT_CHANGED")
    signature = bundle.contract["parameter_storage"]["base_dtype_signature"]
    if signature != training_contract["parameter_storage"]["base_dtype_signature"]:
        raise MainTrainError("RELOAD_BASE_DTYPE_SIGNATURE_CHANGED")
    if adapter_digest(bundle.model) != checkpoint["adapter_sha256"]:
        raise MainTrainError("RELOADED_ADAPTER_WEIGHTS_DIFFER")
    record = records[-1]
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
        raise MainTrainError("TRAINED_EXAMPLE_GENERATION_CONTEXT_EXHAUSTED")
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        generated = bundle.model.generate(**moved, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
    continuation = generated[0, prompt_length:]
    count = continuation.shape[-1]
    if count < 1:
        raise MainTrainError("RELOAD_GENERATED_NO_TOKENS")
    tokenizer = getattr(bundle.processor, "tokenizer", bundle.processor)
    raw = tokenizer.decode(continuation, skip_special_tokens=False)
    thinking = verify_native_no_thinking_output(raw)
    atomic_json(directory / "reload.generated.private.json", {"trained_example_sample_id": record["sample_id"],
        "generated_ids": continuation.tolist(), "generated_text": raw})
    _verify_checkpoint(config, directory)
    _validate(config)
    result = {"status": "SUCCEEDED", **{k: config[k] for k in IDENTITY_KEYS}, "pid": os.getpid(), "process_identity": _child_identity(), "probe_scope": "TRAINING_EXAMPLE_SANITY_CHECK",
        "fresh_process_required": True, "adapter_weights_verified": True, "adapter_sha256": checkpoint["adapter_sha256"],
        "trained_example_sample_id": record["sample_id"], "trained_example_loss": float(loss), "target_tokens": tokens,
        "finite_forward": True, "base_dtype_signature": signature, "generated_tokens": count, "generation_max_new_tokens": max_new_tokens,
        "generated_ids_sha256": canonical_hash(continuation.tolist()), "no_thinking_output": thinking,
        "dev_test_read": False, "runtime_versions": runtime_versions(), "finished_at": utcnow()}
    atomic_json(receipt_path, result)
    return result



def _reload_proof(config, directory, metrics, records):
    value = _read(directory / "reload.json")
    checkpoint = _verify_checkpoint(config, directory)
    record = records[-1]
    generated = _read(directory / "reload.generated.private.json")
    ids = generated.get("generated_ids", [])
    from .training.v6_native import verify_native_no_thinking_output
    require(value.get("status") == "SUCCEEDED" and all(value.get(k) == config[k] for k in IDENTITY_KEYS)
        and value.get("probe_scope") == "TRAINING_EXAMPLE_SANITY_CHECK"
        and value.get("trained_example_sample_id") == generated.get("trained_example_sample_id") == record["sample_id"] == config["data"]["probe_sample_id"]
        and value.get("adapter_sha256") == checkpoint["adapter_sha256"] == metrics["adapter_sha256"]
        and value.get("finite_forward") is True and value.get("adapter_weights_verified") is True
        and value.get("fresh_process_required") is True and value.get("dev_test_read") is False
        and type(value.get("pid")) is int and value["pid"] > 0
        and value["pid"] not in {metrics["training_pid"], metrics["process_isolation"]["coordinator_pid"]}
        and math.isfinite(value.get("trained_example_loss", math.nan))
        and value.get("target_tokens") == record["native"]["supervised_tokens_after_shift"]
        and value.get("base_dtype_signature") == metrics["native_contract"]["parameter_storage"]["base_dtype_signature"]
        and value.get("runtime_versions") == config["runtime_versions"] and ids
        and all(type(token) is int and token >= 0 for token in ids)
        and value.get("generated_tokens") == len(ids) and value.get("generated_ids_sha256") == canonical_hash(ids)
        and type(value.get("generation_max_new_tokens")) is int and 0 < len(ids) <= value["generation_max_new_tokens"] <= 32
        and value.get("no_thinking_output") == verify_native_no_thinking_output(generated["generated_text"]), "FINAL_RELOAD_PROOF_INVALID")
    isolation = metrics["process_isolation"]
    require(isolation.get("coordinator_cuda_initialized") is False and isolation.get("training_exited_before_reload") is True
        and isolation.get("train", {}).get("exit_verified") is True and isolation["train"].get("exit_code") == 0
        and isolation["train"].get("pid") == metrics["training_pid"]
        and isolation.get("reload", {}).get("exit_verified") is True and isolation["reload"].get("exit_code") == 0
        and isolation["reload"].get("pid") == value["pid"]
        and isolation["reload"]["process_identity"] == value.get("process_identity"), "SEQUENTIAL_CHILD_EXIT_PROOF_REQUIRED")
    return value


def _verify_output(config, directory, records):
    metrics = _read(directory / "metrics.json")
    _metrics_proof(config, metrics, records)
    child = _read(directory / "train.json")
    original = _read(directory / "train.metrics.json")
    require(child.get("scope") == "NATIVE_MAIN_ONE_PASS_TRAIN_ONLY" and child.get("status") == "SUCCEEDED"
        and all(child.get(k) == config[k] for k in IDENTITY_KEYS)
        and child.get("metrics_sha256") == file_hash(directory / "train.metrics.json")
        and child.get("training_pid") == metrics["training_pid"]
        and child.get("process_identity") == metrics["process_isolation"]["train"]["process_identity"]
        and child.get("retry_start") == "original_base"
        and child.get("optimizer_steps") == config["recipe"]["optimizer_steps"]
        and child.get("samples_processed") == config["recipe"]["examples"]
        and original == {k: v for k, v in metrics.items() if k not in {"process_isolation", "gpu_accounting"}},
        "TRAIN_CHILD_METRICS_PROOF_INVALID")
    checkpoint = _verify_checkpoint(config, directory)
    require(checkpoint["adapter_sha256"] == metrics["adapter_sha256"]
        and _read(directory / "final_selection.json") == _final_selection(config, directory), "FINAL_SELECTION_PROOF_INVALID")
    _reload_proof(config, directory, metrics, records)
    return metrics


def _completed_output(config, records):
    output = Path(config["output_dir"])
    require((output / "completion.json").is_file(), "INCOMPLETE_OUTPUT_NO_OPTIMIZER_RNG_RESUME")
    marker = _read(output / "completion.json")
    require(marker.get("status") == "SUCCEEDED" and marker.get("scope") == SCOPE
        and all(marker.get(k) == config[k] for k in IDENTITY_KEYS)
        and marker.get("optimizer_steps") == config["recipe"]["optimizer_steps"]
        and marker.get("samples_processed") == config["recipe"]["examples"]
        and marker.get("excluded_count") == 0 and marker.get("full_one_pass_verified") is True
        and marker.get("final_checkpoint_verified") is True and marker.get("fresh_process_reload_verified") is True
        and marker.get("v6_main_cell_training_completed") is True and marker.get("campaign_terminal") is False
        and marker.get("dev_test_read") is False, "EXISTING_FINAL_RECEIPT_IDENTITY_MISMATCH")
    actual = _tree_files(output)
    artifacts = marker.get("artifacts", [])
    expected = {str(_absolute(a["path"]).relative_to(output)): a["sha256"] for a in artifacts}
    require(len(expected) == len(artifacts) and actual == expected and {"config.json", "train.json", "train.metrics.json",
        "metrics.json", "checkpoint.json", "final_selection.json", "reload.json", "reload.generated.private.json", "attempt.json"} <= actual.keys(),
        "EXISTING_FINAL_ARTIFACT_CHANGED")
    require(_read(output / "config.json") == config, "EXISTING_FINAL_CONFIG_MISMATCH")
    _verify_output(config, output, records)
    return {**marker, "artifacts": artifacts + [{"path": str(output / "completion.json"), "sha256": file_hash(output / "completion.json")}]}


def verify_result(config, *, repo=None):
    """CPU-only exact-artifact and semantic validator for supervisor aggregates."""
    validate_config(config, repo=repo)
    freeze, plan, _ = _freeze(config["data"]["common_freeze"]["path"])
    _, _, records, _, _ = _inputs(freeze, plan, config["cell_id"])
    return _completed_output(config, records)


def run_train(config_path, receipt_path, *, invoke=None):
    config_path, receipt_path = _absolute(config_path), _absolute(receipt_path)
    config, initial_bytes = _read(config_path), file_hash(config_path)
    require(str(receipt_path) == config["success_receipt"], "SUCCESS_RECEIPT_PATH_MISMATCH")
    _require_parent_without_cuda()
    rows, records = _validate(config)
    output = _absolute(config["output_dir"])
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with output.with_name(output.name + ".worker.lock").open("a") as lock:
        os.chmod(lock.name, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if output.exists():
            result = _completed_output(config, records)
            require(file_hash(config_path) == initial_bytes, "CONFIG_CHANGED_DURING_REUSE")
            require(not receipt_path.exists() or _read(receipt_path) == result, "EXISTING_FINAL_RECEIPT_MISMATCH")
            if not receipt_path.exists(): atomic_json(receipt_path, result)
            return result
        require(not receipt_path.exists(), "SUCCESS_RECEIPT_WITHOUT_FINAL_OUTPUT")
        staging = Path(tempfile.mkdtemp(prefix=output.name + ".attempt-", dir=output.parent))
        old_umask = os.umask(0o077)
        telemetry = telemetry_module.TelemetryWriter()
        try:
            atomic_json(staging / "attempt.json", {"status": "STARTED", **{k: config[k] for k in IDENTITY_KEYS},
                "attempt_id": uuid.uuid4().hex, "retry_start": "original_base", "optimizer_rng_resume_supported": False,
                "started_at": utcnow()})
            atomic_json(staging / "config.json", config)
            telemetry = telemetry_module.attempt_writer(config, staging)
            telemetry.event("model_loading")
            telemetry.observe_steps()
            _progress(config, 0, 0, "start_original_base_train_child")
            ledger = BudgetLedger(config["gpu_ledger_path"], policy=ResourcePolicy())
            ledger.start(config["model_id"], config["run_id"], "native_main_train")
            try:
                train, train_exit = _process("train", config_path, staging, config, invoke)
            finally:
                ledger.finish()
                telemetry.stop_observer()
            require(train.get("metrics_sha256") == file_hash(staging / "train.metrics.json"), "TRAIN_METRICS_HASH_MISMATCH")
            metrics = _read(staging / "train.metrics.json")
            _metrics_proof(config, metrics, records)
            require(metrics.get("training_pid") == train_exit["pid"] and train.get("scope") == "NATIVE_MAIN_ONE_PASS_TRAIN_ONLY",
                "TRAIN_PROCESS_PROOF_INVALID")
            metrics["process_isolation"] = {"coordinator_pid": os.getpid(), "coordinator_cuda_initialized": False,
                "training_exited_before_reload": True, "train": train_exit}
            metrics["gpu_accounting"] = {"ledger_path": config["gpu_ledger_path"], "train_seconds": ledger.session_seconds,
                "snapshot_after_train": ledger.snapshot()}
            atomic_json(staging / "metrics.json", metrics)
            _validate(config)
            _checkpoint(config, staging, metrics)
            atomic_json(staging / "final_selection.json", _final_selection(config, staging))
            telemetry.record_artifacts(staging)
            telemetry.snapshot(training_succeeded=True)
            telemetry.event("checkpoint_saved")
            telemetry.event("reload_started")
            _progress(config, config["recipe"]["optimizer_steps"], len(records), "fresh_final_reload_child", str(staging / "checkpoint.json"))
            reload_ledger = BudgetLedger(config["gpu_ledger_path"], policy=ResourcePolicy())
            reload_ledger.start(config["model_id"], config["run_id"], "native_main_reload")
            try:
                _, reload_exit = _process("reload", config_path, staging, config, invoke)
            finally:
                reload_ledger.finish()
            metrics["process_isolation"]["reload"] = reload_exit
            metrics["gpu_accounting"].update(reload_seconds=reload_ledger.session_seconds, snapshot_after_reload=reload_ledger.snapshot())
            atomic_json(staging / "metrics.json", metrics)
            _validate(config)
            require(file_hash(config_path) == initial_bytes, "CONFIG_CHANGED_DURING_ONE_PASS")
            _verify_output(config, staging, records)
            telemetry.record_artifacts(staging, reload_verified=True)
            telemetry.event("reload_succeeded")
            telemetry.finish(metrics, reload_succeeded=True)
            for path in staging.rglob("*"):
                if path.is_file():
                    os.chmod(path, 0o600)
                    with path.open("rb") as stream: os.fsync(stream.fileno())
                elif path.is_dir(): os.chmod(path, 0o700)
            result = {"schema_version": "medvlm.v6.main-train-receipt.v1", "status": "SUCCEEDED", "scope": SCOPE,
                **{k: config[k] for k in IDENTITY_KEYS}, "cell_id": config["cell_id"],
                "optimizer_steps": config["recipe"]["optimizer_steps"], "samples_processed": config["recipe"]["examples"],
                "excluded_count": 0, "full_one_pass_verified": True, "final_checkpoint_verified": True,
                "fresh_process_reload_verified": True, "v6_main_cell_training_completed": True,
                "dev_test_read": False, "campaign_terminal": False,
                "artifacts": [{"path": str(output / name), "sha256": sha} for name, sha in _tree_files(staging).items()], "finished_at": utcnow()}
            atomic_json(staging / "completion.json", result)
            for directory in [p for p in staging.rglob("*") if p.is_dir()] + [staging]:
                fd = os.open(directory, os.O_DIRECTORY)
                try: os.fsync(fd)
                finally: os.close(fd)
            os.rename(staging, output)
            fd = os.open(output.parent, os.O_DIRECTORY)
            try: os.fsync(fd)
            finally: os.close(fd)
            result = _completed_output(config, records)
            atomic_json(receipt_path, result)
            return result
        except BaseException as exc:
            telemetry.stop_observer()
            if telemetry.status.get("phase") == "reloading": telemetry.snapshot(reload_succeeded=False)
            telemetry.event("failed")
            if staging.exists():
                atomic_json(staging / "failure.private.json", {"exception_type": type(exc).__name__,
                    "exception_message": str(exc), "traceback": traceback.format_exc()})
                atomic_json(staging / "failure.json", {"status": "FAILED", "exception_type": type(exc).__name__,
                    "reason_code": str(exc) if isinstance(exc, MainTrainError) else "NATIVE_FAILURE",
                    "retry_start": "original_base", "optimizer_rng_resume_supported": False, "finished_at": utcnow()})
            raise
        finally:
            telemetry.close()
            os.umask(old_umask)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--receipt", required=True)
    stages = parser.add_mutually_exclusive_group()
    stages.add_argument("--train-checkpoint")
    stages.add_argument("--reload-checkpoint")
    args = parser.parse_args(argv)
    try:
        stage = args.train_checkpoint or args.reload_checkpoint
        if args.train_checkpoint: result = train_checkpoint(_absolute(args.config), _absolute(stage), _absolute(args.receipt))
        elif args.reload_checkpoint: result = reload_checkpoint(_absolute(args.config), _absolute(stage), _absolute(args.receipt))
        else: result = run_train(args.config, args.receipt)
    except Exception as exc:
        stage = args.train_checkpoint or args.reload_checkpoint
        if stage:
            atomic_json(Path(stage) / "stage.failure.private.json", {"exception_type": type(exc).__name__,
                "exception_message": str(exc), "traceback": traceback.format_exc()})
        print(json.dumps({"status": "FAILED", "scope": SCOPE, "exception_type": type(exc).__name__,
            **({"reason_code": str(exc)} if isinstance(exc, MainTrainError) else {})}), file=sys.stderr)
        return 1
    print(json.dumps({"status": result["status"], "scope": SCOPE}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
