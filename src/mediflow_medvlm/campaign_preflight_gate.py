"""CPU-only audit of the fixed eight-model, ten-cell native preflight/smoke gate.

Planning: build_config(active_manifest_path, state_path, private_output, repo=SNAPSHOT),
then save it once and pass activation_manifest(config_path, current_manifest) to the
existing Supervisor.activate API. Neither planner nor worker advances the queue.
The config binds only terminal prerequisite records, so activation and the gate's
own attempt cannot invalidate its input snapshot. No intersection is frozen here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import sys
import tempfile

from . import campaign_gpu_smoke as gpu
from . import campaign_preflight as cpu
from .campaign_supervisor import (HASH_FIELDS, TERMINAL, atomic_json, canonical_hash,
                                  file_hash, validate_manifest, verify_source, verify_success)
from .campaign_v6 import CELLS, cell_spec
from .runtime import utcnow

SCHEMA = "medvlm.v6.preflight-gate-config.v1"
SCOPE = "NATIVE_PREFLIGHT_SMOKE_AGGREGATE_ONLY"
JOB_ID = "v6-preflight_smoke"
SOURCE_FILES = tuple(dict.fromkeys((*gpu.SOURCE_FILES, "campaign_preflight_gate.py")))
IDENTITY_KEYS = cpu.IDENTITY_KEYS
OUTPUT_FILES = {"config.json", "summary.json"}
CELL_IDS = [cell[0] for cell in CELLS]
PREPARE_IDS = ["prepare-" + cell for cell in CELL_IDS if cell not in {"V2", "V3"}]
PREREQUISITE_IDS = PREPARE_IDS + [f"v6-native-{kind}-{cell}"
    for kind in ("preflight", "smoke") for cell in CELL_IDS]


class GateError(ValueError):
    """A controlled error code, without private sample or prediction contents."""


def _require(condition, code):
    if not condition:
        raise GateError(code)


def _absolute(path):
    path = Path(path)
    _require(path.is_absolute() and ".." not in path.parts, "ABSOLUTE_NORMALIZED_PATH_REQUIRED")
    return path


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _equal(first, second):
    return canonical_hash(first) == canonical_hash(second)


def source_identity(repo=None):
    package = Path(__file__).resolve().parent if repo is None else Path(repo) / "src/mediflow_medvlm"
    files = {"src/mediflow_medvlm/" + name: file_hash(package / name) for name in SOURCE_FILES}
    return files, canonical_hash(files)


def _prepare_id(cell):
    return "prepare-" + {"V2": "T3", "V3": "T4"}.get(cell, cell)


def _snapshot(manifest, state):
    validate_manifest(manifest)
    _require(manifest.get("campaign_id") == state.get("campaign_id"), "CAMPAIGN_IDENTITY_MISMATCH")
    declared = manifest.get("cells", [])
    _require(len(declared) == 10 and {cell.get("cell_id") for cell in declared} == set(CELL_IDS),
             "EXACT_TEN_DECLARED_CELLS_REQUIRED")
    _require(all(all(cell.get(key) == value for key, value in cell_spec(cell["cell_id"]).items())
                 for cell in declared) and len({cell["model_id"] for cell in declared}) == 8,
             "EXACT_EIGHT_NATIVE_MODELS_REQUIRED")
    jobs = {job["job_id"]: job for job in manifest["jobs"]}
    for phase, expected in (("model_preparation", PREPARE_IDS),
            ("native_preflight", ["v6-native-preflight-" + cell for cell in CELL_IDS]),
            ("native_smoke", ["v6-native-smoke-" + cell for cell in CELL_IDS])):
        _require({job["job_id"] for job in jobs.values() if job["phase"] == phase} == set(expected),
                 "EXACT_PREREQUISITE_JOB_SET_REQUIRED")
    gate = jobs.get(JOB_ID, {})
    _require(gate.get("phase") == "preflight_smoke" and gate.get("cell_id") == "ALL_V6"
             and gate.get("run_id") == JOB_ID and gate.get("dependencies") == PREREQUISITE_IDS
             and gate.get("dependency_policy") == "terminal", "AGGREGATE_GATE_IDENTITY_MISMATCH")
    for job_id in PREREQUISITE_IDS:
        job, record = jobs[job_id], state["jobs"][job_id]
        _require(record.get("status") in TERMINAL, "PREREQUISITE_NOT_TERMINAL")
        _require(all(type(record.get(key)) is int and record[key] >= 0
                     for key in ("attempts", "failure_count", "recovery_attempts")), "INVALID_CUMULATIVE_COUNTERS")
        if job_id in PREPARE_IDS:
            cell = job_id.removeprefix("prepare-")
            _require(job["cell_id"] == cell and job["run_id"] == "v6-" + job_id
                     and not job.get("uses_gpu"), "PREPARATION_JOB_IDENTITY_MISMATCH")
        else:
            cell = job["cell_id"]
            smoke = job["phase"] == "native_smoke"
            _require(cell in CELL_IDS and job_id == f"v6-native-{'smoke' if smoke else 'preflight'}-{cell}"
                     and job["run_id"] == job_id and all(job.get(k) == v for k, v in cell_spec(cell).items())
                     and job.get("uses_gpu") is smoke
                     and job.get("dependencies") == ["v6-native-preflight-" + cell if smoke else _prepare_id(cell)],
                     "NATIVE_JOB_IDENTITY_MISMATCH")
    return {"declaration": {k: v for k, v in manifest.items() if k != "jobs"},
            "jobs": {key: jobs[key] for key in PREREQUISITE_IDS},
            "states": {key: state["jobs"][key] for key in PREREQUISITE_IDS}}


def _active(state_path):
    state = _read(state_path)
    path = _absolute(state["manifest_path"])
    manifest = _read(path)
    _require(canonical_hash(manifest) == state["manifest_hash"], "ACTIVE_MANIFEST_HASH_MISMATCH")
    return manifest, state


def _success(job, record):
    path = _absolute(job["success_receipt"])
    _require(not path.is_symlink(), "SYMLINK_RECEIPT_FORBIDDEN")
    verification = record.get("verification", {})
    _require(record.get("exit_code") == 0 and record["attempts"] > 0
             and verification.get("receipt") == str(path)
             and verification.get("receipt_sha256") == file_hash(path), "TERMINAL_SUCCESS_PROOF_MISMATCH")
    receipt = _read(path)
    artifacts = receipt.get("artifacts", [])
    _require(len({item["path"] for item in artifacts}) == len(artifacts)
             and all(not _absolute(item["path"]).is_symlink() for item in artifacts), "INVALID_ARTIFACT_PATH_SET")
    verify_success(job)  # Rehash every receipt artifact, including adapter weights.
    _require(verification.get("artifact_count") == len(artifacts), "VERIFIED_ARTIFACT_COUNT_MISMATCH")
    return receipt


def _native_config(job, receipt, smoke):
    argv = job["argv"]
    module = "mediflow_medvlm.campaign_gpu_smoke" if smoke else "mediflow_medvlm.campaign_preflight"
    _require(argv[1:3] == ["-m", module] and argv.count("--config") == 1, "NATIVE_COMMAND_MISMATCH")
    path = _absolute(argv[argv.index("--config") + 1])
    config = _read(path)
    _require(config.get("schema_version") == (gpu.SCHEMA if smoke else cpu.SCHEMA)
             and config.get("config_hash") == canonical_hash({k: v for k, v in config.items() if k != "config_hash"}),
             "NATIVE_CONFIG_HASH_MISMATCH")
    _require(all(_equal(config.get(key), job.get(key)) for key in (*IDENTITY_KEYS, "source_files"))
             and all(config.get(key) == value for key, value in cell_spec(job["cell_id"]).items())
             and canonical_hash(config["model"]) == config["model_hash"]
             and canonical_hash(config["data"]) == config["data_hash"], "NATIVE_CONFIG_JOB_IDENTITY_MISMATCH")
    _require(config.get("max_length") == 2048 and config.get("fixed_template_date") == "2026-11-14"
             and config.get("device") == ("cuda" if smoke else "cpu")
             and config.get("local_files_only") is True and config.get("trust_remote_code") is False,
             "NATIVE_FROZEN_RUNTIME_POLICY_MISMATCH")
    if smoke:
        _require(config.get("scope") == gpu.SCOPE
                 and _equal(config.get("recipe"), {**gpu.RECIPE, "learning_rate": 2e-4 if config["task"] == "text" else 1e-4}),
                 "NATIVE_SMOKE_RECIPE_MISMATCH")
    completed = gpu._completed_output(config) if smoke else cpu._completed_output(config)
    _require(_equal(completed, receipt), "NATIVE_COMPLETION_RECEIPT_MISMATCH")
    return config, {"path": str(path), "sha256": file_hash(path)}


def _finite(values, length):
    return isinstance(values, list) and len(values) == length and all(
        type(value) in (int, float) and math.isfinite(value) for value in values)


def _gpu_proof(config, receipt, preconfig, records):
    output = _absolute(config["output_dir"])
    metrics, reload, checkpoint = (_read(output / name) for name in ("metrics.json", "reload.json", "checkpoint.json"))
    _require(receipt.get("scope") == gpu.SCOPE and all(receipt.get(key) is True for key in
        ("fresh_process_reload_verified", "gpu_smoke_verified", "adapter_changed"))
        and all(receipt.get(key) is False for key in ("common_train_freeze_decided", "dev_test_read",
            "v6_main_training_completed", "usable_as_main_initialization", "campaign_terminal")), "GPU_SMOKE_SCOPE_OR_PROOF_MISMATCH")
    for value in (receipt, metrics, checkpoint):
        _require(type(value.get("optimizer_steps")) is int and value["optimizer_steps"] == 20
                 and type(value.get("samples_processed")) is int and value["samples_processed"] == 320,
                 "FULL_20_UPDATES_320_SAMPLES_REQUIRED")
    _require(receipt.get("excluded_count") == metrics.get("excluded_count") == 0
             and metrics.get("adapter_changed") is True and metrics.get("all_projection_tensors_changed") is True,
             "FULL_SMOKE_WITHOUT_DROPPED_EXAMPLES_REQUIRED")
    _require(_equal(config["model"], preconfig["model"])
             and config["data"].get("preflight_config_hash") == preconfig["config_hash"]
             and all(_equal(config["data"].get(key), value) for key, value in preconfig["data"].items())
             and _equal(config["data"].get("selection"), gpu._selection(preconfig, records)),
             "CPU_GPU_MODEL_DATA_SELECTION_MISMATCH")
    selected = config["data"]["selection"]["train_records"]
    expected_tokens = [row["native"]["supervised_tokens_after_shift"] for row in selected]
    _require(_finite(metrics.get("example_losses"), 320)
             and _equal(metrics.get("example_supervised_tokens"), expected_tokens)
             and all(type(n) is int and n > 0 for n in expected_tokens)
             and metrics.get("supervised_tokens") == sum(expected_tokens)
             and metrics.get("sequence_tokens") == sum(row["native"]["sequence_tokens"] for row in selected)
             and math.isfinite(metrics.get("loss_mean", math.nan))
             and math.isclose(metrics["loss_mean"], sum(metrics["example_losses"]) / 320, rel_tol=1e-9, abs_tol=1e-9),
             "FULL_TARGET_TOKEN_LOSS_PROOF_MISMATCH")
    rates = metrics.get("learning_rates")
    _require(_finite(rates, 20) and all(math.isclose(value, config["recipe"]["learning_rate"] * (1 - step / 20),
                 rel_tol=1e-9, abs_tol=1e-12) for step, value in enumerate(rates))
             and metrics.get("final_learning_rate") == 0
             and all(config["runtime_versions"].get(key) == value for key, value in preconfig["runtime_versions"].items()),
             "OBSERVED_RECIPE_OR_RUNTIME_MISMATCH")
    contract = metrics["native_contract"]
    architecture = _read(Path(preconfig["output_dir"]) / "architecture.json")
    _require(all(_equal(contract.get(key), architecture.get(key)) for key in ("architecture", "template", "thinking"))
             and contract.get("weights_loaded") is True and contract.get("cuda_residency_verified") is True
             and contract.get("files", {}).get("all_prepared_files_rehashed") is True
             and contract.get("files", {}).get("model_files_hash") == config["model"]["model_files_hash"]
             and contract.get("revision") == config["model"]["revision"]
             and contract.get("snapshot_path") == config["model"]["snapshot_path"]
             and contract.get("quantization", {}).get("method") == "nf4"
             and contract["quantization"].get("double_quant") is True
             and contract["quantization"].get("compute_dtype") == "bfloat16", "NATIVE_GPU_CONTRACT_MISMATCH")
    paths = contract["architecture"]["exact_target_modules"]
    _require(paths and len(set(paths)) == len(paths)
             and _equal(contract.get("coverage", {}).get("exact_target_modules"), paths)
             and contract["coverage"].get("quantized_targets_verified") is True
             and contract["coverage"].get("base_frozen") is True
             and contract["coverage"].get("vision_projector_embeddings_head_frozen") is True
             and contract["coverage"].get("target_count") == len(paths)
             and contract["architecture"].get("target_count") == len(paths)
             and contract["coverage"].get("target_modules_sha256") == canonical_hash(paths),
             "NATIVE_GPU_PROJECTION_COVERAGE_MISMATCH")
    before, after = metrics.get("initial_projection_sha256", {}), metrics.get("final_projection_sha256", {})
    seen = metrics.get("nonzero_gradients_observed", {})
    reports = metrics.get("gradient_reports", [])
    _require(set(before) == set(after) == set(seen) == set(paths)
             and all(all(before[path].get(side) and before[path][side] != after[path].get(side)
                         and seen[path].get(side) is True for side in ("A", "B")) for path in paths)
             and len(reports) == 20 and _finite(metrics.get("gradient_norms_before_clip"), 20),
             "PROJECTION_UPDATE_PROOF_MISMATCH")
    observed = {path: {side: False for side in ("A", "B")} for path in paths}
    for report in reports:
        projections = report.get("projections", [])
        _require(report.get("all_finite") is True and len(projections) == len(paths)
                 and {row.get("path") for row in projections} == set(paths)
                 and all(row.get("tensors", {}).get(side, {}).get("finite") is True
                         for row in projections for side in ("A", "B")), "FINITE_PROJECTION_GRADIENT_PROOF_MISMATCH")
        for row in projections:
            for side in ("A", "B"):
                _require(type(row["tensors"][side].get("nonzero")) is bool, "INVALID_GRADIENT_NONZERO_FLAG")
                observed[row["path"]][side] |= row["tensors"][side]["nonzero"]
    _require(_equal(observed, seen), "NONZERO_GRADIENT_OBSERVATION_MISMATCH")
    _require(all(reload.get(key) == config[key] and checkpoint.get(key) == config[key] for key in IDENTITY_KEYS)
             and reload.get("status") == "SUCCEEDED"
             and all(reload.get(key) is True for key in ("fresh_process_required", "adapter_weights_verified", "finite_forward"))
             and reload.get("dev_test_read") is False
             and type(reload.get("pid")) is int and type(metrics.get("training_pid")) is int
             and reload["pid"] > 0 and metrics["training_pid"] > 0 and reload["pid"] != metrics["training_pid"]
             and metrics.get("initial_adapter_sha256") != metrics.get("adapter_sha256")
             and checkpoint.get("adapter_sha256") == metrics.get("adapter_sha256") == reload.get("adapter_sha256")
             and reload.get("base_dtype_signature") == contract["parameter_storage"]["base_dtype_signature"]
             and reload.get("runtime_versions") == config["runtime_versions"]
             and math.isfinite(reload.get("held_train_loss", math.nan)), "FRESH_RELOAD_PROOF_MISMATCH")
    held = config["data"]["selection"]["held_train_record"]
    generated = _read(output / "reload.generated.private.json")
    ids = generated.get("generated_ids", [])
    _require(reload.get("held_train_sample_id") == generated.get("held_train_sample_id") == held["sample_id"]
             and reload.get("target_tokens") == held["native"]["supervised_tokens_after_shift"]
             and ids and all(type(token) is int and token >= 0 for token in ids)
             and reload.get("generated_tokens") == len(ids)
             and reload.get("generated_ids_sha256") == canonical_hash(ids)
             and type(reload.get("generation_max_new_tokens")) is int
             and len(ids) <= reload["generation_max_new_tokens"] <= 32,
             "HELD_TRAIN_GENERATION_PROOF_MISMATCH")
    from .training.v6_native import verify_native_no_thinking_output
    _require(_equal(reload.get("no_thinking_output"), verify_native_no_thinking_output(generated.get("generated_text"))),
             "NO_THINKING_OUTPUT_PROOF_MISMATCH")
    return {"optimizer_steps": 20, "samples_processed": 320, "excluded_count": 0,
            "fresh_reload_verified": True, "supervised_tokens": sum(expected_tokens),
            "adapter_checkpoint": {"path": str(output / "checkpoint.json"), "sha256": file_hash(output / "checkpoint.json")}}


def _audit(snapshot):
    jobs, states = snapshot["jobs"], snapshot["states"]
    evidence, prepared, preflight = {}, {}, {}
    for job_id in PREREQUISITE_IDS:
        job, record = jobs[job_id], states[job_id]
        evidence[job_id] = {"status": record["status"]}
        if record["status"] != "SUCCEEDED":
            continue
        verify_source(job)
        _require(all(states[dep]["status"] == "SUCCEEDED" for dep in job.get("dependencies", [])
                     if dep in states), "SUCCESS_HAS_UNSUCCESSFUL_DEPENDENCY")
        receipt = _success(job, record)
        evidence[job_id].update(receipt={"path": job["success_receipt"], "sha256": file_hash(job["success_receipt"])})
        if job_id in PREPARE_IDS:
            cell = job["cell_id"]
            _require(receipt.get("model_id") == cell_spec(cell)["model_id"]
                     and receipt.get("scope") == "MODEL_PREPARATION_ONLY", "PREPARED_MODEL_IDENTITY_MISMATCH")
            prepared[job_id] = cpu._prepared(job["success_receipt"])
            continue
        smoke = job["phase"] == "native_smoke"
        config, reference = _native_config(job, receipt, smoke)
        evidence[job_id]["config"] = reference
        cell = job["cell_id"]
        _require(_equal(config["model"], prepared[_prepare_id(cell)]), "PREPARED_NATIVE_MODEL_BINDING_MISMATCH")
        if not smoke:
            checked_config, records, _ = gpu._preflight(job["success_receipt"])
            _require(_equal(checked_config, config) and receipt.get("scope") == "TRAIN_PREFLIGHT_ELIGIBILITY_ONLY",
                     "CPU_PREFLIGHT_PROOF_MISMATCH")
            summary = _read(Path(config["output_dir"]) / "summary.json")
            eligible = [record["sample_id"] for record in records if record.get("eligible") is True]
            excluded = [record["sample_id"] for record in records if record.get("eligible") is False]
            _require(len(eligible) + len(excluded) == len(records)
                     and summary.get("original_count") == config["data"]["expected_count"] == len(records)
                     and summary.get("eligible_count") == receipt.get("eligible_count") == len(eligible)
                     and summary.get("excluded_count") == receipt.get("excluded_count") == len(excluded)
                     and _equal(summary.get("eligible_ids"), eligible) and _equal(summary.get("excluded_ids"), excluded)
                     and summary.get("excluded_ids_sha256") == canonical_hash(excluded), "CPU_ELIGIBILITY_DENOMINATOR_MISMATCH")
            for record in records:
                if record.get("eligible") is True:
                    native = record.get("native", {})
                    _require(all(native.get(key) is True for key in ("full_target_preserved", "prefix_verified",
                                 "prompt_image_pad_mask_verified", "all_tensors_cpu"))
                             and type(native.get("sequence_tokens")) is int and 1 < native["sequence_tokens"] <= 2048
                             and type(native.get("supervised_tokens_after_shift")) is int
                             and 0 < native["supervised_tokens_after_shift"] < native["sequence_tokens"],
                             "CPU_FULL_TARGET_NATIVE_PROOF_MISMATCH")
            _require(summary.get("max_length") == config.get("max_length") == 2048
                     and summary.get("full_targets_preserved") is True and summary.get("dev_test_read") is False
                     and summary.get("common_train_freeze_decided") is False, "CPU_TARGET_CONTEXT_PROOF_MISMATCH")
            preflight[cell] = (config, records)
            evidence[job_id]["eligibility"] = {"path": str(Path(config["output_dir"]) / "eligibility.jsonl"),
                "sha256": file_hash(Path(config["output_dir"]) / "eligibility.jsonl"),
                "eligible_count": summary["eligible_count"], "eligible_ids_sha256": summary["eligible_ids_sha256"],
                "excluded_count": summary["excluded_count"], "data_hash": config["data_hash"]}
        else:
            preconfig, records = preflight[cell]
            cpu_job = jobs["v6-native-preflight-" + cell]
            _require(config["data"].get("preflight_receipt") == cpu_job["success_receipt"]
                     and config["data"].get("preflight_receipt_sha256") == file_hash(cpu_job["success_receipt"]),
                     "GPU_PREFLIGHT_RECEIPT_BINDING_MISMATCH")
            evidence[job_id]["smoke"] = _gpu_proof(config, receipt, preconfig, records)
    cells = []
    for cell in CELL_IDS:
        ids = [_prepare_id(cell), "v6-native-preflight-" + cell, "v6-native-smoke-" + cell]
        statuses = {key: states[key]["status"] for key in ids}
        supported = all(status == "SUCCEEDED" for status in statuses.values())
        row = {**cell_spec(cell), "status": "SUPPORTED" if supported else
               "FAILED" if "FAILED" in statuses.values() else "BLOCKED_PREREQUISITE",
               "prerequisite_statuses": statuses}
        if supported:
            row.update(model=prepared[ids[0]], eligibility=evidence[ids[1]]["eligibility"],
                       gpu_smoke=evidence[ids[2]]["smoke"])
        cells.append(row)
    return {"evidence": evidence, "cells": cells,
            "prepared_model_identities": {key: {k: value[k] for k in ("model_id", "revision", "model_files_hash")}
                                           for key, value in prepared.items()}}


def build_config(manifest_path, state_path, output_dir, *, repo=None, success_receipt=None):
    """Pure planner; save returned JSON once at a new immutable config path."""
    manifest_path, state_path, output_dir = map(_absolute, (manifest_path, state_path, output_dir))
    manifest, state = _active(state_path)
    _require(_absolute(state["manifest_path"]).resolve() == manifest_path.resolve(), "PLANNING_MANIFEST_NOT_ACTIVE")
    snapshot = _snapshot(manifest, state)
    _require(state["jobs"][JOB_ID]["status"] == "IMPLEMENTATION_PENDING"
             and all(type(state["jobs"][JOB_ID].get(key)) is int and state["jobs"][JOB_ID][key] == 0
                     for key in ("attempts", "failure_count", "recovery_attempts"))
             and not state["jobs"][JOB_ID].get("history"),
             "ONLY_UNSTARTED_AGGREGATE_GATE_MAY_BE_PLANNED")
    audit = _audit(snapshot)
    _require(_equal(_snapshot(*_active(state_path)), snapshot), "PREREQUISITES_CHANGED_DURING_AUDIT")
    campaign_root = _absolute(manifest["campaign_root"])
    receipt = _absolute(success_receipt or campaign_root / "receipts" / (JOB_ID + ".json"))
    _private(output_dir, campaign_root)
    _private(receipt, campaign_root)
    files, source_hash = source_identity(repo)
    config = {"schema_version": SCHEMA, "job_id": JOB_ID, "run_id": JOB_ID,
        "campaign_id": manifest["campaign_id"], "scope": SCOPE, "uses_gpu": False,
        "state_path": str(state_path), "output_dir": str(output_dir), "success_receipt": str(receipt),
        "registration_manifest": {"path": str(manifest_path), "sha256": file_hash(manifest_path)},
        "prerequisites": snapshot, "evidence": audit["evidence"],
        "source_files": files, "source_hash": source_hash,
        "data_hash": canonical_hash({"prerequisites": snapshot, "evidence": audit["evidence"]}),
        "model_hash": canonical_hash(audit["prepared_model_identities"]),
        "common_train_freeze_decided": False}
    config["config_hash"] = canonical_hash(config)
    return config


def _private(path, campaign_root):
    _require(path.resolve().is_relative_to(campaign_root.resolve())
             and not any(part in {"reports", "public_summary", ".git"} for part in path.parts), "PRIVATE_CAMPAIGN_OUTPUT_REQUIRED")


def activation_manifest(config_path, manifest, *, python=None, repo=None):
    """Return an activate-compatible manifest; caller owns review/registration."""
    config_path = _absolute(config_path)
    config = _read(config_path)
    _require(_equal(_snapshot(manifest, _read(config["state_path"])), config["prerequisites"]),
             "ACTIVATION_PREREQUISITE_SNAPSHOT_MISMATCH")
    files, source_hash = source_identity(repo)
    _require(files == config["source_files"] and source_hash == config["source_hash"], "ACTIVATION_SOURCE_MISMATCH")
    revised = copy.deepcopy(manifest)
    job = next(row for row in revised["jobs"] if row["job_id"] == JOB_ID)
    job.update(implementation_pending=False, uses_gpu=False, gpu_lock_mode="wrapper",
        argv=[str(python or sys.executable), "-m", "mediflow_medvlm.campaign_preflight_gate",
              "--config", str(config_path), "--receipt", config["success_receipt"]],
        cwd=str(Path(repo).resolve() if repo is not None else Path(__file__).resolve().parents[2]),
        source_files=files, **{key: config[key] for key in HASH_FIELDS}, success_receipt=config["success_receipt"])
    validate_manifest(revised)
    return revised


def _validate(config):
    _require(config.get("schema_version") == SCHEMA and config.get("scope") == SCOPE
             and config.get("job_id") == config.get("run_id") == JOB_ID
             and config.get("uses_gpu") is False and config.get("common_train_freeze_decided") is False
             and config.get("config_hash") == canonical_hash({k: v for k, v in config.items() if k != "config_hash"}),
             "GATE_CONFIG_IDENTITY_MISMATCH")
    _require(_equal(source_identity(), (config["source_files"], config["source_hash"])), "GATE_SOURCE_CHANGED")
    reference = config["registration_manifest"]
    _require(file_hash(_absolute(reference["path"])) == reference["sha256"], "REGISTRATION_MANIFEST_CHANGED")
    manifest, state = _active(_absolute(config["state_path"]))
    snapshot = _snapshot(manifest, state)
    _require(_equal(snapshot, config["prerequisites"]), "TERMINAL_PREREQUISITE_SNAPSHOT_CHANGED")
    audit = _audit(snapshot)
    _require(_equal(audit["evidence"], config["evidence"])
             and config["data_hash"] == canonical_hash({"prerequisites": snapshot, "evidence": audit["evidence"]})
             and config["model_hash"] == canonical_hash(audit["prepared_model_identities"]), "GATE_INPUT_IDENTITY_MISMATCH")
    _require(_equal(_snapshot(*_active(config["state_path"])), snapshot), "PREREQUISITES_CHANGED_DURING_AUDIT")
    for key in IDENTITY_KEYS:
        supplied = os.environ.get("MEDVLM_SUPERVISOR_" + key.upper())
        _require(supplied is None or supplied == config[key], "SUPERVISOR_IDENTITY_MISMATCH_" + key.upper())
    for key in ("output_dir", "success_receipt"):
        _private(_absolute(config[key]), _absolute(manifest["campaign_root"]))
    return audit


def _completed(config):
    output = _absolute(config["output_dir"])
    marker = _read(output / "completion.json")
    _require(marker.get("status") == "SUCCEEDED" and marker.get("scope") == SCOPE
             and all(marker.get(key) == config[key] for key in IDENTITY_KEYS), "EXISTING_GATE_IDENTITY_MISMATCH")
    _require({p.name for p in output.iterdir()} == OUTPUT_FILES | {"completion.json"}
             and not any(p.is_symlink() for p in output.iterdir()), "EXISTING_GATE_ARTIFACT_SET_MISMATCH")
    expected = {str(output / name): file_hash(output / name) for name in OUTPUT_FILES}
    _require(marker.get("artifacts") == [{"path": path, "sha256": expected[path]} for path in sorted(expected)]
             and _equal(_read(output / "config.json"), config), "EXISTING_GATE_ARTIFACT_CHANGED")
    return {**marker, "artifacts": marker["artifacts"] + [{"path": str(output / "completion.json"),
                                                               "sha256": file_hash(output / "completion.json")}]}


def run_gate(config_path, receipt_path):
    config_path, receipt_path = _absolute(config_path), _absolute(receipt_path)
    config = _read(config_path)
    _require(str(receipt_path) == config["success_receipt"], "GATE_RECEIPT_PATH_MISMATCH")
    initial_bytes = file_hash(config_path)
    output = _absolute(config["output_dir"])
    audit = _validate(config)
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with output.with_name(output.name + ".worker.lock").open("a") as lock:
        os.chmod(lock.name, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if output.exists():
            result = _completed(config)
            _require(file_hash(config_path) == initial_bytes, "CONFIG_CHANGED_DURING_GATE_REUSE")
            _require(not receipt_path.exists() or _equal(_read(receipt_path), result), "EXISTING_GATE_RECEIPT_MISMATCH")
            atomic_json(receipt_path, result)
            return result
        _require(not receipt_path.exists(), "GATE_RECEIPT_WITHOUT_OUTPUT")
        staging = Path(tempfile.mkdtemp(prefix=output.name + ".candidate-", dir=output.parent))
        supported = [row["cell_id"] for row in audit["cells"] if row["status"] == "SUPPORTED"]
        counts = {phase: dict(Counter(record["status"] for key, record in config["prerequisites"]["states"].items()
                  if config["prerequisites"]["jobs"][key]["phase"] == phase))
                  for phase in ("model_preparation", "native_preflight", "native_smoke")}
        summary = {"schema_version": "medvlm.v6.preflight-gate-summary.v1", "scope": SCOPE,
            "aggregate_audit_succeeded": True, "all_prerequisites_terminal": True,
            "declared_unique_models": 8, "declared_cells": 10, "prerequisite_jobs": 28,
            "cells": audit["cells"], "supported_cells": supported, "supported_count": len(supported),
            "all_cells_supported": len(supported) == 10, "cell_status_counts": dict(Counter(row["status"] for row in audit["cells"])),
            "prerequisite_status_counts": counts, "terminal_prerequisites": config["prerequisites"]["states"],
            "common_train_freeze_decided": False, "v6_main_training_completed": False,
            "comparison_cohort_frozen": False, "usable_as_main_initialization": False,
            "dev_test_read": False, "gpu_work_performed": False, "base_weight_bytes_rehashed": False,
            "receipt_artifact_bytes_rehashed": True, "finished_at": utcnow()}
        atomic_json(staging / "config.json", config)
        atomic_json(staging / "summary.json", summary)
        _require(_equal(_validate(config), audit) and file_hash(config_path) == initial_bytes, "INPUT_CHANGED_DURING_GATE")
        result = {"schema_version": "medvlm.v6.preflight-gate-receipt.v1", "status": "SUCCEEDED", "scope": SCOPE,
            **{key: config[key] for key in IDENTITY_KEYS}, "supported_cells": supported,
            "all_cells_supported": len(supported) == 10, "common_train_freeze_decided": False,
            "campaign_terminal": False, "artifacts": [{"path": str(output / name), "sha256": file_hash(staging / name)}
                                                        for name in sorted(OUTPUT_FILES)], "finished_at": utcnow()}
        atomic_json(staging / "completion.json", result)
        os.rename(staging, output)
        fd = os.open(output.parent, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        result = _completed(config)
        atomic_json(receipt_path, result)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = run_gate(args.config, args.receipt)
    except Exception as exc:
        value = {"status": "FAILED", "scope": SCOPE, "exception_type": type(exc).__name__}
        if isinstance(exc, GateError):
            value["reason_code"] = str(exc)
        print(json.dumps(value), file=sys.stderr)
        return 1
    print(json.dumps({"status": result["status"], "scope": SCOPE, "supported_cells": result["supported_cells"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
