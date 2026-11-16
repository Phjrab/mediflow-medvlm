"""CPU terminal audit for a recorded namespace reusing an external train freeze.

Unlike the historical full-scaffold gate, this binds ten independent recorded
jobs to their *actual* archived training sources and to the external historical
freeze's original job and terminal state. It never rewrites training identities,
reads held-out sample files, runs inference, or advances the supervisor.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import tempfile

from . import campaign_main_train as main_train
from . import campaign_preflight_gate as gate
from .campaign_supervisor import (HASH_FIELDS, TERMINAL, atomic_json, canonical_hash,
                                  file_hash, validate_manifest, verify_source)
from .campaign_v6 import CELLS, SEEDS, cell_spec
from .runtime import utcnow

JOB_ID = "v6-first_seed_final"
CHILD_IDS = ["v6-first-seed-" + cell[0] for cell in CELLS]
SCHEMA = "medvlm.v6.recorded-first-seed-config.v1"
SCOPE = "RECORDED_NAMESPACE_FIRST_SEED_TERMINAL_AUDIT_ONLY"
IDENTITY_KEYS = gate.IDENTITY_KEYS
SOURCE_FILES = (*main_train.SOURCE_FILES, "campaign_recorded_first_seed.py")
_require, _read, _absolute, _equal = gate._require, gate._read, gate._absolute, gate._equal


def source_identity(repo=None):
    package = Path(__file__).resolve().parent if repo is None else Path(repo) / "src/mediflow_medvlm"
    files = {"src/mediflow_medvlm/" + name: file_hash(package / name) for name in SOURCE_FILES}
    return files, canonical_hash(files)


def _inputs(manifest, state, *, verify_results=False):
    validate_manifest(manifest)
    _require(manifest.get("campaign_id") == state.get("campaign_id")
        and manifest.get("seeds") == list(SEEDS), "RECORDED_CAMPAIGN_IDENTITY_MISMATCH")
    cells = manifest.get("cells", [])
    _require([c.get("cell_id") for c in cells] == [c[0] for c in CELLS]
        and all(all(c.get(k) == v for k, v in cell_spec(c["cell_id"]).items())
            and c.get("seed") == SEEDS[0] for c in cells), "EXACT_FIRST_SEED_PLAN_REQUIRED")
    jobs = {j["job_id"]: j for j in manifest["jobs"]}
    _require([j["job_id"] for j in manifest["jobs"] if j["phase"] == "native_main_train"] == CHILD_IDS
        and main_train.common.JOB_ID not in jobs, "RECORDED_NAMESPACE_EXACT_TEN_REQUIRED")
    aggregate = jobs[JOB_ID]
    _require(aggregate.get("phase") == "first_seed_final" and aggregate.get("cell_id") == "ALL_V6"
        and aggregate.get("run_id") == JOB_ID and aggregate.get("dependencies") == CHILD_IDS
        and aggregate.get("dependency_policy") == "terminal", "RECORDED_AGGREGATE_IDENTITY_MISMATCH")
    external = manifest.get("external_common_freeze", {})
    freeze_config, plan, reference = main_train._freeze(external["path"])
    _require({k: reference[k] for k in ("path", "sha256")} == external,
        "EXTERNAL_FREEZE_RECEIPT_MISMATCH")
    old_manifest, old_state = gate._active(freeze_config["state_path"])
    old_job = next(j for j in old_manifest["jobs"] if j["job_id"] == main_train.common.JOB_ID)
    old_record = old_state["jobs"][old_job["job_id"]]
    _require(canonical_hash(old_job) == reference["job_sha256"]
        and canonical_hash(old_record) == reference["terminal_state_sha256"], "ORIGINAL_FREEZE_JOB_STATE_MISMATCH")
    freeze = {"receipt": reference, "job": old_job, "terminal_state": old_record,
        "config": {"path": str(Path(freeze_config["output_dir"]) / "config.json"),
                   "sha256": file_hash(Path(freeze_config["output_dir"]) / "config.json")},
        "freeze_id": plan["freeze_id"]}
    root = _absolute(manifest["campaign_root"])
    children, rows = {}, []
    for cell, job_id in zip(CELLS, CHILD_IDS):
        job, record = jobs[job_id], state["jobs"][job_id]
        _require(job.get("job_id") == job.get("run_id") == job_id and job.get("phase") == "native_main_train"
            and all(job.get(k) == v for k, v in cell_spec(cell[0]).items())
            and job.get("dependencies") == [] and job.get("external_prerequisite") == external
            and job.get("dependency_policy") == "success" and job.get("seed") == SEEDS[0]
            and job.get("uses_gpu") is True and job.get("requires_gpu") is True
            and job.get("gpu_lock_mode") == "wrapper" and job.get("implementation_pending") is False,
            "RECORDED_CHILD_IDENTITY_MISMATCH")
        argv = job.get("argv", [])
        _require(len(argv) == 7 and argv[1:4] == ["-m", "mediflow_medvlm.campaign_main_train", "--config"]
            and argv[5:] == ["--receipt", job["success_receipt"]], "RECORDED_CHILD_COMMAND_MISMATCH")
        path = _absolute(argv[4])
        _require(not path.is_symlink(), "SYMLINK_CHILD_CONFIG_FORBIDDEN")
        config = _read(path)
        main_train.validate_config_identity(config, repo=job["cwd"])
        verify_source(job)
        _require(all(_equal(config.get(k), job.get(k)) for k in (*IDENTITY_KEYS, "source_files"))
            and config.get("success_receipt") == job["success_receipt"]
            and config["data"].get("common_freeze") == reference
            and config["data"].get("freeze_id") == plan["freeze_id"], "RECORDED_CHILD_FREEZE_OR_JOB_BINDING_MISMATCH")
        for name in ("output_dir", "success_receipt"):
            gate._private(_absolute(config[name]), root)
        _require(record.get("status") in TERMINAL and all(type(record.get(k)) is int and record[k] >= 0
            for k in ("attempts", "failure_count", "recovery_attempts")), "CHILD_NOT_TERMINAL_OR_INVALID_COUNTERS")
        child = {"job": copy.deepcopy(job), "config": {"path": str(path), "sha256": file_hash(path)},
                 "terminal_state": copy.deepcopy(record)}
        row = {**cell_spec(cell[0]), "job_id": job_id, "seed": SEEDS[0], "status": record["status"],
               **{k: record[k] for k in ("attempts", "failure_count", "recovery_attempts")}}
        if record["status"] == "SUCCEEDED":
            receipt = gate._success(job, record)
            child["receipt"] = {"path": job["success_receipt"], "sha256": file_hash(job["success_receipt"])}
            if verify_results:
                _require(_equal(main_train.verify_result(config, repo=job["cwd"]), receipt),
                    "RECORDED_CHILD_FINAL_RESULT_MISMATCH")
            row.update(final_checkpoint_verified=verify_results, full_pass_verified=verify_results,
                fresh_process_reload_verified=verify_results, receipt=child["receipt"], config=child["config"])
        else:
            row.update(final_checkpoint_verified=False, full_pass_verified=False,
                fresh_process_reload_verified=False, error=record.get("error"), error_signature=record.get("error_signature"))
        children[job_id] = child
        rows.append(row)
    return {"campaign_id": manifest["campaign_id"], "campaign_root": str(root),
        "declaration_sha256": canonical_hash({k: v for k, v in manifest.items() if k != "jobs"}),
        "external_freeze": freeze, "children": children}, rows


def build_config(state_path, output_dir, *, repo=None, success_receipt=None):
    manifest, state = gate._active(state_path)
    inputs, _ = _inputs(manifest, state)
    root = _absolute(manifest["campaign_root"])
    output = _absolute(output_dir)
    receipt = _absolute(success_receipt or root / "receipts" / (JOB_ID + ".json"))
    for path in (output, receipt): gate._private(path, root)
    files, source = source_identity(repo)
    result = {"schema_version": SCHEMA, "scope": SCOPE, "job_id": JOB_ID, "run_id": JOB_ID,
        "uses_gpu": False, "state_path": str(_absolute(state_path)), "output_dir": str(output),
        "success_receipt": str(receipt), "prerequisites": inputs, "source_files": files,
        "source_hash": source, "data_hash": canonical_hash(inputs),
        "model_hash": canonical_hash({k: v["job"]["model_hash"] for k, v in inputs["children"].items()})}
    result["config_hash"] = canonical_hash(result)
    return result


def _check(config, *, repo=None, verify_results=False):
    _require(config.get("schema_version") == SCHEMA and config.get("scope") == SCOPE
        and config.get("job_id") == config.get("run_id") == JOB_ID and config.get("uses_gpu") is False
        and config.get("config_hash") == canonical_hash({k: v for k, v in config.items() if k != "config_hash"})
        and _equal(source_identity(repo), (config["source_files"], config["source_hash"])), "RECORDED_GATE_CONFIG_MISMATCH")
    manifest, state = gate._active(config["state_path"])
    inputs, rows = _inputs(manifest, state, verify_results=verify_results)
    _require(_equal(inputs, config["prerequisites"]) and config["data_hash"] == canonical_hash(inputs)
        and config["model_hash"] == canonical_hash({k: v["job"]["model_hash"] for k, v in inputs["children"].items()}),
        "RECORDED_GATE_INPUT_CHANGED")
    for path in (config["output_dir"], config["success_receipt"]):
        gate._private(_absolute(path), _absolute(manifest["campaign_root"]))
    for key in IDENTITY_KEYS:
        supplied = os.environ.get("MEDVLM_SUPERVISOR_" + key.upper())
        _require(supplied is None or supplied == config[key], "SUPERVISOR_IDENTITY_MISMATCH")
    return rows


def activation_manifest(config_path, manifest, *, repo=None, python=None):
    config_path, config = _absolute(config_path), _read(config_path)
    _check(config, repo=repo)
    active, _ = gate._active(config["state_path"])
    _require(_equal(active, manifest), "ACTIVATION_REQUIRES_CURRENT_MANIFEST")
    revised = copy.deepcopy(manifest)
    job = next(j for j in revised["jobs"] if j["job_id"] == JOB_ID)
    _require(job.get("implementation_pending") is True and not job.get("argv"), "UNSTARTED_RECORDED_GATE_REQUIRED")
    job.update(implementation_pending=False, uses_gpu=False, gpu_lock_mode="wrapper",
        argv=[str(python or __import__("sys").executable), "-m", "mediflow_medvlm.campaign_recorded_first_seed",
              "--config", str(config_path), "--receipt", config["success_receipt"]],
        cwd=str(Path(repo).resolve() if repo is not None else Path(__file__).resolve().parents[2]),
        source_files=config["source_files"], **{k: config[k] for k in HASH_FIELDS}, success_receipt=config["success_receipt"])
    validate_manifest(revised)
    return revised


def _summary(rows):
    counts = {status: sum(r["status"] == status for r in rows) for status in sorted({r["status"] for r in rows})}
    return {"scope": SCOPE, "aggregate_audit_succeeded": True, "first_seed_stage_terminal": True,
        "all_cells_succeeded": counts.get("SUCCEEDED") == 10, "v6_main_training_completed": counts.get("SUCCEEDED") == 10,
        "cell_status_counts": counts, "cells": rows, "dev_test_read": False,
        "gpu_work_performed": False, "campaign_terminal": False}


def run_gate(config_path, receipt_path):
    config, before = _read(config_path), file_hash(config_path)
    _require(str(_absolute(receipt_path)) == config["success_receipt"], "RECORDED_RECEIPT_PATH_MISMATCH")
    rows = _check(config, verify_results=True)
    output = _absolute(config["output_dir"])
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with output.with_name(output.name + ".worker.lock").open("a") as lock:
        os.chmod(lock.name, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        summary = _summary(rows)
        if not output.exists():
            _require(not Path(receipt_path).exists(), "RECEIPT_WITHOUT_RECORDED_OUTPUT")
            staging = Path(tempfile.mkdtemp(prefix=output.name + ".candidate-", dir=output.parent))
            atomic_json(staging / "config.json", config)
            atomic_json(staging / "summary.json", {**summary, "finished_at": utcnow()})
            _require(file_hash(config_path) == before and _equal(_check(config), [{**r,
                "final_checkpoint_verified": False, "full_pass_verified": False,
                "fresh_process_reload_verified": False} if r["status"] == "SUCCEEDED" else r for r in rows]),
                "RECORDED_INPUT_CHANGED_DURING_AUDIT")
            marker = {"schema_version": "medvlm.v6.recorded-first-seed-receipt.v1", "status": "SUCCEEDED", "scope": SCOPE,
                **{k: config[k] for k in IDENTITY_KEYS}, "all_cells_succeeded": summary["all_cells_succeeded"],
                "first_seed_stage_terminal": True, "campaign_terminal": False,
                "artifacts": [{"path": str(output / n), "sha256": file_hash(staging / n)}
                              for n in ("config.json", "summary.json")]}
            atomic_json(staging / "completion.json", marker)
            os.rename(staging, output)
            fd = os.open(output.parent, os.O_DIRECTORY)
            try: os.fsync(fd)
            finally: os.close(fd)
        _require({p.name for p in output.iterdir()} == {"config.json", "summary.json", "completion.json"}
            and not any(p.is_symlink() for p in output.iterdir()) and _equal(_read(output / "config.json"), config),
            "EXISTING_RECORDED_OUTPUT_MISMATCH")
        stored = _read(output / "summary.json")
        _require(_equal({k: v for k, v in stored.items() if k != "finished_at"}, summary), "EXISTING_RECORDED_AUDIT_CHANGED")
        result = _read(output / "completion.json")
        expected_artifacts = [{"path": str(output / n), "sha256": file_hash(output / n)} for n in ("config.json", "summary.json")]
        _require(result.get("status") == "SUCCEEDED" and result.get("scope") == SCOPE
            and all(result.get(k) == config[k] for k in IDENTITY_KEYS)
            and result.get("all_cells_succeeded") == summary["all_cells_succeeded"]
            and result.get("first_seed_stage_terminal") is True and result.get("campaign_terminal") is False
            and result.get("artifacts") == expected_artifacts and file_hash(config_path) == before,
            "EXISTING_RECORDED_ARTIFACT_CHANGED")
        result = {**result, "artifacts": expected_artifacts + [{"path": str(output / "completion.json"),
                  "sha256": file_hash(output / "completion.json")}]}
        _require(not Path(receipt_path).exists() or _equal(_read(receipt_path), result), "EXISTING_RECORDED_RECEIPT_CHANGED")
        if not Path(receipt_path).exists(): atomic_json(receipt_path, result)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run_gate(args.config, args.receipt)
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "scope": SCOPE, "exception_type": type(exc).__name__,
            **({"reason_code": str(exc)} if isinstance(exc, (gate.GateError, main_train.MainTrainError)) else {})}))
        return 1
    print(json.dumps({"status": result["status"], "all_cells_succeeded": result["all_cells_succeeded"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
