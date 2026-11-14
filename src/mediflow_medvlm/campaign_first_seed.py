"""Plan ten independent first-seed runs and audit their terminal results on CPU.

``build_refinement`` returns an unregistered candidate and child config payloads.
Save the child configs, build/save the aggregate config, then use
``activation_manifest`` and one ``Supervisor.refine_first_seed`` registration.
The aggregate config never snapshots mutable child attempt records. Neither
planning nor the aggregate worker advances the queue or imports GPU libraries.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import fcntl
import json
import os
from pathlib import Path
import sys
import tempfile

from . import campaign_common_train_freeze as freeze
from . import campaign_preflight_gate as gate
from .campaign_supervisor import (HASH_FIELDS, TERMINAL, atomic_json, canonical_hash,
    file_hash, validate_manifest, verify_source, verify_gpu_source_rebind)
from .campaign_v6 import CELLS, SEEDS, cell_spec
from .runtime import utcnow

JOB_ID = "v6-first_seed_final"
CHILD_IDS = ["v6-first-seed-" + cell[0] for cell in CELLS]
SCHEMA = "medvlm.v6.first-seed-gate-config.v1"
SCOPE = "FIRST_SEED_TERMINAL_AUDIT_ONLY"
IDENTITY_KEYS = gate.IDENTITY_KEYS
OUTPUT_FILES = {"config.json", "summary.json"}
_require, _read, _absolute, _equal = gate._require, gate._read, gate._absolute, gate._equal
REBIND_KEYS = {"cwd", "source_files", "source_hash", "argv", "config_hash"}


def _main():
    from . import campaign_main_train
    return campaign_main_train


def source_identity(repo=None):
    package = Path(__file__).resolve().parent if repo is None else Path(repo) / "src/mediflow_medvlm"
    names = tuple(dict.fromkeys((*_main().SOURCE_FILES, "campaign_first_seed.py")))
    files = {"src/mediflow_medvlm/" + name: file_hash(package / name) for name in names}
    return files, canonical_hash(files)


def _declaration(manifest):
    declared = manifest.get("cells", [])
    _require(manifest.get("seeds") == list(SEEDS) and len(declared) == 10
        and [c.get("cell_id") for c in declared] == [c[0] for c in CELLS]
        and all(all(_equal(c.get(k), v) for k, v in cell_spec(c["cell_id"]).items())
            and type(c.get("seed")) is int and c["seed"] == SEEDS[0] for c in declared),
        "EXACT_FIRST_SEED_TEN_CELL_PLAN_REQUIRED")
    return {k: v for k, v in manifest.items() if k != "jobs"}


def _freeze_proof(manifest, state):
    _require(state.get("campaign_id") == manifest.get("campaign_id"), "CAMPAIGN_IDENTITY_MISMATCH")
    job = next(j for j in manifest["jobs"] if j["job_id"] == freeze.JOB_ID)
    record = state["jobs"][freeze.JOB_ID]
    _require(record.get("status") == "SUCCEEDED", "COMMON_FREEZE_SUCCESS_REQUIRED")
    verify_source(job)  # Its archived source, never the newly imported supervisor.
    receipt = gate._success(job, record)
    argv = job["argv"]
    _require(argv[1:3] == ["-m", "mediflow_medvlm.campaign_common_train_freeze"]
        and argv.count("--config") == 1, "COMMON_FREEZE_COMMAND_MISMATCH")
    path = _absolute(argv[argv.index("--config") + 1])
    config = _read(path)
    _require(config.get("schema_version") == freeze.SCHEMA and config.get("scope") == freeze.SCOPE
        and config.get("config_hash") == canonical_hash({k: v for k, v in config.items() if k != "config_hash"})
        and all(_equal(config.get(k), job.get(k)) for k in (*IDENTITY_KEYS, "source_files")),
        "COMMON_FREEZE_CONFIG_BINDING_MISMATCH")
    _require(_equal(freeze._completed(config), receipt), "COMMON_FREEZE_COMPLETION_MISMATCH")
    return {"job": job, "state": record, "config": {"path": str(path), "sha256": file_hash(path)},
            "receipt": {"path": job["success_receipt"], "sha256": file_hash(job["success_receipt"])}}


def _child_config(job, *, validate_inputs=True):
    cell = job.get("cell_id")
    _require(cell in [c[0] for c in CELLS] and job.get("job_id") == job.get("run_id") == "v6-first-seed-" + cell
        and job.get("phase") == "native_main_train" and job.get("dependencies") == [freeze.JOB_ID]
        and job.get("dependency_policy") == "success" and job.get("implementation_pending") is False
        and job.get("uses_gpu") is True and job.get("requires_gpu") is True
        and job.get("gpu_lock_mode") == "wrapper" and type(job.get("seed")) is int and job["seed"] == SEEDS[0]
        and all(_equal(job.get(k), v) for k, v in cell_spec(cell).items()), "MAIN_CHILD_IDENTITY_MISMATCH")
    allowed = {"job_id", "run_id", "phase", "dependencies", "dependency_policy", "implementation_pending",
        "uses_gpu", "requires_gpu", "gpu_lock_mode", "seed", "argv", "cwd", "success_receipt", "source_files",
        *HASH_FIELDS, *cell_spec(cell)}
    _require(not set(job) - allowed, "UNEXPECTED_MAIN_CHILD_FIELDS")
    argv = job["argv"]
    _require(len(argv) == 7 and argv[1:4] == ["-m", "mediflow_medvlm.campaign_main_train", "--config"]
        and argv[5:] == ["--receipt", job["success_receipt"]], "MAIN_CHILD_COMMAND_MISMATCH")
    path = _absolute(argv[4])
    _require(not path.is_symlink(), "SYMLINK_CONFIG_FORBIDDEN")
    config = _read(path)
    worker = _main()
    (worker.validate_config if validate_inputs else worker.validate_config_identity)(config, repo=job["cwd"])
    _require(all(_equal(config.get(k), job.get(k)) for k in (*IDENTITY_KEYS, "source_files"))
        and all(_equal(config.get(k), v) for k, v in cell_spec(cell).items())
        and config.get("success_receipt") == job["success_receipt"]
        and type(config.get("recipe", {}).get("seed")) is int and config["recipe"]["seed"] == SEEDS[0],
        "MAIN_CONFIG_JOB_BINDING_MISMATCH")
    verify_source(job)
    return config, {"path": str(path), "sha256": file_hash(path)}


def _inputs(manifest, state, *, validate_inputs=True):
    validate_manifest(manifest)
    declaration = _declaration(manifest)
    common = _freeze_proof(manifest, state)
    jobs = {j["job_id"]: j for j in manifest["jobs"]}
    _require([j["job_id"] for j in manifest["jobs"] if j["phase"] == "native_main_train"] == CHILD_IDS,
        "EXACT_TEN_ORDERED_MAIN_CHILDREN_REQUIRED")
    aggregate = jobs[JOB_ID]
    _require(aggregate.get("phase") == "first_seed_final" and aggregate.get("cell_id") == "ALL_V6"
        and aggregate.get("run_id") == JOB_ID and aggregate.get("dependencies") == CHILD_IDS
        and aggregate.get("dependency_policy") == "terminal", "FIRST_SEED_GATE_IDENTITY_MISMATCH")
    children = {}
    for job_id in CHILD_IDS:
        config, reference = _child_config(jobs[job_id], validate_inputs=validate_inputs)
        common_ref = config.get("data", {}).get("common_freeze", {})
        _require(common_ref.get("path") == common["receipt"]["path"]
            and common_ref.get("sha256") == common["receipt"]["sha256"]
            and common_ref.get("job_sha256") == canonical_hash(common["job"])
            and common_ref.get("terminal_state_sha256") == canonical_hash(common["state"]),
            "MAIN_COMMON_FREEZE_BINDING_MISMATCH")
        children[job_id] = {"job": jobs[job_id], "config": reference,
            "scientific_config_hash": canonical_hash({k: v for k, v in config.items()
                if k not in {"source_files", "source_hash", "config_hash"}})}
    return copy.deepcopy({"declaration": declaration, "common_freeze": common, "children": children})


def _compare_inputs(expected, actual):
    _require(_equal(expected["declaration"], actual["declaration"])
        and _equal(expected["common_freeze"], actual["common_freeze"]), "FROZEN_DECLARATION_CHANGED")
    for job_id in CHILD_IDS:
        before, after = expected["children"][job_id], actual["children"][job_id]
        _require(file_hash(_absolute(before["config"]["path"])) == before["config"]["sha256"],
            "REGISTERED_MAIN_CONFIG_CHANGED")
        _require(before["scientific_config_hash"] == after["scientific_config_hash"], "MAIN_SCIENTIFIC_CONFIG_CHANGED")
        if not _equal(before, after):
            verify_gpu_source_rebind(before["job"], after["job"], validate_inputs=False)


def build_refinement(manifest_path, state_path, config_dir, output_dir, *, repo=None, python=None):
    """Return {manifest, configs}; callers save configs before building the gate."""
    manifest_path, state_path, config_dir, output_dir = map(_absolute,
        (manifest_path, state_path, config_dir, output_dir))
    manifest, state = gate._active(state_path)
    _require(str(manifest_path) == state["manifest_path"], "PLANNING_MANIFEST_NOT_ACTIVE")
    _declaration(manifest)
    common = _freeze_proof(manifest, state)
    old = next(j for j in manifest["jobs"] if j["job_id"] == JOB_ID)
    _require(old.get("implementation_pending") is True and not old.get("argv")
        and state["jobs"][JOB_ID]["status"] == "IMPLEMENTATION_PENDING", "ONLY_UNSTARTED_GATE_MAY_BE_PLANNED")
    root = _absolute(manifest["campaign_root"])
    for path in (config_dir, output_dir):
        gate._private(path, root)
    revised, configs, children = copy.deepcopy(manifest), {}, []
    cwd = str(Path(repo).resolve() if repo is not None else Path(__file__).resolve().parents[2])
    for cell in CELLS:
        cell_id, job_id = cell[0], "v6-first-seed-" + cell[0]
        receipt = str(root / "receipts" / (job_id + ".json"))
        config = _main().build_config(common["receipt"]["path"], cell_id, output_dir / cell_id,
            repo=repo, success_receipt=receipt, job_id=job_id)
        path = config_dir / (cell_id + ".json")
        configs[str(path)] = config
        children.append({"job_id": job_id, "run_id": job_id, "phase": "native_main_train", **cell_spec(cell_id),
            "seed": SEEDS[0], "dependencies": [freeze.JOB_ID], "dependency_policy": "success",
            "implementation_pending": False, "uses_gpu": True, "requires_gpu": True, "gpu_lock_mode": "wrapper",
            "argv": [str(python or sys.executable), "-m", "mediflow_medvlm.campaign_main_train", "--config", str(path), "--receipt", receipt],
            "cwd": cwd, "success_receipt": receipt, "source_files": config["source_files"],
            **{k: config[k] for k in HASH_FIELDS}})
    position = next(i for i, j in enumerate(revised["jobs"]) if j["job_id"] == JOB_ID)
    revised["jobs"][position]["dependencies"] = CHILD_IDS.copy()
    revised["jobs"][position:position] = children
    validate_manifest(revised)
    _require(_equal(_freeze_proof(*gate._active(state_path)), common), "FREEZE_CHANGED_DURING_PLANNING")
    return {"manifest": revised, "configs": configs}


def build_config(manifest, state_path, output_dir, *, repo=None, success_receipt=None):
    """Bind immutable child configs, never their not-yet-terminal attempt states."""
    state_path, output_dir = _absolute(state_path), _absolute(output_dir)
    state = _read(state_path)
    prerequisites = _inputs(manifest, state)
    root = _absolute(manifest["campaign_root"])
    receipt = _absolute(success_receipt or root / "receipts" / (JOB_ID + ".json"))
    for path in (output_dir, receipt):
        gate._private(path, root)
    files, source = source_identity(repo)
    config = {"schema_version": SCHEMA, "scope": SCOPE, "job_id": JOB_ID, "run_id": JOB_ID,
        "uses_gpu": False, "state_path": str(state_path), "output_dir": str(output_dir),
        "success_receipt": str(receipt), "prerequisites": prerequisites,
        "source_files": files, "source_hash": source, "data_hash": canonical_hash(prerequisites),
        "model_hash": canonical_hash({k: prerequisites["children"][k]["job"]["model_hash"] for k in CHILD_IDS})}
    config["config_hash"] = canonical_hash(config)
    return config


def activation_manifest(config_path, manifest, *, repo=None, python=None):
    path, config = _absolute(config_path), _read(config_path)
    _check_config(config, manifest, _read(config["state_path"]), repo=repo)
    revised = copy.deepcopy(manifest)
    job = next(j for j in revised["jobs"] if j["job_id"] == JOB_ID)
    job.update(implementation_pending=False, uses_gpu=False, gpu_lock_mode="wrapper",
        argv=[str(python or sys.executable), "-m", "mediflow_medvlm.campaign_first_seed", "--config", str(path), "--receipt", config["success_receipt"]],
        cwd=str(Path(repo).resolve() if repo is not None else Path(__file__).resolve().parents[2]),
        source_files=config["source_files"], **{k: config[k] for k in HASH_FIELDS}, success_receipt=config["success_receipt"])
    validate_manifest(revised)
    return revised


def _check_config(config, manifest, state, *, repo=None, validate_inputs=True):
    _require(config.get("schema_version") == SCHEMA and config.get("scope") == SCOPE
        and config.get("job_id") == config.get("run_id") == JOB_ID and config.get("uses_gpu") is False
        and config.get("config_hash") == canonical_hash({k: v for k, v in config.items() if k != "config_hash"})
        and _equal(source_identity(repo), (config["source_files"], config["source_hash"])), "FIRST_SEED_GATE_CONFIG_MISMATCH")
    _compare_inputs(config["prerequisites"], _inputs(manifest, state, validate_inputs=validate_inputs))
    _require(config["data_hash"] == canonical_hash(config["prerequisites"])
        and config["model_hash"] == canonical_hash({k: config["prerequisites"]["children"][k]["job"]["model_hash"] for k in CHILD_IDS}),
        "FIRST_SEED_GATE_INPUT_HASH_MISMATCH")
    for key in ("output_dir", "success_receipt"):
        gate._private(_absolute(config[key]), _absolute(manifest["campaign_root"]))


def validate_refinement(original, revised, state, *, validate_inputs=True):
    """Strict structural/config validation used inside the supervisor lock."""
    validate_manifest(original)
    _declaration(original)
    _require(len(original["jobs"]) == 43 and _equal(_declaration(original), _declaration(revised)),
        "FIRST_SEED_REFINEMENT_CANNOT_CHANGE_METADATA")
    before, after = ({j["job_id"]: j for j in m["jobs"]} for m in (original, revised))
    _require(not set(CHILD_IDS).intersection(before), "FIRST_SEED_CHILDREN_ALREADY_PRESENT")
    order = [j["job_id"] for j in original["jobs"]]
    position = order.index(JOB_ID)
    order[position:position] = CHILD_IDS
    _require([j["job_id"] for j in revised["jobs"]] == order, "EXACT_TEN_CHILD_INSERTION_REQUIRED")
    old = before[JOB_ID]
    _require(old.get("phase") == "first_seed_final" and old.get("cell_id") == "ALL_V6"
        and old.get("run_id") == JOB_ID and old.get("dependencies") == [freeze.JOB_ID]
        and old.get("dependency_policy") == "terminal" and old.get("implementation_pending") is True
        and not old.get("argv"), "ORIGINAL_UNIMPLEMENTED_FIRST_SEED_GATE_REQUIRED")
    allowed = {"dependencies", "implementation_pending", "uses_gpu", "gpu_lock_mode", "argv", "cwd",
        "source_files", *HASH_FIELDS, "success_receipt"}
    for job_id, job in before.items():
        _require(_equal({k: v for k, v in job.items() if job_id != JOB_ID or k not in allowed},
            {k: v for k, v in after[job_id].items() if job_id != JOB_ID or k not in allowed}),
            "FIRST_SEED_REFINEMENT_CHANGED_OLD_JOB")
    job = after[JOB_ID]
    _require(job.get("implementation_pending") is False and job.get("uses_gpu") is False
        and job.get("gpu_lock_mode") == "wrapper" and len(job.get("argv", [])) == 7
        and job["argv"][1:4] == ["-m", "mediflow_medvlm.campaign_first_seed", "--config"]
        and job["argv"][5:] == ["--receipt", job["success_receipt"]], "EXECUTABLE_CPU_FIRST_SEED_GATE_REQUIRED")
    path = _absolute(job["argv"][4])
    config = _read(path)
    _check_config(config, revised, state, repo=job["cwd"], validate_inputs=validate_inputs)
    _require(all(_equal(config.get(k), job.get(k)) for k in (*IDENTITY_KEYS, "source_files"))
        and job["success_receipt"] == config["success_receipt"], "FIRST_SEED_GATE_JOB_BINDING_MISMATCH")
    return {"original_gate_hash": canonical_hash(old), "revised_gate_hash": canonical_hash(job),
        "child_job_hashes": {k: canonical_hash(after[k]) for k in CHILD_IDS},
        "config_files": {**{k: config["prerequisites"]["children"][k]["config"] for k in CHILD_IDS},
            JOB_ID: {"path": str(path), "sha256": file_hash(path)}},
        "children": config["prerequisites"]["children"], "common_freeze": config["prerequisites"]["common_freeze"]}


def _audit(config):
    manifest, state = gate._active(config["state_path"])
    _check_config(config, manifest, state, validate_inputs=False)
    for key in IDENTITY_KEYS:
        supplied = os.environ.get("MEDVLM_SUPERVISOR_" + key.upper())
        _require(supplied is None or supplied == config[key], "SUPERVISOR_IDENTITY_MISMATCH")
    snapshot, rows = {}, []
    jobs = {j["job_id"]: j for j in manifest["jobs"]}
    for cell, job_id in zip(CELLS, CHILD_IDS):
        record = state["jobs"][job_id]
        _require(record.get("status") in TERMINAL, "MAIN_CHILD_NOT_TERMINAL")
        _require(all(type(record.get(k)) is int and record[k] >= 0 for k in
            ("attempts", "failure_count", "recovery_attempts")), "INVALID_MAIN_CUMULATIVE_COUNTERS")
        row = {**cell_spec(cell[0]), "job_id": job_id, "seed": SEEDS[0], "status": record["status"],
            "attempts": record["attempts"], "failure_count": record["failure_count"],
            "recovery_attempts": record["recovery_attempts"]}
        if record["status"] == "SUCCEEDED":
            receipt = gate._success(jobs[job_id], record)
            child_config, reference = _child_config(jobs[job_id], validate_inputs=False)
            _require(_equal(_main().verify_result(child_config, repo=jobs[job_id]["cwd"]), receipt),
                "MAIN_FINAL_RESULT_MISMATCH")
            row.update(final_checkpoint_verified=True, full_pass_verified=True, fresh_process_reload_verified=True,
                receipt={"path": jobs[job_id]["success_receipt"], "sha256": file_hash(jobs[job_id]["success_receipt"])},
                config=reference)
        else:
            row.update(final_checkpoint_verified=False, full_pass_verified=False, fresh_process_reload_verified=False,
                error=record.get("error"), error_signature=record.get("error_signature"))
        rows.append(row)
        snapshot[job_id] = copy.deepcopy(record)
    return {"cells": rows, "terminal_records": snapshot}


def _summary(audit):
    counts = dict(Counter(row["status"] for row in audit["cells"]))
    return {"scope": SCOPE, "aggregate_audit_succeeded": True, "first_seed_stage_terminal": True,
        "all_cells_succeeded": counts.get("SUCCEEDED") == 10,
        "v6_main_training_completed": counts.get("SUCCEEDED") == 10,
        "cell_status_counts": counts, **audit, "dev_test_read": False, "gpu_work_performed": False,
        "campaign_terminal": False}


def _completed(config):
    output = _absolute(config["output_dir"])
    result = _read(output / "completion.json")
    _require(result.get("status") == "SUCCEEDED" and result.get("scope") == SCOPE
        and all(result.get(k) == config[k] for k in IDENTITY_KEYS)
        and {p.name for p in output.iterdir()} == OUTPUT_FILES | {"completion.json"}
        and not any(p.is_symlink() for p in output.iterdir()) and _equal(_read(output / "config.json"), config),
        "EXISTING_FIRST_SEED_OUTPUT_MISMATCH")
    _require(result.get("artifacts") == [{"path": str(output / name), "sha256": file_hash(output / name)}
        for name in sorted(OUTPUT_FILES)], "EXISTING_FIRST_SEED_ARTIFACT_CHANGED")
    summary = _read(output / "summary.json")
    audit = _audit(config)
    expected = _summary(audit)
    _require(_equal({k: v for k, v in summary.items() if k != "finished_at"}, expected)
        and result.get("schema_version") == "medvlm.v6.first-seed-gate-receipt.v1"
        and result.get("all_cells_succeeded") is expected["all_cells_succeeded"]
        and result.get("first_seed_stage_terminal") is True and result.get("campaign_terminal") is False,
        "EXISTING_FIRST_SEED_AUDIT_CHANGED")
    return {**result, "artifacts": result["artifacts"] + [{"path": str(output / "completion.json"),
        "sha256": file_hash(output / "completion.json")}]}


def run_gate(config_path, receipt_path):
    config_path, receipt_path = _absolute(config_path), _absolute(receipt_path)
    config, initial = _read(config_path), file_hash(config_path)
    _require(str(receipt_path) == config["success_receipt"], "FIRST_SEED_RECEIPT_PATH_MISMATCH")
    audit = _audit(config)
    output = _absolute(config["output_dir"])
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with output.with_name(output.name + ".worker.lock").open("a") as lock:
        os.chmod(lock.name, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if output.exists():
            result = _completed(config)
            _require(file_hash(config_path) == initial and (not receipt_path.exists()
                or _equal(_read(receipt_path), result)), "FIRST_SEED_REUSE_CHANGED")
            if not receipt_path.exists():
                atomic_json(receipt_path, result)
            return result
        _require(not receipt_path.exists(), "FIRST_SEED_RECEIPT_WITHOUT_OUTPUT")
        staging = Path(tempfile.mkdtemp(prefix=output.name + ".candidate-", dir=output.parent))
        summary = {**_summary(audit), "finished_at": utcnow()}
        all_success = summary["all_cells_succeeded"]
        atomic_json(staging / "config.json", config)
        atomic_json(staging / "summary.json", summary)
        _require(file_hash(config_path) == initial and _equal(_audit(config), audit), "FIRST_SEED_INPUT_CHANGED_DURING_AUDIT")
        marker = {"schema_version": "medvlm.v6.first-seed-gate-receipt.v1", "status": "SUCCEEDED", "scope": SCOPE,
            **{k: config[k] for k in IDENTITY_KEYS}, "all_cells_succeeded": all_success,
            "first_seed_stage_terminal": True, "campaign_terminal": False,
            "artifacts": [{"path": str(output / name), "sha256": file_hash(staging / name)} for name in sorted(OUTPUT_FILES)]}
        atomic_json(staging / "completion.json", marker)
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
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run_gate(args.config, args.receipt)
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "scope": SCOPE, "exception_type": type(exc).__name__,
            **({"reason_code": str(exc)} if isinstance(exc, gate.GateError) else {})}), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
