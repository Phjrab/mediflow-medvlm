"""One audited cap decision, twenty independent once-bumped roles, CPU freeze.

All executed initial modules stay immutable. Planning and CPU gates never load
CUDA or mutate the Supervisor. GPU work is one configured model/role; only the
Supervisor owns the queue and the inherited physical worker/GPU leases.
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
import traceback

from . import campaign_validation_caps as initial
from .campaign_supervisor import (HASH_FIELDS, TERMINAL, atomic_json, canonical_hash,
    file_hash, verify_source, validate_manifest)
from .campaign_v6 import CELLS, cell_spec
from .runtime import BudgetLedger, ResourcePolicy, utcnow

SCHEMA = "medvlm.v6.validation-cap-refinement-config.v1"
SCOPES = {"decision": "VALIDATION_CAP_DECISION_ONLY",
    "bumped": "VALIDATION_ONLY_NATIVE_ONCE_BUMPED_CAP_RERUN",
    "freeze": "VALIDATION_CAP_PROTOCOL_FREEZE_ONLY"}
PHASES = {"decision": "validation_cap_decision", "bumped": "validation_caps_bumped", "freeze": "validation_caps"}
DECISION_ID, FREEZE_ID = "v6-caps-decision", "v6-validation_caps"
BUMPED_IDS = [f"v6-caps-bumped-{c[0]}-{role}" for c in CELLS for role in initial.ROLES]
SOURCE_FILES = (*initial.SOURCE_FILES, "campaign_validation_cap_refinement.py")
IDENTITY_KEYS = initial.IDENTITY_KEYS
_require, _read, _absolute, _equal = initial._require, initial._read, initial._absolute, initial._equal
_ref, _reference = initial._ref, initial._reference


def source_identity(repo=None):
    package = Path(__file__).resolve().parent if repo is None else Path(repo) / "src/mediflow_medvlm"
    files = {"src/mediflow_medvlm/" + name: file_hash(package / name) for name in SOURCE_FILES}
    return files, canonical_hash(files)


def _protocol(reference, old_repo):
    protocol = _ref(reference)
    _require(protocol.get("schema_version") == initial.PROTOCOL_SCHEMA and protocol.get("scope") == initial.SCOPE
        and protocol.get("protocol_hash") == canonical_hash({k:v for k,v in protocol.items() if k != "protocol_hash"})
        and (protocol["source_files"], protocol["source_hash"]) == initial.source_identity(old_repo)
        and protocol.get("initial_caps") == initial.INITIAL_CAPS and protocol.get("maximum_caps") == initial.MAXIMUM_CAPS
        and protocol.get("eval_context") == 4096 and protocol.get("train_context") == 2048
        and protocol.get("test_files_read") is False, "ORIGINAL_INITIAL_PROTOCOL_CHANGED")
    return protocol


def _active(state_path):
    manifest, state = initial.gate._active(state_path)
    validate_manifest(manifest)
    _require(manifest.get("campaign_id") == state.get("campaign_id"), "CAP_CAMPAIGN_IDENTITY_MISMATCH")
    return manifest, state


def _child_config(job, module):
    argv = job.get("argv", [])
    _require(len(argv) == 7 and argv[1:4] == ["-m", "mediflow_medvlm." + module, "--config"]
        and argv[5:] == ["--receipt", job["success_receipt"]], "CAP_CHILD_COMMAND_CHANGED")
    reference = _reference(argv[4])
    config = _ref(reference)
    _require(config.get("config_hash") == canonical_hash({k:v for k,v in config.items() if k != "config_hash"})
        and all(_equal(config.get(k), job.get(k)) for k in (*IDENTITY_KEYS, "source_files"))
        and config.get("success_receipt") == job["success_receipt"], "CAP_CHILD_CONFIG_JOB_BINDING_CHANGED")
    verify_source(job)
    return config, reference


def _terminal(record):
    _require(record.get("status") in TERMINAL and all(type(record.get(k)) is int and record[k] >= 0
        for k in ("attempts", "failure_count", "recovery_attempts")), "CAP_CHILD_NOT_TERMINAL_OR_INVALID_COUNTERS")


def _audit_initial(protocol_ref, state_path, captured=None, *, deep=False):
    """A local proof snapshot binds only already-terminal initial jobs."""
    manifest, state = _active(state_path)
    jobs = {j["job_id"]:j for j in manifest["jobs"]}
    _require([j["job_id"] for j in manifest["jobs"] if j.get("calibration_round") == "initial"] == initial.INITIAL_IDS,
             "EXACT_INITIAL_TWENTY_JOBS_REQUIRED")
    old_repo = jobs[initial.INITIAL_IDS[0]]["cwd"]
    protocol = _protocol(protocol_ref, old_repo)
    _require(manifest["campaign_id"] == protocol["campaign_id"]
        and manifest["campaign_root"] == protocol["campaign_root"], "INITIAL_PROTOCOL_CAMPAIGN_CHANGED")
    proofs = {}
    for cell in CELLS:
        for role in initial.ROLES:
            key = f"v6-caps-initial-{cell[0]}-{role}"
            job, record = jobs[key], state["jobs"][key]
            _terminal(record)
            config, reference = _child_config(job, "campaign_validation_caps")
            _require(config.get("schema_version") == initial.SCHEMA and config.get("calibration_round") == "initial"
                and config.get("role") == role and config.get("data", {}).get("protocol") == protocol_ref
                and config.get("job_id") == key and config.get("seed") == 20261023
                and all(config.get(k) == v and job.get(k) == v for k,v in cell_spec(cell[0]).items())
                and job.get("phase") == "validation_caps_calibration" and job.get("uses_gpu") is True,
                "INITIAL_ROLE_SCIENCE_CHANGED")
            prior = captured[key] if captured is not None else None
            if prior is not None:
                _require(_equal(prior["job"], job) and _equal(prior["terminal_state"], record)
                    and prior["config"] == reference, "IMMUTABLE_INITIAL_TERMINAL_PROOF_CHANGED")
            proof = {"job": copy.deepcopy(job), "terminal_state": copy.deepcopy(record), "config": reference,
                     "receipt": None, "task_counts": None, "technical_complete": False}
            if record["status"] == "SUCCEEDED":
                receipt_ref = _reference(job["success_receipt"])
                receipt = initial.gate._success(job, record)  # Rehash only initial eval artifacts on shallow paths.
                if deep:
                    _require(_equal(initial.verify_result(config, repo=job["cwd"]), receipt), "INITIAL_DEEP_RESULT_MISMATCH")
                    metrics = _read(Path(config["output_dir"]) / "metrics.json")
                    proof.update(receipt=receipt_ref, task_counts=metrics["task_counts"],
                                 technical_complete=receipt["technical_complete"])
                else:
                    _require(prior is not None and prior["receipt"] == receipt_ref, "DEEP_INITIAL_PROOF_REQUIRED")
                    proof.update(receipt=receipt_ref, task_counts=prior["task_counts"], technical_complete=prior["technical_complete"])
            elif deep:
                initial.validate_config(config, repo=job["cwd"])
            if prior is not None:
                _require(_equal(proof, prior), "INITIAL_AUDIT_BYTES_OR_COUNTS_CHANGED")
            proofs[key] = proof
            progress = os.environ.get("MEDVLM_SUPERVISOR_PROGRESS_PATH")
            if deep and progress:
                atomic_json(progress, {"status":"RUNNING", "stage":"validation_cap_initial_cpu_audit",
                    "audited_conditions":len(proofs), "total_conditions":20,
                    "gpu_work_performed":False, "updated_at":utcnow()})
    return protocol, proofs, old_repo


def _decision(proofs, protocol):
    conditions = {key: {"status":p["terminal_state"]["status"], "task_counts":p["task_counts"]} for key,p in proofs.items()}
    proposed = initial.cap_decision_from_counts(protocol, conditions)
    return {"initial_protocol_hash": protocol["protocol_hash"], "caps": proposed["caps"],
        "bumped_task_types": proposed["bumped_task_types"], "bump_rounds": proposed["bump_rounds"],
        "universal_across_models_and_roles": True, "failed_initial_conditions": proposed["failed_conditions"],
        "technical_incomplete_initial_conditions": [key for key,p in proofs.items() if not p["technical_complete"]],
        "initial_task_counts": {key:p["task_counts"] for key,p in proofs.items()},
        "initial_receipts": {key:p["receipt"] for key,p in proofs.items()},
        "rule": "length_terminated*100 > scheduled; once only; EOS at boundary is stop",
        "test_files_read": False, "full_dev_completed": False, "family_selection_completed": False,
        "clinical_quality_certified": False, "campaign_terminal": False}


def _seal(kind, data, output_dir, state_path, *, repo=None, success_receipt=None, cell_id=None, role=None):
    protocol = _ref(data["protocol"])
    root = _absolute(protocol["campaign_root"])
    job_id = DECISION_ID if kind == "decision" else FREEZE_ID if kind == "freeze" else f"v6-caps-bumped-{cell_id}-{role}"
    output, receipt = _absolute(output_dir), _absolute(success_receipt or root / "receipts" / (job_id + ".json"))
    for path in (output, receipt): initial.gate._private(path, root)
    files, source = source_identity(repo)
    value = {"schema_version": SCHEMA, "scope": SCOPES[kind], "kind": kind, "phase": PHASES[kind],
        "job_id": job_id, "run_id": job_id, "cell_id": cell_id if kind == "bumped" else "ALL_V6",
        "seed": 20261023, "uses_gpu": kind == "bumped", "state_path": str(_absolute(state_path)),
        "output_dir": str(output), "success_receipt": str(receipt), "data": data,
        "resource_policy": ResourcePolicy().as_dict(), "source_files": files, "source_hash": source,
        "data_hash": canonical_hash(data), "model_hash": canonical_hash(protocol["models"])}
    if kind == "bumped":
        decision_config = _ref(data["decision_config"])
        initial_config = _ref(decision_config["data"]["initial_proofs"][f"v6-caps-initial-{cell_id}-{role}"]["config"])
        value.update(**cell_spec(cell_id), role=role, calibration_round="bumped", device="cuda",
            eval_context=4096, fixed_template_date="2026-11-14", model=initial_config["model"],
            decode=initial_config["decode"], local_files_only=True, trust_remote_code=False,
            gpu_ledger_path=initial_config["gpu_ledger_path"], supervisor_lock_paths=initial_config["supervisor_lock_paths"],
            model_hash=initial_config["model_hash"])
    value["config_hash"] = canonical_hash(value)
    return value


def build_decision_config(protocol_path, state_path, output_dir, *, repo=None, success_receipt=None):
    protocol_ref = _reference(protocol_path)
    protocol, proofs, old_repo = _audit_initial(protocol_ref, state_path, deep=True)
    data = {"protocol": protocol_ref, "protocol_hash": protocol["protocol_hash"], "protocol_source_repo": old_repo,
            "initial_proofs": proofs, "expected_decision": _decision(proofs, protocol)}
    return _seal("decision", data, output_dir, state_path, repo=repo, success_receipt=success_receipt)


def _decision_config(reference, *, repo=None):
    config = _ref(reference)
    validate_config(config, repo=repo)
    _require(config["kind"] == "decision", "DECISION_CONFIG_REQUIRED")
    return config


def build_bumped_config(decision_config_path, cell_id, role, output_dir, *, repo=None, success_receipt=None):
    decision_ref = _reference(decision_config_path)
    decision_config = _decision_config(decision_ref, repo=repo)
    _require(cell_id in [c[0] for c in CELLS] and role in initial.ROLES, "BUMPED_CELL_ROLE_REQUIRED")
    protocol = _ref(decision_config["data"]["protocol"])
    selected = initial._selected(protocol, cell_spec(cell_id)["task"], set(decision_config["data"]["expected_decision"]["bumped_task_types"]))
    initial_config = _ref(decision_config["data"]["initial_proofs"][f"v6-caps-initial-{cell_id}-{role}"]["config"])
    data = {"protocol": decision_config["data"]["protocol"], "protocol_hash": protocol["protocol_hash"],
        "decision_config": decision_ref, "decision_receipt": decision_config["success_receipt"],
        "expected_decision": decision_config["data"]["expected_decision"], "partition": "validation",
        "image_root": initial_config["data"]["image_root"],
        "ordered_sample_ids": [r["sample_id"] for r,_ in selected],
        "ordered_sample_ids_sha256": canonical_hash([r["sample_id"] for r,_ in selected])}
    return _seal("bumped", data, output_dir, decision_config["state_path"], repo=repo,
                 success_receipt=success_receipt, cell_id=cell_id, role=role)


build_rerun_config = build_bumped_config


def build_freeze_config(decision_config_path, bumped_config_paths, state_path, output_dir, *, repo=None, success_receipt=None):
    decision_ref = _reference(decision_config_path)
    parent = _decision_config(decision_ref, repo=repo)
    references = [_reference(path) for path in bumped_config_paths]
    configs = [_ref(reference) for reference in references]
    _require([c.get("job_id") for c in configs] == BUMPED_IDS and str(_absolute(state_path)) == parent["state_path"],
             "EXACT_ORDERED_TWENTY_BUMPED_CONFIGS_REQUIRED")
    for config in configs:
        validate_config(config, repo=repo)
        _require(config["data"]["decision_config"] == decision_ref, "BUMPED_CONFIG_DECISION_CHANGED")
    data = {"protocol": parent["data"]["protocol"], "protocol_hash": parent["data"]["protocol_hash"],
        "decision_config": decision_ref, "decision_receipt": parent["success_receipt"], "bumped_configs": references}
    return _seal("freeze", data, output_dir, state_path, repo=repo, success_receipt=success_receipt)


def validate_config(config, *, repo=None):
    kind = config.get("kind")
    _require(kind in SCOPES and config.get("schema_version") == SCHEMA and config.get("scope") == SCOPES[kind]
        and config.get("config_hash") == canonical_hash({k:v for k,v in config.items() if k != "config_hash"})
        and (config.get("source_files"),config.get("source_hash")) == source_identity(repo), "CAP_REFINEMENT_CONFIG_HASH_OR_SOURCE_CHANGED")
    if kind == "decision":
        data = config["data"]
        protocol = _protocol(data["protocol"], data["protocol_source_repo"])
        _require(list(data["initial_proofs"]) == initial.INITIAL_IDS
            and _equal(data["expected_decision"], _decision(data["initial_proofs"], protocol)), "CAP_DECISION_RULE_OR_MATRIX_CHANGED")
        for key,p in data["initial_proofs"].items():
            _terminal(p["terminal_state"])
            _require(p["job"]["job_id"] == key and _reference(p["config"]["path"]) == p["config"], "CAP_INITIAL_CONFIG_REFERENCE_CHANGED")
        expected = _seal(kind, data, config["output_dir"], config["state_path"], repo=repo, success_receipt=config["success_receipt"])
    elif kind == "bumped":
        expected = build_bumped_config(config["data"]["decision_config"]["path"], config["cell_id"], config["role"],
            config["output_dir"], repo=repo, success_receipt=config["success_receipt"])
    else:
        expected = build_freeze_config(config["data"]["decision_config"]["path"],
            [r["path"] for r in config["data"]["bumped_configs"]], config["state_path"], config["output_dir"],
            repo=repo, success_receipt=config["success_receipt"])
    _require(_equal(config,expected), "CAP_REFINEMENT_SCIENTIFIC_CONFIG_CHANGED")
    return config


def job_from_config(config_path, *, repo=None, python=None):
    path, config = _absolute(config_path), _read(config_path)
    validate_config(config, repo=repo)
    kind = config["kind"]
    deps = initial.INITIAL_IDS if kind == "decision" else [DECISION_ID] if kind == "bumped" else [DECISION_ID,*BUMPED_IDS]
    job = {"job_id":config["job_id"], "run_id":config["run_id"], "cell_id":config["cell_id"],
        "kind":kind, "phase":config["phase"], "seed":20261023, "dependencies":list(deps),
        "dependency_policy":"success" if kind == "bumped" else "terminal", "implementation_pending":False,
        "uses_gpu":kind == "bumped", "requires_gpu":kind == "bumped", "gpu_lock_mode":"wrapper",
        "argv":[str(python or sys.executable),"-m","mediflow_medvlm.campaign_validation_cap_refinement","--config",str(path),"--receipt",config["success_receipt"]],
        "cwd":str(Path(repo).resolve() if repo else Path(__file__).resolve().parents[2]),
        "success_receipt":config["success_receipt"], "source_files":config["source_files"],
        **{k:config[k] for k in HASH_FIELDS}}
    if kind == "bumped":job.update(**cell_spec(config["cell_id"]),role=config["role"],calibration_round="bumped",requires_gpu=True)
    return job


def _env(config):
    for key in IDENTITY_KEYS:
        value = os.environ.get("MEDVLM_SUPERVISOR_"+key.upper())
        _require(value is None or value == config[key], "CAP_SUPERVISOR_IDENTITY_CHANGED")


def _decision_summary(config, *, deep=False):
    protocol, proofs, _ = _audit_initial(config["data"]["protocol"],config["state_path"],config["data"]["initial_proofs"],deep=deep)
    result = _decision(proofs,protocol)
    _require(_equal(result, config["data"]["expected_decision"]), "CAP_DECISION_CHANGED_AFTER_PLANNING")
    return {**result,"initial_proof_hash":canonical_hash(proofs),"cap_protocol_frozen":False,
            "decision_audit_succeeded":True,"scope":SCOPES["decision"]}


def _cpu_artifacts(config):
    names = ["config.json","summary.json"] + (["protocol.final.json"] if config["kind"] == "freeze" else [])
    return [_reference(Path(config["output_dir"])/name) for name in sorted(names)]


def _cpu_completed(config, expected):
    output = _absolute(config["output_dir"])
    artifacts = _cpu_artifacts(config)
    marker = _read(output/"completion.json")
    _require(not output.is_symlink() and not any(p.is_symlink() for p in output.iterdir())
        and {p.name for p in output.iterdir()} == {Path(r["path"]).name for r in artifacts}|{"completion.json"}
        and _equal(_read(output/"config.json"),config)
        and _equal({k:v for k,v in _read(output/"summary.json").items() if k != "finished_at"},expected)
        and marker.get("schema_version") == "medvlm.v6.validation-cap-refinement-receipt.v1"
        and marker.get("status") == "SUCCEEDED" and marker.get("scope") == config["scope"]
        and all(marker.get(k) == config[k] for k in IDENTITY_KEYS)
        and marker.get("artifacts") == artifacts and marker.get("campaign_terminal") is False
        and marker.get("test_files_read") is False
        and marker.get("caps") == expected["caps"]
        and marker.get("cap_protocol_frozen") is (config["kind"] == "freeze"),
        "CAP_CPU_COMPLETION_CHANGED")
    if config["kind"] == "freeze":
        _require(_equal(_read(output/"protocol.final.json"),expected["final_protocol"])
            and marker.get("all_conditions_technical_complete") is expected["all_conditions_technical_complete"], "FROZEN_CAP_PROTOCOL_ARTIFACT_CHANGED")
    return {**marker,"artifacts":artifacts+[_reference(output/"completion.json")]}


def _actual_decision(config):
    parent = _ref(config["data"]["decision_config"])
    validate_config(parent)
    _require(parent["success_receipt"] == config["data"]["decision_receipt"], "CAP_DECISION_RECEIPT_ROUTE_CHANGED")
    manifest,state = _active(config["state_path"])
    job = next((j for j in manifest["jobs"] if j["job_id"] == DECISION_ID),None)
    _require(job is not None and DECISION_ID in state["jobs"], "CAP_DECISION_JOB_NOT_REGISTERED")
    record = state["jobs"][DECISION_ID]
    _require(record.get("status") == "SUCCEEDED", "SUCCESSFUL_ACTUAL_CAP_DECISION_REQUIRED")
    actual_config, ref = _child_config(job,"campaign_validation_cap_refinement")
    _require(ref == config["data"]["decision_config"] and _equal(actual_config,parent), "ACTUAL_CAP_DECISION_CONFIG_CHANGED")
    receipt = initial.gate._success(job,record)
    summary = _decision_summary(parent)
    _require(_equal(_cpu_completed(parent,summary),receipt), "ACTUAL_CAP_DECISION_RESULT_CHANGED")
    return parent,summary,_reference(parent["success_receipt"])


def _effective(config):
    parent,decision,reference = _actual_decision(config)
    protocol = _ref(config["data"]["protocol"])
    _require(_equal(config["data"].get("expected_decision",parent["data"]["expected_decision"]),parent["data"]["expected_decision"]), "RERUN_DECISION_CHANGED")
    selected = initial._selected(protocol,config["task"],set(decision["bumped_task_types"]))
    _require([r["sample_id"] for r,_ in selected] == config["data"]["ordered_sample_ids"], "RERUN_AFFECTED_IDS_CHANGED")
    return protocol,decision,selected,reference


def _native_proof(config,protocol,metrics,selected):
    if not selected:
        _require(metrics.get("gpu_accounting") is None and metrics.get("no_gpu_work_needed") is True,
                 "EMPTY_RERUN_MUST_NOT_CLAIM_GPU_WORK")
        return
    from .evaluation.v6_native import normalized_base_dtype_signature
    contract = metrics.get("native_contract",{})
    expected = _ref(protocol["models"][config["cell_id"]]["metrics"])["native_contract"]
    _require(all(_equal(contract.get(k),expected.get(k)) for k in ("architecture","template","thinking","loader","revision","snapshot_path","processor_class","quantization","native_runtime_compatibility"))
        and contract.get("weights_loaded") is True and contract.get("cuda_residency_verified") is True
        and contract.get("evaluation_role") == config["role"]
        and contract.get("parameter_storage",{}).get("trainable_parameters") == 0
        and normalized_base_dtype_signature(contract["parameter_storage"]) == normalized_base_dtype_signature(expected["parameter_storage"]), "RERUN_NATIVE_CONTRACT_CHANGED")
    _require(contract.get("evaluation_adapter_sha256") == (protocol["models"][config["cell_id"]]["adapter_sha256"] if config["role"] == "tuned" else None)
        and (contract["parameter_storage"]["adapter_parameters"] > 0 if config["role"] == "tuned" else contract["parameter_storage"]["adapter_parameters"] == 0), "RERUN_ROLE_ADAPTER_CHANGED")
    account = metrics.get("gpu_accounting",{})
    _require(type(account.get("session_seconds")) in (int,float) and 0 <= account["session_seconds"] < float("inf")
        and account.get("ledger_path") == config["gpu_ledger_path"]
        and account.get("snapshot",{}).get("resource_policy") == ResourcePolicy().as_dict(), "RERUN_GPU_ACCOUNTING_CHANGED")


def _bumped_completed(config):
    protocol,decision,selected,decision_ref = _effective(config)
    output = _absolute(config["output_dir"])
    marker = _read(output/"completion.json")
    names = ("config.json","metrics.json","predictions.private.jsonl")
    artifacts = [_reference(output/name) for name in names]
    _require(not output.is_symlink() and not any(p.is_symlink() for p in output.iterdir())
        and {p.name for p in output.iterdir()} == set(names)|{"completion.json"}
        and _equal(_read(output/"config.json"),config)
        and marker.get("schema_version") == "medvlm.v6.validation-cap-refinement-receipt.v1"
        and marker.get("status") == "SUCCEEDED" and marker.get("scope") == config["scope"]
        and all(marker.get(k) == config[k] for k in IDENTITY_KEYS)
        and marker.get("artifacts") == artifacts and marker.get("decision_receipt") == decision_ref
        and marker.get("test_files_read") is False and marker.get("campaign_terminal") is False,
        "BUMPED_RERUN_COMPLETION_CHANGED")
    rows = [json.loads(line) for line in (output/"predictions.private.jsonl").read_text().splitlines()]
    counts = initial._verify_rows(config,protocol,rows,selected,decision["caps"])
    metrics = _read(output/"metrics.json")
    _native_proof(config,protocol,metrics,selected)
    _require(metrics.get("task_counts") == counts and metrics.get("scheduled_count") == len(selected)
        and metrics.get("technical_failure_count") == sum(r["status"] != "SUCCEEDED" for r in rows)
        and marker.get("scheduled_count") == len(selected)
        and marker.get("skipped_no_task_cap_change") is (not selected)
        and marker.get("technical_complete") is all(r["status"] == "SUCCEEDED" for r in rows), "BUMPED_RERUN_DENOMINATOR_CHANGED")
    return {**marker,"artifacts":artifacts+[_reference(output/"completion.json")]}


def _freeze_summary(config, *, deep=False):
    parent,decision,decision_ref = _actual_decision(config)
    protocol,proofs,_ = _audit_initial(parent["data"]["protocol"],config["state_path"],parent["data"]["initial_proofs"],deep=deep)
    manifest,state = _active(config["state_path"])
    jobs = {j["job_id"]:j for j in manifest["jobs"]}
    conditions,terminal_records,rerun_receipts = {},{},{}
    for reference,key in zip(config["data"]["bumped_configs"],BUMPED_IDS):
        child = _ref(reference)
        job,record = jobs[key],state["jobs"][key]
        _terminal(record)
        actual,actual_ref = _child_config(job,"campaign_validation_cap_refinement")
        _require(actual_ref == reference and _equal(actual,child), "FROZEN_RERUN_JOB_CHANGED")
        terminal_records[key] = copy.deepcopy(record)
        initial_key = f"v6-caps-initial-{child['cell_id']}-{child['role']}"
        prior = proofs[initial_key]
        initial_config = _ref(prior["config"])
        initial_rows = [json.loads(line) for line in (Path(initial_config["output_dir"])/"predictions.private.jsonl").read_text().splitlines()] if prior["terminal_state"]["status"] == "SUCCEEDED" else []
        rerun_rows = []
        if record["status"] == "SUCCEEDED":
            receipt = initial.gate._success(job,record)
            _require(_equal(_bumped_completed(child),receipt), "FROZEN_RERUN_RESULT_CHANGED")
            rerun_rows = [json.loads(line) for line in (Path(child["output_dir"])/"predictions.private.jsonl").read_text().splitlines()]
            rerun_receipts[key] = _reference(job["success_receipt"])
        else:rerun_receipts[key] = None
        original_by_id = {r["sample_id"]:r for r in initial_rows}
        rerun_by_id = {r["sample_id"]:r for r in rerun_rows}
        final_rows,unavailable = [],[]
        for row,stratum in initial._selected(protocol,child["task"]):
            bumped = row["task_type"] in decision["bumped_task_types"]
            result = (rerun_by_id if bumped else original_by_id).get(row["sample_id"])
            if result is None:
                # An unavailable planned row is an explicit failure, never a
                # lower-cap prediction silently substituted after rerun failure.
                unavailable.append(row["sample_id"])
                result = initial._failure(row,stratum,decision["caps"][row["task_type"]],
                    "BUMPED_JOB_UNAVAILABLE" if bumped else "INITIAL_JOB_UNAVAILABLE")
            _require(result["max_new_tokens"] == decision["caps"][row["task_type"]], "FINAL_ROW_WRONG_GLOBAL_CAP")
            final_rows.append(result)
        counts = initial._task_counts(final_rows)
        complete = all(r["status"] == "SUCCEEDED" for r in final_rows)
        conditions[key] = {"cell_id":child["cell_id"],"role":child["role"],
            "initial_status":prior["terminal_state"]["status"],"bumped_status":record["status"],
            "initial_task_counts":prior["task_counts"],"final_task_counts":counts,
            "planned_count":len(final_rows),"unavailable_count":len(unavailable),"unavailable_ids_sha256":canonical_hash(unavailable),
            "technical_complete":complete,"clinical_quality_certified":False}
    final_protocol = {"schema_version":"medvlm.v6.validation-caps-frozen.v1",
        "initial_protocol":config["data"]["protocol"],"initial_protocol_hash":protocol["protocol_hash"],
        "caps":decision["caps"],"bumped_task_types":decision["bumped_task_types"],"bump_rounds":decision["bump_rounds"],
        "decision_receipt":decision_ref,"rerun_receipts":rerun_receipts,
        "eval_context":4096,"train_context":2048,"decode":protocol["decode"],
        "fixed_template_date":protocol["fixed_template_date"],"task_grouping":protocol["task_grouping"],
        "condition_coverage":conditions,"cap_protocol_frozen":True,"further_bump_allowed":False,
        "test_files_read":False,"full_dev_completed":False,"family_selection_completed":False,
        "test_unlocked":False,"clinical_quality_certified":False,"v6_completed":False,"campaign_terminal":False}
    final_protocol["protocol_hash"] = canonical_hash(final_protocol)
    return {"scope":SCOPES["freeze"],"cap_freeze_audit_succeeded":True,"cap_protocol_frozen":True,
        "caps":decision["caps"],"final_protocol":final_protocol,"conditions":conditions,"terminal_records":terminal_records,
        "all_conditions_technical_complete":all(c["technical_complete"] for c in conditions.values()),
        "test_files_read":False,"full_dev_completed":False,"family_selection_completed":False,
        "test_unlocked":False,"clinical_quality_certified":False,"v6_completed":False,"campaign_terminal":False}


def verify_result(config, *, repo=None):
    validate_config(config,repo=repo)
    if config["kind"] == "bumped":return _bumped_completed(config)
    summary = _decision_summary(config) if config["kind"] == "decision" else _freeze_summary(config)
    return _cpu_completed(config,summary)


def run_job(config_path,receipt_path):
    path,receipt_path = _absolute(config_path),_absolute(receipt_path)
    before,config = file_hash(path),_read(path)
    validate_config(config)
    _env(config)
    _require(str(receipt_path) == config["success_receipt"], "REFINEMENT_RECEIPT_ROUTE_CHANGED")
    if config["kind"] == "bumped":initial.first.main_train.smoke._inherited_supervisor_fds(config)
    output = _absolute(config["output_dir"])
    output.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    with output.with_name(output.name+".worker.lock").open("a") as lock:
        os.chmod(lock.name,0o600)
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not output.exists():
            _require(not receipt_path.exists(), "REFINEMENT_RECEIPT_WITHOUT_OUTPUT")
            staging = Path(tempfile.mkdtemp(prefix=output.name+".attempt-",dir=output.parent))
            mask = os.umask(0o077)
            try:
                atomic_json(staging/"config.json",config)
                if config["kind"] == "bumped":
                    protocol,decision,selected,decision_ref = _effective(config)
                    ledger = None
                    with (staging/"predictions.private.jsonl").open("x") as stream:
                        if selected:
                            ledger = BudgetLedger(config["gpu_ledger_path"],policy=ResourcePolicy())
                            ledger.start(config["model_id"],config["run_id"],"validation_cap_once_bumped")
                            try:rows,native = initial._execute(config,protocol,selected,decision["caps"],stream)
                            finally:ledger.finish()
                        else:
                            rows,native = [],{"no_gpu_work_needed":True}
                            stream.flush();os.fsync(stream.fileno())
                    counts = initial._verify_rows(config,protocol,rows,selected,decision["caps"])
                    metrics = {"task_counts":counts,"scheduled_count":len(selected),
                        "technical_failure_count":sum(r["status"] != "SUCCEEDED" for r in rows),**native,
                        "gpu_accounting":{"session_seconds":ledger.session_seconds,"ledger_path":config["gpu_ledger_path"],"snapshot":ledger.snapshot()} if ledger else None,
                        "finished_at":utcnow()}
                    _native_proof(config,protocol,metrics,selected)
                    atomic_json(staging/"metrics.json",metrics)
                    names = ("config.json","metrics.json","predictions.private.jsonl")
                    extra = {"decision_receipt":decision_ref,"cell_id":config["cell_id"],"role":config["role"],
                        "calibration_round":"bumped","scheduled_count":len(selected),
                        "technical_complete":all(r["status"] == "SUCCEEDED" for r in rows),"skipped_no_task_cap_change":not selected}
                else:
                    summary = _decision_summary(config,deep=True) if config["kind"] == "decision" else _freeze_summary(config,deep=True)
                    atomic_json(staging/"summary.json",{**summary,"finished_at":utcnow()})
                    names = ["config.json","summary.json"]
                    if config["kind"] == "freeze":
                        atomic_json(staging/"protocol.final.json",summary["final_protocol"])
                        names.append("protocol.final.json")
                    names = sorted(names)
                    extra = {"caps":summary["caps"],"cap_protocol_frozen":config["kind"] == "freeze"}
                    if config["kind"] == "freeze":extra["all_conditions_technical_complete"] = summary["all_conditions_technical_complete"]
                _require(file_hash(path) == before,"REFINEMENT_CONFIG_CHANGED_DURING_EXECUTION")
                validate_config(config)
                marker = {"schema_version":"medvlm.v6.validation-cap-refinement-receipt.v1","status":"SUCCEEDED","scope":config["scope"],
                    **{k:config[k] for k in IDENTITY_KEYS},**extra,"test_files_read":False,"campaign_terminal":False,
                    "artifacts":[{"path":str(output/name),"sha256":file_hash(staging/name)} for name in names]}
                atomic_json(staging/"completion.json",marker)
                os.rename(staging,output)
                fd = os.open(output.parent,os.O_DIRECTORY)
                try:os.fsync(fd)
                finally:os.close(fd)
            except BaseException as exc:
                if staging.exists():atomic_json(staging/"failure.private.json",{"status":"FAILED","exception_type":type(exc).__name__,"exception_message":str(exc),"traceback":traceback.format_exc(),"finished_at":utcnow()})
                raise
            finally:os.umask(mask)
        result = verify_result(config)
        _require(file_hash(path) == before and (not receipt_path.exists() or _equal(_read(receipt_path),result)),"REFINEMENT_REUSE_CHANGED")
        if not receipt_path.exists():atomic_json(receipt_path,result)
        return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--receipt",type=Path,required=True)
    args=parser.parse_args(argv)
    try:result=run_job(args.config,args.receipt)
    except Exception as exc:
        print(json.dumps({"status":"FAILED","exception_type":type(exc).__name__,**({"reason_code":str(exc)} if isinstance(exc,initial.gate.GateError) else {})}))
        return 1
    print(json.dumps({"status":result["status"],"scope":result["scope"],"cap_protocol_frozen":result.get("cap_protocol_frozen",False)}))
    return 0


if __name__ == "__main__":raise SystemExit(main())
