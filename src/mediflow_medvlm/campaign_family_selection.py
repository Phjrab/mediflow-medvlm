"""CPU-only R2 family/extra-seed investment freeze from recorded validation.

Only absolute Tuned generation scores rank candidates. Bootstrap and choice
likelihood are pinned auxiliary evidence, never ranking inputs. This module has
no test reader, training action, queue writer, or GPU import.
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

from . import campaign_bootstrap as bootstrap
from . import campaign_choice_likelihood as choice
from . import campaign_full_dev as dev
from .campaign_supervisor import HASH_FIELDS, atomic_json, canonical_hash, file_hash
from .runtime import ResourcePolicy

SCHEMA = "medvlm.v6.family-selection-config.v1"
RECEIPT_SCHEMA = "medvlm.v6.family-selection-receipt.v1"
SCOPE = "VALIDATION_ONLY_RECORDED_FAMILY_AND_EXTRA_SEED_FREEZE"
JOB_ID = "v6-family_selection"
SEEDS = (20261114, 20261115)
SOURCE_FILES = tuple(dict.fromkeys((*dev.SOURCE_FILES, "campaign_bootstrap.py", "campaign_choice_likelihood.py", "campaign_family_selection.py")))
IDENTITY_KEYS = dev.IDENTITY_KEYS
_require, _equal = dev._require, dev._equal
PROCEDURE = {
    "version":"r2-family-selection.v1", "first_seed":20261023, "additional_seeds":list(SEEDS),
    "primary":"absolute Tuned generation validation macro_score; never Base-to-Tuned gain",
    "text":"K=mean(71875_mcq strict all-row accuracy,71875_short lexical F1,71875_long lexical F1); E=mean(pubmedqa_labeled strict all-row accuracy,medquad lexical F1); macro_score=0.5*K+0.5*E",
    "vqa":"macro_score=0.5*(slake_open normalized EM+slake_closed normalized EM)",
    "denominator":"all scheduled rows in every stratum, including invalid/abstention/technical failures",
    "eligibility":"Tuned native role SUCCEEDED and technical_complete true with zero technical failures; Base failures do not exclude Tuned",
    "family":"one eligible representative per declared family, then top two different families separately for text and vqa",
    "ties":"unrounded macro_score descending; unrounded mean of the two classification strict accuracies (text) or scheduled-weighted overall normalized EM (vqa) descending; model ID Unicode lexicographic ascending",
    "shortfall":"select available eligible families only, report fewer than two and all excluded candidates; no replacement using test",
    "auxiliary":"bootstrap and choice likelihood audit pinned separately; neither scores nor technical completeness affect generation ranking",
    "interpretation":"research investment rule; not clinical accuracy or statistical superiority",
}
FLAGS = {"family_selection_completed":True, "extra_seed_targets_frozen":True,
         "test_files_read":False, "test_unlocked":False, "additional_seeds_completed":False,
         "r2_full_evaluation_completed":False, "v6_completed":False, "campaign_terminal":False,
         "clinical_quality_certified":False}


def relevant_source_files(): return SOURCE_FILES


def source_identity(repo=None):
    package = Path(__file__).resolve().parent if repo is None else Path(repo)/"src/mediflow_medvlm"
    files = {"src/mediflow_medvlm/"+name:file_hash(package/name) for name in SOURCE_FILES}
    return files, canonical_hash(files)


def _absolute(path):
    path = dev._absolute(path)
    _require(not path.is_symlink() and not any(p.is_symlink() for p in path.parents), "SELECTION_SYMLINK_FORBIDDEN")
    _require(not any(p.lower() in {"test", "tests"} or p.lower().startswith("test.") for p in path.parts), "SELECTION_TEST_PATH_FORBIDDEN")
    return path


def _read(path): return dev._read(_absolute(path))
def _reference(path): return {"path":str(_absolute(path)), "sha256":file_hash(_absolute(path))}


def _ref(reference):
    _require(isinstance(reference, dict) and _reference(reference["path"]) == reference, "SELECTION_REFERENCE_CHANGED")
    return _read(reference["path"])


def _cpu():
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and "torch" not in sys.modules, "SELECTION_FRESH_CPU_PROCESS_REQUIRED")


def _audit_job(job, record, module):
    """Registered native config, original source, canonical success, exact attempt."""
    _require(record.get("status") == "SUCCEEDED", "SELECTION_NATIVE_SUCCESS_REQUIRED")
    argv = job.get("argv", [])
    _require(len(argv) == 7, "SELECTION_NATIVE_COMMAND_REQUIRED")
    _absolute(argv[4]); _absolute(job["success_receipt"])
    for name in job.get("source_files", {}): _absolute(Path(job["cwd"])/name)
    config, reference = dev.caps._child_config(job, module)
    _absolute(reference["path"]); _absolute(job["success_receipt"])
    for item in _read(job["success_receipt"]).get("artifacts", []): _absolute(item["path"])
    receipt = dev.caps.initial.gate._success(job, record)
    _require(receipt.get("test_files_read") is False and receipt.get("test_unlocked") is False,
             "SELECTION_VALIDATION_ONLY_RECEIPT_REQUIRED")
    attempt = _absolute(record["attempt_dir"]); root = _absolute(config["state_path"]).parent
    _require(attempt.is_relative_to(root/"attempts"/job["job_id"]), "SELECTION_NATIVE_ATTEMPT_ROUTE_CHANGED")
    launch, exited = _read(attempt/"launch.json"), _read(attempt/"exit.json")
    token = record.get("launch_token")
    _require(bool(token) and launch.get("launch_token") == token == exited.get("launch_token")
             and exited.get("exit_code") == 0 and launch.get("attempt") == record["attempts"]
             and launch.get("campaign_id") == root.name and launch.get("job_hash") == canonical_hash(job)
             and _equal(launch.get("job"), job), "SELECTION_NATIVE_ATTEMPT_IDENTITY_CHANGED")
    proof = {"job":copy.deepcopy(job), "config":reference, "receipt":_reference(job["success_receipt"]),
             "artifacts":copy.deepcopy(receipt["artifacts"]), "launch":_reference(attempt/"launch.json"),
             "exit":_reference(attempt/"exit.json"),
             "terminal_record":{k:copy.deepcopy(record[k]) for k in ("status","attempts","failure_count","recovery_attempts","attempt_dir","launch_token","exit_code","verification")}}
    return config, receipt, proof


def _rate(value):
    _require(type(value) in (int,float) and math.isfinite(value) and 0 <= value <= 1, "SELECTION_INVALID_RATE")
    return float(value)


def score_components(strata, axis):
    expected = ("71875_mcq","71875_short","71875_long","pubmedqa_labeled","medquad") if axis == "text" else ("slake_open","slake_closed")
    _require(axis in {"text","vqa"} and set(strata) == set(expected), "SELECTION_EXACT_STRATA_REQUIRED")
    _require(all(type(s.get("scheduled")) is int and s["scheduled"] > 0
                 and type(s.get("technical_failure_count")) is int and 0 <= s["technical_failure_count"] <= s["scheduled"] for s in strata.values()), "SELECTION_INVALID_DENOMINATOR")
    if axis == "text":
        values = {key:_rate(strata[key]["strict_all_row_accuracy"] if key in {"71875_mcq","pubmedqa_labeled"} else strata[key]["lexical_f1"]) for key in expected}
        k = sum(values[key] for key in expected[:3])/3
        e = (values["pubmedqa_labeled"]+values["medquad"])/2
        primary, tie = 0.5*k+0.5*e, (values["71875_mcq"]+values["pubmedqa_labeled"])/2
        components = {"K":k,"E":e,**values}
    else:
        values = {key:_rate(strata[key]["normalized_em"]) for key in expected}
        primary = (values["slake_open"]+values["slake_closed"])/2
        tie = sum(values[key]*strata[key]["scheduled"] for key in expected)/sum(s["scheduled"] for s in strata.values())
        components = values
    return {"macro_score":primary, "tie_break_score":tie, "score_components":components,
            "scheduled":sum(s["scheduled"] for s in strata.values()),
            "strata":copy.deepcopy(strata)}


def select_candidates(candidates):
    """Pure deterministic R2 selector; exact declared matrix, no external readers."""
    _require([c.get("cell_id") for c in candidates] == [c[0] for c in dev.CELLS], "SELECTION_EXACT_TEN_CANDIDATES_REQUIRED")
    for c in candidates:
        _require(all(c.get(k) == v for k,v in dev.cell_spec(c["cell_id"]).items()), "SELECTION_DECLARED_MODEL_FAMILY_CHANGED")
        _require(type(c.get("eligible")) is bool, "SELECTION_ELIGIBILITY_REQUIRED")
        if c["eligible"]: _rate(c["macro_score"]); _rate(c["tie_break_score"])
    ordered = lambda rows: sorted(rows, key=lambda c:(-c["macro_score"],-c["tie_break_score"],c["model_id"]))
    representatives, rankings, selected, shortfalls = {}, {}, {}, {}
    for axis in ("text","vqa"):
        reps = {}
        for c in ordered([c for c in candidates if c["task"] == axis and c["eligible"]]): reps.setdefault(c["family"], c)
        rank = ordered(list(reps.values())); representatives[axis] = {family:c["cell_id"] for family,c in sorted(reps.items())}
        rankings[axis] = [c["cell_id"] for c in rank]; selected[axis] = rankings[axis][:2]
        shortfalls[axis] = {"requested":2,"selected":len(selected[axis]),"reason":None if len(rank)>=2 else "FEWER_THAN_TWO_TECHNICALLY_COMPLETE_FAMILIES"}
    selected_models = [copy.deepcopy(next(c for c in candidates if c["cell_id"] == key)) for axis in ("text","vqa") for key in selected[axis]]
    plan = [{**{k:c[k] for k in ("cell_id","model_id","family","task")}, "seed":seed} for c in selected_models for seed in SEEDS]
    return {"selected_cells_by_axis":selected,"selected_models":selected_models,"additional_seed_plan":plan,
            "additional_seed_count":len(plan),"family_representatives":representatives,"family_rankings":rankings,"shortfalls":shortfalls,
            "candidates":copy.deepcopy(candidates),"procedure":copy.deepcopy(PROCEDURE),**FLAGS}


def _collect(full_dev_path, bootstrap_path, choice_path):
    aggregate = _read(full_dev_path); manifest, state = dev.caps._active(aggregate["state_path"])
    jobs = {job["job_id"]:job for job in manifest["jobs"]}; root = _absolute(aggregate["state_path"]).parent
    _require(manifest["campaign_id"] == root.name and root.name.startswith("v6-forge-recorded-"), "SELECTION_RECORDED_NAMESPACE_REQUIRED")
    aggregate, dev_receipt, native = _audit_job(jobs[dev.AGGREGATE_ID], state["jobs"][dev.AGGREGATE_ID], "campaign_full_dev")
    _require(native["config"] == _reference(full_dev_path) and aggregate.get("kind") == "aggregate", "SELECTION_REGISTERED_FULL_DEV_REQUIRED")
    original = _ref(aggregate["data"]["initial_protocol"])
    _require(original["campaign_id"] == root.name and original["campaign_root"] == str(root), "SELECTION_INITIAL_PROTOCOL_NAMESPACE_CHANGED")
    for axis in ("text","vqa"): _absolute(original["validation_references"][axis]["path"])
    for model in original["models"].values():
        for key in ("config","receipt","checkpoint","metrics","reload","final_selection"): _absolute(model[key]["path"])
    _require(_equal(dev.verify_result(aggregate, repo=jobs[dev.AGGREGATE_ID]["cwd"]),dev_receipt), "SELECTION_NATIVE_FULL_DEV_CHANGED")
    summary = _read(Path(aggregate["output_dir"])/"summary.json")
    _require(summary.get("aggregate_audit_succeeded") is True and set(summary["conditions"]) == set(dev.ROLE_IDS), "SELECTION_EXACT_FULL_DEV_MATRIX_REQUIRED")
    validation = {}
    for axis in ("text","vqa"):
        _absolute(original["validation_references"][axis]["path"])
        rows = dev._rows(original,axis)
        _require(all(row.get("split") == "validation" for row,_ in rows), "SELECTION_VALIDATION_ROWS_ONLY")
        ids = [row["sample_id"] for row,_ in rows]
        _require(len(ids) == len(set(ids)), "SELECTION_DUPLICATE_VALIDATION_IDS")
        validation[axis] = {"reference":{k:original["validation_references"][axis][k] for k in ("path","sha256")},
                            "ordered_sample_ids":ids,"ordered_sample_ids_sha256":canonical_hash(ids),"strata_counts":dict(Counter(s for _,s in rows))}
    role_configs = [_ref(ref) for ref in aggregate["data"]["roles"]]
    _require([c["job_id"] for c in role_configs] == dev.ROLE_IDS, "SELECTION_ORDERED_NATIVE_ROLES_CHANGED")
    role_refs = dict(zip(dev.ROLE_IDS,aggregate["data"]["roles"]))
    candidates = []; role_proofs = []
    for cell in dev.CELLS:
        key = f"v6-full-dev-{cell[0]}-tuned"; child = next(c for c in role_configs if c["job_id"] == key)
        condition = summary["conditions"][key]; spec = dev.cell_spec(cell[0]); strata = condition["strata"]
        _require({k:s["scheduled"] for k,s in strata.items()} == validation[spec["task"]]["strata_counts"], "SELECTION_ALL_PLANNED_DENOMINATORS_CHANGED")
        _require(child["data"]["initial_protocol"] == aggregate["data"]["initial_protocol"]
                 and child["data"]["ordered_sample_ids"] == validation[spec["task"]]["ordered_sample_ids"], "SELECTION_COMMON_VALIDATION_CHANGED")
        scored = score_components(strata,spec["task"])
        eligible = condition["status"] == "SUCCEEDED" and condition["technical_complete"] is True and not any(s["technical_failure_count"] for s in strata.values())
        reason = None if eligible else "TUNED_NATIVE_ROLE_UNAVAILABLE" if condition["status"] != "SUCCEEDED" else "TUNED_TECHNICALLY_INCOMPLETE"
        proof = None
        if condition["status"] == "SUCCEEDED":
            actual, rec, proof = _audit_job(jobs[key],state["jobs"][key],"campaign_full_dev")
            _require(_equal(actual,child) and rec["technical_complete"] == condition["technical_complete"], "SELECTION_TUNED_RESULT_CHANGED")
        role_proofs.append({"cell_id":cell[0],"native":proof,"terminal_record":copy.deepcopy(summary["terminal_records"][key])})
        candidates.append({**spec,**scored,"eligible":eligible,"exclusion_reason":reason,"native_status":condition["status"],
                           "technical_complete":condition["technical_complete"],"full_dev_role":role_refs[key],
                           "full_dev_receipt":condition["receipt"],"protocol":copy.deepcopy(child["data"]["evaluation_protocol"]),
                           "trained_cell":copy.deepcopy(original["models"][cell[0]])})
    auxiliary = {}
    for key,path,module in ((bootstrap.JOB_ID,bootstrap_path,"campaign_bootstrap"),(choice.AGGREGATE_ID,choice_path,"campaign_choice_likelihood")):
        conf,rec,proof = _audit_job(jobs[key],state["jobs"][key],module)
        _require(proof["config"] == _reference(path) and conf["state_path"] == aggregate["state_path"], "SELECTION_REGISTERED_AUXILIARY_CHANGED")
        if key == bootstrap.JOB_ID:
            _require(conf.get("procedure") == bootstrap.PROCEDURE and conf["data"]["aggregate_config"] == native["config"]
                     and rec.get("bootstrap_auxiliary_completed") is True, "SELECTION_BOOTSTRAP_PROVENANCE_CHANGED")
        else:
            _require(conf.get("kind") == "aggregate" and conf.get("procedure") == choice.PROCEDURE
                     and conf["data"]["initial_protocol"] == aggregate["data"]["initial_protocol"]
                     and rec.get("scope") == choice.SCOPES["aggregate"], "SELECTION_CHOICE_PROVENANCE_CHANGED")
        auxiliary[key] = proof
    return {"full_dev":native,"tuned_roles":role_proofs,"bootstrap":auxiliary[bootstrap.JOB_ID],"choice_likelihood":auxiliary[choice.AGGREGATE_ID],
            "initial_protocol":aggregate["data"]["initial_protocol"],"final_protocol":aggregate["data"]["protocol"],"validation":validation,
            "candidates":candidates}, aggregate


def prepare_config(full_dev_aggregate_config_path, bootstrap_config_path, choice_aggregate_config_path, output_dir, *, repo=None, success_receipt=None):
    data, aggregate = _collect(full_dev_aggregate_config_path,bootstrap_config_path,choice_aggregate_config_path)
    root = _absolute(aggregate["state_path"]).parent; output = _absolute(output_dir)
    receipt = _absolute(success_receipt or root/"receipts"/(JOB_ID+".json"))
    for path in (output,receipt): dev.caps.initial.gate._private(path,root)
    _require(output.is_relative_to(root/"family_selection") and output != root/"family_selection"
             and receipt == root/"receipts"/(JOB_ID+".json"), "SELECTION_DEDICATED_DESTINATION_REQUIRED")
    files, source = source_identity(repo)
    config = {"schema_version":SCHEMA,"scope":SCOPE,"kind":"family_selection","phase":"family_selection",
              "job_id":JOB_ID,"run_id":JOB_ID,"cell_id":"ALL_V6","seed":20261023,"uses_gpu":False,"device":"cpu",
              "state_path":aggregate["state_path"],"output_dir":str(output),"success_receipt":str(receipt),
              "data":data,"data_hash":canonical_hash(data),"model_hash":aggregate["model_hash"],"procedure":copy.deepcopy(PROCEDURE),
              "resource_policy":ResourcePolicy().as_dict(),"source_files":files,"source_hash":source}
    config["config_hash"] = canonical_hash(config)
    return config


def validate_config(config, *, repo=None):
    _require(config.get("schema_version") == SCHEMA and config.get("config_hash") == canonical_hash({k:v for k,v in config.items() if k != "config_hash"})
             and (config.get("source_files"),config.get("source_hash")) == source_identity(repo), "SELECTION_CONFIG_HASH_OR_SOURCE_CHANGED")
    expected = prepare_config(config["data"]["full_dev"]["config"]["path"],config["data"]["bootstrap"]["config"]["path"],
                              config["data"]["choice_likelihood"]["config"]["path"],config["output_dir"],repo=repo,success_receipt=config["success_receipt"])
    _require(_equal(config,expected), "SELECTION_FIXED_PROCEDURE_OR_PROVENANCE_CHANGED")
    return config


def job_from_config(config_path, *, repo=None, python=None):
    path = _absolute(config_path); config = _read(path); validate_config(config,repo=repo)
    return {"job_id":JOB_ID,"run_id":JOB_ID,"cell_id":"ALL_V6","phase":"family_selection","kind":"family_selection","seed":20261023,
            "dependencies":[dev.AGGREGATE_ID,bootstrap.JOB_ID,choice.AGGREGATE_ID],"dependency_policy":"success",
            "implementation_pending":False,"uses_gpu":False,"requires_gpu":False,"gpu_lock_mode":"wrapper",
            "argv":[str(python or sys.executable),"-m","mediflow_medvlm.campaign_family_selection","--config",str(path),"--receipt",config["success_receipt"]],
            "cwd":str(Path(repo).resolve() if repo else Path(__file__).resolve().parents[2]),"success_receipt":config["success_receipt"],
            "source_files":config["source_files"],**{k:config[k] for k in HASH_FIELDS}}


def _selection(config):
    return {"schema_version":"medvlm.v6.family-selection.v1","scope":SCOPE,**select_candidates(config["data"]["candidates"]),
            "validation":config["data"]["validation"],"initial_protocol":config["data"]["initial_protocol"],
            "final_protocol":config["data"]["final_protocol"],"selection_code_hash":config["source_hash"],
            "native_evidence_hash":config["data_hash"]}


def verify_result(config, *, repo=None):
    validate_config(config,repo=repo); output = _absolute(config["output_dir"]); selection = _selection(config)
    names = ("config.json","selection.json")
    _require(set(p.name for p in output.iterdir()) == set(names)|{"completion.json"}
             and all(not p.is_symlink() for p in output.iterdir()), "SELECTION_OUTPUT_SET_CHANGED")
    marker = _read(output/"completion.json")
    _require(_equal(_read(output/"config.json"),config) and _equal(_read(output/"selection.json"),selection)
             and marker.get("schema_version") == RECEIPT_SCHEMA and marker.get("status") == "SUCCEEDED" and marker.get("scope") == SCOPE
             and all(marker.get(k) == config[k] for k in IDENTITY_KEYS) and all(marker.get(k) == v for k,v in FLAGS.items())
             and marker.get("selection_sha256") == file_hash(output/"selection.json")
             and marker.get("artifacts") == [_reference(output/name) for name in names]
             and all(_equal(marker.get(key),selection[key]) for key in ("selected_cells_by_axis","selected_models","additional_seed_plan","additional_seed_count")), "SELECTION_RESULT_CHANGED")
    return {**marker,"artifacts":marker["artifacts"]+[_reference(output/"completion.json")]}


def run_job(config_path, receipt_path):
    _cpu(); path = _absolute(config_path); config = _read(path); before = file_hash(path)
    validate_config(config); dev.caps._env(config)
    receipt = _absolute(receipt_path); _require(str(receipt) == config["success_receipt"], "SELECTION_RECEIPT_ROUTE_CHANGED")
    output = _absolute(config["output_dir"]); output.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    with output.with_name(output.name+".worker.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not output.exists():
            _require(not receipt.exists(), "SELECTION_RECEIPT_WITHOUT_OUTPUT")
            staging = Path(tempfile.mkdtemp(prefix=output.name+".attempt-",dir=output.parent)); mask = os.umask(0o077)
            try:
                selection = _selection(config)
                atomic_json(staging/"config.json",config); atomic_json(staging/"selection.json",selection)
                validate_config(config); _cpu(); _require(file_hash(path) == before, "SELECTION_CONFIG_CHANGED_DURING_EXECUTION")
                marker = {"schema_version":RECEIPT_SCHEMA,"status":"SUCCEEDED","scope":SCOPE,**{k:config[k] for k in IDENTITY_KEYS},**FLAGS,
                          **{key:selection[key] for key in ("selected_cells_by_axis","selected_models","additional_seed_plan","additional_seed_count")},
                          "selection_sha256":file_hash(staging/"selection.json"),
                          "artifacts":[{"path":str(output/name),"sha256":file_hash(staging/name)} for name in ("config.json","selection.json")]}
                atomic_json(staging/"completion.json",marker); os.rename(staging,output)
                fd = os.open(output.parent,os.O_DIRECTORY)
                try: os.fsync(fd)
                finally: os.close(fd)
            finally: os.umask(mask)
        result = verify_result(config); _cpu()
        _require(file_hash(path) == before and (not receipt.exists() or _equal(_read(receipt),result)), "SELECTION_COMPLETED_REUSE_CHANGED")
        if not receipt.exists(): atomic_json(receipt,result)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True); parser.add_argument("--receipt",type=Path,required=True)
    args = parser.parse_args(argv)
    try:
        _require("torch" not in sys.modules,"SELECTION_FRESH_CPU_PROCESS_REQUIRED"); os.environ["CUDA_VISIBLE_DEVICES"] = ""
        result = run_job(args.config,args.receipt)
    except Exception as exc:
        print(json.dumps({"status":"FAILED","exception_type":type(exc).__name__,"reason_code":str(exc) if isinstance(exc,dev.caps.initial.gate.GateError) else "SELECTION_FAILURE"})); return 1
    print(json.dumps({"status":result["status"],"selected_cells_by_axis":result["selected_cells_by_axis"],"additional_seed_count":result["additional_seed_count"]})); return 0


if __name__ == "__main__": raise SystemExit(main())
