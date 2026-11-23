"""Recorded full common validation, twenty native roles and one CPU audit.

No test reader, family selector, extra scheduler, or training mutation lives here.
GPU jobs use the executed native decoder and frozen caps. Forge import is one
CPU-only attempt after native completion, recorded separately from scientific
success; it never turns a reload or calibration into a quality evaluation.
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
import subprocess
import tempfile
import traceback

from . import campaign_validation_cap_refinement as caps
from . import telemetry_import
from .evaluation import metrics as scoring
from .evaluation.v2 import normalize_vqa
from .campaign_supervisor import HASH_FIELDS, TERMINAL, atomic_json, canonical_hash, file_hash, verify_source
from .campaign_v6 import CELLS, cell_spec
from .runtime import BudgetLedger, ResourcePolicy, utcnow

SCHEMA="medvlm.v6.full-dev-config.v1"
PROTOCOL_VERSION="medvlm.v6.full-dev.v1"
SCOPES={"role":"VALIDATION_ONLY_NATIVE_FULL_DEV","aggregate":"VALIDATION_ONLY_FULL_DEV_TERMINAL_AUDIT"}
ROLE_IDS=[f"v6-full-dev-{cell[0]}-{role}" for cell in CELLS for role in caps.initial.ROLES]
AGGREGATE_ID="v6-full_dev"
SOURCE_FILES=tuple(dict.fromkeys((*caps.SOURCE_FILES,"evaluation/metrics.py","telemetry.py","telemetry_import.py","campaign_full_dev.py")))
IDENTITY_KEYS=caps.IDENTITY_KEYS
_require,_read,_absolute,_equal=caps._require,caps._read,caps._absolute,caps._equal
_ref,_reference=caps._ref,caps._reference


def source_identity(repo=None):
    package=Path(__file__).resolve().parent if repo is None else Path(repo)/"src/mediflow_medvlm"
    files={"src/mediflow_medvlm/"+name:file_hash(package/name) for name in SOURCE_FILES}
    return files,canonical_hash(files)


def _cap_proof(receipt_path,state_path):
    manifest,state=caps._active(state_path)
    job=next(j for j in manifest["jobs"] if j["job_id"]==caps.FREEZE_ID)
    record=state["jobs"][caps.FREEZE_ID]
    _require(record.get("status")=="SUCCEEDED" and job["success_receipt"]==str(_absolute(receipt_path)),"SUCCEEDED_CAP_FREEZE_REQUIRED")
    config,config_ref=caps._child_config(job,"campaign_validation_cap_refinement")
    _require(config.get("kind")=="freeze" and config.get("scope")==caps.SCOPES["freeze"]
        and (config["source_files"],config["source_hash"])==caps.source_identity(job["cwd"]),"CAP_FREEZE_SOURCE_OR_KIND_CHANGED")
    receipt=caps.initial.gate._success(job,record)
    output=Path(config["output_dir"])
    final_ref=_reference(output/"protocol.final.json")
    final=_ref(final_ref);summary=_read(output/"summary.json")
    _require(receipt.get("scope")==caps.SCOPES["freeze"] and receipt.get("cap_protocol_frozen") is True
        and final.get("schema_version")=="medvlm.v6.validation-caps-frozen.v1"
        and final.get("protocol_hash")==canonical_hash({k:v for k,v in final.items() if k!="protocol_hash"})
        and final.get("cap_protocol_frozen") is True and final.get("further_bump_allowed") is False
        and final.get("test_files_read") is False and final.get("test_unlocked") is False
        and final.get("eval_context")==4096 and final.get("train_context")==2048
        and final.get("fixed_template_date")=="2026-11-14"
        and final.get("decode")=={"do_sample":False,"num_beams":1,"use_cache":True,"max_time":None,"warmup_requests":0}
        and all(final["caps"].get(task) in (initial, caps.initial.MAXIMUM_CAPS[task]) for task,initial in caps.initial.INITIAL_CAPS.items())
        and _equal(summary.get("final_protocol"),final) and summary.get("cap_freeze_audit_succeeded") is True
        and config["data"]["protocol"]==final["initial_protocol"],"CAP_FREEZE_PROTOCOL_OR_SUMMARY_CHANGED")
    original=_ref(final["initial_protocol"])
    _require(original.get("protocol_hash")==final["initial_protocol_hash"]
        and original["protocol_hash"]==canonical_hash({k:v for k,v in original.items() if k!="protocol_hash"}),"CAP_INITIAL_PROTOCOL_CHANGED")
    proof={"job":copy.deepcopy(job),"terminal_state":copy.deepcopy(record),"config":config_ref,
        "receipt":_reference(receipt_path),"final_protocol":final_ref}
    return proof,final,original


def _rows(original,axis):
    ref=original["validation_references"][axis]
    rows=caps.initial._validation_rows({k:ref[k] for k in ("path","sha256")},1516 if axis=="text" else 512)
    return [(row,caps.initial._stratum(row)) for row in rows]


def _evaluation_protocol(final_ref,final,original,axis,*,repo=None):
    selected=_rows(original,axis);rows=[row for row,_ in selected]
    files,_=source_identity(repo)
    value={"protocol_version":PROTOCOL_VERSION,"split":"validation","cap_protocol":final_ref,
        "cap_protocol_hash":final["protocol_hash"],"caps":final["caps"],"eval_context":4096,
        "decode":final["decode"],"fixed_template_date":"2026-11-14",
        "sample_ids":[row["sample_id"] for row in rows],
        "input_digests":{row["sample_id"]:canonical_hash(row["input"]) for row in rows},
        "reference_digest":canonical_hash([(row["sample_id"],row["supervision"]) for row in rows]),
        "files":{axis:original["validation_references"][axis]["sha256"]},
        "parser":"existing anchored full-string parse_answer; no answer substring extraction",
        "normalization":"NFKC casefold whitespace; VQA terminal punctuation only; preserve numbers/units/negation",
        "scorer_source_files":{k:v for k,v in files.items() if k.endswith(("evaluation/metrics.py","evaluation/v2.py","campaign_full_dev.py"))},
        "planned_failures_in_denominator":True,"further_bump_allowed":False,"test_files_read":False}
    value["protocol_hash"]=canonical_hash(value)
    return value


def _training(original,cell_id):
    reference=original["models"][cell_id]
    _require(reference.get("eligible_for_calibration") is True,"AUDITED_ORIGINAL_FINAL_MODEL_REQUIRED")
    config=_ref(reference["config"])
    for key in ("checkpoint","reload","final_selection","metrics"):_ref(reference[key])
    _require(reference["model_hash"]==config["model_hash"] and reference["revision"]==config["model"]["revision"],"DEV_MODEL_REFERENCE_CHANGED")
    return config,reference


def build_config(cap_freeze_receipt_path,state_path,cell_id,role,output_dir,*,repo=None,success_receipt=None,forge_output_root=None):
    _require(cell_id in [c[0] for c in CELLS] and role in caps.initial.ROLES,"FULL_DEV_CELL_ROLE_REQUIRED")
    proof,final,original=_cap_proof(cap_freeze_receipt_path,state_path)
    train,model_ref=_training(original,cell_id);spec=cell_spec(cell_id)
    evaluation=_evaluation_protocol(proof["final_protocol"],final,original,spec["task"],repo=repo)
    selected=_rows(original,spec["task"])
    root=_absolute(original["campaign_root"]);data_root=Path(train["gpu_ledger_path"]).parent.parent
    job_id=f"v6-full-dev-{cell_id}-{role}";output=_absolute(output_dir)
    receipt=_absolute(success_receipt or root/"receipts"/(job_id+".json"))
    for path in (output,receipt):caps.initial.gate._private(path,root)
    telemetry_root=forge_output_root or os.environ.get("MEDVLM_TELEMETRY_ROOT")
    _require(bool(telemetry_root),"EXPLICIT_EXISTING_FORGE_TELEMETRY_ROOT_REQUIRED")
    telemetry_path=_absolute(telemetry_root)
    _require(not telemetry_path.resolve().is_relative_to(data_root.resolve())
        and not data_root.resolve().is_relative_to(telemetry_path.resolve()),"FORGE_OUTPUT_MUST_BE_DISJOINT_FROM_SCIENTIFIC_DATA")
    forge={"output_root":str(_absolute(telemetry_root)),"data_root":str(data_root),"campaign_id":original["campaign_id"],
           "receipt_path":str(root/"receipts"/(job_id+".forge-import.json"))}
    data={"cap_freeze":proof,"protocol":proof["final_protocol"],"initial_protocol":final["initial_protocol"],
        "validation":{k:original["validation_references"][spec["task"]][k] for k in ("path","sha256")},
        "ordered_sample_ids":[r["sample_id"] for r,_ in selected],"ordered_sample_ids_sha256":canonical_hash([r["sample_id"] for r,_ in selected]),
        "count":len(selected),"partition":"validation","image_root":train["data"]["image_root"],
        "trained_cell":model_ref,"first_seed_receipt":original["first_seed_receipt"],"evaluation_protocol":evaluation}
    files,source=source_identity(repo)
    config={"schema_version":SCHEMA,"scope":SCOPES["role"],"kind":"role","phase":"full_dev_evaluation",**spec,
        "job_id":job_id,"run_id":job_id,"role":role,"seed":20261023,"uses_gpu":True,"device":"cuda",
        "state_path":str(_absolute(state_path)),"output_dir":str(output),"success_receipt":str(receipt),"data":data,
        "model":train["model"],"model_hash":canonical_hash({"model":train["model"],"role":role,"final_checkpoint":model_ref["checkpoint"] if role=="tuned" else None}),
        "eval_context":4096,"split":"validation","protocol_version":PROTOCOL_VERSION,"protocol_hash":evaluation["protocol_hash"],
        "fixed_template_date":"2026-11-14","decode":final["decode"],"local_files_only":True,"trust_remote_code":False,
        "precision":"nf4_double_quant_bfloat16","resource_policy":ResourcePolicy().as_dict(),
        "gpu_ledger_path":str(data_root/"runtime/gpu_budget.json"),
        "supervisor_lock_paths":[str(data_root/"v6_v7/worker.lock"),str(data_root/"runtime/medvlm_gpu.lock")],
        "forge":forge,"source_files":files,"source_hash":source,"data_hash":canonical_hash(data)}
    config["config_hash"]=canonical_hash(config)
    return config


def build_aggregate_config(cap_freeze_receipt_path,state_path,role_config_paths_ordered20,output_dir,*,repo=None,success_receipt=None):
    proof,final,original=_cap_proof(cap_freeze_receipt_path,state_path)
    refs=[_reference(path) for path in role_config_paths_ordered20]
    configs=[_ref(ref) for ref in refs]
    _require([c.get("job_id") for c in configs]==ROLE_IDS,"EXACT_ORDERED_TWENTY_DEV_CONFIGS_REQUIRED")
    for config in configs:
        validate_config(config,repo=repo)
        _require(config["data"]["cap_freeze"]==proof,"DEV_CONFIG_CAP_PROOF_CHANGED")
    data={"cap_freeze":proof,"protocol":proof["final_protocol"],"initial_protocol":final["initial_protocol"],"roles":refs}
    output=_absolute(output_dir);root=_absolute(original["campaign_root"])
    receipt=_absolute(success_receipt or root/"receipts"/(AGGREGATE_ID+".json"))
    for path in (output,receipt):caps.initial.gate._private(path,root)
    files,source=source_identity(repo)
    config={"schema_version":SCHEMA,"scope":SCOPES["aggregate"],"kind":"aggregate","phase":"full_dev",
        "job_id":AGGREGATE_ID,"run_id":AGGREGATE_ID,"cell_id":"ALL_V6","seed":20261023,"uses_gpu":False,
        "state_path":str(_absolute(state_path)),"output_dir":str(output),"success_receipt":str(receipt),"data":data,
        "model_hash":canonical_hash(original["models"]),"resource_policy":ResourcePolicy().as_dict(),
        "source_files":files,"source_hash":source,"data_hash":canonical_hash(data)}
    config["config_hash"]=canonical_hash(config)
    return config


def validate_config(config,*,repo=None):
    _require(config.get("schema_version")==SCHEMA and config.get("config_hash")==canonical_hash({k:v for k,v in config.items() if k!="config_hash"})
        and (config.get("source_files"),config.get("source_hash"))==source_identity(repo),"DEV_CONFIG_HASH_OR_SOURCE_CHANGED")
    if config.get("kind")=="role":
        expected=build_config(config["data"]["cap_freeze"]["receipt"]["path"],config["state_path"],config["cell_id"],config["role"],config["output_dir"],
            repo=repo,success_receipt=config["success_receipt"],forge_output_root=config["forge"]["output_root"])
    else:
        _require(config.get("kind")=="aggregate","DEV_CONFIG_KIND_REQUIRED")
        expected=build_aggregate_config(config["data"]["cap_freeze"]["receipt"]["path"],config["state_path"],
            [r["path"] for r in config["data"]["roles"]],config["output_dir"],repo=repo,success_receipt=config["success_receipt"])
    _require(_equal(expected,config),"FULL_DEV_SCIENTIFIC_IDENTITY_CHANGED")
    return config


def job_from_config(config_path,*,repo=None,python=None):
    path,config=_absolute(config_path),_read(config_path);validate_config(config,repo=repo)
    role=config["kind"]=="role"
    job={"job_id":config["job_id"],"run_id":config["run_id"],"cell_id":config["cell_id"],"phase":config["phase"],"kind":config["kind"],"seed":20261023,
        "dependencies":[caps.FREEZE_ID] if role else [caps.FREEZE_ID,*ROLE_IDS],"dependency_policy":"success" if role else "terminal",
        "implementation_pending":False,"uses_gpu":role,"requires_gpu":role,"gpu_lock_mode":"wrapper",
        "argv":[str(python or sys.executable),"-m","mediflow_medvlm.campaign_full_dev","--config",str(path),"--receipt",config["success_receipt"]],
        "cwd":str(Path(repo).resolve() if repo else Path(__file__).resolve().parents[2]),"success_receipt":config["success_receipt"],
        "source_files":config["source_files"],**{k:config[k] for k in HASH_FIELDS}}
    if role:job.update(**cell_spec(config["cell_id"]),role=config["role"])
    return job


def score_predictions(selected,predictions):
    samples=[row for row,_ in selected]
    _require([p["sample_id"] for p in predictions]==[s["sample_id"] for s in samples],"DEV_SCORING_IDS_CHANGED")
    parsed=[]
    for sample,pred in zip(samples,predictions):
        if pred["status"]=="SUCCEEDED":answer,status=scoring.parse_answer(pred.get("generated_text",""),sample["task_type"],sample["input"].get("options"))
        else:answer,status=None,"technical_failure"
        parsed.append({"sample_id":sample["sample_id"],"parsed_answer":answer,"status":status,"attempted":pred.get("attempted",True),
            "finish_reason":pred.get("finish_reason"),"timing":{"e2e_seconds":pred.get("elapsed_seconds")}})
    scored=scoring.score_rows(samples,parsed,vqa_normalizer=normalize_vqa)
    for score,sample,pred,parsed_row,(_,stratum) in zip(scored,samples,predictions,parsed,selected):
        gold=sample["supervision"].get("option_id") if sample["task_type"]=="medical_mcq" else sample["supervision"]["answer"]
        score.update(stratum=stratum,raw_exact_match=float(pred["status"]=="SUCCEEDED" and pred.get("generated_text","")==str(gold)))
    strata={}
    for stratum in sorted({s for _,s in selected}):
        rows=[r for r in scored if r["stratum"]==stratum];n=len(rows);valid=sum(r["valid"] for r in rows);correct=sum(r["exact_match"] for r in rows)
        counts=Counter(r["status"] for r in rows);classification=rows[0]["task_type"] in {"medical_mcq","context_qa"}
        strata[stratum]={"scheduled":n,"valid_answers":valid,"valid_format_rate":valid/n,"answered_accuracy":correct/valid if valid else None,
            "strict_all_row_accuracy":correct/n if classification else None,"normalized_em":correct/n,
            "raw_em":sum(r["raw_exact_match"] for r in rows)/n,"lexical_f1":sum(r["token_f1"] for r in rows)/n if not classification else None,
            "invalid_format_count":counts["invalid_format"],"invalid_format_rate":counts["invalid_format"]/n,
            "abstention_count":counts["abstained"],"abstention_rate":counts["abstained"]/n,
            "technical_failure_count":counts["technical_failure"],"technical_failure_rate":counts["technical_failure"]/n,
            "truncated_count":sum(r["finish_reason"]=="length" for r in rows),
            "truncated_rate":sum(r["finish_reason"]=="length" for r in rows)/n,"clinical_accuracy":None}
    return scored,{"cohorts":scoring.aggregate(scored),"vqa_subgroups":scoring.vqa_subgroups(scored),"strata":strata,
        "scheduled":len(samples),"predictions":len(predictions),"technical_failure_count":sum(r["status"]=="technical_failure" for r in scored),
        "likelihood_auxiliary":{"status":"PENDING_NOT_IMPLEMENTED","generation_validity_affected":False},"human_review":{"performed":0},"clinical_quality_certified":False}


def _execute(config,original,selected,limits,stream):
    from .evaluation.v6_native import load_evaluation_bundle
    import torch
    ref=original["models"][config["cell_id"]];train=_ref(ref["config"]);metrics=_ref(ref["metrics"])
    fatal=None;rows=[]
    try:
        bundle=load_evaluation_bundle(config["model"],config["cell_id"],role=config["role"],expected_contract=metrics["native_contract"],
            final_directory=train["output_dir"],expected_adapter_sha256=ref["adapter_sha256"],fixed_template_date="2026-11-14")
        torch.cuda.reset_peak_memory_stats()
    except Exception as exc:bundle,fatal=None,"MODEL_LOAD_"+type(exc).__name__
    for row,stratum in selected:
        try:result=caps.initial._failure(row,stratum,limits[row["task_type"]],fatal,fatal=True) if fatal else caps.initial._predict_one(bundle,row,stratum,limits[row["task_type"]],config)
        except Exception as exc:result=caps.initial._failure(row,stratum,limits[row["task_type"]],"POSTPROCESS_"+type(exc).__name__,fatal=True)
        rows.append(result);stream.write(json.dumps(result,ensure_ascii=False,sort_keys=True)+'\n');stream.flush()
        if result.get("model_fatal"):fatal=result["reason_code"]
        progress=os.environ.get("MEDVLM_SUPERVISOR_PROGRESS_PATH")
        if progress:atomic_json(progress,{**{k:config[k] for k in IDENTITY_KEYS},"status":"RUNNING","stage":"full_dev_evaluation",
            "rows_completed":len(rows),"scheduled_rows":len(selected),"technical_failures":sum(r["status"]!="SUCCEEDED" for r in rows),"evaluation_completed":None,"updated_at":utcnow()})
    stream.flush();os.fsync(stream.fileno())
    if fatal:raise caps.initial.CalibrationFatal(fatal)
    return rows,{"native_contract":bundle.contract,"peak_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved()}


def _material(config):
    proof,final,original=_cap_proof(config["data"]["cap_freeze"]["receipt"]["path"],config["state_path"])
    _require(_equal(proof,config["data"]["cap_freeze"]),"IMMUTABLE_CAP_FREEZE_PROOF_CHANGED")
    return final,original


def _native_completed(config):
    final,original=_material(config);selected=_rows(original,config["task"]);output=Path(config["output_dir"])
    names=("config.json","evaluation_protocol.json","metrics.json","predictions.private.jsonl","run_manifest.json","scored.private.jsonl")
    marker=_read(output/"completion.json")
    _require(not output.is_symlink() and not any(p.is_symlink() for p in output.iterdir())
        and {p.name for p in output.iterdir()}==set(names)|{"completion.json"}
        and marker.get("schema_version")=="medvlm.v6.full-dev-receipt.v1" and marker.get("status")=="SUCCEEDED" and marker.get("scope")==config["scope"]
        and all(marker.get(k)==config[k] for k in IDENTITY_KEYS) and marker.get("protocol_hash")==config["protocol_hash"]
        and marker.get("test_files_read") is False and marker.get("test_unlocked") is False
        and marker.get("full_dev_completed") is False and marker.get("r2_full_evaluation_completed") is False
        and marker.get("clinical_quality_certified") is False
        and marker.get("evaluation_completed") is True and marker.get("family_selection_completed") is False and marker.get("campaign_terminal") is False
        and marker.get("artifacts")==[_reference(output/name) for name in names]
        and _equal(_read(output/"config.json"),config) and _equal(_read(output/"evaluation_protocol.json"),config["data"]["evaluation_protocol"]),"FULL_DEV_COMPLETION_CHANGED")
    predictions=[json.loads(line) for line in (output/"predictions.private.jsonl").read_text().splitlines()]
    caps.initial._verify_rows(config,original,predictions,selected,final["caps"])
    scored,expected=score_predictions(selected,predictions)
    metrics=_read(output/"metrics.json")
    _require(all(_equal(metrics.get(k),v) for k,v in expected.items())
        and _equal([json.loads(line) for line in (output/"scored.private.jsonl").read_text().splitlines()],scored)
        and metrics.get("protocol_hash")==config["protocol_hash"] and metrics.get("task_counts")==caps.initial._task_counts(predictions)
        and metrics.get("scheduled_count")==len(selected) and marker.get("scheduled_count")==len(selected)
        and marker.get("technical_complete") is all(p["status"]=="SUCCEEDED" for p in predictions),"FULL_DEV_SCORES_OR_DENOMINATORS_CHANGED")
    caps._native_proof(config,original,metrics,selected)
    run=_read(output/"run_manifest.json")
    _require(_equal(run,_run_manifest(config,output)),"FULL_DEV_RUN_MANIFEST_CHANGED")
    return {**marker,"artifacts":marker["artifacts"]+[_reference(output/"completion.json")]}


def _run_manifest(config,output):
    return {"schema_version":"medvlm.v6.full-dev-run.v1","status":"SUCCEEDED","run_id":config["run_id"],"model_id":config["model_id"],"role":config["role"],
        "protocol_hash":config["protocol_hash"],"protocol_version":PROTOCOL_VERSION,"split":"validation","precision":config["precision"],
        "data_hash":config["data_hash"],"artifact_hashes":{name:file_hash(Path(output)/name) for name in ("metrics.json","predictions.private.jsonl","scored.private.jsonl")}}


def _launch_forge_import(config_path,config):
    """Fresh CPU interpreter after GPU accounting closes, without lease FDs."""
    subprocess.run([sys.executable,"-m","mediflow_medvlm.campaign_full_dev","--config",str(config_path),
        "--receipt",config["success_receipt"],"--forge-import-only"],env=dict(os.environ,CUDA_VISIBLE_DEVICES=""),
        check=False,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,close_fds=True)


def _dataset_hash(protocol):
    return canonical_hash({"sample_ids":protocol["sample_ids"],"input_digests":protocol["input_digests"],
        "reference_digest":protocol["reference_digest"],"source_file_hashes":sorted(protocol["files"].values())})


def _forge_import_once(config):
    """Persist attempt intent before the sole supported CPU import call."""
    _require(os.environ.get("CUDA_VISIBLE_DEVICES")=="" and "torch" not in sys.modules,"FORGE_FRESH_CPU_PROCESS_REQUIRED")
    validate_config(config)
    receipt=Path(config["forge"]["receipt_path"]);receipt.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    with receipt.with_suffix(receipt.suffix+'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if receipt.exists():return _read(receipt)
        output=Path(config["output_dir"])
        hashes={name:file_hash(output/name) for name in ("run_manifest.json","metrics.json","completion.json")}
        base={"schema_version":"medvlm.full-dev.forge-import.v1","run_id":config["run_id"],"source_hashes":hashes,"attempted_at":utcnow(),
            "scope":"POST_COMPLETION_CPU_IMPORT_ONLY","scientific_evaluation_status":"SUCCEEDED",
            "config_hash":config["config_hash"],"source_hash":config["source_hash"],"protocol_hash":config["protocol_hash"],
            "cuda_visible_devices":"","torch_imported":False,
            "coverage_note":"native q_type strata are retained in native metrics; legacy Forge exposes supported source/task cohorts and VQA subgroups"}
        atomic_json(receipt,{**base,"status":"STARTED","reason":"one import attempt; interrupted intent is not automatically retried"})
        try:
            _native_completed(config)
            catalog=telemetry_import.load_protocol_catalog(config["forge"]["data_root"])
            protocol=config["data"]["evaluation_protocol"]
            catalog[config["protocol_hash"]]={"split":"validation","protocol_version":PROTOCOL_VERSION,"data_hash":_dataset_hash(protocol),"sha256":file_hash(output/"evaluation_protocol.json")}
            result=telemetry_import.import_directory(output,config["forge"]["output_root"],config["forge"]["campaign_id"],expected_hashes=hashes,protocol_catalog=catalog)
            _require("torch" not in sys.modules,"CPU_IMPORT_UNEXPECTED_TORCH_IMPORT")
            destination=Path(config["forge"]["output_root"])/result["campaign_id"]/result["run_id"]/result["attempt_id"]
            status={**base,"status":"SUCCEEDED","result":result,"finished_at":utcnow(),
                "imported_artifacts":[_reference(destination/name) for name in ("manifest.json","evaluations.json","summary.json","status.json")]}
        except Exception as exc:status={**base,"status":"FAILED","exception_type":type(exc).__name__,"reason":"native evaluation preserved; Forge import failed","finished_at":utcnow()}
        atomic_json(receipt,status);return status


def _import_coverage(config,native_succeeded):
    path=Path(config["forge"]["receipt_path"])
    if not path.exists():return {"status":"NOT_RECORDED","receipt":None}
    value=_read(path);status=value.get("status")
    _require(native_succeeded and value.get("schema_version")=="medvlm.full-dev.forge-import.v1"
        and value.get("run_id")==config["run_id"] and value.get("scope")=="POST_COMPLETION_CPU_IMPORT_ONLY"
        and status in {"STARTED","FAILED","SUCCEEDED"} and value.get("scientific_evaluation_status")=="SUCCEEDED"
        and all(value.get(k)==config[k] for k in ("config_hash","source_hash","protocol_hash"))
        and value.get("cuda_visible_devices")=="" and value.get("torch_imported") is False
        and value.get("source_hashes")=={name:file_hash(Path(config["output_dir"])/name) for name in ("run_manifest.json","metrics.json","completion.json")},
        "FORGE_IMPORT_PROVENANCE_CHANGED")
    if status=="SUCCEEDED":
        result=value.get("result",{});root=Path(config["forge"]["output_root"])
        _require(result.get("campaign_id")==config["forge"]["campaign_id"] and result.get("run_id")==config["run_id"]
            and isinstance(result.get("attempt_id"),str) and result["attempt_id"].startswith("hist-")
            and "/" not in result["attempt_id"] and ".." not in result["attempt_id"],"FORGE_IMPORT_RESULT_CHANGED")
        destination=root/result["campaign_id"]/result["run_id"]/result["attempt_id"]
        _require(value.get("imported_artifacts")==[_reference(destination/name) for name in ("manifest.json","evaluations.json","summary.json","status.json")],"FORGE_IMPORTED_ARTIFACT_CHANGED")
        manifest=_read(destination/"manifest.json");exported_status=_read(destination/"status.json")
        source_hashes={a["artifact_id"]:a["sha256"] for a in manifest.get("source_artifacts",[])}
        _require(all(source_hashes.get(name)==sha for name,sha in value["source_hashes"].items())
            and manifest.get("run_id")==config["run_id"] and manifest.get("campaign_id")==config["forge"]["campaign_id"]
            and manifest.get("data_hash")==_dataset_hash(config["data"]["evaluation_protocol"])
            and manifest.get("eval_protocol")==config["protocol_hash"] and exported_status.get("evaluation_completed") is True
            and exported_status.get("training_succeeded") is None,"FORGE_EXPORTED_METADATA_CHANGED")
    return {"status":status,"receipt":_reference(path)}


def _summary(config):
    final,original=_material(config);manifest,state=caps._active(config["state_path"]);jobs={j["job_id"]:j for j in manifest["jobs"]}
    conditions={};scored_by_role={};records={};imports={}
    for ref,key in zip(config["data"]["roles"],ROLE_IDS):
        child=_ref(ref);job,record=jobs[key],state["jobs"][key];caps._terminal(record)
        actual,actual_ref=caps._child_config(job,"campaign_full_dev")
        _require(actual_ref==ref and _equal(actual,child),"FULL_DEV_REGISTERED_ROLE_CHANGED")
        selected=_rows(original,child["task"]);records[key]=copy.deepcopy(record)
        if record["status"]=="SUCCEEDED":
            receipt=caps.initial.gate._success(job,record)
            _require(_equal(_native_completed(child),receipt),"FULL_DEV_TERMINAL_RESULT_CHANGED")
            metrics=_read(Path(child["output_dir"])/"metrics.json")
            scored=[json.loads(line) for line in (Path(child["output_dir"])/"scored.private.jsonl").read_text().splitlines()]
            condition={"status":"SUCCEEDED","technical_complete":receipt["technical_complete"],"strata":metrics["strata"],"scheduled":len(selected),"receipt":_reference(job["success_receipt"])}
        else:
            failures=[caps.initial._failure(row,stratum,final["caps"][row["task_type"]],"ROLE_JOB_UNAVAILABLE") for row,stratum in selected]
            for failure in failures:failure["attempted"]=False
            scored,metrics=score_predictions(selected,failures)
            condition={"status":record["status"],"technical_complete":False,"strata":metrics["strata"],"scheduled":len(selected),"receipt":None,
                "error":record.get("error"),"error_signature":record.get("error_signature"),"unavailable_count":len(selected)}
        conditions[key]=condition;scored_by_role[key]=scored
        imports[key]=_import_coverage(child,record["status"]=="SUCCEEDED")
    transitions={}
    for cell in CELLS:
        left=scored_by_role[f"v6-full-dev-{cell[0]}-base"];right=scored_by_role[f"v6-full-dev-{cell[0]}-tuned"]
        _require([r["sample_id"] for r in left]==[r["sample_id"] for r in right],"FULL_DEV_PAIRED_IDS_CHANGED")
        grouped={}
        for a,b in zip(left,right):
            key=a["stratum"];grouped.setdefault(key,{"scheduled":0,"correct_transitions":{},"valid_transitions":{}});g=grouped[key];g["scheduled"]+=1
            for name,x,y in (("correct_transitions",bool(a["exact_match"]),bool(b["exact_match"])),("valid_transitions",a["valid"],b["valid"])):
                label=f"{int(x)}->{int(y)}";g[name][label]=g[name].get(label,0)+1
        transitions[cell[0]]={"strata":grouped,"include_unavailable_as_incorrect":True,
            "both_roles_technical_complete":all(conditions[f"v6-full-dev-{cell[0]}-{role}"]["technical_complete"] for role in caps.initial.ROLES)}
    return {"scope":SCOPES["aggregate"],"full_dev_stage_terminal":True,"generation_evaluation_terminal":True,
        "full_dev_completed":False,"r2_full_evaluation_completed":False,"aggregate_audit_succeeded":True,
        "all_roles_succeeded":all(c["status"]=="SUCCEEDED" for c in conditions.values()),
        "all_roles_technical_complete":all(c["technical_complete"] for c in conditions.values()),"conditions":conditions,"terminal_records":records,
        "paired_transitions":transitions,"forge_imports":imports,"forge_import_complete":all(v["status"]=="SUCCEEDED" for v in imports.values()),
        "likelihood_auxiliary":{"status":"PENDING_NOT_IMPLEMENTED","generation_validity_affected":False},
        "separate_speed_protocol":{"status":"PENDING_NOT_EXECUTED","generation_latency_only":True},
        "bootstrap_confidence_intervals":{"status":"PENDING_NOT_COMPUTED"},
        "human_review":{"status":"NOT_PERFORMED","performed":0},
        "family_selection_completed":False,"test_unlocked":False,"test_files_read":False,"clinical_quality_certified":False,"v6_completed":False,"campaign_terminal":False}


def _aggregate_completed(config,summary):
    output=Path(config["output_dir"]);marker=_read(output/"completion.json");names=("config.json","summary.json")
    _require({p.name for p in output.iterdir()}==set(names)|{"completion.json"} and not output.is_symlink() and not any(p.is_symlink() for p in output.iterdir())
        and _equal(_read(output/"config.json"),config) and _equal({k:v for k,v in _read(output/"summary.json").items() if k!="finished_at"},summary)
        and marker.get("schema_version")=="medvlm.v6.full-dev-receipt.v1" and marker.get("status")=="SUCCEEDED" and marker.get("scope")==config["scope"]
        and all(marker.get(k)==config[k] for k in IDENTITY_KEYS) and marker.get("artifacts")==[_reference(output/name) for name in names]
        and marker.get("all_roles_succeeded") is summary["all_roles_succeeded"] and marker.get("full_dev_stage_terminal") is True
        and marker.get("all_roles_technical_complete") is summary["all_roles_technical_complete"]
        and marker.get("generation_evaluation_terminal") is True and marker.get("full_dev_completed") is False
        and marker.get("r2_full_evaluation_completed") is False
        and marker.get("clinical_quality_certified") is False
        and marker.get("test_files_read") is False and marker.get("family_selection_completed") is False
        and marker.get("test_unlocked") is False and marker.get("campaign_terminal") is False,"FULL_DEV_AGGREGATE_CHANGED")
    return {**marker,"artifacts":marker["artifacts"]+[_reference(output/"completion.json")]}


def verify_result(config,*,repo=None):
    validate_config(config,repo=repo)
    return _native_completed(config) if config["kind"]=="role" else _aggregate_completed(config,_summary(config))


def run_job(config_path,receipt_path):
    path,receipt_path=_absolute(config_path),_absolute(receipt_path);config=_read(path);before=file_hash(path)
    validate_config(config);caps._env(config)
    _require(str(receipt_path)==config["success_receipt"],"DEV_RECEIPT_ROUTE_CHANGED")
    role=config["kind"]=="role"
    if role:caps.initial.first.main_train.smoke._inherited_supervisor_fds(config)
    output=Path(config["output_dir"]);output.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    with output.with_name(output.name+'.worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not output.exists():
            _require(not receipt_path.exists(),"DEV_RECEIPT_WITHOUT_OUTPUT")
            staging=Path(tempfile.mkdtemp(prefix=output.name+'.attempt-',dir=output.parent));mask=os.umask(0o077)
            try:
                atomic_json(staging/"config.json",config)
                if role:
                    final,original=_material(config);selected=_rows(original,config["task"])
                    caps.initial._rehash_receipt(config["data"]["trained_cell"]["receipt"])
                    ledger=BudgetLedger(config["gpu_ledger_path"],policy=ResourcePolicy());ledger.start(config["model_id"],config["run_id"],"native_full_validation")
                    try:
                        with (staging/"predictions.private.jsonl").open('x') as stream:predictions,native=_execute(config,original,selected,final["caps"],stream)
                    finally:ledger.finish()
                    caps.initial._verify_rows(config,original,predictions,selected,final["caps"])
                    scored,metrics=score_predictions(selected,predictions)
                    with (staging/"scored.private.jsonl").open('x') as stream:
                        for row in scored:stream.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')
                        stream.flush();os.fsync(stream.fileno())
                    metrics.update(**native,task_counts=caps.initial._task_counts(predictions),scheduled_count=len(selected),protocol_hash=config["protocol_hash"],
                        gpu_accounting={"session_seconds":ledger.session_seconds,"ledger_path":config["gpu_ledger_path"],"snapshot":ledger.snapshot()})
                    caps._native_proof(config,original,metrics,selected)
                    atomic_json(staging/"metrics.json",metrics);atomic_json(staging/"evaluation_protocol.json",config["data"]["evaluation_protocol"])
                    run=_run_manifest(config,staging)
                    atomic_json(staging/"run_manifest.json",run)
                    names=("config.json","evaluation_protocol.json","metrics.json","predictions.private.jsonl","run_manifest.json","scored.private.jsonl")
                    extra={"scheduled_count":len(selected),"technical_complete":all(p["status"]=="SUCCEEDED" for p in predictions),"protocol_hash":config["protocol_hash"],"evaluation_completed":True}
                else:
                    summary=_summary(config);atomic_json(staging/"summary.json",{**summary,"finished_at":utcnow()});names=("config.json","summary.json")
                    extra={"full_dev_stage_terminal":True,"generation_evaluation_terminal":True,"full_dev_completed":False,"r2_full_evaluation_completed":False,"all_roles_succeeded":summary["all_roles_succeeded"],"all_roles_technical_complete":summary["all_roles_technical_complete"]}
                _require(file_hash(path)==before,"DEV_CONFIG_CHANGED_DURING_EXECUTION");validate_config(config)
                marker={"schema_version":"medvlm.v6.full-dev-receipt.v1","status":"SUCCEEDED","scope":config["scope"],**{k:config[k] for k in IDENTITY_KEYS},**extra,
                    "test_files_read":False,"family_selection_completed":False,"test_unlocked":False,"campaign_terminal":False,
                    "full_dev_completed":False,"r2_full_evaluation_completed":False,"clinical_quality_certified":False,
                    "artifacts":[{"path":str(output/name),"sha256":file_hash(staging/name)} for name in names]}
                atomic_json(staging/"completion.json",marker);os.rename(staging,output)
                fd=os.open(output.parent,os.O_DIRECTORY)
                try:os.fsync(fd)
                finally:os.close(fd)
            except BaseException as exc:
                if staging.exists():atomic_json(staging/"failure.private.json",{"status":"FAILED","exception_type":type(exc).__name__,"exception_message":str(exc),"traceback":traceback.format_exc(),"finished_at":utcnow()})
                raise
            finally:os.umask(mask)
        result=verify_result(config)
        _require(file_hash(path)==before and (not receipt_path.exists() or _equal(_read(receipt_path),result)),"DEV_COMPLETED_REUSE_CHANGED")
        if not receipt_path.exists():atomic_json(receipt_path,result)
        if role:
            try:_launch_forge_import(path,config)
            except Exception:pass  # Import coverage cannot invalidate scientific artifacts.
        return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True);parser.add_argument('--receipt',type=Path,required=True);parser.add_argument('--forge-import-only',action='store_true');args=parser.parse_args(argv)
    try:
        if args.forge_import_only:
            result=_forge_import_once(_read(args.config));print(json.dumps({'status':result['status'],'scope':result['scope']}));return 0 if result['status']=='SUCCEEDED' else 1
        result=run_job(args.config,args.receipt)
    except Exception as exc:
        print(json.dumps({"status":"FAILED","exception_type":type(exc).__name__,**({"reason_code":str(exc)} if isinstance(exc,caps.initial.gate.GateError) else {})}));return 1
    print(json.dumps({"status":result["status"],"scope":result["scope"],"evaluation_completed":result.get("evaluation_completed")}));return 0


if __name__=='__main__':raise SystemExit(main())
