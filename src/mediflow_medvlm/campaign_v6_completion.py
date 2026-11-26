"""Finite V6 completion workers: selected fresh seeds, sealed test, honest report.

Planning never launches a worker. Every runtime operation is one canonical
Supervisor job. The first-seed native train kernels run with isolated globals;
the executed historical module and its functions are never monkeypatched.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import contextvars
import functools
import csv
import fcntl
import hashlib
import inspect
import io
import json
import math
import os
from pathlib import Path
import statistics
import time
import traceback
import secrets
import sys
import tempfile
import types

from . import campaign_main_train as train
from . import campaign_full_dev as dev
from . import campaign_bootstrap as bootstrap
from . import campaign_choice_likelihood as choice
from .campaign_supervisor import HASH_FIELDS, atomic_json, canonical_hash, file_hash
from .runtime import BudgetLedger, ResourcePolicy, utcnow

SCHEMA = "medvlm.v6.completion-config.v1"
SEEDS = (20261114, 20261115)
TRAIN_SCOPE = "NATIVE_SELECTED_EXTRA_SEED_ONE_PASS_FINAL_ONLY"
SCOPES = {"train":TRAIN_SCOPE,"additional_audit":"SELECTED_EXTRA_SEEDS_TERMINAL_AUDIT",
          "test_freeze":"V6_SELECTED_FAMILIES_FROZEN_TEST_PROTOCOL",
          "evaluation":"V6_FROZEN_TEST_NATIVE_GENERATION",
          "diagnostic":"V6_FIXED_SUBSET_SPEED_AND_IMAGE_DIAGNOSTIC",
          "likelihood":"V6_FROZEN_TEST_CHOICE_LIKELIHOOD_AUXILIARY",
          "test_audit":"V6_ALL_FROZEN_TEST_CONDITIONS_TERMINAL_AUDIT","report":"V6_TERMINAL_EVIDENCE_REPORT"}
IDS = {"additional_audit":"v6-additional_seeds","test_freeze":"v6-test-freeze","test_audit":"v6-test","report":"v6-report"}
EXPECTED = {"text":4530,"vqa":1512}
TEST_SOURCES = {"text":{"kormedmcqa":3009,"aihub_71875":957,"pubmedqa_labeled":500,"medquad":64},
                "vqa":{"slake":1061,"vqa_rad":451}}
CLONED = ("_progress","_updates","_train","_finite","_metrics_proof","_checkpoint","_verify_checkpoint",
          "_final_selection","_child_identity","train_checkpoint","_invoke","reload_checkpoint",
          "_reload_proof","_verify_output","_completed_output","run_train")
SOURCE_FILES = tuple(dict.fromkeys((*bootstrap.SOURCE_FILES,*choice.SOURCE_FILES,*train.SOURCE_FILES,"campaign_family_selection.py","campaign_v6_completion.py","evaluation/expert_review.py","evaluation/review_v2.py")))
_require, _read, _absolute, _equal = dev._require, dev._read, dev._absolute, dev._equal
_reference, _ref = dev._reference, dev._ref
IDENTITY_KEYS = dev.IDENTITY_KEYS
LIMITATIONS = ["71762 is excluded from V6; its original parent/patient/augmentation lineage remains unknown.",
    "71762 fixed random question-answer pairing and lexical F1 do not establish clinical correctness.",
    "Within-seed item bootstrap intervals do not estimate independent training-seed variation.",
    "Human review not performed remains 0; Jetson deployment is USER_DEFERRED.",
    "Previously observed public benchmarks are not newly blind tests; VQA English is not Korean VQA evidence.",
    "SLAKE dataset-level MedGemma pretraining exposure is documented; exact test exposure is unknown."]


_TRANSACTION=contextvars.ContextVar("v6_completion_read_transaction",default=None)


def _transaction(function):
    @functools.wraps(function)
    def wrapped(*args,**kwargs):
        if _TRANSACTION.get() is not None:return function(*args,**kwargs)
        state={"cache":{},"refs":{}};token=_TRANSACTION.set(state)
        try:
            result=function(*args,**kwargs)
            for path,digest in state["refs"].items():
                _require(file_hash(path)==digest,"COMPLETION_REFERENCE_CHANGED_DURING_AUDIT")
            return result
        finally:_TRANSACTION.reset(token)
    return wrapped


def _memo(function):
    @functools.wraps(function)
    def wrapped(*args,**kwargs):
        state=_TRANSACTION.get()
        if state is None:return function(*args,**kwargs)
        key=(function.__name__,repr(args),repr(sorted(kwargs.items())))
        if key not in state["cache"]:state["cache"][key]=function(*args,**kwargs)
        return copy.deepcopy(state["cache"][key])
    return wrapped


def _ref(reference):
    value=dev._ref(reference);state=_TRANSACTION.get()
    if state is not None:
        before=state["refs"].setdefault(reference["path"],reference["sha256"])
        _require(before==reference["sha256"],"COMPLETION_REFERENCE_CONFLICT_IN_AUDIT")
    return value


def _track_pins(value):
    state=_TRANSACTION.get()
    if state is None:return
    if isinstance(value,dict):
        if {"path","sha256"}<=set(value) and isinstance(value["path"],str) and Path(value["path"]).is_absolute():
            digest=value["sha256"]
            if isinstance(digest,str) and len(digest)==64:
                previous=state["refs"].setdefault(value["path"],digest)
                _require(previous==digest,"COMPLETION_REFERENCE_CONFLICT_IN_AUDIT")
        for item in value.values():_track_pins(item)
    elif isinstance(value,(list,tuple)):
        for item in value:_track_pins(item)


def _track_native(job,config,receipt,state):
    _track_pins([config,receipt,_reference(job["success_receipt"])])
    _track_pins([{"path":str(Path(job["cwd"])/name),"sha256":digest} for name,digest in job["source_files"].items()])
    _track_pins([_reference(config["state_path"]),_reference(state["manifest_path"])])


def source_identity(repo=None):
    package = Path(__file__).resolve().parent if repo is None else Path(repo)/"src/mediflow_medvlm"
    files = {"src/mediflow_medvlm/"+name:file_hash(package/name) for name in SOURCE_FILES}
    return files, canonical_hash(files)


def native_function_contract():
    """Path-independent source hashes; clone code-object identity is checked live."""
    return {name:hashlib.sha256(inspect.getsource(getattr(train,name)).encode()).hexdigest() for name in CLONED}


@_memo
def _registered(config_path, module, verifier):
    ref = _reference(config_path); config = _ref(ref)
    manifest,state = dev.caps._active(config["state_path"])
    job = next(j for j in manifest["jobs"] if j["job_id"] == config["job_id"])
    record = state["jobs"][job["job_id"]]
    _require(record.get("status") == "SUCCEEDED", "COMPLETION_SUCCESSFUL_PREREQUISITE_REQUIRED")
    actual,pin = dev.caps._child_config(job,module)
    _require(pin["sha256"] == ref["sha256"] and _equal(actual,config), "COMPLETION_REGISTERED_CONFIG_CHANGED")
    receipt = dev.caps.initial.gate._success(job,record)
    _require(_equal(verifier(config,repo=job["cwd"]),receipt), "COMPLETION_PREREQUISITE_RESULT_CHANGED")
    _track_native(job,config,receipt,state)
    return config,{"config":pin,"receipt":_reference(job["success_receipt"])},receipt


@_memo
def _selection(path):
    from . import campaign_family_selection as family
    config,proof,receipt = _registered(path,"campaign_family_selection",family.verify_result)
    selected = _read(Path(config["output_dir"])/"selection.json")
    _require(receipt.get("family_selection_completed") is True and receipt.get("extra_seed_targets_frozen") is True
        and receipt.get("test_files_read") is False and receipt.get("test_unlocked") is False,
        "COMPLETION_VALIDATION_ONLY_FROZEN_SELECTION_REQUIRED")
    cells = selected["selected_cells_by_axis"]
    _require(set(cells)=={"text","vqa"} and all(len(v)<=2 and len(v)==len(set(v)) for v in cells.values()), "COMPLETION_MAX_TWO_FAMILIES_PER_AXIS")
    for axis,ids in cells.items():
        specs=[dev.cell_spec(cell) for cell in ids]
        _require(all(s["task"]==axis for s in specs) and len({s["family"] for s in specs})==len(specs), "COMPLETION_DISTINCT_SELECTED_FAMILIES_REQUIRED")
    expected=[{**dev.cell_spec(cell),"seed":seed} for axis in ("text","vqa") for cell in cells[axis] for seed in SEEDS]
    declared=selected["additional_seed_plan"]
    _require(len(declared)<=8 and [{k:r[k] for k in (*dev.cell_spec("T1"),"seed")} for r in declared]==expected,
             "COMPLETION_EXACT_FROZEN_EXTRA_SEED_PLAN_REQUIRED")
    proof["selection"]=_reference(Path(config["output_dir"])/"selection.json")
    initial_ref=config["data"]["initial_protocol"]
    initial=_ref(initial_ref)
    return config,proof,selected,initial


def _destination(root,kind,job_id,output_dir,receipt=None):
    output=_absolute(output_dir);receipt=_absolute(receipt or root/"receipts"/(job_id+".json"))
    allowed=root/"completion"/kind
    for path in (output,receipt):
        dev.caps.initial.gate._private(path,root)
        _require(not path.is_symlink() and not any(p.is_symlink() for p in path.parents),"COMPLETION_SYMLINK_DESTINATION_FORBIDDEN")
    _require(output.is_relative_to(allowed) and output!=allowed and receipt==root/"receipts"/(job_id+".json"),"COMPLETION_DEDICATED_DESTINATION_REQUIRED")
    return output,receipt


def _base(kind,selection_path,output_dir,job_id,*,repo=None,success_receipt=None,data=None,seed=20261023,cell="ALL_V6"):
    selection,proof,chosen,initial=_selection(selection_path)
    root=_absolute(initial["campaign_root"]);output,receipt=_destination(root,kind,job_id,output_dir,success_receipt)
    material={"selection":proof,**(data or {})};files,source=source_identity(repo)
    config={"schema_version":SCHEMA,"scope":SCOPES[kind],"kind":kind,"phase":{"additional_audit":"additional_seeds","test_audit":"test"}.get(kind,kind),"job_id":job_id,"run_id":job_id,
        "cell_id":cell,"seed":seed,"uses_gpu":kind in {"train","evaluation","diagnostic","likelihood"},"device":"cuda" if kind in {"train","evaluation","diagnostic","likelihood"} else "cpu",
        "state_path":selection["state_path"],"output_dir":str(output),"success_receipt":str(receipt),"data":material,
        "resource_policy":ResourcePolicy().as_dict(),"source_files":files,"source_hash":source,
        "data_hash":canonical_hash(material),"model_hash":canonical_hash(initial["models"])}
    config["config_hash"]=canonical_hash(config)
    return config,chosen,initial


def _hash_config(config):
    config["data_hash"]=canonical_hash(config["data"])
    config.pop("config_hash",None);config["config_hash"]=canonical_hash(config)
    return config


def _seed_inputs(template,seed):
    freeze,plan,_=train._freeze(template["data"]["common_freeze"]["path"])
    _,rows,first_records,axis,_=train._inputs(freeze,plan,template["cell_id"])
    ids=axis["sample_ids"];order=axis["training_orders"][str(seed)]
    _require(_equal(order,train.common._orders(ids)[str(seed)]),"COMPLETION_FROZEN_EXTRA_ORDER_CHANGED")
    by_id={r["sample_id"]:r for r in first_records}
    records=[by_id[ids[i]] for i in order["indices"]]
    return rows,records,order


@_transaction
def prepare_train_config(selection_config_path,cell_id,seed,output_dir,*,repo=None,success_receipt=None):
    _require(type(seed) is int and seed in SEEDS,"COMPLETION_ADDITIONAL_SEED_REQUIRED")
    config,selected,initial=_base("train",selection_config_path,output_dir,f"v6-additional-{cell_id}-{seed}",repo=repo,
        success_receipt=success_receipt,seed=seed,cell=cell_id)
    _require(any(r["cell_id"]==cell_id and r["seed"]==seed for r in selected["additional_seed_plan"]),"COMPLETION_UNSELECTED_TRAINING_FORBIDDEN")
    reference=initial["models"][cell_id]["config"];template=_ref(reference)
    _require(template.get("adapter_initialization") is None and template["recipe"]["seed"]==20261023,"COMPLETION_ORIGINAL_BASE_TEMPLATE_REQUIRED")
    rows,records,order=_seed_inputs(template,seed)
    for key in ("model","max_length","fixed_template_date","local_files_only","trust_remote_code","runtime_versions","gpu_ledger_path","supervisor_lock_paths"):
        config[key]=copy.deepcopy(template[key])
    config.update(**dev.cell_spec(cell_id),adapter_initialization=None,model_hash=template["model_hash"])
    config["recipe"]={**copy.deepcopy(template["recipe"]),"seed":seed}
    config["data"].update({k:copy.deepcopy(v) for k,v in template["data"].items()})
    config["data"].update(first_seed_template=reference,sample_order_sha256=order["sample_order_sha256"],indices_sha256=order["indices_sha256"],
        update_example_counts=order["update_example_counts"],probe_sample_id=records[-1]["sample_id"],native_function_contract=native_function_contract())
    return _hash_config(config)


def _training_validate(config):
    validate_config(config)
    for key in IDENTITY_KEYS:
        supplied=os.environ.get("MEDVLM_SUPERVISOR_"+key.upper())
        _require(supplied is None or supplied==config[key],"COMPLETION_SUPERVISOR_IDENTITY_CHANGED")
    return _seed_inputs(_ref(config["data"]["first_seed_template"]),config["seed"])[:2]


def _training_process(stage,config_path,directory,config,invoke=None):
    train._require_parent_without_cuda()
    args=[sys.executable,"-m","mediflow_medvlm.campaign_v6_completion","--config",str(config_path),"--"+stage+"-checkpoint",str(directory),"--receipt",str(directory/(stage+".json"))]
    result=train._invoke(args,config,invoke)
    for name in ("stdout","stderr"):
        data=getattr(result,name,b"") or b"";(directory/(stage+"."+name+".private.log")).write_bytes(data.encode() if isinstance(data,str) else data)
    _require(result.returncode==0 and (directory/(stage+".json")).is_file(),"COMPLETION_"+stage.upper()+"_PROCESS_FAILED")
    value=_read(directory/(stage+".json"));pid=value.get("training_pid") if stage=="train" else value.get("pid")
    identity=value.get("process_identity",{})
    _require(value.get("status")=="SUCCEEDED" and all(value.get(k)==config[k] for k in IDENTITY_KEYS)
        and type(pid) is int and pid>0 and pid!=os.getpid() and identity.get("worker_pid")==pid
        and identity.get("boot_identity") and identity.get("process_start_time"),"COMPLETION_CHILD_IDENTITY_INVALID")
    if getattr(result,"pid",None) is not None:_require(result.pid==pid and result.process_identity==identity,"COMPLETION_ACTUAL_CHILD_IDENTITY_MISMATCH")
    train._require_parent_without_cuda()
    return value,{"pid":pid,"process_identity":identity,"exit_code":result.returncode,"exit_verified":True}


def training_runtime(config):
    """Clone only audited function code objects into an isolated globals table."""
    _require(config["data"]["native_function_contract"]==native_function_contract(),"COMPLETION_NATIVE_FUNCTION_CODE_CHANGED")
    _require(config["source_files"]["src/mediflow_medvlm/campaign_main_train.py"]==file_hash(train.__file__),"COMPLETION_NATIVE_MODULE_SOURCE_CHANGED")
    namespace=dict(vars(train))
    for name in CLONED:
        original=getattr(train,name)
        _require(isinstance(original,types.FunctionType) and original.__module__==train.__name__,"COMPLETION_CLONE_ALLOWLIST_REQUIRED")
        cloned=types.FunctionType(original.__code__,namespace,name,original.__defaults__,original.__closure__)
        cloned.__kwdefaults__=copy.copy(original.__kwdefaults__)
        _require(cloned.__code__ is original.__code__ and cloned.__globals__ is not original.__globals__,"COMPLETION_NATIVE_CLONE_CODE_IDENTITY_REQUIRED")
        namespace[name]=cloned
    namespace.update(SCHEMA=SCHEMA,SCOPE=TRAIN_SCOPE,validate_config=validate_config,_validate=_training_validate,_process=_training_process)
    return types.SimpleNamespace(**{name:namespace[name] for name in CLONED})


@_transaction
def prepare_additional_audit_config(selection_config_path,train_config_paths,output_dir,*,repo=None,success_receipt=None):
    refs=[_reference(path) for path in train_config_paths];configs=[_ref(ref) for ref in refs]
    config,selected,_=_base("additional_audit",selection_config_path,output_dir,IDS["additional_audit"],repo=repo,success_receipt=success_receipt,data={"train_configs":refs})
    _require([(c["cell_id"],c["seed"]) for c in configs]==[(r["cell_id"],r["seed"]) for r in selected["additional_seed_plan"]],"COMPLETION_EXACT_SELECTED_TRAIN_CONFIGS_REQUIRED")
    for child in configs:validate_config(child,repo=repo)
    return config


@_memo
def _terminal(config,reference):
    child=_ref(reference);manifest,state=dev.caps._active(config["state_path"])
    job=next(j for j in manifest["jobs"] if j["job_id"]==child["job_id"]);record=state["jobs"][job["job_id"]]
    dev.caps._terminal(record);actual,pin=dev.caps._child_config(job,"campaign_v6_completion")
    _require(pin==reference and _equal(actual,child),"COMPLETION_TERMINAL_CONFIG_BINDING_CHANGED")
    if record["status"]=="SUCCEEDED":
        result=verify_result(child,repo=job["cwd"])
        _require(_equal(result,dev.caps.initial.gate._success(job,record)),"COMPLETION_TERMINAL_RESULT_CHANGED")
    else:result=None
    if result:_track_native(job,child,result,state)
    else:_track_pins([reference,_reference(config["state_path"]),_reference(state["manifest_path"])])
    return child,{"status":record["status"],"state_sha256":canonical_hash(record),"receipt":_reference(job["success_receipt"]) if result else None,
        "error":record.get("error"),"error_signature":record.get("error_signature"),
        "attempts":record["attempts"],"failure_count":record["failure_count"],"recovery_attempts":record["recovery_attempts"]},result


def _additional_summary(config):
    conditions={}
    for ref in config["data"]["train_configs"]:
        child,condition,_=_terminal(config,ref);conditions[child["job_id"]]=condition
    return {"additional_seeds_terminal":True,"planned":len(conditions),"conditions":conditions,
        "replacement_after_failure_allowed":False,"test_inference_performed":False}


# Historical selection metadata is immutable and supplies ONLY the original 64 IDs.
MEDQUAD_SELECTION_SHA256="aa9faa47f3ee09ae4682222bde26c8290bb6294634da6aa6d5792bfd3c4a4ba0"
MEDQUAD_IDS_SHA256="53d02c7690f7fff7b05a4d2dec66ccca2669a679915212662b727aa35d56423d"
DIAGNOSTIC_PROCEDURE={"version":"v6.fixed-diagnostic.v1","seed":20261023,"subset_per_task":32,
    "subset_order":"canonical_hash([seed,sample_id]) ascending, sample_id tie; fixed before outcomes",
    "warmup_requests":2,"measured_repeats":3,"batch_size":1,"same_gpu_required":True,
    "timing":"CUDA synchronize immediately before and after whole native encode/generate/decode; monotonic wall time",
    "memory":"torch.cuda.reset_peak_memory_stats before each measured request; max_memory_allocated/reserved after synchronized request",
    "ttft":None,"decode_only_latency":None,"ttft_unavailable_reason":"NOT_INSTRUMENTED",
    "image_ablation":"same 32 VQA inputs with image/no_image; native evaluation encoder; unsupported native path recorded separately"}


def _test_lineage(selection_path):
    _,_,_,initial=_selection(selection_path)  # This gate precedes every test metadata/row read.
    template=_ref(initial["models"][next(iter(initial["models"]))]["config"])
    freeze,_,_=train._freeze(template["data"]["common_freeze"]["path"])
    protected=_ref(freeze["protection_references"]["protected_freeze"])
    mixed=_ref(freeze["protection_references"]["text_freeze"])
    ancestor_ref=freeze["protection_references"]["text_ancestor_freeze"];ancestor=_ref(ancestor_ref)
    selection_ref=_reference(Path(ancestor_ref["path"]).parent/"evaluation/text-test-v1.json")
    _require(selection_ref["sha256"]==MEDQUAD_SELECTION_SHA256,"COMPLETION_ORIGINAL_MEDQUAD_SELECTION_CHANGED")
    historic=_ref(selection_ref);ids=[sid for sid in historic["sample_ids"] if sid.startswith("medquad/")]
    _require(len(ids)==64 and len(set(ids))==64 and canonical_hash(ids)==MEDQUAD_IDS_SHA256,
        "COMPLETION_ORIGINAL_MEDQUAD_64_REQUIRED")
    return {"kormedmcqa":protected["splits"]["kormedmcqa/test"],"text":mixed["splits"]["mix_v2/test"],
        "pubmedqa":ancestor["splits"]["pubmedqa_labeled/test"],
        "medquad":{**ancestor["splits"]["medquad/test"],"selected_ids":ids,"selection_metadata":selection_ref},
        "slake":protected["splits"]["slake/test"],"vqa_rad":protected["splits"]["vqa_rad/test"]}


def _test_stratum(row):
    if row["task_type"]=="medical_vqa":
        answer_type=str(row.get("metadata",{}).get("answer_type","")).casefold()
        _require(answer_type in {"open","closed"},"COMPLETION_VQA_ANSWER_TYPE_REQUIRED")
        return row["source_id"]+"_"+answer_type
    return dev.caps.initial._stratum(row)


@_memo
def _test_rows(entries):
    axes={"text":[],"vqa":[]};refs={}
    for key in ("kormedmcqa","text","pubmedqa","medquad","slake","vqa_rad"):
        entry=entries[key];path=_absolute(entry.get("snapshot_path",entry.get("path")))
        _require(not path.is_symlink() and not any(p.is_symlink() for p in path.parents) and path.name=="test.jsonl"
            and file_hash(path)==entry["sha256"],"COMPLETION_FULL_TEST_SOURCE_CHANGED")
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        _require(len(rows)==entry["count"] and all(r.get("split")=="test" for r in rows),"COMPLETION_TEST_PARTITION_REQUIRED")
        if key=="medquad":
            metadata=_ref(entry["selection_metadata"]);ids=entry["selected_ids"]
            _require(ids==[sid for sid in metadata["sample_ids"] if sid.startswith("medquad/")] and len(ids)==64
                and len(set(ids))==64,"COMPLETION_MEDQUAD_SELECTION_IDS_CHANGED")
            by_id={r["sample_id"]:r for r in rows}
            _require(set(ids)<=set(by_id),"COMPLETION_ORIGINAL_MEDQUAD_IDS_MISSING")
            rows=[by_id[sid] for sid in ids]
        axes["vqa" if key in {"slake","vqa_rad"} else "text"].extend(rows);refs[key]=_reference(path)
    selected={}
    for axis,rows in axes.items():
        _require(len(rows)==EXPECTED[axis] and len({r["sample_id"] for r in rows})==len(rows)
            and dict(Counter(r["source_id"] for r in rows))==TEST_SOURCES[axis],"COMPLETION_FULL_TEST_POPULATION_REQUIRED")
        strata=Counter(_test_stratum(r) for r in rows)
        if axis=="text":_require(all(strata[k]==v for k,v in {"71875_mcq":777,"71875_short":83,"71875_long":97}.items()),"COMPLETION_71875_TEST_STRATA_REQUIRED")
        selected[axis]=[(r,_test_stratum(r)) for r in rows]
    return selected,refs


def _diagnostic_ids(selected):
    quotas={"text":{"kormedmcqa":8,"71875_mcq":4,"71875_short":4,"71875_long":4,"pubmedqa_labeled":8,"medquad":4},
            "vqa":{"slake_open":8,"slake_closed":8,"vqa_rad_open":8,"vqa_rad_closed":8}}
    result={}
    for axis,rows in selected.items():
        ids=[]
        for stratum,count in quotas[axis].items():
            members=sorted([row for row,s in rows if s==stratum],key=lambda row:(canonical_hash([20261023,row["sample_id"]]),row["sample_id"]))
            _require(len(members)>=count,"COMPLETION_DIAGNOSTIC_STRATUM_QUOTA_UNAVAILABLE")
            ids.extend(row["sample_id"] for row in members[:count])
        result[axis]=ids
    return result


@_transaction
def prepare_test_freeze_config(selection_config_path,additional_audit_config_path,output_dir,*,repo=None,success_receipt=None):
    additional,proof,_=_registered(additional_audit_config_path,"campaign_v6_completion",verify_result)
    _,selection_proof,_,_=_selection(selection_config_path)
    _require(additional["kind"]=="additional_audit" and additional["data"]["selection"]==selection_proof,"COMPLETION_AUDIT_SELECTION_CHANGED")
    audit=_read(Path(_ref(proof["config"])["output_dir"])/"summary.json")
    _require(audit.get("additional_seeds_terminal") is True,"COMPLETION_EXTRA_SEED_TERMINAL_AUDIT_REQUIRED")
    entries=_test_lineage(selection_config_path);selected,refs=_test_rows(entries)
    data={"additional_audit":proof,"test_entries":entries,"test_sources":refs,
        "ordered_test_ids":{axis:[r["sample_id"] for r,_ in rows] for axis,rows in selected.items()},
        "counts":{axis:len(rows) for axis,rows in selected.items()},"diagnostic_ids":_diagnostic_ids(selected)}
    return _base("test_freeze",selection_config_path,output_dir,IDS["test_freeze"],repo=repo,success_receipt=success_receipt,data=data)[0]


def _test_protocol(config):
    _,_,_,initial=_selection(config["data"]["selection"]["config"]["path"])
    final_ref=initial.get("final_protocol")
    selection_config=_ref(config["data"]["selection"]["config"])
    # Family selection pins the existing final cap reference directly.
    final_ref=selection_config["data"]["final_protocol"] if final_ref is None else final_ref
    final=_ref(final_ref);selected,_=_test_rows(config["data"]["test_entries"])
    value={"schema_version":"medvlm.v6.test-protocol.v1","split":"test","selection":config["data"]["selection"],
        "additional_audit":config["data"]["additional_audit"],"cap_protocol":final_ref,"caps":final["caps"],"eval_context":4096,
        "decode":final["decode"],"fixed_template_date":"2026-11-14","test_sources":config["data"]["test_sources"],
        "ordered_test_ids":config["data"]["ordered_test_ids"],
        "input_digests":{axis:{r["sample_id"]:canonical_hash(r["input"]) for r,_ in rows} for axis,rows in selected.items()},
        "reference_digest":{axis:canonical_hash([(r["sample_id"],r["supervision"]) for r,_ in rows]) for axis,rows in selected.items()},
        "diagnostic_ids":config["data"]["diagnostic_ids"],"diagnostic_procedure":copy.deepcopy(DIAGNOSTIC_PROCEDURE),
        "bootstrap_procedure":copy.deepcopy(bootstrap.PROCEDURE),"choice_likelihood_procedure":copy.deepcopy(choice.PROCEDURE),
        "all_planned_failures_in_denominator":True,
        "parser_and_scorer_source":{k:v for k,v in config["source_files"].items() if k.endswith(("evaluation/metrics.py","evaluation/v2.py"))},
        "already_observed_public_benchmarks_not_new_blind_test":True,"post_test_reselection_allowed":False}
    value["protocol_hash"]=canonical_hash(value)
    return value


def _model_ref(selection_path,cell_id,seed,train_config_path=None):
    _,_,_,initial=_selection(selection_path)
    if seed==20261023:
        _require(train_config_path is None,"COMPLETION_FIRST_SEED_MODEL_REPLACEMENT_FORBIDDEN")
        return initial["models"][cell_id]
    _require(train_config_path is not None,"COMPLETION_SELECTED_SEED_TRAIN_PROOF_REQUIRED")
    config,proof,_=_registered(train_config_path,"campaign_v6_completion",verify_result)
    _require(config["kind"]=="train" and config["cell_id"]==cell_id and config["seed"]==seed,"COMPLETION_TRAIN_SEED_IDENTITY_CHANGED")
    output=Path(config["output_dir"]);metrics=_read(output/"metrics.json")
    return {"config":proof["config"],"receipt":proof["receipt"],"checkpoint":_reference(output/"checkpoint.json"),
        "metrics":_reference(output/"metrics.json"),"reload":_reference(output/"reload.json"),"final_selection":_reference(output/"final_selection.json"),
        "adapter_sha256":metrics["adapter_sha256"],"model_hash":config["model_hash"],"revision":config["model"]["revision"]}


def _prepare_native_config(kind,test_freeze_config_path,cell_id,role,training_seed,output_dir,*,train_config_path=None,repo=None,success_receipt=None):
    freeze,proof,_=_registered(test_freeze_config_path,"campaign_v6_completion",verify_result)
    _require(freeze["kind"]=="test_freeze" and role in {"base","tuned"} and training_seed in (20261023,*SEEDS) and (training_seed==20261023 or role=="tuned"),"COMPLETION_TEST_CONDITION_REQUIRED")
    selection_path=freeze["data"]["selection"]["config"]["path"]
    reference=_model_ref(selection_path,cell_id,training_seed,train_config_path)
    native=_ref(reference["config"]);protocol_ref=_reference(Path(freeze["output_dir"])/"protocol.json");protocol=_ref(protocol_ref)
    axis=dev.cell_spec(cell_id)["task"]
    data={"test_freeze":proof,"protocol":protocol_ref,"model_ref":reference,"training_config":_reference(train_config_path) if train_config_path else None,
        "ordered_sample_ids":protocol["ordered_test_ids"][axis],"image_root":native["data"]["image_root"]}
    config,_,_=_base(kind,selection_path,output_dir,f"v6-test-{kind+'-' if kind!='evaluation' else ''}{cell_id}-{role}-{training_seed}",repo=repo,success_receipt=success_receipt,data=data,seed=training_seed,cell=cell_id)
    config.update(**dev.cell_spec(cell_id),local_files_only=True,trust_remote_code=False,precision="nf4_double_quant_bfloat16",role=role,model=native["model"],model_hash=canonical_hash({"model":native["model"],"role":role,"checkpoint":reference["checkpoint"] if role=="tuned" else None}),
        eval_context=4096,split="test",decode=protocol["decode"],fixed_template_date="2026-11-14",gpu_ledger_path=native["gpu_ledger_path"],
        supervisor_lock_paths=native["supervisor_lock_paths"],protocol_hash=protocol["protocol_hash"])
    return _hash_config(config)


@_transaction
def prepare_evaluation_config(test_freeze_config_path,cell_id,role,training_seed,output_dir,*,train_config_path=None,repo=None,success_receipt=None):
    return _prepare_native_config("evaluation",test_freeze_config_path,cell_id,role,training_seed,output_dir,
        train_config_path=train_config_path,repo=repo,success_receipt=success_receipt)


@_transaction
def prepare_diagnostic_config(test_freeze_config_path,cell_id,role,output_dir,*,repo=None,success_receipt=None):
    config=_prepare_native_config("diagnostic",test_freeze_config_path,cell_id,role,20261023,output_dir,repo=repo,success_receipt=success_receipt)
    protocol=_ref(config["data"]["protocol"])
    config["data"].update(diagnostic_ids=protocol["diagnostic_ids"][config["task"]],procedure=copy.deepcopy(DIAGNOSTIC_PROCEDURE))
    return _hash_config(config)


@_transaction
def prepare_likelihood_config(test_freeze_config_path,cell_id,role,training_seed,output_dir,*,train_config_path=None,repo=None,success_receipt=None):
    _require(dev.cell_spec(cell_id)["task"]=="text","COMPLETION_LIKELIHOOD_TEXT_ONLY")
    config=_prepare_native_config("likelihood",test_freeze_config_path,cell_id,role,training_seed,output_dir,
        train_config_path=train_config_path,repo=repo,success_receipt=success_receipt)
    config["data"].update(procedure=copy.deepcopy(choice.PROCEDURE),native={"trained_cell":config["data"]["model_ref"]})
    return _hash_config(config)


def _test_matrix(chosen,audit):
    conditions={(cell[0],role,20261023) for cell in dev.CELLS for role in ("base","tuned")}
    for record in chosen["additional_seed_plan"]:
        if audit["conditions"][f"v6-additional-{record['cell_id']}-{record['seed']}"]["status"]=="SUCCEEDED":
            conditions.add((record["cell_id"],"tuned",record["seed"]))
    return conditions


@_transaction
def prepare_test_audit_config(selection_config_path,test_freeze_config_path,evaluation_config_paths,output_dir,*,
                          diagnostic_config_paths=(),likelihood_config_paths=(),repo=None,success_receipt=None):
    freeze,proof,_=_registered(test_freeze_config_path,"campaign_v6_completion",verify_result)
    _,selection_proof,chosen,_=_selection(selection_config_path)
    _require(freeze["kind"]=="test_freeze" and freeze["data"]["selection"]==selection_proof,"COMPLETION_REPORT_SELECTION_CHANGED")
    audit=_read(Path(_ref(freeze["data"]["additional_audit"]["config"])["output_dir"])/"summary.json")
    matrix=_test_matrix(chosen,audit)
    expected={"evaluation":matrix,"diagnostic":{(c[0],r,20261023) for c in dev.CELLS for r in ("base","tuned")},
              "likelihood":{item for item in matrix if dev.cell_spec(item[0])["task"]=="text"}}
    data={"test_freeze":proof}
    for kind,paths in (("evaluation",evaluation_config_paths),("diagnostic",diagnostic_config_paths),("likelihood",likelihood_config_paths)):
        refs=[_reference(path) for path in paths];configs=[_ref(ref) for ref in refs]
        _require({(c["cell_id"],c["role"],c["seed"]) for c in configs}==expected[kind] and len(configs)==len(expected[kind]),"COMPLETION_EXACT_"+kind.upper()+"_MATRIX_REQUIRED")
        for child in configs:
            _require(child["kind"]==kind and child["data"]["test_freeze"]==proof,"COMPLETION_REPORT_COMMON_TEST_PROTOCOL_REQUIRED")
            validate_config(child,repo=repo)
        data[{"evaluation":"evaluations","diagnostic":"diagnostics","likelihood":"likelihoods"}[kind]]=refs
    return _base("test_audit",selection_config_path,output_dir,IDS["test_audit"],repo=repo,success_receipt=success_receipt,data=data)[0]


@_transaction
def prepare_report_config(test_audit_config_path,output_dir,*,evidence=None,repo=None,success_receipt=None):
    audit,proof,_=_registered(test_audit_config_path,"campaign_v6_completion",verify_result)
    _require(audit["kind"]=="test_audit","COMPLETION_TEST_AUDIT_REQUIRED")
    data={**copy.deepcopy(audit["data"]),"test_audit":proof,"evidence":{k:_reference(v) for k,v in (evidence or {}).items()}}
    data.pop("selection")
    return _base("report",audit["data"]["selection"]["config"]["path"],output_dir,IDS["report"],repo=repo,success_receipt=success_receipt,data=data)[0]


def _test_audit_summary(config):
    conditions={}
    for kind in ("evaluations","diagnostics","likelihoods"):
        conditions[kind]={}
        for ref in config["data"][kind]:
            child,condition,_=_terminal(config,ref);conditions[kind][child["job_id"]]=condition
    return {"test_conditions_terminal":True,"conditions":conditions,"post_test_reselection_allowed":False,
            "v6_completed":False,"test_files_read":True}


@_transaction
def validate_config(config,*,repo=None):
    _require(config.get("schema_version")==SCHEMA and config.get("config_hash")==canonical_hash({k:v for k,v in config.items() if k!="config_hash"})
        and (config.get("source_files"),config.get("source_hash"))==source_identity(repo),"COMPLETION_CONFIG_HASH_OR_SOURCE_CHANGED")
    kind=config["kind"];selection=config["data"]["selection"]["config"]["path"]
    common={"repo":repo,"success_receipt":config["success_receipt"]}
    if kind=="train":expected=prepare_train_config(selection,config["cell_id"],config["seed"],config["output_dir"],**common)
    elif kind=="additional_audit":expected=prepare_additional_audit_config(selection,[r["path"] for r in config["data"]["train_configs"]],config["output_dir"],**common)
    elif kind=="test_freeze":expected=prepare_test_freeze_config(selection,config["data"]["additional_audit"]["config"]["path"],config["output_dir"],**common)
    elif kind=="diagnostic":expected=prepare_diagnostic_config(config["data"]["test_freeze"]["config"]["path"],config["cell_id"],config["role"],config["output_dir"],**common)
    elif kind in {"evaluation","likelihood"}:expected=(prepare_evaluation_config if kind=="evaluation" else prepare_likelihood_config)(config["data"]["test_freeze"]["config"]["path"],config["cell_id"],config["role"],config["seed"],config["output_dir"],
        train_config_path=config["data"]["training_config"]["path"] if config["data"]["training_config"] else None,**common)
    elif kind=="test_audit":expected=prepare_test_audit_config(selection,config["data"]["test_freeze"]["config"]["path"],[r["path"] for r in config["data"]["evaluations"]],config["output_dir"],
        diagnostic_config_paths=[r["path"] for r in config["data"]["diagnostics"]],likelihood_config_paths=[r["path"] for r in config["data"]["likelihoods"]],**common)
    else:
        _require(kind=="report","COMPLETION_KIND_REQUIRED")
        expected=prepare_report_config(config["data"]["test_audit"]["config"]["path"],config["output_dir"],
            evidence={k:v["path"] for k,v in config["data"]["evidence"].items()},**common)
    _require(_equal(config,expected),"COMPLETION_SCIENTIFIC_IDENTITY_CHANGED")
    return config


def job_from_config(config_path,*,repo=None,python=None):
    path=_absolute(config_path);config=_read(path);validate_config(config,repo=repo);kind=config["kind"]
    deps=[_ref(config["data"]["selection"]["config"])["job_id"]]
    if kind=="additional_audit":deps+=[_ref(r)["job_id"] for r in config["data"]["train_configs"]]
    elif kind in {"test_freeze","evaluation","diagnostic","likelihood","test_audit","report"}:
        key="additional_audit" if kind=="test_freeze" else "test_freeze";deps.append(_ref(config["data"][key]["config"])["job_id"])
        if kind in {"evaluation","likelihood"} and config["data"]["training_config"]:deps.append(_ref(config["data"]["training_config"])["job_id"])
        if kind=="report":deps.append(_ref(config["data"]["test_audit"]["config"])["job_id"])
        if kind=="test_audit":
            freeze=_ref(config["data"]["test_freeze"]["config"]);deps.append(_ref(freeze["data"]["additional_audit"]["config"])["job_id"])
            deps += [_ref(r)["job_id"] for key in ("evaluations","diagnostics","likelihoods") for r in config["data"][key]]
    return {**{k:config[k] for k in (*IDENTITY_KEYS,"cell_id","phase","kind","seed","uses_gpu","source_files","success_receipt")},
        **{k:config[k] for k in ("model_id","family","task","role") if k in config},
        "requires_gpu":config["uses_gpu"],"gpu_lock_mode":"wrapper","dependencies":deps,"dependency_policy":"terminal" if kind in {"additional_audit","test_audit"} else "success",
        "implementation_pending":False,"cwd":str(Path(repo).resolve() if repo else Path(__file__).resolve().parents[2]),
        "argv":[str(python or sys.executable),"-m","mediflow_medvlm.campaign_v6_completion","--config",str(path),"--receipt",config["success_receipt"]]}


def _evaluation(config,stream):
    ref=config["data"]["model_ref"];protocol=_ref(config["data"]["protocol"])
    freeze=_ref(config["data"]["test_freeze"]["config"]);selected,_=_test_rows(freeze["data"]["test_entries"]);selected=selected[config["task"]]
    original={"models":{config["cell_id"]:ref}}
    # Exact native full-dev decoder, caps and parser; test IDs are supplied explicitly.
    predictions,contract=dev._execute(config,original,selected,protocol["caps"],stream)
    dev.caps.initial._verify_rows(config,{"eval_context":4096},predictions,selected,protocol["caps"])
    scored,summary=dev.score_predictions(selected,predictions)
    return scored,summary,contract,selected


def _no_image(bundle,row,stratum,cap,config):
    """The existing native no_image encoder, never a blank image substitute."""
    import torch
    from .evaluation.v2 import encode_evaluation_sample,generation_finish
    from .training.v6_native import verify_native_no_thinking_output
    result=dev.caps.initial._failure(row,stratum,cap,"NO_IMAGE_UNINITIALIZED")
    try:
        encoded,metadata=encode_evaluation_sample(row,bundle.processor,bundle.architecture,
            max_length=config["eval_context"]-cap,image_root=config["data"]["image_root"],image_condition="no_image")
        _require(not any(any(w in k.casefold() for w in ("pixel","image","vision","video")) for k in encoded),"NO_IMAGE_TENSORS_FORBIDDEN")
    except Exception as exc:
        unsupported=isinstance(exc,(ValueError,NotImplementedError)) and any(token in str(exc).casefold() for token in ("image is required", "image inputs are required", "requires an image", "at least one image"))
        return {**result,"status":"UNSUPPORTED_NATIVE_NO_IMAGE" if unsupported else "TECHNICAL_FAILURE","reason_code":type(exc).__name__+":"+str(exc)}
    ids=encoded["input_ids"][0].tolist();moved={k:v.to(bundle.model.device) if hasattr(v,"to") else v for k,v in encoded.items()}
    tokenizer=getattr(bundle.processor,"tokenizer",bundle.processor);eos=getattr(bundle.model.generation_config,"eos_token_id",None)
    if eos is None:eos=tokenizer.eos_token_id
    pad=getattr(bundle.model.generation_config,"pad_token_id",None)
    if pad is None:pad=getattr(tokenizer,"pad_token_id",None)
    guard=dev.caps.initial.GenerationScoreGuard()
    try:
        torch.cuda.synchronize()
        with torch.inference_mode(),torch.autocast(device_type="cuda",dtype=torch.bfloat16):
            generated=bundle.model.generate(**moved,max_new_tokens=cap,do_sample=False,num_beams=1,use_cache=True,
                eos_token_id=eos,pad_token_id=pad,max_time=None,remove_invalid_values=False,logits_processor=[guard])
        torch.cuda.synchronize();continuation=generated[0,len(ids):];tokens=continuation.tolist()
        _require(tokens and len(tokens)<=cap and generated[0,:len(ids)].tolist()==ids
            and guard.checked_steps==len(tokens),"NO_IMAGE_NATIVE_GENERATION_INVALID")
    except ValueError as exc:
        unsupported=any(token in str(exc).casefold() for token in ("image is required","image inputs are required","requires an image","at least one image"))
        if not unsupported:raise
        return {**result,"status":"UNSUPPORTED_NATIVE_NO_IMAGE","reason_code":"NATIVE_GENERATION_"+str(exc)}
    # CUDA/OOM/other runtime failures propagate to the canonical retry/failure gate.
    raw=tokenizer.decode(continuation,skip_special_tokens=False);text=tokenizer.decode(continuation,skip_special_tokens=True)
    try:thinking=verify_native_no_thinking_output(raw);status="SUCCEEDED";reason=None
    except ValueError:thinking={"explicit_nonempty_thinking_detected":True};status="TECHNICAL_FAILURE";reason="NATIVE_THINKING_OUTPUT_PRESENT"
    return {**result,"status":status,"reason_code":reason,"model_fatal":False,"input_tokens":len(ids),"input_metadata":metadata,
        "generated_tokens":len(tokens),"generated_ids":tokens,"generated_ids_sha256":canonical_hash(tokens),"generated_text":text,
        "generated_raw_text":raw,"eos_token_ids":eos,"input_tensor_shapes":{k:list(v.shape) for k,v in encoded.items() if hasattr(v,"shape")},
        "input_tensor_dtypes":{k:str(v.dtype) for k,v in encoded.items() if hasattr(v,"dtype")},"finish_reason":generation_finish(len(tokens),tokens[-1],cap,eos),"no_thinking_output":thinking,
        "generation_scores_valid":True,"raw_forward_logits_finite_verified":False,"generation_scores_checked_steps":guard.checked_steps,"pixel_tensor_keys":[]}


def _diagnostic_summary(config,rows):
    ids=config["data"]["diagnostic_ids"];procedure=config["data"]["procedure"];measured=[r for r in rows if r["stage"]=="measured"]
    expected=[(repeat,sid) for repeat in range(procedure["measured_repeats"]) for sid in ids]
    _require([(r["repeat"],r["prediction"]["sample_id"]) for r in measured]==expected,"COMPLETION_DIAGNOSTIC_MEASURED_MATRIX_CHANGED")
    warmup=[r for r in rows if r["stage"]=="warmup"]
    _require([r["prediction"]["sample_id"] for r in warmup]==ids[:procedure["warmup_requests"]],"COMPLETION_DIAGNOSTIC_WARMUP_CHANGED")
    ablation=[r for r in rows if r["stage"]=="no_image"]
    _require([r["prediction"]["sample_id"] for r in ablation]==(ids if config["task"]=="vqa" else []),"COMPLETION_DIAGNOSTIC_IMAGE_MATRIX_CHANGED")
    _require(len(rows)==len(measured)+len(warmup)+len(ablation),"COMPLETION_DIAGNOSTIC_UNKNOWN_STAGE")
    durations=[r["synchronized_e2e_seconds"] for r in measured]
    _require(all(type(v) in (float,int) and math.isfinite(v) and v>=0 for v in durations),"COMPLETION_DIAGNOSTIC_FINITE_TIMES_REQUIRED")
    selected={r["sample_id"]:(r,s) for r,s in _rows_for(config)[config["task"]]}
    baseline={r["prediction"]["sample_id"]:r["prediction"] for r in measured if r["repeat"]==0}
    scores=None
    if ablation:
        supported=[r for r in ablation if r["prediction"]["status"]!="UNSUPPORTED_NATIVE_NO_IMAGE"]
        # Report unavailable cases explicitly; do not treat an unsupported path as a quality score.
        scores={"scheduled":32,"supported":len(supported),"unsupported":32-len(supported),
            "technical_failures":sum(r["prediction"]["status"]=="TECHNICAL_FAILURE" for r in ablation),
            "unsupported_reasons":dict(Counter(r["prediction"]["reason_code"] for r in ablation if r not in supported)),
            "paired_cases":[{"sample_id":item["prediction"]["sample_id"],"image":dev.score_predictions([selected[item["prediction"]["sample_id"]]],[baseline[item["prediction"]["sample_id"]]])[0][0],
                "no_image":dev.score_predictions([selected[item["prediction"]["sample_id"]]],[item["prediction"]])[0][0]} for item in supported],
            "clinical_correctness_verified":False}
    return {"procedure":procedure,"scheduled_measured_requests":len(expected),"successful_requests":sum(r["prediction"]["status"]=="SUCCEEDED" for r in measured),
        "technical_failures":sum(r["prediction"]["status"]!="SUCCEEDED" for r in measured),"latency_p50_seconds":bootstrap._quantile(durations,.5),
        "latency_p95_seconds":bootstrap._quantile(durations,.95),"sample_count":len(durations),"by_task":{task:{"scheduled":len(items),"p50_seconds":bootstrap._quantile([r["synchronized_e2e_seconds"] for r in items],.5),
        "p95_seconds":bootstrap._quantile([r["synchronized_e2e_seconds"] for r in items],.95)} for task in sorted({r["prediction"]["task_type"] for r in measured})
        for items in [[r for r in measured if r["prediction"]["task_type"]==task]]},"ttft_seconds":None,"decode_only_seconds":None,
        "input_native_tokens":sum(r["prediction"].get("input_tokens",0) for r in measured),"output_native_tokens":sum(r["prediction"]["generated_tokens"] for r in measured),
        "output_characters":sum(len(r["prediction"].get("generated_text","")) for r in measured),
        "output_utf8_bytes":sum(len(r["prediction"].get("generated_text","").encode()) for r in measured),
        "input_serialized_utf8_bytes":sum(len(json.dumps(selected[r["prediction"]["sample_id"]][0]["input"],ensure_ascii=False).encode()) for r in measured),
        "input_characters":sum(len(json.dumps(selected[r["prediction"]["sample_id"]][0]["input"],ensure_ascii=False)) for r in measured),
        "eos_stops":sum(r["prediction"]["finish_reason"]=="stop" for r in measured),"truncated":sum(r["prediction"]["finish_reason"]=="length" for r in measured),
        "peak_allocated_bytes":max(r["peak_allocated_bytes"] for r in measured),"peak_reserved_bytes":max(r["peak_reserved_bytes"] for r in measured),
        "image_ablation_32":scores,"whole_generation_tokens_per_second_is_not_decode_only":True}


def _diagnostic(config,stream):
    import torch
    from .evaluation.v6_native import load_evaluation_bundle
    ref=config["data"]["model_ref"];native=_ref(ref["config"]);metrics=_ref(ref["metrics"])
    bundle=load_evaluation_bundle(config["model"],config["cell_id"],role=config["role"],expected_contract=metrics["native_contract"],
        final_directory=native["output_dir"],expected_adapter_sha256=ref["adapter_sha256"],fixed_template_date="2026-11-14")
    selected={row["sample_id"]:(row,stratum) for row,stratum in _rows_for(config)[config["task"]]}
    ids=config["data"]["diagnostic_ids"];protocol=_ref(config["data"]["protocol"]);rows=[]
    schedule=[("warmup",0,sid) for sid in ids[:2]]+[("measured",repeat,sid) for repeat in range(3) for sid in ids]
    if config["task"]=="vqa":schedule += [("no_image",0,sid) for sid in ids]
    for stage,repeat,sid in schedule:
        row,stratum=selected[sid];cap=protocol["caps"][row["task_type"]]
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.monotonic()
        prediction=(_no_image if stage=="no_image" else dev.caps.initial._predict_one)(bundle,row,stratum,cap,config)
        torch.cuda.synchronize()
        item={"stage":stage,"repeat":repeat,"prediction":prediction,"synchronized_e2e_seconds":time.monotonic()-started,
            "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved()}
        rows.append(item);stream.write(json.dumps(item,sort_keys=True,ensure_ascii=False)+"\n");stream.flush()
        if prediction.get("model_fatal"):raise dev.caps.initial.CalibrationFatal(prediction["reason_code"])
    stream.flush();os.fsync(stream.fileno());properties=torch.cuda.get_device_properties(0)
    return _diagnostic_summary(config,rows),{"native_contract":bundle.contract,"gpu_identity":{"name":properties.name,
        "total_memory":properties.total_memory,"uuid":str(getattr(properties,"uuid","")),"compute_capability":list(torch.cuda.get_device_capability(0))}}


def _likelihood(config,stream):
    selected=_rows_for(config)["text"]
    rows,native=choice._execute(config,{"models":{config["cell_id"]:config["data"]["model_ref"]}},selected,stream)
    summary=choice._metrics(selected,rows)
    summary.update(procedure=copy.deepcopy(choice.PROCEDURE),test_files_read=True)
    choice._native_proof(config,native)
    if native["fatal_reason"]:raise dev.caps.initial.CalibrationFatal(native["fatal_reason"])
    return summary,{"native":native},rows


def seed_statistics(values_by_seed):
    """Missing/failed seeds stay explicit; no best-seed substitution."""
    required=(20261023,*SEEDS);available={str(seed):values_by_seed.get(seed) for seed in required}
    valid=[v for v in available.values() if v is not None]
    _require(all(isinstance(v,(int,float)) and math.isfinite(v) for v in valid),"COMPLETION_FINITE_SEED_VALUES_REQUIRED")
    complete=len(valid)==3
    return {"values":available,"successful_seeds":len(valid),"required_seeds":3,"three_seed_complete":complete,
        "mean":statistics.mean(valid) if complete else None,"sample_std":statistics.stdev(valid) if complete else None,
        "minimum":min(valid) if complete else None,"maximum":max(valid) if complete else None,
        "independent_seed_variation":True,"best_seed_selected":False}


def _rows_for(config):
    freeze=_ref(config["data"]["test_freeze"]["config"])
    return _test_rows(freeze["data"]["test_entries"])[0]


def _failure_scores(selected,protocol,reason):
    predictions=[dev.caps.initial._failure(row,stratum,protocol["caps"][row["task_type"]],reason) for row,stratum in selected]
    for prediction in predictions:prediction["attempted"]=False
    return dev.score_predictions(selected,predictions)


def _test_grouping(selected, axis):
    """Build transitive components from observed source fields, never scored group_id."""
    _require(axis in {"text", "vqa"}, "BOOTSTRAP_AXIS_REQUIRED")
    ids = [row["sample_id"] for row, _ in selected]
    _require(len(ids) == len(set(ids)) and bool(ids), "BOOTSTRAP_UNIQUE_NONEMPTY_IDS_REQUIRED")
    parent = list(range(len(ids))); tokens_seen = {}; evidence = []; reason_counts = Counter()
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    def join(a, b):
        a, b = root(a), root(b)
        if a != b: parent[max(a, b)] = min(a, b)
    for i, (row, stratum) in enumerate(selected):
        _require(row.get("split") == "test", "COMPLETION_TEST_GROUPING_ONLY")
        source = row["source_id"]; meta = row.get("metadata", {}); groups = row.get("group_ids", {})
        tokens = set(); fields = set()
        def add(namespace, key, value):
            for item in bootstrap._values(value):
                if item is not None and item != "" and item != []:
                    _require(isinstance(item, (str, int)), "BOOTSTRAP_GROUP_ID_MUST_BE_SCALAR")
                    tokens.add((namespace, key, str(item))); fields.add(key)
        for key in (("document", "question", "duplicate_cluster", "parent") if axis == "text" else ("image", "duplicate_cluster")):
            add(source, "group_ids."+key, groups.get(key))
        if axis == "text":
            for key in ("source_document_id", "source_question_id", "source_qa_id", "pmid"):
                namespace = source
                if source == "medquad" and key in {"source_document_id", "source_question_id"}:
                    context = meta.get("source_xml_file")
                    if not context:
                        reason_counts["MEDQUAD_LOCAL_ID_CONTEXT_UNAVAILABLE_IGNORED"] += bool(meta.get(key))
                        continue
                    namespace = canonical_hash([source, meta.get("subset"), context])
                add(namespace, "metadata."+key, meta.get(key))
            if source == "medquad": add(source, "provenance.source_url", row.get("provenance", {}).get("source_url"))
            if not any("document" in key or key == "metadata.pmid" for key in fields):
                reason_counts["SOURCE_DOCUMENT_GROUP_UNAVAILABLE"] += 1
            if not any("question" in key or key == "metadata.source_qa_id" for key in fields):
                reason_counts["SOURCE_QUESTION_GROUP_UNAVAILABLE"] += 1
        else:
            for mapping_name, mapping in (("metadata", meta), ("provenance", row.get("provenance", {}))):
                for key in ("image_sha256", "decoded_pixel_hash"):
                    # Same digest namespace across metadata/provenance and sources.
                    add("GLOBAL_IMAGE", key, mapping.get(key))
            for image in row.get("input", {}).get("images", []):
                add(source, "input.images", image)
            if not any("image" in key or "pixel_hash" in key for key in fields):
                reason_counts["IMAGE_GROUP_UNAVAILABLE"] += 1
        fallback = not tokens
        if fallback:
            reason_counts["NO_OBSERVED_GROUP_USING_SAMPLE_ID_SINGLETON"] += 1
            add(source, "sample_id_fallback", row["sample_id"])
        for token in tokens:
            if token in tokens_seen: join(i, tokens_seen[token])
            else: tokens_seen[token] = i
        evidence.append({"sample_id": row["sample_id"], "source_id": source, "stratum": stratum,
                         "source_row_sha256": canonical_hash(row), "tokens_sha256": canonical_hash(sorted(tokens)),
                         "observed_fields": sorted(fields), "singleton_fallback": fallback,
                         "grouping_quality": row.get("grouping_quality", {})})
    components = defaultdict(list)
    for i in range(len(ids)): components[root(i)].append(i)
    units = list(components.values())
    profiles = defaultdict(list)
    for unit_id, indices in enumerate(units):
        for i in indices: evidence[i]["component_index"] = unit_id
        sources = tuple(sorted({selected[i][0]["source_id"] for i in indices}))
        strata = tuple(sorted({selected[i][1] for i in indices}))
        profiles[(sources, strata)].append(unit_id)
    projection = [{"indices": indices, "sample_ids_sha256": canonical_hash([ids[i] for i in indices])} for indices in units]
    return {"axis": axis, "scheduled": len(ids), "group_count": len(units),
            "stratum_counts": dict(sorted(Counter(stratum for _, stratum in selected).items())),
            "group_size_counts": {str(size):count for size, count in sorted(Counter(len(unit) for unit in units).items())},
            "unavailable_reason_counts": dict(sorted(reason_counts.items())),
            "singleton_fallback_count": sum(e["singleton_fallback"] for e in evidence),
            "source_evidence_sha256": canonical_hash(evidence), "components_sha256": canonical_hash(projection),
            "profiles": [{"sources": list(sources), "strata": list(strata), "group_count": len(members)} for (sources, strata), members in profiles.items()],
            "patient_or_study_independence_certified": False}, units, list(profiles.values()), evidence


def _report_external(config):
    # Owner attestations are identified as such. A generation receipt cannot stand in for a CI/preservation proof.
    expected={"code_report_ci","existing_result_preservation"};supplied=config["data"]["evidence"]
    _require(set(supplied)<=expected,"COMPLETION_UNKNOWN_EXTERNAL_EVIDENCE")
    proof={};missing=[]
    for name in sorted(expected):
        ref=supplied.get(name)
        if ref is None:missing.append(name);continue
        value=_ref(ref)
        _require(value.get("schema_version")=="medvlm.v6.owner-evidence.v1" and value.get("kind")==name
            and value.get("status")=="SUCCEEDED" and value.get("source_hash")==config["source_hash"],"COMPLETION_TYPED_OWNER_EVIDENCE_REQUIRED")
        if name=="code_report_ci":
            commit=value.get("source_commit","");runs=value.get("ci_runs",[])
            _require(len(commit)==40 and all(c in "0123456789abcdef" for c in commit) and len(runs)>=3
                and value.get("pushed_commit")==commit and value.get("cpu_tests",{}).get("failed")==0
                and value.get("cpu_tests",{}).get("passed",0)>0 and all(r.get("head_sha")==commit
                and r.get("conclusion")=="success" and str(r.get("url","")).startswith("https://github.com/") for r in runs),
                "COMPLETION_EXACT_COMMIT_CI_EVIDENCE_REQUIRED")
        else:
            artifacts=value.get("protected_artifacts",[])
            _require(value.get("existing_v1_v5_preserved") is True and value.get("approved_release_preserved") is True
                and artifacts and all(_reference(item["path"])==item for item in artifacts),"COMPLETION_PRESERVATION_EVIDENCE_REQUIRED")
        proof[name]={"reference":ref,"evidence_level":"OWNER_RECORDED_EXTERNAL_EVIDENCE","proof":value}
    return proof,missing


@_transaction
def _report(config):
    selected=_rows_for(config);protocol=_ref(_reference(Path(_ref(config["data"]["test_freeze"]["config"])["output_dir"])/"protocol.json"))
    _,_,chosen,initial=_selection(config["data"]["selection"]["config"]["path"])
    chosen_cells={r["cell_id"] for r in chosen["selected_models"]}
    conditions={};scores={};rows=[];by_condition={};resources={};likelihood={}
    for ref in config["data"]["evaluations"]:
        child,condition,result=_terminal(config,ref);axis=child["task"]
        if result:
            summary=_read(Path(child["output_dir"])/"metrics.json")
            scored=[json.loads(line) for line in (Path(child["output_dir"])/"scored.private.jsonl").read_text().splitlines()]
            condition.update(technical_complete=summary["technical_failure_count"]==0,receipt=_reference(child["success_receipt"]),
                predictions=_reference(Path(child["output_dir"])/"predictions.private.jsonl"),scored=_reference(Path(child["output_dir"])/"scored.private.jsonl"))
        else:
            scored,summary=_failure_scores(selected[axis],protocol,"TEST_ROLE_JOB_UNAVAILABLE")
            condition.update(technical_complete=False,all_rows_unavailable=True)
        condition.update(cohorts=summary["cohorts"],strata=summary["strata"],scheduled=len(selected[axis]),vqa_subgroups=summary["vqa_subgroups"])
        for name,value in summary["strata"].items():
            rows.append({"cell_id":child["cell_id"],"model_id":child["model_id"],"family":child["family"],"task":axis,
                         "role":child["role"],"seed":child["seed"],"stratum":name,"job_status":condition["status"],**value})
            if child["role"]=="tuned" and child["cell_id"] in chosen_cells:
                
                for metric in ("normalized_em","lexical_f1","strict_all_row_accuracy","valid_format_rate"):
                    if value[metric] is not None:scores.setdefault((child["cell_id"],name,metric),{})[child["seed"]]=value[metric] if result else None
        conditions[child["job_id"]]=condition;by_condition[(child["cell_id"],child["role"],child["seed"])]=scored
    for key,target in (("diagnostics",resources),("likelihoods",likelihood)):
        for ref in config["data"][key]:
            child,condition,result=_terminal(config,ref)
            if result:condition["metrics"]=_read(Path(child["output_dir"])/"metrics.json")
            target[child["job_id"]]=condition
    paired={};grouping={};transitions={}
    for axis in ("text","vqa"):
        grouping[axis],units,profiles,_=_test_grouping(selected[axis],axis)
        for cell,*_ in dev.CELLS:
            if dev.cell_spec(cell)["task"]!=axis:continue
            left=by_condition[(cell,"base",20261023)];right=by_condition[(cell,"tuned",20261023)]
            paired[cell]=bootstrap.bootstrap_pair(selected[axis],left,right,units,profiles)
            grouped={}
            for a,b in zip(left,right):
                item=grouped.setdefault(a["stratum"],{"scheduled":0,"correct_transitions":{},"valid_transitions":{}});item["scheduled"]+=1
                for name,x,y in (("correct_transitions",a["exact_match"]==1.,b["exact_match"]==1.),("valid_transitions",a["valid"],b["valid"])):
                    label=f"{int(x)}->{int(y)}";item[name][label]=item[name].get(label,0)+1
            transitions[cell]=grouped
    external,missing=_report_external(config)
    identities=[]
    for job,condition in resources.items():
        metrics=condition.get("metrics")
        if metrics is None:missing.append("speed_unavailable:"+job);continue
        if metrics["technical_failures"]:missing.append("speed_technical_failures:"+job)
        identities.append(metrics.get("gpu_identity",{}).get("uuid"))
        ablation=metrics.get("image_ablation_32")
        if ablation and ablation.get("technical_failures",0):missing.append("image_ablation_technical_failures:"+job)
    if not identities or any(not identity for identity in identities):missing.append("same_physical_gpu_identity_unverified")
    elif len(set(identities))!=1:missing.append("speed_gpu_identity_mismatch")
    additional=_read(Path(_ref(_ref(config["data"]["test_freeze"]["config"])["data"]["additional_audit"]["config"])["output_dir"])/"summary.json")
    training={cell:{"model_id":dev.cell_spec(cell)["model_id"],"checkpoint":ref["checkpoint"],"reload":ref["reload"],"config":ref["config"],
                    "metrics":_ref(ref["metrics"])} for cell,ref in initial["models"].items()}
    additional_training={}
    for ref in _ref(_ref(config["data"]["test_freeze"]["config"])["data"]["additional_audit"]["config"])["data"]["train_configs"]:
        child,condition,result=_terminal(config,ref);entry={"config":ref,**condition}
        if result:
            output=Path(child["output_dir"])
            entry.update(checkpoint=_reference(output/"checkpoint.json"),reload=_reference(output/"reload.json"),
                final_selection=_reference(output/"final_selection.json"),metrics=_read(output/"metrics.json"))
        additional_training[child["job_id"]]=entry
    return {"scope":SCOPES["report"],"report_aggregation_completed":True,"evaluation_conditions":conditions,
        "first_seed_table":[r for r in rows if r["seed"]==20261023],"all_seed_table":rows,
        "selected_three_seed_confirmation":[{"cell_id":cell,"stratum":name,"metric":metric,**seed_statistics(values),
            "evaluation_status_by_seed":{str(seed):conditions.get(f"v6-test-{cell}-tuned-{seed}",{"status":"UNAVAILABLE_TRAINING_FAILED_OR_MISSING","technical_complete":False})
                for seed in (20261023,*SEEDS)},"all_three_evaluations_technical_complete":all(conditions.get(f"v6-test-{cell}-tuned-{seed}",{}).get("technical_complete") is True for seed in (20261023,*SEEDS))}
            for (cell,name,metric),values in sorted(scores.items())],
        "paired_transitions":transitions,"paired_group_intervals":paired,"grouping":grouping,"bootstrap_procedure":copy.deepcopy(bootstrap.PROCEDURE),
        "diagnostic_conditions":resources,"likelihood_conditions":likelihood,"additional_seed_audit":additional,"training_artifacts":training,"additional_training_artifacts":additional_training,
        "manifest_counts":{"first_seed_task_cells":10,"first_seed_unique_models":len({s["model_id"] for s in training.values()}),
                           "selected_additional_training_conditions":len(chosen["additional_seed_plan"]),"test_conditions":len(conditions)},
        "required_external_evidence":external,"missing_required_evidence":missing,"v6_completed":not missing,
        "v6_terminal":not missing,"v7_ready":not missing,"human_review_performed":0,"clinical_quality_certified":False,
        "anonymous_packet_prepared":True,"jetson_status":"USER_DEFERRED","limitations":LIMITATIONS,
        "test_files_read":True,"post_test_reselection_allowed":False,"training_seed_selection_from_test":False}


REPORT_NAMES=("RESULTS_KO.md","READOUT_KO.md","INDEX_KO.md","HANDOFF.md","WORKLOG.md","comparison.csv","seed_confirmation.csv",
              "accuracy_change.svg","format_compliance.svg","open_closed.svg","speed_quality.svg","vram_quality.svg",
              "anonymous_packet.private.json","anonymous_key.private.json","inference_examples.json")


def _csv_text(rows):
    stream=io.StringIO(newline="");writer=csv.DictWriter(stream,fieldnames=sorted({k for row in rows for k in row}))
    writer.writeheader();writer.writerows(rows);return stream.getvalue()


def _plot(title,points,xlabel="condition",ylabel="rate"):
    """Standalone numeric SVG; every point and axis is also in the JSON/CSV."""
    from xml.sax.saxutils import escape
    height=max(500,110+len(points)*18)
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="{height}"><rect width="1100" height="{height}" fill="white"/>',
        f'<text x="30" y="30">{escape(title)}</text><text x="30" y="{height-20}">{escape(xlabel+" / "+ylabel)}</text>']
    if not points:parts.append('<text x="30" y="100">UNAVAILABLE: no successful measured conditions</text>')
    for i,(label,x,y) in enumerate(points):
        parts.append(f'<text x="30" y="{65+i*18}">{escape(str(label))}: {x:.6g}, {y:.6g}</text>')
    # Table-like point labels remain legible for all20 conditions; dots show the same coordinates.
    xmax=max([abs(x) for _,x,_ in points]+[1.]);ymax=max([abs(y) for _,_,y in points]+[1.])
    for label,x,y in points:
        px=650+350*x/xmax;py=430-350*y/ymax
        parts.append(f'<circle cx="{px:.4f}" cy="{py:.4f}" r="4" fill="#2563eb"><title>{escape(str(label))}</title></circle>')
    parts.append('</svg>');return "".join(parts)


def _packet(config,key):
    from .evaluation.expert_review import REVIEW_FIELDS
    selected=_rows_for(config);protocol=_ref(_reference(Path(_ref(config["data"]["test_freeze"]["config"])["output_dir"])/"protocol.json"))
    mapping=key["mapping"];packets=[];cached=[]
    for ref in config["data"]["evaluations"]:
        child,_,result=_terminal(config,ref)
        predictions={r["sample_id"]:r for r in (json.loads(line) for line in (Path(child["output_dir"])/"predictions.private.jsonl").read_text().splitlines())} if result else {}
        cached.append((child,predictions))
    _require(set(mapping)=={_ref(ref)["job_id"] for ref in config["data"]["evaluations"]},"COMPLETION_ANONYMOUS_KEY_CONDITIONS_CHANGED")
    _require(len(set(mapping.values()))==len(mapping) and all(v.startswith("response-") for v in mapping.values()),"COMPLETION_ANONYMOUS_CODES_REQUIRED")
    for axis,rows in selected.items():
        wanted=protocol["diagnostic_ids"][axis]
        for row,stratum in rows:
            if row["sample_id"] not in wanted:continue
            answers=[]
            for child,predictions in cached:
                if child["task"]!=axis:continue
                prediction=predictions.get(row["sample_id"],{})
                answers.append({"code":mapping[child["job_id"]],"text":prediction.get("generated_text",""),
                    "available":prediction.get("status")=="SUCCEEDED","ratings":{field:None for field in REVIEW_FIELDS}})
            packets.append({"case_id":row["sample_id"],"task":axis,"stratum":stratum,"input":row["input"],
                "reference":row["supervision"],"responses":sorted(answers,key=lambda item:item["code"])})
    return {"schema_version":"medvlm.v6.anonymous-packet.v1","private_only":True,"human_review_performed":0,
        "model_identity_key_separate":True,"rubric_fields":REVIEW_FIELDS,"cases":packets}


@_transaction
def _render_report(config,summary,key):
    table=summary["first_seed_table"];by_cell={};points={}
    for row in table:
        item=by_cell.setdefault((row["cell_id"],row["role"]),[0,0,0]);n=row["scheduled"]
        item[0]+=row["normalized_em"]*n;item[1]+=row["valid_format_rate"]*n;item[2]+=n
    deltas=[];formats=[];openclosed=[];speed=[];vram=[]
    for i,(cell,role) in enumerate(sorted(by_cell)):
        correct,valid,n=by_cell[(cell,role)];label=cell+"/"+role;formats.append((label,float(i),valid/n))
        if role=="tuned":
            a=by_cell[(cell,"base")];deltas.append((cell,float(i),correct/n-a[0]/a[2]))
    for row in table:
        if row["task"]=="vqa":openclosed.append((row["cell_id"]+"/"+row["role"]+"/"+row["stratum"],float(len(openclosed)),row["normalized_em"]))
    for job,condition in summary["diagnostic_conditions"].items():
        if "metrics" not in condition:continue
        child=next(_ref(r) for r in config["data"]["diagnostics"] if _ref(r)["job_id"]==job);m=condition["metrics"]
        c,_,n=by_cell[(child["cell_id"],child["role"])];label=child["cell_id"]+"/"+child["role"]
        speed.append((label,m["latency_p50_seconds"],c/n));vram.append((label,m["peak_reserved_bytes"]/2**30,c/n))
    findings=[]
    for cell,*_ in dev.CELLS:
        a=by_cell[(cell,"base")];b=by_cell[(cell,"tuned")];findings.append(f"- {cell}: scheduled normalized EM {a[0]/a[2]:.4f} → {b[0]/b[2]:.4f} (Δ {b[0]/b[2]-a[0]/a[2]:+.4f}); valid format {a[1]/a[2]:.4f} → {b[1]/b[2]:.4f}.")
    status="증거 확인 완료" if summary["v6_completed"] else "미완료: "+", ".join(summary["missing_required_evidence"])
    body=("# V6 결과\n\nA. 완료 범위: "+json.dumps(summary["manifest_counts"],ensure_ascii=False)+"\n\n"
        "B. 동일 seed 20261023의 Base/Tuned 전체 예정 분모 비교는 comparison.csv에 있습니다. 실패도 분모에 포함됩니다. "
        "선택 모델의 세 seed는 별도 확인 표이며 단일 seed 순위와 섞지 않습니다.\n\n"+"\n".join(findings)+"\n\n"
        "C. 계열·크기·의료 특화는 관찰된 조건의 차이입니다. 통계적 우월성 또는 임상 정확도를 인증하지 않습니다. "
        "악화 및 형식 준수 변화는 accuracy_change.svg, format_compliance.svg와 paired_transitions에 보존됩니다.\n\n"
        "D. 같은 장비의 고정 32문항, warmup 2회, 3회 반복으로 속도/VRAM을 측정합니다. "
        "TTFT와 decode-only는 미측정입니다. 성공 측정 조건만 speed_quality.svg/vram_quality.svg에 있습니다. "
        "용도 후보는 필요한 source/형식/지연 조건별 표로 판단하며 모든 면의 단일 최고 모델을 지정하지 않습니다.\n\n"
        "E. 현재 판정: "+status+". 사람 검토는 0건이며 점수를 채우지 않았습니다. Jetson USER_DEFERRED. "
        "실패/미실행은 summary.json의 조건별 상태와 원래 native terminal record에 있습니다.\n\n"
        "F. config.json, summary.json, checkpoint/reload 참조, inference_examples.json, HANDOFF.md를 사용합니다. "
        "외부 commit/push/CI/보존 증거는 required_external_evidence에 명시됩니다.\n\n"+
        "\n".join("- "+value for value in LIMITATIONS)+"\n")
    examples=[]
    for axis in ("text","vqa"):
        ref=next(r for r in config["data"]["evaluations"] if _ref(r)["task"]==axis and _ref(r)["role"]=="tuned")
        child=_ref(ref);examples.append({"task":axis,"config":ref,"native_worker_argv":[sys.executable,"-m","mediflow_medvlm.campaign_v6_completion","--config",ref["path"],"--receipt",child["success_receipt"]],
            "execution_requires_canonical_supervisor_lease":True,"inference_scope":"full pinned local test task; no checkpoint selection"})
    return {"RESULTS_KO.md":body,"READOUT_KO.md":body,"INDEX_KO.md":"# V6\n\n[결과](RESULTS_KO.md) · [인계](HANDOFF.md) · comparison.csv · summary.json\n",
        "HANDOFF.md":"# V6 인계\n\n기존 canonical Supervisor의 등록된 config/receipt를 사용합니다. 직접 worker 중복 실행을 하지 않습니다. "
            "각 성공 학습의 checkpoint/fresh reload/adapter 경로는 summary.json training_artifacts와 additional_seed_audit에 고정됩니다. "
            "config 및 artifact hash가 일치할 때만 재사용하며 실패 학습은 original Base부터 재시도합니다. optimizer/RNG resume은 지원하지 않습니다. "
            "V7은 별도 승인된 구현 단계이며 이 worker가 실행하지 않습니다.\n",
        "WORKLOG.md":"# V6 worklog\n\n선택/추가 seed/test freeze/각 조건/집계의 원래 receipt와 terminal record hash를 summary.json에서 확인합니다. "
            "기존 v1~v5와 승인 Release는 수정하지 않습니다. 이 문서는 실행 증거를 대체하지 않습니다.\n",
        "comparison.csv":_csv_text(table),"seed_confirmation.csv":_csv_text(summary["selected_three_seed_confirmation"]),
        "accuracy_change.svg":_plot("Tuned minus Base normalized EM",deltas,ylabel="change"),
        "format_compliance.svg":_plot("Valid format / all scheduled",formats),"open_closed.svg":_plot("VQA source OPEN/CLOSED normalized EM",openclosed),
        "speed_quality.svg":_plot("Measured speed versus scheduled normalized EM",speed,xlabel="synchronized p50 seconds"),
        "vram_quality.svg":_plot("Measured VRAM versus scheduled normalized EM",vram,xlabel="peak reserved GiB"),
        "anonymous_packet.private.json":json.dumps(_packet(config,key),ensure_ascii=False,sort_keys=True,indent=2)+"\n",
        "anonymous_key.private.json":json.dumps(key,sort_keys=True,indent=2)+"\n",
        "inference_examples.json":json.dumps(examples,sort_keys=True,indent=2)+"\n"}


def _artifact_names(kind):
    if kind=="evaluation":return ("config.json","metrics.json","predictions.private.jsonl","scored.private.jsonl")
    if kind in {"diagnostic","likelihood"}:return ("config.json","metrics.json","records.private.jsonl")
    result=("config.json","summary.json")
    if kind=="test_freeze":result+=("protocol.json",)
    if kind=="report":result+=REPORT_NAMES
    return result


def _gpu_metrics_proof(config,metrics,records):
    kind=config["kind"];selected=_rows_for(config)[config["task"]];protocol=_ref(config["data"]["protocol"])
    if kind=="evaluation":
        dev.caps.initial._verify_rows(config,{"eval_context":4096},records,selected,protocol["caps"])
        scored,expected=dev.score_predictions(selected,records)
        _require(all(_equal(metrics.get(k),v) for k,v in expected.items()),"COMPLETION_TEST_METRICS_CHANGED")
    elif kind=="diagnostic":
        expected=_diagnostic_summary(config,records)
        _require(all(_equal(metrics.get(k),v) for k,v in expected.items()),"COMPLETION_DIAGNOSTIC_METRICS_CHANGED")
        ids=set(config["data"]["diagnostic_ids"]);subset=[pair for pair in selected if pair[0]["sample_id"] in ids]
        by_id={row["sample_id"]:(row,stratum) for row,stratum in subset}
        for item in records:
            prediction=item["prediction"];row,stratum=by_id[prediction["sample_id"]]
            _require(prediction.get("stratum")==stratum and prediction.get("input_sha256")==canonical_hash(row["input"]),"COMPLETION_DIAGNOSTIC_INPUT_CHANGED")
            if item["stage"]!="no_image":dev.caps.initial._verify_rows(config,{"eval_context":4096},[prediction],[(row,stratum)],protocol["caps"])
            elif prediction["status"]=="SUCCEEDED":
                dev.caps.initial._verify_rows(config,{"eval_context":4096},[prediction],[(row,stratum)],protocol["caps"])
                _require(prediction["pixel_tensor_keys"]==[] and prediction.get("input_metadata",{}).get("images")==[],"COMPLETION_NO_IMAGE_PIXELS_FORBIDDEN")
            else:_require(prediction["status"] in {"TECHNICAL_FAILURE","UNSUPPORTED_NATIVE_NO_IMAGE"} and prediction.get("reason_code"),"COMPLETION_NO_IMAGE_UNAVAILABLE_REASON_REQUIRED")
        selected=subset
        _require(metrics.get("gpu_identity",{}).get("name") and type(metrics["gpu_identity"].get("total_memory")) is int,"COMPLETION_DIAGNOSTIC_GPU_IDENTITY_REQUIRED")
    else:
        expected=choice._metrics(selected,records);expected.update(procedure=choice.PROCEDURE,test_files_read=True)
        _require(all(_equal(metrics.get(k),v) for k,v in expected.items()),"COMPLETION_LIKELIHOOD_METRICS_CHANGED")
        choice._native_proof(config,metrics["native"])
        _require(metrics["native"]["model_load_succeeded"] is True and metrics["native"]["fatal_reason"] is None,"COMPLETION_FATAL_LIKELIHOOD_CANNOT_SUCCEED")
        account=metrics.get("gpu_accounting",{})
        _require(account.get("ledger_path")==config["gpu_ledger_path"] and type(account.get("session_seconds")) in (int,float)
            and 0<=account["session_seconds"]<float("inf") and account.get("snapshot",{}).get("resource_policy")==config["resource_policy"],"COMPLETION_LIKELIHOOD_GPU_ACCOUNTING_CHANGED")
        return
    dev.caps._native_proof(config,{"models":{config["cell_id"]:config["data"]["model_ref"]}},metrics,selected)


@_transaction
def verify_result(config,*,repo=None):
    validate_config(config,repo=repo)
    if config["kind"]=="train":
        _,records,_=_seed_inputs(_ref(config["data"]["first_seed_template"]),config["seed"]);return training_runtime(config)._completed_output(config,records)
    output=Path(config["output_dir"]);kind=config["kind"];names=_artifact_names(kind)
    _require(not output.is_symlink() and {p.name for p in output.iterdir()}==set(names)|{"completion.json"}
        and not any(p.is_symlink() for p in output.iterdir()) and _equal(_read(output/"config.json"),config),"COMPLETION_ARTIFACT_SET_CHANGED")
    marker=_read(output/"completion.json")
    _require(marker.get("schema_version")=="medvlm.v6.completion-receipt.v1" and marker.get("status")=="SUCCEEDED" and marker.get("scope")==config["scope"]
        and all(marker.get(k)==config[k] for k in IDENTITY_KEYS) and marker.get("artifacts")==[_reference(output/name) for name in names]
        and marker.get("clinical_quality_certified") is False and marker.get("campaign_terminal") is False
        and marker.get("test_files_read") is (kind!="additional_audit"),"COMPLETION_RECEIPT_CHANGED")
    if config["uses_gpu"]:
        metrics=_read(output/"metrics.json")
        records=[json.loads(line) for line in (output/("predictions.private.jsonl" if kind=="evaluation" else "records.private.jsonl")).read_text().splitlines()]
        _gpu_metrics_proof(config,metrics,records)
        _require(metrics.get("protocol_hash")==config["protocol_hash"] and marker.get("v6_completed") is False,"COMPLETION_GPU_PROTOCOL_CHANGED")
        if kind=="evaluation":
            scored,_=dev.score_predictions(_rows_for(config)[config["task"]],records)
            _require(_equal(scored,[json.loads(line) for line in (output/"scored.private.jsonl").read_text().splitlines()]),"COMPLETION_TEST_SCORES_CHANGED")
    else:
        if kind=="additional_audit":expected=_additional_summary(config)
        elif kind=="test_audit":expected=_test_audit_summary(config)
        elif kind=="test_freeze":
            expected={"test_protocol_frozen":True,"test_inference_performed":False,"test_files_read":True}
            _require(_equal(_read(output/"protocol.json"),_test_protocol(config)),"COMPLETION_TEST_PROTOCOL_CHANGED")
        else:
            expected=_report(config);key=_read(output/"anonymous_key.private.json")
            for name,text in _render_report(config,expected,key).items():
                _require((output/name).read_text()==text,"COMPLETION_RENDERED_REPORT_CHANGED")
        _require(_equal(_read(output/"summary.json"),expected) and marker.get("v6_completed") is expected.get("v6_completed",False),"COMPLETION_SUMMARY_CHANGED")
    return {**marker,"artifacts":marker["artifacts"]+[_reference(output/"completion.json")]}


def run_job(config_path,receipt_path):
    path=_absolute(config_path);config=_read(path);before=file_hash(path);validate_config(config);dev.caps._env(config)
    _require(str(_absolute(receipt_path))==config["success_receipt"],"COMPLETION_RECEIPT_ROUTE_CHANGED")
    if config["kind"]=="train":return training_runtime(config).run_train(path,receipt_path)
    gpu=config["uses_gpu"];kind=config["kind"]
    if gpu:train._inherited_supervisor_fds(config)
    else:_require(os.environ.get("CUDA_VISIBLE_DEVICES")=="" and "torch" not in sys.modules,"COMPLETION_FRESH_CPU_REQUIRED")
    output=Path(config["output_dir"]);output.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    with output.with_name(output.name+".worker.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not output.exists():
            _require(not Path(receipt_path).exists(),"COMPLETION_RECEIPT_WITHOUT_OUTPUT")
            staging=Path(tempfile.mkdtemp(prefix=output.name+".attempt-",dir=output.parent));mask=os.umask(0o077)
            try:
                atomic_json(staging/"config.json",config)
                if gpu:
                    ledger=BudgetLedger(config["gpu_ledger_path"],policy=ResourcePolicy());ledger.start(config["model_id"],config["run_id"],"native_v6_test_"+kind)
                    try:
                        filename="predictions.private.jsonl" if kind=="evaluation" else "records.private.jsonl"
                        with (staging/filename).open("x") as stream:
                            if kind=="evaluation":scored,summary,native,_=_evaluation(config,stream)
                            elif kind=="diagnostic":summary,native=_diagnostic(config,stream)
                            else:summary,native,_=_likelihood(config,stream)
                    finally:ledger.finish()
                    summary.update(**native,protocol_hash=config["protocol_hash"],gpu_accounting={"ledger_path":config["gpu_ledger_path"],"session_seconds":ledger.session_seconds,"snapshot":ledger.snapshot()})
                    records=[json.loads(line) for line in (staging/filename).read_text().splitlines()]
                    _gpu_metrics_proof(config,summary,records)
                    if kind=="evaluation":
                        with (staging/"scored.private.jsonl").open("x") as stream:
                            for row in scored:stream.write(json.dumps(row,sort_keys=True,ensure_ascii=False)+"\n")
                            stream.flush();os.fsync(stream.fileno())
                    atomic_json(staging/"metrics.json",summary)
                else:
                    if kind=="additional_audit":summary=_additional_summary(config)
                    elif kind=="test_audit":summary=_test_audit_summary(config)
                    elif kind=="test_freeze":
                        summary={"test_protocol_frozen":True,"test_inference_performed":False,"test_files_read":True};atomic_json(staging/"protocol.json",_test_protocol(config))
                    else:
                        summary=_report(config);key={"private_only":True,"mapping":{_ref(ref)["job_id"]:"response-"+secrets.token_hex(8) for ref in config["data"]["evaluations"]}}
                        for name,text in _render_report(config,summary,key).items():(staging/name).write_text(text)
                    atomic_json(staging/"summary.json",summary)
                _require(file_hash(path)==before,"COMPLETION_CONFIG_CHANGED_DURING_EXECUTION");validate_config(config);names=_artifact_names(kind)
                marker={"schema_version":"medvlm.v6.completion-receipt.v1","status":"SUCCEEDED","scope":config["scope"],**{k:config[k] for k in IDENTITY_KEYS},
                    "v6_completed":summary.get("v6_completed",False),"campaign_terminal":False,"clinical_quality_certified":False,
                    "test_files_read":kind!="additional_audit","artifacts":[{"path":str(output/name),"sha256":file_hash(staging/name)} for name in names]}
                atomic_json(staging/"completion.json",marker);os.rename(staging,output)
                descriptor=os.open(output.parent,os.O_DIRECTORY)
                try:os.fsync(descriptor)
                finally:os.close(descriptor)
            except BaseException as exc:
                if staging.exists():atomic_json(staging/"failure.private.json",{"status":"FAILED","exception_type":type(exc).__name__,"reason_private":str(exc),"traceback":traceback.format_exc()})
                raise
            finally:os.umask(mask)
        result=verify_result(config)
        _require(file_hash(path)==before and (not Path(receipt_path).exists() or _equal(_read(receipt_path),result)),"COMPLETION_EXISTING_RECEIPT_CHANGED")
        if not Path(receipt_path).exists():atomic_json(receipt_path,result)
        return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--config",required=True);parser.add_argument("--receipt",required=True)
    stages=parser.add_mutually_exclusive_group();stages.add_argument("--train-checkpoint");stages.add_argument("--reload-checkpoint");args=parser.parse_args(argv)
    try:
        config=_read(args.config)
        if args.train_checkpoint or args.reload_checkpoint:
            _require(config["kind"]=="train","COMPLETION_TRAIN_CHILD_KIND_REQUIRED")
            runtime=training_runtime(config);fn=runtime.train_checkpoint if args.train_checkpoint else runtime.reload_checkpoint
            result=fn(_absolute(args.config),_absolute(args.train_checkpoint or args.reload_checkpoint),_absolute(args.receipt))
        else:
            if not config["uses_gpu"]:os.environ["CUDA_VISIBLE_DEVICES"]=""
            result=run_job(args.config,args.receipt)
    except Exception as exc:
        print(json.dumps({"status":"FAILED","exception_type":type(exc).__name__,**({"reason_code":str(exc)} if isinstance(exc,dev.caps.initial.gate.GateError) else {})}),file=sys.stderr);return 1
    print(json.dumps({"status":result["status"],"scope":result["scope"]}));return 0


if __name__=="__main__":raise SystemExit(main())
