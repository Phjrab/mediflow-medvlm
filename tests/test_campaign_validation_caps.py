"""Calibration contracts and native generation helpers; CPU only, synthetic rows."""
from contextlib import nullcontext
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mediflow_medvlm import campaign_validation_caps as c
from mediflow_medvlm import campaign_supervisor as supervisor

REPO = Path(__file__).resolve().parents[1]


def write(path, value):
    c.atomic_json(path, value)
    return Path(path)


def row(index, source, task, qtype=None, answer_type=None):
    return {"sample_id": source + "/" + str(index), "source_id": source, "task_type": task,
        "split": "validation", "input": {"question": "Question " + str(index), "images": []},
        "supervision": {"answer": "Private synthetic target"},
        "metadata": {"q_type": qtype, "answer_type": answer_type}}


@pytest.fixture
def proposal(tmp_path):
    groups = {"71875_mcq": [row(i, "aihub_71875", "medical_mcq", "1") for i in range(587)],
        "71875_short": [row(i+587, "aihub_71875", "medical_qa", "2") for i in range(93)],
        "71875_long": [row(i+680, "aihub_71875", "medical_qa", "3") for i in range(78)],
        "medquad": [row(i, "medquad", "medical_qa") for i in range(708)],
        "pubmedqa_labeled": [row(i, "pubmedqa_labeled", "context_qa") for i in range(50)],
        "slake_open": [row(i, "slake", "medical_vqa", answer_type="OPEN") for i in range(330)],
        "slake_closed": [row(i+330, "slake", "medical_vqa", answer_type="CLOSED") for i in range(182)]}
    refs, strata = {}, {}
    for axis in ("text", "vqa"):
        rows = [r for name, group in groups.items() if name.startswith("slake") == (axis == "vqa") for r in group]
        path = tmp_path / axis / "validation.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        refs[axis] = {**c._reference(path), "full_validation_count": len(rows)}
    for name, group in groups.items():
        chosen = sorted(group, key=lambda r: (hashlib.sha256((
            "medvlm-v6-recorded-cap-calibration-v1/20261023/"+r["sample_id"]).encode()).hexdigest(),r["sample_id"]))[:64]
        ids = [r["sample_id"] for r in chosen]
        strata[name] = {"axis": "vqa" if name.startswith("slake") else "text", "task_type": chosen[0]["task_type"],
            "full_validation_count": len(group), "calibration_count": len(ids),
            "ordered_sample_ids": ids, "ordered_ids_sha256": c.canonical_hash(ids)}
    proposal = {"schema_version": "medvlm.recorded-validation-calibration-ids.proposal.v1",
        "validation_only": True, "test_files_read": False, "selection_uses_question_or_answer_content": False,
        "seed": 20261023, "initial_caps": c.INITIAL_CAPS, "maximum_caps": c.MAXIMUM_CAPS,
        "references": refs, "strata": strata}
    proposal["proposal_hash"] = c.canonical_hash(proposal)
    return write(tmp_path / "proposal.json", proposal)


@pytest.fixture
def planned(tmp_path, proposal, monkeypatch):
    root = tmp_path / "private/v6_v7/campaigns/recorded"
    data_root = tmp_path / "private"
    children, rows = {}, []
    for cell in c.CELLS:
        cell_id = cell[0]
        output = root / "training" / cell_id
        model = {"model_id": cell[1], "revision": "a"*40, "cache_dir": str(data_root / "models/hub")}
        train = {"model": model, "model_hash": c.canonical_hash(model), "output_dir": str(output),
            "gpu_ledger_path": str(data_root / "runtime/gpu_budget.json"), "data": {"image_root": None}}
        config_ref = c._reference(write(output / "config.json", train))
        checkpoint = {"final_checkpoint": True, "complete": True, "adapter_sha256": "final-adapter",
                      "adapter_files": {"adapter_model.safetensors": "sha"}, "processor_files": {"tokenizer.json": "sha"}}
        write(output / "checkpoint.json", checkpoint)
        for name in ("metrics.json", "reload.json", "final_selection.json"):
            write(output / name, {"native_contract": {"parameter_storage": {"trainable_parameters": 0, "adapter_parameters": 0, "base_dtypes": []}}})
        receipt_ref = c._reference(write(output / "receipt.json", {"status": "SUCCEEDED", "artifacts": [config_ref]}))
        children["v6-first-seed-" + cell_id] = {"config": config_ref, "receipt": receipt_ref}
        rows.append({"cell_id": cell_id, "status": "SUCCEEDED", "final_checkpoint_verified": True,
                     "full_pass_verified": True, "fresh_process_reload_verified": True})
    prereq = {"campaign_id": "recorded", "campaign_root": str(root), "children": children}
    aggregate_config = {"prerequisites": prereq, "state_path": str(root / "state.json")}
    aggregate_config["config_hash"] = c.canonical_hash(aggregate_config)
    config_ref = c._reference(write(root / "audit/config.json", aggregate_config))
    summary_ref = c._reference(write(root / "audit/summary.json", {"aggregate_audit_succeeded": True,
        "first_seed_stage_terminal": True, "cells": rows}))
    receipt = write(root / "receipts/v6-first_seed_final.json", {"status": "SUCCEEDED", "scope": c.first.SCOPE,
        "first_seed_stage_terminal": True, "campaign_terminal": False, "artifacts": [config_ref, summary_ref]})
    monkeypatch.setattr(c.gate, "_active", lambda _: ({}, {}))
    monkeypatch.setattr(c.first, "_inputs", lambda *args: (prereq, rows))
    protocol = c.build_protocol(proposal, receipt, repo=REPO)
    protocol_path = write(root / "protocol.json", protocol)
    # Installed-wheel CI still hashes the checked-out source explicitly.
    real_source = c.source_identity
    monkeypatch.setattr(c, "source_identity", lambda repo=None: real_source(repo or REPO))
    config = c.build_config(protocol_path, "T1", "base", root / "calibration/T1-base", repo=REPO)
    config_path = write(root / "configs/T1-base.json", config)
    return SimpleNamespace(root=root, data_root=data_root, proposal=proposal, protocol=protocol,
        protocol_path=protocol_path, config=config, path=config_path)


def test_frozen_ids_no_targets_and_native_task_groups(planned):
    p = planned
    assert p.protocol["selection_recipe"]["namespace"] == "medvlm-v6-recorded-cap-calibration-v1/20261023/"
    text = c._selected(p.protocol, "text")
    assert len(text) == 306
    assert Counter(r[0]["task_type"] for r in text) == {"medical_mcq":64,"medical_qa":192,"context_qa":50}
    assert len(c._selected(p.protocol, "vqa")) == 128
    assert p.protocol["decode"] == {"do_sample":False,"num_beams":1,"use_cache":True,"max_time":None,"warmup_requests":0}
    assert p.config["eval_context"] == 4096
    assert p.config["supervisor_lock_paths"] == [str(p.data_root / "v6_v7/worker.lock"),str(p.data_root / "runtime/medvlm_gpu.lock")]
    assert p.config["data"]["partition"] == "validation"


def test_exact_twenty_initial_jobs_and_source_wrapper(planned):
    p = planned
    jobs = []
    for cell in c.CELLS:
        for role in c.ROLES:
            config = c.build_config(p.protocol_path, cell[0], role, p.root / "runs" / cell[0] / role, repo=REPO)
            path = write(p.root / "configs" / (cell[0] + role + ".json"), config)
            job = c.job_from_config(path, repo=REPO)
            assert job["dependencies"] == [c.first.JOB_ID]
            assert job["role"] == role
            assert "src/mediflow_medvlm/campaign_supervisor.py" in job["source_files"]
            jobs.append(job)
    assert [j["job_id"] for j in jobs] == c.INITIAL_IDS
    supervisor.validate_manifest({"campaign_id":"fixture", "jobs":[
        {"job_id":c.first.JOB_ID,"run_id":c.first.JOB_ID,"phase":"audit","cell_id":"ALL_V6","implementation_pending":True},*jobs]})


@pytest.mark.parametrize("field,value", [("eval_context",2048),("role","other"),("seed",42),
    ("uses_gpu",False),("decode",{"do_sample":True}),("supervisor_lock_paths",["/tmp/fake"]),
    ("gpu_ledger_path","/tmp/fake-ledger")])
def test_config_cannot_change_science_or_leases(planned,field,value):
    changed = copy.deepcopy(planned.config)
    changed[field] = value
    changed["config_hash"] = c.canonical_hash({k:v for k,v in changed.items() if k!="config_hash"})
    with pytest.raises(ValueError):c.validate_config(changed,repo=REPO)


def test_proposal_ids_cannot_be_reselected_and_test_path_denied(proposal):
    original = c._read(proposal)
    changed = copy.deepcopy(original)
    changed["strata"]["medquad"]["ordered_sample_ids"].reverse()
    changed["proposal_hash"] = c.canonical_hash({k:v for k,v in changed.items() if k!="proposal_hash"})
    write(proposal,changed)
    with pytest.raises(ValueError,match="PREDECLARED_CALIBRATION_IDS_CHANGED"):c._proposal(proposal)
    changed=copy.deepcopy(original)
    changed["references"]["text"]["path"]=str(proposal.parent / "test.jsonl")
    changed["proposal_hash"]=c.canonical_hash({k:v for k,v in changed.items() if k!="proposal_hash"})
    write(proposal,changed)
    with pytest.raises(ValueError,match="ONLY_EXACT_FROZEN_VALIDATION_FILE_ALLOWED"):c._proposal(proposal)


def test_gpu_lease_guard_precedes_ledger_or_model_load(planned,monkeypatch):
    def rejected(_):raise ValueError("NO_OWNER_LEASE")
    monkeypatch.setattr(c.first.main_train.smoke,"_inherited_supervisor_fds",rejected)
    monkeypatch.setattr(c,"BudgetLedger",lambda *a,**k: pytest.fail("lease rejection must precede ledger"))
    monkeypatch.setattr(c,"_execute",lambda *a: pytest.fail("lease rejection must precede GPU load"))
    with pytest.raises(ValueError,match="NO_OWNER_LEASE"):c.run_job(planned.path,planned.config["success_receipt"])
    assert not Path(planned.config["output_dir"]).exists()


@pytest.fixture
def generation(monkeypatch):
    torch = pytest.importorskip("torch")
    from mediflow_medvlm.evaluation import v2
    monkeypatch.setattr(torch.cuda,"synchronize",lambda:None)
    monkeypatch.setattr(torch,"autocast",lambda **kwargs:nullcontext())
    forwarded={}
    class Tokenizer:
        eos_token_id=99
        def decode(self,ids,skip_special_tokens=False):return "answer" if skip_special_tokens else "raw answer"
    class Model:
        device=torch.device("cpu")
        generation_config=SimpleNamespace(eos_token_id=[2,3])
        def generate(self,**kwargs):
            forwarded.update(kwargs)
            for step in range(2):
                for processor in kwargs["logits_processor"]: processor(kwargs["input_ids"],torch.zeros((1,10)))
            return torch.tensor([[10,11,7,3]])
    bundle=SimpleNamespace(model=Model(),processor=Tokenizer(),architecture={})
    encoded={"input_ids":torch.tensor([[10,11]]),"attention_mask":torch.tensor([[1,1]])}
    monkeypatch.setattr(v2,"encode_evaluation_sample",lambda *a,**k:(encoded,{}))
    return SimpleNamespace(torch=torch,bundle=bundle,encoded=encoded,forwarded=forwarded)


def test_native_generation_greedy_effective_eos_boundary_and_no_timeout(generation):
    p=generation
    r=row(0,"medquad","medical_qa")
    result=c._predict_one(p.bundle,r,"medquad",2,{"eval_context":4096,"task":"text","data":{"image_root":None}})
    assert result["status"]=="SUCCEEDED"
    assert result["finish_reason"]=="stop"  # EOS at cap boundary.
    assert result["eos_token_ids"]==[2,3]
    assert result["generated_raw_text"]=="raw answer"
    assert p.forwarded["do_sample"] is False and p.forwarded["num_beams"]==1 and p.forwarded["use_cache"] is True
    assert p.forwarded["max_time"] is None
    assert result["generation_scores_valid"] is True
    assert result["generation_scores_checked_steps"] == 2


def test_vlm_forwards_every_pixel_tensor(generation):
    p=generation
    p.encoded.update(pixel_values=p.torch.zeros(1,3,2,2),pixel_attention_mask=p.torch.ones(1,2,2))
    result=c._predict_one(p.bundle,row(0,"slake","medical_vqa"),"slake_open",256,
        {"eval_context":4096,"task":"vqa","data":{"image_root":"/private/images"}})
    assert result["status"]=="SUCCEEDED"
    assert p.forwarded["pixel_values"] is p.encoded["pixel_values"]
    assert p.forwarded["pixel_attention_mask"] is p.encoded["pixel_attention_mask"]
    assert set(result["pixel_tensor_keys"])=={"pixel_values","pixel_attention_mask"}


def test_full_input_overflow_is_failure_without_generate(generation):
    p=generation
    p.encoded["input_ids"]=p.torch.zeros((1,4096),dtype=p.torch.long)
    result=c._predict_one(p.bundle,row(0,"medquad","medical_qa"),"medquad",256,
        {"eval_context":4096,"task":"text","data":{"image_root":None}})
    assert result["status"]=="TECHNICAL_FAILURE" and "CONTEXT_OVERFLOW" in result["reason_code"]
    assert not p.forwarded


def test_technical_failures_remain_planned_denominator():
    rows=[{"task_type":"medical_qa","status":"TECHNICAL_FAILURE","finish_reason":None},
          {"task_type":"medical_qa","status":"SUCCEEDED","finish_reason":"length"}]
    assert c._task_counts(rows)=={"medical_qa":{"scheduled":2,"successful":1,"technical_failures":1,"length_terminated":1}}


from collections import Counter


def test_model_load_failure_preserves_every_planned_row_before_failing(planned,tmp_path,monkeypatch):
    # This branch fails before any tensor/CUDA operation. Keep its accounting
    # assertion active in the lightweight CI install without optional torch.
    import sys
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    from mediflow_medvlm.evaluation import v6_native
    def fail(*a,**k):raise RuntimeError("synthetic loader failure")
    monkeypatch.setattr(v6_native,"load_evaluation_bundle",fail)
    selected=c._selected(planned.protocol,"text")
    path=tmp_path/"failure-predictions.jsonl"
    with path.open("w") as stream:
        with pytest.raises(c.CalibrationFatal):c._execute(planned.config,planned.protocol,selected,c.INITIAL_CAPS,stream)
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows)==306 and all(r["status"]=="TECHNICAL_FAILURE" for r in rows)
    assert all(r["model_fatal"] is True for r in rows)


def test_unexpected_decode_failure_preserves_remaining_rows(planned,tmp_path,monkeypatch):
    torch = pytest.importorskip("torch")
    from mediflow_medvlm.evaluation import v6_native
    monkeypatch.setattr(v6_native,"load_evaluation_bundle",lambda *a,**k:SimpleNamespace(contract={}))
    monkeypatch.setattr(torch.cuda,"reset_peak_memory_stats",lambda:None)
    def fail(*a,**k):raise KeyError("synthetic decoder failure")
    monkeypatch.setattr(c,"_predict_one",fail)
    selected=c._selected(planned.protocol,"text")
    path=tmp_path/"decode-failure.jsonl"
    with path.open("w") as stream:
        with pytest.raises(c.CalibrationFatal):c._execute(planned.config,planned.protocol,selected,c.INITIAL_CAPS,stream)
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows)==306 and all(r["reason_code"]=="POSTPROCESS_KeyError" for r in rows)


def test_bumped_round_cannot_run_without_reviewed_cpu_decision(planned):
    with pytest.raises(ValueError,match="CALIBRATION_INITIAL_ROUND_ONLY"):
        c.build_config(planned.protocol_path,"T1","base",planned.root/"bumped",repo=REPO,
            calibration_round="bumped",decision_reference={"path":"/untrusted","sha256":"fake"})


@pytest.mark.parametrize("scores", [[[0.,float('nan')]], [[0.,float('inf')]], [[float('-inf'),float('-inf')]], [[0.,1.],[float('-inf'),float('-inf')]]])
def test_current_step_scores_reject_nan_posinf_or_no_finite_candidate(scores):
    torch=pytest.importorskip("torch")
    check=c.GenerationScoreGuard()
    with pytest.raises(ValueError,match="INVALID_CURRENT_STEP_GENERATION_SCORES"):
        check(torch.tensor([[1]]),torch.tensor(scores))
    assert check.checked_steps==0


def test_current_step_score_guard_accepts_negative_inf_mask_without_raw_logit_claim():
    torch=pytest.importorskip("torch")
    check=c.GenerationScoreGuard()
    scores=torch.tensor([[0.,float('-inf'),1.]])
    assert check(torch.tensor([[1]]),scores) is scores
    assert check.checked_steps==1


def test_role_worker_receipt_exact_denominators_hashes_and_idempotent(planned,monkeypatch):
    p=planned
    monkeypatch.setattr(c.first.main_train.smoke,"_inherited_supervisor_fds",lambda _: (10,11))
    ledger_calls=[]
    class FixtureLedger:
        session_seconds=.01
        def __init__(self,path,policy):self.path=path
        def start(self,*args):ledger_calls.append(args)
        def finish(self):pass
        def snapshot(self):return {"resource_policy":c.ResourcePolicy().as_dict()}
    monkeypatch.setattr(c,"BudgetLedger",FixtureLedger)
    def execute(config,protocol,selected,caps,stream):
        result=[]
        for row,stratum in selected:
            value=c._failure(row,stratum,caps[row["task_type"]],None)
            value.update(status="SUCCEEDED",generated_ids=[7,2],generated_tokens=2,
                generated_ids_sha256=c.canonical_hash([7,2]),finish_reason="stop",eos_token_ids=[2],input_tokens=10,
                no_thinking_output={"explicit_nonempty_thinking_detected":False},generation_scores_valid=True,
                raw_forward_logits_finite_verified=False,generation_scores_checked_steps=2)
            result.append(value)
            stream.write(json.dumps(value)+'\n')
        return result,{"native_contract":{"weights_loaded":True,"cuda_residency_verified":True,
            "evaluation_role":"base","evaluation_adapter_sha256":None,
            "parameter_storage":{"trainable_parameters":0,"adapter_parameters":0,"base_dtypes":[]}}}
    monkeypatch.setattr(c,"_execute",execute)
    result=c.run_job(p.path,p.config["success_receipt"])
    assert result["scheduled_count"]==306 and result["technical_complete"] is True
    assert result["test_files_read"] is False and len(ledger_calls)==1
    assert c.run_job(p.path,p.config["success_receipt"])==result
    assert len(ledger_calls)==1
    predictions=Path(p.config["output_dir"])/"predictions.private.jsonl"
    predictions.write_text(predictions.read_text().replace('"generated_tokens": 2','"generated_tokens": 1',1))
    with pytest.raises(ValueError,match="CAP_COMPLETION_ARTIFACT_CHANGED"):
        c.verify_result(p.config,repo=REPO)


def test_once_only_universal_cap_bump_uses_planned_task_denominators(planned):
    conditions={}
    for cell in c.CELLS:
        scheduled=Counter(row["task_type"] for row,_ in c._selected(planned.protocol,c.cell_spec(cell[0])["task"]))
        for role in c.ROLES:
            conditions[f"v6-caps-initial-{cell[0]}-{role}"]={"status":"SUCCEEDED","task_counts":{
                task:{"scheduled":n,"successful":n,"technical_failures":0,"length_terminated":0} for task,n in scheduled.items()}}
    conditions["v6-caps-initial-T1-base"]["task_counts"]["medical_qa"]["length_terminated"]=1
    assert c.cap_decision_from_counts(planned.protocol,conditions)["bumped_task_types"]==[]
    conditions["v6-caps-initial-T1-base"]["task_counts"]["medical_qa"]["length_terminated"]=2
    conditions["v6-caps-initial-V4-tuned"]["task_counts"]["medical_vqa"]["length_terminated"]=2
    result=c.cap_decision_from_counts(planned.protocol,conditions)
    assert result["caps"]=={"medical_mcq":64,"context_qa":64,"medical_qa":512,"medical_vqa":512}
    assert result["bump_rounds"]==1 and result["universal_across_models_and_roles"] is True
    assert result["cap_protocol_frozen"] is False
    bad=copy.deepcopy(conditions)
    bad["v6-caps-initial-T1-base"]["task_counts"]["medical_qa"]["scheduled"]-=1
    with pytest.raises(ValueError,match="CAP_CALIBRATION_DENOMINATOR_CHANGED"):
        c.cap_decision_from_counts(planned.protocol,bad)
