"""CPU proof, once-bump, subset/empty rerun, and failure-aware freeze contracts."""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from mediflow_medvlm import campaign_validation_cap_refinement as r
from mediflow_medvlm import campaign_supervisor as supervisor
from test_campaign_validation_caps import planned as initial_planned, proposal

REPO=Path(__file__).resolve().parents[1]


def write(path,value):r.atomic_json(path,value);return Path(path)


def native_metrics(role,protocol,cell):
    return {"native_contract":{"weights_loaded":True,"cuda_residency_verified":True,
        "evaluation_role":role,"evaluation_adapter_sha256":protocol["models"][cell]["adapter_sha256"] if role=="tuned" else None,
        "parameter_storage":{"trainable_parameters":0,"adapter_parameters":16 if role=="tuned" else 0,"base_dtypes":[]}}}


def predictions(config,protocol,selected,caps,trigger=False):
    rows=[];seen={}
    for row,stratum in selected:
        task=row["task_type"];cap=caps[task];seen[task]=seen.get(task,0)+1
        truncated=bool(trigger) and (trigger is True or task in trigger) and seen[task] <= (2 if task in {"medical_qa","medical_vqa"} else 1)
        ids=[7]*cap if truncated else [7,2]
        result=r.initial._failure(row,stratum,cap,None)
        result.update(status="SUCCEEDED",generated_ids=ids,generated_tokens=len(ids),generated_ids_sha256=r.canonical_hash(ids),
            finish_reason="length" if truncated else "stop",eos_token_ids=[2],input_tokens=10,
            no_thinking_output={"explicit_nonempty_thinking_detected":False},generation_scores_valid=True,
            raw_forward_logits_finite_verified=False,generation_scores_checked_steps=len(ids))
        rows.append(result)
    return rows


def publish_initial(config,protocol,trigger):
    output=Path(config["output_dir"]);output.mkdir(parents=True,exist_ok=True)
    write(output/"config.json",config)
    rows=predictions(config,protocol,r.initial._selected(protocol,config["task"]),r.initial.INITIAL_CAPS,trigger)
    (output/"predictions.private.jsonl").write_text(''.join(json.dumps(row)+'\n' for row in rows))
    metrics={"task_counts":r.initial._task_counts(rows),"scheduled_count":len(rows),"technical_failure_count":0,
        **native_metrics(config["role"],protocol,config["cell_id"]),"gpu_accounting":{"session_seconds":.001,
        "ledger_path":config["gpu_ledger_path"],"snapshot":{"resource_policy":r.ResourcePolicy().as_dict()}}}
    write(output/"metrics.json",metrics)
    marker={"status":"SUCCEEDED","scope":r.initial.SCOPE,**{k:config[k] for k in r.IDENTITY_KEYS},
        "protocol_hash":protocol["protocol_hash"],"scheduled_count":len(rows),"skipped_no_task_cap_change":False,
        "technical_complete":True,"test_files_read":False,"campaign_terminal":False,
        "artifacts":[r._reference(output/name) for name in ("config.json","metrics.json","predictions.private.jsonl")]}
    write(output/"completion.json",marker)
    write(config["success_receipt"],{**marker,"artifacts":marker["artifacts"]+[r._reference(output/"completion.json")]})


def state_record(job,status="SUCCEEDED"):
    value={"status":status,"attempts":1,"failure_count":0 if status=="SUCCEEDED" else 1,"recovery_attempts":0,"exit_code":0 if status=="SUCCEEDED" else 1,"history":[]}
    if status=="SUCCEEDED":value["verification"]=supervisor.verify_success(job)
    return value


@pytest.fixture
def planned(initial_planned,monkeypatch):
    p=initial_planned
    jobs=[{"job_id":r.initial.first.JOB_ID,"run_id":r.initial.first.JOB_ID,"phase":"first_seed_final","cell_id":"ALL_V6","dependencies":[],"implementation_pending":True}]
    records={r.initial.first.JOB_ID:{"status":"SUCCEEDED","attempts":1,"failure_count":0,"recovery_attempts":0}}
    for cell in r.CELLS:
        for role in r.initial.ROLES:
            config=r.initial.build_config(p.protocol_path,cell[0],role,p.root/"initial"/cell[0]/role,repo=REPO)
            path=write(p.root/"initial-configs"/(cell[0]+role+'.json'),config)
            job=r.initial.job_from_config(path,repo=REPO)
            publish_initial(config,p.protocol,role=="base" and cell[0] in {"T1","V1"})
            jobs.append(job);records[job["job_id"]]=state_record(job)
    manifest={"campaign_id":"recorded","campaign_root":str(p.root),"jobs":jobs}
    state={"campaign_id":"recorded","manifest_path":str(p.root/"manifest.json"),"manifest_hash":r.canonical_hash(manifest),"jobs":records}
    write(p.root/"manifest.json",manifest);state_path=write(p.root/"state.json",state)
    real_active=r._active
    # The upstream synthetic protocol fixture mocks the historical helper only;
    # the new-stage live-manifest read must use its real canonical pointer check.
    monkeypatch.setattr(r,"_active",lambda path:(lambda s:(r._read(s["manifest_path"]),s))(r._read(path)))
    decision=r.build_decision_config(p.protocol_path,state_path,p.root/"decision-output",repo=REPO)
    decision_path=write(p.root/"decision-config.json",decision)
    return SimpleNamespace(base=p,root=p.root,state_path=state_path,manifest=manifest,state=state,
        decision=decision,decision_path=decision_path,old_active=real_active)


def update_state(p,jobs,records):
    manifest=copy.deepcopy(p.manifest);manifest["jobs"].extend(jobs)
    state=r._read(p.state_path);state["jobs"].update(records)
    state["manifest_hash"]=r.canonical_hash(manifest)
    write(state["manifest_path"],manifest);write(p.state_path,state)
    p.manifest=manifest


def run_decision(p):
    job=r.job_from_config(p.decision_path,repo=REPO)
    update_state(p,[job],{r.DECISION_ID:{"status":"PENDING","attempts":0,"failure_count":0,"recovery_attempts":0}})
    result=r.run_job(p.decision_path,p.decision["success_receipt"])
    state=r._read(p.state_path);state["jobs"][r.DECISION_ID]=state_record(job);write(p.state_path,state)
    return result


def add_bumped(p):
    configs=[];jobs=[]
    for cell in r.CELLS:
        for role in r.initial.ROLES:
            config=r.build_bumped_config(p.decision_path,cell[0],role,p.root/"bumped"/cell[0]/role,repo=REPO)
            path=write(p.root/"bumped-configs"/(cell[0]+role+'.json'),config)
            configs.append(path);jobs.append(r.job_from_config(path,repo=REPO))
    update_state(p,jobs,{j["job_id"]:{"status":"PENDING","attempts":0,"failure_count":0,"recovery_attempts":0} for j in jobs})
    return configs,jobs


def mock_gpu(monkeypatch):
    calls=[]
    monkeypatch.setattr(r.initial.first.main_train.smoke,"_inherited_supervisor_fds",lambda _: (10,11))
    class Ledger:
        session_seconds=.001
        def __init__(self,*a,**k):pass
        def start(self,*a):calls.append(a)
        def finish(self):pass
        def snapshot(self):return {"resource_policy":r.ResourcePolicy().as_dict()}
    monkeypatch.setattr(r,"BudgetLedger",Ledger)
    def execute(config,protocol,selected,caps,stream):
        rows=predictions(config,protocol,selected,caps)
        stream.write(''.join(json.dumps(row)+'\n' for row in rows))
        return rows,native_metrics(config["role"],protocol,config["cell_id"])
    monkeypatch.setattr(r.initial,"_execute",execute)
    return calls


def test_deep_initial_audit_global_once_bump_and_cpu_progress(planned,monkeypatch):
    p=planned;progress=p.root/"cpu-progress.json"
    monkeypatch.setenv("MEDVLM_SUPERVISOR_PROGRESS_PATH",str(progress))
    before={key:copy.deepcopy(value) for key,value in p.state["jobs"].items()}
    result=run_decision(p)
    assert result["caps"]==r.initial.MAXIMUM_CAPS and result["cap_protocol_frozen"] is False
    summary=r._read(Path(p.decision["output_dir"])/"summary.json")
    assert summary["bumped_task_types"]==sorted(r.initial.INITIAL_CAPS)
    assert summary["technical_incomplete_initial_conditions"]==[]
    assert r._read(progress)["audited_conditions"]==20 and r._read(progress)["gpu_work_performed"] is False
    assert all(r._read(p.state_path)["jobs"][key]==value for key,value in before.items())
    assert r.run_job(p.decision_path,p.decision["success_receipt"])==result


@pytest.mark.parametrize("changed",["terminal","config","receipt","predictions","source-counts"])
def test_initial_proof_tamper_blocks_decision_without_success(planned,changed):
    p=planned;key=r.initial.INITIAL_IDS[0];proof=p.decision["data"]["initial_proofs"][key]
    if changed=="terminal":
        state=r._read(p.state_path);state["jobs"][key]["attempts"]=2;write(p.state_path,state)
    elif changed=="config":
        config=r._ref(proof["config"]);config["eval_context"]=2048;write(proof["config"]["path"],config)
    elif changed=="receipt":
        receipt=r._ref(proof["receipt"]);receipt["technical_complete"]=False;write(proof["receipt"]["path"],receipt)
    elif changed=="predictions":
        config=r._ref(proof["config"]);path=Path(config["output_dir"])/"predictions.private.jsonl";path.write_text(path.read_text()+'\n')
    else:
        config=copy.deepcopy(p.decision);config["data"]["initial_proofs"][key]["task_counts"]["medical_qa"]["scheduled"]-=1
        config["config_hash"]=r.canonical_hash({k:v for k,v in config.items() if k!="config_hash"});write(p.decision_path,config)
    with pytest.raises(ValueError):r.run_job(p.decision_path,p.decision["success_receipt"])
    assert not Path(p.decision["success_receipt"]).exists()


def test_twenty_explicit_bumped_jobs_and_exact_selected_ids(planned):
    p=planned;paths=[];jobs=[]
    for cell in r.CELLS:
        for role in r.initial.ROLES:
            config=r.build_bumped_config(p.decision_path,cell[0],role,p.root/"rerun"/cell[0]/role,repo=REPO)
            assert len(config["data"]["ordered_sample_ids"])==(306 if cell[0].startswith('T') else 128)
            assert config["eval_context"]==4096 and config["decode"]["max_time"] is None
            assert config["supervisor_lock_paths"]==[str(p.base.data_root/"v6_v7/worker.lock"),str(p.base.data_root/"runtime/medvlm_gpu.lock")]
            path=write(p.root/"configs"/(cell[0]+role+'.json'),config);paths.append(path);jobs.append(r.job_from_config(path,repo=REPO))
    assert [job["job_id"] for job in jobs]==r.BUMPED_IDS
    assert all(job["dependencies"]==[r.DECISION_ID] and job["uses_gpu"] and job["requires_gpu"] for job in jobs)
    freeze=r.build_freeze_config(p.decision_path,paths,p.state_path,p.root/"freeze",repo=REPO)
    assert freeze["data"]["bumped_configs"]==[r._reference(path) for path in paths]


def test_bumped_requires_actual_decision_then_uses_same_worker_once(planned,monkeypatch):
    p=planned;config=r.build_bumped_config(p.decision_path,"T1","base",p.root/"rerun-T1",repo=REPO)
    path=write(p.root/"rerun.json",config);calls=mock_gpu(monkeypatch)
    with pytest.raises((ValueError,StopIteration)):r.run_job(path,config["success_receipt"])
    assert calls==[]
    run_decision(p)
    result=r.run_job(path,config["success_receipt"])
    assert result["scheduled_count"]==306 and result["technical_complete"] is True and len(calls)==1
    assert r.run_job(path,config["success_receipt"])==result and len(calls)==1


def test_freeze_retains_terminal_failed_rerun_without_lower_cap_fallback(planned,monkeypatch):
    p=planned;run_decision(p);paths,jobs=add_bumped(p);calls=mock_gpu(monkeypatch)
    state=r._read(p.state_path)
    for path,job in zip(paths,jobs):
        config=r._read(path)
        if job["job_id"]==r.BUMPED_IDS[0]:state["jobs"][job["job_id"]]=state_record(job,"FAILED");continue
        r.run_job(path,config["success_receipt"])
        state["jobs"][job["job_id"]]=state_record(job)
    write(p.state_path,state)
    freeze=r.build_freeze_config(p.decision_path,paths,p.state_path,p.root/"freeze",repo=REPO)
    path=write(p.root/"freeze-config.json",freeze)
    result=r.run_job(path,freeze["success_receipt"])
    summary=r._read(Path(freeze["output_dir"])/"summary.json")
    failed=summary["conditions"][r.BUMPED_IDS[0]]
    assert failed["planned_count"]==306 and failed["unavailable_count"]==306
    assert sum(v["technical_failures"] for v in failed["final_task_counts"].values())==306
    assert result["cap_protocol_frozen"] is True and result["all_conditions_technical_complete"] is False
    assert summary["clinical_quality_certified"] is False and summary["v6_completed"] is False
    assert summary["test_unlocked"] is False and summary["family_selection_completed"] is False
    assert summary["final_protocol"]["further_bump_allowed"] is False
    assert len(calls)==19


def replan(p,tasks):
    state=r._read(p.state_path)
    for job in p.manifest["jobs"]:
        if job.get("calibration_round")!="initial":continue
        config=r._read(job["argv"][4])
        publish_initial(config,p.base.protocol,tasks if job["job_id"]=="v6-caps-initial-T1-base" else False)
        state["jobs"][job["job_id"]]=state_record(job)
    write(p.state_path,state)
    p.decision=r.build_decision_config(p.base.protocol_path,p.state_path,p.root/"decision-output",repo=REPO)
    write(p.decision_path,p.decision)


def test_empty_global_change_truthful_skip_never_loads_cuda_or_ledger(planned,monkeypatch):
    p=planned;replan(p,set());run_decision(p)
    config=r.build_bumped_config(p.decision_path,"T1","base",p.root/"empty-rerun",repo=REPO)
    assert config["data"]["ordered_sample_ids"]==[]
    path=write(p.root/"empty.json",config)
    monkeypatch.setattr(r.initial.first.main_train.smoke,"_inherited_supervisor_fds",lambda _: (10,11))
    monkeypatch.setattr(r,"BudgetLedger",lambda *a,**k:pytest.fail("empty subset must not write GPU ledger"))
    monkeypatch.setattr(r.initial,"_execute",lambda *a:pytest.fail("empty subset must not initialize CUDA"))
    result=r.run_job(path,config["success_receipt"])
    assert result["scheduled_count"]==0 and result["skipped_no_task_cap_change"] is True
    metrics=r._read(Path(config["output_dir"])/"metrics.json")
    assert metrics["gpu_accounting"] is None and metrics["no_gpu_work_needed"] is True
    assert (Path(config["output_dir"])/"predictions.private.jsonl").read_bytes()==b''


def test_only_changed_task_ids_rerun_and_empty_vqa_uses_no_gpu(planned,monkeypatch):
    p=planned;replan(p,{"medical_qa"});run_decision(p)
    expected=p.decision["data"]["expected_decision"]
    assert expected["caps"]=={"medical_mcq":64,"context_qa":64,"medical_qa":512,"medical_vqa":256}
    calls=mock_gpu(monkeypatch)
    config=r.build_bumped_config(p.decision_path,"T4","tuned",p.root/"subset-text",repo=REPO)
    path=write(p.root/"subset-text.json",config)
    assert len(config["data"]["ordered_sample_ids"])==192
    result=r.run_job(path,config["success_receipt"])
    rows=[json.loads(line) for line in (Path(config["output_dir"])/"predictions.private.jsonl").read_text().splitlines()]
    assert result["scheduled_count"]==192 and all(row["task_type"]=="medical_qa" and row["max_new_tokens"]==512 for row in rows)
    config=r.build_bumped_config(p.decision_path,"V1","base",p.root/"subset-vqa",repo=REPO)
    path=write(p.root/"subset-vqa.json",config)
    assert r.run_job(path,config["success_receipt"])["skipped_no_task_cap_change"] is True
    assert len(calls)==1


@pytest.mark.parametrize("field,value",[("caps",{}),("cap_protocol_frozen",True),("test_files_read",True),("schema_version","other")])
def test_cpu_completion_marker_cannot_claim_false_policy(planned,field,value):
    p=planned;run_decision(p)
    path=Path(p.decision["output_dir"])/"completion.json";marker=r._read(path);marker[field]=value;write(path,marker)
    with pytest.raises(ValueError,match="CAP_CPU_COMPLETION_CHANGED"):r.verify_result(p.decision,repo=REPO)


def test_decision_reuse_rehashes_initial_artifacts_even_when_counts_unchanged(planned):
    p=planned;run_decision(p)
    proof=p.decision["data"]["initial_proofs"][r.initial.INITIAL_IDS[0]];config=r._ref(proof["config"])
    path=Path(config["output_dir"])/"predictions.private.jsonl"
    lines=path.read_text().splitlines();row=json.loads(lines[0]);row["private_extra"]="changed bytes with same count";lines[0]=json.dumps(row)
    path.write_text('\n'.join(lines)+'\n')
    with pytest.raises(ValueError,match="artifact hash mismatch"):r.verify_result(p.decision,repo=REPO)


def test_successful_rerun_with_technical_row_is_not_technical_complete(planned,monkeypatch):
    p=planned;run_decision(p);calls=mock_gpu(monkeypatch)
    def execute(config,protocol,selected,caps,stream):
        rows=predictions(config,protocol,selected,caps)
        row,stratum=selected[0];rows[0]=r.initial._failure(row,stratum,caps[row["task_type"]],"CONTEXT_OVERFLOW")
        stream.write(''.join(json.dumps(row)+'\n' for row in rows))
        return rows,native_metrics(config["role"],protocol,config["cell_id"])
    monkeypatch.setattr(r.initial,"_execute",execute)
    config=r.build_bumped_config(p.decision_path,"T1","base",p.root/"technical-row",repo=REPO)
    path=write(p.root/"technical-row.json",config);result=r.run_job(path,config["success_receipt"])
    assert result["status"]=="SUCCEEDED" and result["technical_complete"] is False
    metrics=r._read(Path(config["output_dir"])/"metrics.json")
    assert metrics["scheduled_count"]==306 and metrics["technical_failure_count"]==1
    assert sum(v["scheduled"] for v in metrics["task_counts"].values())==306 and len(calls)==1


def test_bumped_fatal_preserves_attempt_and_never_publishes_success(planned,monkeypatch):
    p=planned;run_decision(p);calls=mock_gpu(monkeypatch)
    def fail(config,protocol,selected,caps,stream):
        for row,stratum in selected:
            stream.write(json.dumps(r.initial._failure(row,stratum,caps[row["task_type"]],"MODEL_FATAL",fatal=True))+'\n')
        raise r.initial.CalibrationFatal("MODEL_FATAL")
    monkeypatch.setattr(r.initial,"_execute",fail)
    config=r.build_bumped_config(p.decision_path,"T1","base",p.root/"fatal-row",repo=REPO)
    path=write(p.root/"fatal-row.json",config)
    with pytest.raises(r.initial.CalibrationFatal):r.run_job(path,config["success_receipt"])
    assert not Path(config["success_receipt"]).exists() and len(calls)==1
    attempts=list(Path(config["output_dir"]).parent.glob('fatal-row.attempt-*'))
    assert len(attempts)==1 and (attempts[0]/"failure.private.json").exists()
    assert len((attempts[0]/"predictions.private.jsonl").read_text().splitlines())==306


def test_no_freeze_before_all_twenty_reruns_terminal(planned):
    p=planned;run_decision(p);paths,jobs=add_bumped(p)
    freeze=r.build_freeze_config(p.decision_path,paths,p.state_path,p.root/"premature-freeze",repo=REPO)
    path=write(p.root/"premature-freeze.json",freeze)
    with pytest.raises(ValueError,match="CAP_CHILD_NOT_TERMINAL_OR_INVALID_COUNTERS"):
        r.run_job(path,freeze["success_receipt"])
    assert not Path(freeze["success_receipt"]).exists()


def test_initial_protocol_cannot_point_calibration_at_test_path(planned):
    p=planned;protocol=copy.deepcopy(p.base.protocol)
    protocol["validation_references"]["text"]["path"]=str(p.root/"test.jsonl")
    protocol["protocol_hash"]=r.canonical_hash({k:v for k,v in protocol.items() if k!="protocol_hash"})
    ref=r._reference(write(p.root/"bad-protocol.json",protocol))
    config=copy.deepcopy(p.decision);config["data"]["protocol"]=ref
    config["config_hash"]=r.canonical_hash({k:v for k,v in config.items() if k!="config_hash"})
    with pytest.raises(ValueError,match="ONLY_EXACT_FROZEN_VALIDATION_FILE_ALLOWED"):
        r.validate_config(config,repo=REPO)


def test_bumped_science_override_rejected_even_with_new_self_hash(planned):
    p=planned;config=r.build_bumped_config(p.decision_path,"T1","base",p.root/"override",repo=REPO)
    for field,value in [("eval_context",2048),("decode",{"do_sample":True}),("seed",1),("calibration_round","third")]:
        changed=copy.deepcopy(config);changed[field]=value
        changed["config_hash"]=r.canonical_hash({k:v for k,v in changed.items() if k!="config_hash"})
        with pytest.raises(ValueError,match="CAP_REFINEMENT_SCIENTIFIC_CONFIG_CHANGED"):
            r.validate_config(changed,repo=REPO)
