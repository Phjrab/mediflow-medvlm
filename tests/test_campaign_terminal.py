"""Finite CPU fixtures: the foreground client never owns queue/worker state."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from mediflow_medvlm import campaign_terminal as t

REPO=Path(__file__).resolve().parents[1]


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+"\n")
    return path


def ref(path): return {"path":str(path),"sha256":t.file_hash(path)}


class Fixture:
    def __init__(self,tmp):
        self.root=tmp/"campaign"; self.root.mkdir(); self.data=tmp/"data"; self.data.mkdir()
        self.operator=tmp/"guard.py"; self.operator.write_text("# reviewed CPU fixture guard\n")
        self.telemetry=tmp/"forge"; self.contract=tmp/"contract.md"; self.contract.write_text("scalar telemetry only\n")
        self.env={"HF_HUB_OFFLINE":"1","MEDVLM_TELEMETRY_ROOT":str(self.telemetry),"MEDVLM_TELEMETRY_CAMPAIGN":self.root.name}
        self.source=self.root/"snapshots/worker.py"; self.source.parent.mkdir(); self.source.write_text("# CPU-only fixture\n")
        self.jobs=[]; self.records={}; self.bindings={}; self.mapping={}; self.ids=[]
        self.add("v6-report","ALL_V6","report")
        for condition in t.CONDITIONS:
            self.mapping[condition]={}
            for kind in ("train","evaluation"):
                key=f"v7-{condition}-{kind}"; self.add(key,condition,kind)
                self.mapping[condition][kind]=[key]; self.ids.append(key)
        self.add("v7-report","ALL_V7","report"); self.ids.append("v7-report")
        self.manifest={"campaign_id":self.root.name,"campaign_root":str(self.root),"jobs":self.jobs}
        self.manifest_path=save(self.root/"manifests/current.json",self.manifest)
        self.state={"campaign_id":self.root.name,"queue_owner":"supervisor","manifest_path":str(self.manifest_path),
            "manifest_hash":t.digest(self.manifest),"jobs":self.records,"manifest_revisions":[],"finalization":"NOT_DONE"}
        self.success("v6-report"); self.persist()
        self.parent=save(tmp/"parent-deployment.json",{"campaign_root":str(self.root),"data_root":str(self.data),"env":self.env,"cwd":str(tmp)})
        self.ready=save(tmp/"ready.json",{"status":"READY","cpu_tests_passed":True,"v7_workers_implemented":True,
            "campaign_root":str(self.root),"manifest_hash":self.state["manifest_hash"],"job_bindings_hash":t.digest(self.bindings)})
        self.token="manual-token-value"
        self.handoff_path=tmp/"handoff.json"
        self.handoff={"schema_version":t.SCHEMA,"campaign_root":str(self.root),"data_root":str(self.data),"campaign_id":self.root.name,
            "manifest":{**ref(self.manifest_path),"manifest_hash":self.state["manifest_hash"]},"model_id":t.MODEL_ID,"conditions":t.CONDITIONS,
            "condition_jobs":self.mapping,"report_job_ids":["v7-report"],"allowed_job_ids":self.ids,"v6_report_job_id":"v6-report",
            "readiness":[ref(self.ready)],"manual_scope_token_sha256":hashlib.sha256(self.token.encode()).hexdigest(),
            "guard_operator":ref(self.operator),"parent_deployment":ref(self.parent),"job_bindings":self.bindings,
            "telemetry":{"root":str(self.telemetry),"campaign_id":self.root.name,"forge_contract":ref(self.contract)}}
        self.route_path=tmp/"deployment.json"; self.write_handoff()
        self.calls=[]; self.output=[]; self.interrupt=None; self.live=False

    def add(self,key,condition,kind):
        config={"job_id":key,"run_id":key,"condition":condition,"model_id":t.MODEL_ID,"model":{"model_id":t.MODEL_ID,"revision":"pinned"},
            "data":{"immutable":"fixture"},"output_dir":str(self.root/"outputs"/key),"source_files":{"worker.py":t.file_hash(self.source)}}
        config.update(model_hash=t.digest(config["model"]),data_hash=t.digest(config["data"]),source_hash=t.digest(config["source_files"]))
        config["config_hash"]=t.digest(config)
        path=save(self.root/"configs"/(key+".json"),config)
        job={"job_id":key,"run_id":key,"cell_id":condition,"phase":"v7_"+kind if key.startswith("v7-") else "report",
            "model_id":t.MODEL_ID,"argv":[sys.executable,"-m","fixture","--config",str(path)],"implementation_pending":False,
            "success_receipt":str(self.root/"receipts"/(key+".json")),**{field:config[field] for field in t.HASH_FIELDS}}
        self.jobs.append(job); self.records[key]={"status":"PENDING","attempts":0,"failure_count":0,"recovery_attempts":0}
        if key.startswith("v7-"): self.bindings[key]={"job_hash":t.digest(job),"config":ref(path),"condition":condition,"kind":kind}

    def write_handoff(self,scope="terminal"):
        save(self.handoff_path,self.handoff)
        prefix=[sys.executable,str(self.operator),"--deployment",str(self.parent),"--deployment-sha256",t.file_hash(self.parent),"--scope",scope]
        if scope=="terminal": prefix += ["--handoff",str(self.handoff_path),"--handoff-sha256",t.file_hash(self.handoff_path)]
        self.route={"campaign_root":str(self.root),"data_root":str(self.data),"operator":str(self.operator),"cwd":self.parent.parent.as_posix(),
            "env":self.env,"status_argv":prefix+["status"],"tick_argv":prefix+["tick"]}
        save(self.route_path,self.route)

    def persist(self): save(self.root/"state.json",self.state)

    def success(self,key):
        job=next(job for job in self.jobs if job["job_id"]==key)
        artifact=save(self.root/"results"/(key+".json"),{"audited":True})
        receipt={"status":"SUCCEEDED",**{k:job[k] for k in ("job_id","run_id",*t.HASH_FIELDS)},"artifacts":[ref(artifact)]}
        path=save(Path(job["success_receipt"]),receipt)
        self.records[key].update(status="SUCCEEDED",exit_code=0,verification={"receipt":str(path),"receipt_sha256":t.file_hash(path),"artifact_count":1})

    def snapshot(self,env):
        return {**copy.deepcopy(self.state),"terminal_scope_v7":self.route["status_argv"][self.route["status_argv"].index("--scope")+1]=="terminal",
            "manual_handoff_verified":env.get(t.TOKEN_ENV)==self.token,"expected_manifest_hash":self.state["manifest_hash"],
            "v7_manual_start_required":True,"guard_operator_sha256":t.file_hash(self.operator)}

    def invoke(self,argv,env,**kwargs):
        action=argv[-1]; self.calls.append((action,copy.deepcopy(env),copy.deepcopy(argv)))
        assert argv==self.route[action+"_argv"] and kwargs["cwd"]==self.route["cwd"]
        if self.interrupt==action: raise KeyboardInterrupt
        if action=="tick":
            assert env.get(t.TOKEN_ENV)==self.token
            assert self.records["v6-report"]["status"]=="SUCCEEDED"
            key=next((key for key in self.ids if self.records[key]["status"]!="SUCCEEDED"),None)
            if key: self.success(key); self.persist()
        return self.snapshot(env)

    def client(self,*,handoff=True,sleep=None):
        return t.TerminalClient(self.route_path,t.file_hash(self.route_path),handoff=self.handoff_path if handoff else None,
            handoff_sha256=t.file_hash(self.handoff_path) if handoff else None,transport=SimpleNamespace(invoke=self.invoke),
            sleep=sleep or (lambda seconds:None),is_alive=lambda identity:bool(identity and self.live),emit=self.output.append,interval=1.)


@pytest.fixture
def f(tmp_path): return Fixture(tmp_path)


def test_import_cpu_only_and_script_default_read_only(f):
    result=subprocess.run([sys.executable,"-c","import sys; from mediflow_medvlm import campaign_terminal; assert 'torch' not in sys.modules"],cwd=REPO,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert f.client().execute()==0
    assert [action for action,_,_ in f.calls]==["status"]
    assert all(t.TOKEN_ENV not in env for _,env,_ in f.calls)
    assert f.output[-1]["v7_completed"] is False


def test_preflight_and_watch_never_dispatch_and_preserve_state(f):
    before=(f.root/"state.json").read_bytes()
    assert f.client().execute("preflight")==0
    assert f.client().execute("watch",max_updates=3)==3
    assert [a for a,_,_ in f.calls]==["status","status"]
    assert (f.root/"state.json").read_bytes()==before
    assert not any(item.get("v7_completed") for item in f.output)


def test_no_dispatch_before_manual_token_and_no_v6_dispatch(f):
    client=f.client()
    for token in (None,"wrong"):
        with pytest.raises(t.TerminalError,match="EXPLICIT_TERMINAL_SCOPE_TOKEN"): client.execute("run",scope_token=token)
    assert not f.calls
    f.records["v6-report"]["status"]="RUNNING"; f.persist()
    assert f.client().execute("run",scope_token=f.token)==2
    assert [a for a,_,_ in f.calls]==["status"]
    assert "V6_NOT_TERMINAL:v6-report" in f.output[-1]["reasons"]


def test_run_four_conditions_train_eval_then_report_receipts_sequentially(f):
    assert f.client().execute("run",scope_token=f.token,max_updates=20)==0
    assert [a for a,_,_ in f.calls]==["status"]+["tick"]*9
    assert all(env[t.TOKEN_ENV]==f.token for _,env,_ in f.calls)
    assert f.output[-1]["v7_completed"] is True and f.output[-1]["v7_status"]=="SUCCEEDED"
    assert all(item["train"]=={"scheduled":1,"succeeded":1} and item["evaluation"]=={"scheduled":1,"succeeded":1} for item in f.output[-1]["conditions"].values())
    assert f.output[-1]["report_jobs"]=={"v7-report":"SUCCEEDED"}
    f.calls.clear()
    assert f.client().execute("run",scope_token=f.token)==0
    assert [a for a,_,_ in f.calls]==["status"]


def test_existing_success_artifact_tampering_blocks_restart(f):
    f.success(f.ids[0]); f.persist()
    receipt=t.read_json(next(j for j in f.jobs if j["job_id"]==f.ids[0])["success_receipt"])
    Path(receipt["artifacts"][0]["path"]).write_text("tampered medical PRIVATE")
    assert f.client().execute("run",scope_token=f.token)==2
    assert [a for a,_,_ in f.calls]==["status"]
    assert "V7_SUCCESS_VERIFICATION_FAILED:"+f.ids[0] in f.output[-1]["reasons"]
    assert "medical PRIVATE" not in json.dumps(f.output)


def test_current_unimplemented_draft_reports_all_blockers_without_success(f):
    for condition in t.CONDITIONS: f.handoff["condition_jobs"][condition]["evaluation"]=[]
    f.handoff["allowed_job_ids"]=[f.mapping[c]["train"][0] for c in t.CONDITIONS]+["v7-report"]
    f.handoff["job_bindings"]={}; f.handoff["manual_scope_token_sha256"]=""
    for key in f.ids:
        f.records[key]["status"]="IMPLEMENTATION_PENDING"
        next(j for j in f.jobs if j["job_id"]==key)["implementation_pending"]=True
    save(f.manifest_path,f.manifest); f.state["manifest_hash"]=t.digest(f.manifest); f.persist()
    f.handoff["manifest"]={**ref(f.manifest_path),"manifest_hash":f.state["manifest_hash"]}
    save(f.ready,{"status":"BLOCKED_PREREQUISITE","cpu_tests_passed":False,"v7_workers_implemented":False})
    f.handoff["readiness"]=[ref(f.ready)]; f.write_handoff(scope="auto")
    assert f.client().execute("preflight")==2
    output=f.output[-1]
    assert not output["v7_completed"] and output["v7_status"]=="BLOCKED_PREREQUISITE"
    assert "V7_JOB_BINDINGS_MISSING" in output["reasons"] and "MANUAL_TOKEN_NOT_ISSUED" in output["reasons"]
    assert all("V7_STAGE_MISSING:"+condition+":evaluation" in output["reasons"] for condition in t.CONDITIONS)
    assert [a for a,_,_ in f.calls]==["status"]
    f.calls.clear(); f.output.clear()
    client=f.client()
    assert client.execute("watch",max_updates=2)==3
    assert [a for a,_,_ in f.calls]==["status"]
    views=[value for value in f.output if value.get("event")=="STATUS"]
    assert len(views)==2 and all(not value["v7_completed"] and value["reasons"] for value in views)


@pytest.mark.parametrize("status",["FAILED","ANALYSIS_REQUIRED","BLOCKED_PREREQUISITE","IMPLEMENTATION_PENDING"])
def test_ready_failures_stop_with_nonzero_without_tick(f,status):
    f.records[f.ids[0]]["status"]=status; f.persist()
    assert f.client().execute("run",scope_token=f.token)==2
    assert [a for a,_,_ in f.calls]==["status"]


def test_readiness_config_parent_env_route_and_manifest_tampering(f):
    client=f.client()
    f.operator.write_text("# tampered\n")
    with pytest.raises(t.TerminalError,match="GUARD_OPERATOR_CHANGED"): client.execute()
    assert not f.calls


def test_bound_manifest_and_job_config_science_tamper_fail_closed(f):
    key=f.ids[0]; path=Path(f.bindings[key]["config"]["path"])
    config=t.read_json(path); config["condition"]="R1"; save(path,config)
    assert f.client().execute("run",scope_token=f.token)==2
    assert [a for a,_,_ in f.calls]==["status"]
    assert "V7_CONFIG_BINDING_CHANGED:"+key in f.output[-1]["reasons"]


@pytest.mark.parametrize("action",["status","tick"])
def test_interrupt_detaches_without_signalling_and_reports_inflight_tick_honestly(f,action):
    f.interrupt=action
    assert f.client().execute("run",scope_token=f.token)==130
    event=f.output[-1]
    assert event["event"]=="DETACHED" and event["worker_signalled"] is False and event["further_tick_requests"] is False
    assert event["inflight_atomic_tick_may_finish"] is (action=="tick")
    assert [a for a,_,_ in f.calls]==(["status"] if action=="status" else ["status","tick"])
    assert f.token not in json.dumps(f.output)


def test_live_worker_monitoring_reads_local_state_without_status_poll_or_tick(f):
    key=f.ids[0]; f.records[key].update(status="RUNNING",attempt_dir=str(f.root/"attempts"/key/"0001"),worker_identity={"worker_pid":123},launch_token="launch")
    f.live=True; f.persist()
    assert f.client().execute("run",scope_token=f.token,max_updates=4)==3
    assert [a for a,_,_ in f.calls]==["status"]


@pytest.mark.parametrize("exit_code",[0,1])
def test_dead_worker_bound_exit_is_awaiting_verification_and_displays_row_progress(f,exit_code):
    key="v6-report"; job=next(j for j in f.jobs if j["job_id"]==key)
    native=f.root/"attempts"/key/"0001"
    f.records[key].update(status="RUNNING",attempts=1,attempt_dir=str(native),launch_token="launch",worker_identity={"worker_pid":123})
    save(native/"launch.json",{"launch_token":"launch","attempt":1,"campaign_id":f.root.name,"job":job,"job_hash":t.digest(job)})
    save(native/"exit.json",{"status":"EXITED","launch_token":"launch","exit_code":exit_code,"error":"PRIVATE ERROR"})
    save(native/"progress.json",{**{k:job[k] for k in ("job_id","run_id",*t.HASH_FIELDS)},
        "rows_completed":601,"scheduled_text_rows":1516,"technical_failures":3,"answer":"PRIVATE ANSWER"})
    f.persist(); before=(f.root/"state.json").read_bytes()
    client=f.client()
    assert client.execute("watch",max_updates=2)==3
    views=[v for v in f.output if v.get("event")=="STATUS"]
    active=views[-1]["active"][0]
    assert active["status"]=="EXITED_AWAITING_VERIFICATION" and active["native_state"]=="RUNNING"
    assert active["exit_code"]==exit_code and active["scope"]=="V6_READ_ONLY"
    assert {k:v["value"] for k,v in active["row_progress"].items()}=={"rows_completed":601,"scheduled_text_rows":1516,"technical_failures":3}
    rendered=t.render_text(views[-1])
    assert "EXITED_AWAITING_VERIFICATION native_state=RUNNING exit_code="+str(exit_code) in rendered
    assert "rows_completed=601" in rendered and "technical_failures=3" in rendered
    assert "PRIVATE" not in json.dumps(f.output)+rendered
    assert [a for a,_,_ in f.calls]==["status"] and (f.root/"state.json").read_bytes()==before
    assert not client._worker_state(f.snapshot({}),key)


@pytest.mark.parametrize("change",["exit_token","launch_job","launch_attempt","live_child","unsafe_exit_code","bad_progress"])
def test_exit_and_row_progress_display_require_current_identity(f,change):
    key=f.ids[0]; job=next(j for j in f.jobs if j["job_id"]==key); native=f.root/"attempts"/key/"0001"
    f.records[key].update(status="RUNNING",attempts=1,attempt_dir=str(native),launch_token="launch",worker_identity={"worker_pid":123})
    launch={"launch_token":"launch","attempt":1,"campaign_id":f.root.name,"job":job,"job_hash":t.digest(job)}
    exited={"status":"EXITED","launch_token":"launch","exit_code":0}
    progress={**{k:job[k] for k in ("job_id","run_id",*t.HASH_FIELDS)},"rows_completed":9,"scheduled_text_rows":10,"technical_failures":False}
    if change=="exit_token": exited["launch_token"]="forged"
    if change=="launch_job": launch["job_hash"]="forged"
    if change=="launch_attempt": launch["attempt"]=2
    if change=="live_child":
        f.live=True; save(native/"heartbeat.json",{"child_identity":{"worker_pid":234}})
    if change=="unsafe_exit_code": exited["exit_code"]="PRIVATE ERROR"
    if change=="bad_progress": progress["config_hash"]="forged"
    save(native/"launch.json",launch); save(native/"exit.json",exited); save(native/"progress.json",progress)
    client=f.client(); client._canonical("status")
    active=client.metrics(f.snapshot({}))[0]
    expected="RUNNING" if change=="live_child" else "ACTIVITY_UNVERIFIED" if change in {"exit_token","launch_job","launch_attempt"} else "EXITED_AWAITING_VERIFICATION"
    assert active["status"]==expected and active["native_state"]=="RUNNING"
    assert active["exit_code"] is None if change in {"exit_token","launch_job","launch_attempt","live_child","unsafe_exit_code"} else active["exit_code"]==0
    assert active["row_progress"]["technical_failures"]["value"] is None
    if change=="bad_progress": assert all(v["value"] is None for v in active["row_progress"].values())
    assert "PRIVATE" not in json.dumps(active) and [a for a,_,_ in f.calls]==["status"]


def test_dead_worker_without_exit_delegates_reconciliation_to_canonical_tick(f):
    key=f.ids[0]; native=f.root/"attempts"/key/"0001"
    f.records[key].update(status="RUNNING",attempts=1,attempt_dir=str(native),launch_token="launch",worker_identity={"worker_pid":123})
    f.persist(); client=f.client()
    def canonical(argv,env,**kwargs):
        if argv[-1]=="tick":
            f.calls.append(("tick",copy.deepcopy(env),copy.deepcopy(argv)))
            f.records[key]["status"]="SOURCE_REPAIR_REQUIRED"; f.persist()
            return f.snapshot(env)
        return f.invoke(argv,env,**kwargs)
    client.transport=SimpleNamespace(invoke=canonical)
    assert client.execute("run",scope_token=f.token,max_updates=3)==2
    assert [a for a,_,_ in f.calls]==["status","tick"]
    views=[v for v in f.output if v.get("event")=="STATUS"]
    assert views[0]["active"][0]["status"]=="ACTIVITY_UNVERIFIED"
    assert "V7_REPAIR_REQUIRED:"+key in views[-1]["reasons"]


def test_scalar_telemetry_exact_attempt_binding_null_reasons_and_no_private_text(f):
    key=f.ids[0]; job=next(j for j in f.jobs if j["job_id"]==key)
    native=f.root/"attempts"/key/"0001"; native.mkdir(parents=True)
    f.records[key].update(status="RUNNING",attempts=1,attempt_dir=str(native),launch_token="launch",worker_identity={"worker_pid":123})
    progress={k:job[k] for k in ("job_id","run_id",*t.HASH_FIELDS)}
    progress.update(optimizer_step=2,total_updates=10,telemetry_attempt_id="uuid",question="PRIVATE QUESTION",answer="PRIVATE ANSWER")
    save(native/"progress.json",progress)
    save(native/"launch.json",{"launch_token":"launch","attempt":1,"job":job,"created_at":"2026-11-26T00:00:00+00:00"})
    config=t.read_json(job["argv"][4]); producer=Path(config["output_dir"]+".attempt-cpu")
    save(producer/"attempt.json",{**{k:job[k] for k in ("job_id","run_id",*t.HASH_FIELDS)},
        "attempt_id":"uuid","started_at":"2026-11-26T00:00:01+00:00"})
    directory=f.telemetry/f.root.name/key/"uuid"; envelope={"schema_version":"mediflow.telemetry.v1","campaign_id":f.root.name,
        "job_id":key,"run_id":key,"attempt_id":"uuid","producer":"mediflow.campaign"}
    save(directory/"manifest.json",{**envelope,"historical":False,"data_hash":job["data_hash"]})
    step={**envelope,"provenance":"live_measured","optimizer_step":2,"total_updates":10,"train_loss":.25,"lr_used":.0002,
        "gradient_norm_before_clip":1.5,"samples_per_second":4.,"samples_processed":8,"samples_total":40,"question":"PRIVATE QUESTION"}
    (directory/"steps.jsonl").write_text(json.dumps(step)+"\n"+'{"partial":')
    resource={**envelope,"provenance":"live_measured","gpu_utilization_percent":None,"gpu_memory_used_bytes":1024,
        "gpu_memory_total_bytes":8192,"host_ram_used_bytes":2048,"host_ram_total_bytes":16384,"availability":{"gpu_utilization_percent":{"reason":"sensor_unsupported"}}}
    (directory/"resources.jsonl").write_text(json.dumps(resource)+"\n")
    f.persist(); f.live=True
    assert f.client().execute("watch",max_updates=1)==3
    view=next(item for item in f.output if item.get("event")=="STATUS")
    metrics=view["active"][0]["metrics"]
    assert metrics["loss"]["value"]==.25 and metrics["learning_rate"]["value"]==.0002
    assert metrics["gpu_percent"]=={"value":None,"source":None,"reason":"sensor_unsupported"}
    assert metrics["eta_seconds"]["value"]==8 and metrics["eta_seconds"]["source"]=="estimate_from_latest_measured_sample_rate"
    assert "PRIVATE" not in json.dumps(f.output)


def test_missing_sensors_and_metrics_are_unavailable_not_zero(f):
    f.records[f.ids[0]]["status"]="RUNNING"; f.persist()
    assert f.client().execute("watch",max_updates=1)==3
    metrics=next(item for item in f.output if item.get("event")=="STATUS")["active"][0]["metrics"]
    assert all(value["value"] is None and value["reason"] for value in metrics.values())


def test_real_transport_preserves_environment_and_never_signals_on_ctrl_c(monkeypatch):
    seen={}
    class Process:
        def wait(self,**kwargs): raise KeyboardInterrupt
        def kill(self): raise AssertionError("kill called")
        def terminate(self): raise AssertionError("terminate called")
    def spawn(argv,**kwargs): seen.update(kwargs); return Process()
    monkeypatch.setattr(t.subprocess,"Popen",spawn)
    with pytest.raises(KeyboardInterrupt): t.CanonicalTransport().invoke(["/pinned/python","/pinned/operator","tick"],{"X":"Y"},cwd="/pinned",interval=1)
    assert seen["start_new_session"] is True and seen["env"]=={"X":"Y"} and seen["cwd"]=="/pinned"


def test_main_default_status_and_controlled_errors_without_backtrace(f,capsys):
    def factory(*args,**kwargs):
        return SimpleNamespace(execute=lambda command,**extra:0 if command=="status" and extra["scope_token"] is None else 99)
    assert t.main(["--deployment",str(f.route_path),"--deployment-sha256",t.file_hash(f.route_path)],client_factory=factory)==0
    assert t.main(["--deployment",str(f.route_path),"--deployment-sha256","0"*64,"preflight"])==2
    assert "Traceback" not in capsys.readouterr().out


def test_only_pinned_runtime_environment_and_manual_token_are_passed(f,monkeypatch):
    monkeypatch.setenv("PYTHONHOME","unreviewed"); monkeypatch.setenv("CUDA_VISIBLE_DEVICES","7")
    monkeypatch.setenv("LD_PRELOAD","unreviewed"); monkeypatch.setenv("OMP_NUM_THREADS","999")
    monkeypatch.setenv(t.TOKEN_ENV,"ambient-token")
    client=f.client(); assert client.execute("preflight")==0
    env=f.calls[0][1]
    assert all(key not in env for key in ("PYTHONHOME","CUDA_VISIBLE_DEVICES","LD_PRELOAD","OMP_NUM_THREADS",t.TOKEN_ENV))
    assert all(env[key]==value for key,value in f.env.items())


@pytest.mark.parametrize("mode",["missing","stale","ambiguous"])
def test_explicit_telemetry_uuid_needs_unique_current_producer_proof(f,mode):
    key=f.ids[0]; job=next(job for job in f.jobs if job["job_id"]==key); native=f.root/"attempts"/key/"0001"
    record={"attempts":1,"launch_token":"token"}; progress={"telemetry_attempt_id":"uuid"}
    save(native/"launch.json",{"attempt":1,"launch_token":"token","job":job,"created_at":"2026-11-26T01:00:00+00:00"})
    config=t.read_json(job["argv"][4]); item={**{k:job[k] for k in ("job_id","run_id",*t.HASH_FIELDS)},"attempt_id":"uuid"}
    if mode!="missing":
        save(Path(config["output_dir"]+".attempt-a")/"attempt.json",{**item,"started_at":"2026-11-26T00:00:00+00:00" if mode=="stale" else "2026-11-26T01:00:01+00:00"})
    if mode=="ambiguous":
        save(Path(config["output_dir"]+".attempt-b")/"attempt.json",{**item,"started_at":"2026-11-26T01:00:02+00:00"})
    steps,resources,reason=f.client()._telemetry(job,record,progress,native)
    assert steps==resources=={} and reason in {"current_attempt_not_bound","ambiguous_attempt_identity"}


@pytest.mark.parametrize("interval",[0,.5,61,float("nan"),True])
def test_interval_constraints(f,interval):
    with pytest.raises(t.TerminalError,match="INTERVAL"):
        t.TerminalClient(f.route_path,t.file_hash(f.route_path),interval=interval)


def test_unknown_canonical_readiness_rejection_is_blocked_without_echo(f):
    client=f.client(); snapshot=f.snapshot({}); client._bind_snapshot(snapshot)
    snapshot["terminal_readiness_reasons"]=["PRIVATE PATIENT third party exception"]
    view=client.view(snapshot)
    assert "CANONICAL_PREFLIGHT_REJECTED" in view["reasons"] and "PRIVATE" not in json.dumps(view)
