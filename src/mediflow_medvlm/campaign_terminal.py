"""Foreground V7 terminal client for the saved, sole canonical dispatcher.

This module never imports a trainer, writes queue state, acquires GPU leases,
samples GPU sensors, or signals a worker. Only explicit ``run`` may invoke the
exact pinned deployment tick. All other commands are read-only. V7 executables
and readiness are external prerequisites; this client does not invent them.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

SCHEMA = "mediflow.v7-terminal-handoff.v1"
CONDITIONS = ["P0", "P1", "R0", "R1"]
MODEL_ID = "Qwen/Qwen3-1.7B"
TOKEN_ENV = "MEDVLM_V7_SCOPE_TOKEN"
HASH_FIELDS = ("source_hash", "config_hash", "data_hash", "model_hash")
BAD_STATES = {"FAILED", "BLOCKED_PREREQUISITE", "ANALYSIS_REQUIRED", "SOURCE_REPAIR_REQUIRED"}
TERMINAL = {"SUCCEEDED", "FAILED", "BLOCKED_PREREQUISITE"}
SAFE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
MAX_JSON = 8 * 1024 * 1024
TAIL_BYTES = 2 * 1024 * 1024
HOST_ENV_FIELDS = ("PATH","HOME","USER","LOGNAME","SHELL","LANG","LC_ALL","LC_CTYPE","TERM",
                   "XDG_RUNTIME_DIR","DBUS_SESSION_BUS_ADDRESS","TMPDIR")
METRIC_FIELDS = {
    "step":"optimizer_step", "total":"total_updates", "loss":"train_loss", "learning_rate":"lr_used",
    "gradient_norm":"gradient_norm_before_clip", "samples_per_second":"samples_per_second",
    "gpu_percent":"gpu_utilization_percent", "gpu_used_bytes":"gpu_memory_used_bytes",
    "gpu_total_bytes":"gpu_memory_total_bytes", "ram_used_bytes":"host_ram_used_bytes",
    "ram_total_bytes":"host_ram_total_bytes", "eta_seconds":"eta_seconds",
}
ROW_PROGRESS_FIELDS = ("rows_completed", "scheduled_text_rows", "technical_failures")


class TerminalError(ValueError):
    """Fixed safe diagnostics, never third-party exception/medical text."""


def require(condition, code):
    if not condition: raise TerminalError(code)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",", ":")).encode()).hexdigest()


def absolute(path):
    path=Path(path)
    require(path.is_absolute() and ".." not in path.parts and not any(p.is_symlink() for p in (path,*path.parents)), "UNSAFE_PATH")
    return path


def file_hash(path):
    h=hashlib.sha256()
    with absolute(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""): h.update(block)
    return h.hexdigest()


def read_json(path):
    path=absolute(path); require(path.stat().st_size<=MAX_JSON,"JSON_TOO_LARGE")
    with path.open("rb") as stream: value=json.load(stream)
    require(isinstance(value,dict),"JSON_OBJECT_REQUIRED")
    return value


def pinned(reference):
    require(isinstance(reference,dict) and {"path","sha256"}<=set(reference),"INVALID_REFERENCE")
    require(file_hash(reference["path"])==reference["sha256"],"PINNED_FILE_CHANGED")
    return read_json(reference["path"])


def v7_job(job):
    return (str(job.get("job_id","")).lower().startswith("v7-") or job.get("cell_id") in CONDITIONS+["ALL_V7"]
            or "v7" in str(job.get("phase","")).lower())


def _finite(value):
    return type(value) in (int,float) and math.isfinite(value)


def _metric(value,source,reason="not_emitted"):
    if reason not in {"not_emitted","not_measured","sensor_unsupported","nvidia_smi_unavailable","proc_unavailable",
                      "cpu_interval_not_measured","nvidia_smi_failed","visible_device_unresolved"}: reason="not_measured"
    return {"value":value if _finite(value) else None,"source":source if _finite(value) else None,
            "reason":None if _finite(value) else reason}


def process_alive(identity):
    """PID/start/boot identity check; unknown permissions conservatively mean live."""
    if not identity or not isinstance(identity,dict) or not identity.get("worker_pid"): return False
    try:
        pid=int(identity["worker_pid"]); text=Path(f"/proc/{pid}/stat").read_text()
        fields=text[text.rfind(")")+2:].split()
        if fields[0]=="Z": return False
        boot=Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        return str(fields[19])==str(identity.get("process_start_time")) and boot==identity.get("boot_identity")
    except (FileNotFoundError,ProcessLookupError,ValueError,KeyError): return False
    except (PermissionError,OSError): return True


class CanonicalTransport:
    """One foreground canonical call, with interrupt-safe detached ownership.

    The short-lived operator gets its own session and real temporary-file FDs.
    Ctrl-C never calls terminate/kill or closes a worker's inherited lease.
    The canonical operator may finish its already-requested atomic operation.
    """
    def invoke(self,argv,env,*,cwd=None,on_wait=None,interval=5.):
        with tempfile.TemporaryFile() as stdout,tempfile.TemporaryFile() as stderr:
            process=subprocess.Popen(argv,cwd=cwd,env=env,stdin=subprocess.DEVNULL,
                                     stdout=stdout,stderr=stderr,start_new_session=True,close_fds=True)
            while True:
                try:
                    code=process.wait(timeout=min(interval,5.)); break
                except subprocess.TimeoutExpired:
                    if on_wait is not None: on_wait()
                # KeyboardInterrupt intentionally propagates without signalling
                # the detached operator or its detached native worker.
            require(code==0,"CANONICAL_OPERATOR_FAILED")
            stdout.seek(0); raw=stdout.read(MAX_JSON+1); require(len(raw)<=MAX_JSON,"CANONICAL_RESPONSE_TOO_LARGE")
            try: value=json.loads(raw)
            except (ValueError,UnicodeDecodeError): raise TerminalError("CANONICAL_RESPONSE_INVALID") from None
            require(isinstance(value,dict),"CANONICAL_RESPONSE_INVALID")
            return value


class TerminalClient:
    def __init__(self,deployment,deployment_sha256,*,handoff=None,handoff_sha256=None,
                 transport=None,sleep=None,is_alive=None,emit=None,interval=5.):
        require(_finite(interval) and 1<=interval<=60,"INTERVAL_MUST_BE_1_TO_60_SECONDS")
        require((handoff is None)==(handoff_sha256 is None),"HANDOFF_AND_SHA_REQUIRED_TOGETHER")
        self.deployment_ref={"path":str(absolute(deployment)),"sha256":deployment_sha256}
        self.route=pinned(self.deployment_ref)
        self.handoff_ref={"path":str(absolute(handoff)),"sha256":handoff_sha256} if handoff is not None else None
        self.handoff=pinned(self.handoff_ref) if self.handoff_ref else None
        self.transport=transport or CanonicalTransport(); self.sleep=sleep or time.sleep
        self.is_alive=is_alive or process_alive; self.emit=emit or (lambda value:print(json.dumps(value,ensure_ascii=False),flush=True))
        self.interval=interval; self.snapshot=None; self.manifest=None; self.scope_token=None
        self.success_cache={}; self.inflight_action=None
        self.root=absolute(self.route["campaign_root"])
        self._validate_route()

    def _validate_route(self):
        status,tick=self.route.get("status_argv"),self.route.get("tick_argv")
        require(isinstance(status,list) and isinstance(tick,list) and len(status)>=3 and status[:-1]==tick[:-1]
                and status[-1]=="status" and tick[-1]=="tick" and all(isinstance(x,str) for x in status+tick),"EXACT_CANONICAL_ROUTE_REQUIRED")
        # The interpreter is an existing saved executable, not a reconstructed
        # command or shell string. Its saved spelling is retained verbatim.
        require(Path(status[0]).is_absolute() and Path(status[0]).is_file() and Path(status[1]).is_absolute(),"ABSOLUTE_CANONICAL_EXECUTABLE_REQUIRED")
        self.operator=absolute(self.route.get("operator",status[1]))
        require(str(self.operator)==status[1],"SAVED_OPERATOR_MISMATCH")
        self.operator_hash=file_hash(self.operator)
        require(isinstance(self.route.get("env",{}),dict) and all(isinstance(k,str) and isinstance(v,str) for k,v in self.route.get("env",{}).items())
                and TOKEN_ENV not in self.route.get("env",{}),"SAVED_ENVIRONMENT_OR_TOKEN_INVALID")
        if self.handoff is not None:
            h=self.handoff
            require(h.get("schema_version")==SCHEMA and h.get("campaign_root")==str(self.root)
                    and h.get("campaign_id")==self.root.name and h.get("data_root")==self.route.get("data_root")
                    and h.get("conditions")==CONDITIONS and h.get("model_id")==MODEL_ID,"HANDOFF_IDENTITY_CHANGED")
            require(h.get("guard_operator")=={"path":str(self.operator),"sha256":self.operator_hash},"GUARD_OPERATOR_CHANGED")
            parent=pinned(h["parent_deployment"])
            require(parent.get("campaign_root")==str(self.root) and parent.get("data_root")==self.route.get("data_root")
                    and parent.get("env",{})==self.route.get("env",{}) and parent.get("cwd")==self.route.get("cwd"),"SAVED_PARENT_ENVIRONMENT_CHANGED")
            argv=status[:-1]
            require(argv.count("--scope")==1 and argv[argv.index("--scope")+1] in {"auto","terminal"},"CANONICAL_GUARD_SCOPE_REQUIRED")
            pairs=[("--deployment",h["parent_deployment"]["path"]),("--deployment-sha256",h["parent_deployment"]["sha256"])]
            if argv[argv.index("--scope")+1]=="terminal":
                pairs.extend([("--handoff",self.handoff_ref["path"]),("--handoff-sha256",self.handoff_ref["sha256"])])
            for option,value in pairs:
                require(argv.count(option)==1 and argv[argv.index(option)+1]==value,"TERMINAL_SAVED_SCOPE_ROUTE_REQUIRED")

    def _recheck_pins(self):
        require(file_hash(self.deployment_ref["path"])==self.deployment_ref["sha256"],"DEPLOYMENT_CHANGED")
        require(file_hash(self.operator)==self.operator_hash,"GUARD_OPERATOR_CHANGED")
        if self.handoff_ref: pinned(self.handoff_ref)

    def _env(self):
        env={key:os.environ[key] for key in HOST_ENV_FIELDS if key in os.environ}
        env.update(self.route.get("env",{}))
        if self.scope_token is not None: env[TOKEN_ENV]=self.scope_token
        return env

    def _canonical(self,action):
        self._recheck_pins()
        self.emit({"event":"CANONICAL_VALIDATION_RUNNING","action":action,"eta_seconds":None,"reason":"validation_duration_not_measured"})
        self.inflight_action=action
        result=self.transport.invoke(self.route[action+"_argv"],self._env(),cwd=self.route.get("cwd"),
            on_wait=self._during_validation,interval=self.interval)
        self.inflight_action=None
        self._bind_snapshot(result); return result

    def _during_validation(self):
        if self.snapshot is not None:
            try: self.emit(self.view(self.local_snapshot(),event="CANONICAL_VALIDATION_RUNNING"))
            except (TerminalError,OSError,ValueError,KeyError):
                self.emit({"event":"CANONICAL_VALIDATION_RUNNING","eta_seconds":None,"reason":"local_snapshot_unavailable"})

    def _bind_snapshot(self,snapshot):
        require(snapshot.get("campaign_id")==self.root.name and snapshot.get("queue_owner")=="supervisor"
                and isinstance(snapshot.get("jobs"),dict),"CANONICAL_NAMESPACE_OR_OWNER_CHANGED")
        path=absolute(snapshot["manifest_path"])
        require(path.is_relative_to(self.root/"manifests"),"MANIFEST_OUTSIDE_CAMPAIGN")
        manifest=read_json(path)
        require(digest(manifest)==snapshot.get("manifest_hash") and manifest.get("campaign_id")==self.root.name
                and (manifest.get("campaign_root",str(self.root))==str(self.root)),"MANIFEST_CHANGED")
        require(set(snapshot["jobs"])=={j["job_id"] for j in manifest["jobs"]}
                and all(isinstance(j.get("job_id"),str) and SAFE.fullmatch(j["job_id"]) for j in manifest["jobs"]),"JOB_SET_CHANGED")
        if self.handoff:
            bound=pinned(self.handoff["manifest"])
            require(digest(bound)==digest(manifest)==self.handoff["manifest"]["manifest_hash"],"HANDOFF_MANIFEST_CHANGED")
        self.manifest,self.snapshot=manifest,snapshot

    def local_snapshot(self):
        """Fast read-only display; never call expensive canonical status per poll."""
        require(self.snapshot is not None,"CANONICAL_STATUS_NOT_VALIDATED")
        self._recheck_pins(); state=read_json(self.root/"state.json")
        require(state.get("manifest_hash")==self.snapshot["manifest_hash"] and state.get("manifest_path")==self.snapshot["manifest_path"]
                and state.get("campaign_id")==self.root.name and state.get("queue_owner")=="supervisor"
                and set(state.get("jobs",{}))==set(self.snapshot["jobs"]),"STATE_REVALIDATION_REQUIRED")
        # Guard authorization belongs to the last canonical check, never to
        # producer progress. The guard rechecks everything atomically at tick.
        return {**state,**{k:self.snapshot.get(k) for k in ("terminal_scope_v7","manual_handoff_verified","expected_manifest_hash","v7_manual_start_required","guard_operator_sha256")}}

    def _verify_success(self,job,record):
        require(record.get("status")=="SUCCEEDED" and record.get("exit_code")==0,"SUCCESS_VERIFICATION_MISSING")
        verification=record.get("verification",{}); path=absolute(job["success_receipt"])
        require(path.is_relative_to(self.root) and verification.get("receipt")==str(path)
                and verification.get("receipt_sha256")==file_hash(path),"SUCCESS_RECEIPT_CHANGED")
        receipt=read_json(path)
        require(receipt.get("status")=="SUCCEEDED" and all(receipt.get(k)==job.get(k) for k in ("job_id","run_id",*HASH_FIELDS))
                and bool(receipt.get("artifacts")) and verification.get("artifact_count")==len(receipt["artifacts"]),"SUCCESS_RECEIPT_IDENTITY_CHANGED")
        signature=[]
        for artifact in receipt["artifacts"]:
            stat=absolute(artifact["path"]).stat(); signature.append((artifact["path"],stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns))
        cache_key=(digest(job),verification["receipt_sha256"])
        if self.success_cache.get(cache_key)==signature: return
        for artifact in receipt["artifacts"]:
            require(absolute(artifact["path"]).is_relative_to(self.root) and file_hash(artifact["path"])==artifact["sha256"],"SUCCESS_ARTIFACT_CHANGED")
        self.success_cache[cache_key]=signature

    def blockers(self,snapshot,*,for_run=False):
        jobs={j["job_id"]:j for j in self.manifest["jobs"]}; records=snapshot["jobs"]; reasons=[]
        if not self.handoff:
            reasons.append("TERMINAL_HANDOFF_MISSING")
            for key,job in jobs.items():
                if v7_job(job) and (job.get("implementation_pending") or records[key]["status"]=="IMPLEMENTATION_PENDING"):
                    reasons.append("V7_IMPLEMENTATION_PENDING:"+key)
            return reasons
        h=self.handoff; allowed=[]
        for condition in CONDITIONS:
            entry=h.get("condition_jobs",{}).get(condition,{})
            for kind in ("train","evaluation"):
                group=entry.get(kind,[])
                if not isinstance(group,list) or not group: reasons.append("V7_STAGE_MISSING:"+condition+":"+kind); continue
                if not all(isinstance(key,str) and key in jobs for key in group): reasons.append("V7_STAGE_JOB_MISSING:"+condition+":"+kind); continue
                allowed.extend(group)
        reports=h.get("report_job_ids",[])
        if not isinstance(reports,list) or not reports: reasons.append("V7_REPORT_MISSING")
        else: allowed.extend(reports)
        if (allowed!=h.get("allowed_job_ids") or len(set(allowed))!=len(allowed)
                or set(allowed)!={key for key,job in jobs.items() if v7_job(job)}): reasons.append("V7_SCOPE_INCOMPLETE")
        bindings=h.get("job_bindings",{})
        if set(bindings)!=set(h.get("allowed_job_ids",[])): reasons.append("V7_JOB_BINDINGS_MISSING")
        for key in h.get("allowed_job_ids",[]):
            if key not in jobs: reasons.append("V7_STAGE_JOB_MISSING:"+key); continue
            job,record=jobs[key],records[key]
            if not v7_job(job): reasons.append("V6_DISPATCH_FORBIDDEN")
            if job.get("implementation_pending") or not job.get("argv") or record.get("status")=="IMPLEMENTATION_PENDING": reasons.append("V7_IMPLEMENTATION_PENDING:"+key)
            if record.get("status") in BAD_STATES: reasons.append("V7_REPAIR_REQUIRED:"+key)
            if key in bindings:
                binding=bindings[key]
                try:
                    require(binding.get("job_hash")==digest(job),"V7_JOB_BINDING_CHANGED")
                    config=pinned(binding["config"]); argv=job.get("argv",[])
                    require(argv.count("--config")==1 and argv[argv.index("--config")+1]==binding["config"]["path"]
                            and config.get("config_hash")==digest({k:v for k,v in config.items() if k!="config_hash"})
                            and all(config.get(k)==job.get(k) for k in ("job_id","run_id",*HASH_FIELDS))
                            and config.get("data_hash")==digest(config.get("data")) and config.get("model_hash")==digest(config.get("model"))
                            and config.get("source_hash")==digest(config.get("source_files")),"V7_CONFIG_BINDING_CHANGED")
                    require(binding.get("kind") in {"train","evaluation","report"} and binding.get("condition") in CONDITIONS+["ALL_V7"]
                            and config.get("condition")==binding["condition"]
                            and (binding["kind"]=="report" or config.get("model_id")==MODEL_ID),"V7_CONDITION_BINDING_CHANGED")
                except (TerminalError,OSError,ValueError,KeyError,TypeError): reasons.append("V7_CONFIG_BINDING_CHANGED:"+key)
            if record.get("status")=="SUCCEEDED":
                try: self._verify_success(job,record)
                except (TerminalError,OSError,ValueError,KeyError,TypeError): reasons.append("V7_SUCCESS_VERIFICATION_FAILED:"+key)
        for key,record in records.items():
            if key not in h.get("allowed_job_ids",[]) and record.get("status") not in TERMINAL: reasons.append("V6_NOT_TERMINAL:"+key)
        report_id=h.get("v6_report_job_id")
        if report_id!="v6-report" or report_id not in jobs or records[report_id].get("status")!="SUCCEEDED": reasons.append("V6_REPORT_NOT_VERIFIED")
        else:
            try: self._verify_success(jobs[report_id],records[report_id])
            except (TerminalError,OSError,ValueError,KeyError,TypeError): reasons.append("V6_REPORT_NOT_VERIFIED")
        if not h.get("readiness"): reasons.append("V7_READINESS_MISSING")
        for ref in h.get("readiness",[]):
            try:
                evidence=pinned(ref)
                require(evidence.get("status")=="READY" and evidence.get("cpu_tests_passed") is True and evidence.get("v7_workers_implemented") is True
                        and evidence.get("campaign_root")==str(self.root) and evidence.get("manifest_hash")==snapshot["manifest_hash"]
                        and evidence.get("job_bindings_hash")==digest(bindings),"V7_READINESS_INCOMPLETE")
            except (TerminalError,OSError,ValueError,KeyError,TypeError): reasons.append("V7_READINESS_INCOMPLETE")
        token_hash=h.get("manual_scope_token_sha256")
        if not isinstance(token_hash,str) or not re.fullmatch(r"[0-9a-f]{64}",token_hash): reasons.append("MANUAL_TOKEN_NOT_ISSUED")
        if (snapshot.get("terminal_scope_v7") is not True or snapshot.get("expected_manifest_hash")!=snapshot["manifest_hash"]
                or snapshot.get("guard_operator_sha256")!=self.operator_hash): reasons.append("CANONICAL_TERMINAL_GUARD_REQUIRED")
        diagnostics=snapshot.get("terminal_readiness_reasons",[])
        if diagnostics and diagnostics!=["explicit foreground terminal scope token required"]:
            reasons.append("CANONICAL_PREFLIGHT_REJECTED")
        telemetry=h.get("telemetry",{})
        if telemetry.get("root")!=self.route.get("env",{}).get("MEDVLM_TELEMETRY_ROOT") or telemetry.get("campaign_id")!=self.route.get("env",{}).get("MEDVLM_TELEMETRY_CAMPAIGN"):
            reasons.append("FORGE_TELEMETRY_ROUTE_CHANGED")
        try: require(file_hash(telemetry["forge_contract"]["path"])==telemetry["forge_contract"]["sha256"],"FORGE_CONTRACT_CHANGED")
        except (TerminalError,OSError,KeyError,TypeError): reasons.append("FORGE_CONTRACT_CHANGED")
        if for_run and snapshot.get("manual_handoff_verified") is not True: reasons.append("MANUAL_HANDOFF_NOT_AUTHORIZED")
        return sorted(set(reasons))

    def _worker_observation(self,snapshot,key):
        record=snapshot["jobs"][key]; health={}; exited=None
        if record.get("attempt_dir"):
            attempt=absolute(record["attempt_dir"]); require(attempt.is_relative_to(self.root/"attempts"),"ATTEMPT_OUTSIDE_CAMPAIGN")
            for filename in ("heartbeat.json","exit.json"):
                path=attempt/filename
                if path.exists():
                    if filename=="heartbeat.json": health=read_json(path)
                    else: exited=read_json(path)
        identities=(record.get("worker_identity"),health.get("worker_identity"),health.get("child_identity"))
        live=any(self.is_alive(identity) for identity in identities)
        if live: return True,None
        token=record.get("launch_token")
        bound_exit=None
        if token and exited and exited.get("launch_token")==token and (attempt/"launch.json").exists():
            launch=read_json(attempt/"launch.json")
            job=next(job for job in self.manifest["jobs"] if job["job_id"]==key)
            if (launch.get("launch_token")==token and launch.get("attempt")==record.get("attempts")
                    and launch.get("campaign_id")==snapshot.get("campaign_id")
                    and launch.get("job_hash")==digest(job) and digest(launch.get("job"))==digest(job)):
                bound_exit=exited
        return live,bound_exit

    def _worker_state(self,snapshot,key):
        # The canonical tick arbitrates absent exit records and launch races
        # under its worker/GPU leases. A stale RUNNING row is not live evidence.
        return self._worker_observation(snapshot,key)[0]

    def _progress(self,job,record):
        if not record.get("attempt_dir"): return {},None
        attempt=absolute(record["attempt_dir"]); require(attempt.is_relative_to(self.root/"attempts"),"ATTEMPT_OUTSIDE_CAMPAIGN")
        path=attempt/"progress.json"
        if not path.exists(): return {},attempt
        value=read_json(path)
        require(all(value.get(key)==job.get(key) for key in ("job_id","run_id",*HASH_FIELDS)),"PROGRESS_IDENTITY_CHANGED")
        return value,attempt

    def _telemetry(self,job,record,progress,native_attempt):
        if not self.handoff or native_attempt is None: return {},{},"current_attempt_not_bound"
        launch_path=native_attempt/"launch.json"
        if not launch_path.exists(): return {},{},"current_attempt_not_bound"
        launch=read_json(launch_path)
        require(launch.get("launch_token")==record.get("launch_token") and launch.get("attempt")==record.get("attempts")
                and digest(launch.get("job"))==digest(job),"LAUNCH_IDENTITY_CHANGED")
        candidates=[]; argv=job.get("argv",[])
        if "--config" in argv:
            config=read_json(argv[argv.index("--config")+1]); output=absolute(config["output_dir"])
            require(output.is_relative_to(self.root),"OUTPUT_OUTSIDE_CAMPAIGN")
            directories=([output] if output.is_dir() else [])+list(output.parent.glob(output.name+".attempt-*"))
            for directory in directories:
                path=absolute(directory)/"attempt.json"
                if not path.exists(): continue
                item=read_json(path)
                if all(item.get(k)==job.get(k) for k in ("job_id","run_id",*HASH_FIELDS)):
                    try:
                        if datetime.fromisoformat(item["started_at"])>=datetime.fromisoformat(launch["created_at"]): candidates.append(item["attempt_id"])
                    except (ValueError,KeyError,TypeError): pass
        explicit=progress.get("telemetry_attempt_id")
        # The producer's private attempt.json, matched to the current native
        # launch, is mandatory even when progress advertises an explicit UUID.
        if explicit is not None and (not candidates or any(value!=explicit for value in candidates)):
            return {},{},"current_attempt_not_bound"
        if len(candidates)!=1: return {},{},"current_attempt_not_bound" if not candidates else "ambiguous_attempt_identity"
        attempt_id=candidates[0]
        require(isinstance(attempt_id,str) and SAFE.fullmatch(attempt_id),"UNSAFE_TELEMETRY_ATTEMPT_ID")
        telemetry=self.handoff["telemetry"]; root=absolute(telemetry["root"])
        require(SAFE.fullmatch(telemetry["campaign_id"]) and SAFE.fullmatch(job["run_id"]),"UNSAFE_TELEMETRY_ID")
        directory=root/telemetry["campaign_id"]/job["run_id"]/attempt_id
        if not (directory/"manifest.json").exists(): return {},{},"telemetry_not_emitted"
        manifest=read_json(directory/"manifest.json")
        expected={"schema_version":"mediflow.telemetry.v1","campaign_id":telemetry["campaign_id"],"job_id":job["job_id"],"run_id":job["run_id"],"attempt_id":attempt_id}
        require(all(manifest.get(k)==v for k,v in expected.items()) and manifest.get("data_hash")==job["data_hash"]
                and manifest.get("historical") is False,"TELEMETRY_IDENTITY_CHANGED")
        def tail(name):
            path=absolute(directory/(name+".jsonl"))
            if not path.exists(): return {}
            with path.open("rb") as stream:
                size=path.stat().st_size; offset=max(0,size-TAIL_BYTES); stream.seek(offset); raw=stream.read(TAIL_BYTES)
            if offset: raw=raw.partition(b"\n")[2]
            lines=raw.split(b"\n")[:-1]
            if not lines: return {}
            last=json.loads(lines[-1]); require(all(last.get(k)==v for k,v in expected.items())
                and last.get("provenance")=="live_measured" and last.get("producer")==manifest.get("producer"),"TELEMETRY_STREAM_IDENTITY_CHANGED")
            return last
        return tail("steps"),tail("resources"),None

    def metrics(self,snapshot):
        jobs={j["job_id"]:j for j in self.manifest["jobs"]}; result=[]
        for key,record in snapshot["jobs"].items():
            if record.get("status") not in {"STARTING","RUNNING"}: continue
            values={name:_metric(None,None) for name in METRIC_FIELDS}; reason=None
            row_progress={name:_metric(None,None) for name in ROW_PROGRESS_FIELDS}
            status="ACTIVITY_UNVERIFIED"; exit_code=None
            try:
                live,exited=self._worker_observation(snapshot,key)
                if live: status=record["status"]
                elif exited is not None:
                    status="EXITED_AWAITING_VERIFICATION"
                    if type(exited.get("exit_code")) is int: exit_code=exited["exit_code"]
                progress,attempt=self._progress(jobs[key],record)
                for name,field in METRIC_FIELDS.items(): values[name]=_metric(progress.get(field),"native_progress")
                for name in ROW_PROGRESS_FIELDS:
                    count=progress.get(name)
                    row_progress[name]=_metric(count if type(count) is int and count>=0 else None,"native_progress")
                steps,resources,reason=self._telemetry(jobs[key],record,progress,attempt)
                for name,field in METRIC_FIELDS.items():
                    source=resources if name.startswith(("gpu_","ram_")) else steps
                    if field in source: values[name]=_metric(source[field],"forge_telemetry",source.get("availability",{}).get(field,{}).get("reason","not_measured"))
                if values["eta_seconds"]["value"] is None and all(_finite(steps.get(k)) for k in ("samples_total","samples_processed","samples_per_second")) and steps["samples_per_second"]>0:
                    values["eta_seconds"]=_metric(max(0,steps["samples_total"]-steps["samples_processed"])/steps["samples_per_second"],"estimate_from_latest_measured_sample_rate")
                if reason:
                    for item in values.values():
                        if item["value"] is None: item["reason"]=reason
            except (TerminalError,OSError,ValueError,KeyError,TypeError): reason="metrics_unavailable_or_identity_changed"
            result.append({"job_id":key,"scope":"V7" if v7_job(jobs[key]) else "V6_READ_ONLY","status":status,
                           "native_state":record["status"],"exit_code":exit_code,"row_progress":row_progress,
                           "metrics":values,"telemetry_reason":reason})
        return result

    def view(self,snapshot,*,event="STATUS"):
        reasons=self.blockers(snapshot,for_run=self.scope_token is not None)
        conditions={}
        for condition in CONDITIONS:
            entry=(self.handoff or {}).get("condition_jobs",{}).get(condition,{"train":["v7-"+condition],"evaluation":[]})
            conditions[condition]={kind:{"scheduled":len(entry.get(kind,[])),"succeeded":sum(snapshot["jobs"].get(key,{}).get("status")=="SUCCEEDED" for key in entry.get(kind,[]))} for kind in ("train","evaluation")}
        allowed=(self.handoff or {}).get("allowed_job_ids",[])
        complete=bool(allowed) and not reasons and all(snapshot["jobs"].get(key,{}).get("status")=="SUCCEEDED" for key in allowed)
        ready=bool(allowed) and not reasons and all(snapshot["jobs"].get(key,{}).get("status")=="PENDING" for key in allowed)
        return {"event":event,"v7_status":"SUCCEEDED" if complete else "BLOCKED_PREREQUISITE" if reasons else "READY_FOR_MANUAL_RUN" if ready else "IN_PROGRESS",
                "v7_completed":complete,"reasons":reasons,"conditions":conditions,"active":self.metrics(snapshot),
                "report_jobs":{key:snapshot["jobs"].get(key,{}).get("status","MISSING") for key in (self.handoff or {}).get("report_job_ids",["v7-report"])}}

    def execute(self,command="status",*,scope_token=None,max_updates=None):
        require(command in {"status","preflight","watch","run"},"UNKNOWN_TERMINAL_COMMAND")
        require(max_updates is None or type(max_updates) is int and max_updates>0,"INVALID_MAX_UPDATES")
        if command=="run":
            require(self.handoff is not None,"TERMINAL_HANDOFF_MISSING")
            require(isinstance(scope_token,str) and bool(scope_token) and hashlib.sha256(scope_token.encode()).hexdigest()==self.handoff.get("manual_scope_token_sha256"),"EXPLICIT_TERMINAL_SCOPE_TOKEN_REQUIRED")
            self.scope_token=scope_token
        try:
            snapshot=self._canonical("status"); updates=0
            while True:
                output=self.view(snapshot,event="PREFLIGHT" if command=="preflight" else "STATUS"); self.emit(output); updates+=1
                if command=="status": return 0
                if output["reasons"] and command!="watch": return 2
                if command=="preflight": return 0
                if output["v7_completed"]: return 0
                if max_updates is not None and updates>=max_updates:
                    self.emit({"event":"MONITORING_LIMIT_REACHED","v7_completed":False}); return 3
                if command=="run" and not any(self._worker_state(snapshot,key) for key in snapshot["jobs"]):
                    # No independent ready-job choice or state transition lives
                    # here. Exact canonical tick decides/reconciles one worker.
                    self._canonical("tick")
                self.sleep(self.interval); snapshot=self.local_snapshot()
        except KeyboardInterrupt:
            self.emit({"event":"DETACHED","v7_completed":False,"worker_signalled":False,
                       "further_tick_requests":False,"inflight_atomic_tick_may_finish":self.inflight_action=="tick"}); return 130


def render_text(value):
    if value.get("event")=="CANONICAL_VALIDATION_RUNNING" and "v7_status" not in value:
        return "Validating canonical dispatcher; ETA unavailable (validation duration not measured)."
    if value.get("event")=="DETACHED":
        return ("Detached without signalling the native worker. No further tick requests. "
                +("The already-requested canonical tick may still finish and start its worker. " if value.get("inflight_atomic_tick_may_finish") else "")
                +"Watch reattaches without dispatch.")
    if value.get("event")=="MONITORING_LIMIT_REACHED": return "Monitoring limit reached; V7 completion remains unverified."
    parts=["V7 "+value.get("v7_status","UNKNOWN")]
    for condition,item in value.get("conditions",{}).items():
        parts.append(condition+" train "+str(item["train"]["succeeded"])+"/"+str(item["train"]["scheduled"])
                     +" eval "+str(item["evaluation"]["succeeded"])+"/"+str(item["evaluation"]["scheduled"]))
    for active in value.get("active",[]):
        metrics=active["metrics"]; fields=[]
        for name,item in metrics.items():
            text=str(round(item["value"],5)) if item["value"] is not None else "unavailable("+str(item["reason"])+")"
            if str(item.get("source","")).startswith("estimate_"): text+="(estimated from measured sample rate)"
            fields.append(name+"="+text)
        for name,item in active.get("row_progress",{}).items():
            if item["value"] is not None: fields.append(name+"="+str(item["value"]))
        parts.append(active["job_id"]+" "+active["scope"]+" "+active["status"]
                     +" native_state="+active.get("native_state",active["status"])
                     +(" exit_code="+str(active["exit_code"]) if active.get("exit_code") is not None else "")
                     +" | "+" ".join(fields))
    if value.get("reasons"): parts.append("Blocked: "+", ".join(value["reasons"]))
    return "\n".join(parts)


def main(argv=None,*,client_factory=TerminalClient):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment",type=Path,required=True); parser.add_argument("--deployment-sha256",required=True)
    parser.add_argument("--handoff",type=Path); parser.add_argument("--handoff-sha256")
    parser.add_argument("--interval",type=float,default=5.); parser.add_argument("--max-updates",type=int)
    parser.add_argument("--json",action="store_true"); parser.add_argument("command",choices=("status","preflight","watch","run"),nargs="?",default="status")
    args=parser.parse_args(argv)
    def emit(value): print(json.dumps(value,ensure_ascii=False,allow_nan=False) if args.json else render_text(value),flush=True)
    try:
        client=client_factory(args.deployment,args.deployment_sha256,handoff=args.handoff,handoff_sha256=args.handoff_sha256,
                              interval=args.interval,emit=emit)
        return client.execute(args.command,scope_token=os.environ.get(TOKEN_ENV) if args.command=="run" else None,max_updates=args.max_updates)
    except TerminalError as exc:
        print(json.dumps({"event":"BLOCKED_PREREQUISITE","reason":str(exc),"v7_completed":False}),flush=True); return 2
    except (OSError,ValueError,KeyError,TypeError):
        print(json.dumps({"event":"BLOCKED_PREREQUISITE","reason":"INPUT_OR_CANONICAL_RESPONSE_UNAVAILABLE","v7_completed":False}),flush=True); return 2


if __name__=="__main__": raise SystemExit(main())
