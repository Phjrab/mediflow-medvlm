"""Lightweight research entrypoint. Heavy dependencies are imported on demand."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from .runtime import atomic_json, canonical_hash, file_hash, utcnow, GPULease


def doctor(output=None):
    result={"checked_at":utcnow(),"os":platform.platform(),"architecture":platform.machine(),"python":platform.python_version(),
            "disk_free_bytes":shutil.disk_usage(".").free,"packages":{},"cuda_available":None}
    for package in ("torch","transformers","peft","bitsandbytes","pyarrow","pillow"):
        try:result["packages"][package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:result["packages"][package]=None
    try:
        import torch
        result["cuda_available"]=torch.cuda.is_available();result["cuda_runtime"]=torch.version.cuda
        if result["cuda_available"]:
            result["gpu_name"]=torch.cuda.get_device_name();result["gpu_total_bytes"]=torch.cuda.get_device_properties(0).total_memory
            result["gpu_free_bytes"]=torch.cuda.mem_get_info()[0];result["bf16_supported"]=torch.cuda.is_bf16_supported()
            x=torch.ones(32,device="cuda",requires_grad=True);(x*x).sum().backward();torch.cuda.synchronize()
            result["forward_backward_finite"]=bool(torch.isfinite(x.grad).all())
    except Exception as error:result["torch_probe_error"]=type(error).__name__
    result["status"]="SUCCEEDED" if result.get("forward_backward_finite") else "BLOCKED_HARDWARE"
    result["evidence_level"]="REAL_GPU_SMOKE" if result.get("forward_backward_finite") else "CPU_TESTED"
    result["medical_training_performed"]=False
    if output:atomic_json(output,result)
    return result


def run_plan(plan_path, state_path="runtime/medvlm_state.json"):
    """Finite explicit argv jobs; immutable outputs and completed work are reused."""
    plan=json.loads(Path(plan_path).read_text());identity=canonical_hash(plan)
    state_file=Path(state_path)
    state=json.loads(state_file.read_text()) if state_file.exists() else {"schema_version":"medvlm.state.v1","jobs":{}}
    state.setdefault("jobs", {})
    if state.get("plan_hash") not in (None,identity):raise ValueError("plan changed: choose a new state path")
    state.update(plan_hash=identity,plan_path=str(Path(plan_path).resolve()))
    with GPULease(str(state_file)+".runner.lock","orchestrator"):
        for job in plan["jobs"]:
            key=job["id"];prior=state["jobs"].get(key,{})
            if prior.get("status")=="SUCCEEDED":
                for path,digest in prior.get("output_hashes",{}).items():
                    if file_hash(path)!=digest:raise ValueError("completed output changed: "+key)
                continue
            blocked=[dep for dep in job.get("requires",[]) if state["jobs"].get(dep,{}).get("status")!="SUCCEEDED"]
            if blocked:
                state["jobs"][key]={"status":"PREPARED_NOT_RUN","reason_code":"DEPENDENCY_NOT_SUCCEEDED","dependencies":blocked}
                atomic_json(state_file,state);continue
            log=state_file.parent/"logs"/(key+".log");log.parent.mkdir(parents=True,exist_ok=True)
            argv=job["command"]
            if not isinstance(argv,list) or not argv or any(not isinstance(x,str) for x in argv):raise ValueError("commands must be explicit argv arrays")
            item={"status":"RUNNING","started_at":utcnow(),"command":argv,"log_path":str(log)}
            state["jobs"][key]=item;atomic_json(state_file,state)
            with log.open("a") as stream:
                process=subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,cwd=job.get("cwd"))
                item["pid"]=process.pid;atomic_json(state_file,state)
                try:code=process.wait(timeout=min(float(job.get("timeout_seconds",21600)),129600))
                except subprocess.TimeoutExpired:
                    # Own process only; give checkpoint-aware workers a graceful stop.
                    process.terminate()
                    try:process.wait(timeout=60)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
                    code=124
            item.update(status="SUCCEEDED" if code==0 else "SKIPPED_BUDGET" if code==124 else "FAILED_REPRODUCIBLE",exit_code=code,finished_at=utcnow(),pid=None)
            manifest=job.get("result_manifest")
            if manifest and Path(manifest).exists():
                result=json.loads(Path(manifest).read_text());item.update({k:result[k] for k in ("status","evidence_level","reason_code") if k in result})
            item["output_hashes"]={p:file_hash(p) for p in job.get("outputs",[]) if Path(p).is_file()}
            atomic_json(state_file,state)
        state["active_process"]=None;state["updated_at"]=utcnow();atomic_json(state_file,state)
    return state


def main(argv=None):
    args=list(sys.argv[1:] if argv is None else argv)
    command=args[0] if args else None
    if command=="data":
        from .data.__main__ import main as delegated
        return delegated(args[1:])
    if command in {"train","models","reload"}:
        from .training.__main__ import main as delegated
        return delegated((["probe"]+args[2:]) if command=="models" and args[1:2]==["probe"] else args)
    if command in {"evaluate","compare","evaluation"}:
        from .evaluation.__main__ import main as delegated
        return delegated(args[1:] if command=="evaluation" else args)
    if command=="device":
        from .devices import main as delegated
        return delegated(args[1:])
    parser=argparse.ArgumentParser(prog="medvlm",description="Data-first offline medical QA/VQA research. Subcommands also have --help.")
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("doctor");p.add_argument("--output")
    for name in ("run","resume"):
        p=sub.add_parser(name);p.add_argument("--plan",required=True);p.add_argument("--state",default="runtime/medvlm_state.json")
    p=sub.add_parser("status");p.add_argument("--state",default="runtime/medvlm_state.json")
    for name in ("data","models","train","reload","evaluation","evaluate","compare","device"):
        sub.add_parser(name,add_help=False)
    parsed=parser.parse_args(args)
    if parsed.command=="doctor":value=doctor(parsed.output)
    elif parsed.command in {"run","resume"}:value=run_plan(parsed.plan,parsed.state)
    else:value=json.loads(Path(parsed.state).read_text())
    print(json.dumps(value,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
