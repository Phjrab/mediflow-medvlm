"""Bounded SSH inventory and text inference, requiring an already trusted host."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
from .runtime import atomic_json, canonical_hash, utcnow

REMOTE_CODE = r'''
import json,sys,platform,pathlib,shutil,time,resource,statistics
payload=json.load(sys.stdin)
result={"architecture":platform.machine(),"python":platform.python_version(),"os":platform.platform()}
for key,path in (("board_model","/proc/device-tree/model"),("l4t","/etc/nv_tegra_release"),("os_release","/etc/os-release")):
 p=pathlib.Path(path);result[key]=p.read_text(errors="replace").replace("\x00","").strip() if p.exists() else None
result["disk_free_bytes"]=shutil.disk_usage(".").free
try:
 import torch
 result.update(torch=torch.__version__,cuda_available=torch.cuda.is_available(),cuda_runtime=torch.version.cuda)
except Exception as error:
 result.update(torch=None,cuda_available=False,torch_error=type(error).__name__)
if payload["action"]=="inventory":
 print(json.dumps(result));sys.exit(0)
if not result["cuda_available"]:
 result.update(status="BLOCKED_HARDWARE",reason_code="GPU_NOT_AVAILABLE");print(json.dumps(result));sys.exit(0)
try:
 from transformers import AutoConfig,AutoTokenizer
 import transformers
 config=payload["model"]
 native=AutoConfig.from_pretrained(config["model_id"],revision=config["revision"],trust_remote_code=False,local_files_only=True)
 if getattr(native,"vision_config",None):raise ValueError("text-device-smoke requires a text model")
 cls=getattr(transformers,native.architectures[0])
 tokenizer=AutoTokenizer.from_pretrained(config["model_id"],revision=config["revision"],trust_remote_code=False,local_files_only=True)
 model=cls.from_pretrained(config["model_id"],revision=config["revision"],trust_remote_code=False,local_files_only=True,dtype=torch.float16).to("cuda").eval()
 if config.get("adapter"):
  from peft import PeftModel
  model=PeftModel.from_pretrained(model,config["adapter"]).eval()
 outputs=[]
 for item in payload["samples"]:
  row={"sample_id":item["sample_id"],"input_hash":item["input_hash"],"warmup":item["warmup"],"ttft_seconds":None}
  started=time.monotonic()
  try:
   rendered=tokenizer.apply_chat_template(item["messages"],tokenize=False,add_generation_prompt=True,enable_thinking=False)
   inputs=tokenizer(rendered,return_tensors="pt").to("cuda")
   if inputs["input_ids"].shape[-1]>2048:raise ValueError("CONTEXT_OVERFLOW")
   with torch.inference_mode():generated=model.generate(**inputs,max_new_tokens=64,max_time=60,do_sample=False)
   torch.cuda.synchronize();tokens=generated[0,inputs["input_ids"].shape[-1]:]
   row.update(status="answered",raw_output=tokenizer.decode(tokens,skip_special_tokens=True),output_tokens=len(tokens),finish_reason="length" if len(tokens)>=64 else "stop")
  except Exception as error:row.update(status="oom" if isinstance(error,torch.OutOfMemoryError) else "load_error",error_reason=type(error).__name__)
  row["e2e_seconds"]=time.monotonic()-started;outputs.append(row)
 result.update(status="SUCCEEDED",samples=outputs,peak_allocated_bytes=torch.cuda.max_memory_allocated(),process_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,shared_memory_note="Jetson shared memory; not discrete VRAM",energy_joules=None)
except Exception as error:result.update(status="BLOCKED_HARDWARE",reason_code="UNSUPPORTED_RUNTIME",error=type(error).__name__)
print(json.dumps(result))
'''


def execute(host, user, remote_python, payload, timeout):
    command = shlex.quote(remote_python) + " -c " + shlex.quote(REMOTE_CODE)
    if os.environ.get("MEDVLM_JETSON_PASSWORD"):
        import paramiko
        client = paramiko.SSHClient(); client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.RejectPolicy())
        client.connect(host, username=user, password=os.environ["MEDVLM_JETSON_PASSWORD"], timeout=8,
                       auth_timeout=8, banner_timeout=8, look_for_keys=True, allow_agent=True)
        try:
            stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
            stdin.write(json.dumps(payload)); stdin.channel.shutdown_write()
            text = stdout.read().decode(); error = stderr.read().decode()
            return stdout.channel.recv_exit_status(), text, error
        finally:
            client.close()
    process = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=8",
                              user+"@"+host, command], input=json.dumps(payload), text=True,
                             capture_output=True, timeout=timeout)
    return process.returncode, process.stdout, process.stderr


def device_run(output, *, action="inventory", host="192.168.50.11", user="jetson2", remote_python="python3", model_config=None, samples_file=None):
    root = Path(output); root.mkdir(parents=True, exist_ok=True); os.chmod(root,0o700)
    started=time.monotonic(); manifest={"started_at":utcnow(),"host":host,"user":user,"action":action,"status":"RUNNING","evidence_level":"IMPLEMENTED_ONLY","device_outcome":"NOT_RUN"}
    payload={"action":action}
    if action=="smoke":
        if not model_config or not samples_file:raise ValueError("smoke requires model config and frozen dev samples")
        from .training.rendering import render_messages
        rows=[json.loads(line) for line in Path(samples_file).read_text().splitlines() if line.strip()]
        if len(rows)<12 or any(r["split"]!="validation" or r["input"].get("images") for r in rows):
            raise ValueError("text device smoke needs >=12 public text validation samples")
        rows=sorted(rows,key=lambda r:canonical_hash([20261023,r["sample_id"]]))[:12]
        payload["model"]=json.loads(Path(model_config).read_text())
        payload["samples"]=[{"sample_id":r["sample_id"],"messages":render_messages(r),"input_hash":canonical_hash(r["input"]),"warmup":i<2} for i,r in enumerate(rows)]
        manifest["input_list_digest"]=canonical_hash([x["input_hash"] for x in payload["samples"]])
    try:
        code,text,error=execute(host,user,remote_python,payload,timeout=900 if action=="smoke" else 30)
        if code:
            (root/"connection_error.txt").write_text(error)
            reason="HOST_KEY_UNVERIFIED" if "Host key verification failed" in error else "AUTH_REQUIRED" if "Permission denied" in error else "SSH_UNREACHABLE"
            manifest.update(status="BLOCKED_ACCESS" if reason!="SSH_UNREACHABLE" else "NETWORK_UNREACHABLE",reason_code=reason,device_outcome="BLOCKED_ACCESS" if reason!="SSH_UNREACHABLE" else "BLOCKED_NETWORK")
        else:
            result=json.loads(text.strip().splitlines()[-1]);atomic_json(root/"inventory.json",{k:v for k,v in result.items() if k!="samples"})
            manifest.update(status=result.get("status","SUCCEEDED"),reason_code=result.get("reason_code"))
            if action=="smoke" and result.get("samples"):
                with (root/"samples.jsonl").open("w") as stream:
                    for row in result["samples"]:stream.write(json.dumps(row,ensure_ascii=False)+"\n")
                measured=[r for r in result["samples"] if not r["warmup"]]
                manifest.update(evidence_level="DEVICE_INFERENCE_MEASURED",device_outcome="MEASURED",measured=len(measured),successful=sum(r["status"]=="answered" for r in measured))
            else:manifest.update(evidence_level="CPU_TESTED",device_outcome="NOT_RUN")
    except Exception as error:
        manifest.update(status="BLOCKED_ACCESS" if "host" in str(error).lower() else "NETWORK_UNREACHABLE",reason_code=type(error).__name__,device_outcome="BLOCKED_ACCESS")
        (root/"connection_error.txt").write_text(str(error))
    manifest.update(finished_at=utcnow(),wall_seconds=time.monotonic()-started)
    atomic_json(root/"run_manifest.json",manifest);atomic_json(root/"summary.json",manifest)
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description="SSH host verification remains enabled. No system modifications.")
    parser.add_argument("action",choices=["inventory","smoke"]);parser.add_argument("--output",required=True)
    parser.add_argument("--host",default="192.168.50.11");parser.add_argument("--user",default="jetson2");parser.add_argument("--remote-python",default="python3")
    parser.add_argument("--model-config");parser.add_argument("--samples-file")
    args=vars(parser.parse_args(argv));print(json.dumps(device_run(**args),ensure_ascii=False))


if __name__=="__main__":main()
