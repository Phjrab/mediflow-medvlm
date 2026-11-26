"""CPU subprocess regression for the native completion CLI reload projection only."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from mediflow_medvlm import campaign_v6_completion as w

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path

@pytest.mark.parametrize('mode',['reload-child','train-child','main-train','main-cpu','reload-proof-error','reload-non-success'])
def test_actual_cli_subprocess_projects_declared_scope_without_mutating_native_receipts(tmp_path,mode):
    cfg={key:'a'*64 for key in w.IDENTITY_KEYS}
    cfg.update(kind='additional_audit' if mode=='main-cpu' else 'train',uses_gpu=mode!='main-cpu',
        scope=w.SCOPES['additional_audit'] if mode=='main-cpu' else w.TRAIN_SCOPE)
    path=write(tmp_path/'config.json',cfg);receipt=tmp_path/'native-receipt.json';directory=tmp_path/'stage';directory.mkdir()
    code=r'''
import json,os,sys
from pathlib import Path
from types import SimpleNamespace
from mediflow_medvlm import campaign_v6_completion as w
mode,config_path,receipt_path,directory=sys.argv[1:]
config=json.loads(Path(config_path).read_text())
identity={key:config[key] for key in w.IDENTITY_KEYS}
# Native reload_checkpoint returns probe_scope and identity fields, never scope.
reload_result={"status":"SUCCEEDED",**identity,"pid":123,"process_identity":{"pid":123,"boot_id":"synthetic","start_ticks":1},
    "probe_scope":"TRAINING_EXAMPLE_SANITY_CHECK","fresh_process_required":True,"adapter_weights_verified":True,
    "adapter_sha256":"b"*64,"trained_example_sample_id":"synthetic/training-only","trained_example_loss":.5,"target_tokens":4,
    "finite_forward":True,"base_dtype_signature":[],"generated_tokens":1,"generation_max_new_tokens":32,
    "generated_ids_sha256":"c"*64,"no_thinking_output":{"explicit_nonempty_thinking_detected":False},
    "dev_test_read":False,"runtime_versions":{},"finished_at":"synthetic"}
assert "scope" not in reload_result
train_result={"status":"SUCCEEDED","scope":"NATIVE_MAIN_ONE_PASS_TRAIN_ONLY",**identity,"training_pid":123,
    "process_identity":{"pid":123,"boot_id":"synthetic","start_ticks":1},"optimizer_steps":500,"samples_processed":7997,
    "metrics_sha256":"d"*64,"retry_start":"original_base","finished_at":"synthetic"}
def child(which,*args):
    assert [str(value) for value in args]==[config_path,directory,receipt_path]
    if mode=="reload-proof-error":raise w.dev.caps.initial.gate.GateError("SYNTHETIC_RELOAD_PROOF_FAILED")
    assert which==("train" if mode=="train-child" else "reload")
    result=dict(train_result if which=="train" else reload_result)
    if mode=="reload-non-success":result["status"]="FAILED"
    Path(receipt_path).write_text(json.dumps(result))
    return result
w.training_runtime=lambda actual:(config==actual and SimpleNamespace(train_checkpoint=lambda *a:child("train",*a),reload_checkpoint=lambda *a:child("reload",*a)))
def run_job(*args):
    assert list(args)==[config_path,receipt_path]
    if mode=="main-cpu":assert os.environ["CUDA_VISIBLE_DEVICES"]=="" and "torch" not in sys.modules
    result={"status":"SUCCEEDED","scope":config["scope"],**identity,"clinical_quality_certified":False}
    Path(receipt_path).write_text(json.dumps(result))
    return result
w.run_job=run_job
argv=["--config",config_path,"--receipt",receipt_path]
if mode in {"reload-child","reload-proof-error","reload-non-success"}:argv += ["--reload-checkpoint",directory]
elif mode=="train-child":argv += ["--train-checkpoint",directory]
raise SystemExit(w.main(argv))
'''
    result=subprocess.run([sys.executable,'-B','-c',code,mode,str(path),str(receipt),str(directory)],capture_output=True,text=True,
        env={key:value for key,value in os.environ.items() if key!='CUDA_VISIBLE_DEVICES'})
    if mode in {'reload-proof-error','reload-non-success'}:
        assert result.returncode==1 and not result.stdout and 'Traceback' not in result.stderr
        error=json.loads(result.stderr)
        assert error['reason_code']==('SYNTHETIC_RELOAD_PROOF_FAILED' if mode=='reload-proof-error' else 'COMPLETION_CLI_RESULT_NOT_SUCCEEDED')
        assert receipt.exists() is (mode=='reload-non-success')
    else:
        assert result.returncode==0,result.stderr
        assert json.loads(result.stdout)=={'status':'SUCCEEDED','scope':cfg['scope']}
        original=json.loads(receipt.read_text())
        if mode=='reload-child':assert 'scope' not in original and original['probe_scope']=='TRAINING_EXAMPLE_SANITY_CHECK' and original['dev_test_read'] is False
        if mode=='train-child':assert original['scope']=='NATIVE_MAIN_ONE_PASS_TRAIN_ONLY' and original['samples_processed']==7997 and original['optimizer_steps']==500
        assert not result.stderr
