#!/usr/bin/env python3
"""Resume the finite, predeclared local v1 campaign without rerunning successes.

Only public native weights are downloaded. No Hub upload, service operation,
system dependency installation or host-key changes are performed.
"""
from __future__ import annotations
import argparse
import copy
import fnmatch
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from mediflow_medvlm.runtime import atomic_json, canonical_hash, directory_digest, file_hash, utcnow, BudgetLedger
from mediflow_medvlm.evaluation.runner import freeze_protocol


def read(path):
    return json.loads(Path(path).read_text())


def completed(path):
    return Path(path).exists() and read(path).get("status")=="SUCCEEDED"


def wait_owned_training(path):
    if not Path(path).exists():return
    state=read(path)
    while state.get("status")=="RUNNING":
        pid=state.get("training_pid")
        try:
            command=Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0",b" ").decode()
        except OSError:
            raise RuntimeError("prior training process exited without terminal manifest; inspect latest checkpoint")
        if state["run_id"] not in command:
            raise RuntimeError("prior PID does not belong to this run; refuse to duplicate work")
        time.sleep(30)
        state=read(path)


def invoke(argv, log):
    with Path(log).open("a") as stream:
        process=subprocess.Popen([sys.executable,*argv],stdout=stream,stderr=subprocess.STDOUT)
        print(json.dumps({"event":"started","pid":process.pid,"argv":argv,"at":utcnow()}),flush=True)
        try:return process.wait(timeout=21600)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:return process.wait(timeout=60)
            except subprocess.TimeoutExpired:process.kill();process.wait();return 124


def download(config, metadata, root, token=None):
    from huggingface_hub import HfApi, snapshot_download, try_to_load_from_cache
    patterns=['*.json','*.txt','*.jinja','*.safetensors','LICENSE','README.md','*.model']
    info=HfApi(token=token).model_info(config['model_id'],revision=config['revision'],files_metadata=True,timeout=30)
    if info.sha!=config['revision']:raise ValueError('download metadata revision changed')
    expected=0;matched=0
    for asset in info.siblings or []:
        if not any(fnmatch.fnmatchcase(asset.rfilename,pattern) for pattern in patterns):continue
        matched+=1
        if type(asset.size) is not int or asset.size<0:
            raise ValueError('download size metadata missing: '+asset.rfilename)
        cached=try_to_load_from_cache(config['model_id'],asset.rfilename,cache_dir=config['cache_dir'],revision=config['revision'])
        if isinstance(cached,str) and Path(cached).is_file():
            if Path(cached).stat().st_size!=asset.size:
                raise ValueError('cached model file size differs from pinned metadata: '+asset.rfilename)
        else:expected+=asset.size
    if not matched:raise ValueError('download metadata contains no selected files')
    usage=shutil.disk_usage(root)
    if usage.free-expected < max(20*1024**3,.15*usage.total):raise RuntimeError("DISK_RESERVE")
    measured=sum(p.stat().st_size for p in (root/'models').rglob('*') if p.is_file() and not p.is_symlink())
    if measured+expected>40*1024**3:raise RuntimeError("MODEL_DOWNLOAD_BUDGET")
    return snapshot_download(config['model_id'],revision=config['revision'],cache_dir=config['cache_dir'],
                             allow_patterns=patterns,max_workers=2,token=token)


def verify_training_cache(config):
    """Verify completed disk evidence using CPU reads only; never reload a model."""
    from mediflow_medvlm.training.config import load_frozen_data, assert_resume_identity
    from mediflow_medvlm.training.models import model_file_digests
    from mediflow_medvlm.training.rendering import image_paths
    out=Path(config['output_dir']);manifest=read(out/'manifest.json')
    digest=canonical_hash(config)
    if manifest.get('status')!='SUCCEEDED':raise ValueError('training cache is not successful')
    if manifest.get('config_hash')!=digest or canonical_hash(read(out/'config.json'))!=digest:
        raise ValueError('completed training config changed')
    for key in ('run_id','pair_id','model_id','revision'):
        if manifest.get(key)!=config[key]:raise ValueError('completed training identity changed: '+key)
    rows,data=load_frozen_data(config)
    for split in rows.values():
        for row in split:
            if row.get('input',{}).get('images'):
                image_paths(row,config.get('image_root'))
    identity=manifest['identity']
    if manifest.get('data')!=data or identity.get('data')!=data:
        raise ValueError('completed training frozen data changed')
    if identity.get('config_sha256')!=digest:
        raise ValueError('checkpoint training config changed')
    model=manifest['model']
    for key in ('model_id','revision','quantization'):
        expected=config.get(key,'none' if key=='quantization' else None)
        if model.get(key)!=expected or identity.get(key)!=expected:
            raise ValueError('completed training model identity changed: '+key)
    if model.get('native_files')!=model_file_digests(config):
        raise ValueError('completed training native model files changed')
    if identity.get('template_sha256')!=model.get('chat_template_sha256'):
        raise ValueError('completed training template changed')
    selection=read(out/'selection.json')
    if selection.get('identity')!=identity or selection.get('run_id')!=config['run_id'] or selection.get('pair_id')!=config['pair_id']:
        raise ValueError('selected checkpoint identity changed')
    checkpoint_path=Path(selection['checkpoint'])
    if not checkpoint_path.resolve().is_relative_to((out/'checkpoints').resolve()):
        raise ValueError('selected checkpoint is outside its training run')
    checkpoint=assert_resume_identity(identity,checkpoint_path)
    for relative,expected in checkpoint['artifacts']['files'].items():
        asset=checkpoint_path/relative
        if not asset.resolve().is_relative_to(checkpoint_path.resolve()) or file_hash(asset)!=expected:
            raise ValueError('selected checkpoint artifact changed')
    adapter=manifest['adapter']
    if Path(adapter['path']).resolve()!=checkpoint_path.resolve() or adapter.get('digest')!=directory_digest(checkpoint_path)['sha256']:
        raise ValueError('selected adapter content changed')
    if selection.get('adapter')!=adapter:
        raise ValueError('selection adapter differs from completed training')
    return manifest,selection


def verify_reload_cache(config, manifest, selection):
    result=read(Path(config['output_dir'])/'reload.json')
    if not selection.get('reload_verified') or not manifest.get('reload_verified') or result.get('status')!='SUCCEEDED':
        raise ValueError('selected adapter reload is not verified')
    if result.get('run_id')!=config['run_id'] or Path(result['checkpoint']).resolve()!=Path(selection['checkpoint']).resolve():
        raise ValueError('reload belongs to another checkpoint')
    if not result.get('fresh_process_required') or result.get('pid')==manifest.get('training_pid'):
        raise ValueError('reload lacks fresh process evidence')
    if result.get('generated_tokens',0)<=0:
        raise ValueError('reload has no generated tokens')
    for key in ('model_id','revision','quantization','dtype','chat_template_sha256','native_files'):
        if result.get('model',{}).get(key)!=manifest['model'].get(key):
            raise ValueError('reloaded model identity changed: '+key)
    return result


def train_stage(config_path, root):
    config=read(config_path);out=Path(config['output_dir']);manifest=out/'manifest.json'
    wait_owned_training(manifest)
    if completed(manifest):
        verify_training_cache(config)
        return True
    if manifest.exists() and read(manifest).get('optimizer_steps',0):
        latest=read(out/'latest.json')['checkpoint']
        argv=['-m','mediflow_medvlm.training','train','--config',str(config_path),'--resume',latest]
    else:
        argv=['-m','mediflow_medvlm.training','train','--config',str(config_path)]
    invoke(argv,root/'runs'/(config['run_id']+'-launch.log'))
    if not completed(manifest):return False
    verify_training_cache(config)
    return True


def reload_stage(config_path, root):
    config=read(config_path);out=Path(config['output_dir']);selection=out/'selection.json'
    if not selection.exists():return False
    manifest,selected=verify_training_cache(config)
    checkpoint=selected['checkpoint']
    if selected.get('reload_verified') and completed(out/'reload.json'):
        verify_reload_cache(config,manifest,selected)
        return True
    invoke(['-m','mediflow_medvlm.training','reload','--config',str(config_path),'--checkpoint',checkpoint],root/'runs'/(config['run_id']+'-reload-launch.log'))
    if not completed(out/'reload.json'):return False
    manifest,selected=verify_training_cache(config)
    verify_reload_cache(config,manifest,selected)
    return True


def evaluate_stage(config_path, protocol_path, root, key, label, mode, selection):
    config=read(config_path);out=root/'runs'/f'{key}-{label}-{mode}-20261023'
    argv=['-m','mediflow_medvlm.evaluation','evaluate','--config',str(config_path),
          '--protocol',str(protocol_path),'--output-dir',str(out),'--pair-id',config['pair_id']]
    if mode=='tuned':argv+=['--adapter',selection['checkpoint']]
    # The evaluator owns its proof/identity gate, including successful resumes.
    # A failed gate can leave an older SUCCEEDED manifest untouched.
    if invoke(argv,root/'runs'/f'{key}-{label}-{mode}-launch.log')!=0:
        raise RuntimeError('evaluation invocation or cache verification failed: '+key+'/'+label+'/'+mode)
    return out


def optional_diagnostics(config_path, root, key, test_protocol, results, selection, policy):
    """Finite extra evidence. Failures never erase the primary paired result."""
    config=read(config_path);outcomes={}
    if key.startswith('qwen3_'):
        try:
            from mediflow_medvlm.evaluation.expert_review import prepare_expert_review
            packet=root/'review'/key
            if not (packet/'manifest.json').exists():
                prepare_expert_review(test_protocol,results['base']/'predictions.jsonl',results['tuned']/'predictions.jsonl',packet)
            outcomes['expert_review']={'status':'PREPARED_NOT_RUN','reason_code':'REVIEWER_NOT_ASSIGNED','packet':str(packet)}
        except Exception as error:outcomes['expert_review']={'status':'FAILED_REPRODUCIBLE','reason_code':type(error).__name__}
        return outcomes
    from mediflow_medvlm.evaluation.image_diagnostics import derive_no_image_protocol, compare_image_conditions
    ledger=BudgetLedger(root/'runtime/gpu_budget.json')
    try:
        if ledger.remaining(config['model_id'])<900:
            outcomes['no_image']={'status':'SKIPPED_BUDGET'}
        else:
            protocol=root/'manifests/evaluation/slake-no-image-v1.json'
            derive_no_image_protocol(test_protocol,protocol,cap=policy['image_diagnostic_cap'])
            conditions={}
            for mode in ('base','tuned'):
                run=evaluate_stage(config_path,protocol,root,key,'no-image',mode,selection)
                destination=root/'runs'/f'{key}-{mode}-image-diagnostic.json'
                compare_image_conditions(test_protocol,protocol,results[mode]/'predictions.jsonl',run/'predictions.jsonl',destination)
                conditions[mode]={'status':read(run/'run_manifest.json')['status'],'comparison':str(destination)}
            outcomes['no_image']=conditions
    except Exception as error:outcomes['no_image']={'status':'FAILED_REPRODUCIBLE','reason_code':type(error).__name__}
    try:
        if ledger.remaining(config['model_id'])<900:
            outcomes['external_vqarad']={'status':'SKIPPED_BUDGET'}
        else:
            aggregate=read(root/'manifests/current_freeze.json')
            source=aggregate['splits']['vqa_rad/test']['snapshot_path']
            protocol=root/'manifests/evaluation/vqarad-test-v1.json'
            freeze_protocol([source],protocol,cap=policy['test_caps']['vqa_rad'])
            external={mode:evaluate_stage(config_path,protocol,root,key,'vqarad',mode,selection) for mode in ('base','tuned')}
            destination=root/'runs'/f'{key}-vqarad-paired-comparison.json'
            code=invoke(['-m','mediflow_medvlm.evaluation','compare','--base',str(external['base']),
                         '--tuned',str(external['tuned']),'--output',str(destination)],root/'runs'/f'{key}-vqarad-compare.log')
            outcomes['external_vqarad']={'status':'SUCCEEDED' if code==0 and all(completed(p/'run_manifest.json') for p in external.values()) else 'FAILED_REPRODUCIBLE','comparison':str(destination)}
    except Exception as error:outcomes['external_vqarad']={'status':'FAILED_REPRODUCIBLE','reason_code':type(error).__name__}
    return outcomes


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data-root',required=True)
    parser.add_argument('--keys',nargs='+',default=['qwen3_17b','qwen3_4b','qwen35_2b','medgemma15_4b','gemma3_4b','smolvlm_500m'])
    parser.add_argument('--skip-download',action='store_true');args=parser.parse_args()
    os.umask(0o077)
    # Read the user's authenticated cache before isolating the model artifact cache.
    from huggingface_hub import get_token
    token=get_token()
    root=Path(args.data_root).resolve();os.environ.update(HF_HOME=str(root/'models/hf-home'),WANDB_DISABLED='true',TOKENIZERS_PARALLELISM='false')
    os.environ['PYTHONPATH']=str(Path(__file__).resolve().parents[1]/'src')
    policy=read('configs/evaluation_policy.json');state_path=Path('runtime/medvlm_state.json')
    state=read(state_path) if state_path.exists() else {'schema_version':'medvlm.state.v1','models':{},'stages':{}}
    state['data_root']=str(root);state['campaign_pid']=os.getpid();state['config_policy_hash']=canonical_hash(policy)
    state['active_process']={'pid':os.getpid(),'role':'finite_campaign'}
    atomic_json(state_path,state)
    for key in args.keys:
        predecessors={'gemma3_4b':['medgemma15_4b'], 'smolvlm_500m':['medgemma15_4b','gemma3_4b']}.get(key,[])
        if any(state['models'].get(parent,{}).get('status')=='SUCCEEDED' for parent in predecessors):
            if key in state['models']:
                state['models'][key]['campaign_disposition']='FALLBACK_NOT_NEEDED'
            else:
                state['models'][key]={'model_id':{'gemma3_4b':'google/gemma-3-4b-it','smolvlm_500m':'HuggingFaceTB/SmolVLM-500M-Instruct'}[key],
                                     'status':'PREPARED_NOT_RUN','evidence_level':'IMPLEMENTED_ONLY','reason_code':'FALLBACK_NOT_NEEDED'}
            atomic_json(state_path,state)
            continue
        state['current_model']=key;state['updated_at']=utcnow();atomic_json(state_path,state)
        config_path=root/'configs'/f'{key}-v1-20261023.json';smoke_path=root/'configs'/f'{key}-smoke-20261023.json'
        cell=copy.deepcopy(state['models'].get(key,{}))
        cell.update(status='RUNNING',phase='initialization',started_at=utcnow())
        cell.pop('reason_code',None);cell.pop('error',None);cell.pop('campaign_disposition',None)
        state['models'][key]=cell;atomic_json(state_path,state)
        try:
            config=read(config_path);freeze=read(config['freeze_manifest']);metadata=read(root/'manifests/models'/f'{key}.json')
            cell.update(model_id=config['model_id'],revision=config['revision'])
            cell.setdefault('evidence_level','IMPLEMENTED_ONLY')
            # Do not load a second model when an earlier worker for this cell is alive.
            wait_owned_training(Path(config['output_dir'])/'manifest.json')
            if not args.skip_download:download(config,metadata,root,token)
            if not train_stage(smoke_path,root) or not reload_stage(smoke_path,root):
                smoke=read(read(smoke_path)['output_dir']+'/manifest.json')
                cell.update(status=smoke['status'] if smoke['status']!='SUCCEEDED' else 'FAILED_REPRODUCIBLE',reason_code=smoke.get('reason_code'),phase='smoke_or_reload')
                continue
            if not train_stage(config_path,root) or not reload_stage(config_path,root):
                training=read(Path(config['output_dir'])/'manifest.json')
                cell.update(status=training['status'] if training['status']!='SUCCEEDED' else 'FAILED_REPRODUCIBLE',reason_code=training.get('reason_code'),phase='training_or_reload')
                continue
            selection=read(Path(config['output_dir'])/'selection.json')
            if not selection['reload_verified']:raise RuntimeError('selected adapter reload not verified')
            sources=['medquad','pubmedqa_labeled','kormedmcqa'] if key.startswith('qwen3_') else ['slake']
            dev_protocol=root/'manifests/evaluation'/('text-validation-v1.json' if key.startswith('qwen3_') else 'slake-validation-v1.json')
            freeze_protocol([freeze['splits'][source+'/validation']['snapshot_path'] for source in sources],
                            dev_protocol,split='validation',cap=0,caps={s:policy['validation_caps'][s] for s in sources})
            dev_results={mode:evaluate_stage(config_path,dev_protocol,root,key,'validation',mode,selection) for mode in ('base','tuned')}
            if not all(completed(p/'run_manifest.json') for p in dev_results.values()):
                cell.update(status='FAILED_REPRODUCIBLE',reason_code='VALIDATION_INFERENCE_FAILED',phase='validation_generation')
                continue
            paths=[freeze['splits'][source+'/test']['snapshot_path'] for source in sources]
            protocol_path=root/'manifests/evaluation'/('text-test-v1.json' if key.startswith('qwen3_') else 'slake-test-v1.json')
            freeze_protocol(paths,protocol_path,cap=0,caps={s:policy['test_caps'][s] for s in sources})
            results={}
            for mode in ['base','tuned']:
                results[mode]=evaluate_stage(config_path,protocol_path,root,key,'test',mode,selection)
            compare_path=root/'runs'/f'{key}-paired-comparison.json'
            compare_exit=invoke(['-m','mediflow_medvlm.evaluation','compare','--base',str(results['base']),'--tuned',str(results['tuned']),'--output',str(compare_path)],root/'runs'/f'{key}-compare.log')
            success=compare_exit==0 and compare_path.exists() and all(completed(p/'run_manifest.json') for p in results.values())
            measured=any(read(p/'run_manifest.json').get('evidence_level')=='REAL_MEDICAL_EVALUATED' for p in results.values())
            cell.update(status='SUCCEEDED' if success else 'FAILED_REPRODUCIBLE',evidence_level='REAL_MEDICAL_EVALUATED' if measured else 'REAL_GPU_SMOKE',phase='paired_evaluation',comparison=str(compare_path) if compare_path.exists() else None,training_manifest=str(Path(config['output_dir'])/'manifest.json'))
            if success:cell['diagnostics']=optional_diagnostics(config_path,root,key,protocol_path,results,selection,policy)
        except Exception as error:
            cell.update(status='FAILED_REPRODUCIBLE',reason_code=type(error).__name__,error=str(error)[:300])
        finally:
            cell['finished_at']=utcnow();state['updated_at']=utcnow();atomic_json(state_path,state)
            print(json.dumps({'event':'model_terminal','key':key,**cell}),flush=True)
    state['active_process']=None;state['campaign_pid']=None;state['updated_at']=utcnow();atomic_json(state_path,state)


if __name__=='__main__':main()
