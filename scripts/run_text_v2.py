#!/usr/bin/env python3
"""Finite Base / EN-v1 / KO-v2 / MIX-v2 campaign with separate smoke reloads."""
from __future__ import annotations
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from mediflow_medvlm.runtime import atomic_json, canonical_hash, file_hash, utcnow, resolve_resource_policy
from mediflow_medvlm.evaluation.runner import freeze_protocol, compare
from mediflow_medvlm.evaluation.v2 import promote_protocol
from complete_v1 import train_stage, reload_stage, verify_training_cache, verify_reload_cache
import complete_v1


def read(path):return json.loads(Path(path).read_text())


def immutable(path,value):
    if path.exists():
        if read(path)!=value:raise ValueError('immutable declaration changed: '+path.name)
    else:atomic_json(path,value)


def review_packet(root,dest,model,condition,base,tuned):
    from mediflow_medvlm.evaluation.review_v2 import prepare_aihub_expert_review
    output=root/'v2/review'/f'{model}-base-vs-{condition}'
    manifest_path=output/'manifest.json'
    if manifest_path.exists():
        receipt=read(manifest_path)
        if len(receipt.get('source_runs',[]))!=2 or receipt.get('selected_by_q_type')!={'2':8,'3':8} or receipt.get('seed')!=20261023:
            raise ValueError('review fixed selection identity changed')
        if set(receipt.get('files_sha256',{}))!={'review.csv','private_key.json','instructions.txt'}:
            raise ValueError('review packet receipt is incomplete')
        if receipt['protocol_file_sha256']!=file_hash(dest/'text-test.json'):
            raise ValueError('review protocol bytes changed')
        expected_paths=[base/'predictions.jsonl',tuned/'predictions.jsonl']
        for source,path in zip(receipt['source_runs'],expected_paths):
            if Path(source['predictions_path']).resolve()!=path.resolve() or file_hash(path.parent/'run_manifest.json')!=source['manifest_sha256']:
                raise ValueError('review source identity changed')
            for name,digest in source['artifact_hashes'].items():
                if file_hash(path.parent/name)!=digest:raise ValueError('review source artifact changed')
        for name,digest in receipt['files_sha256'].items():
            if file_hash(output/name)!=digest:raise ValueError('review packet was changed; preserve and inspect it')
    else:
        receipt=prepare_aihub_expert_review(dest/'text-test.json',base/'predictions.jsonl',tuned/'predictions.jsonl',output)
    return {'status':receipt['status'],'selected_count':receipt['selected_count'],'human_review':receipt['human_review'],
            'manifest':str(manifest_path),'manifest_sha256':file_hash(manifest_path)}


def prepare(root,corpora_path):
    corpus=read(corpora_path);dest=root/'v2/text';dest.mkdir(parents=True,exist_ok=True)
    if corpus['seed']!=20261023 or corpus['max_length']!=2048:raise ValueError('unexpected corpus selection recipe')
    if {k:v['train_count'] for k,v in corpus['corpora'].items()}!={'ko':4000,'mix':8000}:
        raise ValueError('declared KO/MIX corpus sizes changed')
    if file_hash(corpus['corpora']['ko']['validation_file'])!=file_hash(corpus['corpora']['mix']['validation_file']):
        raise ValueError('KO and MIX must use the same checkpoint-selection validation')
    configs=dest/'configs';configs.mkdir(exist_ok=True)
    policy=resolve_resource_policy().as_dict()
    entries={}
    for model in ('qwen3_17b','qwen3_4b'):
        original=read(root/f'runs/{model}-v1-20261023/config.json')
        for condition in ('ko','mix'):
            data=corpus['corpora'][condition]
            for phase in ('smoke','full'):
                run_id=f'{model}-{condition}-{phase}-v2-20261026'
                config={k:v for k,v in original.items() if k not in {'total_gpu_hours','per_model_gpu_hours','wall_seconds'}}
                config.update(run_id=run_id,pair_id=f'{model}-text-v2-20261026',experiment_id='medvlm-ko-mix-v2-20261026',
                              output_dir=str(dest/'runs'/run_id),training_schema_version='medvlm.training.v2',
                              training_recipe='one_pass',mode='smoke' if phase=='smoke' else 'v2',
                              train_file=data['train_file'],validation_file=data['validation_file'],freeze_manifest=data['freeze_manifest'],
                              train_samples_cap=data['train_count'],max_optimizer_steps=20 if phase=='smoke' else math.ceil(data['train_count']/16),
                              validation_samples=48,checkpoint_every_steps=100,max_length=2048,
                              max_epochs=1,seed=20261023,resource_policy=policy,
                              parent_condition='original_base',initial_adapter=None)
                path=configs/(run_id+'.json');immutable(path,config)
                entries[run_id]={'config':str(path),'config_sha256':file_hash(path),
                                 'model':model,'condition':condition,'phase':phase,
                                 'selected_train_rows':data['train_count'],'max_optimizer_steps':config['max_optimizer_steps'],
                                 'full_pass_supervised_tokens':data['supervised_tokens']}
        eval_config={k:v for k,v in original.items() if k not in {'total_gpu_hours','per_model_gpu_hours','wall_seconds'}}
        eval_config.update(resource_policy=policy,max_length=4096,pair_id=f'{model}-text-v2-20261026')
        immutable(configs/(model+'-evaluation.json'),eval_config)
    med=read(root/'runs/medgemma15_4b-v1-20261023/config.json')
    text_original=read(root/'runs/qwen3_17b-v1-20261023/config.json')
    med={k:v for k,v in med.items() if k not in {'total_gpu_hours','per_model_gpu_hours','wall_seconds'}}
    med.update(resource_policy=policy,max_length=4096,pair_id='medgemma15_4b-text-base-v2-20261026')
    # Warmups use only English text validation, without loading a VQA adapter.
    for field in ('train_file','validation_file','freeze_manifest'):med[field]=text_original[field]
    immutable(configs/'medgemma15_4b-evaluation.json',med)
    aggregate=read(root/'manifests/current_freeze.json')
    paths=[aggregate['splits'][key+'/test']['snapshot_path'] for key in ('kormedmcqa','pubmedqa_labeled','medquad')]
    paths.append(corpus['aihub_test_file'])
    frozen=freeze_protocol(paths,dest/'text-selection.json',cap=0,
                           caps={'kormedmcqa':3009,'pubmedqa_labeled':500,'medquad':64,'aihub_71875':957})
    # Set the finite text generation budget before any v2 text predictions.
    frozen={k:v for k,v in frozen.items() if k!='protocol_hash'}
    frozen['max_new_tokens']={**frozen['max_new_tokens'],'medical_qa':256}
    frozen['protocol_hash']=canonical_hash(frozen)
    protocol=promote_protocol(frozen,dest/'text-test.json',prompt_policy='native-task-instructions.v1',
                              validation_evidence={'policy':'task-specific budget fixed prospectively; no test-driven search'},
                              previously_observed_test=True)
    plan={'schema_version':'medvlm.text-campaign.v2','seed':20261023,'resource_policy':policy,
          'corpora_manifest_sha256':file_hash(corpora_path),'runs':entries,
          'conditions':['base','en_v1','ko','mix'],'medgemma_condition':'base_text_only',
          'selected_training':{'ko':4000,'mix':8000},'mix_language_ratio':'same KO4000 + EN4000; 1:1 by examples',
          'effective_batch':16,'epochs':1,'smoke_updates_per_new_cell':20,
          'checkpoint_selection':'identical 48 source-balanced validation rows; seed 20261023',
          'learning_rate':0.0002,
          'lora':{'rank':8,'alpha':16,'dropout':0.05},'max_train_length':2048,'max_eval_input_length':4096,
          'evaluation_protocol_sha256':file_hash(dest/'text-test.json'),'evaluation_count_per_condition':len(protocol['sample_ids']),
          'evaluation_budgets':protocol['max_new_tokens'],'additional_seeds':'not selected',
          'limitations':['single seed','KO and MIX have different total exposure','v1 tests previously observed',
                         'AIHub generation/review provenance does not establish clinical accuracy','human_review not performed']}
    immutable(dest/'plan.json',plan)
    return dest,plan


def main():
    started=time.monotonic()
    parser=argparse.ArgumentParser();parser.add_argument('--data-root',required=True)
    parser.add_argument('--corpora-manifest');parser.add_argument('--phase',choices=['prepare','train','evaluate','all'],default='all')
    args=parser.parse_args();root=Path(args.data_root).resolve()
    policy=resolve_resource_policy()
    path=Path(args.corpora_manifest) if args.corpora_manifest else root/'v2/manifests/text-corpora-v2.json'
    dest=root/'v2/text';dest.mkdir(parents=True,exist_ok=True)
    lock=(dest/'campaign.lock').open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('text campaign already has an active owner')
    lock.seek(0);lock.truncate();lock.write(str(os.getpid()));lock.flush()
    dest,plan=prepare(root,path)
    immutable(root/'v2/review/plan.json',{'schema_version':'medvlm.review-plan.v2','seed':20261023,
              'models':['qwen3_17b','qwen3_4b'],'comparisons':['base-vs-ko','base-vs-mix'],
              'source':'aihub_71875','q_type_quotas':{'2':8,'3':8},'selected_by':'sha256(seed,sample_id)',
              'human_review':'not_performed','test_used_for_selection':False})
    state=read(dest/'state.json') if (dest/'state.json').exists() else {'plan_sha256':canonical_hash(plan),'training':{},'evaluation':{},'errors':{}}
    if state['plan_sha256']!=canonical_hash(plan):raise ValueError('campaign state plan mismatch')
    state.setdefault('reviews',{})
    prior_wall=float(state.get('campaign_wall_seconds',0.0))
    deadline=policy.deadline(started,prior_wall,scope='campaign')
    complete_v1._campaign_deadline=deadline
    def save():
        state.update(updated_at=utcnow(),campaign_pid=os.getpid(),campaign_wall_seconds=prior_wall+max(0.0,time.monotonic()-started))
        atomic_json(dest/'state.json',state)
    save()
    if args.phase=='prepare':
        state['campaign_pid']=None;state.setdefault('status','PREPARED_NOT_RUN');atomic_json(dest/'state.json',state);return 0
    if args.phase in ('all','train'):
        for model in ('qwen3_17b','qwen3_4b'):
            for condition in ('ko','mix'):
                for phase in ('smoke','full'):
                    run_id=f'{model}-{condition}-{phase}-v2-20261026';entry=plan['runs'][run_id]
                    state['active']=run_id;save()
                    if deadline is not None and time.monotonic()>=deadline:
                        state['training'][run_id]={'status':'SKIPPED_BUDGET','reason_code':'CAMPAIGN_WALL_SECONDS'}
                        state['active']=None;save();continue
                    try:
                        ok=train_stage(entry['config'],root) and reload_stage(entry['config'],root)
                        config=read(entry['config']);manifest=read(Path(config['output_dir'])/'manifest.json')
                        state['training'][run_id]={'status':'SUCCEEDED' if ok else 'FAILED_REPRODUCIBLE',
                            'output_dir':config['output_dir'],'optimizer_steps':manifest.get('optimizer_steps'),
                            'samples_processed':manifest.get('samples_processed'),'supervised_tokens':manifest.get('supervised_tokens'),
                            'reload_verified':manifest.get('reload_verified')}
                    except Exception as error:
                        state['training'][run_id]={'status':'FAILED_REPRODUCIBLE','reason_code':type(error).__name__}
                        (dest/(run_id+'-wrapper-failure.txt')).write_text(str(error));ok=False
                    state['active']=None;save()
                    print(json.dumps({'event':'training_stage_finished','run':run_id,**state['training'][run_id]}),flush=True)
                    if not ok:break
    if args.phase in ('all','evaluate'):
        for model in ('qwen3_17b','qwen3_4b','medgemma15_4b'):
            conditions=('base',) if model=='medgemma15_4b' else ('base','en_v1','ko','mix')
            outputs={}
            for condition in conditions:
                run_id=f'{model}-{condition}-text-v2-20261026';out=dest/'evaluations'/run_id
                state['active']=run_id;save()
                try:
                    adapter=None
                    if condition=='en_v1':adapter=read(root/f'runs/{model}-v1-20261023/selection.json')['checkpoint']
                    elif condition in ('ko','mix'):
                        train_id=f'{model}-{condition}-full-v2-20261026'
                        config=read(plan['runs'][train_id]['config'])
                        manifest,selection=verify_training_cache(config);verify_reload_cache(config,manifest,selection)
                        adapter=selection['checkpoint']
                    argv=[sys.executable,'-m','mediflow_medvlm.evaluation','evaluate','--config',str(dest/'configs'/(model+'-evaluation.json')),
                          '--protocol',str(dest/'text-test.json'),'--output-dir',str(out),'--pair-id',f'{model}-text-v2-20261026']
                    if adapter:argv+=['--adapter',adapter]
                    with (dest/(run_id+'.log')).open('a') as stream:
                        remaining=None if deadline is None else max(0.0,deadline-time.monotonic())
                        if remaining is not None and remaining<=0:
                            state['evaluation'][run_id]={'status':'SKIPPED_BUDGET','reason_code':'CAMPAIGN_WALL_SECONDS'}
                            state['active']=None;save();continue
                        limits=[x for x in (policy.job_timeout_seconds,remaining) if x is not None]
                        process=subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT)
                        state['active_pid']=process.pid;save()
                        print(json.dumps({'event':'evaluation_started','run':run_id,'pid':process.pid}),flush=True)
                        try:code=process.wait(timeout=min(limits) if limits else None)
                        except subprocess.TimeoutExpired:
                            process.terminate()
                            try:process.wait(timeout=60)
                            except subprocess.TimeoutExpired:process.kill();process.wait()
                            code=124
                    manifest=read(out/'run_manifest.json')
                    status=manifest['status'] if not code else 'FAILED_REPRODUCIBLE'
                    state['evaluation'][run_id]={'status':status,'exit_code':code,'output_dir':str(out),'predictions':manifest.get('predictions')}
                    if code==124:
                        expired=deadline is not None and time.monotonic()>=deadline
                        state['evaluation'][run_id].update(status='SKIPPED_BUDGET' if expired else 'FAILED_REPRODUCIBLE',
                            reason_code='CAMPAIGN_WALL_SECONDS' if expired else 'JOB_TIMEOUT_SECONDS')
                    if status=='SUCCEEDED':outputs[condition]=out
                except Exception as error:
                    state['evaluation'][run_id]={'status':'FAILED_REPRODUCIBLE','reason_code':type(error).__name__}
                    (dest/(run_id+'-wrapper-failure.txt')).write_text(str(error))
                state['active']=None;state['active_pid']=None;save()
                print(json.dumps({'event':'evaluation_finished','run':run_id,**state['evaluation'][run_id]}),flush=True)
            for condition,out in outputs.items():
                if condition=='base' or 'base' not in outputs:continue
                try:
                    compare(outputs['base'],out,dest/f'{model}-base-vs-{condition}.json')
                    previous=state['errors'].pop(model+'-'+condition,None)
                    if previous:state.setdefault('resolved_errors',[]).append({'cell':model+'-'+condition,'previous':previous,'resolved_at':utcnow()})
                except (OSError,ValueError,KeyError) as error:state['errors'][model+'-'+condition]=type(error).__name__;save()
                if condition in ('ko','mix'):
                    key=model+'-base-vs-'+condition
                    try:
                        previous_packet=state['reviews'].get(key)
                        if previous_packet and file_hash(previous_packet['manifest'])!=previous_packet['manifest_sha256']:
                            raise ValueError('previous review manifest changed; preserve and inspect it')
                        state['reviews'][key]=review_packet(root,dest,model,condition,outputs['base'],out)
                        previous=state['errors'].pop(key+'-review',None)
                        if previous:state.setdefault('resolved_errors',[]).append({'cell':key+'-review','previous':previous,'resolved_at':utcnow()})
                    except (OSError,ValueError,KeyError) as error:state['errors'][key+'-review']=type(error).__name__
                    save()
    state['status']='SUCCEEDED' if len(state['training'])==8 and len(state['evaluation'])==9 and len(state['reviews'])==4 and not state['errors'] and all(x['status']=='SUCCEEDED' for group in ('training','evaluation') for x in state[group].values()) else 'PARTIAL'
    state['campaign_wall_seconds']=prior_wall+max(0.0,time.monotonic()-started)
    state['campaign_pid']=None;state['updated_at']=utcnow();atomic_json(dest/'state.json',state)
    return 0


if __name__=='__main__':raise SystemExit(main())
