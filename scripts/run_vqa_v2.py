#!/usr/bin/env python3
"""Finite validation-first reevaluation of the two preserved v1 VLM pairs."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from mediflow_medvlm.runtime import atomic_json, canonical_hash, file_hash, utcnow, resolve_resource_policy
from mediflow_medvlm.evaluation.runner import (freeze_protocol, load_protocol, compare,
    validated_predictions, verify_completed_evaluation)
from mediflow_medvlm.evaluation.v2 import promote_protocol, VQA_PROMPT
from mediflow_medvlm.evaluation.image_diagnostics import derive_no_image_protocol, compare_image_conditions


def read(path):
    return json.loads(Path(path).read_text())


def immutable(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError('immutable campaign declaration changed')
    else:
        atomic_json(path, value)


def verify_validation_decision(plan, decision, expected_protocols, *, run_root):
    """Verify frozen validation evidence without rerunning or changing the decision.

    Older decisions may omit ready_models; readiness is derived from the complete
    model/mode schedule. Failed cells stay failed and cannot qualify another model.
    Native cache identity is checked again by evaluate before each test run; here
    the immutable validation manifest and its artifact receipts are the evidence.
    """
    if decision.get('plan_sha256') != canonical_hash(plan):
        raise ValueError('validation decision plan mismatch')
    if decision.get('test_used_for_selection') is not False:
        raise ValueError('validation decision used test results')
    budget = decision.get('chosen_max_vqa_tokens')
    if type(budget) is not int or budget not in (128, 256):
        raise ValueError('invalid validation token budget')
    budgets = (128, 256) if budget == 256 else (128,)
    models = tuple(plan['models'])
    expected = {(model, mode, limit) for model in models for mode in ('base', 'tuned') for limit in budgets}
    observations = decision.get('observations')
    if not isinstance(observations, list) or len(observations) != len(expected):
        raise ValueError('incomplete validation observation schedule')
    protocols = {}
    for limit in budgets:
        protocol, rows = load_protocol(expected_protocols[limit])
        if (protocol.get('split') != 'validation' or len(rows) != plan['validation_samples']
                or protocol.get('seed') != plan['seed']
                or protocol.get('protocol_version') != 'medvlm.eval.v2'
                or protocol.get('prompt_policy') != plan['prompt_policy']
                or protocol.get('normalization_policy') != 'nfkc-casefold-whitespace-terminal-punctuation.v2'
                or protocol.get('max_new_tokens', {}).get('medical_vqa') != limit
                or protocol.get('test_used_for_selection') is not False
                or protocol.get('do_sample') is not False):
            raise ValueError('validation protocol differs from the frozen plan')
        protocols[limit] = protocol
    seen = {}
    for record in observations:
        run = Path(record['run']).resolve()
        matches = [key for key in expected if run == (Path(run_root) /
            f'{key[0]}-validation{key[2]}-{key[1]}-v2-20261026').resolve()]
        if len(matches) != 1:
            raise ValueError('unexpected validation run path')
        model, mode, limit = key = matches[0]
        if key in seen or record.get('model') != model or record.get('mode') != mode:
            raise ValueError('duplicate or mislabeled validation observation')
        if record.get('status') not in {'SUCCEEDED', 'FAILED_REPRODUCIBLE', 'SKIPPED_BUDGET'}:
            raise ValueError('invalid validation observation status')
        manifest_path = run / 'run_manifest.json'
        manifest = None
        if record.get('manifest_sha256') is not None:
            if file_hash(manifest_path) != record['manifest_sha256']:
                raise ValueError('validation manifest changed')
            manifest = read(manifest_path)
            if (manifest.get('run_id') != run.name or manifest.get('protocol_hash') != protocols[limit]['protocol_hash']
                    or manifest.get('scheduled') != plan['validation_samples']):
                raise ValueError('validation manifest/protocol mismatch')
        elif manifest_path.exists():
            raise ValueError('validation manifest has no frozen receipt')
        predictions = validated_predictions(run / 'predictions.jsonl', protocols[limit], manifest)
        errors = sum(p.get('status') not in {'answered', 'abstained', 'invalid_format'} for p in predictions)
        lengths = sum(p.get('finish_reason') == 'length' for p in predictions)
        # The first wrapper did not store these two redundant counts. Its manifest,
        # artifact hashes, full ID coverage and actual row statuses remain required.
        for field, value in (('predictions', len(predictions)), ('technical_errors', errors)):
            if field in record and record[field] != value:
                raise ValueError('validation observation count mismatch: ' + field)
        if record.get('length_count') != lengths:
            raise ValueError('validation length count mismatch')
        if record['status'] == 'SUCCEEDED':
            if (manifest is None or manifest.get('status') != 'SUCCEEDED'
                    or len(predictions) != plan['validation_samples'] or errors):
                raise ValueError('validation success lacks complete successful evidence')
            verify_completed_evaluation(manifest, protocols[limit], run,
                                        manifest['model']['native_files']['sha256'])
        seen[key] = record
    if set(seen) != expected:
        raise ValueError('incomplete validation observation schedule')
    selected_budget = 256 if any(record['length_count'] for key, record in seen.items() if key[2] == 128) else 128
    if budget != selected_budget:
        raise ValueError('validation token budget violates the frozen rule')
    ready = [model for model in models if all(seen[(model, mode, limit)]['status'] == 'SUCCEEDED'
             for mode in ('base', 'tuned') for limit in budgets)]
    if 'ready_models' in decision and decision['ready_models'] != ready:
        raise ValueError('validation readiness differs from verified observations')
    return ready


def resolve_comparison_error(state, key):
    """Keep previous failures as history while allowing a successful retry to finish."""
    previous = state.get('comparison_errors', {}).pop(key, None)
    if previous is not None:
        state.setdefault('resolved_comparison_errors', []).append(
            {'key': key, 'error': previous, 'resolved_at': utcnow()})


def job_timeout(policy, campaign_deadline, *, now=None):
    remaining = None if campaign_deadline is None else max(0.0, campaign_deadline - (time.monotonic() if now is None else now))
    limits = [value for value in (policy.job_timeout_seconds, remaining) if value is not None]
    return min(limits) if limits else None


def main():
    started = time.monotonic()
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--validation-only', action='store_true')
    args=parser.parse_args()
    root=Path(args.data_root).resolve(); dest=root/'v2/vqa'
    dest.mkdir(parents=True,exist_ok=True)
    campaign_lock=(dest/'campaign.lock').open('a+')
    try:fcntl.flock(campaign_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('this VQA campaign already has an active owner')
    campaign_lock.seek(0);campaign_lock.truncate();campaign_lock.write(str(os.getpid()));campaign_lock.flush()
    protocols=dest/'protocols'; protocols.mkdir(exist_ok=True)
    models=('qwen35_2b','medgemma15_4b')
    plan={'schema_version':'medvlm.vqa-followup.v2','models':list(models),
          'conditions':['base','tuned_v1'],'seed':20261023,
          'training':'none; reuse preserved original base and selected v1 adapter',
          'precision':'same NF4 within each base/tuned pair',
          'validation_samples':32,'initial_vqa_max_new_tokens':128,
          'budget_rule':'if any completed validation answer has finish_reason length, validate 256 once and use 256; otherwise 128',
          'prompt_policy':VQA_PROMPT,'normalization':'NFKC/casefold/whitespace/terminal .!? only',
          'test_sources':{'slake':1061,'vqa_rad':451},'no_image_samples':32,
          'test_exposure':'contains v1 observed test samples; not a blind benchmark',
          'test_used_for_selection':False,'resource_policy':resolve_resource_policy().as_dict()}
    immutable(dest/'plan.json',plan)
    state=read(dest/'state.json') if (dest/'state.json').exists() else {'plan_sha256':canonical_hash(plan),'runs':{}}
    if state['plan_sha256']!=canonical_hash(plan):raise ValueError('state plan mismatch')
    resource_policy=resolve_resource_policy()
    prior_wall=float(state.get('campaign_wall_seconds',0.0))
    deadline=resource_policy.deadline(started,prior_wall,scope='campaign')
    def save():
        state['updated_at']=utcnow()
        state['campaign_wall_seconds']=prior_wall+max(0.0,time.monotonic()-started)
        atomic_json(dest/'state.json',state)
    configs={m:root/f'runs/{m}-v1-20261023/config.json' for m in models}
    def run(model, stage, mode, protocol):
        key=f'{model}-{stage}-{mode}-v2-20261026'; output=dest/'runs'/key
        output.parent.mkdir(exist_ok=True)
        timeout=job_timeout(resource_policy,deadline)
        if timeout is not None and timeout <= 0:
            state['runs'][key]={'status':'SKIPPED_BUDGET','reason_code':'CAMPAIGN_WALL_SECONDS',
                                'output_dir':str(output),'protocol_sha256':file_hash(protocol)}
            save();return output,False
        argv=[sys.executable,'-m','mediflow_medvlm.evaluation','evaluate','--config',str(configs[model]),
              '--protocol',str(protocol),'--output-dir',str(output),'--pair-id',f'{model}-evaluation-v2-20261026']
        if mode=='tuned':
            try:argv+=['--adapter',read(root/f'runs/{model}-v1-20261023/selection.json')['checkpoint']]
            except (OSError,ValueError,KeyError) as error:
                state['runs'][key]={'status':'FAILED_REPRODUCIBLE','reason_code':'SELECTION_'+type(error).__name__}
                save();return output,False
        state['active']=key; save()
        log=dest/(key+'.log')
        with log.open('a') as stream:
            process=subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT)
            state['active_pid']=process.pid;save()
            print(json.dumps({'event':'started','run':key,'pid':process.pid,'at':utcnow()}),flush=True)
            try:code=process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:process.wait(timeout=60)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                code=124
        manifest=read(output/'run_manifest.json') if (output/'run_manifest.json').exists() else {}
        result={'status':manifest.get('status','FAILED_REPRODUCIBLE'),'exit_code':code,
                'output_dir':str(output),'protocol_sha256':file_hash(protocol),'log':str(log)}
        if code:result['status']='FAILED_REPRODUCIBLE'
        if code==124:
            expired=deadline is not None and time.monotonic()>=deadline
            result.update(status='SKIPPED_BUDGET' if expired else 'FAILED_REPRODUCIBLE',
                          reason_code='CAMPAIGN_WALL_SECONDS' if expired else 'JOB_TIMEOUT_SECONDS')
        state['runs'][key]=result; state['active']=None;state['active_pid']=None;save()
        print(json.dumps({'event':'finished','run':key,**result}),flush=True)
        return output,result['status']=='SUCCEEDED'
    v1val=read(root/'manifests/evaluation/slake-validation-v1.json')
    frozen=freeze_protocol(list(v1val['files']),protocols/'validation-selection.json',split='validation',cap=32)
    def validation(budget):
        protocol=protocols/f'validation-{budget}.json'
        promote_protocol(frozen,protocol,prompt_policy=VQA_PROMPT,max_vqa_tokens=budget,previously_observed_test=False)
        records=[]
        for model in models:
            for mode in ('base','tuned'):
                output,ok=run(model,f'validation{budget}',mode,protocol)
                predictions=[json.loads(line) for line in (output/'predictions.jsonl').read_text().splitlines()] if (output/'predictions.jsonl').exists() else []
                errors=sum(p.get('status') not in {'answered','abstained','invalid_format'} for p in predictions)
                ok=ok and len(predictions)==32 and errors==0
                cell=state['runs'][output.name]
                records.append({'model':model,'mode':mode,'status':'SUCCEEDED' if ok else cell['status'] if cell['status']=='SKIPPED_BUDGET' else 'FAILED_REPRODUCIBLE',
                                'reason_code':cell.get('reason_code'),
                                'predictions':len(predictions),'length_count':sum(p.get('finish_reason')=='length' for p in predictions),
                                'technical_errors':errors,
                                'run':str(output),'manifest_sha256':file_hash(output/'run_manifest.json') if (output/'run_manifest.json').exists() else None})
        return records
    decision_path=dest/'validation-decision.json'
    if decision_path.exists():
        decision=read(decision_path)
    else:
        observations=validation(128)
        budget=256 if any(x['length_count'] for x in observations) else 128
        if budget==256:
            more=validation(256);observations+=more
        ready=[model for model in models if all(x['status']=='SUCCEEDED' for x in observations if x['model']==model)]
        decision={'plan_sha256':canonical_hash(plan),'chosen_max_vqa_tokens':budget,
                  'observations':observations,'ready_models':ready,
                  'test_used_for_selection':False,'frozen_at':utcnow()}
        atomic_json(decision_path,decision)
    ready=verify_validation_decision(plan,decision,
        {limit:protocols/f'validation-{limit}.json' for limit in (128,256)},run_root=dest/'runs')
    state['status']='VALIDATION_COMPLETE' if len(ready)==len(models) else 'PARTIAL_VALIDATION';save()
    if args.validation_only:return 0
    aggregate=read(root/'manifests/current_freeze.json')
    for label,source in (('slake','slake'),('vqarad','vqa_rad')):
        path=aggregate['splits'][source+'/test']['snapshot_path']
        frozen=freeze_protocol([path],protocols/f'{label}-selection.json',cap=0)
        protocol=protocols/f'{label}-test.json'
        promote_protocol(frozen,protocol,prompt_policy=VQA_PROMPT,max_vqa_tokens=decision['chosen_max_vqa_tokens'],
                         validation_evidence={'path':str(decision_path),'sha256':file_hash(decision_path)})
        for model in models:
            if model not in ready:continue
            results={}
            for mode in ('base','tuned'):
                output,ok=run(model,label,mode,protocol)
                if ok:results[mode]=output
            if len(results)==2:
                try:
                    compare(results['base'],results['tuned'],dest/f'{model}-{label}-comparison.json')
                    resolve_comparison_error(state,model+'-'+label);save()
                except (OSError,ValueError,KeyError) as error:
                    state.setdefault('comparison_errors',{})[model+'-'+label]=type(error).__name__;save()
            if label=='slake':
                noimage=protocols/'slake-no-image.json'
                derive_no_image_protocol(protocol,noimage,cap=32)
                for mode in ('base','tuned'):
                    output,ok=run(model,'noimage',mode,noimage)
                    if ok and mode in results:
                        try:
                            compare_image_conditions(protocol,noimage,results[mode]/'predictions.jsonl',output/'predictions.jsonl',dest/f'{model}-{mode}-image-diagnostic.json')
                            resolve_comparison_error(state,model+'-'+mode+'-image');save()
                        except (OSError,ValueError,KeyError) as error:
                            state.setdefault('comparison_errors',{})[model+'-'+mode+'-image']=type(error).__name__;save()
    state['status']='SUCCEEDED' if len(ready)==len(models) and not state.get('comparison_errors') and all(v['status']=='SUCCEEDED' for v in state['runs'].values()) else 'PARTIAL_WITH_FAILURES'
    save();return 0


if __name__=='__main__':raise SystemExit(main())
