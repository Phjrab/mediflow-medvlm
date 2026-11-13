#!/usr/bin/env python3
"""One finite 2-model x 2-weight x 2-instruction x 2-budget validation diagnostic."""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from mediflow_medvlm.data.io import read_jsonl, write_jsonl
from mediflow_medvlm.evaluation.format_diagnostics import diagnostic_text, select_validation, descriptive_style
from mediflow_medvlm.evaluation.metrics import parse_answer, score_rows, aggregate
from mediflow_medvlm.evaluation.runner import load_protocol, verify_adapter_parent
from mediflow_medvlm.evaluation.v2 import generation_finish
from mediflow_medvlm.runtime import (atomic_json, canonical_hash, file_hash, directory_digest,
    utcnow, resolve_resource_policy, GPULease, BudgetLedger, append_event)


def read(path): return json.loads(Path(path).read_text())


def immutable(path, value):
    if path.exists():
        if read(path) != value: raise ValueError('IMMUTABLE_DIAGNOSTIC_CHANGED: '+path.name)
    else: atomic_json(path,value)


def code_identity():
    repo=Path(__file__).resolve().parents[1]
    names=['scripts/run_format_diagnostic_v5.py','src/mediflow_medvlm/evaluation/format_diagnostics.py',
           'src/mediflow_medvlm/training/models.py','src/mediflow_medvlm/training/rendering.py',
           'src/mediflow_medvlm/evaluation/metrics.py','src/mediflow_medvlm/runtime.py',
           'src/mediflow_medvlm/evaluation/runner.py','src/mediflow_medvlm/evaluation/v2.py',
           'src/mediflow_medvlm/data/io.py']
    return {name:file_hash(repo/name) for name in names}


def prepare(root):
    out=root/'v5';out.mkdir(mode=0o700,exist_ok=True)
    if (out/'plan.json').exists():
        plan=read(out/'plan.json')
        if plan['code_sha256']!=code_identity(): raise ValueError('DIAGNOSTIC_SOURCE_CHANGED')
        for name,sha in plan['files_sha256'].items():
            if file_hash(name)!=sha: raise ValueError('DIAGNOSTIC_INPUT_CHANGED')
        if file_hash(out/'validation.jsonl')!=plan['validation_sha256']:raise ValueError('VALIDATION_CHANGED')
        return plan
    corpus_config=read(root/'v2/text/configs/qwen3_17b-ko-full-v2-20261026.json')
    aggregate_freeze=read(root/'manifests/current_freeze.json')
    paths=[Path(corpus_config['validation_file']),Path(aggregate_freeze['splits']['kormedmcqa/validation']['snapshot_path'])]
    _,test_rows=load_protocol(root/'v2/text/text-test.json')
    rows=select_validation([r for p in paths for r in read_jsonl(p)],test_rows)
    write_jsonl(out/'validation.jsonl',rows)
    v4=read(root/'v4/state.json')
    if v4['status']!='SUCCEEDED' or v4['errors']:raise ValueError('FINISHED_V4_REQUIRED')
    configs={};adapters={};files={str(p):file_hash(p) for p in paths}
    files[str(root/'v2/text/text-test.json')]=file_hash(root/'v2/text/text-test.json')
    for model in ('qwen3_17b','qwen3_4b'):
        p=root/f'v2/text/configs/{model}-evaluation.json';configs[model]=str(p);files[str(p)]=file_hash(p)
        adapters[model]=v4['training'][f'{model}-category-random-full-v4-20261112']['selected_checkpoint']
    historical={str(p):file_hash(p) for folder in (root/'v4',Path(__file__).resolve().parents[1]/'reports/public_summary')
                for p in folder.rglob('*') if p.is_file() and p.suffix!='.lock'}
    immutable(out/'historical.json',historical)
    plan={'schema':'medvlm.format-diagnostic.v5','declared_at':utcnow(),'seed':20261023,
          'authorization':'2026-11-13 user: 다음 단계 진행및 지금까지의 결과들 다 깃허브에 올라가있나?',
          'selection':'24 existing validation rows: 8 AIHub MCQ, 8 PubMedQA, 2 per KorMed profession; seed hash; exact test IDs/inputs excluded',
          'validation_sha256':file_hash(out/'validation.jsonl'),'code_sha256':code_identity(),
          'files_sha256':files,'configs':configs,'adapters':adapters,
          'adapter_digests':{m:directory_digest(p)['sha256'] for m,p in adapters.items()},
          'models':['qwen3_17b','qwen3_4b'],'conditions':['base','tuned'],
          'instruction_languages':['original','ko'],'max_new_tokens':[16,256],
          'expected_prediction_records':384,'historical_sha256':file_hash(out/'historical.json'),
          'resource_policy':resolve_resource_policy().as_dict(),
          'parser':'unchanged strict-label.v1; no relaxed rescoring',
          'selection_of_new_checkpoint_or_test_protocol':False,'test_inference':False,
          'limitations':['Small previously available validation subset; not a blind test or clinical validation.',
                         'Korean translation changes instruction tokens; not an isolated medical reasoning measurement.',
                         'Different training tasks and token exposures confound attribution to random QA alignment.']}
    immutable(out/'plan.json',plan)
    return plan


def verify_worker(out,plan,model,condition):
    folder=out/'runs'/f'{model}-{condition}';manifest=read(folder/'manifest.json')
    if manifest['plan_sha256']!=canonical_hash(plan) or manifest['status']!='SUCCEEDED':raise ValueError('WORKER_NOT_VERIFIED')
    if file_hash(folder/'predictions.jsonl')!=manifest['predictions_sha256']:raise ValueError('PREDICTIONS_CHANGED')
    rows=read_jsonl(out/'validation.jsonl');preds=read_jsonl(folder/'predictions.jsonl')
    if len(preds)!=96:raise ValueError('MISSING_DIAGNOSTIC_PREDICTIONS')
    indexed={};by_id={r['sample_id']:r for r in rows}
    for p in preds:
        key=(p['instruction_language'],p['maximum'],p['sample_id'])
        if key in indexed:raise ValueError('DUPLICATE_DIAGNOSTIC_PREDICTION')
        row=by_id[p['sample_id']]
        if p['instruction_text_sha256']!=canonical_hash(diagnostic_text(row,p['instruction_language'])):raise ValueError('INSTRUCTION_CHANGED')
        if p['output_tokens']!=len(p['generated_token_ids']) or not 0<p['output_tokens']<=p['maximum']:raise ValueError('TOKEN_COUNT_CHANGED')
        indexed[key]=p
    prefix_matches=0
    for row in rows:
        for language in plan['instruction_languages']:
            a=indexed[(language,16,row['sample_id'])];b=indexed[(language,256,row['sample_id'])]
            if a['input_hash']!=canonical_hash(row['input']) or b['input_hash']!=a['input_hash']:raise ValueError('INPUT_CHANGED')
            if a['prompt_token_sha256']!=b['prompt_token_sha256']:raise ValueError('BUDGET_PROMPT_CHANGED')
            prefix_matches+=a['generated_token_ids']==b['generated_token_ids'][:len(a['generated_token_ids'])]
            for p in (a,b):
                answer,status=parse_answer(p['raw_output'],row['task_type'],row['input'].get('options'))
                if (answer,status)!=(p['parsed_answer'],p['status']):raise ValueError('STRICT_PARSER_CHANGED')
    return {**manifest,'budget_pair_prefix_matches':prefix_matches,'budget_pairs':48},preds


def worker(root,model_key,condition):
    import torch
    from mediflow_medvlm.training.models import load_model
    from mediflow_medvlm.training.rendering import native_encode
    plan=prepare(root);out=root/'v5';folder=out/'runs'/f'{model_key}-{condition}';folder.mkdir(parents=True,mode=0o700,exist_ok=True)
    if (folder/'manifest.json').exists() and read(folder/'manifest.json')['status']=='SUCCEEDED':
        verify_worker(out,plan,model_key,condition);return
    config=read(plan['configs'][model_key]);adapter=plan['adapters'][model_key] if condition=='tuned' else None
    if adapter and directory_digest(adapter)['sha256']!=plan['adapter_digests'][model_key]:raise ValueError('ADAPTER_CHANGED')
    run_id=f'{model_key}-{condition}-format-validation-v5-20261113'
    policy=resolve_resource_policy(config)
    if policy.as_dict()!=plan['resource_policy']:raise ValueError('RESOURCE_POLICY_CHANGED')
    ledger=BudgetLedger(root/'runtime/gpu_budget.json',policy=policy)
    manifest={'status':'RUNNING','plan_sha256':canonical_hash(plan),'run_id':run_id,'pid':os.getpid(),
              'started_at':utcnow(),'adapter':adapter,'test_inference':False,'human_review':'not_performed'}
    atomic_json(folder/'manifest.json',manifest)
    path=folder/'predictions.jsonl';existing=read_jsonl(path) if path.exists() else []
    done={(p['instruction_language'],p['maximum'],p['sample_id']) for p in existing}
    rows=read_jsonl(out/'validation.jsonl')
    try:
        with GPULease(root/'runtime/medvlm_gpu.lock',run_id):
            ledger.start(config['model_id'],run_id,'format_validation_diagnostic')
            try:
                model,processor,meta=load_model(config,training=False,adapter_path=adapter)
                model.eval();tokenizer=getattr(processor,'tokenizer',processor)
                manifest['model']=meta
                if adapter:manifest['verified_parent']=verify_adapter_parent(adapter,config,meta)
                # Validate retained partial predictions before treating their IDs as done.
                for p in existing:
                    row=next(r for r in rows if r['sample_id']==p['sample_id'])
                    if p['input_hash']!=canonical_hash(row['input']) or p['instruction_text_sha256']!=canonical_hash(diagnostic_text(row,p['instruction_language'])):
                        raise ValueError('PARTIAL_DIAGNOSTIC_INPUT_CHANGED')
                    if tokenizer.decode(p['generated_token_ids'],skip_special_tokens=True)!=p['raw_output']:
                        raise ValueError('PARTIAL_DIAGNOSTIC_OUTPUT_CHANGED')
                if len(done)!=len(existing):raise ValueError('DUPLICATE_PARTIAL_DIAGNOSTIC')
                # Warmup uses a declared validation item, outside measurement.
                warm=native_encode(processor,[{'role':'user','content':diagnostic_text(rows[0],'original')}],generation_prompt=True)
                warm={k:v.to(model.device) for k,v in warm.items()}
                with torch.inference_mode():model.generate(**warm,do_sample=False,max_new_tokens=4,pad_token_id=tokenizer.pad_token_id)
                torch.cuda.synchronize()
                for language in plan['instruction_languages']:
                    for maximum in plan['max_new_tokens']:
                        for row in rows:
                            if (language,maximum,row['sample_id']) in done:continue
                            if not ledger.can_continue(config['model_id']):raise RuntimeError('RESOURCE_LIMIT_REACHED')
                            began=time.monotonic();text=diagnostic_text(row,language)
                            encoded=native_encode(processor,[{'role':'user','content':text}],generation_prompt=True)
                            if encoded['input_ids'].shape[-1]>config['max_length']:raise ValueError('CONTEXT_OVERFLOW')
                            prompt_hash=canonical_hash({k:v.tolist() for k,v in encoded.items()})
                            encoded={k:v.to(model.device) for k,v in encoded.items()}
                            with torch.inference_mode():generated=model.generate(**encoded,do_sample=False,max_new_tokens=maximum,max_time=120,pad_token_id=tokenizer.pad_token_id)
                            torch.cuda.synchronize();ids=generated[0,encoded['input_ids'].shape[-1]:].tolist()
                            raw=tokenizer.decode(ids,skip_special_tokens=True)
                            parsed,status=parse_answer(raw,row['task_type'],row['input'].get('options'))
                            elapsed=time.monotonic()-began
                            finish=generation_finish(len(ids),ids[-1] if ids else None,maximum,model.generation_config.eos_token_id)
                            if elapsed>=120:raise TimeoutError('DIAGNOSTIC_GENERATION_TIMEOUT')
                            append_event(path,{'sample_id':row['sample_id'],'input_hash':canonical_hash(row['input']),
                                'instruction_language':language,'maximum':maximum,'prompt_token_sha256':prompt_hash,
                                'instruction_text_sha256':canonical_hash(text),'raw_output':raw,'generated_token_ids':ids,
                                'parsed_answer':parsed,'status':status,'output_tokens':len(ids),'finish_reason':finish,
                                'timing':{'e2e_seconds':elapsed},'attempted':True})
                manifest.update(status='SUCCEEDED',predictions_sha256=file_hash(path),predictions=len(read_jsonl(path)))
            finally:ledger.finish()
    except Exception as error:
        manifest.update(status='FAILED_REPRODUCIBLE',reason_code=type(error).__name__)
        (folder/'failure.txt').write_text(str(error));raise
    finally:
        manifest['finished_at']=utcnow();atomic_json(folder/'manifest.json',manifest)
    verify_worker(out,plan,model_key,condition)


def report(root,public):
    out=root/'v5';plan=prepare(root);rows=read_jsonl(out/'validation.jsonl');results={};native_pairs={};worker_receipts={}
    for model in plan['models']:
        for condition in plan['conditions']:
            receipt,preds=verify_worker(out,plan,model,condition);worker_receipts[model+'/'+condition]=receipt
            native_pairs[(model,condition)]={(p['instruction_language'],p['maximum'],p['sample_id']):p['prompt_token_sha256'] for p in preds}
            for language in plan['instruction_languages']:
                for maximum in plan['max_new_tokens']:
                    cell=[p for p in preds if p['instruction_language']==language and p['maximum']==maximum]
                    scores=score_rows(rows,cell);key=f'{model}/{condition}/{language}/{maximum}'
                    results[key]={'model':model,'condition':condition,'language':language,'maximum':maximum,
                        'n':len(cell),'valid':sum(p['status']=='answered' for p in cell),
                        'strict_correct':int(sum(p['exact_match'] for p in scores)),
                        'style':descriptive_style(cell),'cohorts':aggregate(scores)}
        if native_pairs[(model,'base')]!=native_pairs[(model,'tuned')]:raise ValueError('BASE_TUNED_PROMPTS_DIFFER')
        for field in ('native_files','chat_template_sha256','dtype'):
            if worker_receipts[model+'/base']['model'][field]!=worker_receipts[model+'/tuned']['model'][field]:raise ValueError('BASE_TUNED_NATIVE_MODEL_DIFFER')
    historical=read(out/'historical.json')
    if file_hash(out/'historical.json')!=plan['historical_sha256']:raise ValueError('HISTORICAL_RECEIPT_CHANGED')
    for path,sha in historical.items():
        if file_hash(path)!=sha:raise ValueError('HISTORICAL_ARTIFACT_CHANGED')
    # Read-only posthoc description of existing test outputs, with no alternative scoring.
    posthoc={}
    _,test=load_protocol(root/'v2/text/text-test.json');classification={r['sample_id'] for r in test if r['task_type'] in {'medical_mcq','context_qa'}}
    for model in plan['models']:
        conditions={}
        for condition in plan['conditions']:
            folder=(root/f'v2/text/evaluations/{model}-base-text-v2-20261026' if condition=='base' else root/f'v4/evaluations/{model}-common-tuned-v4-20261112')
            m=read(folder/'run_manifest.json')
            for name,sha in m['artifact_hashes'].items():
                if file_hash(folder/name)!=sha:raise ValueError('HISTORICAL_PREDICTION_CHANGED')
            p=[p for p in read_jsonl(folder/'predictions.jsonl') if p['sample_id'] in classification]
            conditions[condition]=p
        left={p['sample_id']:p for p in conditions['base']};right={p['sample_id']:p for p in conditions['tuned']}
        if left.keys()!=right.keys():raise ValueError('HISTORICAL_PAIR_IDS_DIFFER')
        if any(left[k]['prompt_token_sha256']!=right[k]['prompt_token_sha256'] for k in left):raise ValueError('HISTORICAL_PAIR_PROMPTS_DIFFER')
        transitions=Counter(('valid' if left[k]['status']=='answered' else 'invalid')+'->'+('valid' if right[k]['status']=='answered' else 'invalid') for k in left)
        posthoc[model]={'base':descriptive_style(conditions['base']),'tuned':descriptive_style(conditions['tuned']),'format_transitions':dict(transitions)}
    budget=read(root/'runtime/gpu_budget.json');owned=[f'{m}-{c}-format-validation-v5-20261113' for m in plan['models'] for c in plan['conditions']]
    if budget.get('active') and budget['active']['run_id'] in owned:raise ValueError('GPU_STILL_ACTIVE')
    gpu={name:budget['run_phases'][name]['format_validation_diagnostic'] for name in owned}
    summary={'status':'SUCCEEDED','schema':'medvlm.format-diagnostic.results.v5','validation_only':True,
             'new_training':False,'new_test_inference':False,'historical_test_scores_changed':False,
             'clinical_accuracy':None,'human_review':'not_performed','scheduled_records':384,
             'plan_sha256':file_hash(out/'plan.json'),'cells':results,'historical_posthoc':posthoc,
             'budget_pair_prefix_checks':{k:{f:v[f] for f in ('budget_pair_prefix_matches','budget_pairs')} for k,v in worker_receipts.items()},
             'gpu_seconds_by_run':gpu,'gpu_seconds_total':sum(gpu.values()),
             'historical_files_verified_unchanged':len(historical),'limitations':plan['limitations']}
    public.mkdir(parents=True,exist_ok=True);atomic_json(public/'run_summary.json',summary)
    lines=['# 71762 후속 형식 진단 — 2026-11-13','','현재 단계: 사전 고정한 검증셋 전용384건 진단을 완료했습니다.','',
        '이번에 한 일: 두 모델 Base/Tuned에 동일24검증문항(세 source 각8개), 기존/한국어 번역 지시문, 16/256토큰을 교차했습니다. 새 학습·test 추론·기존 점수 보정은 하지 않았습니다.','',
        '확인 결과: 유효응답은 기존 strict parser가 받아들인 형식입니다. 정확도 수치는 이 작은 검증 부분집합의 엄격한 문항 점수이며 임상 정확도나 새 test 성능이 아닙니다.','',
        '| 모델 | 조건 | 지시문 | token 상한 | 유효응답/24 | 엄격 정답/24 | 길이 종료 |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for c in results.values():lines.append(f"| {c['model']} | {c['condition']} | {c['language']} | {c['maximum']} | {c['valid']} | {c['strict_correct']} | {c['style']['length_terminated']} |")
    matched=sum(v['budget_pair_prefix_matches'] for v in worker_receipts.values())
    lines+=['',f"실제 GPU 점유 {sum(gpu.values())/3600:.3f}시간. 과거 파일{len(historical)}개 hash 보존을 확인했습니다. 같은 조건의 Base/Tuned native prompt가 일치했고, 16→256의 greedy token prefix는 {matched}/192쌍 일치했습니다.",'',
        '한계: 검증 문항24개뿐이며 기존에 이용 가능한 validation입니다. 번역은 native instruction token을 바꾸며, 문체·과제 분포·학습 노출량 차이와 무작위 QA 대응의 효과를 분리하지 못합니다. 기존 test 출력의 재분석은 관측 후 기술 통계이며 relaxed parser 정확도를 만들지 않았습니다.','',
        '다음 단계: 이 진단은 학습·정식 test 프로토콜을 자동 선택하지 않습니다. 결과에 따른 무한 재학습·문항 재추첨·추가seed를 실행하지 않습니다.','',
        '재개 명령(저장소에서): `PYTHONPATH=src ../.venv-medvlm/bin/python scripts/run_format_diagnostic_v5.py --data-root ../medvlm-data`','',
        '사용자가 할 일: 필수 입력은 없습니다. 전문가 검토는 미실행이며 Jetson은 사용자 보류입니다.','']
    (public/'RESULTS_KO.md').write_text('\n'.join(lines))
    atomic_json(out/'verification.json',{'status':'SUCCEEDED','records':384,'historical_files_unchanged':len(historical),'report_sha256':file_hash(public/'run_summary.json')})
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--worker',choices=['qwen3_17b','qwen3_4b']);p.add_argument('--condition',choices=['base','tuned']);p.add_argument('--prepare-only',action='store_true');a=p.parse_args();root=a.data_root.resolve()
    if a.worker:worker(root,a.worker,a.condition);return
    out=root/'v5';out.mkdir(mode=0o700,exist_ok=True)
    with (out/'campaign.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);plan=prepare(root)
        if a.prepare_only:print(json.dumps({'status':'PREPARED_NOT_RUN','records':384}));return
        state=read(out/'state.json') if (out/'state.json').exists() else {'completed':{},'errors':{}}
        for model in plan['models']:
            for condition in plan['conditions']:
                key=model+'/'+condition;state.update(status='RUNNING',active=key);atomic_json(out/'state.json',state)
                try:
                    folder=out/'runs'/f'{model}-{condition}';folder.mkdir(parents=True,exist_ok=True)
                    with (folder/'worker.log').open('a') as log:
                        process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--data-root',str(root),'--worker',model,'--condition',condition],stdout=log,stderr=subprocess.STDOUT)
                        state['pid']=process.pid;atomic_json(out/'state.json',state);code=process.wait()
                    if code:raise RuntimeError('WORKER_EXIT_'+str(code))
                    verify_worker(out,plan,model,condition);state['completed'][key]='SUCCEEDED';state['errors'].pop(key,None)
                except Exception as error:state['errors'][key]=type(error).__name__
                atomic_json(out/'state.json',state);print(json.dumps({'completed':len(state['completed']),'errors':state['errors']}),flush=True)
        state.update(active=None,pid=None,status='SUCCEEDED' if len(state['completed'])==4 and not state['errors'] else 'PARTIAL');atomic_json(out/'state.json',state)
        if state['status']=='SUCCEEDED':report(root,Path(__file__).resolve().parents[1]/'reports/public_summary/v5')
        else:raise RuntimeError('DIAGNOSTIC_INCOMPLETE')


if __name__=='__main__':main()
