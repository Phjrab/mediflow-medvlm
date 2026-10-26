#!/usr/bin/env python3
"""Read-only v1 artifact reanalysis. Public outputs contain aggregates, not text."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import re
import statistics

from mediflow_medvlm.runtime import atomic_json, canonical_hash, directory_digest, file_hash
from mediflow_medvlm.evaluation.runner import load_protocol, validated_predictions
from mediflow_medvlm.evaluation.metrics import score_rows, normalize


def read(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verified_pair(root, key, scope, protocol_name):
    pp = root / 'manifests/evaluation' / protocol_name
    protocol, samples = load_protocol(pp)
    config = read(root / 'configs' / f'{key}-v1-20261023.json')
    train = read(root / 'runs' / f'{key}-v1-20261023' / 'manifest.json')
    results, receipts = [], []
    for condition in ('base', 'tuned'):
        folder = root / 'runs' / f'{key}-{scope}-{condition}-20261023'
        run = read(folder / 'run_manifest.json')
        require(run['status'] == 'SUCCEEDED', 'v1 run is not successful')
        for name, expected in run['artifact_hashes'].items():
            require(file_hash(folder / name) == expected, 'recorded v1 artifact hash changed: '+folder.name+'/'+name)
        require(run['config_hash'] == canonical_hash(config) == train['config_hash'], 'config identity changed')
        require(run['protocol_hash'] == protocol['protocol_hash'], 'protocol identity changed')
        require(run['model_id'] == config['model_id'] and run['model_revision'] == config['revision'], 'model identity changed')
        require(run['model']['native_files']['sha256'] == train['model']['native_files']['sha256'], 'recorded native model identity changed')
        identity_keys = ('config_hash','protocol_hash','model_id','model_revision','adapter_path','pair_id','renderer_version','image_condition')
        identity = {k: run[k] for k in identity_keys}
        if run.get('adapter_path'):
            identity['adapter_digest'] = run['adapter_digest']
            require(directory_digest(run['adapter_path'])['sha256'] == run['adapter_digest'], 'adapter artifact changed')
        require(canonical_hash(identity) == run['identity_hash'], 'run identity hash changed')
        preds = validated_predictions(folder / 'predictions.jsonl', protocol, run)
        require(len(preds) == run['predictions'] == len(samples), 'incomplete v1 predictions')
        results.append((run, preds))
        receipts.append({'run_id':run['run_id'],'model_id':run['model_id'],'model_revision':run['model_revision'],
                         'run_manifest_sha256':file_hash(folder/'run_manifest.json'),'predictions_sha256':file_hash(folder/'predictions.jsonl'),
                         'identity_hash':run['identity_hash'],'native_model_recorded_digest':run['model']['native_files']['sha256'],
                         'adapter_digest':run.get('adapter_digest'),'protocol_hash':protocol['protocol_hash'],
                         'protocol_file_sha256':file_hash(pp),'scheduled':len(samples),'precision':run['precision'],
                         'generation':run['generation'],'verification':'recorded prediction/artifact hashes, frozen input/reference/image bytes, config and run/model identities, adapter bytes'})
    for field in ('model_id','model_revision','config_hash','protocol_hash','pair_id','precision','generation'):
        require(results[0][0][field] == results[1][0][field], 'pair conditions differ: '+field)
    return samples, results, receipts


def correctness_partition(samples, left, right):
    a = {r['sample_id']:r for r in score_rows(samples,left)}
    b = {r['sample_id']:r for r in score_rows(samples,right)}
    groups = defaultdict(list)
    for sid, x in a.items():
        y=b[sid]
        key = ('both_valid' if x['valid'] and y['valid'] else 'invalid_to_valid' if y['valid'] else
               'valid_to_invalid' if x['valid'] else 'both_invalid')
        groups[key].append((x,y))
    result=[]
    for key in ('both_valid','invalid_to_valid','valid_to_invalid','both_invalid'):
        pairs=groups[key];n=len(pairs);ac=sum(x['exact_match'] for x,y in pairs);bc=sum(y['exact_match'] for x,y in pairs)
        result.append({'partition':key,'support':n,'base_correct':int(ac),'tuned_correct':int(bc),
                       'base_correct_fraction':ac/n if n else None,'tuned_correct_fraction':bc/n if n else None,
                       'improved':sum(y['exact_match']>x['exact_match'] for x,y in pairs),
                       'regressed':sum(y['exact_match']<x['exact_match'] for x,y in pairs),
                       'unchanged':sum(y['exact_match']==x['exact_match'] for x,y in pairs),
                       'net_correct_delta':int(bc-ac),'contribution_to_total_delta':(bc-ac)/len(samples),
                       'base_statuses':dict(Counter(x['status'] for x,y in pairs)),
                       'tuned_statuses':dict(Counter(y['status'] for x,y in pairs))})
    return result


def distribution(values):
    values=sorted(values)
    return {'n':len(values),'mean':statistics.mean(values) if values else None,
            'median':statistics.median(values) if values else None,
            'p95':values[max(0,math.ceil(.95*len(values))-1)] if values else None,'max':max(values) if values else None}


def vqa_diagnostics(samples,preds):
    lookup={p['sample_id']:p for p in preds};scores={r['sample_id']:r for r in score_rows(samples,preds)}
    output=[]
    for group in ('ALL','OPEN','CLOSED'):
        rows=[s for s in samples if group=='ALL' or str(s.get('metadata',{}).get('answer_type','')).upper()==group]
        if not rows:continue
        pairs=[(s,lookup[s['sample_id']],scores[s['sample_id']]) for s in rows]
        contains=[]
        for s,p,_ in pairs:
            gold=normalize(s['supervision']['answer']);answer=normalize(p.get('raw_output') or p.get('parsed_answer') or '')
            # Unicode word boundaries prevent e.g. 'no' matching 'normal'. Negation is not removed.
            contains.append(bool(gold and re.search(r'(?<!\w)'+re.escape(gold)+r'(?!\w)',answer)))
        output.append({'answer_type':group,'scheduled':len(rows),'valid':sum(r['valid'] for _,_,r in pairs),
                       'lexical_exact_matches':int(sum(r['exact_match'] for _,_,r in pairs)),
                       'lexical_exact_match':sum(r['exact_match'] for _,_,r in pairs)/len(rows),
                       'lexical_token_f1':sum(r['token_f1'] for _,_,r in pairs)/len(rows),
                       'reference_phrase_contains_count':sum(contains),'reference_phrase_contains_fraction':sum(contains)/len(rows),
                       'reference_phrase_is_accuracy':False,'truncated_count':sum(p.get('finish_reason')=='length' for _,p,_ in pairs),
                       'output_characters':distribution([len(p.get('raw_output') or '') for _,p,_ in pairs]),
                       'output_whitespace_words':distribution([len((p.get('raw_output') or '').split()) for _,p,_ in pairs]),
                       'generated_tokens':distribution([p['output_tokens'] for _,p,_ in pairs if 'output_tokens' in p]),
                       'reference_characters':distribution([len(str(s['supervision']['answer'])) for s,_,_ in pairs])})
    return output


def training_counts(root,key):
    folder=root/'runs'/f'{key}-v1-20261023';config=read(folder/'config.json');m=read(folder/'manifest.json')
    require(canonical_hash(config)==m['config_hash'],'training config changed')
    require(file_hash(config['freeze_manifest'])==m['data']['freeze_manifest_sha256'],'training freeze changed')
    require(file_hash(config['train_file'])==m['data']['train']['sha256'],'training data changed')
    rows=jsonl(config['train_file']);by_id={r['sample_id']:r for r in rows}
    audit=jsonl(folder/'mask_audit.jsonl');excluded=read(folder/'excluded.json')
    ordered=[r['sample_id'] for r in audit];excluded_ids={r['sample_id'] for r in excluded}
    require(not set(ordered)&excluded_ids and len(set(ordered))==len(ordered),'preflight duplicated samples')
    order=list(range(len(rows)));random.Random(config['seed']).shuffle(order)
    candidate=[rows[i]['sample_id'] for i in order[:min(config.get('train_samples_cap',20000),20000)]]
    require([sid for sid in candidate if sid not in excluded_ids]==ordered,'sampler/preflight order changed')
    require(canonical_hash(ordered)==m['training_selection']['sample_order_sha256'],'sample order hash changed')
    cpdir=Path(read(folder/'latest.json')['checkpoint']);cp=read(cpdir/'checkpoint.json')
    require(cp['sample_order_sha256']==canonical_hash(ordered),'checkpoint order differs')
    processed=ordered[:cp['next_sample_position']]
    require(len(processed)==cp['samples_processed']==m['samples_processed'],'processed counts differ')
    events=[x for x in jsonl(folder/'events.jsonl') if x.get('event')=='optimizer_update']
    require(sum(x['samples'] for x in events)==len(processed),'event processed count differs')
    require(sum(x['target_tokens'] for x in events)==m['supervised_tokens'],'event supervised token total differs')
    counts=[]
    for source in sorted({r['source_id'] for r in rows}):
        belongs=lambda sid:by_id[sid]['source_id']==source
        counts.append({'source_id':source,'prepared':sum(r['source_id']==source for r in rows),
                       'selected_for_preflight':sum(belongs(sid) for sid in candidate),
                       'rejected':sum(belongs(sid) for sid in excluded_ids),
                       'rejection_reasons':dict(Counter(x['reason'] for x in excluded if belongs(x['sample_id']))),
                       'usable':sum(belongs(sid) for sid in ordered),'processed':sum(belongs(sid) for sid in processed),
                       'supervised_tokens':sum(x['supervised_tokens_after_shift'] for x in audit if belongs(x['sample_id']))})
    return {'model':key,'training_selection':m['training_selection'],'sources':counts,
            'max_length':config['max_length'],'sample_cap':config.get('train_samples_cap',20000),
            'effective_batch':config['gradient_accumulation_steps'],'max_optimizer_steps':config['max_optimizer_steps'],
            'actual_optimizer_steps':m['optimizer_steps'],'expected_updates':math.ceil(len(ordered)/config['gradient_accumulation_steps']),
            'remaining_unprocessed_usable':len(ordered)-len(processed),'status':m['status'],
            'cause':'50 whole samples rejected by native template CONTEXT_OVERFLOW; no step-cap or budget omission',
            'receipts':{n:file_hash(folder/n) for n in ('config.json','manifest.json','excluded.json','mask_audit.jsonl','events.jsonl')},
            'checkpoint_json_sha256':file_hash(cpdir/'checkpoint.json'),'train_sha256':file_hash(config['train_file'])}


def main(root,output):
    receipts=[];korean=[];vqa=[]
    for key in ('qwen3_17b','qwen3_4b'):
        samples,runs,proof=verified_pair(root,key,'test','text-test-v1.json');receipts+=proof
        selected=[s for s in samples if s['source_id']=='kormedmcqa'];ids={s['sample_id'] for s in selected}
        preds=[[p for p in pp if p['sample_id'] in ids] for _,pp in runs]
        korean.append({'model':key,'support':len(selected),'partitions':correctness_partition(selected,*preds)})
    for key,scope,name in [('medgemma15_4b','test','slake-test-v1.json'),('qwen35_2b','vqarad','vqarad-test-v1.json'),('medgemma15_4b','vqarad','vqarad-test-v1.json')]:
        samples,runs,proof=verified_pair(root,key,scope,name);receipts+=proof
        for condition,(_,preds) in zip(('base','tuned'),runs):
            vqa.append({'model':key,'source':samples[0]['source_id'],'condition':condition,'groups':vqa_diagnostics(samples,preds)})
    result={'schema_version':'medvlm.v1-reanalysis.v2','status':'SUCCEEDED','evidence_level':'VERIFIED_EXISTING_PREDICTIONS',
            'new_training_or_inference':False,'human_review':'not_performed','clinical_accuracy':None,
            'recorded_model_identity_verified':True,'base_weight_bytes_rehashed':False,
            'limitations':['Post-hoc descriptive analysis of previously observed v1 tests; no new blind test.',
                          'Validity partitions are descriptive, not a causal attribution of medical knowledge.',
                          'Reference phrase occurrence can be negated, contradicted or quoted and is not accuracy.',
                          'Native model identity is compared to recorded training receipts; no new weight load or inference.'],
            'receipts':receipts,'kormedmcqa_validity':korean,'vqa_diagnostics':vqa,
            'training_8450_to_8400':[training_counts(root,k) for k in ('qwen3_17b','qwen3_4b')]}
    output.mkdir(parents=True,exist_ok=True);atomic_json(output/'v1_reanalysis.json',result)
    lines=['# v1 예측 후속 재분석','', '기존 성공 예측과 동결 프로토콜을 읽어 재집계했다. 새 학습·추론은 하지 않았다. 원문·예측 본문은 공개 보고서에 포함하지 않는다.','',
           '## KorMedMCQA: 형식 유효성과 정답 변화','', '| 모델 | 구분 | support | Base 정답 | Tuned 정답 | 개선 | 퇴보 | 전체 정답률 기여 |','|---|---|---:|---:|---:|---:|---:|---:|']
    for item in korean:
        for p in item['partitions']:
            lines.append(f"| {item['model']} | {p['partition']} | {p['support']} | {p['base_correct']} | {p['tuned_correct']} | {p['improved']} | {p['regressed']} | {100*p['contribution_to_total_delta']:.2f}%p |")
    lines+=['','유효 응답은 기존 parser의 answered 상태와 parsed answer를 기준으로 한다. invalid에는 형식 오류·응답 보류·실패가 포함되며 상세 상태 수는 JSON에 있다. both_valid의 변화는 형식 전환 없이 정답이 바뀐 부분이다. invalid→valid에서 맞힌 수는 출력 유효성 회복과 함께 발생한 변화이며 내용 지식과의 인과적 분리는 아니다.','',
            '## VQA: 답 길이·정답 구절·잘림·OPEN/CLOSED','', '| 모델/source | 조건 | 구분 | n | lexical EM | 구절 포함 | 잘림 | 출력 단어 중앙값 |','|---|---|---|---:|---:|---:|---:|---:|']
    for item in vqa:
        for g in item['groups']:
            lines.append(f"| {item['model']}/{item['source']} | {item['condition']} | {g['answer_type']} | {g['scheduled']} | {100*g['lexical_exact_match']:.2f}% | {100*g['reference_phrase_contains_fraction']:.2f}% | {g['truncated_count']} | {g['output_whitespace_words']['median']} |")
    lines+=['','구절 포함은 NFKC·대소문자·공백 정규화 후 단어 경계가 있는 연속 정답 구절의 출현이다. 부정·대안 나열·인용도 포함될 수 있으므로 정확도가 아니다. 숫자·단위·부정을 지우지 않았다. MedGemma Base의 EM 0%는 32-token 생성 상한과 답변 형태의 영향을 받으며 의료 능력 0%를 의미하지 않는다. VQA-RAD의 향상은 주로 CLOSED에서 발생했고 OPEN 22개는 매우 작은 표본이다. 자유서술 일반화 개선을 입증하지 않는다.','',
            '## 8,450 준비 문항과 실제 8,400 처리 문항','', '| 모델 | source | 준비/선택 | 제외 | 처리 | supervised tokens |','|---|---|---:|---:|---:|---:|']
    for item in result['training_8450_to_8400']:
        for s in item['sources']:
            lines.append(f"| {item['model']} | {s['source_id']} | {s['prepared']}/{s['selected_for_preflight']} | {s['rejected']} | {s['processed']} | {s['supervised_tokens']} |")
    lines+=['','두 모델 모두 native template 적용 후 max_length=2048을 넘는 50문항을 통째로 제외했다. 남은 8,400개를 16개 effective batch로 525회 업데이트했고 미처리 usable 문항은 0개다. 1,000-step 상한이나 GPU 시간 상한으로 50개가 누락된 것이 아니다. 질문·답변을 잘라서 학습하지 않았다.','',
            '## 무결성과 해석 범위','', '예측/기존 metrics의 기록 SHA, config hash, protocol·입력·reference·이미지 byte hash, run/model revision과 기록된 native 모델 digest, adapter bytes를 검증했다. sampler 순서·preflight 제외·최종 checkpoint 처리 위치·optimizer event의 표본 및 token 합계를 대조했다. base 모델 가중치를 새로 로드하거나 전체 바이트를 재해시하지 않았다. 검증 receipt는 JSON에 있다. 전문가 검토는 수행하지 않았다.','',
            '재생성: `PYTHONPATH=src python scripts/analyze_v1_followup.py --data-root ../medvlm-data --output reports/public_summary/v2`','']
    (output/'V1_REANALYSIS_KO.md').write_text('\n'.join(lines),encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=main(args.data_root.resolve(),args.output)
    print(json.dumps({'status':result['status'],'verified_runs':len(result['receipts'])}))
