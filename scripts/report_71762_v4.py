#!/usr/bin/env python3
"""Verify finished v4 evidence and export aggregates plus private blind reviews."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import io
import json
import math
from pathlib import Path

from complete_v1 import verify_training_cache, verify_reload_cache
from mediflow_medvlm.data.io import digest, read_jsonl
from mediflow_medvlm.data.schema import normalized, validate_record
from mediflow_medvlm.evaluation.metrics import score_rows, aggregate, korean_summary
from mediflow_medvlm.evaluation.runner import load_protocol, validated_predictions, compare
from mediflow_medvlm.runtime import atomic_json, canonical_hash, file_hash


def read(path):
    return json.loads(Path(path).read_text())


def immutable_bytes(path, value):
    if path.exists():
        if path.read_bytes() != value:
            raise ValueError('IMMUTABLE_REVIEW_CHANGED')
    else:
        path.write_bytes(value)


def equivalent(left, right):
    """Allow only float summation roundoff between Python 3.10 and 3.12."""
    if isinstance(left, float) and isinstance(right, float):
        return (math.isfinite(left) and math.isfinite(right) and
                math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(equivalent(left[k], right[k]) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(equivalent(a,b) for a,b in zip(left,right))
    return type(left) is type(right) and left == right


def review_packet(root, model, protocol_path, base, tuned):
    out = root / 'v4/review' / model
    out.mkdir(parents=True, exist_ok=True)
    protocol, rows = load_protocol(protocol_path)
    sources = {label: {r['sample_id']: r for r in read_jsonl(folder/'predictions.jsonl')}
               for label, folder in [('base', base), ('tuned', tuned)]}
    rows = sorted(rows, key=lambda r: digest([20261023, r['sample_id']]))[:16]
    plan_path = root/'v4/review/plan.json'
    if plan_path.exists():
        plan = read(plan_path)
        if ([r['sample_id'] for r in rows] != plan['sample_ids'] or
                file_hash(protocol_path) != plan['protocol_sha256']):
            raise ValueError('PROSPECTIVE_REVIEW_SELECTION_CHANGED')
    buffer = io.StringIO(newline='')
    fields = ['row_id', 'question', 'output_A', 'output_B', 'randomly_assigned_reference',
              'relevance_A', 'correctness_A', 'safety_A', 'clarity_A',
              'relevance_B', 'correctness_B', 'safety_B', 'clarity_B', 'preferred', 'comment']
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    key = []
    for index, row in enumerate(rows, 1):
        labels = ['base', 'tuned']
        if int(digest([20261023, model, row['sample_id']]), 16) % 2:
            labels.reverse()
        row_id = f'{model}-{index:02d}'
        writer.writerow({'row_id': row_id, 'question': row['input']['question'],
                         'output_A': sources[labels[0]][row['sample_id']].get('raw_output', ''),
                         'output_B': sources[labels[1]][row['sample_id']].get('raw_output', ''),
                         'randomly_assigned_reference': row['supervision']['answer']})
        key.append({'row_id': row_id, 'sample_id': row['sample_id'], 'A': labels[0], 'B': labels[1]})
    # Prediction field is validated below rather than silently exporting blanks.
    for label in ('base', 'tuned'):
        for row in rows:
            pred = sources[label][row['sample_id']]
            if pred['status'] == 'answered' and not pred.get('raw_output'):
                raise ValueError('REVIEW_PREDICTION_OUTPUT_FIELD_MISSING')
    instructions = ('이 CSV의 전달 사본만 편집하세요. A/B 조건은 무작위로 가렸습니다.\n'
        'reference는 같은 카테고리에서 무작위 선택한 답변이며 문항별 검증된 정답이 아닙니다.\n'
        '의료 전문가가 질문 관련성·의학적 정확성·안전성·명료성을 각각 0~2점으로 평가하세요.\n'
        '0=문제 있음, 1=혼합/불확실, 2=적절. 판단 불가 항목은 비워 두고 comment에 이유를 적으세요.\n'
        'preferred는 A/B/tie/unsure 중 하나를 기재하세요. 참조와의 문구 일치를 정확성으로 채점하지 마세요.\n'
        '임상 검토는 아직 수행하지 않았으며 이 packet은 실제 임상 사용을 승인하는 자료가 아닙니다.\n')
    immutable_bytes(out/'review.csv', ('\ufeff'+buffer.getvalue()).encode())
    immutable_bytes(out/'private_key.json', (json.dumps(key,ensure_ascii=False,sort_keys=True)+'\n').encode())
    immutable_bytes(out/'instructions.txt', instructions.encode())
    manifest = {'model': model, 'rows': len(rows), 'human_review': 'not_performed',
                'protocol_sha256': file_hash(protocol_path),
                'sources': {label: {'manifest_sha256': file_hash(folder/'run_manifest.json'),
                                    'predictions_sha256': file_hash(folder/'predictions.jsonl')}
                            for label, folder in [('base', base), ('tuned', tuned)]},
                'files_sha256': {name: file_hash(out/name) for name in ('review.csv','private_key.json','instructions.txt')}}
    immutable_bytes(out/'manifest.json', (json.dumps(manifest,ensure_ascii=False,sort_keys=True)+'\n').encode())
    return {'rows': len(rows), 'human_review': 'not_performed', 'manifest_sha256': file_hash(out/'manifest.json')}


def verify_evaluation(folder, protocol_path):
    manifest = read(folder/'run_manifest.json')
    if manifest['status'] != 'SUCCEEDED':
        raise ValueError('EVALUATION_NOT_SUCCESSFUL')
    protocol, rows = load_protocol(protocol_path)
    if manifest['protocol_hash'] != protocol['protocol_hash']:
        raise ValueError('EVALUATION_PROTOCOL_MISMATCH')
    for name, value in manifest['artifact_hashes'].items():
        if file_hash(folder/name) != value:
            raise ValueError('EVALUATION_ARTIFACT_CHANGED')
    predictions = validated_predictions(folder/'predictions.jsonl', protocol, manifest)
    if len(predictions) != len(rows):
        raise ValueError('MISSING_PREDICTIONS')
    if any(p.get('vision_ingested') is not False or p.get('pixel_tensor_keys') for p in predictions):
        raise ValueError('UNEXPECTED_VISION_IN_TEXT_EVALUATION')
    by_id = {r['sample_id']: r for r in rows}
    for prediction in predictions:
        expected = protocol['max_new_tokens'][by_id[prediction['sample_id']]['task_type']]
        if prediction.get('generation_settings') != {'do_sample': False, 'max_new_tokens': expected}:
            raise ValueError('EVALUATION_GENERATION_POLICY_CHANGED')
    scores = score_rows(rows, predictions)
    metrics = read(folder/'metrics.json')
    if scores != metrics['scores'] or not equivalent(aggregate(scores), metrics['cohorts']):
        raise ValueError('METRIC_RECALCULATION_MISMATCH')
    return {'scheduled': len(rows), 'predictions': len(predictions), 'cohorts': metrics['cohorts'],
            'korean_summary': korean_summary(metrics['cohorts']),
            'artifact_hashes': manifest['artifact_hashes'], 'manifest_sha256': file_hash(folder/'run_manifest.json')}


def verify_corpus(root, corpus):
    import hashlib
    frozen=read(corpus['freeze_manifest'])
    groups={}; counts={}; all_ids=set()
    for key,entry in frozen['splits'].items():
        split=key.rsplit('/',1)[1]
        rows=read_jsonl(entry['snapshot_path'])
        if len(rows)!=entry['count'] or file_hash(entry['snapshot_path'])!=entry['sha256']:
            raise ValueError('FROZEN_SPLIT_INTEGRITY')
        for row in rows:
            meta=row['metadata'];q=meta['question_component'];a=meta['answer_component']
            if validate_record(row) or row['sample_id'] in all_ids:
                raise ValueError('INVALID_FROZEN_CANONICAL')
            all_ids.add(row['sample_id'])
            if q['category']!=a['category'] or q['split']!=split or a['split']!=split:
                raise ValueError('PAIRED_CATEGORY_OR_SPLIT_MISMATCH')
            if q['archive']!=a['archive'] or q['archive']!=('TL.zip' if split=='train' else 'VL.zip'):
                raise ValueError('ORIGINAL_SOURCE_SPLIT_MISMATCH')
            if row['input'].get('images') or set(row['input'])!={'question','images'}:
                raise ValueError('NON_QUESTION_INPUT_OR_LABEL_LEAKAGE')
            target='\n\n'.join(meta['answer_sections'].get(k) or '' for k in ('intro','body','conclusion'))
            if row['supervision']['answer']!=target:
                raise ValueError('ANSWER_SECTIONS_CHANGED')
            for text,component in [(row['input']['question'],q),(target,a)]:
                if hashlib.sha256(normalized(text).encode()).hexdigest()!=component['text_sha256']:
                    raise ValueError('SOURCE_COMPONENT_TEXT_CHANGED')
            if row['group_ids']['parent'] is not None or meta['true_qa_correspondence_verified'] is not False:
                raise ValueError('UNSUPPORTED_TRUE_PAIR_OR_PARENT_CLAIM')
            for kind,value in [('question',q['text_sha256']),('answer',a['text_sha256'])]:
                groups.setdefault((kind,value),set()).add(split)
        counts[split]=len(rows)
    if any(len(values)>1 for values in groups.values()):
        raise ValueError('EXACT_COMPONENT_CROSS_SPLIT')
    train_ids={r['sample_id'] for r in read_jsonl(frozen['splits']['aihub71762_category_random_v4/train']['snapshot_path'])}
    blocked=set()
    for name in ('semantic/audit.json','internal_semantic_audit.json'):
        blocked.update(r['sample_id'] for r in read(root/'v4'/name)['candidate_pairs'])
    for audit in read(root/'v4/lexical_audit.json').values():
        for name in ('confirmed_overlaps','near_candidates'):
            blocked.update(r['sample_id'] for r in audit.get(name,[]))
    if train_ids & blocked:
        raise ValueError('QUARANTINED_CANDIDATE_ENTERED_TRAIN')
    return {'status':'SUCCEEDED','counts':counts,'exact_component_cross_split':0,
            'selected_train_quarantined_candidates':0,'all_source_texts_and_answer_sections_preserved':True,
            'true_qa_or_parent_relationship_inferred':False}


def gpu_accounting(root, plan, state):
    ledger=read(root/'runtime/gpu_budget.json')
    owned={run_id:read(entry['config'])['model_id'] for run_id,entry in plan['runs'].items()}
    for run_id in state['evaluation']:
        model=next(m for m in plan['models'] if run_id.startswith(m+'-'))
        owned[run_id]=read(root/f'v2/text/configs/{model}-evaluation.json')['model_id']
    if ledger.get('active') and ledger['active'].get('run_id') in owned:
        raise ValueError('OWNED_GPU_STAGE_STILL_ACTIVE')
    phases={run_id:ledger['run_phases'].get(run_id,{}) for run_id in owned}
    for run_id,values in phases.items():
        required={'train','reload'} if run_id in plan['runs'] else {'evaluation'}
        if not required <= values.keys() or any(values[k]<=0 for k in required):
            raise ValueError('OWNED_RUN_GPU_LEDGER_INCOMPLETE')
    delta={}
    for run_id,values in phases.items():
        model=owned[run_id]
        delta[model]=delta.get(model,0)+sum(values.values())
    path=root/'v4/gpu_completion.json'
    if path.exists():
        completion=read(path)
        if completion['owned_run_phases']!=phases:
            raise ValueError('COMPLETED_RUN_GPU_ACCOUNTING_CHANGED')
        if any(ledger['models'].get(m,0)<seconds for m,seconds in completion['cumulative_models_at_completion'].items()):
            raise ValueError('CUMULATIVE_GPU_LEDGER_REWOUND')
    else:
        completion={'owned_run_phases':phases,'cumulative_models_at_completion':ledger['models']}
        atomic_json(path,completion)
    return ledger,delta,completion


def report(root, output):
    out = root/'v4'
    state, plan, corpus = (read(out/name) for name in ('state.json','plan.json','corpus.json'))
    if state['status'] != 'SUCCEEDED' or state['errors']:
        raise ValueError('CAMPAIGN_REQUIRED_STAGES_NOT_COMPLETE')
    if state['plan_sha256'] != canonical_hash(plan) or file_hash(out/'corpus.json') != plan['corpus_sha256']:
        raise ValueError('CAMPAIGN_DECLARATION_CHANGED')
    if file_hash(Path(__file__).with_name('run_71762_v4.py')) != plan['runner_sha256']:
        raise ValueError('CAMPAIGN_SOURCE_CHANGED')
    if file_hash(Path(__file__).with_name('prepare_71762_v4.py')) != corpus['identity']['preparation_code_sha256']:
        raise ValueError('PREPARATION_SOURCE_CHANGED')
    frozen = read(corpus['freeze_manifest'])
    corpus_verification=verify_corpus(root,corpus)
    if file_hash(corpus['freeze_manifest']) != corpus['freeze_manifest_sha256']:
        raise ValueError('CORPUS_FREEZE_CHANGED')
    for entry in frozen['splits'].values():
        if file_hash(entry['snapshot_path']) != entry['sha256']:
            raise ValueError('CORPUS_ROWS_CHANGED')
    for key, name in [('lexical_audit_sha256','lexical_audit.json'), ('token_audit_sha256','token_audit.json'),
                      ('semantic_audit_sha256','semantic/audit.json'), ('internal_semantic_sha256','internal_semantic_audit.json')]:
        if file_hash(out/name) != frozen['provenance'][key]:
            raise ValueError('DATA_SCREEN_RECEIPT_CHANGED')
    for path, expected in frozen['provenance']['protected_files'].items():
        if file_hash(path) != expected:
            raise ValueError('PROTECTED_EXTERNAL_HELDOUT_CHANGED')
    historical = read(out/'historical_baseline.json')
    for name, expected in historical['files_sha256'].items():
        if file_hash(name) != expected:
            raise ValueError('HISTORICAL_ARTIFACT_CHANGED')
    training = {}
    for run_id, entry in plan['runs'].items():
        if file_hash(entry['config']) != entry['config_sha256']:
            raise ValueError('RUN_CONFIG_CHANGED')
        config = read(entry['config'])
        manifest, selection = verify_training_cache(config)
        reload = verify_reload_cache(config, manifest, selection)
        expected_steps = 20 if entry['phase'] == 'smoke' else 250
        if manifest['optimizer_steps'] != expected_steps or manifest['samples_processed'] != expected_steps*16:
            raise ValueError('INCOMPLETE_DECLARED_TRAINING')
        if manifest.get('adapter_changed') is not True or manifest.get('test_used_for_selection') is not False:
            raise ValueError('ADAPTER_UPDATE_OR_VALIDATION_SELECTION_NOT_VERIFIED')
        if entry['phase'] == 'full' and manifest['supervised_tokens'] != entry['supervised_tokens_full_pass']:
            raise ValueError('DECLARED_SUPERVISED_TOKEN_COUNT_MISMATCH')
        training[run_id] = {'status': manifest['status'], 'optimizer_steps': manifest['optimizer_steps'],
              'samples_processed': manifest['samples_processed'], 'supervised_tokens': manifest['supervised_tokens'],
              'selected_checkpoint': Path(selection['checkpoint']).name,
              'training_pid': manifest['training_pid'], 'reload_pid': reload['pid'],
              'fresh_process_reload': True, 'adapter_changed': manifest.get('adapter_changed'),
              'manifest_sha256': file_hash(Path(config['output_dir'])/'manifest.json')}
    evaluations, controls, reviews = {}, {}, {}
    for model in plan['models']:
        outputs = {}
        for dataset, conditions in (('random',('base','tuned')), ('common',('tuned',))):
            protocol = out/'random-heldout-test.json' if dataset == 'random' else root/'v2/text/text-test.json'
            for condition in conditions:
                key = f'{model}-{dataset}-{condition}-v4-20261112'
                folder = out/'evaluations'/key
                evaluations[key] = verify_evaluation(folder, protocol)
                outputs[(dataset, condition)] = folder
        base = root/f'v2/text/evaluations/{model}-base-text-v2-20261026'
        controls[model] = verify_evaluation(base, root/'v2/text/text-test.json')
        compare(base, outputs[('common','tuned')], out/f'{model}-common-base-vs-tuned.json')
        compare(outputs[('random','base')], outputs[('random','tuned')], out/f'{model}-random-base-vs-tuned.json')
        reviews[model] = review_packet(root, model, out/'random-heldout-test.json',
                                       outputs[('random','base')], outputs[('random','tuned')])
    ledger,delta,completion=gpu_accounting(root,plan,state)
    gpu_receipt=out/'gpu_completion.json'
    counts = {}
    answer_reuse = {}
    for key, entry in frozen['splits'].items():
        records = read_jsonl(entry['snapshot_path'])
        counts[key] = {'source': dict(Counter(r['source_id'] for r in records)),
                       'task': dict(Counter(r['task_type'] for r in records)),
                       'language': dict(Counter(r['language'] for r in records))}
        answers = Counter(r['metadata']['answer_component']['text_sha256'] for r in records)
        answer_reuse[key] = {'unique_answer_texts':len(answers), 'reused_answer_rows':sum(n-1 for n in answers.values()),
                             'max_answer_reuse':max(answers.values(),default=0)}
    summary = {'schema':'medvlm.aihub71762.result.v4','status':'SUCCEEDED',
               'authorization': 'Explicit user exception on 2026-11-12',
               'method':'provider-described category-random pairing; independent implementation',
               'clinical_accuracy':None,'human_review':'not_performed',
               'data':{key: corpus[key] for key in ('token_counts_by_model','domain_counts','semantic_quarantined_rows_train_and_validation',
                          'protected_unique_rows','protected_kormed_rows','leakage_limitations')},
               'training':training,'evaluations':evaluations,'reused_base_controls':controls,'review_packets':reviews,
               'counts_by_source_task_language':counts,'selected_answer_reuse':answer_reuse,
               'gpu_seconds_delta_by_model':delta,'gpu_seconds_delta_total':sum(delta.values()),
               'gpu_seconds_cumulative_total':sum(completion['cumulative_models_at_completion'].values()),
               'gpu_scope':'shared ledger occupancy for declared v4 train/reload/evaluation run IDs only',
               'gpu_completion_sha256':file_hash(gpu_receipt),'resource_policy':ledger['current_policy'],
               'historical_files_verified_unchanged':len(historical['files_sha256']),
               'limitations':plan['limitations'], 'official_preprocessing_code_reviewed':False}
    output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/'run_summary.json',summary)
    atomic_json(output/'verification.json',{'status':'SUCCEEDED','training_and_reload':len(training),
                'new_evaluations':len(evaluations),'reused_controls':len(controls),'review_packets':len(reviews),
                'historical_files_unchanged':len(historical['files_sha256']),'errors':[],
                'plan_sha256':file_hash(out/'plan.json'),'corpus_sha256':file_hash(out/'corpus.json'),
                'corpus_verification':corpus_verification})
    lines = ['# 71762 별도 무작위 연결 실험 — 2026-11-12', '',
             '현재 단계: 선언한 두 모델의 학습·새 process 재로드·평가를 완료했습니다.', '',
             '이번에 한 일: 사용자 예외 승인에 따라 제공처 방식의 별도 구현으로 학습4,000/검증105/파생평가256을 동결했습니다. 두 모델 각각20update smoke와250update 1pass 본학습 및 별도 재로드를 수행했습니다. 새 평가6개와 기존 Base 대조2개를 검증했습니다.', '',
             '확인 결과: 아래 숫자는 실제 채점 결과입니다. 새 71762의 무작위 참조와의 문자열 일치는 임상 정답률이 아닙니다.', '',
             '| 모델 | 평가 | 조건 | 문항수 | 정답/문자열 일치 | coverage | lexical F1 |',
             '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for key, entry in evaluations.items():
        if '-random-' in key:
            cohort = entry['cohorts'][0]
            lines.append(f"| {key.split('-random-')[0]} | 71762 무작위 참조 | {'Base' if '-base-' in key else 'Tuned'} | {cohort['scheduled']} | {cohort['exact_matches']} | {cohort['coverage']:.2%} | {cohort['lexical_token_f1']:.4f} |")
        else:
            korean = entry['korean_summary']
            lines.append(f"| {key.split('-common-')[0]} | KorMedMCQA | Tuned | {korean['scheduled']} | {korean['correct']} | {korean['coverage']:.2%} | — |")
    for model, entry in controls.items():
        korean = entry['korean_summary']
        lines.append(f"| {model} | KorMedMCQA | 기존 Base 재사용 | {korean['scheduled']} | {korean['correct']} | {korean['coverage']:.2%} | — |")
    lines += ['', '외부 자료별 유지 성능:', '',
              '| 모델 | 자료/문항 유형 | 문항수 | Base | 71762 Tuned | 지표 |',
              '| --- | --- | ---: | ---: | ---: | --- |']
    for model, base_entry in controls.items():
        tuned_entry = evaluations[f'{model}-common-tuned-v4-20261112']
        for source, task in [('pubmedqa_labeled','context_qa'),('aihub_71875','medical_mcq'),
                             ('aihub_71875','medical_qa'),('medquad','medical_qa')]:
            left = next(c for c in base_entry['cohorts'] if c['source_id']==source and c['task_type']==task)
            right = next(c for c in tuned_entry['cohorts'] if c['source_id']==source and c['task_type']==task)
            field = 'end_to_end_correct_fraction' if task in {'context_qa','medical_mcq'} else 'lexical_token_f1'
            metric = '정답률(형식 실패 포함)' if field=='end_to_end_correct_fraction' else 'lexical F1'
            lines.append(f"| {model} | {source}/{task} | {left['scheduled']} | {left[field]:.4f} | {right[field]:.4f} | {metric} |")
    lines += ['', '실행·미실행·한계:', '',
              f"- 실제 수행: 학습4회·새 process 재로드4회, 새 평가6개, 대조2개 검증. GPU 추가 {sum(delta.values())/3600:.3f}시간. 과거 파일{len(historical['files_sha256'])}개 hash 보존 확인.",
              '- 미실행: 공식 make_train_data.py 다운로드·실행, 추가seed, 전체 원본 학습, 새로운 MedGemma/VLM 학습, 사람의 임상 검토. Jetson은 사용자 보류입니다.',
              '- 차단/한계: 증강 parent는 미제공이며 복구하지 않았습니다. 의미 유사도 후보 격리는 의학적/번역 동등성 검증이 아닙니다. 무작위 Q→A 대응의 개별 적합성, 답변 재서술 누수와 기반모델 과거 노출은 배제할 수 없습니다.',
              '- 두 모델은 원본 Base에서 독립 SFT했습니다. 이전 EN/KO/MIX와 학습 token 노출량이 달라 차이를 데이터 방식의 인과 효과로 해석하지 않습니다.',
              '- 사람 검토표2개·32비교행은 동일16문항을 사용하며 점수는 비어 있습니다. private A/B 키는 공개하지 않습니다.', '',
              '다음 단계: 아래 재개 명령은 완료한 산출물을 검증 후 재사용합니다. 무한 재학습이나 평가 성적에 따른 재추첨은 하지 않습니다.', '',
              '```bash', 'export PYTHONPATH="$PWD/src"',
              'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_DISABLED=true',
              '../.venv-medvlm/bin/python scripts/run_71762_v4.py --data-root ../medvlm-data --phase all',
              '../.venv-medvlm/bin/python scripts/report_71762_v4.py --data-root ../medvlm-data --output reports/public_summary/v4',
              '```', '', '사용자가 할 일: 실행 재개를 위한 추가 파일·승인은 필요하지 않습니다. 임상적 정확성 판단이 필요하면 의료 전문가가 검토표 전달 사본을 작성해야 하며, 현재 사람 검토는0건입니다.', '']
    (output/'RESULTS_KO.md').write_text('\n'.join(lines))
    print(json.dumps({'status':'SUCCEEDED','training':len(training),'evaluations':len(evaluations),'output':str(output)}),flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    report(args.data_root.resolve(), args.output.resolve())


if __name__=='__main__':
    main()
