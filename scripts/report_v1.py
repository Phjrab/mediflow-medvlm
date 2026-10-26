#!/usr/bin/env python3
"""Generate aggregate-only Korean handoff artifacts from actual local evidence."""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import tempfile
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone

from mediflow_medvlm.runtime import atomic_json, canonical_hash, file_hash, resolve_resource_policy

KEYS = ('qwen3_17b','qwen3_4b','qwen35_2b','qwen3vl_2b','qwen25vl_3b','medgemma15_4b','gemma3_4b','smolvlm_500m')
MODELS = {'qwen3_17b':'Qwen/Qwen3-1.7B','qwen3_4b':'Qwen/Qwen3-4B',
          'qwen35_2b':'Qwen/Qwen3.5-2B','medgemma15_4b':'google/medgemma-1.5-4b-it','gemma3_4b':'google/gemma-3-4b-it',
          'qwen3vl_2b':'Qwen/Qwen3-VL-2B-Instruct','qwen25vl_3b':'Qwen/Qwen2.5-VL-3B-Instruct',
          'smolvlm_500m':'HuggingFaceTB/SmolVLM-500M-Instruct'}


def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def portable(value, root):
    if isinstance(value,dict):return {k:portable(v,root) for k,v in value.items()}
    if isinstance(value,list):return [portable(v,root) for v in value]
    if isinstance(value,str):return value.replace(str(root),'${MEDVLM_DATA_ROOT}')
    return value


def pct(value):
    return 'N/A' if value is None else f'{value*100:.2f}%'


def points(value):
    return 'N/A' if value is None else f'{value*100:.2f}%p'


def metric(row):
    classification = row['task_type'] in {'medical_mcq','context_qa'}
    return row.get('end_to_end_correct_fraction' if classification else 'end_to_end_lexical_exact_match')


def resource_policy_description(summary):
    """Distinguish historical v1 constraints from the currently resolved policy."""
    policy=summary.get('current_resource_policy') or resolve_resource_policy().as_dict()
    historical='v1 실행 당시 제약은 전체 24 GPU시간·기반모델별 6 GPU시간이었다. 과거 config와 사용량 기록은 보존했다.'
    if policy['budget_mode']=='unlimited':
        current='현재 정책은 누적 GPU 시간 상한 없음(unlimited)이며 total_gpu_hours/per_model_gpu_hours와 run/job/campaign 시간 상한은 null이다.'
    else:
        current=(f"현재 finite 정책은 전체 {policy['total_gpu_hours']:g} GPU시간·기반모델별 {policy['per_model_gpu_hours']:g} GPU시간이다. "
                 'run/job/campaign 상한의 실제 값은 current_resource_policy에 기록했다.')
    return historical+' '+current+' 시간 상한 변경으로 사용량 원장을 초기화하지 않는다.'


def weight_inventory(config):
    if not config.get('cache_dir') or not config.get('revision'):
        return {'status':'PREPARED_NOT_RUN','reason_code':'NO_CONFIG'}
    snapshot=Path(config['cache_dir'])/('models--'+config['model_id'].replace('/','--'))/'snapshots'/config['revision']
    index=read(snapshot/'model.safetensors.index.json',{})
    if not index and not (snapshot/'model.safetensors').is_file():
        return {'status':'PREPARED_NOT_RUN','evidence_level':'LOCAL_FILE_PRESENCE_ONLY',
                'expected_weight_files':None,'present_weight_files':0,'present_bytes':0,
                'reason_code':'WEIGHTS_NOT_PRESENT_SHARD_COUNT_UNRESOLVED'}
    names=sorted(set(index.get('weight_map',{}).values())) or ['model.safetensors']
    present=[name for name in names if (snapshot/name).is_file()]
    return {'status':'SUCCEEDED' if len(present)==len(names) else 'PREPARED_NOT_RUN',
            'evidence_level':'LOCAL_FILE_PRESENCE_ONLY','expected_weight_files':len(names),
            'present_weight_files':len(present),'present_bytes':sum((snapshot/name).stat().st_size for name in present),
            'note':'Presence is not proof of model load, optimizer update, or adapter reload.'}


def prediction_runtime(run):
    path=run/'predictions.jsonl'
    if not path.exists():return None
    latencies=[];tokens=0
    with path.open() as stream:
        for line in stream:
            record=json.loads(line)
            latency=record.get('timing',{}).get('e2e_seconds')
            if latency is not None:
                latencies.append(latency);tokens+=record.get('output_tokens',0)
    latencies.sort();seconds=sum(latencies)
    return {'measured_attempts':len(latencies),'generated_tokens':tokens,'summed_e2e_seconds':seconds,
            'generated_tokens_per_summed_e2e_second':tokens/seconds if seconds else None,
            'p95_e2e_seconds_nearest_rank':latencies[math.ceil(.95*len(latencies))-1] if latencies else None,
            'decode_only_tokens_per_second':None,'ttft_seconds':None,'energy_joules':None}


def collect(root, repo):
    campaign=read(repo/'runtime/medvlm_state.json',{})
    summary={'schema_version':'medvlm.public-result.v1','generated_at':datetime.now(timezone.utc).isoformat(),
             'preliminary_single_seed':True,'human_review':'not_performed','clinical_validation':False,
             'data':read(repo/'reports/public_summary/data_freeze.json'), 'models':{},'evaluations':[],
             'data_repairs':read(repo/'reports/public_summary/vqarad_image_provenance_repair.json'),
             'posthoc_format_diagnostics':read(repo/'reports/public_summary/medgemma_format_diagnostic.json'),
             'paired':{},'external_paired':{},'incomplete_comparisons':{},'external_evaluations':[],'image_diagnostics':{},'changes':{},
             'campaign':{k:campaign.get(k) for k in ('stages','current_model','updated_at','active_process')},
             'device':{k:v for k,v in read(root/'runs/device/jetson-inventory-20261023/summary.json',{}).items()
                                  if k in ('status','reason_code','evidence_level','device_outcome','wall_seconds','finished_at')},
             'budget':read(root/'runtime/gpu_budget.json'),
             'current_resource_policy':resolve_resource_policy().as_dict(),
             'historical_v1_resource_limits':{'scope':'constraints_at_original_v1_execution',
                                              'total_gpu_hours':24,'per_model_gpu_hours':6}}
    for key in KEYS:
        train=root/'runs'/f'{key}-v1-20261023'; smoke=root/'runs'/f'{key}-smoke-20261023'
        manifest=read(train/'manifest.json',{}); selection=read(train/'selection.json',{})
        sm=read(smoke/'manifest.json',{})
        model=manifest.get('model',{})
        wanted=('status','evidence_level','reason_code','optimizer_steps','microsteps','samples_processed','supervised_tokens',
                'wall_seconds','gpu_seconds','peak_allocated_bytes','peak_reserved_bytes','config_hash','data','adapter',
                'best_checkpoint','validation_history','initial_validation','training_selection','validation_selection','selected_validation_loss','adapter_changed',
                'effective_batch','resume_from','training_pid','finished_at','gpu','host','versions','actual_epochs')
        record={k:manifest.get(k) for k in wanted if k in manifest}
        record['peak_allocated_bytes']=manifest.get('peak_cuda_allocated_bytes')
        record['peak_reserved_bytes']=manifest.get('peak_cuda_reserved_bytes')
        record['recipe']=read(root/'configs'/f'{key}-v1-20261023.json',{})
        record['weights']=weight_inventory(record['recipe'])
        record['validation_curve']=[]
        events=train/'events.jsonl'
        if events.exists():
            for line in events.read_text().splitlines():
                event=json.loads(line)
                if event['event'] in ('checkpoint','validation_before_updates'):
                    record['validation_curve'].append(event)
        record.update(model_id=MODELS[key],revision=manifest.get('revision'),status=manifest.get('status','PREPARED_NOT_RUN'),
                      campaign={k:v for k,v in campaign.get('models',{}).get(key,{}).items()
                                if k in ('status','reason_code','phase','evidence_level','campaign_disposition','diagnostics')},
                      manifest_sha256=file_hash(train/'manifest.json') if manifest else None,
                      smoke={k:sm.get(k) for k in ('status','evidence_level','optimizer_steps','adapter_changed')},
                      reload=read(train/'reload.json',{}).get('status','PREPARED_NOT_RUN'),
                      selected_checkpoint=selection.get('checkpoint'),selection=selection,reload_verified=selection.get('reload_verified',False),
                      model={k:model.get(k) for k in ('dtype','quantization','native_files','chat_template_sha256','trainable_parameters','frozen_parameters','vision_encoder_frozen','loader','processor_class')},
                      target_modules_count=len(model['exact_target_modules']) if 'exact_target_modules' in model else None,
                      target_modules_sha256=canonical_hash(model['exact_target_modules']) if 'exact_target_modules' in model else None,
                      code={k:manifest.get('code',{}).get(k) for k in ('commit','dirty','source_tree_sha256','training_code_sha256')})
        summary['models'][key]=record
        for mode in ('base','tuned'):
            run=root/'runs'/f'{key}-test-{mode}-20261023'
            rm=read(run/'run_manifest.json',{})
            metrics=read(run/'metrics.json',{})
            summary['evaluations'].append({'key':key,'model_id':MODELS[key],'mode':mode,'run_id':run.name,
                'status':rm.get('status','PREPARED_NOT_RUN'),'reason_code':rm.get('reason_code'),
                'evidence_level':rm.get('evidence_level','IMPLEMENTED_ONLY'),'protocol_hash':rm.get('protocol_hash'),
                'identity_hash':rm.get('identity_hash'),'scheduled':rm.get('scheduled'), 'predictions':rm.get('predictions'),
                'load_seconds':rm.get('load_seconds'),'wall_seconds':rm.get('wall_seconds'),'warmup':rm.get('warmup'),
                'peak_allocated_bytes':rm.get('peak_allocated_bytes'),'peak_reserved_bytes':rm.get('peak_reserved_bytes'),
                'ttft_seconds':None,'energy_joules':None,'cohorts':metrics.get('cohorts',[]),
                'korean_original_eval':metrics.get('korean_original_eval'),'vqa_subgroups':metrics.get('vqa_subgroups',[]),
                'measurement':rm.get('measurement')})
            summary['evaluations'][-1]['runtime']=prediction_runtime(run)
        pair=read(root/'runs'/f'{key}-paired-comparison.json')
        pair_complete=record['campaign'].get('status')=='SUCCEEDED' and all(
            read(root/'runs'/f'{key}-test-{mode}-20261023/run_manifest.json',{}).get('status')=='SUCCEEDED'
            for mode in ('base','tuned'))
        if pair:
            if pair_complete:summary['paired'][key]=pair
            else:summary['incomplete_comparisons'][key]=pair
        external=read(root/'runs'/f'{key}-vqarad-paired-comparison.json')
        if external:
            external_complete=all(read(root/'runs'/f'{key}-vqarad-{mode}-20261023/run_manifest.json',{}).get('status')=='SUCCEEDED' for mode in ('base','tuned'))
            if external_complete:summary['external_paired'][key]=external
            else:summary['incomplete_comparisons'][key+'-vqarad']=external
        for mode in ('base','tuned'):
            run=root/'runs'/f'{key}-vqarad-{mode}-20261023'
            rm=read(run/'run_manifest.json',{})
            metrics=read(run/'metrics.json',{})
            if rm:
                summary['external_evaluations'].append({'key':key,'model_id':MODELS[key],'mode':mode,
                    'status':rm.get('status'),'reason_code':rm.get('reason_code'),
                    'peak_allocated_bytes':rm.get('peak_allocated_bytes'),'cohorts':metrics.get('cohorts',[]),
                    'vqa_subgroups':metrics.get('vqa_subgroups',[])})
        for mode in ('base','tuned'):
            diagnostic=read(root/'runs'/f'{key}-{mode}-image-diagnostic.json')
            if diagnostic:summary['image_diagnostics'][key+'-'+mode]=diagnostic
        left=read(root/'runs'/f'{key}-test-base-20261023/metrics.json',{}).get('scores',[])
        right={x['sample_id']:x for x in read(root/'runs'/f'{key}-test-tuned-20261023/metrics.json',{}).get('scores',[])}
        if left and right:
            by_source={}
            for a in left:
                b=right.get(a['sample_id'])
                if b is None:continue
                source=by_source.setdefault(a['source_id'],{'improved':0,'unchanged':0,'regressed':0,'opaque_examples':[]})
                label='improved' if b['exact_match']>a['exact_match'] else 'regressed' if b['exact_match']<a['exact_match'] else 'unchanged'
                source[label]+=1
                if label!='unchanged' and len(source['opaque_examples'])<6:
                    source['opaque_examples'].append({'sample_id':a['sample_id'],'change':label,'base_status':a['status'],'tuned_status':b['status']})
            summary['changes'][key]=by_source
    for name in KEYS:
        probe=read(root/'manifests/models'/f'{name}.json',{})
        if not summary['models'][name]['revision']:
            summary['models'][name]['revision']=probe.get('revision')
        summary['models'][name]['access_probe']={k:probe.get(k) for k in ('status','reason_code','http_status','access_verified_config','processor_class','declared_license','source_url','checked_at','evidence_level','native_processor_cpu_status','weights_loaded','forward_tested','backward_tested')}
        summary['models'][name]['license_evidence']=probe.get('license_evidence')
        summary['models'][name]['weight_access_probe']=read(root/'manifests/models'/f'{name}-weight-access.json')
    return portable(summary,root)


def csv_rows(summary):
    rows=[]
    for ev in summary['evaluations']+summary['external_evaluations']:
        for c in ev['cohorts']:
            rows.append({'model':ev['model_id'],'condition':ev['mode'],'run_status':ev['status'],
                         **{k:c.get(k) for k in ('source_id','task_type','language','subject','scheduled','attempted','completed','valid_answers','correct','exact_matches','coverage','answered_accuracy','answered_lexical_exact_match','end_to_end_correct_fraction','end_to_end_lexical_exact_match','lexical_token_f1','macro_f1','invalid_format_rate','technical_failure_rate','not_attempted_count','truncated_count')},
                         'status_counts':json.dumps(c['statuses'],sort_keys=True),
                         'median_e2e_seconds':c['e2e_latency_seconds']['median'],'peak_allocated_bytes':ev['peak_allocated_bytes']})
    return rows


def report(summary, root, repo):
    data=summary['data'];lines=['# 공개 의료 QA/VQA 로컬 학습 실행 보고서','',
      f"생성 시각: {summary['generated_at']}. 실제 artifact에서 자동 집계. 단일 학습 seed 20261023의 예비 연구 결과다.",'',
      '## 실제 데이터 확보·정제','',f"정제 canonical {data['record_count']:,}개. 감사 critical 오류 {data['critical_error_count']}개. aggregate freeze `{data['freeze_id']}`.",'',
      '| 소스 | 원본 QA 행 | 학습 | 검증 | 시험 | 사용 조건 |','|---|---:|---:|---:|---:|---|']
    sources={**read(repo/'reports/public_summary/data_sources.json',{}),**read(repo/'reports/public_summary/vision_sources.json',{}).get('sources',{})}
    for name in ('medquad','pubmedqa_labeled','kormedmcqa','slake','vqa_rad'):
        source=sources.get(name,{});count=source.get('canonical_counts',source.get('used_counts',{}))
        original=source.get('actual_downloaded_count',source.get('raw_qa_count',source.get('raw_counts')))
        if isinstance(original,dict):original=sum(x for x in original.values() if isinstance(x,int))
        if original is None:original=source.get('actual_downloaded_rows',source.get('original_qa_rows',source.get('qa_records')))
        lines.append(f"| {name} | {original if original is not None else '원본 manifest 참조'} | {count.get('train',0)} | {count.get('validation',0)} | {count.get('test',0)} | {source.get('resolved_applicable_license','source manifest 참조')} |")
    if summary['data_repairs']:
        lines += ['', '외부 VQA-RAD 평가를 준비할 때 이미지 해시 필드 누락을 발견해 provenance를 수정하고 새 aggregate freeze를 만들었다. 원본316파일과451개 영상 참조를 재검증했다. 질문·정답·ID·이미지·core 학습 분할은 그대로이며 이전 freeze도 보존했다. 수정 전 외부 추론·점수는 없었다. 상세 변경 이력은 `vqarad_image_provenance_repair.json`이다.']
    lines += ['', '학습용 코퍼스로 MedQuAD/PubMedQA와 SLAKE를 준비했다. 실제 사용·완료 여부는 모델별 실행 표를 따른다. KorMedMCQA는 한국어 원문 평가 전용, VQA-RAD는 외부 평가 전용이다. 합성·번역 의료 학습 데이터는 추가하지 않았다. 환자 독립성은 확인할 수 없으며, 이미지 유사 후보를 보수적으로 같은 그룹에 묶어 학습 데이터를 제외했다. 소스별 정확한 제외 수·권리·revision·분할 해시는 DATA_REPORT_KO.md와 공개 manifest를 참조한다.', '',
      '## 실제 학습 및 재로드', '', '| 모델 | v1 학습 상태 / 전체 셀 상태 | 실제 update | 처리 sample 횟수 | target token | 재로드 |', '|---|---|---:|---:|---:|---|']
    for key,m in summary['models'].items():
        cell=m['campaign'];disposition=cell.get('reason_code') or cell.get('campaign_disposition') or cell.get('phase','')
        lines.append(f"| {m['model_id']} | {m['status']} / {cell.get('status','PREPARED_NOT_RUN')} {disposition} | {m.get('optimizer_steps','—')} | {m.get('samples_processed','—')} | {m.get('supervised_tokens','—')} | {m.get('reload','미실행')} |")
    lines += ['', '학습 횟수와 평가 완료 수는 별개다. smoke와 본 학습도 별도 run으로 보존했다. 기본 설정은 NF4 QLoRA, SmolVLM은 BF16 LoRA이며 vision encoder/projector는 동결했다. 모델별 실제 설정·모듈·메모리·adapter digest는 ADAPTER_CARDS.md와 run_summary.json에 보존했다. 선택은 고정 validation loss에만 근거하고 test는 선택에 사용하지 않았다.', '',
      '## 동일 조건 Base/Tuned 시험 평가', '', '| 모델 | source / 언어 / 직군 | 조건 | 예정/시도/완료/유효 | 주 지표 | coverage | 형식 오류 | 기술 실패 | E2E 중앙값(초) |', '|---|---|---|---|---:|---:|---:|---:|---:|']
    for e in summary['evaluations']:
        if not e['cohorts']:
            lines.append(f"| {e['model_id']} | — | {e['mode']} | {e['status']} | N/A | N/A | N/A | N/A | N/A |")
        for c in e['cohorts']:
            label=' / '.join(str(c[k]) for k in ('source_id','language','subject') if c.get(k))
            den='/'.join(str(c[k]) for k in ('scheduled','attempted','completed','valid_answers'))
            latency=c['e2e_latency_seconds']['median']
            lines.append(f"| {e['model_id']} | {label} | {e['mode']} | {den} | {pct(metric(c))} | {pct(c['coverage'])} | {pct(c['invalid_format_rate'])} | {pct(c['technical_failure_rate'])} | {latency:.3f} |" if latency is not None else f"| {e['model_id']} | {label} | {e['mode']} | {den} | {pct(metric(c))} | {pct(c['coverage'])} | {pct(c['invalid_format_rate'])} | {pct(c['technical_failure_rate'])} | N/A |")
    lines += ['', 'PubMedQA·KorMedMCQA 주 지표는 전체 예정 문항 기준 정답률이다. MedQuAD·SLAKE 주 지표는 보수적인 문자열 exact match이며 임상 정답률이 아니다. lexical token F1, 유효 응답 기준 지표, 잘림과 실패 분모는 CSV/JSON에 함께 보존했다. 평가 대상이 없거나 실행하지 않은 셀을 0점으로 표시하지 않았다.', '',
      '**MedGemma 기본 모델의 SLAKE EM 0/256은 임상 정답률 0%라는 뜻이 아니다.** 사후 형식 진단에서 기본 응답은 중앙값14단어, 정답과 tuned 응답은1단어였다. 기본72건이32-token 생성 상한에 도달했고 정상 종료184건도 엄격한 EM은0이었다. 기본141건에는 gold 문자열이 단어 경계를 지키는 구절로 포함됐지만, 포함 여부는 정답 판정이 아니다. 이미지 입력·해시·동일 prompt 연결에서 명확한 구현 오류는 발견하지 못했다. 형식과 답 길이의 불일치가 큰 영향을 준 증거이며, 내용 오류와 형식의 기여도를 분리한 수치는 없다. 따라서 0→69.92% lexical EM 변화를 같은 크기의 의학적 능력 향상으로 해석하지 않는다. 고정 NF4/32-token 조건은 공식 BF16/장문 생성 예시의 성능 재현도 아니다. 설정·parser·평가 문항은 결과를 본 뒤 변경하지 않았다. 집계 정의와 근거 해시는 `medgemma_format_diagnostic.json`에 있다.', '',
      '| 모델 | source | 문항 / 관측 그룹 | Tuned − Base | 그룹 bootstrap 95% CI |', '|---|---|---:|---:|---|']
    for key,pair in summary['paired'].items():
        for c in pair['comparison']:
            lines.append(f"| {MODELS[key]} | {c['source_id']} | {c['n']} / {c['group_count']} | {points(c['delta'])} | {points(c['ci95'][0])} ~ {points(c['ci95'][1])} |")
    lines += ['', '| 모델 | source | 개선 | 동일 | 퇴보 |', '|---|---|---:|---:|---:|']
    for key,sources in summary['changes'].items():
        for source,counts in sources.items():
            lines.append(f"| {MODELS[key]} | {source} | {counts['improved']} | {counts['unchanged']} | {counts['regressed']} |")
    lines += ['', '개선·퇴보는 같은 문항의 이산 정답/lexical EM 변화에 대한 사후 집계다. 예시 opaque ID와 상태 변화는 run_summary.json에 보존하고 원문·응답은 로컬 제한 artifact에만 둔다. 한국어 성능 변화에는 정답 내용과 출력 형식 준수의 변화가 함께 포함된다. VQA lexical 개선에도 답변 길이·형식과 생성상한 잘림 감소가 영향을 줄 수 있으므로 시각적 추론 능력만의 변화로 단정하지 않는다. 잘림 수는 CSV/JSON에서 확인할 수 있다.']
    lines += ['', 'CI는 관측된 문서/이미지 그룹의 문항 표본 불확실성만 반영한다. 학습 seed 변동성·새 환자 임상 일반화는 측정하지 않았다. MedGemma의 SLAKE dataset-level 사전학습 노출은 알려져 있으나 정확한 test 노출 여부는 미상이다. 이번 실행 일반 모델들의 과거 노출도 미상으로 남겼다.', '',
      '## VQA 하위집단', '',
      '| 모델 | source | 조건 | 구분 | 값 | 예정/유효 | lexical EM | token F1 |',
      '|---|---|---|---|---|---:|---:|---:|']
    subgroups=0
    for ev in summary['evaluations']+summary['external_evaluations']:
        for group in ev.get('vqa_subgroups',[]):
            c=group['metrics'];subgroups+=1
            lines.append(f"| {ev['model_id']} | {group['source_id']} | {ev['mode']} | {group['dimension']} | {group['value'] or '미상'} | {c['scheduled']}/{c['valid_answers']} | {pct(metric(c))} | {pct(c['lexical_token_f1'])} |")
    if not subgroups:lines.append('| — | — | PREPARED_NOT_RUN | — | — | — | N/A | N/A |')
    lines += ['', '질문 유형과 영상 종류는 원본 metadata를 사용했다. 서로 겹치는 구분이므로 행들의 예정 수를 합산하지 않는다. 작은 하위집단의 차이는 임상 일반화를 뜻하지 않는다.', '',
      '## 추가 영상 진단·외부 평가', '',
      '| 모델 | 조건 | 예정 문항 | 차이와 해석 |', '|---|---|---:|---|']
    for key,d in summary['image_diagnostics'].items():
        for c in d['paired_image_minus_no_image']:
            value='N/A (이미지 제외 추론 전체 미지원)' if d['unsupported_ablation_count']==d['scheduled'] else 'lexical EM '+points(c['delta'])
            lines.append(f"| {key} | 같은 모델의 이미지 포함 − 제외 | {c['n']} | {value}; 미지원 {d['unsupported_ablation_count']}개 |")
    for key,pair in summary['external_paired'].items():
        for c in pair['comparison']:
            lines.append(f"| {MODELS[key]} | VQA-RAD 외부 test, Tuned − Base | {c['n']} | lexical EM {points(c['delta'])}, CI {points(c['ci95'][0])} ~ {points(c['ci95'][1])} |")
    if not summary['image_diagnostics'] and not summary['external_paired']:lines.append('| — | PREPARED_NOT_RUN | — | N/A |')
    for key,model in summary['models'].items():
        for name,outcome in model['campaign'].get('diagnostics',{}).items():
            if outcome.get('status') in ('FAILED_REPRODUCIBLE','SKIPPED_BUDGET','BLOCKED_HARDWARE','BLOCKED_ACCESS'):
                lines.append(f"| {MODELS[key]} | {name} | — | {outcome['status']}: {outcome.get('reason_code','예산 제한')} |")
    if summary['external_evaluations']:
        lines += ['', '| 모델 | VQA-RAD 조건 | 예정/시도/완료/유효 | lexical EM | token F1 | 기술 실패 |',
                  '|---|---|---:|---:|---:|---:|']
        for ev in summary['external_evaluations']:
            for c in ev['cohorts']:
                den='/'.join(str(c[k]) for k in ('scheduled','attempted','completed','valid_answers'))
                lines.append(f"| {ev['model_id']} | {ev['mode']} ({ev['status']}) | {den} | {pct(metric(c))} | {pct(c['lexical_token_f1'])} | {pct(c['technical_failure_rate'])} |")
    lines += ['', 'VQA-RAD 외부 평가는 OSF `phrase_type=test_freeform/test_para`에서 사전 정책으로 선택한64건이다. MedGemma 공식 카드가 보고한 balanced split 성적과 같은 시험이라고 주장하지 않는다.',
      '', '이미지 제외 조건에는 빈 이미지를 넣지 않는다. 동일 응답만으로 시각 정보가 무시됐다고 단정하지 않는다. 추가 진단은 본 평가 및 남은 모델 예산을 우선한 뒤 실행한다. 전문가 검토표는 로컬에만 준비하며 실제 검토 점수는 비어 있다.', '',
      '## 실행 비용·환경', '', 'RTX 3060 12GB에서 실제 CUDA 연산으로 학습 가능성을 확인했다. NVML kernel/userspace mismatch 때문에 nvidia-smi는 실패했지만 PyTorch forward/backward는 성공했다. 드라이버는 변경하지 않았다.', '',
      '| 기반 모델 | 누적 GPU 점유(시간) |', '|---|---:|']
    for model,seconds in (summary.get('budget') or {}).get('models',{}).items():lines.append(f'| {model} | {seconds/3600:.4f} |')
    lines += ['', 'GPU 사용량은 load/smoke/train/reload/evaluation을 포함한 단일 worker 점유 시간으로 계측한다. 위 표는 현재 원장에 기록된 모델별 누적 사용량이며 후속 실행도 포함할 수 있다. 진행 중인 active session의 미정산 시간은 이 표에 추가하지 않는다. '+resource_policy_description(summary)+' TTFT와 전력·에너지는 측정하지 않아 null이다. tokens/E2E는 전처리와 생성의 합산 처리량이며 순수 decode throughput이 아니다.', '',
      '## v1 실행 당시 수행·미실행·차단', '',
      '다음 항목은 v1 종료 당시 기록이다. 2026-10-26 이후 데이터 접근·권리·실행 범위와 실제 후속 결과는 `docs/agent/HANDOFF.md` 및 `reports/public_summary/v2/`를 따른다.', '',
      '- 실제 수행: 공개 원본 확보, 정제·중복/누수 감사, immutable split freeze, CPU 계약 검사, 위 표에 명시된 GPU 학습·재로드·평가.',
      '- 접근 재확인: 사용자 새 계정과 추가 승인으로 MedGemma 1.5와 Gemma 3 접근 모두 성공. 과거401/403 시도는 이력으로 보존. 일반 VLM은 Qwen3.5→Qwen3-VL→Qwen2.5-VL, 의료 VLM은 MedGemma→Gemma3 순으로 대체한다. Smol은 아직 완료하지 못한 VLM 셀의 마지막 대안이며 두 셀로 중복 계산하지 않는다.',
      '- 접근 차단: AIHub 71875/71762 인증 및 허용 범위 미확보.',
      '- 권리 확인 보류: MedQA/MedMCQA의 외부 시험·교재 파일 적용 권리 미확정. ROCOv2/PMC-VQA는 등록·권리 조사만 수행.',
      '- 장치 차단: Jetson SSH 호스트 키 검증 자료 없음. 호스트 검증을 우회하지 않았으며 장치 identity·실제 추론 결과는 없다.',
      '- 미실행: 전문가 사실성·안전성 검토, 추가 seed, 임상 검증, 서비스 배포. 자동 점수를 전문가 판정으로 대체하지 않았다.', '',
      '핵심 Python3.10 검사139passed/6skipped/6subtests와 별도 native CPU16passed 기록은 `verification.json`에 있다. 주 평가4쌍·외부 평가2쌍의 실제 예측4,560건에 대한 파일SHA, 고정 문항, 동일 pair 입력·생성 조건, 선정 adapter 연결의 최종 검증은 `final_artifact_verification.json`에 있다. 네 모델 모두 실제 의료 데이터 smoke20 update와 별도 process 재로드를 통과했다. 미실행 검사는 성공 수에 포함하지 않았다.', '',
      '전용 branch는 `codex/medical-finetune-v1-20261023`이다. 최종 전달 commit·remote SHA 일치·CI 결과는 전달 폴더의 `git_receipt.json`에 별도 보존한다. 각 학습/평가 run의 code commit과 source snapshot은 당시 실행 코드의 근거이며 최종 문서 commit과 구분한다.', '',
      '후속 연구의 우선순위는 준비한 익명 전문가 검토표의 사실성·안전성 판정, 그리고 별도 validation에서 응답 길이·형식을 사전 고정한 새 VQA 실험이다. 현재 결과를 보고 기존 test에 맞춘 설정 변경으로 이번 점수를 대체하지 않는다. 추가 seed는 선택적 후속 실험이며 이번 v1 완료의 선행 조건이 아니다.', '',
      '## 재개', '', '정확한 로컬 경로와 명령은 HANDOFF.md에 기록했다. 공개 저장소에서는 다음과 같이 기존 환경을 재사용한다.', '',
      '```bash', 'export MEDVLM_DATA_ROOT="$(realpath ../medvlm-data)"', 'PYTHONPATH=src ../.venv-medvlm/bin/python -m mediflow_medvlm status',
      'PYTHONPATH=src ../.venv-medvlm/bin/python scripts/complete_v1.py --data-root "$MEDVLM_DATA_ROOT"',
      'PYTHONPATH=src ../.venv-medvlm/bin/python scripts/report_v1.py --data-root "$MEDVLM_DATA_ROOT" --output reports/public_summary --plot', '```', '',
      '완료한 run은 동일 config/data/model identity 확인 후 재사용한다. 진행 중인 소유 프로세스가 있으면 중복 실행하지 않는다. 예산 ledger와 checkpoint·source snapshot을 함께 보존한다.']
    lines += ['', '## 실제 시험 추론 계측', '',
              '| 모델 | 조건 | 계측 시도 수 | 생성 token | 합산 E2E(초) | token / 합산 E2E | E2E p95(초) | peak allocated / reserved bytes |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for ev in summary['evaluations']:
        timing=ev.get('runtime')
        if not timing:continue
        rate=timing['generated_tokens_per_summed_e2e_second'];p95=timing['p95_e2e_seconds_nearest_rank']
        lines.append(f"| {ev['model_id']} | {ev['mode']} | {timing['measured_attempts']} | {timing['generated_tokens']} | {timing['summed_e2e_seconds']:.3f} | {rate:.3f} | {p95:.3f} | {ev['peak_allocated_bytes']} / {ev['peak_reserved_bytes']} |" if rate is not None and p95 is not None else f"| {ev['model_id']} | {ev['mode']} | {timing['measured_attempts']} | {timing['generated_tokens']} | N/A | N/A | N/A | N/A |")
    lines += ['', 'E2E p95는 계측된 문항 지연의 nearest-rank 값이다. model load와 validation warmup은 제외하고 전처리·생성·parsing은 포함한다. peak memory는 run 전체 단위이며 source별 별도 peak로 해석하지 않는다.']
    return '\n'.join(lines)+'\n'


def adapter_cards(summary):
    lines=['# 모델·adapter 실행 카드','',
           '공개 의료 데이터의 오프라인 연구용이다. 모델이나 adapter 가중치를 공개 업로드하지 않았다. 자동 lexical 점수와 객관식 결과는 임상 검증이 아니며 전문가 검토는 미실행이다. 각 adapter에는 기반 모델 및 학습 데이터의 이용 조건이 함께 적용된다. `${MEDVLM_DATA_ROOT}`는 실제 로컬 data root다.','']
    for key,m in summary['models'].items():
        recipe=m['recipe'];native=m['model'];adapter=m.get('adapter',{}) or {};probe=m['access_probe']
        lines.extend([f"## {m['model_id']}",'',f"학습 상태 `{m['status']}`, 재로드 `{m['reload']}`, 전체 셀 `{m['campaign'].get('status','PREPARED_NOT_RUN')}`.",'',
          '| 항목 | 실제 값 또는 사전 설정 |','|---|---|',
          f"| Base revision | `{m['revision']}` |",
          f"| 로컬 native weights 파일 | {m['weights']['status']} ({m['weights'].get('present_weight_files',0)}/{m['weights'].get('expected_weight_files') or '총수 미확인'} files, {m['weights'].get('present_bytes',0)} bytes); 로드 성공과 별개 |",
          f"| Base native files digest | `{(native.get('native_files') or {}).get('sha256','미측정')}` |",
          f"| Processor / loader | {native.get('processor_class') or probe.get('processor_class')} / {native.get('loader')} |",
          f"| Native chat template SHA256 | `{native.get('chat_template_sha256')}` |",
          f"| Adapter 경로 | `{adapter.get('path','미생성')}` |",
          f"| Adapter digest / parent | `{adapter.get('digest')}` / `{adapter.get('base_revision')}` |",
          f"| 학습 source / config hash | `{m['code'].get('training_code_sha256')}` / `{m.get('config_hash')}` |",
          f"| 설정 seed / rank / alpha / dropout | {recipe.get('seed')} / {recipe.get('lora_rank')} / {recipe.get('lora_alpha')} / {recipe.get('lora_dropout')} |",
          f"| 설정 dtype / quantization / max length | {recipe.get('dtype')} / {recipe.get('quantization')} / {recipe.get('max_length')} |",
          f"| 실제 학습 / 동결 parameter 수 | {native.get('trainable_parameters')} / {native.get('frozen_parameters')} |",
          f"| 실제 target module 수 / SHA256 | {m['target_modules_count']} / `{m['target_modules_sha256']}` |",
          f"| 실제 sample pass / target token / update | {m.get('samples_processed')} / {m.get('supervised_tokens')} / {m.get('optimizer_steps')} |",
          f"| 실제 epoch / effective batch | {m.get('actual_epochs')} / {m.get('effective_batch')} |",
          f"| 학습 wall seconds / GPU seconds | {m.get('wall_seconds')} / {m.get('gpu_seconds')} |",
          f"| Peak CUDA allocated / reserved bytes | {m.get('peak_allocated_bytes')} / {m.get('peak_reserved_bytes')} |",
          f"| 선언 license / 원본 근거 | {probe.get('declared_license')} / {probe.get('source_url')} |",''])
        evidence=m.get('license_evidence')
        if evidence:lines.extend([f"추가 이용 조건 근거: {evidence.get('exact_name')}. {evidence.get('scope_note','')} [{evidence.get('license_id','license')}]({evidence.get('source_url')}).",''])
        lines.extend(['원본 설정, 정확한 target module 목록, 환경 버전, train/dev digest, validation 선정과 source snapshot은 해당 로컬 run의 `config.json`, `manifest.json`, `selection.json`, `reload.json`에 보존한다. 동결 parameter 수는 로드한 quantized tensor의 numel 합으로, 원본 모델 규모와 다를 수 있다. 미실행 모델의 null은 측정값이 아니다.',''])
    return '\n'.join(lines)


def plots(summary,destination):
    os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/f'medvlm-matplotlib-{os.getuid()}'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    sources=('pubmedqa_labeled','kormedmcqa','medquad','slake')
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for ax,source in zip(axes.flat,sources):
        series=[]
        for key in KEYS:
            vals=[];ns=[]
            for mode in ('base','tuned'):
                e=next(e for e in summary['evaluations'] if e['key']==key and e['mode']==mode)
                cohort=[c for c in e['cohorts'] if c['source_id']==source]
                n=sum(c['scheduled'] for c in cohort);ns.append(n)
                vals.append(sum(metric(c)*c['scheduled'] for c in cohort)/n if n else None)
            if any(v is not None for v in vals):series.append((key,vals,ns))
        for i,(key,vals,ns) in enumerate(series):
            for j,(value,color) in enumerate(zip(vals,('#8ca5b6','#286b54'))):
                if value is not None:
                    ax.bar(i+(j-.5)*.35,value,width=.34,color=color,label=('Base' if j==0 else 'Tuned') if i==0 else None)
                    ax.annotate(f'{value:.1%}',(i+(j-.5)*.35,value),xytext=(0,3),textcoords='offset points',ha='center',fontsize=8)
            ax.text(i,.97,f'n={ns[0]}/{ns[1]}',ha='center',va='top',fontsize=8)
        ax.set_xticks(range(len(series)),[x[0] for x in series],rotation=15)
        ax.set_ylim(0,1);ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_title(source+(' | end-to-end accuracy' if source in sources[:2] else ' | lexical exact match'))
        if not series:ax.text(.5,.5,'NOT RUN',transform=ax.transAxes,ha='center',color='#666666')
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    for ax in axes.flat:
        handles,labels=ax.get_legend_handles_labels()
        if handles:
            fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False)
            break
    fig.suptitle('Frozen paired test evaluation | single seed, preliminary; lexical scores are not clinical accuracy',fontsize=11)
    fig.savefig(destination/'paired_scores.png',dpi=170)
    fig.savefig(destination/'paired_scores.svg')
    svg=destination/'paired_scores.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);p.add_argument('--output',required=True);p.add_argument('--plot',action='store_true');args=p.parse_args()
    root=Path(args.data_root).resolve();repo=Path(__file__).resolve().parents[1];out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    summary=collect(root,repo);atomic_json(out/'run_summary.json',summary)
    (out/'RESULTS_KO.md').write_text(report(summary,root,repo),encoding='utf-8')
    (out/'ADAPTER_CARDS.md').write_text(adapter_cards(summary),encoding='utf-8')
    rows=csv_rows(summary)
    if rows:
        with (out/'paired_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    if args.plot:plots(summary,out)
    print(json.dumps({'output':str(out),
        'optimizer_updated_models':sum(x.get('optimizer_steps',0)>0 for x in summary['models'].values()),
        'trained_and_reloaded_models':sum(x['status']=='SUCCEEDED' and x['reload']=='SUCCEEDED' and x['reload_verified'] for x in summary['models'].values()),
        'evaluated_pairs':len(summary['paired'])}))

if __name__=='__main__':main()
