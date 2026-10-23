#!/usr/bin/env python3
"""Generate aggregate-only Korean handoff artifacts from actual local evidence."""
from __future__ import annotations
import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone

from mediflow_medvlm.runtime import atomic_json, canonical_hash, file_hash

KEYS = ('qwen3_17b','qwen3_4b','qwen35_2b','medgemma15_4b','gemma3_4b','smolvlm_500m')
MODELS = {'qwen3_17b':'Qwen/Qwen3-1.7B','qwen3_4b':'Qwen/Qwen3-4B',
          'qwen35_2b':'Qwen/Qwen3.5-2B','medgemma15_4b':'google/medgemma-1.5-4b-it','gemma3_4b':'google/gemma-3-4b-it',
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


def metric(row):
    classification = row['task_type'] in {'medical_mcq','context_qa'}
    return row.get('end_to_end_correct_fraction' if classification else 'end_to_end_lexical_exact_match')


def collect(root, repo):
    summary={'schema_version':'medvlm.public-result.v1','generated_at':datetime.now(timezone.utc).isoformat(),
             'preliminary_single_seed':True,'human_review':'not_performed','clinical_validation':False,
             'data':read(repo/'reports/public_summary/data_freeze.json'), 'models':{},'evaluations':[],
             'paired':{},'external_paired':{},'image_diagnostics':{},'changes':{},
             'device':{k:v for k,v in read(root/'runs/device/jetson-inventory-20261023/summary.json',{}).items()
                                  if k in ('status','reason_code','evidence_level','device_outcome','wall_seconds','finished_at')},
             'budget':read(root/'runtime/gpu_budget.json')}
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
        record.update(model_id=MODELS[key],revision=manifest.get('revision'),status=manifest.get('status','PREPARED_NOT_RUN'),
                      manifest_sha256=file_hash(train/'manifest.json') if manifest else None,
                      smoke={k:sm.get(k) for k in ('status','evidence_level','optimizer_steps','adapter_changed')},
                      reload=read(train/'reload.json',{}).get('status','PREPARED_NOT_RUN'),
                      selected_checkpoint=selection.get('checkpoint'),selection=selection,reload_verified=selection.get('reload_verified',False),
                      model={k:model.get(k) for k in ('dtype','quantization','native_files','chat_template_sha256','trainable_parameters','frozen_parameters','vision_encoder_frozen','loader','processor_class')},
                      target_modules_count=len(model.get('exact_target_modules',[])),
                      target_modules_sha256=canonical_hash(model.get('exact_target_modules',[])),
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
        pair=read(root/'runs'/f'{key}-paired-comparison.json')
        if pair:summary['paired'][key]=pair
        external=read(root/'runs'/f'{key}-vqarad-paired-comparison.json')
        if external:summary['external_paired'][key]=external
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
        summary['models'][name]['access_probe']={k:probe.get(k) for k in ('status','reason_code','http_status','access_verified_config','processor_class','declared_license','source_url')}
    return portable(summary,root)


def csv_rows(summary):
    rows=[]
    for ev in summary['evaluations']:
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
    lines += ['', '학습에는 MedQuAD/PubMedQA와 SLAKE만 사용했다. KorMedMCQA는 한국어 원문 평가 전용, VQA-RAD는 외부 평가 전용이다. 합성·번역 의료 학습 데이터는 추가하지 않았다. 환자 독립성은 확인할 수 없으며, 이미지 유사 후보를 보수적으로 같은 그룹에 묶어 학습 데이터를 제외했다. 소스별 정확한 제외 수·권리·revision·분할 해시는 DATA_REPORT_KO.md와 공개 manifest를 참조한다.', '',
      '## 실제 학습 및 재로드', '', '| 모델 | v1 상태 | 실제 update | 처리 sample 횟수 | target token | 재로드 |', '|---|---|---:|---:|---:|---|']
    for key,m in summary['models'].items():
        lines.append(f"| {m['model_id']} | {m['status']} | {m.get('optimizer_steps','—')} | {m.get('samples_processed','—')} | {m.get('supervised_tokens','—')} | {m.get('reload','미실행')} |")
    lines += ['', '학습 횟수와 평가 완료 수는 별개다. smoke와 본 학습도 별도 run으로 보존했다. Qwen은 NF4 QLoRA, SmolVLM은 BF16 LoRA이며 vision encoder/projector는 동결했다. 선택은 고정 validation loss에만 근거하고 test는 선택에 사용하지 않았다.', '',
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
      '| 모델 | source | 문항 / 관측 그룹 | Tuned − Base | 그룹 bootstrap 95% CI |', '|---|---|---:|---:|---|']
    for key,pair in summary['paired'].items():
        for c in pair['comparison']:
            lines.append(f"| {MODELS[key]} | {c['source_id']} | {c['n']} / {c['group_count']} | {pct(c['delta'])} | {pct(c['ci95'][0])} ~ {pct(c['ci95'][1])} |")
    lines += ['', 'CI는 관측된 문서/이미지 그룹의 문항 표본 불확실성만 반영한다. 학습 seed 변동성·새 환자 임상 일반화는 측정하지 않았다. MedGemma의 SLAKE dataset-level 사전학습 노출은 알려져 있으나 정확한 test 노출 여부는 미상이다. 이번 실행 일반 모델들의 과거 노출도 미상으로 남겼다.', '',
      '## 추가 영상 진단·외부 평가', '',
      '| 모델 | 조건 | 예정 문항 | 차이와 해석 |', '|---|---|---:|---|']
    for key,d in summary['image_diagnostics'].items():
        for c in d['paired_image_minus_no_image']:
            lines.append(f"| {key} | 같은 모델의 이미지 포함 − 제외 | {c['n']} | lexical EM {pct(c['delta'])}; 미지원 {d['unsupported_ablation_count']}개 |")
    for key,pair in summary['external_paired'].items():
        for c in pair['comparison']:
            lines.append(f"| {MODELS[key]} | VQA-RAD 외부 test, Tuned − Base | {c['n']} | lexical EM {pct(c['delta'])}, CI {pct(c['ci95'][0])} ~ {pct(c['ci95'][1])} |")
    if not summary['image_diagnostics'] and not summary['external_paired']:lines.append('| — | PREPARED_NOT_RUN | — | N/A |')
    lines += ['', '이미지 제외 조건에는 빈 이미지를 넣지 않는다. 동일 응답만으로 시각 정보가 무시됐다고 단정하지 않는다. 추가 진단은 본 평가 및 남은 모델 예산을 우선한 뒤 실행한다. 전문가 검토표는 로컬에만 준비하며 실제 검토 점수는 비어 있다.', '',
      '## 실행 비용·환경', '', 'RTX 3060 12GB에서 실제 CUDA 연산으로 학습 가능성을 확인했다. NVML kernel/userspace mismatch 때문에 nvidia-smi는 실패했지만 PyTorch forward/backward는 성공했다. 드라이버는 변경하지 않았다.', '',
      '| 기반 모델 | 누적 GPU 점유(시간) |', '|---|---:|']
    for model,seconds in (summary.get('budget') or {}).get('models',{}).items():lines.append(f'| {model} | {seconds/3600:.4f} |')
    lines += ['', 'GPU 예산은 load/smoke/train/reload/evaluation을 포함한 단일 worker 점유 시간으로 계측한다. 상한은 전체24시간·기반모델별6시간이다. TTFT와 전력·에너지는 측정하지 않아 null이다. tokens/E2E는 전처리와 생성의 합산 처리량이며 순수 decode throughput이 아니다.', '',
      '## 실제 수행·미실행·차단', '',
      '- 실제 수행: 공개 원본 확보, 정제·중복/누수 감사, immutable split freeze, CPU 계약 검사, 위 표에 명시된 GPU 학습·재로드·평가.',
      '- 접근 재확인: 사용자 새 계정과 추가 승인으로 MedGemma 1.5와 Gemma 3 접근 모두 성공. 과거401/403 시도는 이력으로 보존. Gemma3→Smol은 MedGemma 실패 시에만 사용한다.',
      '- 접근 차단: AIHub 71875/71762 인증 및 허용 범위 미확보.',
      '- 권리 확인 보류: MedQA/MedMCQA의 외부 시험·교재 파일 적용 권리 미확정. ROCOv2/PMC-VQA는 등록·권리 조사만 수행.',
      '- 장치 차단: Jetson SSH 호스트 키 검증 자료 없음. 호스트 검증을 우회하지 않았으며 장치 identity·실제 추론 결과는 없다.',
      '- 미실행: 전문가 사실성·안전성 검토, 추가 seed, 임상 검증, 서비스 배포. 자동 점수를 전문가 판정으로 대체하지 않았다.', '',
      '## 재개', '', '정확한 로컬 경로와 명령은 HANDOFF.md에 기록했다. 공개 저장소에서는 다음과 같이 기존 환경을 재사용한다.', '',
      '```bash', 'export MEDVLM_DATA_ROOT="$(realpath ../medvlm-data)"', 'PYTHONPATH=src ../.venv-medvlm/bin/python -m mediflow_medvlm status',
      'PYTHONPATH=src ../.venv-medvlm/bin/python scripts/complete_v1.py --data-root "$MEDVLM_DATA_ROOT" --skip-download',
      'PYTHONPATH=src ../.venv-medvlm/bin/python scripts/report_v1.py --data-root "$MEDVLM_DATA_ROOT" --output reports/public_summary --plot', '```', '',
      '완료한 run은 동일 config/data/model identity 확인 후 재사용한다. 진행 중인 소유 프로세스가 있으면 중복 실행하지 않는다. 예산 ledger와 checkpoint·source snapshot을 함께 보존한다.']
    return '\n'.join(lines)+'\n'


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
        if series:ax.legend(loc='upper right')
        else:ax.text(.5,.5,'NOT RUN',transform=ax.transAxes,ha='center',color='#666666')
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Frozen paired test evaluation | single seed, preliminary; lexical scores are not clinical accuracy',fontsize=11)
    fig.savefig(destination/'paired_scores.png',dpi=170)
    fig.savefig(destination/'paired_scores.svg')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);p.add_argument('--output',required=True);p.add_argument('--plot',action='store_true');args=p.parse_args()
    root=Path(args.data_root).resolve();repo=Path(__file__).resolve().parents[1];out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    summary=collect(root,repo);atomic_json(out/'run_summary.json',summary)
    (out/'RESULTS_KO.md').write_text(report(summary,root,repo),encoding='utf-8')
    rows=csv_rows(summary)
    if rows:
        with (out/'paired_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    if args.plot:plots(summary,out)
    print(json.dumps({'output':str(out),'trained':sum(x.get('optimizer_steps',0)>0 for x in summary['models'].values()),'evaluated_pairs':len(summary['paired'])}))

if __name__=='__main__':main()
