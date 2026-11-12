#!/usr/bin/env python3
"""Deliver verified aggregate reports and editable local review copies only."""
import argparse
import json
from pathlib import Path
import shutil

from mediflow_medvlm.runtime import atomic_json, file_hash


def copy_once(source, target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if file_hash(source)!=file_hash(target):
            raise ValueError('DELIVERY_CHANGED_PRESERVE_EXISTING: '+target.name)
    else:
        shutil.copyfile(source,target)


def export(root, reports, output):
    summary=json.loads((reports/'run_summary.json').read_text())
    verification=json.loads((reports/'verification.json').read_text())
    if summary['status']!='SUCCEEDED' or verification['errors']:
        raise ValueError('VERIFIED_COMPLETED_REPORTS_REQUIRED')
    output.mkdir(parents=True,exist_ok=True)
    for path in reports.rglob('*'):
        if path.is_file():
            relative=path.relative_to(reports)
            if (str(relative) not in {'RESULTS_KO.md','run_summary.json','verification.json'} and
                    not (relative.parts[0]=='figures' and relative.suffix in {'.png','.svg'})):
                raise ValueError('UNRECOGNIZED_PUBLIC_REPORT_FILE')
            copy_once(path,output/path.relative_to(reports))
    for model in ('qwen3_17b','qwen3_4b'):
        folder=root/'v4/review'/model
        manifest=json.loads((folder/'manifest.json').read_text())
        for name in ('review.csv','instructions.txt'):
            if file_hash(folder/name)!=manifest['files_sha256'][name]:
                raise ValueError('ORIGINAL_REVIEW_PACKET_CHANGED')
            # Existing editable copies may contain human ratings; never replace
            # or treat them as original machine-checked clinical judgements.
            target=output/'review'/model/name
            if not target.exists():
                copy_once(folder/name,target)
    state={'status':'COMPLETED_DECLARED_EXPERIMENT','actually_completed':{
        'paired_sample_preparation':True,'train_rows':4000,'validation_rows':105,'derived_test_rows':256,
        'training_and_reload_stages':4,'new_evaluations':6,'reused_base_controls':2,
        'blind_review_packets':2,'human_reviewed_rows_at_experiment_completion':0},
        'not_run':{'official_make_train_data_py':'NOT_OBTAINED; provider-described method independently implemented',
                   'full_source_training':'NOT_SELECTED_SAMPLED_CORPUS',
                   'additional_seeds':'NOT_SELECTED','new_vlm_training':'OUTSIDE_THIS_EXPERIMENT',
                   'clinical_review':'NOT_PERFORMED','jetson':'USER_DEFERRED'},
        'unavailable_information':{'original_qa_mapping':'provider states no fixed per-question mapping',
                                   'augmentation_parent_ids':'provider states not provided; cannot guarantee lineage isolation'},
        'execution_blockers':[],'next_action_requires_user_input':False}
    atomic_json(output/'stage_status.json',state)
    repo=Path(__file__).resolve().parents[1]
    commands=[f'cd {repo}', 'export PYTHONPATH="$PWD/src"',
              'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_DISABLED=true',
              f'{root.parent/".venv-medvlm/bin/python"} scripts/prepare_71762_v4.py --data-root {root}',
              f'{root.parent/".venv-medvlm/bin/python"} scripts/run_71762_v4.py --data-root {root} --phase all',
              f'{root.parent/".venv-medvlm/bin/python"} scripts/report_71762_v4.py --data-root {root} --output {reports}']
    (output/'RESUME_KO.md').write_text('# 71762 v4 재개·검증\n\n현재 선언 실험은 완료했습니다. 아래 명령은 성공 산출물을 검증 후 재사용합니다. 이미 실행 중인 worker가 있으면 잠금으로 중복 실행을 거절합니다.\n\n```bash\n'+'\n'.join(commands)+'\n```\n\n과거 v1/v2/v3 보고서는 당시 상태로 보존합니다. 현재 GPU 누적량이 증가했으므로 옛 v3 보고서를 당시의 GPU 불변성 조건으로 재생성하지 마세요. 현재 실적 검증에는 v4 보고서를 사용합니다.\n\nreview/의 CSV 전달 사본만 편집하세요. private A/B 키는 이 전달물에 포함하지 않았으며 사람의 평가값을 자동으로 채우지 않았습니다.\n')
    files={str(p.relative_to(output)):{'bytes':p.stat().st_size,'sha256':file_hash(p)}
           for p in output.rglob('*') if p.is_file() and p.name!='output_manifest.json'}
    atomic_json(output/'output_manifest.json',{'schema':'medvlm.aihub71762.delivery.v4','files':files,
                 'raw_archives_or_weights_or_full_prediction_files_exported':False,
                 'selected_review_texts_included_locally':True,'private_ab_keys_exported':False})
    print(json.dumps({'status':'SUCCEEDED','output':str(output),'files':len(files)}))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--reports',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();export(args.data_root.resolve(),args.reports.resolve(),args.output.resolve())
