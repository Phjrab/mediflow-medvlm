"""Author-distributed Korean examination QA, strictly evaluation-only in v1."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import urllib.request
from .io import atomic_json,download,now,sha256
from .schema import sample,input_digest

REPOSITORY='sean0042/KorMedMCQA'


def fetch_kormedmcqa(data_root):
    root=Path(data_root)/'raw'/'kormedmcqa';root.mkdir(parents=True,exist_ok=True)
    meta=root/'source_api.json'
    if not meta.exists():
        download(f'https://huggingface.co/api/datasets/{REPOSITORY}',meta,max_bytes=1024**2)
    info=json.loads(meta.read_text());revision=info['sha']
    if info.get('cardData',{}).get('license')!='cc-by-nc-2.0':
        raise ValueError('BLOCKED_LICENSE: dataset license requires renewed review')
    files=[]
    for item in info['siblings']:
        name=item['rfilename']
        if not (name=='README.md' or name.endswith('.parquet')):continue
        path=root/revision/name
        download(f'https://huggingface.co/datasets/{REPOSITORY}/resolve/{revision}/{name}',path,max_bytes=5*1024**2)
        files.append({'path':str(path.relative_to(root)), 'bytes':path.stat().st_size,'sha256':sha256(path)})
    return root/revision,revision,files


def build_kormedmcqa(repository:Path,revision:str,fetched_at:str):
    try: import pyarrow.parquet as pq
    except ImportError as exc: raise RuntimeError('Install project data extra (pyarrow) before reading official parquet') from exc
    output=[];counts=Counter();subject_counts={};all_ids=defaultdict(list);content=defaultdict(list);exclusions=[]
    for subject in ('doctor','nurse','pharm','dentist'):
        subject_counts[subject]={}
        for path in sorted((repository/subject).glob('*.parquet')):
            split=path.name.split('-')[0];rows=pq.read_table(path).to_pylist();subject_counts[subject][split]=len(rows)
            counts['raw_qa']+=len(rows)
            file_hash=sha256(path)
            for row in rows:
                oid='/'.join(str(row[k]) for k in ('subject','year','period','q_number'))
                index=int(row['answer'])
                if index not in range(1,6):raise ValueError('Unexpected 1-based answer range')
                option='ABCDE'[index-1]
                rec=sample('kormedmcqa',revision,oid,'medical_mcq',row['question'],option,'validation' if split=='dev' else split,language='ko',options={k:row[k] for k in 'ABCDE'},option_id=option,
                    source_url=f'https://huggingface.co/datasets/{REPOSITORY}',document=f'original_question:{oid}',original_split=split,
                    metadata={k:row[k] for k in ('subject','year','period','q_number')},license_ref=f'https://huggingface.co/datasets/{REPOSITORY}/blob/{revision}/README.md',fetched_at=fetched_at,raw_sha256=file_hash,annotation='original_licensing_exam_answer')
                all_ids[oid].append(split);content[input_digest(rec)].append((oid,split))
                if split not in {'dev','test'}:
                    counts[f'excluded_{split}_evaluation_only']+=1
                    exclusions.append({'original_id':oid,'reason':f'{split}_excluded_from_v1_SFT_and_zero_shot_eval'})
                    continue
                output.append(rec)
    # Keep official test immutable, remove matching dev questions if present.
    test_keys={input_digest(r) for r in output if r['split']=='test'}
    cleaned=[]
    for row in output:
        if row['split']=='validation' and input_digest(row) in test_keys:
            exclusions.append({'sample_id':row['sample_id'],'reason':'dev_exact_overlap_test'});counts['dev_exact_overlap_test']+=1
        else:cleaned.append(row)
    duplicate_groups=[{'original_id':oid,'splits':splits} for oid,splits in all_ids.items() if len(splits)>1]
    return cleaned,{'counts':dict(counts),'subject_original_splits':subject_counts,'unique_original_ids':len(all_ids),'unique_normalized_inputs':len(content),'original_id_duplicate_groups':duplicate_groups,'fewshot_dev_overlap_rows':sum('fewshot' in v and 'dev' in v for v in all_ids.values()),'content_duplicate_groups':[{'members':[{'original_id':oid,'split':sp} for oid,sp in v]} for v in content.values() if len(v)>1],'exclusions':exclusions,'selection_policy':{'sft_allowed':False,'zero_shot':True,'official_dev_maps_to':'validation','answer_index_base':1,'answer_conversion':'1=A,2=B,3=C,4=D,5=E'},'license_scope':'CC-BY-NC-2.0 author distribution, attribution and noncommercial offline research only. No automatic commercial kiosk rights for underlying examination questions.'}
