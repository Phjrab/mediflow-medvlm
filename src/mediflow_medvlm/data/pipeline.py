"""Acquisition, schema/group audits, content-addressed data freeze and summary."""
from __future__ import annotations
from collections import Counter,defaultdict
from pathlib import Path
import json
import re
import shutil
from .io import atomic_json, digest, git_snapshot, now, read_jsonl, sha256, write_jsonl
from .schema import input_digest, normalized, validate_record
from .medquad import build_medquad
from .pubmedqa import build_pubmedqa
from .kormedmcqa import fetch_kormedmcqa, build_kormedmcqa

SOURCES=('medquad','pubmedqa_labeled','kormedmcqa')
BUILDERS={'medquad':build_medquad,'pubmedqa_labeled':build_pubmedqa,'kormedmcqa':build_kormedmcqa}
SOURCE_URLS={'medquad':'https://github.com/abachaa/MedQuAD','pubmedqa_labeled':'https://github.com/pubmedqa/pubmedqa','kormedmcqa':'https://huggingface.co/datasets/sean0042/KorMedMCQA'}


def public_paths(value,data_root):
    """Git summaries use a portable root; restricted local evidence stays exact."""
    if isinstance(value,str):return value.replace(str(Path(data_root).resolve()),'${MEDVLM_DATA_ROOT}')
    if isinstance(value,list):return [public_paths(item,data_root) for item in value]
    if isinstance(value,dict):return {public_paths(key,data_root):public_paths(item,data_root) for key,item in value.items()}
    return value


def manifest_path(root,source):return Path(root)/'manifests'/'sources'/f'{source}.json'


def fetch(root,source):
    path=manifest_path(root,source)
    if path.exists():
        result=json.loads(path.read_text())
        if result.get('acquisition_status')=='SUCCEEDED':return result
    repo,revision,files=fetch_kormedmcqa(root) if source=='kormedmcqa' else git_snapshot(source,root)
    license_name={'medquad':'CC-BY-4.0','pubmedqa_labeled':'MIT repository distribution; underlying abstract rights retained','kormedmcqa':'CC-BY-NC-2.0'}[source]
    result={'schema_version':'medvlm.source.v1','source_id':source,'official_url':SOURCE_URLS[source],'resolved_revision':revision,'retrieved_at':now(),'raw_path':str(repo),'files':files,'total_bytes':sum(f['bytes'] for f in files),'access_status':'public_author_distribution','acquisition_status':'SUCCEEDED','status':'PREPARED_NOT_RUN','evidence_level':'CPU_TESTED','declared_license':license_name,'resolved_applicable_license':license_name,'code_license':'MIT' if source=='pubmedqa_labeled' else None,'original_content_license':'see per-sample source URL; retained underlying rights','license_evidence':[SOURCE_URLS[source]+'/blob/'+revision+'/'+('LICENSE.txt' if source=='medquad' else 'LICENSE' if source=='pubmedqa_labeled' else 'README.md')],'raw_redistribution':False,'fetched_files_sha256':digest(files)}
    atomic_json(path,result);return result


def verify_raw(manifest):
    root=Path(manifest['raw_path'])
    # HF manifest paths include their pinned revision directory.
    for f in manifest['files']:
        path=root/f['path']
        if not path.exists() and root.name==manifest['resolved_revision']:path=root.parent/f['path']
        if not path.is_file() or path.stat().st_size!=f['bytes'] or sha256(path)!=f['sha256']:
            raise ValueError(f'Immutable raw checksum mismatch: {f["path"]}')


def write_source(data_root:Path,source_id:str,records:list,stats:dict):
    data_root=Path(data_root);out=data_root/'processed'/source_id
    errors=[{'sample_id':r.get('sample_id'),'errors':validate_record(r)} for r in records if validate_record(r)]
    if errors:raise ValueError(f'Canonical schema errors ({len(errors)}), first: {errors[:1]}')
    paths={}
    for split in ('train','validation','test'):
        rows=sorted([r for r in records if r['split']==split],key=lambda r:r['sample_id'])
        path=out/f'{split}.jsonl'
        # Never destroy previously frozen bytes even when a new processing version is requested.
        if path.exists() and (data_root/'manifests'/'current_freeze.json').exists():
            history=data_root/'processed_history'/source_id/sha256(path)/path.name
            if not history.exists():history.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,history)
        write_jsonl(path,rows)
        paths[split]={'path':str(path),'sha256':sha256(path),'count':len(rows)}
    stats={**stats,'canonical_counts':{k:v['count'] for k,v in paths.items()},'canonical_files':paths,'status':'SUCCEEDED','evidence_level':'CPU_TESTED'}
    atomic_json(data_root/'manifests'/'build'/f'{source_id}.json',stats)
    srcpath=manifest_path(data_root,source_id)
    if srcpath.exists():
        manifest=json.loads(srcpath.read_text());manifest.update({'status':'SUCCEEDED','used_counts':stats['canonical_counts'],'excluded_counts':dict(Counter(x['reason'] for x in stats.get('exclusions',[]))),'actual_downloaded_count':stats.get('counts',{}).get('raw_qa'),'original_split_counts':stats.get('original_split_counts',stats.get('subject_original_splits')),'license_scope':stats.get('license_scope')});atomic_json(srcpath,manifest)
    return {k:v for k,v in stats.items() if k not in {'exclusions','content_duplicate_groups','original_id_duplicate_groups'}}


def build(root,source):
    manifest=json.loads(manifest_path(root,source).read_text());verify_raw(manifest)
    records,stats=BUILDERS[source](Path(manifest['raw_path']),manifest['resolved_revision'],manifest['retrieved_at'])
    return write_source(Path(root),source,records,stats)


def source_files(root):
    return sorted(p for p in (Path(root)/'processed').glob('*/*.jsonl') if p.parent.name not in {'text','vlm'} and p.stem in {'train','validation','test'})


def _near_candidates(records,max_pairs=5000000):
    """Conservative exact token-shingle Jaccard candidates; never silently merge."""
    buckets=defaultdict(list);shingles={};lookup={r['sample_id']:r for r in records}
    pairs=set();examined=0
    for row in records:
        if row['task_type']=='medical_vqa':continue
        words=normalized(row['input']['question']).split()
        if len(words)<6:continue
        grams={' '.join(words[i:i+3]) for i in range(len(words)-2)}
        sid=row['sample_id'];shingles[sid]=grams
        signatures=sorted(digest(g)[:16] for g in grams)[:3]
        for sig in signatures:
            key=(len(words)//5,sig)
            for other in buckets[key][:100]:pairs.add(tuple(sorted((sid,other))))
            buckets[key].append(sid)
    matches=[]
    for a,b in sorted(pairs):
        if examined>=max_pairs:break
        ra,rb=lookup[a],lookup[b]
        examined+=1
        aa,bb=shingles[a],shingles[b];score=len(aa&bb)/len(aa|bb)
        if score>=.95 and input_digest(ra)!=input_digest(rb):
            matches.append({'sample_ids':[a,b],'question_trigram_jaccard':round(score,5),'cross_split':ra['split']!=rb['split'],'action':'candidate_only_manual_review','reason':'Different inputs can differ in medical units, negation, options or context; no automatic semantic merge'})
    return {'method':'word_trigram_jaccard','threshold':.95,'candidate_pairs_generated':len(pairs),'candidate_pairs_examined':examined,'candidate_pair_budget':max_pairs,'candidate_budget_reached':len(pairs)>max_pairs,'bucket_member_comparison_cap':100,'candidates':matches,'clinical_equivalence_verified':False}


def audit(root):
    root=Path(root);records=[];files={}
    for p in source_files(root):
        rows=read_jsonl(p);records.extend(rows);files[f'{p.parent.name}/{p.stem}']={'path':str(p),'sha256':sha256(p),'count':len(rows)}
    errors=[];ids=Counter();groups=defaultdict(set);inputs=defaultdict(list);checked_images={}
    if not records:errors.append({'reason':'no_actual_canonical_records'})
    for row in records:
        sid=row.get('sample_id');ids[sid]+=1
        for err in validate_record(row):errors.append({'sample_id':sid,'reason':err})
        for key,value in row.get('group_ids',{}).items():
            if value is None:continue
            for v in value if isinstance(value,list) else [value]:groups[(row['source_id'],key,str(v))].add(row['split'])
        pixel_hash = row.get('metadata',{}).get('decoded_pixel_hash')
        if pixel_hash:
            groups[('cross_source','decoded_pixel_hash',pixel_hash)].add(row['split'])
        byte_hash = row.get('metadata',{}).get('image_sha256')
        if byte_hash:
            groups[('cross_source','image_sha256',byte_hash)].add(row['split'])
        inputs[input_digest(row)].append(row)
        for image in row['input'].get('images',[]):
            p=Path(image if isinstance(image,str) else image.get('path',image.get('local_path','')))
            if not p.is_file():errors.append({'sample_id':sid,'reason':'image_missing'})
            else:
                if p not in checked_images:checked_images[p]=sha256(p)
                expected=row.get('metadata',{}).get('image_sha256')
                if expected and checked_images[p]!=expected:errors.append({'sample_id':sid,'reason':'image_checksum_mismatch'})
        if row['task_type']=='context_qa' and row['supervision'].get('rationale'):
            contexts=row['input'].get('context',[])
            if row['supervision']['rationale'] in contexts:errors.append({'sample_id':sid,'reason':'rationale_leaked'})
    errors.extend({'sample_id':sid,'reason':'duplicate_sample_id','count':n} for sid,n in ids.items() if n>1)
    cross_groups=[{'source_id':s,'group_type':k,'group_digest':digest(v),'splits':sorted(sp)} for (s,k,v),sp in groups.items() if len(sp)>1]
    errors.extend({'reason':'known_group_cross_split',**g} for g in cross_groups)
    duplicates=[];conflicts=[]
    for key,rows in inputs.items():
        if len(rows)<2:continue
        entry={'input_digest':key,'sample_ids':[r['sample_id'] for r in rows],'splits':sorted({r['split'] for r in rows})}
        duplicates.append(entry)
        if len(entry['splits'])>1:errors.append({'reason':'exact_input_cross_split',**entry})
        if len({normalized(r['supervision']['answer']) for r in rows})>1:
            conflicts.append(entry);errors.append({'reason':'label_conflict',**entry})
    distributions={}
    for source in sorted({r['source_id'] for r in records}):
        selected=[r for r in records if r['source_id']==source]
        lengths=sorted(len(r['input']['question'])+len(str(r['input'].get('context') or '')) for r in selected)
        distributions[source]={'split':dict(Counter(r['split'] for r in selected)),'task':dict(Counter(r['task_type'] for r in selected)),'language':dict(Counter(r['language'] for r in selected)),'label':dict(Counter(r['supervision']['answer'] for r in selected)) if source in {'pubmedqa_labeled','kormedmcqa'} else None,'input_char_length':{'min':min(lengths,default=0),'median':lengths[len(lengths)//2] if lengths else 0,'p95':lengths[int(len(lengths)*.95)] if lengths else 0,'max':max(lengths,default=0)},'derivation':dict(Counter(r['derivation'] for r in selected))}
    result={'schema_version':'medvlm.audit.v1','status':'SUCCEEDED' if not errors else 'FAILED_REPRODUCIBLE','evidence_level':'CPU_TESTED','record_count':len(records),'files':files,'critical_error_count':len(errors),'critical_errors':errors,'exact_duplicate_input_groups':duplicates,'label_conflict_groups':conflicts,'group_cross_split':cross_groups,'distributions':distributions,'text_near_duplicate_candidates':_near_candidates(records),'patient_group_verified':False,'limitations':['Patient-level independence is not established when patient IDs are absent.','Text near-duplicate screening is lexical candidate generation, not verified semantic equivalence.','Tokenizer-specific truncation is checked by the training renderer; not estimated from character length.']}
    atomic_json(root/'manifests'/'audit.json',result);return result


def freeze(root):
    root=Path(root);result=audit(root)
    if result['critical_error_count']:raise ValueError(f'Freeze refused: {result["critical_error_count"]} critical audit errors')
    splits={}
    for p in source_files(root):
        rows=read_jsonl(p)
        splits[f'{p.parent.name}/{p.stem}']={'path':str(p),'sha256':sha256(p),'count':len(rows),'ordered_sample_ids':[r['sample_id'] for r in rows], 'input_content_digest':digest([input_digest(r) for r in rows]),'original_content_digest':digest([r['provenance'].get('local_asset_sha256') for r in rows]),'group_relation_digest':digest([(r['sample_id'],r['group_ids'],r['split']) for r in rows]),'image_content_digest':digest([r.get('metadata',{}).get('image_sha256') for r in rows if r['input'].get('images')])}
    for split in ('train','validation','test'):
        rows=[]
        for p in source_files(root):
            if p.stem!=split:continue
            rows.extend(r for r in read_jsonl(p) if r['task_type']!='medical_vqa' and r['source_id']!='kormedmcqa')
        rows.sort(key=lambda r:r['sample_id'])
        path=root/'processed'/'text'/f'{split}.jsonl';write_jsonl(path,rows)
        splits[f'text/{split}']={'path':str(path),'sha256':sha256(path),'count':len(rows),'ordered_sample_ids':[r['sample_id'] for r in rows],'input_content_digest':digest([input_digest(r) for r in rows]),'note':'Derived union, excluded from source audit to avoid double counting; Korean evaluation source is not SFT.'}
    # Content-addressed copies preserve the bytes referenced by older manifests
    # even if a later processing version updates the convenient working paths.
    for entry in splits.values():
        snapshot=root/'frozen'/entry['sha256']/Path(entry['path']).name
        if not snapshot.exists():
            snapshot.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(entry['path'],snapshot)
        if sha256(snapshot)!=entry['sha256']:raise ValueError('Frozen snapshot checksum mismatch')
        entry['snapshot_path']=str(snapshot)
    code_files=sorted(Path(__file__).parent.glob('*.py'))
    included_sources={p.parent.name for p in source_files(root)}
    sources={p.stem:json.loads(p.read_text()).get('resolved_revision') for p in (root/'manifests'/'sources').glob('*.json') if p.stem in included_sources and json.loads(p.read_text()).get('resolved_revision')}
    identity={'schema_version':'medvlm.freeze.v1','sources':sources,'seed':20261023,'splits':splits,'transformation_code_sha256':digest({p.name:sha256(p) for p in code_files}),'selection_policy':'source official splits, immutable test; MedQuAD document/duplicate components; whole-group caps; text union omits eval-only KorMedMCQA','audit_sha256':sha256(root/'manifests'/'audit.json')}
    frozen={**identity,'freeze_id':digest(identity),'status':'SUCCEEDED','evidence_level':'CPU_TESTED'}
    immutable=root/'manifests'/f'freeze-{frozen["freeze_id"]}.json'
    if immutable.exists() and json.loads(immutable.read_text())!=frozen:raise ValueError('Content-addressed manifest collision')
    atomic_json(immutable,frozen);atomic_json(root/'manifests'/'current_freeze.json',frozen)
    summary={'schema_version':'medvlm.data.public-summary.v1','status':'SUCCEEDED','evidence_level':'CPU_TESTED','freeze_id':frozen['freeze_id'],'manifest_path':str(immutable),'sources':sources,'splits':{k:{a:b for a,b in v.items() if a not in {'ordered_sample_ids'}} for k,v in splits.items()},'critical_error_count':result['critical_error_count'],'record_count':result['record_count'],'patient_group_verified':False,'distributions':result['distributions'],'near_duplicate_candidate_count':len(result['text_near_duplicate_candidates']['candidates']),'raw_data_redistributed':False}
    atomic_json(Path.cwd()/'reports'/'public_summary'/'data_freeze.json',public_paths(summary,root))
    return summary
