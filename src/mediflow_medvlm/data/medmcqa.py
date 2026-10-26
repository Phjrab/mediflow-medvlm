"""Prepare a bounded optional MedMCQA cohort from the official author ZIP.

This module never changes a v1 freeze or enrolls data in Korean SFT. Official
test rows are retained only in raw ZIP and protect against train overlap.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import zipfile

from .io import atomic_json, digest, disk_guard, now, sha256, write_jsonl
from .schema import normalized, sample, validate_record

REPO_REVISION = 'c59ef14ca1990266c4107c7864b45a20fd93e5e0'
ARCHIVE_SHA256 = '16c1fbc6f47d548d2af7837b18e893aa45f45c0be9bda0a9adfff3c625bf9262'
SOURCE_URL = 'https://drive.google.com/uc?export=download&id=15VkJdq5eyWIkfb_aoD3oS8i4tScbHYky'
LICENSE_URL = 'https://github.com/medmcqa/medmcqa/issues/2#issuecomment-1329300257'
SEED = 20261023
MEMBERS = ('train.json', 'dev.json', 'test.json')


def iter_member(archive, member):
    with archive.open(member) as stream:
        for line_number, line in enumerate(stream, 1):
            if len(line)>2*1024**2:
                raise ValueError('source record exceeds 2MiB bound')
            if line.strip():
                value=json.loads(line)
                if not isinstance(value,dict):raise ValueError('source record must be object')
                yield line_number,value


def option_key(raw):
    """Order-independent exact lexical key, preserving numbers/units/negation."""
    question=raw.get('question');options=[raw.get('op'+letter) for letter in 'abcd']
    if not isinstance(question,str) or not question.strip():return None
    if not all(isinstance(v,str) and v.strip() for v in options):return None
    return digest([normalized(question).casefold(),sorted(normalized(v).casefold() for v in options)])


def valid_label(raw):
    if raw.get('choice_type')!='single':return None,'not_single_choice'
    if option_key(raw) is None:return None,'missing_question_or_option'
    cop=raw.get('cop')
    if type(cop) is not int or not 1<=cop<=4:return None,'invalid_one_based_cop'
    return 'ABCD'[cop-1],None


def prepare(data_root, archive_path, *, train_cap=20000, expected_archive_sha256=ARCHIVE_SHA256):
    """Full raw audit, deterministic capped optional train, official dev only."""
    if type(train_cap) is not int or not 1<=train_cap<=20000:raise ValueError('train cap must be 1..20000')
    root=Path(data_root).resolve();archive_path=Path(archive_path).resolve()
    archive_hash=sha256(archive_path)
    if archive_hash!=expected_archive_sha256:raise ValueError('official archive identity changed')
    destination=root/'v2'/'optional'/'medmcqa'/archive_hash
    manifest_path=destination/'manifest.json'
    if manifest_path.exists():
        previous=json.loads(manifest_path.read_text())
        if previous['train_cap']!=train_cap:raise ValueError('cannot overwrite another capped cohort')
        for split,entry in previous['canonical_files'].items():
            if sha256(destination/'processed'/f'{split}.jsonl')!=entry['sha256']:raise ValueError('prepared cohort modified')
        return previous
    # Guard global raw/images cap and free-space reserve before any transformed output.
    disk_guard(root/'raw'/'medmcqa_v2',512*1024**2)
    revision='official-zip-'+archive_hash
    raw_counts=Counter();cop_counts={};choice_counts={};heldout=set();labels=defaultdict(set)
    member_hashes={};raw_id_counts={};quarantine=defaultdict(Counter)
    with zipfile.ZipFile(archive_path) as archive:
        names=archive.namelist()
        if len(names)!=len(set(names)) or not set(MEMBERS)<=set(names):raise ValueError('missing or duplicate official archive members')
        if any(name not in MEMBERS and not name.startswith('__MACOSX/') for name in names):raise ValueError('unexpected official archive member')
        if sum(x.file_size for x in archive.infolist())>256*1024**2:raise ValueError('expanded archive exceeds 256MiB')
        bad=archive.testzip()
        if bad:raise ValueError('archive CRC mismatch')
        # No extraction: bounded streams avoid archive path/symlink traversal entirely.
        for member in MEMBERS:
            with archive.open(member) as stream:
                h=hashlib.sha256()
                for chunk in iter(lambda:stream.read(1024**2),b''):h.update(chunk)
                member_hashes[member]=h.hexdigest()
            cop=Counter();choice=Counter();ids=Counter()
            for _,raw in iter_member(archive,member):
                raw_counts[member]+=1;cop[str(raw.get('cop'))]+=1;choice[str(raw.get('choice_type'))]+=1
                ids[str(raw.get('id'))]+=1;k=option_key(raw)
                if member!='train.json' and k:heldout.add(k)
                label,error=valid_label(raw)
                if member!='test.json' and not error:
                    labels[k].add(normalized(raw['op'+label.lower()]).casefold())
            cop_counts[member]=dict(cop);choice_counts[member]=dict(choice);raw_id_counts[member]=ids
        conflicts={k for k,v in labels.items() if len(v)>1}
        candidates={};seen={};eligible_counts=Counter()
        for member in ('train.json','dev.json'):
            split='train' if member=='train.json' else 'validation';candidates[split]=[];seen[split]=set()
            for line,raw in iter_member(archive,member):
                label,error=valid_label(raw);key=option_key(raw)
                if not error and (raw.get('id') is None or raw_id_counts[member][str(raw['id'])]>1):error='missing_or_duplicate_source_id'
                if not error and key in conflicts:error='conflicting_answer_text'
                if not error and split=='train' and key in heldout:error='official_heldout_input_overlap'
                if not error and key in seen[split]:error='within_split_exact_input_duplicate'
                if error:quarantine[split][error]+=1;continue
                seen[split].add(key);eligible_counts[split]+=1
                # Keep only IDs and lexical keys in the first selection pass.
                candidates[split].append((digest([SEED,str(raw['id'])]),line,key))
        selected={split:{line for _,line,_ in sorted(items)[:train_cap if split=='train' else len(items)]} for split,items in candidates.items()}
        quarantine['train']['deterministic_optional_train_cap']=eligible_counts['train']-len(selected['train'])
        destination.mkdir(parents=True,exist_ok=True)
        records={};subject_counts={}
        fetched=now()
        for member in ('train.json','dev.json'):
            split='train' if member=='train.json' else 'validation';records[split]=[]
            for line,raw in iter_member(archive,member):
                if line not in selected[split]:continue
                label,_=valid_label(raw);key=option_key(raw)
                options={letter:raw['op'+letter.lower()] for letter in 'ABCD'}
                row=sample('medmcqa',revision,split+':'+str(raw['id']),'medical_mcq',raw['question'],options[label],split,
                           original_split='train' if split=='train' else 'dev',options=options,option_id=label,
                           source_url=SOURCE_URL,document='medmcqa:'+key,license_ref=LICENSE_URL,fetched_at=fetched,
                           raw_sha256=member_hashes[member],metadata={'source_record_id':raw['id'],'original_line':line,
                           'cop_base':1,'source_cop':raw['cop'],'subject_name':raw.get('subject_name'),'topic_name':raw.get('topic_name'),
                           'choice_type':'single','declared_data_license':'MIT','optional_only':True,'core_ko_mix_included':False})
                row['supervision']['rationale']=raw.get('exp') or None
                row['group_ids']['duplicate_cluster']='medmcqa:'+key
                row['provenance']['transformation_version']='medvlm.medmcqa.optional.v2.1'
                if validate_record(row):raise ValueError('canonical validation failed')
                records[split].append(row)
            records[split].sort(key=lambda x:digest([SEED,x['metadata']['source_record_id']]))
            subject_counts[split]=dict(Counter(str(x['metadata']['subject_name']) for x in records[split]))
        require_disjoint={r['group_ids']['duplicate_cluster'] for r in records['train']} & {r['group_ids']['duplicate_cluster'] for r in records['validation']}
        if require_disjoint:raise ValueError('optional split leakage')
        files={}
        for split,rows in {**records,'test':[]}.items():
            path=destination/'processed'/f'{split}.jsonl';write_jsonl(path,rows)
            files[split]={'count':len(rows),'sha256':sha256(path)}
    manifest={'schema_version':'medvlm.medmcqa.optional.v2','status':'SUCCEEDED','evidence_level':'CPU_TESTED',
              'source_id':'medmcqa','repo_revision':REPO_REVISION,'source_revision':revision,'archive_sha256':archive_hash,
              'archive_bytes':archive_path.stat().st_size,'source_url':SOURCE_URL,'retrieved_via':'official_author_README_Google_Drive',
              'license':'MIT','license_evidence':[LICENSE_URL,f'https://github.com/medmcqa/medmcqa/blob/{REPO_REVISION}/LICENSE.md'],
              'license_scope':'Official team explicitly confirmed MIT for current dataset; underlying exam/book rights are not individually enumerated.',
              'raw_counts':dict(raw_counts),'raw_member_sha256':member_hashes,'source_cop_counts':cop_counts,'choice_type_counts':choice_counts,
              'cop_base_evidence':f'https://github.com/medmcqa/medmcqa/blob/{REPO_REVISION}/dataset.py',
              'cop_base':1,'canonical_files':files,'eligible_after_cleaning':dict(eligible_counts),
              'excluded_counts':{split:dict(c) for split,c in quarantine.items()},'conflict_input_group_count':len(conflicts),
              'subject_counts':subject_counts,'train_cap':train_cap,'seed':SEED,'test_scoring':'DISALLOWED_OFFICIAL_LABELS_UNAVAILABLE',
              'test_raw_preserved':True,'core_ko_mix_included':False,'training_executed':False,'raw_redistribution':False,
              'near_duplicate_and_translation_screening':'NOT_PERFORMED; optional cohort not approved for core training mixture',
              'input_normalization':'NFC/whitespace/casefold; option-order independent; retain numbers, units, punctuation, negation',
              'first_prepared_at':fetched,'critical_error_count':0}
    atomic_json(manifest_path,manifest)
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data-root',type=Path,required=True);parser.add_argument('--archive',type=Path,required=True);parser.add_argument('--train-cap',type=int,default=20000)
    args=parser.parse_args();result=prepare(args.data_root,args.archive,train_cap=args.train_cap)
    print(json.dumps({'status':result['status'],'raw_counts':result['raw_counts'],'canonical_files':result['canonical_files'],'excluded_counts':result['excluded_counts']}))
