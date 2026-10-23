"""Tiny local Forge JSONL example with a lossless provenance sidecar.

This exports eligible frozen training rows only. It never uploads to Forge,
changes benchmark splits, or puts original medical text into the repository.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from itertools import zip_longest
import json
import os
from pathlib import Path

from .io import digest, read_jsonl, sha256
from ..training.rendering import user_text, assistant_target

MAX_ROWS = 1000
MAX_BYTES = 128 * 1024
MAX_FIELD_CHARS = 16384
MAX_RECORD_CHARS = 32768
FORGE_CONTRACT = 'https://github.com/Phjrab/forge-finetune-dashboard/blob/d9a4656a5bee38cc9903e8caf5c4b680709423d2/workspace.py'


def _bytes(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8') + b'\n'


def _immutable_write(path,content):
    if path.exists():
        if path.read_bytes()!=content:
            raise ValueError('Existing export differs; immutable export not overwritten')
        return
    temporary=path.with_suffix(path.suffix+'.part');temporary.write_bytes(content);os.replace(temporary,path)


def export_forge(data_root,freeze_manifest,rows=4):
    if not 1<=rows<=MAX_ROWS:raise ValueError('Forge export requires 1..1000 rows')
    root=Path(data_root).resolve();freeze_path=Path(freeze_manifest).resolve()
    frozen=json.loads(freeze_path.read_text())
    identity={k:v for k,v in frozen.items() if k not in {'freeze_id','status','evidence_level'}}
    if digest(identity)!=frozen['freeze_id']:raise ValueError('Frozen manifest digest mismatch')
    candidates=defaultdict(list)
    for key,entry in frozen['splits'].items():
        source,split=key.split('/',1)
        if split!='train' or source in {'text','vlm'}:continue
        path=Path(entry.get('snapshot_path',entry['path']))
        if sha256(path)!=entry['sha256']:raise ValueError('Frozen training snapshot checksum mismatch')
        for row in read_jsonl(path):
            if row['split']!='train':raise ValueError('Non-training row in training snapshot')
            if row['input'].get('images') or not row.get('restrictions',{}).get('sft_allowed',False):continue
            candidates[source].append(row)
    for items in candidates.values():items.sort(key=lambda x:x['sample_id'])
    ordered=[r for batch in zip_longest(*(candidates[s] for s in sorted(candidates))) for r in batch if r is not None]
    output=[];sidecar=[];excluded=Counter();size=0
    for row in ordered:
        if len(output)==rows:break
        user=user_text(row);answer=assistant_target(row)
        if any(len(text)>MAX_FIELD_CHARS for text in (user,answer)) or len(user)+len(answer)>MAX_RECORD_CHARS:
            excluded['forge_field_or_row_character_limit']+=1;continue
        if any((ord(c)<32 and c not in '\n\r\t') or ord(c)==127 for text in (user,answer) for c in text):
            excluded['binary_control_character']+=1;continue
        value={'messages':[{'role':'user','content':user},{'role':'assistant','content':answer}]}
        encoded=_bytes(value)
        if size+len(encoded)>MAX_BYTES:
            excluded['forge_utf8_byte_limit']+=1;continue
        size+=len(encoded);output.append(encoded)
        sidecar.append({'line':len(output),'export_row_sha256':digest(value),**{k:row[k] for k in ('sample_id','source_id','source_revision','original_id','task_type','language','original_split','split','group_ids','grouping_quality','provenance','derivation','review_status','restrictions')},'metadata':row.get('metadata',{})})
    if len(output)!=rows:raise ValueError('Insufficient eligible frozen training rows within Forge limits')
    directory=root/'exports'/'forge'/f'{frozen["freeze_id"][:12]}-{rows}-text'
    directory.mkdir(parents=True,exist_ok=True);os.chmod(directory,0o700)
    if not directory.resolve().is_relative_to(root/'exports'):raise ValueError('Export path escapes data-root/exports')
    data=directory/'train.jsonl';mapping=directory/'provenance.json'
    _immutable_write(data,b''.join(output));_immutable_write(mapping,_bytes({'schema_version':'medvlm.forge-provenance.v1','freeze_id':frozen['freeze_id'],'records':sidecar}))
    result={'schema_version':'medvlm.forge-example.v1','status':'SUCCEEDED','evidence_level':'CPU_TESTED','rows':len(output),'bytes':size,'row_limit':MAX_ROWS,'byte_limit':MAX_BYTES,'kind':'LLM','synthetic':False,'purpose':'small compatibility example only; no Forge import, upload or training performed','freeze_id':frozen['freeze_id'],'freeze_manifest':str(freeze_path),'freeze_manifest_sha256':sha256(freeze_path),'data_path':str(data),'data_sha256':sha256(data),'sidecar_path':str(mapping),'sidecar_sha256':sha256(mapping),'source_counts':dict(Counter(r['source_id'] for r in sidecar)),'excluded_before_selection':dict(excluded),'forge_contract':FORGE_CONTRACT,'canonical_provenance_preserved_in_sidecar':True}
    _immutable_write(directory/'manifest.json',_bytes(result))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--freeze-manifest',type=Path,required=True)
    parser.add_argument('--rows',type=int,default=4)
    args=parser.parse_args();print(json.dumps(export_forge(args.data_root,args.freeze_manifest,args.rows),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
