"""Non-destructive reproduction of the author's seed-0 PQA-L fold algorithm."""
import json
import math
import random
from pathlib import Path
from collections import Counter
from .io import sha256
from .schema import sample, normalized


def official_folds(dataset, fold=0):
    if not 0 <= fold < 10: raise ValueError('fold must be 0..9')
    rng=random.Random(0)
    def split(data,n):
        buckets={k:[] for k in ('yes','no','maybe')}
        for pmid,row in data.items():buckets[row['final_decision']].append(pmid)
        slices={}
        for label,ids in buckets.items():
            rng.shuffle(ids);width=math.ceil(len(ids)/n)
            slices[label]=[ids[i*width:(i+1)*width] if i<n-1 else ids[i*width:] for i in range(n)]
        result=[{pid:data[pid] for label in buckets for pid in slices[label][i]} for i in range(n)]
        if len(result[-1]) != len(result[0]):
            for i in range(n-1):
                pick=rng.choice(list(result[i]));result[-1][pick]=result[i].pop(pick)
        return result
    pool,test=split(dataset,2)
    folds=split(pool,10)
    train={pid:r for i,part in enumerate(folds) if i!=fold for pid,r in part.items()}
    return {'train':train,'validation':folds[fold],'test':test}


def build_pubmedqa(repository:Path, revision:str, fetched_at:str, fold=0):
    path=repository/'data'/'ori_pqal.json'
    dataset=json.loads(path.read_text());parts=official_folds(dataset,fold)
    output=[];exclusions=[];removed=Counter()
    file_hash=sha256(path)
    for split,rows in parts.items():
        for pmid,row in rows.items():
            question=row['QUESTION'];long=row.get('LONG_ANSWER','')
            labels=row.get('LABELS',[]);contexts=[]
            for i,c in enumerate(row['CONTEXTS']):
                label=labels[i] if i<len(labels) else ''
                if 'CONCLUSION' in label.upper() or normalized(c) in {normalized(question),normalized(long)}:
                    removed['conclusion_or_question_context_segments']+=1;continue
                contexts.append(c)
            if not contexts:
                exclusions.append({'original_id':pmid,'reason':'no_context_without_conclusion'});continue
            rec=sample('pubmedqa_labeled',revision,pmid,'context_qa',question,row['final_decision'],split,
                context=contexts,source_url=f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/',document=f'PMID:{pmid}',original_split=f'pqal_fold{fold}/'+('dev_set' if split=='validation' else 'train_set' if split=='train' else 'test_set'),
                metadata={'pmid':pmid,'year':row.get('YEAR'),'meshes':row.get('MESHES',[]),'context_labels':labels,'fold':fold,'official_split_seed':0},
                license_ref=f'https://github.com/pubmedqa/pubmedqa/blob/{revision}/LICENSE',fetched_at=fetched_at,raw_sha256=file_hash,annotation='expert_annotated_PQA-L')
            # LONG_ANSWER remains out of model input and is not needed for yes/no/maybe supervision.
            output.append(rec)
    return output,{'counts':{'raw_qa':len(dataset),'used':len(output),**dict(removed)},'original_split_counts':{k:len(v) for k,v in parts.items()},'exclusions':exclusions,'selection_policy':{'official_seed':0,'fold':fold,'algorithm_url':f'https://github.com/pubmedqa/pubmedqa/blob/{revision}/preprocess/split_dataset.py','algorithm_sha256':sha256(repository/'preprocess'/'split_dataset.py'),'non_destructive_reimplementation':True},'license_scope':'PQA-L distributed by author in MIT repository. Underlying article abstract copyright remains with original holders; retained PMID provenance, local noncommercial research use, no raw redistribution or public-domain claim.'}
