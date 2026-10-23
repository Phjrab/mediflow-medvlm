"""Read only released public answers; never reconstruct copyright removals."""
from collections import Counter, defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET
from .io import digest, sha256
from .schema import sample, input_digest, normalized

EXCLUDED = {'10_MPlus_ADAM_QA', '11_MPlusDrugs_QA', '12_MPlusHerbsSupplements_QA'}


class UnionFind:
    def __init__(self): self.parent = {}
    def find(self, x):
        self.parent.setdefault(x, x)
        if self.parent[x] != x: self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, x, y):
        a,b = self.find(x), self.find(y)
        self.parent[max(a,b)] = min(a,b)


def build_medquad(repository: Path, revision: str, fetched_at: str, seed=20261023, train_cap=8000):
    records, exclusions = [], []
    counts, subset_counts = Counter(), Counter()
    for path in sorted(repository.glob('*/*.xml')):
        if not path.parent.name[0].isdigit(): continue
        root = ET.parse(path).getroot()
        subset = path.parent.name
        doc_id = f'{subset}/{root.get("id", path.stem)}'
        file_hash = sha256(path)
        for i, pair in enumerate(root.findall('.//QAPair')):
            counts['raw_qa'] += 1
            subset_counts[subset] += 1
            q,a = pair.find('Question'), pair.find('Answer')
            question = ''.join(q.itertext()).strip() if q is not None else ''
            answer = ''.join(a.itertext()).strip() if a is not None else ''
            # The release reuses some document/qid pairs across XML fragments.
            # Keep their source document group but disambiguate the physical QA.
            oid = f'{subset}/{path.stem}/{q.get("qid", str(i)) if q is not None else i}/{i}'
            if not answer: counts['observed_empty_answers'] += 1
            reason = 'copyright_removed_subset' if subset in EXCLUDED else 'empty_answer' if not answer else 'empty_question' if not question else None
            if reason:
                exclusions.append({'original_id':oid,'reason':reason}); counts[reason] += 1; continue
            records.append(sample('medquad', revision, oid, 'medical_qa', question, answer, 'train',
                source_url=root.get('url'), document=doc_id, metadata={'subset':subset,'source_document_id':root.get('id'),'source_xml_file':str(path.relative_to(repository)),'q_type':q.get('qtype'), 'source_question_id':q.get('qid')},
                license_ref=f'https://github.com/abachaa/MedQuAD/blob/{revision}/LICENSE.txt', fetched_at=fetched_at,raw_sha256=file_hash,annotation='NIH_website_source_answer'))
    counts['public_nonempty_before_dedup'] = len(records)
    uf, keys, source_documents = UnionFind(), defaultdict(list), {}
    for row in records:
        keys[input_digest(row)].append(row)
        uf.find(row['group_ids']['document'])
        url = row['provenance'].get('source_url')
        if url:
            if url in source_documents:
                uf.union(source_documents[url], row['group_ids']['document'])
            source_documents[url] = row['group_ids']['document']
    for key, rows in keys.items():
        for row in rows[1:]: uf.union(rows[0]['group_ids']['document'],row['group_ids']['document'])
    kept=[]
    for key, rows in sorted(keys.items()):
        if len({normalized(r['supervision']['answer']) for r in rows}) > 1:
            exclusions.extend({'sample_id':r['sample_id'],'reason':'label_conflict'} for r in rows)
            counts['label_conflict'] += len(rows); continue
        kept.append(sorted(rows,key=lambda r:r['sample_id'])[0])
        exclusions.extend({'sample_id':r['sample_id'],'reason':'exact_duplicate_input'} for r in sorted(rows,key=lambda r:r['sample_id'])[1:])
        counts['exact_duplicate_input'] += len(rows)-1
    groups=defaultdict(list)
    for row in kept:
        component=uf.find(row['group_ids']['document'])
        row['group_ids']['duplicate_cluster'] = digest(component)
        groups[component].append(row)
    output=[]
    train_used=0
    for component,rows in sorted(groups.items(),key=lambda kv:digest([seed,kv[0]])):
        fraction=int(digest([seed,'split',component])[:16],16)/2**64
        split='test' if fraction < .1 else 'validation' if fraction < .2 else 'train'
        if split=='train' and train_used+len(rows)>train_cap:
            exclusions.extend({'sample_id':r['sample_id'],'reason':'train_cap_whole_group'} for r in rows)
            counts['train_cap_whole_group']+=len(rows);continue
        if split=='train':train_used+=len(rows)
        for row in rows:row['split']=split
        output.extend(rows)
    return output, {'counts':dict(counts),'raw_subsets':dict(subset_counts),'exclusions':exclusions,'selection_policy':{'seed':seed,'ratios':[.8,.1,.1],'grouping':'document plus exact-input connected components','train_cap':train_cap,'cap_unit':'whole_connected_component','stratification':'source distributions audited; hash split without forced row balancing'},'license_scope':'Released public answers under author CC-BY-4.0; removed subsets never reconstructed; individual source URLs preserved.'}
