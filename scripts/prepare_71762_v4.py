#!/usr/bin/env python3
"""Local CPU screening and a separate finite 71762 experimental corpus."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import fcntl
import json
from pathlib import Path

from mediflow_medvlm.data.aihub import freeze_corpus, leakage_audit, semantic_audit, token_audit
from mediflow_medvlm.data.aihub71762_experiment import build_pool, SEED
from mediflow_medvlm.data.io import read_jsonl, sha256, digest
from mediflow_medvlm.evaluation.runner import load_protocol
from mediflow_medvlm.runtime import atomic_json


def read(path):
    return json.loads(Path(path).read_text())


def balanced(rows, count):
    groups = defaultdict(list)
    for row in rows:
        groups[row['metadata']['domain']].append(row)
    groups = {key: deque(sorted(items, key=lambda r: digest([SEED, r['sample_id']])))
              for key, items in sorted(groups.items())}
    result = []
    while len(result) < count:
        progressed = False
        for items in groups.values():
            if items and len(result) < count:
                result.append(items.popleft())
                progressed = True
        if not progressed:
            break
    return result


def blocked_ids(report):
    return ({r['sample_id'] for r in report.get('confirmed_overlaps', [])} |
            {r['sample_id'] for r in report.get('near_candidates', [])} |
            {r['sample_id'] for r in report.get('candidate_pairs', [])})


def prepare(root):
    out = root / 'v4'
    audit = root / 'v3/prepared/aihub_71762/ef07a830a830126d8f00669db250381f5355bfa3510c0e5ac9c9f2e39f8f00eb/audit.json'
    pool = build_pool(audit, out / 'authorization.json', out / 'pool')
    identity = {'pool_receipt_sha256': sha256(out / 'pool/pool.json'),
                'preparation_code_sha256': sha256(__file__), 'seed': SEED,
                'train_target': 4000, 'validation_target_before_semantic_screen': 128, 'test_target': 256,
                'length_limit': 2048, 'near_jaccard': .90, 'semantic_cosine': .92, 'semantic_top_k': 5}
    if (out / 'corpus.json').exists():
        previous = read(out / 'corpus.json')
        if previous['identity'] != identity:
            raise ValueError('COMPLETED_CORPUS_IDENTITY_CHANGED')
        freeze = read(previous['freeze_manifest'])
        if sha256(previous['freeze_manifest']) != previous['freeze_manifest_sha256']:
            raise ValueError('FROZEN_MANIFEST_CHANGED')
        for entry in freeze['splits'].values():
            if sha256(entry['snapshot_path']) != entry['sha256']:
                raise ValueError('FROZEN_CORPUS_CHANGED')
        return previous
    def progress(stage, **values):
        atomic_json(out / 'preparation_state.json', {'stage': stage, **values})
        print(json.dumps({'stage': stage, **values}), flush=True)
    rows = {split: read_jsonl(entry['path']) for split, entry in pool['splits'].items()}
    configs = {model: read(root / f'runs/{model}-v1-20261023/config.json')
               for model in ('qwen3_17b', 'qwen3_4b')}
    token_path = out / 'token_audit.json'
    if token_path.exists():
        token = read(token_path)
    else:
        progress('native_tokenization', rows=sum(map(len, rows.values())))
        token = token_audit([r for split in rows.values() for r in split], configs, max_length=2048)
        atomic_json(token_path, token)
    eligible = set(token['common_eligible_sample_ids'])
    rows = {split: [r for r in records if r['sample_id'] in eligible] for split, records in rows.items()}
    old_freeze = read(root / 'manifests/freeze-1bf58dbc8baca7a3f5865accf5a86e9fba0b84f8abe3c986ed34b67d6c3d87c5.json')
    protected = {}
    protected_files = {}
    for split in ('validation', 'test'):
        entry = old_freeze['splits']['kormedmcqa/' + split]
        path = entry.get('snapshot_path', entry['path'])
        if sha256(path) != entry['sha256']:
            raise ValueError('PROTECTED_BENCHMARK_CHANGED')
        protected_files[path] = entry['sha256']
        protected.update({r['sample_id']: r for r in read_jsonl(path)})
    if len(protected) != 4068:
        raise ValueError('FULL_KORMED_4068_REQUIRED')
    old_protocol, common = load_protocol(root / 'v2/text/text-test.json')
    protected.update({r['sample_id']: r for r in common})
    protected_files.update(old_protocol['files'])
    external = list(protected.values())
    progress('lexical_external_screen', protected=len(external), candidates=sum(map(len, rows.values())))
    lexical = {}
    for split in rows:
        rows[split], lexical[split] = leakage_audit(rows[split], external)
    test = balanced(rows['test'], 256)
    valid_pool, lexical['validation_vs_test'] = leakage_audit(rows['validation'], test)
    valid = balanced(valid_pool, 128)
    train, lexical['train_vs_derived_heldout'] = leakage_audit(rows['train'], valid + test)
    atomic_json(out / 'lexical_audit.json', lexical)
    if len(test) != 256 or len(valid) < 48:
        raise ValueError('INSUFFICIENT_LENGTH_AND_LEXICAL_ELIGIBLE_HELDOUT')
    # Token eligibility and lexical screens do not inspect model predictions.
    semantic_candidates = train + valid
    semantic_protected = external + test
    progress('semantic_screen', candidates=len(semantic_candidates), protected=len(semantic_protected))
    sem = semantic_audit(semantic_candidates, semantic_protected,
                         root / 'v2/manifests/multilingual_embedding_model.json', out / 'semantic')
    blocked = blocked_ids(sem)
    # The same already measured vectors additionally screen train vs checkpoint
    # validation. No second model pass or different similarity threshold.
    import numpy as np
    policy = read(out / 'semantic/policy.json')
    vectors = np.load(out / 'semantic/embeddings.npy', allow_pickle=False)
    positions = {sid: i for i, sid in enumerate(policy['sample_ids'])}
    right = vectors[[positions[r['sample_id']] for r in valid]]
    internal = []
    for start in range(0, len(train), 256):
        left = train[start:start+256]
        scores = vectors[[positions[r['sample_id']] for r in left]] @ right.T
        for row, values in zip(left, scores):
            for index in np.argsort(-values, kind='stable')[:5]:
                if float(values[index]) >= .92:
                    blocked.add(row['sample_id'])
                    internal.append({'sample_id': row['sample_id'], 'protected_sample_id': valid[index]['sample_id'],
                                     'cosine': float(values[index]), 'confirmed_duplicate': False})
    atomic_json(out / 'internal_semantic_audit.json', {'candidate_pairs': internal, 'threshold': .92,
                'top_k': 5, 'candidate_rows': len({r['sample_id'] for r in internal}),
                'vectors_sha256': sha256(out / 'semantic/embeddings.npy'),
                'semantic_equivalence_verified': False, 'translation_equivalence_verified': False})
    train = balanced([r for r in train if r['sample_id'] not in blocked], 4000)
    valid = [r for r in valid if r['sample_id'] not in blocked]
    if len(train) != 4000 or len(valid) < 48:
        raise ValueError('INSUFFICIENT_SCREENED_FINITE_CORPUS')
    splits = {'train': train, 'validation': valid, 'test': test}
    for records in splits.values():
        for row in records:
            row['group_ids']['answer_text'] = row['metadata']['answer_component']['text_sha256']
    provenance = {**identity, 'authorization_sha256': sha256(out / 'authorization.json'),
                  'lexical_audit_sha256': sha256(out / 'lexical_audit.json'),
                  'token_audit_sha256': sha256(token_path), 'semantic_audit_sha256': sha256(out / 'semantic/audit.json'),
                  'internal_semantic_sha256': sha256(out / 'internal_semantic_audit.json'),
                  'protected_files': protected_files, 'true_qa_correspondence_verified': False,
                  'augmentation_parent_available': False, 'test_used_for_training_or_checkpoint_selection': False}
    frozen = freeze_corpus(out, 'aihub71762_category_random_v4', splits, provenance=provenance)
    measurements = {}
    for key, model in token['models'].items():
        entries = token['encodings'][model['encoding_identity']]['samples']
        measurements[key] = {split: {'rows': len(records),
                    'supervised_tokens': sum(entries[r['sample_id']]['supervised_tokens'] for r in records),
                    'sequence_tokens': sum(entries[r['sample_id']]['sequence_tokens'] for r in records)}
                    for split, records in splits.items()}
    result = {'schema': 'medvlm.aihub71762.corpus.v4', 'status': 'SUCCEEDED', 'identity': identity,
              'freeze_manifest': frozen['manifest_path'], 'freeze_manifest_sha256': frozen['manifest_sha256'],
              'splits': frozen['splits'], 'token_counts_by_model': measurements,
              'semantic_quarantined_rows_train_and_validation': len(blocked),
              'source_counts': {split: dict(Counter(r['source_id'] for r in records)) for split, records in splits.items()},
              'domain_counts': {split: dict(Counter(r['metadata']['domain'] for r in records)) for split, records in splits.items()},
              'protected_unique_rows': len(external), 'protected_kormed_rows': 4068,
              'true_qa_correspondence_verified': False, 'augmentation_lineage_verified': False,
              'clinical_validation': 'not_performed', 'official_preprocessing_code_reviewed': False,
              'all_source_training': False, 'heldout_is_official_test': False,
              'leakage_limitations': ['Missing augmentation lineage prevents parent-group guarantees.',
                 'Lexical and multilingual semantic screening finds candidates, not verified medical/translation equivalence.',
                 'Answer paraphrase overlap and historical base-model exposure are not ruled out.']}
    atomic_json(out / 'corpus.json', result)
    progress('SUCCEEDED', counts={split: len(records) for split, records in splits.items()})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True, type=Path)
    args = parser.parse_args()
    root = args.data_root.resolve()
    with (root / 'v4/preparation.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = prepare(root)
    print(json.dumps({'status': result['status'], 'freeze': result['freeze_manifest']}), flush=True)


if __name__ == '__main__':
    main()
