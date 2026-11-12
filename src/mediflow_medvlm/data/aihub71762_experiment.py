"""Explicitly authorized 71762 category-random-pairing experiment.

This never changes the audited component inventory or infers a true QA/parent
identifier. Only label copies are used. Sampling is fixed before token/leakage
screening or model scores and preserves the provider's category constraints.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
import heapq
import fcntl
import json
from pathlib import Path, PurePosixPath
import sqlite3

from .aihub import _private
from .io import atomic_json, digest, sha256, write_jsonl
from .schema import normalized, sample, validate_record

VERSION = 'medvlm.aihub71762.category-random.v1'
SEED = 20261023


def check_authorization(value):
    if (value.get('dataset_id') != '71762' or
            value.get('decision') != 'ALLOW_SEPARATE_CATEGORY_RANDOM_EXPERIMENT' or
            value.get('human_instruction') != '71762는 제공처 방식으로 별도 실험 진행'):
        raise ValueError('EXPLICIT_71762_CATEGORY_RANDOM_AUTHORIZATION_REQUIRED')
    return value


def category_key(member, domain, intention):
    parts = PurePosixPath(member).parts
    if (len(parts) != 5 or parts[0] not in {'1.질문', '2.답변'} or
            any(p in {'..', '.', ''} for p in parts)):
        raise ValueError('UNEXPECTED_CATEGORY_PATH')
    if normalized(parts[1]) != normalized(domain) or normalized(parts[3]) != normalized(intention):
        raise ValueError('PATH_TAG_CATEGORY_MISMATCH')
    return tuple(normalized(p) for p in parts[1:4])


def component_split(original_split, text_hash, seed=SEED):
    if original_split == 'train':
        return 'train'
    if original_split != 'validation':
        raise ValueError('UNEXPECTED_ORIGINAL_SPLIT')
    # Apply the same rule to question and answer exact-text groups. VL alone
    # provides both derived dev/test; this is not a delivered official test.
    return 'validation' if int(digest([seed, 'heldout', text_hash]), 16) % 2 == 0 else 'test'


def _keep(heap, row, count, seed):
    priority = int(digest([seed, row['archive'], row['member']]), 16)
    value = (-priority, row['archive'], row['member'], row)
    if len(heap) < count:
        heapq.heappush(heap, value)
    elif value[:3] > heap[0][:3]:
        heapq.heapreplace(heap, value)


def component_pools(connection, *, questions_per_category=4, answers_per_category=8, seed=SEED):
    """Bounded per-category priority reservoirs; exact cross-split exclusion."""
    cross = set(connection.execute('''SELECT kind,text_sha256 FROM components
        GROUP BY kind,text_sha256 HAVING COUNT(DISTINCT split)>1'''))
    pools = defaultdict(list)
    seen = set()
    counts = Counter()
    query = '''SELECT archive,member,split,kind,text_sha256,domain,intention
               FROM components ORDER BY archive,member'''
    for archive, member, original, kind, text_hash, domain, intention in connection.execute(query):
        counts['observed_' + kind] += 1
        if archive not in {'TL.zip', 'VL.zip'}:
            raise ValueError('LABEL_ONLY_COMPONENT_INVENTORY_REQUIRED')
        if (kind, text_hash) in cross:
            counts['excluded_cross_split_' + kind] += 1
            continue
        if (kind, text_hash) in seen:
            counts['excluded_duplicate_' + kind] += 1
            continue
        seen.add((kind, text_hash))
        key = category_key(member, domain, intention)
        split = component_split(original, text_hash, seed)
        row = {'archive': archive, 'member': member, 'original_split': original,
               'kind': kind, 'text_sha256': text_hash, 'category': key, 'split': split}
        _keep(pools[(split, kind, key)], row,
              questions_per_category if kind == 'question' else answers_per_category, seed)
    result = {k: [entry[3] for entry in sorted(v, key=lambda x: (-x[0], x[1], x[2]))]
              for k, v in pools.items()}
    return result, {'counts': dict(counts), 'cross_split_exact_component_groups': len(cross),
                    'reservoir_categories': len(pools),
                    'questions_per_category_cap': questions_per_category,
                    'answers_per_category_cap': answers_per_category,
                    'full_source_exact_cross_split_screen': True}


def pair_components(pools, *, seed=SEED):
    """One seeded random answer per question; answer reuse is allowed/reported."""
    pairs, missing = [], Counter()
    for (split, kind, key), questions in sorted(pools.items()):
        if kind != 'question':
            continue
        answers = pools.get((split, 'answer', key), [])
        if not answers:
            missing[split] += len(questions)
            continue
        for question in questions:
            position = int(digest([seed, 'answer-choice', question['archive'], question['member']]), 16) % len(answers)
            answer = answers[position]
            if question['category'] != answer['category'] or question['split'] != answer['split']:
                raise ValueError('CROSS_CATEGORY_OR_SPLIT_PAIR')
            pairs.append((question, answer))
    return pairs, dict(missing)


def balanced_pairs(pairs, count, seed=SEED):
    domains = defaultdict(list)
    for question, answer in pairs:
        domains[question['category'][0]].append((question, answer))
    queues = {}
    for domain, rows in sorted(domains.items()):
        queues[domain] = deque(sorted(rows, key=lambda p: digest([seed, p[0]['archive'], p[0]['member']])))
    selected = []
    while len(selected) < count:
        progressed = False
        for queue in queues.values():
            if queue and len(selected) < count:
                selected.append(queue.popleft())
                progressed = True
        if not progressed:
            break
    return selected


def canonical_pair(connection, question, answer, *, revision, authorization_sha256, fetched_at):
    if question['category'] != answer['category'] or question['split'] != answer['split']:
        raise ValueError('CROSS_CATEGORY_OR_SPLIT_PAIR')
    def content(item):
        row = connection.execute('''SELECT c.content_json,a.sha256 FROM components c
            JOIN assets a ON c.archive=a.archive AND c.member=a.member
            WHERE c.archive=? AND c.member=?''', (item['archive'], item['member'])).fetchone()
        if row is None:
            raise ValueError('COMPONENT_OR_ASSET_MISSING')
        return json.loads(row[0]), row[1]
    q, q_sha = content(question)
    a, a_sha = content(answer)
    sections = a['answer']
    target = '\n\n'.join(sections.get(k) or '' for k in ('intro', 'body', 'conclusion'))
    identity = [question['archive'], question['member'], answer['archive'], answer['member']]
    row = sample('aihub_71762_category_random', revision, digest(identity), 'medical_qa',
                 q['question'], target, question['split'], language='ko',
                 original_split=question['original_split'], fetched_at=fetched_at,
                 source_url='https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71762',
                 license_ref='https://www.aihub.or.kr/intrcn/guid/usagepolicy.do?currMenu=151',
                 raw_sha256=q_sha, annotation='provider_recommended_category_random_pairing',
                 metadata={'domain': question['category'][0], 'disease_name': question['category'][1],
                           'intention': question['category'][2], 'pairing_method': 'category_random_pairing',
                           'true_qa_correspondence_verified': False, 'augmentation_parent_available': False,
                           'authorization_sha256': authorization_sha256,
                           'question_component': {**question, 'raw_sha256': q_sha},
                           'answer_component': {**answer, 'raw_sha256': a_sha},
                           'answer_sections': sections, 'sampling_seed': SEED})
    row['derivation'] = 'category_random_pairing'
    row['group_ids']['duplicate_cluster'] = question['text_sha256']
    row['grouping_quality'].update(status='proxy', level='exact_question_text',
                                   augmentation_parent_verified=False)
    row['provenance']['transformation_version'] = VERSION
    if validate_record(row):
        raise ValueError('INVALID_PAIRED_CANONICAL')
    return row


def build_pool(audit_manifest, authorization_path, output_dir, *, train_candidates=12000, heldout_candidates=512):
    out = _private(output_dir)
    with (out / 'pool.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('PAIRING_POOL_ALREADY_OWNED') from None
        return _build_pool(audit_manifest, authorization_path, out,
                           train_candidates=train_candidates, heldout_candidates=heldout_candidates)


def _build_pool(audit_manifest, authorization_path, output_dir, *, train_candidates, heldout_candidates):
    authorization_path = Path(authorization_path)
    check_authorization(json.loads(authorization_path.read_text()))
    audit_manifest = Path(audit_manifest)
    audit = json.loads(audit_manifest.read_text())
    database = Path(audit['database']['path'])
    if sha256(database) != audit['database']['sha256']:
        raise ValueError('AUDITED_COMPONENT_DATABASE_CHANGED')
    identity = {'schema': VERSION, 'audit_sha256': sha256(audit_manifest),
                'database_sha256': audit['database']['sha256'],
                'authorization_sha256': sha256(authorization_path), 'code_sha256': sha256(__file__),
                'helper_code_sha256': {name: sha256(Path(__file__).with_name(name))
                                       for name in ('aihub.py', 'io.py', 'schema.py')},
                'seed': SEED, 'train_candidates': train_candidates, 'heldout_candidates': heldout_candidates,
                'question_pool_per_category': 4, 'answer_pool_per_category': 8}
    out = _private(output_dir)
    snapshot = out / 'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    source_copy = snapshot / Path(__file__).name
    if source_copy.exists() and sha256(source_copy) != identity['code_sha256']:
        raise ValueError('PAIRING_SOURCE_SNAPSHOT_CHANGED')
    if not source_copy.exists():
        source_copy.write_bytes(Path(__file__).read_bytes())
    receipt = out / 'pool.json'
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        if previous['identity'] != identity:
            raise ValueError('IMMUTABLE_PAIRING_POOL_IDENTITY_CHANGED')
        for entry in previous['splits'].values():
            if sha256(entry['path']) != entry['sha256']:
                raise ValueError('IMMUTABLE_PAIRING_POOL_CHANGED')
        return previous
    c = sqlite3.connect('file:' + str(database.resolve()) + '?mode=ro', uri=True)
    try:
        pools, inventory = component_pools(c)
        pairs, missing = pair_components(pools)
        splits = {}
        for split in ('train', 'validation', 'test'):
            selected = balanced_pairs([p for p in pairs if p[0]['split'] == split],
                                      train_candidates if split == 'train' else heldout_candidates)
            records = [canonical_pair(c, q, a, revision=digest(identity),
                                     authorization_sha256=identity['authorization_sha256'],
                                     fetched_at='2026-11-12') for q, a in selected]
            path = out / (split + '.jsonl')
            if path.exists():
                raise ValueError('INCOMPLETE_POOL_EXISTS_PRESERVE_FOR_REVIEW')
            write_jsonl(path, records)
            answer_counts = Counter(r['metadata']['answer_component']['text_sha256'] for r in records)
            splits[split] = {'path': str(path.resolve()), 'sha256': sha256(path), 'count': len(records),
                             'domain_counts': dict(Counter(r['metadata']['domain'] for r in records)),
                             'unique_answer_texts': len(answer_counts),
                             'answer_reuse_rows': sum(n-1 for n in answer_counts.values()),
                             'max_answer_reuse': max(answer_counts.values(), default=0)}
        result = {'schema': VERSION, 'identity': identity, 'splits': splits, 'inventory': inventory,
                  'missing_answer_pool_question_rows': missing, 'status': 'PAIRED_POOL_NOT_YET_TRAIN_ELIGIBLE',
                  'true_qa_correspondence_verified': False, 'augmentation_lineage_verified': False,
                  'sampling': 'domain_round_robin_from_four_hash_priority_questions_per_category; '
                              'seeded_choice_among_eight_hash_priority_answer_components_per_category',
                  'heldout_policy': 'official_validation_only; exact_text_group_hash_parity; not_official_test'}
        atomic_json(receipt, result)
        return result
    finally:
        c.close()
