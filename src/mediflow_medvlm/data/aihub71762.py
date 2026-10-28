"""Bounded local 71762 delivery import and unpaired-component audit.

The delivered questions and answers have independent identifiers. This module
never manufactures a QA pair, a parent group, or an SFT target from categories,
numeric filename suffixes, file order, or participant identifiers. Original
JSON remains in the immutable ZIPs; the private SQLite inventory holds only
question/answer components and provenance needed for a later evidenced join.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
import zipfile

from .aihub import _private, _safe_json, _safe_member, credential_free_url
from .io import DATA_BYTES_CAP, atomic_json, digest, disk_guard, now, read_jsonl, sha256
from .schema import normalized

SCHEMA = 'medvlm.aihub71762.v3.1'
JSON_LIMIT = 64 * 1024**2
ROLE_NAMES = {'TS.zip': ('train', 'source'), 'TL.zip': ('train', 'label'),
              'VS.zip': ('validation', 'source'), 'VL.zip': ('validation', 'label'),
              'Other.zip': (None, 'reference_documents')}
ID_PATTERN = re.compile(r'HC-([QA])-\d+$')
LINK_KEY = re.compile(r'(question|answer|parent|origin|augment|reference|ref).*(id|name|key)|'
                      r'(id|name|key).*(question|answer|parent|origin|augment|reference)', re.I)


def _json(path, value):
    _private(Path(path).parent)
    atomic_json(path, value)
    os.chmod(path, 0o600)


def _sha_text(value):
    return hashlib.sha256(normalized(value).encode()).hexdigest()


def _safe_info(info):
    _safe_member(info.filename, streaming_root_marker=True)
    if stat.S_ISLNK(info.external_attr >> 16):
        raise ValueError('UNSAFE_ARCHIVE_SYMLINK')
    if info.flag_bits & 1:
        raise ValueError('ENCRYPTED_ARCHIVE_UNSUPPORTED')


def import_delivery(archive, data_root, *, version, delivery=None):
    """Validate outer CRCs and install nested ZIPs; inner audit is separate.

    Only archive-level metadata is kept in memory, rather than millions of
    member dictionaries. Every inner member is checked by audit_delivery().
    """
    source = Path(archive)
    if source.is_symlink() or not source.is_file() or not version:
        raise ValueError('LOCAL_ORIGINAL_ZIP_AND_VERSION_REQUIRED')
    source = source.resolve()
    delivery = dict(delivery or {'method': 'user_provided_local'})
    for key in delivery:
        if any(part in key.lower() for part in ('token', 'secret', 'password', 'api_key')):
            raise ValueError('SECRET_IN_DELIVERY_METADATA')
        if 'url' in key.lower():
            delivery[key] = credential_free_url(delivery[key])
    root = _private(data_root)
    source_sha = sha256(source)
    identity = {'schema_version': SCHEMA, 'dataset_id': '71762', 'version': version,
                'source': {'path': str(source), 'bytes': source.stat().st_size, 'sha256': source_sha}}
    import_id = digest(identity)
    manifest_path = root / 'manifests/imports' / (import_id + '.json')
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        for item in previous['archives']:
            if sha256(item['path']) != item['sha256']:
                raise ValueError('IMPORTED_ARCHIVE_CHANGED')
        return previous
    target_root = _private(root / 'raw/aihub_71762/nested' / source_sha)
    archives = []
    with zipfile.ZipFile(source) as outer:
        infos = outer.infolist()
        expanded = sum(i.file_size for i in infos)
        if source.stat().st_size + expanded > DATA_BYTES_CAP:
            raise ValueError('ARCHIVE_EXPANSION_EXCEEDS_DATA_CAP')
        seen = set()
        for info in infos:
            _safe_info(info)
            logical = str(_safe_member(info.filename, streaming_root_marker=True))
            if logical in seen:
                raise ValueError('DUPLICATE_ARCHIVE_MEMBER')
            seen.add(logical)
            if info.is_dir():
                continue
            name = Path(logical).name
            if name not in ROLE_NAMES:
                # Unexpected delivery artifacts must be inspected explicitly.
                raise ValueError('UNRECOGNIZED_71762_DELIVERY_MEMBER')
            if any(a['name'] == name for a in archives):
                raise ValueError('DUPLICATE_DELIVERY_ARCHIVE_ROLE')
            target = target_root / name
            temporary = target.with_suffix('.zip.part')
            if target.is_symlink() or temporary.is_symlink():
                raise ValueError('UNSAFE_IMPORT_SYMLINK')
            disk_guard(target_root, info.file_size if not target.exists() else 0)
            h = hashlib.sha256()
            with outer.open(info) as src:
                if target.exists():
                    for block in iter(lambda: src.read(1024**2), b''):
                        h.update(block)
                else:
                    with temporary.open('wb') as dst:
                        for block in iter(lambda: src.read(1024**2), b''):
                            h.update(block)
                            dst.write(block)
                    os.chmod(temporary, 0o600)
                    os.replace(temporary, target)
            if target.stat().st_size != info.file_size or sha256(target) != h.hexdigest():
                raise ValueError('NESTED_ARCHIVE_HASH_MISMATCH')
            with zipfile.ZipFile(target) as inner:
                member_count = len(inner.infolist())
                inner_expanded = sum(i.file_size for i in inner.infolist())
            if inner_expanded > DATA_BYTES_CAP:
                raise ValueError('ARCHIVE_EXPANSION_EXCEEDS_DATA_CAP')
            split, role = ROLE_NAMES[name]
            archives.append({'name': name, 'path': str(target), 'split': split, 'role': role,
                             'bytes': target.stat().st_size, 'sha256': h.hexdigest(),
                             'outer_member': logical, 'outer_crc32': f'{info.CRC:08x}',
                             'entries': member_count, 'expanded_bytes': inner_expanded})
    # Limit total logical expansion, including source/label duplicates.
    if sum(a['expanded_bytes'] for a in archives) > DATA_BYTES_CAP:
        raise ValueError('ARCHIVE_EXPANSION_EXCEEDS_DATA_CAP')
    result = {**identity, 'import_id': import_id, 'status': 'SUCCEEDED',
              'status_scope': 'local_delivery_and_outer_CRC_only',
              'evidence_level': 'ACTUAL_LOCAL_DELIVERY_VERIFIED',
              'delivery': delivery,
              'received_at': now(), 'manifest_path': str(manifest_path),
              'archives': archives,
              'members': [{'relative_path': a['outer_member'], 'path': a['path'], 'bytes': a['bytes'],
                           'sha256': a['sha256'], 'original_split': a['split'],
                           'excluded_path': a['role'] == 'reference_documents'} for a in archives],
              'member_manifest_mode': 'archive_level_import; bounded_SQLite_members_in_audit',
              'archive_integrity': 'outer_members_CRC_verified; inner_members_pending_full_audit',
              'reader_validation': 'NOT_RUN', 'pairing_verified': False,
              'originals_preserved': True, 'raw_redistribution': False,
              'official_source': 'https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71762'}
    _json(manifest_path, result)
    return result


def _schema_paths(value, prefix=''):
    """Schema only: no demographics or clinical text in aggregate reports."""
    if isinstance(value, dict):
        for key, item in value.items():
            path = prefix + '.' + key if prefix else key
            yield path, type(item).__name__
            if isinstance(item, dict):
                yield from _schema_paths(item, path)


def component(obj, member):
    """Normalize a separate component, preserving answer section contents."""
    if not isinstance(obj, dict):
        raise ValueError('JSON_OBJECT_REQUIRED')
    first = member.lstrip('/').split('/')[0]
    kind = {'1.질문': 'question', '2.답변': 'answer'}.get(first)
    if kind is None:
        raise ValueError('UNRECOGNIZED_COMPONENT_DIRECTORY')
    sid = obj.get('fileName')
    match = ID_PATTERN.fullmatch(sid) if isinstance(sid, str) else None
    if not match or match[1] != ('Q' if kind == 'question' else 'A'):
        raise ValueError('INVALID_COMPONENT_ID')
    if Path(member).stem != sid:
        raise ValueError('FILENAME_ID_MISMATCH')
    for key in ('disease_category', 'intention'):
        if not isinstance(obj.get(key), str) or not obj[key].strip():
            raise ValueError('INVALID_COMPONENT_CATEGORY')
    if not isinstance(obj.get('disease_name'), dict):
        raise ValueError('INVALID_DISEASE_NAME_STRUCTURE')
    if kind == 'question':
        if not isinstance(obj.get('question'), str) or not obj['question'].strip():
            raise ValueError('EMPTY_OR_INVALID_QUESTION')
        content = {'question': obj['question']}
        text = obj['question']
    else:
        answer = obj.get('answer')
        if not isinstance(answer, dict) or not isinstance(answer.get('body'), str) or not answer['body'].strip():
            raise ValueError('EMPTY_OR_INVALID_ANSWER_BODY')
        if any(v is not None and not isinstance(v, str) for v in answer.values()):
            raise ValueError('INVALID_ANSWER_SECTION_TYPE')
        content = {'answer': answer}
        text = '\n\n'.join(answer.get(k) or '' for k in ('intro', 'body', 'conclusion'))
    references = {}
    def find_references(items, prefix=''):
        for key, value in items.items():
            if key in {'fileName', 'participantsInfo'}:
                continue
            path = prefix + '.' + key if prefix else key
            if LINK_KEY.search(key) or key.lower() in {'qid', 'aid', 'parent', 'parentid', 'originalid'}:
                references[path] = value
            elif isinstance(value, dict):
                find_references(value, path)
    find_references(obj)
    return {'kind': kind, 'source_id': sid, 'content': content, 'text_sha256': _sha_text(text),
            'domain': obj['disease_category'], 'intention': obj['intention'],
            'references': references}


def _protected(path, *, expected_count=4068):
    if path is None:
        return {}, {'status': 'NOT_REQUESTED', 'count': 0}
    manifest = json.loads(Path(path).read_text())
    keys = {}; receipts = {}
    for split in ('validation', 'test'):
        item = manifest['splits']['kormedmcqa/' + split]
        file = Path(item.get('snapshot_path', item['path']))
        if sha256(file) != item['sha256']:
            raise ValueError('PROTECTED_KORMED_FILE_CHANGED')
        rows = read_jsonl(file)
        if len(rows) != item['count']:
            raise ValueError('PROTECTED_KORMED_COUNT_CHANGED')
        for row in rows:
            keys.setdefault(_sha_text(row['input']['question']), []).append(row['sample_id'])
        receipts[split] = {'count': len(rows), 'sha256': item['sha256']}
    count = sum(x['count'] for x in receipts.values())
    if count != expected_count:
        raise ValueError('FULL_OFFICIAL_KORMED_DEV_TEST_4068_REQUIRED')
    return keys, {'status': 'EXACT_QUESTION_ONLY_SCREEN', 'count': count, 'files': receipts,
                  'manifest_sha256': sha256(path), 'heldout_modified': False,
                  'near_semantic_status': 'NOT_RUN_NO_PAIRED_TRAIN_CANDIDATES',
                  'clinical_or_translation_equivalence_verified': False}


def audit_delivery(import_manifest, data_root, *, protected_freeze=None, license_ref):
    """Check every inner CRC/JSON, index assets, and keep unpaired components.

    At most one ZIP central directory, small batches and schema counters are
    resident. SQLite stores per-member hashes, avoiding a multi-million-item
    Python manifest. A restart discards only the incomplete archive's derived
    rows and reuses fully audited archive receipts in this identity.
    """
    if not license_ref:
        raise ValueError('SOURCE_RIGHTS_EVIDENCE_REQUIRED')
    root = _private(data_root)
    imported = json.loads(Path(import_manifest).read_text())
    if imported.get('schema_version') != SCHEMA or imported.get('dataset_id') != '71762':
        raise ValueError('BOUNDED_71762_IMPORT_REQUIRED')
    if sha256(imported['source']['path']) != imported['source']['sha256']:
        raise ValueError('DELIVERY_SOURCE_CHANGED')
    protected, protection = _protected(protected_freeze)
    identity = {'import_manifest_sha256': sha256(import_manifest), 'code_sha256': sha256(__file__),
                'helper_code_sha256': sha256(Path(__file__).with_name('aihub.py')),
                'protected': protection, 'license_ref': license_ref}
    audit_id = digest(identity)
    folder = _private(root / 'prepared/aihub_71762' / audit_id)
    database = folder / 'components.sqlite3'
    receipt_path = folder / 'audit.json'
    if receipt_path.exists():
        previous = json.loads(receipt_path.read_text())
        if sha256(database) != previous['database']['sha256']:
            raise ValueError('IMMUTABLE_COMPONENT_INVENTORY_CHANGED')
        for item in imported['archives']:
            if sha256(item['path']) != item['sha256']:
                raise ValueError('IMPORTED_ARCHIVE_CHANGED')
        return previous
    snapshot = _private(folder / 'source_snapshot') / 'aihub71762.py'
    if snapshot.exists() and snapshot.read_bytes() != Path(__file__).read_bytes():
        raise ValueError('IMMUTABLE_TRANSFORM_SNAPSHOT_CHANGED')
    if not snapshot.exists():
        snapshot.write_bytes(Path(__file__).read_bytes()); os.chmod(snapshot, 0o600)
    disk_guard(folder, sum(a['expanded_bytes'] for a in imported['archives'] if a['role'] == 'label') * 2)
    connection = sqlite3.connect(database)
    os.chmod(database, 0o600)
    connection.execute('PRAGMA cache_size=-65536')
    connection.execute('PRAGMA temp_store=FILE')
    connection.executescript('''
      CREATE TABLE IF NOT EXISTS assets (
        archive TEXT, member TEXT, sha256 TEXT, bytes INTEGER, crc INTEGER,
        issue TEXT, PRIMARY KEY(archive,member));
      CREATE TABLE IF NOT EXISTS components (
        archive TEXT, member TEXT, split TEXT, kind TEXT, source_id TEXT,
        text_sha256 TEXT, content_json TEXT, domain TEXT, intention TEXT,
        references_json TEXT, PRIMARY KEY(archive,member));
      CREATE TABLE IF NOT EXISTS protected_hits (
        archive TEXT, member TEXT, protected_ids_json TEXT);
      CREATE TABLE IF NOT EXISTS completed_archives (
        archive TEXT PRIMARY KEY, sha256 TEXT, receipt_json TEXT);
    ''')
    try:
        ordered = sorted(imported['archives'], key=lambda a: ({'label': 0, 'source': 1, 'reference_documents': 2}[a['role']], a['name']))
        for archive in ordered:
            name = archive['name']
            if sha256(archive['path']) != archive['sha256']:
                raise ValueError('IMPORTED_ARCHIVE_CHANGED')
            old = connection.execute('SELECT sha256 FROM completed_archives WHERE archive=?', (name,)).fetchone()
            if old:
                if old[0] != archive['sha256']:
                    raise ValueError('RESUME_ARCHIVE_IDENTITY_CHANGED')
                continue
            for table in ('assets', 'components', 'protected_hits'):
                connection.execute(f'DELETE FROM {table} WHERE archive=?', (name,))
            connection.commit()
            counts = Counter(); schemas = Counter(); fields = Counter(); issues = Counter(); boms = Counter()
            reference_keys = Counter(); sections = Counter(); assets = []; components = []; hits = []
            with zipfile.ZipFile(archive['path']) as source:
                for info in source.infolist():
                    _safe_info(info)
                    member = str(_safe_member(info.filename, streaming_root_marker=True))
                    if info.is_dir():
                        continue
                    counts['files'] += 1
                    h = hashlib.sha256(); raw = bytearray()
                    parse = archive['role'] == 'label' and info.filename.lower().endswith('.json')
                    if parse and info.file_size > JSON_LIMIT:
                        raise ValueError('JSON_DOCUMENT_SIZE_LIMIT')
                    with source.open(info) as stream:
                        for block in iter(lambda: stream.read(1024**2), b''):
                            h.update(block)
                            if parse: raw.extend(block)
                    issue = None
                    if parse:
                        counts['json'] += 1
                        try:
                            obj, events = _safe_json(bytes(raw)); boms.update(events)
                            if not isinstance(obj, dict): raise ValueError('JSON_OBJECT_REQUIRED')
                            schemas[json.dumps(sorted(obj), ensure_ascii=False)] += 1
                            fields.update(path + ':' + kind for path, kind in _schema_paths(obj))
                            row = component(obj, member)
                            counts[row['kind']] += 1
                            reference_keys.update(row['references'].keys())
                            if row['kind'] == 'answer':
                                sections.update(k for k, v in row['content']['answer'].items() if isinstance(v, str) and v.strip())
                            components.append((name, member, archive['split'], row['kind'], row['source_id'],
                                               row['text_sha256'], json.dumps(row['content'], ensure_ascii=False, separators=(',', ':')),
                                               row['domain'], row['intention'], json.dumps(row['references'], ensure_ascii=False, separators=(',', ':'))))
                            if row['kind'] == 'question' and row['text_sha256'] in protected:
                                hits.append((name, member, json.dumps(protected[row['text_sha256']])));
                                counts['protected_exact_question_hits'] += 1
                        except (ValueError, UnicodeDecodeError, TypeError, KeyError) as exc:
                            # Exceptions can contain source text. Persist only a bounded reason.
                            reason = str(exc)
                            issue = reason if re.fullmatch('[A-Z_]{4,80}', reason) else 'INVALID_COMPONENT_JSON_OR_SCHEMA'
                            issues[issue] += 1
                    assets.append((name, member, h.hexdigest(), info.file_size, info.CRC, issue))
                    if len(assets) >= 2000:
                        _flush(connection, assets, components, hits)
                    if counts['files'] % 50000 == 0:
                        _json(folder / 'progress.json', {'status': 'RUNNING', 'archive': name,
                              'files_verified': counts['files'], 'counts': dict(counts), 'at': now(), 'pid': os.getpid()})
                _flush(connection, assets, components, hits)
            local_receipt = {'archive': name, 'counts': dict(counts), 'top_level_schemas': dict(schemas),
                             'field_types': dict(fields), 'invalid_components': dict(issues),
                             'bom_events': dict(boms), 'reference_key_candidates': dict(reference_keys),
                             'answer_nonempty_sections': dict(sections), 'all_file_CRCs_verified': True}
            connection.execute('INSERT INTO completed_archives VALUES (?,?,?)',
                               (name, archive['sha256'], json.dumps(local_receipt, ensure_ascii=False)))
            connection.commit()
            _json(folder / 'progress.json', {'status': 'ARCHIVE_COMPLETED', 'archive': name,
                  'counts': dict(counts), 'at': now(), 'pid': os.getpid()})
        connection.executescript('''
          CREATE INDEX IF NOT EXISTS component_ids ON components(kind,source_id,split);
          CREATE INDEX IF NOT EXISTS component_text ON components(kind,text_sha256,split);
        ''')
        connection.commit()
        receipts = [json.loads(row[0]) for row in connection.execute('SELECT receipt_json FROM completed_archives ORDER BY archive')]
        if connection.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise ValueError('COMPONENT_INVENTORY_INTEGRITY_FAILURE')
        by_split = [dict(zip(('split', 'kind', 'count'), row)) for row in connection.execute('SELECT split,kind,count(*) FROM components GROUP BY split,kind')]
        duplicate_ids = [dict(zip(('kind', 'groups', 'excess_rows'), row)) for row in connection.execute('SELECT kind,count(*),sum(n-1) FROM (SELECT kind,source_id,count(*) n FROM components GROUP BY kind,source_id HAVING n>1) GROUP BY kind')]
        exact_duplicates = [dict(zip(('kind', 'groups', 'excess_rows'), row)) for row in connection.execute('SELECT kind,count(*),sum(n-1) FROM (SELECT kind,text_sha256,count(*) n FROM components GROUP BY kind,text_sha256 HAVING n>1) GROUP BY kind')]
        exact_cross_split = [dict(zip(('kind', 'text_groups', 'rows'), row)) for row in connection.execute('SELECT kind,count(*),sum(n) FROM (SELECT kind,text_sha256,count(*) n FROM components GROUP BY kind,text_sha256 HAVING count(DISTINCT split)>1) GROUP BY kind')]
        raw_label = []
        names = {a['name'] for a in imported['archives']}
        for raw_name, label_name in (('TS.zip', 'TL.zip'), ('VS.zip', 'VL.zip')):
            if {raw_name, label_name} <= names:
                raw_label.append({'source_archive': raw_name, 'label_archive': label_name,
                    'source_files_without_label': connection.execute('SELECT count(*) FROM assets r LEFT JOIN assets l ON l.archive=? AND l.member=r.member WHERE r.archive=? AND l.member IS NULL', (label_name, raw_name)).fetchone()[0],
                    'label_files_without_source': connection.execute('SELECT count(*) FROM assets l LEFT JOIN assets r ON r.archive=? AND r.member=l.member WHERE l.archive=? AND r.member IS NULL', (raw_name, label_name)).fetchone()[0],
                    'matching_path_different_bytes': connection.execute('SELECT count(*) FROM assets r JOIN assets l ON l.archive=? AND l.member=r.member WHERE r.archive=? AND r.sha256!=l.sha256', (label_name, raw_name)).fetchone()[0]})
        protection['exact_question_component_hits'] = connection.execute('SELECT count(*) FROM protected_hits').fetchone()[0]
        candidate_keys = dict(sum((Counter(r['reference_key_candidates']) for r in receipts), Counter()))
        normalized_count = connection.execute('SELECT count(*) FROM components').fetchone()[0]
        files_count = connection.execute('SELECT count(*) FROM assets').fetchone()[0]
    finally:
        connection.close()
    result = {'schema_version': SCHEMA, 'source_id': 'aihub_71762', 'source_version': imported['version'],
              'audit_id': audit_id, 'identity': identity, 'import_manifest': str(import_manifest),
              'source_zip_sha256': imported['source']['sha256'], 'actual_acquired_bytes': imported['source']['bytes'],
              'status': 'UNRESOLVED_PAIRING', 'delivery_status': 'SUCCEEDED',
              'component_audit_status': 'SUCCEEDED', 'component_normalization': 'UNPAIRED_ONLY',
              'evidence_level': 'ACTUAL_FULL_SOURCE_CRC_AND_LABEL_SCHEMA_AUDIT',
              'verified_asset_files': files_count, 'normalized_components': normalized_count,
              'component_split_counts': by_split, 'archive_receipts': receipts,
              'source_label_content_comparison': raw_label, 'duplicate_component_ids': duplicate_ids,
              'exact_component_text_duplicates': exact_duplicates, 'exact_component_text_cross_split': exact_cross_split,
              'reference_key_candidates': candidate_keys, 'protected_kormed': protection,
              'pairing': {'verified': False, 'question_answer_pairs_created': 0,
                          'reason': 'No evidenced question-to-answer foreign key and augmentation-parent contract supplied',
                          'category_or_position_join_used': False, 'numeric_filename_suffix_join_used': False,
                          'participant_id_used_as_patient_or_parent': False},
              'canonical_sft_records': 0, 'training_status': 'NOT_RUN_UNRESOLVED_PAIRING',
              'supervised_tokens': None, 'new_freeze_created': False,
              'official_test_supplied': False, 'original_splits_preserved': True,
              'source_documents_converted_to_qa': False, 'human_review': 'not_performed',
              'per_record_generation_or_expert_review': 'unknown; official source includes BART augmentation and mixed authorship',
              'rights_evidence': license_ref, 'originals_preserved': True, 'raw_redistribution': False,
              'database': {'path': str(database), 'bytes': database.stat().st_size, 'sha256': sha256(database)},
              'manifest_path': str(receipt_path), 'completed_at': now()}
    _json(receipt_path, result)
    _json(folder / 'progress.json', {'status': result['status'], 'component_audit_status': 'SUCCEEDED', 'at': now(), 'pid': None})
    return result


def _flush(connection, assets, components, hits):
    try:
        connection.executemany('INSERT INTO assets VALUES (?,?,?,?,?,?)', assets)
        connection.executemany('INSERT INTO components VALUES (?,?,?,?,?,?,?,?,?,?)', components)
        connection.executemany('INSERT INTO protected_hits VALUES (?,?,?)', hits)
        connection.commit()
    except sqlite3.IntegrityError:
        raise ValueError('DUPLICATE_ARCHIVE_MEMBER') from None
    assets.clear(); components.clear(); hits.clear()
