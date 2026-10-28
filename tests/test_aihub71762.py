"""Synthetic contracts; actual 71762 evidence is stored outside Git."""
import io
import json
from pathlib import Path
import sqlite3
import zipfile

import pytest

from mediflow_medvlm.data import aihub71762 as source


def question(suffix='1'):
    return {'fileName': 'HC-Q-' + suffix, 'disease_category': '합성 분류',
            'disease_name': {'kor': '합성 질환', 'eng': 'fixture'}, 'intention': '합성 의도',
            'participantsInfo': {'participantID': 'fixture-person'},
            'question': '2 mg을 사용하지 않은 합성 조건은?'}


def answer(suffix='1'):
    return {'fileName': 'HC-A-' + suffix, 'disease_category': '합성 분류',
            'disease_name': {'kor': '합성 질환', 'eng': 'fixture'}, 'intention': '합성 의도',
            'answer': {'intro': ' 합성 서론 ', 'body': '2 mg을 사용하지 않았다.', 'conclusion': None}}


def bundle(tmp_path, monkeypatch, *, bad_json=False, unsafe=False, same_logical_name=False):
    monkeypatch.setattr(source, 'disk_guard', lambda *a: None)
    q = json.dumps(question(), ensure_ascii=False)
    a = json.dumps(answer(), ensure_ascii=False)
    entries = {'1.질문/HC-Q-1.json': q, '2.답변/HC-A-1.json': a}
    if bad_json:
        entries['1.질문/HC-Q-2.json'] = '{"fileName":1,"\\ufefffileName":2}'
    if unsafe:
        entries['../escape.json'] = '{}'
    if same_logical_name:
        entries['/1.질문/HC-Q-1.json'] = q
    nested = io.BytesIO()
    with zipfile.ZipFile(nested, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for name, value in entries.items():
            z.writestr(name, value)
    original = tmp_path / 'delivery.zip'
    with zipfile.ZipFile(original, 'w') as z:
        for role in ('TS', 'TL', 'VS', 'VL'):
            z.writestr(role + '.zip', nested.getvalue())
    return original


def test_realistic_independent_ids_do_not_make_pairs_and_sections_are_preserved(tmp_path, monkeypatch):
    original = bundle(tmp_path, monkeypatch)
    before = original.read_bytes()
    imported = source.import_delivery(original, tmp_path / 'data', version='fixture')
    report = source.audit_delivery(imported['manifest_path'], tmp_path / 'data', license_ref='synthetic_only')
    assert report['delivery_status'] == report['component_audit_status'] == 'SUCCEEDED'
    assert report['status'] == 'UNRESOLVED_PAIRING'
    assert report['canonical_sft_records'] == report['pairing']['question_answer_pairs_created'] == 0
    assert not report['pairing']['numeric_filename_suffix_join_used']
    assert not report['pairing']['participant_id_used_as_patient_or_parent']
    assert report['normalized_components'] == 4 and report['verified_asset_files'] == 8
    assert report['reference_key_candidates'] == {}
    assert all(item['matching_path_different_bytes'] == 0 for item in report['source_label_content_comparison'])
    assert {x['kind'] for x in report['exact_component_text_cross_split']} == {'question', 'answer'}
    with sqlite3.connect(report['database']['path']) as connection:
        text = connection.execute("SELECT content_json FROM components WHERE kind='answer' LIMIT 1").fetchone()[0]
    assert json.loads(text)['answer'] == answer()['answer']
    assert original.read_bytes() == before
    # Only unpaired component fields are exported, never participantsInfo.
    assert 'fixture-person' not in text
    assert source.audit_delivery(imported['manifest_path'], tmp_path / 'data', license_ref='synthetic_only') == report


def test_invalid_json_is_quarantined_without_dropping_valid_components(tmp_path, monkeypatch):
    imported = source.import_delivery(bundle(tmp_path, monkeypatch, bad_json=True), tmp_path / 'data', version='fixture')
    report = source.audit_delivery(imported['manifest_path'], tmp_path / 'data', license_ref='synthetic_only')
    assert report['normalized_components'] == 4
    assert sum(sum(x['invalid_components'].values()) for x in report['archive_receipts']) == 2
    assert report['canonical_sft_records'] == 0


@pytest.mark.parametrize('unsafe,same_logical_name', [(True, False), (False, True)])
def test_inner_path_traversal_and_ambiguous_logical_members_are_rejected(tmp_path, monkeypatch, unsafe, same_logical_name):
    imported = source.import_delivery(bundle(tmp_path, monkeypatch, unsafe=unsafe, same_logical_name=same_logical_name), tmp_path / 'data', version='fixture')
    with pytest.raises(ValueError, match='UNSAFE_ARCHIVE_PATH|DUPLICATE_ARCHIVE_MEMBER'):
        source.audit_delivery(imported['manifest_path'], tmp_path / 'data', license_ref='synthetic_only')
    assert not (tmp_path / 'escape.json').exists()


def test_reference_candidates_are_reported_without_becoming_verified_links():
    q = question()
    q['metadata'] = {'answer_id': 'HC-A-1', 'original_parent_id': 'HC-Q-0'}
    row = source.component(q, '1.질문/HC-Q-1.json')
    assert row['references'] == {'metadata.answer_id': 'HC-A-1', 'metadata.original_parent_id': 'HC-Q-0'}
    assert row['content'] == {'question': q['question']}
    assert 'parent_group' not in row


def test_changed_inventory_or_imported_bytes_are_not_reused(tmp_path, monkeypatch):
    imported = source.import_delivery(bundle(tmp_path, monkeypatch), tmp_path / 'data', version='fixture')
    report = source.audit_delivery(imported['manifest_path'], tmp_path / 'data', license_ref='synthetic_only')
    database = Path(report['database']['path'])
    with database.open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='IMMUTABLE_COMPONENT_INVENTORY_CHANGED'):
        source.audit_delivery(imported['manifest_path'], tmp_path / 'data', license_ref='synthetic_only')


def test_resume_reuses_only_completed_archive_and_replaces_partial_derived_rows(tmp_path, monkeypatch):
    imported = source.import_delivery(bundle(tmp_path, monkeypatch), tmp_path / 'data', version='fixture')
    original_flush = source._flush
    calls = []
    def interrupted(connection, assets, components, hits):
        original_flush(connection, assets, components, hits)
        calls.append(True)
        if len(calls) == 2:
            raise RuntimeError('synthetic interruption after a committed partial archive')
    monkeypatch.setattr(source, '_flush', interrupted)
    with pytest.raises(RuntimeError, match='synthetic interruption'):
        source.audit_delivery(imported['manifest_path'], tmp_path / 'data', license_ref='synthetic_only')
    monkeypatch.setattr(source, '_flush', original_flush)
    report = source.audit_delivery(imported['manifest_path'], tmp_path / 'data', license_ref='synthetic_only')
    assert report['normalized_components'] == 4 and report['verified_asset_files'] == 8
    assert all(x['all_file_CRCs_verified'] for x in report['archive_receipts'])
