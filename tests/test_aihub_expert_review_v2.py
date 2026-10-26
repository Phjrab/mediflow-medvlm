"""AI-Hub review is fixed, model-blinded, local, hash-verified and unscored."""
import csv
import json
import stat
from pathlib import Path

import pytest

from mediflow_medvlm.evaluation.expert_review import REVIEW_FIELDS
from mediflow_medvlm.evaluation.review_v2 import prepare_aihub_expert_review
from mediflow_medvlm.evaluation.runner import freeze_protocol
from mediflow_medvlm.evaluation.v2 import V2_RENDERER, promote_protocol
from mediflow_medvlm.runtime import atomic_json, canonical_hash, directory_digest, file_hash


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')


def receipt(folder, manifest):
    manifest['artifact_hashes'] = {name: file_hash(folder / name) for name in ('predictions.jsonl', 'metrics.json')
                                 if (folder / name).exists()}
    atomic_json(folder / 'run_manifest.json', manifest)


def fixture(tmp_path, short=10, long=10):
    root = tmp_path / 'data'; (root / 'v2/text').mkdir(parents=True)
    rows = [{'sample_id': f'fixture-{qtype}-{i}', 'source_id': 'aihub_71875', 'task_type': 'medical_qa',
             'split': 'test', 'language': 'ko', 'input': {'question': f'=question {qtype}-{i}; 1.5 mg, not 15 mg', 'images': []},
             'supervision': {'answer': '+reference answer'}, 'metadata': {'q_type': qtype}}
            for qtype, count in [('2', short), ('3', long)] for i in range(count)]
    rows += [{'sample_id': 'excluded-mcq', 'source_id': 'aihub_71875', 'task_type': 'medical_mcq',
              'split': 'test', 'input': {'question': 'MCQ', 'options': {'A': 'one'}, 'images': []},
              'supervision': {'answer': 'one', 'option_id': 'A'}, 'metadata': {'q_type': '1'}},
             {'sample_id': 'excluded-source', 'source_id': 'medquad', 'task_type': 'medical_qa',
              'split': 'test', 'input': {'question': 'Other source', 'images': []},
              'supervision': {'answer': 'answer'}, 'metadata': {'q_type': '2'}}]
    source = root / 'test.jsonl'; write_rows(source, rows)
    frozen = freeze_protocol([source], root / 'v2/text/selection.json', cap=0)
    protocol_path = root / 'v2/text/protocol.json'
    protocol = promote_protocol(frozen, protocol_path, prompt_policy='native-task-instructions.v1')
    adapter = root / 'v2/runs/tuned/checkpoints/selected'; adapter.mkdir(parents=True)
    (adapter / 'adapter.safetensors').write_bytes(b'synthetic artifact, no model loaded')
    native_files = {'config.json': {'sha256': 'a' * 64, 'bytes': 10}, 'model.safetensors': {'sha256': 'b' * 64, 'bytes': 20}}
    paths = []
    for role in ('base', 'tuned'):
        folder = root / 'v2/text/evaluations' / role; folder.mkdir(parents=True)
        manifest = {'run_id': role, 'identity_hash': role + '-identity', 'pair_id': 'fixture-pair',
                    'protocol_hash': protocol['protocol_hash'], 'model_id': 'fixture/model', 'model_revision': 'c' * 40,
                    'config_hash': 'config', 'precision': 'none', 'renderer_version': V2_RENDERER, 'image_condition': 'image',
                    'generation': {'do_sample': False, 'budgets': protocol['max_new_tokens']},
                    'status': 'SUCCEEDED', 'scheduled': len(rows),
                    'adapter_path': str(adapter) if role == 'tuned' else None,
                    'model': {'native_files': {'sha256': canonical_hash(native_files), 'files': native_files},
                              'chat_template_sha256': 'd' * 64, 'dtype': 'bfloat16'}}
        if role == 'tuned': manifest['adapter_digest'] = directory_digest(adapter)['sha256']
        predictions = [{'sample_id': row['sample_id'], 'input_hash': canonical_hash(row['input']),
                        'run_id': role, 'pair_id': 'fixture-pair', 'model_identity': manifest['identity_hash'],
                        'status': 'answered', 'raw_output': '@synthetic answer', 'parsed_answer': '@synthetic answer'} for row in rows]
        path = folder / 'predictions.jsonl'; write_rows(path, predictions)
        atomic_json(folder / 'metrics.json', {'fixture': True}); receipt(folder, manifest)
        paths.append(path)
    return root, protocol_path, paths, rows


def packet_table(path):
    with (path / 'review.csv').open(encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def test_fixed_strata_blinding_blank_rubric_and_private_permissions(tmp_path):
    root, protocol, paths, rows = fixture(tmp_path)
    inputs = [protocol, *paths, *(path.parent / 'run_manifest.json' for path in paths)]
    before = {path: file_hash(path) for path in inputs}
    output = root / 'v2/review/fixture-pair'
    result = prepare_aihub_expert_review(protocol, *paths, output)
    table = packet_table(output)
    keys = json.loads((output / 'private_key.json').read_text())['items']
    expected = [row['sample_id'] for qtype in ('2', '3')
                for row in sorted((r for r in rows if r['source_id'] == 'aihub_71875' and r['metadata']['q_type'] == qtype),
                                  key=lambda r: canonical_hash([20261023, r['sample_id']]))[:8]]
    assert [row['sample_id'] for row in keys] == expected
    assert result['selected_count'] == 16 and result['selected_by_q_type'] == {'2': 8, '3': 8}
    assert result['eligible_by_q_type'] == {'2': 10, '3': 10}
    assert result['status'] == 'PREPARED_NOT_RUN' and result['human_review'] == 'not_performed'
    assert result['human_review_count'] == 0 and result['clinical_scores'] is None and result['clinical_accuracy'] is None
    assert result['review_scores_generated'] is False
    assert all(row[field] == '' for row in table for field in REVIEW_FIELDS)
    assert all(not any(field in row for field in ('run_id', 'sample_id', 'model_id', 'A_role', 'B_role')) for row in table)
    assert {key['A_role'] for key in keys} == {'baseline', 'tuned'}
    assert all(row['question'].startswith("'=question") and row['reference_answer'].startswith("'+")
               and row['answer_A'].startswith("'@") and row['answer_B'].startswith("'@") for row in table)
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in output.iterdir())
    assert all(file_hash(path) == digest for path, digest in before.items())
    assert all(file_hash(output / name) == digest for name, digest in result['files_sha256'].items())
    other = root / 'v2/review/copy'; prepare_aihub_expert_review(protocol, *paths, other)
    assert (output / 'review.csv').read_bytes() == (other / 'review.csv').read_bytes()
    with pytest.raises(ValueError, match='overwrite'): prepare_aihub_expert_review(protocol, *paths, output)


def test_failed_and_missing_answers_are_retained_without_selection_changes(tmp_path):
    root, protocol, paths, _ = fixture(tmp_path)
    first = root / 'v2/review/first'; prepare_aihub_expert_review(protocol, *paths, first)
    keys = json.loads((first / 'private_key.json').read_text())['items']
    failed_id, missing_id = keys[0]['sample_id'], keys[1]['sample_id']
    predictions = [json.loads(line) for line in paths[1].read_text().splitlines()]
    for row in predictions:
        if row['sample_id'] == failed_id: row.update(status='oom', raw_output='', parsed_answer=None)
    predictions = [row for row in predictions if row['sample_id'] != missing_id]
    write_rows(paths[1], predictions)
    manifest = json.loads((paths[1].parent / 'run_manifest.json').read_text()); manifest['status'] = 'FAILED_REPRODUCIBLE'
    receipt(paths[1].parent, manifest)
    second = root / 'v2/review/second'; prepare_aihub_expert_review(protocol, *paths, second)
    assert json.loads((second / 'private_key.json').read_text())['items'] == keys
    table = packet_table(second)
    statuses = [row[field] for row in table for field in ('status_A', 'status_B')]
    assert statuses.count('oom') == 1 and statuses.count('not_attempted') == 1
    assert sum(row[field] == '[NO MODEL ANSWER]' for row in table for field in ('answer_A', 'answer_B')) == 2
    assert all(row[field] == '' for row in table for field in REVIEW_FIELDS)


@pytest.mark.parametrize('mutation', ['native_files', 'chat_template_sha256', 'dtype', 'renderer', 'pair',
                                    'prediction_hash', 'metrics_hash', 'adapter_hash', 'baseline_adapter', 'running',
                                    'prediction_identity', 'missing_receipts'])
def test_unmatched_or_modified_evidence_is_rejected_before_output(tmp_path, mutation):
    root, protocol, paths, _ = fixture(tmp_path)
    manifest_path = paths[1].parent / 'run_manifest.json'; manifest = json.loads(manifest_path.read_text())
    if mutation == 'native_files': manifest['model']['native_files']['sha256'] = 'changed'
    elif mutation in {'chat_template_sha256', 'dtype'}: manifest['model'][mutation] = 'changed'
    elif mutation == 'renderer': manifest['renderer_version'] = 'another-renderer'
    elif mutation == 'pair': manifest['pair_id'] = 'another-pair'
    elif mutation == 'prediction_hash': paths[1].write_text(paths[1].read_text() + '\n')
    elif mutation == 'metrics_hash': (paths[1].parent / 'metrics.json').write_text('{}')
    elif mutation == 'adapter_hash': (Path(manifest['adapter_path']) / 'adapter.safetensors').write_bytes(b'changed')
    elif mutation == 'baseline_adapter':
        base = json.loads((paths[0].parent / 'run_manifest.json').read_text()); base['adapter_path'] = manifest['adapter_path']
        atomic_json(paths[0].parent / 'run_manifest.json', base)
    elif mutation == 'running': manifest['status'] = 'RUNNING'
    elif mutation == 'prediction_identity':
        rows = [json.loads(line) for line in paths[1].read_text().splitlines()]; rows[0]['model_identity'] = 'another-run'
        write_rows(paths[1], rows); manifest['artifact_hashes']['predictions.jsonl'] = file_hash(paths[1])
    elif mutation == 'missing_receipts': manifest.pop('artifact_hashes')
    atomic_json(manifest_path, manifest)
    output = root / 'v2/review/rejected'
    with pytest.raises(ValueError): prepare_aihub_expert_review(protocol, *paths, output)
    assert not output.exists()


@pytest.mark.parametrize('cap', [0, 8, 15, 17, True, 16.0])
def test_review_cap_is_exactly_sixteen(tmp_path, cap):
    with pytest.raises(ValueError, match='fixed at 16'):
        prepare_aihub_expert_review('absent', 'absent', 'absent', tmp_path, cap=cap)


def test_incomplete_stratum_does_not_silently_change_the_fixed_packet(tmp_path):
    root, protocol, paths, _ = fixture(tmp_path, short=7)
    with pytest.raises(ValueError, match='at least 8'):
        prepare_aihub_expert_review(protocol, *paths, root / 'v2/review/incomplete')


def test_output_must_stay_outside_git_and_under_the_same_data_review_root(tmp_path):
    root, protocol, paths, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match='same data_root'):
        prepare_aihub_expert_review(protocol, *paths, tmp_path / 'other/v2/review/packet')
    repo = root / 'v2/review/repository'; repo.mkdir(parents=True)
    (repo / '.git').write_text('gitdir: elsewhere')
    with pytest.raises(ValueError, match='outside Git'):
        prepare_aihub_expert_review(protocol, *paths, repo / 'packet')
    link = root / 'v2/review/link'; link.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        prepare_aihub_expert_review(protocol, *paths, link)
