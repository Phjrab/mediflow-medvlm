"""Reporting must not turn pending/altered evidence into completed public scores."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest
from mediflow_medvlm.runtime import canonical_hash, file_hash
from mediflow_medvlm.evaluation.metrics import score_rows
from mediflow_medvlm.evaluation.v2 import normalize_vqa

SCRIPT = Path(__file__).parents[1] / 'scripts/report_v2.py'
sys.path.insert(0, str(SCRIPT.parent))
spec = importlib.util.spec_from_file_location('report_v2', SCRIPT)
report = importlib.util.module_from_spec(spec); spec.loader.exec_module(report)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False))


def sample(sid='s', source='fixture', task='medical_qa', answer='answer'):
    return {'sample_id': sid, 'source_id': source, 'task_type': task, 'language': 'en',
            'input': {'question': 'PRIVATE_QUESTION'}, 'supervision': {'answer': answer},
            'group_ids': {'document': sid}, 'metadata': {}}


def evaluation_fixture(root):
    config = {'model_id': 'fixture/model', 'revision': 'revision'}
    cp = root / 'config.json'; dump(cp, config)
    s = sample(); data = root / 'data.jsonl'; data.write_text(json.dumps(s) + '\n')
    protocol = {'files': {str(data): file_hash(data)}, 'sample_ids': ['s'],
                'input_digests': {'s': canonical_hash(s['input'])}, 'image_files': {'s': []},
                'reference_digest': canonical_hash([s['supervision']])}
    protocol['protocol_hash'] = canonical_hash(protocol)
    pp = root / 'protocol.json'; dump(pp, protocol)
    origin = {'model_id': config['model_id'], 'revision': 'revision',
              'model': {'native_files': {'sha256': 'native_digest'}, 'chat_template_sha256': 'template'}}
    dump(root / 'runs/fixture-v1-20261023/manifest.json', origin)
    folder = root / 'v2/run'; identity = {'config_hash': canonical_hash(config), 'protocol_hash': protocol['protocol_hash'],
        'model_id': config['model_id'], 'model_revision': 'revision', 'adapter_path': None,
        'pair_id': 'pair', 'renderer_version': 'renderer', 'image_condition': 'image'}
    manifest = {**identity, 'run_id': 'run', 'identity_hash': canonical_hash(identity), 'model': origin['model'],
                'status': 'SUCCEEDED', 'scheduled': 1, 'predictions': 1, 'precision': 'nf4', 'generation': {'do_sample': False}}
    prediction = {'sample_id': 's', 'input_hash': canonical_hash(s['input']), 'run_id': 'run', 'pair_id': 'pair',
        'model_identity': manifest['identity_hash'], 'status': 'answered', 'parsed_answer': 'answer', 'raw_output': 'answer',
        'vision_ingested': False, 'pixel_tensor_keys': []}
    folder.mkdir(parents=True)
    (folder / 'predictions.jsonl').write_text(json.dumps(prediction) + '\n')
    dump(folder / 'metrics.json', {'scores': score_rows([s], [prediction], vqa_normalizer=normalize_vqa)})
    manifest['artifact_hashes'] = {n: file_hash(folder / n) for n in ('predictions.jsonl', 'metrics.json')}
    dump(folder / 'run_manifest.json', manifest)
    return folder, pp, cp


def test_verified_completed_artifact_and_tamper_rejection(tmp_path):
    folder, protocol, config = evaluation_fixture(tmp_path)
    evidence = report.Evidence(tmp_path)
    good = evidence.evaluation('fixture', 'base', 'text', folder, protocol, config, expected_count=1)
    assert good['verified'] and good['metrics']['cohorts'][0]['end_to_end_lexical_exact_match'] == 1
    report.validate_public(good)
    assert 'PRIVATE_QUESTION' not in json.dumps(good)
    # Preserve IDs and count; changing the actual prediction bytes must still fail.
    with (folder / 'predictions.jsonl').open('a') as stream: stream.write('\n')
    altered = report.Evidence(tmp_path).evaluation('fixture', 'base', 'text', folder, protocol, config, expected_count=1)
    assert altered['status'] == 'FAILED_VERIFICATION' and not altered['verified'] and altered['metrics'] is None


def test_running_prediction_is_not_scored(tmp_path):
    folder, protocol, config = evaluation_fixture(tmp_path)
    manifest = report.read(folder / 'run_manifest.json'); manifest['status'] = 'RUNNING'; dump(folder / 'run_manifest.json', manifest)
    (folder / 'predictions.jsonl').write_text('half a json line')
    result = report.Evidence(tmp_path).evaluation('fixture', 'base', 'text', folder, protocol, config, expected_count=1)
    assert result['status'] == 'RUNNING' and not result['verified'] and result['metrics'] is None


def test_text_run_rejects_hash_consistent_pixel_input(tmp_path):
    folder, protocol, config = evaluation_fixture(tmp_path)
    path = folder / 'predictions.jsonl'
    prediction = json.loads(path.read_text())
    prediction.update(vision_ingested=True, pixel_tensor_keys=['pixel_values'])
    path.write_text(json.dumps(prediction) + '\n')
    manifest = report.read(folder / 'run_manifest.json')
    manifest['artifact_hashes']['predictions.jsonl'] = file_hash(path)
    dump(folder / 'run_manifest.json', manifest)
    result = report.Evidence(tmp_path).evaluation('fixture', 'base', 'text', folder, protocol, config, expected_count=1)
    assert result['status'] == 'FAILED_VERIFICATION' and not result['verified']


def test_vqa_strict_normalized_and_failed_denominators():
    samples = [sample('a', 'slake', 'medical_vqa', 'yes'), sample('b', 'slake', 'medical_vqa', 'no')]
    for s in samples: s['metadata']['answer_type'] = 'CLOSED'
    predictions = [{'sample_id': 'a', 'status': 'answered', 'parsed_answer': 'Yes.', 'raw_output': 'Yes.', 'finish_reason': 'length'},
                   {'sample_id': 'b', 'status': 'oom', 'parsed_answer': None, 'attempted': True}]
    bundle, _, _ = report.metric_bundle(samples, predictions)
    metric = bundle['vqa'][0]
    assert metric['scheduled'] == 2 and metric['strict_lexical_em'] == 0
    assert metric['v2_normalized_lexical_em'] == .5 and metric['truncated_count'] == 1
    assert bundle['cohorts'][0]['coverage'] == .5 and bundle['cohorts'][0]['technical_failure_rate'] == .5


def test_aihub_qtypes_stay_separate_and_do_not_export_gold():
    samples = [sample(str(i), 'aihub_71875', 'medical_mcq' if i == 1 else 'medical_qa', 'SECRET_GOLD') for i in (1,2,3)]
    samples[0]['supervision']['option_id'] = 'A'
    for i,s in enumerate(samples, 1): s['metadata']['q_type'] = str(i)
    predictions = [{'sample_id': s['sample_id'], 'status': 'invalid_format', 'parsed_answer': None} for s in samples]
    bundle, _, _ = report.metric_bundle(samples, predictions)
    assert [x['type'] for x in bundle['aihub_q_types']] == ['MCQ','short','long']
    report.validate_public(bundle)
    assert 'SECRET_GOLD' not in json.dumps(bundle)


def test_public_boundary_refuses_raw_paths_and_v1_output(tmp_path):
    for private in ({'raw_output': 'anything'}, {'nested': '/home/alice/private'}, {'token': 'Bearer abc'}):
        with pytest.raises(ValueError): report.validate_public(private)
    with pytest.raises(ValueError, match='OUTPUT_MUST_BE_V2_DIRECTORY'):
        report.write_reports({}, {}, {}, tmp_path / 'public_summary')


def test_incomplete_reload_is_pending_not_verified(tmp_path):
    name = 'fixture-ko-smoke-v2-20261026'
    folder = tmp_path / 'v2/text/runs' / name
    dump(folder / 'manifest.json', {'status': 'SUCCEEDED', 'optimizer_steps': 20, 'reload_verified': False})
    result = report.Evidence(tmp_path).training('fixture','ko','smoke',{})
    assert result['status'] == 'TRAINED_AWAITING_RELOAD' and not result['verified']


def test_training_reload_receipts_and_data_tamper(tmp_path):
    from mediflow_medvlm.runtime import directory_digest
    run_id = 'fixture-ko-full-v2-20261026'
    folder = tmp_path / 'v2/text/runs' / run_id
    training = tmp_path / 'train.jsonl'; training.write_text('{}\n')
    validation = tmp_path / 'validation.jsonl'; validation.write_text('{}\n')
    freeze = tmp_path / 'freeze.json'; dump(freeze, {'fixture': True})
    config = {'model_id': 'fixture/model', 'revision': 'rev', 'train_file': str(training),
              'validation_file': str(validation), 'freeze_manifest': str(freeze)}
    dump(folder / 'config.json', config)
    identity = {'config_sha256': canonical_hash(config)}
    checkpoint = folder / 'checkpoints/step-000250'
    dump(checkpoint / 'checkpoint.json', {'identity': identity, 'optimizer_steps': 250})
    (checkpoint / 'adapter.safetensors').write_bytes(b'fixture adapter')
    digest = directory_digest(checkpoint)['sha256']
    manifest = {'status': 'SUCCEEDED', 'config_hash': canonical_hash(config), 'model_id': 'fixture/model', 'revision': 'rev',
        'data': {'freeze_manifest_sha256': file_hash(freeze), 'train': {'sha256': file_hash(training)}, 'validation': {'sha256': file_hash(validation)}},
        'identity': identity, 'reload_verified': True, 'test_used_for_selection': False,
        'model': {'native_files': {'sha256': 'native'}}}
    dump(folder / 'manifest.json', manifest)
    dump(folder / 'selection.json', {'run_id': run_id, 'identity': identity, 'checkpoint': str(checkpoint),
        'adapter': {'digest': digest}, 'test_used': False, 'reload_verified': True})
    dump(folder / 'reload.json', {'status': 'SUCCEEDED', 'fresh_process_required': True, 'checkpoint': str(checkpoint),
        'model': {'native_files': {'sha256': 'native'}}})
    result = report.Evidence(tmp_path).training('fixture','ko','full',{})
    assert result['verified'] and result['reload_verified']
    training.write_text('{"changed":true}\n')
    result = report.Evidence(tmp_path).training('fixture','ko','full',{})
    assert result['status'] == 'FAILED_VERIFICATION' and not result['verified']


def test_review_pending_four_does_not_claim_human_review(tmp_path):
    result = report.review_summary(tmp_path, report.Evidence(tmp_path))
    assert result['planned_packets'] == 4 and result['verified_prepared_packets'] == 0
    assert result['prepared_review_pairs'] == result['human_review_count'] == 0
    assert result['human_review'] == 'not_performed'


def test_review_packet_receipts_blank_scores_and_public_counts(tmp_path):
    helper_spec = importlib.util.spec_from_file_location('review_fixture_helpers', Path(__file__).with_name('test_aihub_expert_review_v2.py'))
    helper = importlib.util.module_from_spec(helper_spec); helper_spec.loader.exec_module(helper)
    root, protocol_path, paths, _ = helper.fixture(tmp_path)
    protocol = root / 'v2/text/text-test.json'; protocol_path.rename(protocol)
    adjusted = []
    for role, old in zip(('base','ko'), paths):
        folder = old.parent.with_name(f'qwen3_17b-{role}-text-v2-20261026'); old.parent.rename(folder)
        path = folder / 'predictions.jsonl'; manifest = report.read(folder / 'run_manifest.json')
        manifest['run_id'] = folder.name
        predictions = report.jsonl(path)
        for p in predictions: p['run_id'] = folder.name
        helper.write_rows(path, predictions); helper.receipt(folder, manifest)
        adjusted.append(path)
    native = report.read(adjusted[0].parent / 'run_manifest.json')['model']['native_files']['sha256']
    dump(root / 'runs/qwen3_17b-v1-20261023/manifest.json', {'model':{'native_files':{'sha256':native}}})
    folder = root / 'v2/review/qwen3_17b-base-vs-ko'
    helper.prepare_aihub_expert_review(protocol, *adjusted, folder)
    summary = report.review_summary(root, report.Evidence(root))
    assert summary['verified_prepared_packets'] == 1 and summary['prepared_review_pairs'] == 16
    assert summary['human_review_count'] == 0 and summary['clinical_accuracy'] is None
    report.validate_public(summary)
    assert all(text not in json.dumps(summary) for text in ('synthetic answer', 'question 2-', 'reference answer', str(root)))
    (folder / 'instructions.txt').write_text('modified after receipt')
    summary = report.review_summary(root, report.Evidence(root))
    assert summary['verified_prepared_packets'] == 0 and summary['packets'][0]['status'] == 'FAILED_VERIFICATION'
