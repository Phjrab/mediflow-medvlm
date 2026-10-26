"""Campaign resumes must retain complete, unmodified validation evidence."""
import copy
import base64
import importlib.util
import json
from pathlib import Path

import pytest

from mediflow_medvlm.evaluation.runner import freeze_protocol
from mediflow_medvlm.evaluation.v2 import VQA_PROMPT, promote_protocol
from mediflow_medvlm.runtime import atomic_json, canonical_hash, file_hash, ResourcePolicy


SPEC = importlib.util.spec_from_file_location('run_vqa_v2', Path(__file__).parents[1] / 'scripts/run_vqa_v2.py')
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def fixture(tmp_path, budget=128):
    plan = {'models': ['first', 'second'], 'validation_samples': 32, 'seed': 20261023,
            'prompt_policy': VQA_PROMPT}
    image = tmp_path / 'fixture.png'
    image.write_bytes(base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aV7cAAAAASUVORK5CYII='))
    rows = [{'sample_id': str(i), 'source_id': 'fixture', 'task_type': 'medical_vqa',
             'split': 'validation', 'language': 'en', 'input': {'question': str(i), 'images': [str(image)]},
             'supervision': {'answer': 'yes'}} for i in range(32)]
    source = tmp_path / 'validation.jsonl'; write_rows(source, rows)
    frozen = freeze_protocol([source], tmp_path / 'selection.json', split='validation', cap=32)
    paths = {}; records = []
    for limit in (128, 256) if budget == 256 else (128,):
        path = tmp_path / f'protocol-{limit}.json'; paths[limit] = path
        protocol = promote_protocol(frozen, path, prompt_policy=VQA_PROMPT, max_vqa_tokens=limit,
                                    previously_observed_test=False)
        for model in plan['models']:
            for mode in ('base', 'tuned'):
                run = tmp_path / 'runs' / f'{model}-validation{limit}-{mode}-v2-20261026'; run.mkdir(parents=True)
                manifest = {'run_id': run.name, 'pair_id': model, 'identity_hash': run.name,
                            'status': 'SUCCEEDED', 'scheduled': 32, 'protocol_hash': protocol['protocol_hash'],
                            'model': {'native_files': {'sha256': 'base'}}}
                predictions = [{'sample_id': sid, 'run_id': run.name, 'pair_id': model,
                                'model_identity': run.name, 'input_hash': protocol['input_digests'][sid],
                                'status': 'answered', 'finish_reason': 'stop'} for sid in protocol['sample_ids']]
                truncated = budget == 256 and limit == 128 and model == 'first' and mode == 'base'
                if truncated: predictions[0]['finish_reason'] = 'length'
                write_rows(run / 'predictions.jsonl', predictions); atomic_json(run / 'metrics.json', {'fixture': True})
                manifest['artifact_hashes'] = {name: file_hash(run / name) for name in ('predictions.jsonl', 'metrics.json')}
                atomic_json(run / 'run_manifest.json', manifest)
                records.append({'model': model, 'mode': mode, 'status': 'SUCCEEDED', 'predictions': 32,
                                'technical_errors': 0, 'length_count': int(truncated), 'run': str(run),
                                'manifest_sha256': file_hash(run / 'run_manifest.json')})
    decision = {'plan_sha256': canonical_hash(plan), 'chosen_max_vqa_tokens': budget,
                'observations': records, 'ready_models': plan['models'], 'test_used_for_selection': False}
    return plan, decision, paths


def verify(tmp_path, plan, decision, paths):
    return campaign.verify_validation_decision(plan, decision, paths, run_root=tmp_path / 'runs')


@pytest.mark.parametrize('budget', [128, 256])
def test_complete_decision_and_legacy_missing_redundant_fields_are_accepted_without_mutation(tmp_path, budget):
    plan, decision, paths = fixture(tmp_path, budget)
    assert verify(tmp_path, plan, decision, paths) == plan['models']
    decision.pop('ready_models')
    for record in decision['observations']:
        record.pop('technical_errors'); record.pop('predictions')
    original = copy.deepcopy(decision)
    assert verify(tmp_path, plan, decision, paths) == plan['models']
    assert decision == original


@pytest.mark.parametrize('mutation', ['plan', 'missing', 'duplicate', 'model', 'length', 'budget', 'readiness', 'test'])
def test_changed_decision_cannot_enable_test(tmp_path, mutation):
    plan, decision, paths = fixture(tmp_path)
    if mutation == 'plan': decision['plan_sha256'] = 'changed'
    elif mutation == 'missing': decision['observations'] = decision['observations'][:2]
    elif mutation == 'duplicate': decision['observations'][1] = decision['observations'][0]
    elif mutation == 'model': decision['observations'][0]['model'] = 'second'
    elif mutation == 'length': decision['observations'][0]['length_count'] = 1
    elif mutation == 'budget': decision['chosen_max_vqa_tokens'] = 256
    elif mutation == 'readiness': decision['ready_models'] = ['first']
    elif mutation == 'test': decision['test_used_for_selection'] = True
    with pytest.raises(ValueError): verify(tmp_path, plan, decision, paths)


@pytest.mark.parametrize('artifact', ['run_manifest.json', 'predictions.jsonl', 'metrics.json'])
def test_modified_validation_artifact_is_rejected(tmp_path, artifact):
    plan, decision, paths = fixture(tmp_path)
    path = Path(decision['observations'][0]['run']) / artifact
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='changed'): verify(tmp_path, plan, decision, paths)


@pytest.mark.parametrize('mutation', ['missing_prediction', 'technical_error', 'wrong_prediction_identity', 'wrong_protocol'])
def test_valid_receipts_cannot_hide_incomplete_or_incorrect_success(tmp_path, mutation):
    plan, decision, paths = fixture(tmp_path)
    record = decision['observations'][0]; run = Path(record['run'])
    manifest = campaign.read(run / 'run_manifest.json')
    predictions = [json.loads(line) for line in (run / 'predictions.jsonl').read_text().splitlines()]
    if mutation == 'missing_prediction': predictions.pop(); record['predictions'] = 31
    elif mutation == 'technical_error': predictions[0]['status'] = 'oom'; record['technical_errors'] = 1
    elif mutation == 'wrong_prediction_identity': predictions[0]['model_identity'] = 'another-run'
    elif mutation == 'wrong_protocol': manifest['protocol_hash'] = 'another-protocol'
    write_rows(run / 'predictions.jsonl', predictions)
    manifest['artifact_hashes']['predictions.jsonl'] = file_hash(run / 'predictions.jsonl')
    atomic_json(run / 'run_manifest.json', manifest); record['manifest_sha256'] = file_hash(run / 'run_manifest.json')
    with pytest.raises(ValueError): verify(tmp_path, plan, decision, paths)


def test_failed_model_does_not_block_other_verified_pair(tmp_path):
    plan, decision, paths = fixture(tmp_path)
    record = decision['observations'][0]
    record['status'] = 'FAILED_REPRODUCIBLE'
    decision['ready_models'] = ['second']
    assert verify(tmp_path, plan, decision, paths) == ['second']


def test_budget_rule_cannot_select_256_without_128_length_evidence(tmp_path):
    plan, decision, paths = fixture(tmp_path, 256)
    record = decision['observations'][0]; run = Path(record['run'])
    predictions = [json.loads(line) for line in (run / 'predictions.jsonl').read_text().splitlines()]
    predictions[0]['finish_reason'] = 'stop'; write_rows(run / 'predictions.jsonl', predictions)
    manifest = campaign.read(run / 'run_manifest.json')
    manifest['artifact_hashes']['predictions.jsonl'] = file_hash(run / 'predictions.jsonl')
    atomic_json(run / 'run_manifest.json', manifest)
    record.update(length_count=0, manifest_sha256=file_hash(run / 'run_manifest.json'))
    with pytest.raises(ValueError, match='frozen rule'): verify(tmp_path, plan, decision, paths)


def test_successful_comparison_retry_clears_only_current_error_and_retains_history():
    state = {'comparison_errors': {'first': 'ValueError', 'second': 'OSError'}}
    campaign.resolve_comparison_error(state, 'first')
    assert state['comparison_errors'] == {'second': 'OSError'}
    assert state['resolved_comparison_errors'][0]['key'] == 'first'
    assert state['resolved_comparison_errors'][0]['error'] == 'ValueError'
    campaign.resolve_comparison_error(state, 'first')
    assert len(state['resolved_comparison_errors']) == 1


def test_unlimited_and_finite_job_timeouts_use_the_stricter_explicit_limit():
    unlimited = ResourcePolicy()
    assert campaign.job_timeout(unlimited, None, now=10**9) is None
    finite = ResourcePolicy(budget_mode='finite', total_gpu_hours=48, per_model_gpu_hours=12,
                            job_timeout_seconds=500, campaign_wall_seconds=1000)
    assert campaign.job_timeout(finite, 1000, now=100) == 500
    assert campaign.job_timeout(finite, 1000, now=900) == 100
    assert campaign.job_timeout(finite, 1000, now=1000) == 0
    assert finite.deadline(100, prior_seconds=900, scope='campaign') == 200


def test_expired_campaign_records_every_cell_without_launching_a_child(tmp_path, monkeypatch):
    fixture(tmp_path)
    original = tmp_path / 'manifests/evaluation/slake-validation-v1.json'
    original.parent.mkdir(parents=True)
    atomic_json(original, {'files': {str(tmp_path / 'validation.jsonl'): file_hash(tmp_path / 'validation.jsonl')}})
    finite = ResourcePolicy(budget_mode='finite', total_gpu_hours=48, per_model_gpu_hours=12,
                            campaign_wall_seconds=1)
    monkeypatch.setattr(campaign, 'resolve_resource_policy', lambda: finite)
    ticks = iter(range(0, 10000, 10))
    monkeypatch.setattr(campaign.time, 'monotonic', lambda: next(ticks))
    def forbidden(*args, **kwargs):
        raise AssertionError('expired campaign launched a child')
    monkeypatch.setattr(campaign.subprocess, 'Popen', forbidden)
    monkeypatch.setattr(campaign.sys, 'argv', ['run_vqa_v2.py', '--data-root', str(tmp_path), '--validation-only'])
    assert campaign.main() == 0
    state = campaign.read(tmp_path / 'v2/vqa/state.json')
    assert len(state['runs']) == 4
    assert all(row['status'] == 'SKIPPED_BUDGET' and row['reason_code'] == 'CAMPAIGN_WALL_SECONDS'
               for row in state['runs'].values())
    assert state['campaign_wall_seconds'] > finite.campaign_wall_seconds
