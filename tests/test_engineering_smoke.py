"""CPU fixtures verify real smoke evidence gates and subprocess orchestration."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mediflow_medvlm.engineering_smoke import _training_code_hash, run_smoke
from mediflow_medvlm.runtime import atomic_json, canonical_hash, file_hash
from mediflow_medvlm.training.config import load_frozen_data


@pytest.fixture
def smoke(tmp_path):
    train, val = tmp_path / 'train.jsonl', tmp_path / 'validation.jsonl'
    for path, split, count in ((train, 'train', 320), (val, 'validation', 1)):
        path.write_text(''.join(json.dumps({'sample_id': split + str(i), 'split': split,
                                          'source_id': 'fixture', 'source_revision': 'fixed'}) + '\n'
                                for i in range(count)))
    freeze = tmp_path / 'freeze.json'
    atomic_json(freeze, {'splits': {'text/train': {'sha256': file_hash(train), 'count': 320},
                                  'text/validation': {'sha256': file_hash(val), 'count': 1}}})
    config = {'run_id': 't1-preliminary-smoke', 'pair_id': 'engineering', 'experiment_id': 'engineering',
              'model_id': 'Qwen/Qwen3-1.7B', 'revision': 'a' * 40, 'mode': 'smoke',
              'training_schema_version': 'medvlm.training.v2', 'training_recipe': 'one_pass',
              'max_optimizer_steps': 20, 'train_samples_cap': 320, 'max_epochs': 1,
              'gradient_accumulation_steps': 16, 'device': 'cuda', 'train_file': str(train),
              'validation_file': str(val), 'freeze_manifest': str(freeze),
              'output_dir': str(tmp_path / 'run'), 'data_root': str(tmp_path),
              'resource_policy': {'budget_mode': 'unlimited'}}
    path = tmp_path / 'config.json'
    atomic_json(path, config)
    output = Path(config['output_dir'])
    _, data = load_frozen_data(config)
    identity = {'config_sha256': canonical_hash(config), 'data': data,
                'model_id': config['model_id'], 'revision': config['revision'],
                'training_code_sha256': _training_code_hash()}
    def make_training(*, status='SUCCEEDED', step=20, changed=True, excluded=0):
        checkpoint = output / 'checkpoints' / f'step-{step:06d}-fixture'
        checkpoint.mkdir(parents=True, exist_ok=True)
        for name in ('adapter_model.safetensors', 'training_state.pt'):
            (checkpoint / name).write_bytes(b'fixture-' + name.encode())
        artifacts = {name: file_hash(checkpoint / name) for name in ('adapter_model.safetensors', 'training_state.pt')}
        atomic_json(checkpoint / 'checkpoint.json', {'complete': True, 'identity': identity,
                    'optimizer_steps': step, 'samples_processed': step * 16,
                    'artifacts': {'files': artifacts}})
        manifest = {'run_id': config['run_id'], 'model_id': config['model_id'], 'revision': config['revision'],
                    'config_hash': canonical_hash(config), 'identity': identity, 'status': status,
                    'optimizer_steps': step, 'samples_processed': step * 16, 'adapter_changed': changed,
                    'training_selection': {'offered': 320, 'usable': 320 - excluded, 'excluded': excluded},
                    'latest_checkpoint': str(checkpoint), 'training_pid': 111111}
        atomic_json(output / 'manifest.json', manifest)
        atomic_json(output / 'latest.json', {'checkpoint': str(checkpoint)})
        atomic_json(output / 'config.json', config)
        return checkpoint
    calls = []
    def invoke(args, *, check):
        calls.append(args)
        stage = args[3]
        if stage == 'train':
            make_training()
        else:
            checkpoint = args[args.index('--checkpoint') + 1]
            atomic_json(output / 'reload.json', {'status': 'SUCCEEDED', 'run_id': config['run_id'],
                        'checkpoint': checkpoint, 'fresh_process_required': True,
                        'generated_tokens': 2, 'pid': 222222})
        return SimpleNamespace(returncode=0)
    return SimpleNamespace(config=config, path=path, output=output, make=make_training,
                           invoke=invoke, calls=calls, receipt=tmp_path / 'receipt.json', identity=identity)


def test_two_subprocesses_and_final_receipt_are_bound(smoke, monkeypatch):
    monkeypatch.setenv('MEDVLM_SUPERVISOR_CONFIG_HASH', file_hash(smoke.path))
    monkeypatch.setenv('MEDVLM_SUPERVISOR_DATA_HASH', file_hash(smoke.config['freeze_manifest']))
    monkeypatch.setenv('MEDVLM_SUPERVISOR_SOURCE_HASH', 'snapshot-hash')
    result = run_smoke(smoke.path, smoke.receipt, invoke=smoke.invoke)
    assert [args[3] for args in smoke.calls] == ['train', 'reload']
    assert result['source_hash'] == 'snapshot-hash'
    assert result['config_hash'] == file_hash(smoke.path)
    assert result['optimizer_steps'] == 20 and result['samples_processed'] == 320
    assert not result['usable_as_main_initialization']
    assert not result['v6_common_freeze_completed']
    assert all(file_hash(a['path']) == a['sha256'] for a in result['artifacts'])


def test_completed_training_is_verified_and_reload_retried_without_train(smoke):
    final = smoke.make()
    # Historical best/selection is not authoritative for final reload.
    atomic_json(smoke.output / 'selection.json', {'checkpoint': '/different/best'})
    result = run_smoke(smoke.path, smoke.receipt, invoke=smoke.invoke)
    assert [args[3] for args in smoke.calls] == ['reload']
    assert result['final_checkpoint'] == str(final)


def test_failed_run_resumes_only_verified_latest_optimizer_checkpoint(smoke):
    prior = smoke.make(status='FAILED_REPRODUCIBLE', step=10)
    run_smoke(smoke.path, smoke.receipt, invoke=smoke.invoke)
    assert smoke.calls[0][-2:] == ['--resume', str(prior)]


@pytest.mark.parametrize('field,value', [('adapter_changed', False), ('samples_processed', 319), ('optimizer_steps', 19)])
def test_incomplete_or_unchanged_success_is_rejected_before_any_subprocess(smoke, field, value):
    smoke.make()
    manifest = json.loads((smoke.output / 'manifest.json').read_text())
    manifest[field] = value
    atomic_json(smoke.output / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='20 updates'):
        run_smoke(smoke.path, smoke.receipt, invoke=smoke.invoke)
    assert smoke.calls == []


def test_exclusions_are_not_silently_accepted(smoke):
    smoke.make(excluded=1)
    with pytest.raises(ValueError, match='without exclusions'):
        run_smoke(smoke.path, smoke.receipt, invoke=smoke.invoke)
    assert not smoke.receipt.exists()


def test_changed_checkpoint_never_resumes_or_reloads(smoke):
    checkpoint = smoke.make(status='FAILED_REPRODUCIBLE', step=10)
    (checkpoint / 'training_state.pt').write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='artifact hash mismatch'):
        run_smoke(smoke.path, smoke.receipt, invoke=smoke.invoke)
    assert smoke.calls == []


def test_supervisor_config_identity_and_prior_config_immutable(smoke, monkeypatch):
    monkeypatch.setenv('MEDVLM_SUPERVISOR_CONFIG_HASH', 'changed')
    with pytest.raises(ValueError, match='supervisor identity'):
        run_smoke(smoke.path, smoke.receipt, invoke=smoke.invoke)
    monkeypatch.delenv('MEDVLM_SUPERVISOR_CONFIG_HASH')
    smoke.make()
    atomic_json(smoke.output / 'config.json', {**smoke.config, 'learning_rate': 123})
    with pytest.raises(ValueError, match='immutable'):
        run_smoke(smoke.path, smoke.receipt, invoke=smoke.invoke)


def test_reload_must_name_final_and_be_a_fresh_process(smoke):
    smoke.make()
    def bad_reload(args, *, check):
        smoke.invoke(args, check=check)
        path = smoke.output / 'reload.json'
        value = json.loads(path.read_text())
        value['pid'] = 111111
        atomic_json(path, value)
        return SimpleNamespace(returncode=0)
    with pytest.raises(ValueError, match='fresh-process'):
        run_smoke(smoke.path, smoke.receipt, invoke=bad_reload)
    assert not smoke.receipt.exists()
