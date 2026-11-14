"""Pure CPU artifact fixtures, never training, CUDA, or campaign transitions."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from mediflow_medvlm import campaign_preflight_gate as gate
from mediflow_medvlm import campaign_v6 as plan
from mediflow_medvlm import campaign_supervisor as supervisor
from mediflow_medvlm.training.v6_native import verify_native_no_thinking_output


def write(path, value):
    gate.atomic_json(path, value)
    return path


def row(manifest, job_id):
    return next(job for job in manifest['jobs'] if job['job_id'] == job_id)


def config_path(job):
    return Path(job['argv'][job['argv'].index('--config') + 1])


def publish_phase(job, config, *, cpu=False):
    output = Path(config['output_dir'])
    marker_path = output / 'completion.json'
    old = json.loads(marker_path.read_text()) if marker_path.exists() else {}
    marker = {**old, 'status': 'SUCCEEDED', **{key: config[key] for key in gate.IDENTITY_KEYS},
        'cell_id': config['cell_id'], 'scope': 'TRAIN_PREFLIGHT_ELIGIBILITY_ONLY' if cpu else gate.gpu.SCOPE,
        'common_train_freeze_decided': False, 'campaign_terminal': False}
    if cpu:
        summary = json.loads((output / 'summary.json').read_text())
        marker.update(eligible_count=summary['eligible_count'], excluded_count=summary['excluded_count'])
    else:
        marker.update(optimizer_steps=20, samples_processed=320, excluded_count=0,
            fresh_process_reload_verified=True, gpu_smoke_verified=True, adapter_changed=True,
            dev_test_read=False, v6_main_training_completed=False, usable_as_main_initialization=False)
    marker['artifacts'] = [{'path': str(path), 'sha256': gate.file_hash(path)}
        for path in sorted(output.rglob('*')) if path.is_file() and path != marker_path]
    write(marker_path, marker)
    receipt = {**marker, 'artifacts': marker['artifacts'] + [{'path': str(marker_path), 'sha256': gate.file_hash(marker_path)}]}
    write(Path(job['success_receipt']), receipt)
    return receipt


@pytest.fixture
def campaign(tmp_path):
    snapshot = tmp_path / 'snapshot'
    package = Path(gate.__file__).parent
    for name in gate.SOURCE_FILES:
        target = snapshot / 'src/mediflow_medvlm' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(package / name, target)
    data = tmp_path / 'private'
    root = data / 'v6_v7'
    manifest = plan.build_plan(data, root, repo=snapshot, python=sys.executable)
    preparations = {}
    for index, job_id in enumerate(gate.PREPARE_IDS, 1):
        job = row(manifest, job_id)
        cell = job['cell_id']; spec = plan.cell_spec(cell); revision = f'{index:040x}'
        cache = data / 'models/hub'
        base = cache / ('models--' + spec['model_id'].replace('/', '--')) / 'snapshots' / revision
        write(base / 'config.json', {'model_type': spec['family']})
        (base / 'model.safetensors').write_bytes(b'CPU_FIXTURE_BASE_WEIGHTS')
        files = {path.name: {'sha256': gate.file_hash(path), 'bytes': path.stat().st_size} for path in base.iterdir()}
        output = root / 'models' / spec['model_id'].replace('/', '--')
        docs = {'pin.json': {'model_id': spec['model_id'], 'revision': revision},
            'native_assets.json': {'model_id': spec['model_id'], 'revision': revision,
                                  'snapshot_path': str(base), 'cache_dir': str(cache)},
            'provenance.json': {'model_id': spec['model_id'], 'revision': revision, 'files': files},
            'disk_check.json': {'fixture': True}}
        artifacts = [{'path': str(write(output / name, value)), 'sha256': gate.file_hash(output / name)} for name, value in docs.items()]
        receipt = {'status': 'SUCCEEDED', **{key: job[key] for key in gate.IDENTITY_KEYS},
            'scope': 'MODEL_PREPARATION_ONLY', 'cell_id': cell, 'model_id': spec['model_id'],
            'model_revision': revision, 'model_files_hash': gate.canonical_hash(files), 'artifacts': artifacts}
        write(Path(job['success_receipt']), receipt)
        preparations[job_id] = job['success_receipt']
    trains = {}
    for task in ('text', 'vqa'):
        rows = [{'sample_id': f'{task}-{i}', 'split': 'train', 'source_id': 'synthetic_cpu_fixture',
                 'task_type': 'medical_vqa' if task == 'vqa' else 'medical_qa',
                 'input': {'question': 'private fixture question', 'images': []},
                 'supervision': {'answer': 'private fixture target'}} for i in range(322)]
        path = data / (task + '.jsonl'); path.write_text(''.join(json.dumps(value) + '\n' for value in rows))
        trains[task] = path, rows
    children = []
    for cell in gate.CELL_IDS:
        spec = plan.cell_spec(cell); path, rows = trains[spec['task']]
        preconfig = gate.cpu.build_config(cell, path, preparations[gate._prepare_id(cell)], root / 'cpu' / cell,
            repo=snapshot, image_root=data / 'images' if spec['task'] == 'vqa' else None, expected_count=322)
        prepath = write(root / 'configs' / ('cpu-' + cell + '.json'), preconfig)
        output = Path(preconfig['output_dir']); output.mkdir(parents=True, exist_ok=True)
        records = [{'sample_id': value['sample_id'], 'position': index, 'eligible': True,
            'input_sha256': gate.canonical_hash(value['input']), 'target_sha256': gate.canonical_hash(value['supervision']),
            'native': {'sequence_tokens': 7 + index % 3, 'supervised_tokens_after_shift': 2 + index % 2,
                       'full_target_preserved': True, 'prefix_verified': True, 'all_tensors_cpu': True, 'prompt_image_pad_mask_verified': True}}
            for index, value in enumerate(rows)]
        (output / 'eligibility.jsonl').write_text(''.join(json.dumps(value) + '\n' for value in records))
        ids = [value['sample_id'] for value in rows]
        paths = ['model.layer.q_proj', 'model.layer.v_proj']
        architecture = {'architecture': {'exact_target_modules': paths, 'target_count': len(paths)},
                        'template': {'fixed_date': '2026-11-14'}, 'thinking': {'native_control_verified': False}}
        write(output / 'architecture.json', architecture)
        write(output / 'fixtures.json', {'fixture': True})
        write(output / 'config.json', preconfig)
        write(output / 'summary.json', {'eligible_ids': ids, 'eligible_ids_sha256': gate.canonical_hash(ids),
            'eligible_count': 322, 'excluded_ids': [], 'excluded_ids_sha256': gate.canonical_hash([]), 'excluded_count': 0, 'original_count': 322,
            'full_targets_preserved': True, 'dev_test_read': False, 'common_train_freeze_decided': False, 'max_length': 2048})
        prejob = {'job_id': preconfig['job_id'], 'run_id': preconfig['run_id'], 'phase': 'native_preflight', **spec,
            'dependencies': [gate._prepare_id(cell)], 'implementation_pending': False, 'uses_gpu': False, 'requires_gpu': False,
            'argv': [sys.executable, '-m', 'mediflow_medvlm.campaign_preflight', '--config', str(prepath)],
            'cwd': str(snapshot), 'source_files': preconfig['source_files'],
            **{key: preconfig[key] for key in supervisor.HASH_FIELDS}, 'success_receipt': str(root / 'receipts' / (preconfig['job_id'] + '.json'))}
        publish_phase(prejob, preconfig, cpu=True)
        children.append(prejob)
        config = gate.gpu.build_config(cell, prejob['success_receipt'], root / 'gpu' / cell, repo=snapshot)
        cfgpath = write(root / 'configs' / ('gpu-' + cell + '.json'), config)
        output = Path(config['output_dir']); output.mkdir(parents=True)
        write(output / 'config.json', config)
        write(output / 'adapter/adapter_config.json', {'base_model_name_or_path': config['model']['snapshot_path']})
        (output / 'adapter/adapter_model.safetensors').write_bytes(b'CPU_FIXTURE_ADAPTER')
        write(output / 'processor/tokenizer.json', {'fixture': True})
        selection = config['data']['selection']; selected = selection['train_records']; held = selection['held_train_record']
        tokens = [value['native']['supervised_tokens_after_shift'] for value in selected]
        reports = [{'all_finite': True, 'projections': [{'path': path,
            'tensors': {side: {'finite': True, 'nonzero': step > 0 or side == 'B'} for side in ('A', 'B')}}
            for path in paths]} for step in range(20)]
        metrics = {'optimizer_steps': 20, 'samples_processed': 320, 'excluded_count': 0,
            'adapter_changed': True, 'all_projection_tensors_changed': True, 'training_pid': 1000,
            'example_losses': [1.] * 320, 'loss_mean': 1., 'example_supervised_tokens': tokens,
            'supervised_tokens': sum(tokens), 'sequence_tokens': sum(value['native']['sequence_tokens'] for value in selected),
            'initial_adapter_sha256': 'a' * 64, 'adapter_sha256': 'b' * 64,
            'initial_projection_sha256': {path: {'A': 'a' * 64, 'B': 'b' * 64} for path in paths},
            'final_projection_sha256': {path: {'A': 'c' * 64, 'B': 'd' * 64} for path in paths},
            'nonzero_gradients_observed': {path: {'A': True, 'B': True} for path in paths},
            'gradient_reports': reports, 'gradient_norms_before_clip': [1.] * 20,
            'learning_rates': [config['recipe']['learning_rate'] * (1 - step / 20) for step in range(20)], 'final_learning_rate': 0.,
            'native_contract': {**architecture, 'weights_loaded': True, 'cuda_residency_verified': True,
                'files': {'all_prepared_files_rehashed': True, 'model_files_hash': config['model']['model_files_hash']},
                'revision': config['model']['revision'], 'snapshot_path': config['model']['snapshot_path'],
                'quantization': {'method': 'nf4', 'double_quant': True, 'compute_dtype': 'bfloat16'},
                'coverage': {'exact_target_modules': paths, 'target_count': len(paths), 'target_modules_sha256': gate.canonical_hash(paths),
                    'quantized_targets_verified': True, 'base_frozen': True, 'vision_projector_embeddings_head_frozen': True},
                'parameter_storage': {'base_dtype_signature': 'f' * 64},
                'gradient_verified': False, 'optimizer_verified': False, 'reload_verified': False}}
        write(output / 'metrics.json', metrics)
        identity = {key: config[key] for key in gate.IDENTITY_KEYS}
        checkpoint = {'complete': True, **identity, 'optimizer_steps': 20, 'samples_processed': 320,
            'adapter_sha256': 'b' * 64, 'adapter_files': {p.name: gate.file_hash(p) for p in (output / 'adapter').iterdir()},
            'processor_files': {p.name: gate.file_hash(p) for p in (output / 'processor').iterdir()}}
        write(output / 'checkpoint.json', checkpoint)
        generated = {'held_train_sample_id': held['sample_id'], 'generated_ids': [2, 3], 'generated_text': 'ready'}
        write(output / 'reload.generated.private.json', generated)
        reload = {'status': 'SUCCEEDED', **identity, 'pid': 2000, 'fresh_process_required': True,
            'adapter_weights_verified': True, 'finite_forward': True, 'dev_test_read': False,
            'adapter_sha256': 'b' * 64, 'held_train_sample_id': held['sample_id'], 'held_train_loss': .5,
            'target_tokens': held['native']['supervised_tokens_after_shift'], 'base_dtype_signature': 'f' * 64,
            'runtime_versions': config['runtime_versions'], 'generated_tokens': 2, 'generation_max_new_tokens': 32,
            'generated_ids_sha256': gate.canonical_hash([2, 3]), 'no_thinking_output': verify_native_no_thinking_output('ready')}
        write(output / 'reload.json', reload)
        gpujob = {'job_id': config['job_id'], 'run_id': config['run_id'], 'phase': 'native_smoke', **spec,
            'dependencies': [prejob['job_id']], 'implementation_pending': False, 'uses_gpu': True, 'requires_gpu': True,
            'argv': [sys.executable, '-m', 'mediflow_medvlm.campaign_gpu_smoke', '--config', str(cfgpath)],
            'cwd': str(snapshot), 'source_files': config['source_files'],
            **{key: config[key] for key in supervisor.HASH_FIELDS}, 'success_receipt': str(root / 'receipts' / (config['job_id'] + '.json'))}
        publish_phase(gpujob, config)
        children.append(gpujob)
    # Preserve the actual declaration/order expected by the existing gate.
    children.sort(key=lambda job: gate.PREREQUISITE_IDS.index(job['job_id']))
    position = next(i for i, job in enumerate(manifest['jobs']) if job['job_id'] == gate.JOB_ID)
    manifest['jobs'][position:position] = children
    row(manifest, gate.JOB_ID)['dependencies'] = gate.PREREQUISITE_IDS.copy()
    owner = supervisor.Supervisor(data)
    owner.initialize(manifest)
    state = supervisor.read_json(owner.state_path)
    for job_id in gate.PREREQUISITE_IDS:
        job = row(manifest, job_id); receipt = json.loads(Path(job['success_receipt']).read_text())
        state['jobs'][job_id].update(status='SUCCEEDED', attempts=3, failure_count=2, recovery_attempts=2,
            history=[{'attempt': 1, 'status': 'FAILED'}, {'attempt': 2, 'status': 'FAILED'}, {'attempt': 3, 'status': 'SUCCEEDED'}],
            exit_code=0, verification={'receipt': job['success_receipt'], 'receipt_sha256': gate.file_hash(job['success_receipt']),
                'artifact_count': len(receipt['artifacts'])})
    write(owner.state_path, state)
    return SimpleNamespace(supervisor=owner, manifest=manifest, snapshot=snapshot, root=root,
                           config_path=root / 'configs/gate.json', output=root / 'gate/aggregate')


def config(case):
    state = supervisor.read_json(case.supervisor.state_path)
    value = gate.build_config(Path(state['manifest_path']), case.supervisor.state_path, case.output, repo=case.snapshot)
    write(case.config_path, value)
    return value


def commit_fixture_manifest(case):
    state = supervisor.read_json(case.supervisor.state_path)
    digest = gate.canonical_hash(case.manifest)
    path = case.root / 'manifests' / (digest + '.json')
    write(path, case.manifest); state.update(manifest_hash=digest, manifest_path=str(path)); write(case.supervisor.state_path, state)


def reseal_gpu(case, cell='T1'):
    job = row(case.manifest, 'v6-native-smoke-' + cell)
    cfg = json.loads(config_path(job).read_text())
    cfg['config_hash'] = gate.canonical_hash({k: v for k, v in cfg.items() if k != 'config_hash'})
    for key in gate.IDENTITY_KEYS:
        if key in job: job[key] = cfg[key]
    write(config_path(job), cfg)
    output = Path(cfg['output_dir']); write(output / 'config.json', cfg)
    for name in ('reload.json', 'checkpoint.json'):
        value = json.loads((output / name).read_text())
        value.update({key: cfg[key] for key in gate.IDENTITY_KEYS}); write(output / name, value)
    receipt = publish_phase(job, cfg)
    state = supervisor.read_json(case.supervisor.state_path)
    state['jobs'][job['job_id']]['verification'].update(receipt_sha256=gate.file_hash(job['success_receipt']), artifact_count=len(receipt['artifacts']))
    write(case.supervisor.state_path, state); commit_fixture_manifest(case)


def test_gate_all_ten_cpu_artifact_audit_preserves_state_and_activate_contract(campaign):
    c = campaign; value = config(c)
    before = c.supervisor.state_path.read_bytes()
    revised = gate.activation_manifest(c.config_path, c.manifest, repo=c.snapshot, python=sys.executable)
    for old, new in zip(c.manifest['jobs'], revised['jobs']):
        if old['job_id'] != gate.JOB_ID: assert old == new
    activated = c.supervisor.activate(revised)
    assert activated['jobs'][gate.JOB_ID]['status'] == 'PENDING'
    state = supervisor.read_json(c.supervisor.state_path)
    state['jobs'][gate.JOB_ID].update(status='RUNNING', attempts=1)
    write(c.supervisor.state_path, state)
    state_bytes = c.supervisor.state_path.read_bytes()
    with patch.object(gate.os, 'system', side_effect=AssertionError('no shell/GPU')), patch.object(gate.sys, 'modules', dict(sys.modules)):
        result = gate.run_gate(c.config_path, Path(value['success_receipt']))
    assert c.supervisor.state_path.read_bytes() == state_bytes
    assert result['supported_cells'] == gate.CELL_IDS and result['all_cells_supported'] is True
    assert result['common_train_freeze_decided'] is False
    assert supervisor.verify_success({**value, 'success_receipt': value['success_receipt']})['artifact_count'] == 3
    summary = json.loads((c.output / 'summary.json').read_text())
    assert summary['declared_unique_models'] == 8 and summary['declared_cells'] == 10
    assert summary['prerequisite_status_counts']['native_smoke'] == {'SUCCEEDED': 10}
    assert summary['cell_status_counts'] == {'SUPPORTED': 10}
    assert all(row['eligibility']['eligible_count'] == 322 for row in summary['cells'])
    assert summary['base_weight_bytes_rehashed'] is False and summary['receipt_artifact_bytes_rehashed'] is True
    assert summary['comparison_cohort_frozen'] is False and summary['v6_main_training_completed'] is False
    assert all(record['attempts'] == 3 and record['failure_count'] == record['recovery_attempts'] == 2
               for record in summary['terminal_prerequisites'].values())
    output_bytes = {p: p.read_bytes() for p in c.output.iterdir()}
    assert gate.run_gate(c.config_path, Path(value['success_receipt'])) == result
    assert all(path.read_bytes() == content for path, content in output_bytes.items())
    assert before != state_bytes  # Only the explicitly simulated supervisor activation/attempt changed state.


@pytest.mark.parametrize('status', ['FAILED', 'BLOCKED_PREREQUISITE'])
def test_terminal_gpu_failure_is_explicit_unsupported_not_success(campaign, status):
    c = campaign; state = supervisor.read_json(c.supervisor.state_path)
    state['jobs']['v6-native-smoke-T1'].update(status=status, error_signature='bound-failure', error='recorded failure', exit_code=0)
    write(c.supervisor.state_path, state)
    value = config(c); result = gate.run_gate(c.config_path, Path(value['success_receipt']))
    assert result['status'] == 'SUCCEEDED' and result['all_cells_supported'] is False
    assert 'T1' not in result['supported_cells'] and len(result['supported_cells']) == 9
    summary = json.loads((c.output / 'summary.json').read_text())
    assert summary['cell_status_counts'][status] == 1
    assert summary['terminal_prerequisites']['v6-native-smoke-T1']['error_signature'] == 'bound-failure'


def test_blocked_placeholder_without_source_is_counted_with_failed_shared_preparation(campaign):
    c = campaign; state = supervisor.read_json(c.supervisor.state_path)
    state['jobs']['prepare-T3'].update(status='FAILED', exit_code=1, error_signature='prepare-failed')
    for cell in ('T3', 'V2'):
        for kind in ('preflight', 'smoke'):
            key = f'v6-native-{kind}-{cell}'
            state['jobs'][key].update(status='BLOCKED_PREREQUISITE', attempts=0, failure_count=0, recovery_attempts=0)
            job = row(c.manifest, key)
            for field in ('cwd', 'source_files', *supervisor.HASH_FIELDS, 'argv'): job.pop(field, None)
            job['implementation_pending'] = True
    write(c.supervisor.state_path, state); commit_fixture_manifest(c)
    value = config(c); result = gate.run_gate(c.config_path, Path(value['success_receipt']))
    assert len(result['supported_cells']) == 8 and 'T3' not in result['supported_cells'] and 'V2' not in result['supported_cells']


@pytest.mark.parametrize('status', ['PENDING', 'RUNNING', 'ANALYSIS_REQUIRED', 'IMPLEMENTATION_PENDING'])
def test_nonterminal_prerequisite_rejected_before_writes(campaign, status):
    state = supervisor.read_json(campaign.supervisor.state_path)
    state['jobs']['v6-native-smoke-T1']['status'] = status; write(campaign.supervisor.state_path, state)
    with pytest.raises(gate.GateError, match='NOT_TERMINAL'): config(campaign)
    assert not campaign.output.exists()


@pytest.mark.parametrize('change', ['missing-cell', 'wrong-model', 'extra-model-job', 'success-after-failed-dependency'])
def test_declared_matrix_and_dependency_tampering_rejected(campaign, change):
    c = campaign
    if change == 'missing-cell': c.manifest['cells'].pop()
    elif change == 'wrong-model': c.manifest['cells'][0]['model_id'] = 'Other/model'
    elif change == 'extra-model-job':
        extra = copy.deepcopy(row(c.manifest, 'prepare-T1')); extra['job_id'] = 'prepare-extra'; c.manifest['jobs'].append(extra)
    else:
        state = supervisor.read_json(c.supervisor.state_path); state['jobs']['prepare-T1']['status'] = 'FAILED'; write(c.supervisor.state_path, state)
    commit_fixture_manifest(c)
    with pytest.raises((gate.GateError, ValueError)): config(c)


@pytest.mark.parametrize('artifact', ['adapter/adapter_model.safetensors', 'processor/tokenizer.json', 'metrics.json',
    'reload.json', 'completion.json'])
def test_success_artifact_tamper_rejected_and_not_downgraded(campaign, artifact):
    c = campaign; value = config(c)
    path = c.root / 'gpu/T1' / artifact; path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises((gate.GateError, ValueError, OSError)):
        gate.run_gate(c.config_path, Path(value['success_receipt']))
    assert not c.output.exists()


@pytest.mark.parametrize('field,value', [('recipe', {'optimizer_steps': 19}), ('max_length', 4096),
    ('fixed_template_date', '2026-11-15'), ('device', 'cpu'), ('trust_remote_code', True), ('local_files_only', False)])
def test_rehashed_wrong_science_config_rejected_semantically(campaign, field, value):
    job = row(campaign.manifest, 'v6-native-smoke-T1'); cfg = json.loads(config_path(job).read_text()); cfg[field] = value
    write(config_path(job), cfg); reseal_gpu(campaign)
    with pytest.raises(gate.GateError, match='POLICY|RECIPE'): config(campaign)


@pytest.mark.parametrize('change', ['reload-false', 'same-pid', 'adapter-digest', 'target-count', 'generated-ids',
    'generated-text', 'nonfinite-loss', 'missing-example', 'zero-gradient-all-steps', 'base-trainable', 'learning-rate'])
def test_resealed_invalid_gpu_evidence_rejected(campaign, change):
    output = campaign.root / 'gpu/T1'
    metrics = json.loads((output / 'metrics.json').read_text()); reload = json.loads((output / 'reload.json').read_text())
    if change == 'reload-false': reload['fresh_process_required'] = False
    elif change == 'same-pid': reload['pid'] = metrics['training_pid']
    elif change == 'adapter-digest': reload['adapter_sha256'] = 'c' * 64
    elif change == 'target-count': metrics['example_supervised_tokens'][0] += 1
    elif change == 'generated-ids':
        value = json.loads((output / 'reload.generated.private.json').read_text()); value['generated_ids'] = [7]; write(output / 'reload.generated.private.json', value)
    elif change == 'generated-text':
        value = json.loads((output / 'reload.generated.private.json').read_text()); value['generated_text'] = '<think>private reasoning</think>'; write(output / 'reload.generated.private.json', value)
    elif change == 'nonfinite-loss': metrics['example_losses'][0] = float('nan')
    elif change == 'missing-example': metrics['example_losses'].pop()
    elif change == 'zero-gradient-all-steps':
        for report in metrics['gradient_reports']:
            for projection in report['projections']: projection['tensors']['A']['nonzero'] = False
    elif change == 'base-trainable': metrics['native_contract']['coverage']['base_frozen'] = False
    elif change == 'learning-rate': metrics['learning_rates'][0] *= .5
    write(output / 'metrics.json', metrics); write(output / 'reload.json', reload); reseal_gpu(campaign)
    with pytest.raises((gate.GateError, ValueError)): config(campaign)


def test_receipt_identity_state_binding_and_raw_config_bytes_are_immutable(campaign):
    c = campaign; value = config(c)
    path = config_path(row(c.manifest, 'v6-native-smoke-T1'))
    path.write_text(path.read_text() + '\n')
    with pytest.raises(gate.GateError, match='INPUT_IDENTITY'):
        gate.run_gate(c.config_path, Path(value['success_receipt']))


def test_terminal_subset_counter_change_is_rejected_while_other_stage_changes_allowed(campaign):
    c = campaign; value = config(c)
    state = supervisor.read_json(c.supervisor.state_path)
    state['jobs']['v6-common_train_freeze']['status'] = 'RUNNING'; state['updated_at'] = 'changed'
    write(c.supervisor.state_path, state)
    assert gate._validate(value)
    state['jobs']['v6-native-smoke-T1']['failure_count'] = 0; write(c.supervisor.state_path, state)
    with pytest.raises(gate.GateError, match='SNAPSHOT_CHANGED'): gate._validate(value)


def test_completion_recovers_rename_receipt_gap_without_rewrite(campaign, monkeypatch):
    c = campaign; value = config(c); real_write = gate.atomic_json
    def crash(path, value):
        if Path(path) == Path(c.root / 'receipts' / (gate.JOB_ID + '.json')): raise OSError('synthetic receipt gap')
        return real_write(path, value)
    monkeypatch.setattr(gate, 'atomic_json', crash)
    with pytest.raises(OSError): gate.run_gate(c.config_path, Path(value['success_receipt']))
    protected = {p: p.read_bytes() for p in c.output.iterdir()}
    monkeypatch.setattr(gate, 'atomic_json', real_write)
    assert gate.run_gate(c.config_path, Path(value['success_receipt']))['status'] == 'SUCCEEDED'
    assert all(path.read_bytes() == content for path, content in protected.items())


@pytest.mark.parametrize('target', ['config.json', 'summary.json', 'completion.json'])
def test_completed_gate_artifact_tampering_rejected(campaign, target):
    c = campaign; value = config(c); gate.run_gate(c.config_path, Path(value['success_receipt']))
    path = c.output / target; path.write_text(path.read_text() + '\nchanged')
    with pytest.raises((gate.GateError, ValueError)):
        gate.run_gate(c.config_path, Path(value['success_receipt']))


def test_cli_error_redaction_and_cpu_only_import(campaign, monkeypatch, capsys):
    monkeypatch.setattr(gate, 'run_gate', lambda *a: (_ for _ in ()).throw(RuntimeError('private question/target')))
    assert gate.main(['--config', str(campaign.config_path), '--receipt', str(campaign.root / 'result.json')]) == 1
    assert 'private question' not in capsys.readouterr().err
    result = subprocess.run([sys.executable, '-c', 'import sys; import mediflow_medvlm.campaign_preflight_gate; assert "torch" not in sys.modules'],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('change', ['target-truncated', 'prefix-unverified', 'context-overflow', 'excluded-denominator'])
def test_resealed_contradictory_cpu_native_proof_rejected(campaign, change):
    c = campaign; job = row(c.manifest, 'v6-native-preflight-T1'); cfg = json.loads(config_path(job).read_text())
    output = Path(cfg['output_dir']); path = output / 'eligibility.jsonl'
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if change == 'target-truncated': records[0]['native']['full_target_preserved'] = False
    elif change == 'prefix-unverified': records[0]['native']['prefix_verified'] = False
    elif change == 'context-overflow': records[0]['native']['sequence_tokens'] = 2049
    else:
        summary = json.loads((output / 'summary.json').read_text()); summary['excluded_count'] = 1; write(output / 'summary.json', summary)
    path.write_text(''.join(json.dumps(record) + '\n' for record in records))
    receipt = publish_phase(job, cfg, cpu=True)
    state = supervisor.read_json(c.supervisor.state_path)
    state['jobs'][job['job_id']]['verification']['receipt_sha256'] = gate.file_hash(job['success_receipt'])
    write(c.supervisor.state_path, state)
    with pytest.raises(gate.GateError, match='CPU_FULL_TARGET|CPU_ELIGIBILITY_DENOMINATOR'):
        config(c)


def test_all_failed_or_blocked_preserves_declared_denominators(campaign):
    c = campaign; state = supervisor.read_json(c.supervisor.state_path)
    for key in gate.PREREQUISITE_IDS:
        state['jobs'][key].update(status='FAILED' if key in gate.PREPARE_IDS else 'BLOCKED_PREREQUISITE',
                                  error_signature='bound-terminal-cause')
        job = row(c.manifest, key)
        for field in ('cwd', 'source_files', *supervisor.HASH_FIELDS, 'argv'): job.pop(field, None)
        job['implementation_pending'] = True
    write(c.supervisor.state_path, state); commit_fixture_manifest(c)
    value = config(c); result = gate.run_gate(c.config_path, Path(value['success_receipt']))
    summary = json.loads((c.output / 'summary.json').read_text())
    assert result['supported_cells'] == [] and result['all_cells_supported'] is False
    assert summary['declared_unique_models'] == 8 and summary['declared_cells'] == 10
    assert summary['prerequisite_status_counts'] == {'model_preparation': {'FAILED': 8},
        'native_preflight': {'BLOCKED_PREREQUISITE': 10}, 'native_smoke': {'BLOCKED_PREREQUISITE': 10}}
    assert summary['cell_status_counts'] == {'FAILED': 10}


def test_success_state_verification_receipt_sha_mismatch_is_not_unsupported(campaign):
    state = supervisor.read_json(campaign.supervisor.state_path)
    state['jobs']['v6-native-smoke-T1']['verification']['receipt_sha256'] = 'stale'
    write(campaign.supervisor.state_path, state)
    with pytest.raises(gate.GateError, match='TERMINAL_SUCCESS_PROOF_MISMATCH'): config(campaign)


def test_wrong_prepared_model_receipt_even_rebound_to_state_is_rejected(campaign):
    job = row(campaign.manifest, 'prepare-T1'); receipt = json.loads(Path(job['success_receipt']).read_text())
    receipt['model_id'] = 'Other/model'; write(Path(job['success_receipt']), receipt)
    state = supervisor.read_json(campaign.supervisor.state_path)
    state['jobs'][job['job_id']]['verification']['receipt_sha256'] = gate.file_hash(job['success_receipt'])
    write(campaign.supervisor.state_path, state)
    with pytest.raises(gate.GateError, match='PREPARED_MODEL_IDENTITY'): config(campaign)


def test_partial_output_and_receipt_without_output_fail_closed(campaign):
    c = campaign; value = config(c)
    write(Path(value['success_receipt']), {'status': 'partial'})
    with pytest.raises(gate.GateError, match='RECEIPT_WITHOUT_OUTPUT'):
        gate.run_gate(c.config_path, Path(value['success_receipt']))
    Path(value['success_receipt']).unlink()
    c.output.mkdir(parents=True); write(c.output / 'summary.json', {'partial': True})
    with pytest.raises((gate.GateError, FileNotFoundError)):
        gate.run_gate(c.config_path, Path(value['success_receipt']))


def test_changed_gate_config_bytes_during_idempotent_reuse_are_rejected(campaign, monkeypatch):
    c = campaign; value = config(c); gate.run_gate(c.config_path, Path(value['success_receipt']))
    validate = gate._validate
    def change_bytes(cfg):
        result = validate(cfg)
        c.config_path.write_text(c.config_path.read_text() + '\n')
        return result
    monkeypatch.setattr(gate, '_validate', change_bytes)
    with pytest.raises(gate.GateError, match='CONFIG_CHANGED_DURING_GATE_REUSE'):
        gate.run_gate(c.config_path, Path(value['success_receipt']))
