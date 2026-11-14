"""CPU-only temporary fixtures for audited GPU source/config registration.

No campaign worker or CUDA process is started by these tests.
"""
import copy
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

from mediflow_medvlm import campaign_supervisor as c
from mediflow_medvlm.campaign_v6 import cell_spec


def job(manifest, cell='T2'):
    return next(row for row in manifest['jobs'] if row['cell_id'] == cell)


def config_path(row):
    return Path(row['argv'][row['argv'].index('--config') + 1])


def seal_config(row, config):
    config['config_hash'] = c.canonical_hash({k: v for k, v in config.items() if k != 'config_hash'})
    row['config_hash'] = config['config_hash']
    c.atomic_json(config_path(row), config)


def bind_receipt(case):
    supervisor, original, revised, path, receipt = case
    receipt.update(manifest_hash=c.canonical_hash(original), replacement_manifest_hash=c.canonical_hash(revised))
    receipt['jobs'] = {before['job_id']: {'job_hash': c.canonical_hash(before),
        'replacement_job_hash': c.canonical_hash(after), 'replacement_source_hash': after['source_hash']}
        for before, after in zip(original['jobs'], revised['jobs']) if before != after}
    c.atomic_json(path, receipt)


@pytest.fixture
def gpu_case(tmp_path):
    old = tmp_path / 'old-snapshot'
    new = tmp_path / 'new-snapshot'
    names = ['src/mediflow_medvlm/campaign_supervisor.py',
             'src/mediflow_medvlm/campaign_gpu_smoke.py', 'src/mediflow_medvlm/training/v6_gpu.py']
    for root in (old, new):
        for name in names:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# synthetic CPU-only snapshot\n' + ('# fixed\n' if root == new else ''))
    def sources(root):
        return {name: c.file_hash(root / name) for name in names}
    queue = []
    for cell in ('T1', 'T2', 'T3'):
        identity = 'v6-native-smoke-' + cell
        row = {'job_id': identity, 'run_id': identity, 'phase': 'native_smoke', **cell_spec(cell),
            'argv': [sys.executable, '-m', 'mediflow_medvlm.campaign_gpu_smoke', '--config',
                     str(tmp_path / ('old-' + cell + '.json')), '--receipt', str(tmp_path / (cell + '-receipt.json'))],
            'cwd': str(old), 'source_files': sources(old), 'source_hash': c.canonical_hash(sources(old)),
            'data_hash': 'data-' + cell, 'model_hash': 'model-' + cell,
            'success_receipt': str(tmp_path / (cell + '-receipt.json')), 'implementation_pending': False,
            'dependencies': [], 'uses_gpu': True, 'requires_gpu': True, 'gpu_lock_mode': 'wrapper'}
        config = {'schema_version': 'medvlm.v6.native-gpu-smoke-config.v1', 'scope': 'NATIVE_GPU_SMOKE_ONLY',
            **{k: row[k] for k in ('job_id', 'run_id', 'cell_id', 'model_id', 'task', 'family', 'source_files',
                                  'source_hash', 'model_hash', 'data_hash')},
            'recipe': {'seed': 20261023, 'optimizer_steps': 20, 'examples': 320},
            'data': {'train_file': '/immutable/train.jsonl'}, 'model': {'revision': 'frozen-pin'},
            'output_dir': str(tmp_path / (cell + '-output')), 'resource_policy': {'max_gpu_hours': None}}
        seal_config(row, config)
        queue.append(row)
    original = {'campaign_id': 'synthetic-native-smoke', 'scheduler_identity': {'id': 'fixed'}, 'jobs': queue}
    supervisor = c.Supervisor(tmp_path / 'private')
    supervisor.initialize(original)
    state = c.read_json(supervisor.state_path)
    # Synthetic stale label models the unreconciled second exit, without a PID.
    state['jobs'][queue[0]['job_id']].update(status='RUNNING', attempts=2, failure_count=1,
        recovery_attempts=1, history=[{'attempt': 1, 'status': 'FAILED'}], launch_token='preserved')
    c.atomic_json(supervisor.state_path, state)
    revised = copy.deepcopy(original)
    for cell in ('T2', 'T3'):
        row = job(revised, cell)
        config = c.read_json(config_path(row))
        row.update(cwd=str(new), source_files=sources(new), source_hash=c.canonical_hash(sources(new)))
        row['argv'][4] = str(tmp_path / ('new-' + cell + '.json'))
        config.update(source_files=row['source_files'], source_hash=row['source_hash'])
        seal_config(row, config)
    evidence = tmp_path / 'astra-execution.json'
    c.atomic_json(evidence, {'actual_model': 'gpt-6-astra', 'reasoning_effort': 'high', 'fixture': True})
    receipt = {'actual_model': 'gpt-6-astra', 'reasoning_effort': 'high', 'original_model': 'gpt-6.1-sol',
        'original_reasoning_effort': 'high', 'escalation_reason': 'Synthetic common GPU failure.',
        'analysis_summary': 'Synthetic source-only fix verified on CPU.',
        'code_fix_explanation': 'Correct synthetic adapter audit without recipe changes.',
        'revision_authorized': True, 'model_execution_receipt': {'path': str(evidence), 'sha256': c.file_hash(evidence)}}
    case = supervisor, original, revised, tmp_path / 'analysis.json', receipt
    bind_receipt(case)
    return case


def reject_unchanged(case, match=None):
    supervisor, _, revised, receipt_path, _ = case
    before = supervisor.state_path.read_bytes()
    versions = {p: p.read_bytes() for p in (supervisor.root / 'manifests').glob('*.json')}
    with pytest.raises((ValueError, OSError), match=match):
        supervisor.revise_gpu_source(revised, receipt_path)
    assert supervisor.state_path.read_bytes() == before
    assert {p: p.read_bytes() for p in (supervisor.root / 'manifests').glob('*.json')} == versions


def test_register_preserves_every_record_counter_config_and_original_manifest(gpu_case):
    supervisor, original, revised, path, receipt = gpu_case
    before = c.read_json(supervisor.state_path)
    protected = {Path(before['manifest_path']): Path(before['manifest_path']).read_bytes(),
                 supervisor.manifest_path: supervisor.manifest_path.read_bytes(),
                 **{config_path(row): config_path(row).read_bytes() for row in original['jobs']}}
    real_write = c.atomic_json
    writes = []
    def observe(path, value):
        if Path(path) == supervisor.state_path:
            assert c.read_json(value['manifest_path']) == revised
            assert c.read_json(supervisor.state_path) == before
        writes.append(Path(path))
        real_write(path, value)
    with patch.object(c, 'atomic_json', side_effect=observe), patch.object(c.subprocess, 'Popen', side_effect=AssertionError('must not launch')):
        result = supervisor.revise_gpu_source(revised, path)
    after = c.read_json(supervisor.state_path)
    assert after['jobs'] == before['jobs']
    assert after['jobs']['v6-native-smoke-T1']['status'] == 'RUNNING'
    assert after['queue_owner'] == before['queue_owner']
    assert after['ownership_handoffs'] == before['ownership_handoffs']
    assert all(p.read_bytes() == value for p, value in protected.items())
    assert writes == [Path(result['manifest_path']), supervisor.state_path]
    audit = result['manifest_revisions'][-1]
    assert audit['job_bindings'] == receipt['jobs']
    assert audit['analysis_receipt_hash'] == c.file_hash(path)
    assert audit['model_execution_receipt'] == receipt['model_execution_receipt']
    assert supervisor.load()[0] == revised
    durable = supervisor.state_path.read_bytes()
    # Repeating even after a target started is an audit read, never a counter reset.
    assert supervisor.revise_gpu_source(revised, path)['manifest_hash'] == result['manifest_hash']
    assert supervisor.state_path.read_bytes() == durable


@pytest.mark.parametrize('field,value', [('recipe', {'seed': 99}), ('data', {'train_file': '/changed'}),
    ('model', {'revision': 'changed'}), ('output_dir', '/changed'), ('resource_policy', {'max_gpu_hours': 1}),
    ('max_length', 4096), ('device', 'cpu'), ('new_science', True)])
def test_rejects_any_config_science_change_even_resealed(gpu_case, field, value):
    row = job(gpu_case[2]); config = c.read_json(config_path(row)); config[field] = value
    seal_config(row, config); bind_receipt(gpu_case)
    reject_unchanged(gpu_case, 'configs may differ only')


@pytest.mark.parametrize('field,value', [('run_id', 'changed'), ('data_hash', 'other-data'),
    ('model_hash', 'other-model'), ('dependencies', ['v6-native-smoke-T1']), ('success_receipt', '/changed'),
    ('uses_gpu', False), ('gpu_lock_mode', 'child'), ('extra_seed', 2)])
def test_rejects_job_science_or_execution_changes(gpu_case, field, value):
    job(gpu_case[2])[field] = value; bind_receipt(gpu_case)
    reject_unchanged(gpu_case)


@pytest.mark.parametrize('field,value', [('status', 'RUNNING'), ('status', 'ANALYSIS_REQUIRED'),
    ('attempts', 1), ('failure_count', 1), ('recovery_attempts', 1), ('attempts', False),
    ('history', [{'attempt': 1}]), ('attempt_dir', '/tmp/synthetic-old-attempt'),
    ('launch_token', 'started'), ('worker_identity', {}), ('verification', {}),
    ('recovery_authorization_hash', 'old-analysis')])
def test_rejects_previously_started_targets(gpu_case, field, value):
    supervisor = gpu_case[0]; state = c.read_json(supervisor.state_path)
    state['jobs']['v6-native-smoke-T2'][field] = value; c.atomic_json(supervisor.state_path, state)
    reject_unchanged(gpu_case, 'never-started')


@pytest.mark.parametrize('field,value', [('actual_model', 'gpt-6.1-sol'), ('reasoning_effort', 'invalid'),
    ('revision_authorized', False), ('analysis_summary', ''), ('code_fix_explanation', ''),
    ('original_model', ''), ('manifest_hash', 'stale'), ('replacement_manifest_hash', 'stale'), ('jobs', {})])
def test_rejects_missing_or_stale_analysis(gpu_case, field, value):
    gpu_case[4][field] = value; c.atomic_json(gpu_case[3], gpu_case[4]); reject_unchanged(gpu_case)


@pytest.mark.parametrize('target', ['source', 'old-source', 'config', 'old-config', 'evidence'])
@pytest.mark.parametrize('repeated', [False, True])
def test_rehashes_old_new_sources_configs_and_evidence(gpu_case, target, repeated):
    supervisor, original, revised, receipt_path, receipt = gpu_case
    if repeated:
        supervisor.revise_gpu_source(revised, receipt_path)
    row = job(original if target.startswith('old-') else revised)
    path = (Path(row['cwd']) / next(iter(row['source_files'])) if 'source' in target else
            Path(receipt['model_execution_receipt']['path']) if target == 'evidence' else config_path(row))
    path.write_text(path.read_text() + '\n# changed bytes\n')
    reject_unchanged(gpu_case)


@pytest.mark.parametrize('change', ['metadata', 'order', 'other-job', 'non-config-argv', 'duplicate-config',
    'same-config', 'config-alias', 'same-cwd', 'source-hash', 'config-source', 'config-hash', 'wrong-native'])
def test_rejects_out_of_scope_manifest_or_binding_changes(gpu_case, change):
    _, original, revised, _, _ = gpu_case; row = job(revised)
    if change == 'metadata': revised['extra'] = 1
    elif change == 'order': revised['jobs'].reverse()
    elif change == 'other-job': job(revised, 'T1')['source_hash'] = 'bad'
    elif change == 'non-config-argv': row['argv'][-1] = '/other-receipt.json'
    elif change == 'duplicate-config': row['argv'] += ['--config', str(config_path(row))]
    elif change == 'same-config': row['argv'][4] = str(config_path(job(original)))
    elif change == 'config-alias':
        alias = config_path(row).with_name('alias.json'); alias.symlink_to(config_path(job(original))); row['argv'][4] = str(alias)
    elif change == 'same-cwd': row['cwd'] = job(original)['cwd']
    elif change == 'source-hash': row['source_hash'] = 'bad'
    elif change in {'config-source', 'config-hash'}:
        config = c.read_json(config_path(row)); config['source_hash' if change == 'config-source' else 'config_hash'] = 'bad'
        if change == 'config-source': seal_config(row, config)
        else: c.atomic_json(config_path(row), config)
    elif change == 'wrong-native': row['phase'] = 'other'
    bind_receipt(gpu_case); reject_unchanged(gpu_case)


@pytest.mark.parametrize('active', ['record', 'heartbeat-child', 'worker-lease', 'gpu-lease'])
def test_rejects_actual_processes_and_leases_without_reconciling(gpu_case, active):
    supervisor = gpu_case[0]; state = c.read_json(supervisor.state_path)
    record = state['jobs']['v6-native-smoke-T1']
    if active == 'record': record['worker_identity'] = c.process_identity()
    if active == 'heartbeat-child':
        record['attempt_dir'] = str(supervisor.root / 'synthetic-attempt')
        c.atomic_json(Path(record['attempt_dir']) / 'heartbeat.json', {'child_identity': c.process_identity()})
    c.atomic_json(supervisor.state_path, state)
    if active.endswith('lease'):
        path = supervisor.root / 'worker.lock' if active == 'worker-lease' else supervisor.data_root / 'runtime/medvlm_gpu.lock'
        with c.try_lock(path): reject_unchanged(gpu_case, 'lease held')
    else: reject_unchanged(gpu_case, 'worker active')


def test_lock_missing_receipt_corrupt_version_and_conflicting_repeat(gpu_case):
    supervisor, _, revised, path, _ = gpu_case
    before = supervisor.state_path.read_bytes()
    with c.try_lock(supervisor.root / 'supervisor.lock'):
        assert supervisor.revise_gpu_source(revised, path) == {'status': 'SUPERVISOR_BUSY'}
    assert supervisor.state_path.read_bytes() == before
    with pytest.raises(ValueError): supervisor.revise_gpu_source(revised, path.with_name('absent.json'))
    version = supervisor.root / 'manifests' / (c.canonical_hash(revised) + '.json')
    c.atomic_json(version, {'corrupted': True})
    reject_unchanged(gpu_case, 'immutable GPU revision')
    version.unlink(); supervisor.revise_gpu_source(revised, path)
    conflict = copy.deepcopy(revised); conflict['extra'] = True
    with pytest.raises(ValueError, match='repeated GPU revision'):
        supervisor.revise_gpu_source(conflict, path)


def test_interrupted_pointer_update_is_recoverable(gpu_case):
    supervisor, _, revised, path, _ = gpu_case; before = supervisor.state_path.read_bytes(); real_write = c.atomic_json
    def interrupt(path, value):
        if Path(path) == supervisor.state_path: raise OSError('synthetic interrupted pointer')
        real_write(path, value)
    with patch.object(c, 'atomic_json', side_effect=interrupt), pytest.raises(OSError):
        supervisor.revise_gpu_source(revised, path)
    assert supervisor.state_path.read_bytes() == before
    result = supervisor.revise_gpu_source(revised, path)
    assert len(result['manifest_revisions']) == 1


def test_cli_uses_registration_only_and_rejects_missing_manifest(gpu_case, capsys):
    supervisor, _, revised, receipt_path, _ = gpu_case; path = receipt_path.with_name('manifest.json'); c.atomic_json(path, revised)
    with patch.object(c.Supervisor, 'tick', side_effect=AssertionError('must not tick')):
        assert c.main(['--data-root', str(supervisor.data_root), 'revise-gpu-source', '--manifest', str(path), '--analysis-receipt', str(receipt_path)]) == 0
    assert 'UNSTARTED_GPU_SOURCE_REVISION' in capsys.readouterr().out
    with pytest.raises(ValueError, match='manifest object'):
        c.main(['--data-root', str(supervisor.data_root), 'revise-gpu-source', '--manifest', str(path.with_name('absent.json')), '--analysis-receipt', str(receipt_path)])


def recovery_case(case):
    supervisor, original, revised, path, receipt = case
    # Apply the same bounded replacement only to T1, leaving pending T2/T3 unchanged.
    candidate = copy.deepcopy(original); replacement = copy.deepcopy(job(revised))
    original_t1 = job(original, 'T1'); new_t1 = job(candidate, 'T1')
    old_config = c.read_json(config_path(original_t1))
    new_t1.update(cwd=replacement['cwd'], source_files=replacement['source_files'], source_hash=replacement['source_hash'])
    new_t1['argv'][4] = str(path.with_name('new-T1.json'))
    old_config.update(source_files=new_t1['source_files'], source_hash=new_t1['source_hash']); seal_config(new_t1, old_config)
    state = c.read_json(supervisor.state_path); record = state['jobs'][original_t1['job_id']]
    record.update(status='ANALYSIS_REQUIRED', attempts=2, failure_count=2, recovery_attempts=1, error_signature='two-failures')
    c.atomic_json(supervisor.state_path, state)
    receipt.update(job_id=original_t1['job_id'], run_id=original_t1['run_id'], job_hash=c.canonical_hash(original_t1),
        replacement_job_hash=c.canonical_hash(new_t1), replacement_source_hash=new_t1['source_hash'],
        manifest_hash=c.canonical_hash(original), replacement_manifest_hash=c.canonical_hash(candidate),
        failure_count=2, error_signature='two-failures', retry_authorized=True)
    c.atomic_json(path, receipt)
    return candidate


def test_gpu_recovery_preserves_counters_history_other_jobs_and_revalidates_repeat(gpu_case):
    supervisor, original, _, path, _ = gpu_case; candidate = recovery_case(gpu_case)
    before = c.read_json(supervisor.state_path)
    result = supervisor.authorize_recovery('v6-native-smoke-T1', path, candidate)
    record = result['jobs']['v6-native-smoke-T1']
    assert (record['attempts'], record['failure_count'], record['recovery_attempts']) == (2, 2, 1)
    assert record['status'] == 'PENDING'
    assert record['history'][:-1] == before['jobs']['v6-native-smoke-T1']['history']
    for cell in ('T2', 'T3'): assert result['jobs']['v6-native-smoke-' + cell] == before['jobs']['v6-native-smoke-' + cell]
    assert c.read_json(supervisor.manifest_path) == original
    durable = supervisor.state_path.read_bytes()
    supervisor.authorize_recovery('v6-native-smoke-T1', path, candidate)
    supervisor.authorize_recovery('v6-native-smoke-T1', path)
    assert supervisor.state_path.read_bytes() == durable
    config = config_path(job(candidate, 'T1')); config.write_text(config.read_text() + '\n# tampered')
    with pytest.raises(ValueError): supervisor.authorize_recovery('v6-native-smoke-T1', path)
    assert supervisor.state_path.read_bytes() == durable


@pytest.mark.parametrize('change', ['science', 'stale-manifest', 'not-held', 'source-only-stale-config', 'active-lease'])
def test_gpu_recovery_rejects_unsafe_rebind(gpu_case, change):
    supervisor, _, _, path, receipt = gpu_case; candidate = recovery_case(gpu_case); row = job(candidate, 'T1')
    if change == 'science':
        config = c.read_json(config_path(row)); config['recipe']['seed'] = 99; seal_config(row, config)
        receipt.update(replacement_job_hash=c.canonical_hash(row), replacement_manifest_hash=c.canonical_hash(candidate)); c.atomic_json(path, receipt)
    if change == 'stale-manifest': receipt['manifest_hash'] = 'stale'; c.atomic_json(path, receipt)
    if change == 'not-held':
        state = c.read_json(supervisor.state_path); state['jobs'][row['job_id']]['status'] = 'RUNNING'; c.atomic_json(supervisor.state_path, state)
    if change == 'source-only-stale-config': row['argv'][4] = str(config_path(job(gpu_case[1], 'T1')))
    before = supervisor.state_path.read_bytes()
    if change == 'active-lease':
        with c.try_lock(supervisor.root / 'worker.lock'), pytest.raises(ValueError, match='lease held'):
            supervisor.authorize_recovery(row['job_id'], path, candidate)
    else:
        with pytest.raises(ValueError): supervisor.authorize_recovery(row['job_id'], path, candidate)
    assert supervisor.state_path.read_bytes() == before


@pytest.mark.parametrize('change', ['missing', 'extra'])
def test_rebind_preserves_complete_captured_source_set(gpu_case, change):
    row = job(gpu_case[2]); files = dict(row['source_files'])
    if change == 'missing': files.pop('src/mediflow_medvlm/training/v6_gpu.py')
    else:
        path = Path(row['cwd']) / 'extra.py'; path.write_text('# extra synthetic file')
        files['extra.py'] = c.file_hash(path)
    row.update(source_files=files, source_hash=c.canonical_hash(files))
    config = c.read_json(config_path(row)); config.update(source_files=files, source_hash=row['source_hash'])
    seal_config(row, config); bind_receipt(gpu_case)
    reject_unchanged(gpu_case, 'complete captured source file set')


def test_rebind_rejects_json_bool_for_numeric_science(gpu_case):
    row = job(gpu_case[2]); config = c.read_json(config_path(row))
    old = c.read_json(config_path(job(gpu_case[1])))
    # Python's equality permits True == 1; canonical JSON equality must not.
    old['recipe']['numeric_control'] = 1; seal_config(job(gpu_case[1]), old)
    config['recipe']['numeric_control'] = True; seal_config(row, config)
    # Bind the re-sealed synthetic original registration, without any real queue.
    supervisor = gpu_case[0]; state = c.read_json(supervisor.state_path)
    old_digest = c.canonical_hash(gpu_case[1]); old_path = supervisor.root / 'manifests' / (old_digest + '.json')
    c.atomic_json(old_path, gpu_case[1]); state.update(manifest_hash=old_digest, manifest_path=str(old_path)); c.atomic_json(supervisor.state_path, state)
    bind_receipt(gpu_case); reject_unchanged(gpu_case, 'configs may differ only')
