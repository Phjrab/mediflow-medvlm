"""Finite detached CPU workers exercise the original native dispatch fence."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from mediflow_medvlm import campaign_supervisor as engine
from test_campaign_supervisor import setup_campaign, wait_exit

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('terminal_guard', ROOT / 'scripts/campaign_control_terminal_guard.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def reference(path):
    return {'path': str(path), 'sha256': guard.file_hash(path)}


def write(path, value):
    engine.atomic_json(path, value)
    return path


def fixture(setup_campaign, tmp_path, *, expanded=False, scope='auto'):
    original, manifest = setup_campaign(jobs=10 if expanded else 6)
    manifest = copy.deepcopy(manifest)
    ids = ['v6-report'] + ([item for c in guard.CONDITIONS for item in (f'v7-{c}', f'v7-{c}-eval')]
                           if expanded else [f'v7-{c}' for c in guard.CONDITIONS]) + ['v7-report']
    for job, key in zip(manifest['jobs'], ids):
        job.update(job_id=key, run_id=key,
                   cell_id='ALL_V6' if key == 'v6-report' else
                        ('ALL_V7' if key == 'v7-report' else key.split('-')[1]),
                   phase='report' if key.endswith('report') else 'v7')
    supervisor = engine.Supervisor(tmp_path / 'guarded')
    supervisor.initialize(manifest)
    control = SimpleNamespace(supervisor=supervisor, engine=engine, root=supervisor.root, data_root=supervisor.data_root)
    parent = write(tmp_path / 'prior.json', {'cpu_fixture': True})
    boundary = guard.Boundary(control, scope=scope, deployment=parent, deployment_sha256=guard.file_hash(parent))
    boundary.install()
    return control, boundary, manifest


def test_auto_finishes_v6_but_never_dispatches_ready_v7(setup_campaign, tmp_path):
    control, boundary, manifest = fixture(setup_campaign, tmp_path)
    original = engine.read_json(control.supervisor.state_path)
    manifest_sha = guard.file_hash(control.supervisor.manifest_path)
    control.supervisor.tick()
    wait_exit(control.supervisor, 'v6-report')
    for _ in range(2):
        result = control.supervisor.tick()
        assert result['jobs']['v6-report']['status'] == 'SUCCEEDED'
        assert set(result['jobs']) == {j['job_id'] for j in manifest['jobs']}
        assert not result['all_terminal']
        assert result['automatic_dispatch_scope'] == 'V6_ONLY'
        assert result['v7_manual_start_required']
        for key in guard.AUTOMATIC_V7_IDS:
            assert result['jobs'][key] == original['jobs'][key]
    assert guard.file_hash(control.supervisor.manifest_path) == manifest_sha
    assert not boundary.dispatching


def test_busy_native_supervisor_lease_does_not_start_anything(setup_campaign, tmp_path):
    control, boundary, _ = fixture(setup_campaign, tmp_path)
    before = control.supervisor.state_path.read_bytes()
    with engine.try_lock(control.root / 'supervisor.lock'):
        assert control.supervisor.tick() == {'status': 'SUPERVISOR_BUSY'}
    assert control.supervisor.state_path.read_bytes() == before
    assert not boundary.dispatching


def test_busy_native_worker_lease_does_not_increment_attempts(setup_campaign, tmp_path):
    control, _, _ = fixture(setup_campaign, tmp_path)
    before = engine.read_json(control.supervisor.state_path)['jobs']
    with engine.try_lock(control.root / 'worker.lock'):
        result = control.supervisor.tick()
    assert result['jobs'] == before


def test_unknown_v7_cohort_rejected_before_state_changes(setup_campaign, tmp_path):
    control, boundary, _ = fixture(setup_campaign, tmp_path, expanded=True)
    before = control.supervisor.state_path.read_bytes()
    with pytest.raises(ValueError, match='V7 cohort changed'):
        control.supervisor.tick()
    assert control.supervisor.state_path.read_bytes() == before
    assert not boundary.dispatching


def test_terminal_without_manual_handoff_cannot_touch_v6_or_v7(setup_campaign, tmp_path):
    control, boundary, _ = fixture(setup_campaign, tmp_path, scope='terminal')
    before = control.supervisor.state_path.read_bytes()
    status = control.supervisor.status()
    assert not status['manual_handoff_verified'] and status['terminal_readiness_reasons']
    with pytest.raises((ValueError, TypeError)):
        control.supervisor.tick()
    assert control.supervisor.state_path.read_bytes() == before
    assert not boundary.dispatching


def prepare_ready(control, boundary, manifest, tmp_path):
    # Fixture V6 receipt is produced and verified by the actual native CPU worker.
    boundary.scope = 'auto'
    boundary.authorize = lambda *a, **k: [j['job_id'] for j in manifest['jobs'] if guard.v7_job(j)]
    # Expanded test cohort needs temporary native V6-only projection for its CPU prerequisite.
    with patch.object(guard, 'AUTOMATIC_V7_IDS', {j['job_id'] for j in manifest['jobs'] if guard.v7_job(j)}):
        control.supervisor.tick()
        wait_exit(control.supervisor, 'v6-report')
        control.supervisor.tick()
    del boundary.authorize
    boundary.scope = 'terminal'
    manifest = copy.deepcopy(manifest)
    bindings = {}
    condition_jobs = {}
    for c in guard.CONDITIONS:
        condition_jobs[c] = {'train': [f'v7-{c}'], 'evaluation': [f'v7-{c}-eval']}
    allowed = [key for c in guard.CONDITIONS for kind in ('train', 'evaluation')
               for key in condition_jobs[c][kind]] + ['v7-report']
    for job in manifest['jobs']:
        key = job['job_id']
        if key not in allowed:
            continue
        condition = job['cell_id']
        kind = 'report' if key == 'v7-report' else ('evaluation' if key.endswith('-eval') else 'train')
        config = {field: job[field] for field in ('job_id', 'run_id', 'source_files', 'source_hash')}
        config.update(condition=condition, model_id='Qwen/Qwen3-1.7B', data={'fixture': key}, model={'fixture': 'base'})
        config.update(data_hash=engine.canonical_hash(config['data']), model_hash=engine.canonical_hash(config['model']))
        config['config_hash'] = engine.canonical_hash(config)
        path = write(tmp_path / (key + '.config.json'), config)
        job.update({field: config[field] for field in ('config_hash', 'data_hash', 'model_hash')})
        job['argv'] += ['--config', str(path)]
        bindings[key] = {'job_hash': engine.canonical_hash(job), 'config': reference(path),
                         'condition': condition, 'kind': kind}
    # Test fixture registration only; production code never writes state or manifests.
    state = engine.read_json(control.supervisor.state_path)
    state['manifest_hash'] = engine.canonical_hash(manifest)
    version = control.root / 'manifests' / (state['manifest_hash'] + '.json')
    write(version, manifest)
    state['manifest_path'] = str(version)
    write(control.supervisor.state_path, state)
    evidence = write(tmp_path / 'ready.json', {'status': 'READY', 'cpu_tests_passed': True,
        'v7_workers_implemented': True, 'campaign_root': str(control.root),
        'manifest_hash': state['manifest_hash'], 'job_bindings_hash': engine.canonical_hash(bindings)})
    h = {'schema_version': 'mediflow.v7-terminal-handoff.v1', 'campaign_root': str(control.root),
         'data_root': str(control.data_root), 'campaign_id': state['campaign_id'],
         'model_id': 'Qwen/Qwen3-1.7B', 'conditions': guard.CONDITIONS,
         'parent_deployment': {'path': str(boundary.deployment), 'sha256': boundary.deployment_sha256},
         'guard_operator': reference(Path(guard.__file__)),
         'manifest': {**reference(version), 'manifest_hash': state['manifest_hash']},
         'condition_jobs': condition_jobs, 'report_job_ids': ['v7-report'], 'allowed_job_ids': allowed,
         'v6_report_job_id': 'v6-report', 'job_bindings': bindings, 'readiness': [reference(evidence)],
         'manual_scope_token_sha256': hashlib.sha256(b'user-foreground').hexdigest()}
    boundary.handoff = write(tmp_path / 'handoff.json', h)
    boundary.handoff_sha256 = guard.file_hash(boundary.handoff)
    return h, manifest, state


def test_manual_scope_rechecks_token_and_never_relaunches_v6(setup_campaign, tmp_path, monkeypatch):
    control, boundary, manifest = fixture(setup_campaign, tmp_path, expanded=True)
    h, manifest, state = prepare_ready(control, boundary, manifest, tmp_path)
    before = control.supervisor.state_path.read_bytes()
    with pytest.raises(ValueError, match='scope token'):
        control.supervisor.tick()
    assert control.supervisor.state_path.read_bytes() == before
    monkeypatch.setenv('MEDVLM_V7_SCOPE_TOKEN', 'user-foreground')
    assert boundary.authorize(manifest, state, token_required=True) == h['allowed_job_ids']
    authorize = boundary.authorize
    locked_checks = []
    def checked(*args, **kwargs):
        with engine.try_lock(control.root / 'supervisor.lock') as lease:
            locked_checks.append(lease is None)
        return authorize(*args, **kwargs)
    with patch.object(boundary, 'authorize', checked):
        result = control.supervisor.tick()
    assert any(locked_checks), 'authorization must run while canonical supervisor lock is held'
    assert result['jobs']['v6-report'] == state['jobs']['v6-report']
    assert result['jobs']['v7-P0']['attempts'] == 1
    wait_exit(control.supervisor, 'v7-P0')


@pytest.mark.parametrize('change', ['config', 'model', 'condition', 'science_closure', 'report', 'v6_pending'])
def test_tampered_prerequisites_cannot_dispatch(setup_campaign, tmp_path, monkeypatch, change):
    control, boundary, manifest = fixture(setup_campaign, tmp_path, expanded=True)
    h, manifest, state = prepare_ready(control, boundary, manifest, tmp_path)
    monkeypatch.setenv('MEDVLM_V7_SCOPE_TOKEN', 'user-foreground')
    if change == 'config':
        Path(h['job_bindings']['v7-P0']['config']['path']).write_text('{}')
    elif change in {'model', 'condition'}:
        h['model_id' if change == 'model' else 'conditions'] = 'other'
    elif change == 'science_closure':
        p = Path(h['readiness'][0]['path']); value = engine.read_json(p)
        value['job_bindings_hash'] = '0' * 64; write(p, value); h['readiness'][0] = reference(p)
    elif change == 'report':
        job = next(j for j in manifest['jobs'] if j['job_id'] == 'v6-report')
        Path(job['success_receipt']).write_text('{}')
    else:
        state['jobs']['v6-report']['status'] = 'PENDING'; write(control.supervisor.state_path, state)
    write(boundary.handoff, h); boundary.handoff_sha256 = guard.file_hash(boundary.handoff)
    before = control.supervisor.state_path.read_bytes()
    with pytest.raises((ValueError, KeyError)):
        control.supervisor.tick()
    assert control.supervisor.state_path.read_bytes() == before
    assert not boundary.dispatching
