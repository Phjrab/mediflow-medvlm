"""Real detached CPU fixtures exercise queue persistence and process ownership."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from unittest.mock import patch

import pytest

from mediflow_medvlm import campaign_supervisor as campaign


FIXTURE = '''import hashlib,json,os,pathlib,sys,time
root=pathlib.Path(sys.argv[1]); mode=sys.argv[2]
(root/'started').write_text(str(os.getpid()))
if mode == 'slow': time.sleep(1.2)
if mode == 'fail': raise SystemExit(7)
if mode == 'pending': raise SystemExit(75)
artifact=root/'artifact.txt';artifact.write_text('actual artifact')
r={'status':'SUCCEEDED','job_id':os.environ['MEDVLM_SUPERVISOR_JOB_ID'],'run_id':os.environ['MEDVLM_SUPERVISOR_RUN_ID'],'artifacts':[{'path':str(artifact),'sha256':hashlib.sha256(artifact.read_bytes()).hexdigest()}]}
for k in ('source_hash','config_hash','data_hash','model_hash'): r[k]=os.environ['MEDVLM_SUPERVISOR_'+k.upper()]
if mode != 'no-receipt': (root/'success.json').write_text(json.dumps(r))
print('fixture finished')
'''


@pytest.fixture
def setup_campaign(tmp_path):
    snapshot = tmp_path / 'snapshot'
    package = snapshot / 'src' / 'mediflow_medvlm'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text('')
    shutil.copy(campaign.__file__, package / 'campaign_supervisor.py')
    (snapshot / 'fixture.py').write_text(FIXTURE)
    (package / 'recovery_fixture.py').write_text(FIXTURE)
    sources = {str(p.relative_to(snapshot)): campaign.file_hash(p) for p in snapshot.rglob('*.py')}

    def make(mode='normal', jobs=1, module=False):
        queue = []
        for n in range(jobs):
            out = tmp_path / f'output-{n}'
            out.mkdir(exist_ok=True)
            queue.append({'job_id': f'job-{n}', 'phase': 'V6_PREPARE', 'cell_id': f'T{n+1}',
                'run_id': f'run-{n}', 'argv': [sys.executable, '-m', 'mediflow_medvlm.recovery_fixture', str(out), mode]
                    if module else [sys.executable, str(snapshot / 'fixture.py'), str(out), mode],
                'cwd': str(snapshot), 'source_files': sources, 'source_hash': campaign.canonical_hash(sources),
                'config_hash': 'frozen-config', 'data_hash': 'frozen-data', 'model_hash': 'frozen-model',
                'success_receipt': str(out / 'success.json'), 'uses_gpu': False, 'dependencies': []})
        manifest = {'campaign_id': 'fixture-v6-v7', 'scheduler_identity': {'id': 'fixture'}, 'jobs': queue}
        supervisor = campaign.Supervisor(tmp_path / 'private')
        supervisor.initialize(manifest)
        return supervisor, manifest
    return make


def wait_exit(supervisor, job='job-0'):
    attempt = Path(supervisor.status()['jobs'][job]['attempt_dir'])
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        receipt = campaign.read_json(attempt / 'exit.json')
        if receipt:
            # Worker fd is released after receipt publication, not before it.
            with campaign.try_lock(supervisor.root / 'worker.lock') as stream:
                if stream is not None:
                    return receipt
        time.sleep(.02)
    pytest.fail('bounded fixture worker did not exit')


def test_normal_completion_verifies_artifact_and_durable_logs(setup_campaign):
    supervisor, _ = setup_campaign()
    first = supervisor.tick()
    assert first['jobs']['job-0']['attempts'] == 1
    assert wait_exit(supervisor)['exit_code'] == 0
    final = campaign.Supervisor(supervisor.data_root).tick()
    assert final['jobs']['job-0']['status'] == 'SUCCEEDED'
    assert final['jobs']['job-0']['verification']['artifact_count'] == 1
    assert final['all_terminal'] and not final['ready_to_stop_scheduler']
    attempt = Path(final['jobs']['job-0']['attempt_dir'])
    assert 'fixture finished' in (attempt / 'stdout.log').read_text()
    heartbeat = campaign.read_json(attempt / 'heartbeat.json')
    assert heartbeat['heartbeat_at'] and heartbeat['last_progress_at'] is None
    assert heartbeat['worker_pid'] != heartbeat['child_identity']['worker_pid']
    assert (attempt / 'stderr.log').exists()
    assert supervisor.tick()['jobs']['job-0']['attempts'] == 1


def test_real_launcher_exits_while_detached_wrapper_and_job_live(setup_campaign):
    supervisor, manifest = setup_campaign('slow')
    env = {**os.environ, 'PYTHONPATH': str(Path(manifest['jobs'][0]['cwd']) / 'src')}
    result = subprocess.run([sys.executable, '-m', 'mediflow_medvlm.campaign_supervisor',
        '--data-root', str(supervisor.data_root), 'tick'], env=env, capture_output=True, text=True, check=True)
    launched = json.loads(result.stdout)
    record = launched['jobs']['job-0']
    assert campaign.same_process(record['worker_identity'])
    duplicate = supervisor.tick()
    assert duplicate['jobs']['job-0']['attempts'] == 1
    assert wait_exit(supervisor)['exit_code'] == 0
    assert supervisor.tick()['jobs']['job-0']['status'] == 'SUCCEEDED'


def test_nonblocking_duplicate_tick_does_not_launch(setup_campaign):
    supervisor, _ = setup_campaign()
    with campaign.try_lock(supervisor.root / 'supervisor.lock') as lock:
        assert lock is not None
        assert supervisor.tick() == {'status': 'SUPERVISOR_BUSY'}
    assert supervisor.status()['jobs']['job-0']['attempts'] == 0


def test_at_most_one_worker_and_late_heartbeat_never_kills(setup_campaign):
    supervisor, _ = setup_campaign('slow', jobs=2)
    state = supervisor.tick()
    attempt = Path(state['jobs']['job-0']['attempt_dir'])
    deadline = time.monotonic() + 3
    while not (attempt / 'heartbeat.json').exists() and time.monotonic() < deadline:
        time.sleep(.01)
    health = campaign.read_json(attempt / 'heartbeat.json')
    health['heartbeat_at'] = '1999-01-01T00:00:00+00:00'
    campaign.atomic_json(attempt / 'heartbeat.json', health)
    with patch('os.kill', side_effect=AssertionError('must never kill')):
        checked = supervisor.tick()
    assert checked['jobs']['job-0']['attempts'] == 1
    assert checked['jobs']['job-1']['attempts'] == 0
    assert campaign.same_process(health['worker_identity'])
    wait_exit(supervisor)
    supervisor.tick()
    wait_exit(supervisor, 'job-1')
    assert supervisor.tick()['all_terminal']


def test_failure_budget_persists_across_supervisor_instances(setup_campaign):
    supervisor, manifest = setup_campaign('fail')
    supervisor.tick()
    assert wait_exit(supervisor)['exit_code'] == 7
    assert campaign.Supervisor(supervisor.data_root).tick()['jobs']['job-0']['attempts'] == 2
    wait_exit(supervisor)
    hold = campaign.Supervisor(supervisor.data_root).tick()['jobs']['job-0']
    assert hold['status'] == 'ANALYSIS_REQUIRED'
    assert hold['failure_count'] == hold['attempts'] == 2
    assert supervisor.tick()['jobs']['job-0']['attempts'] == 2
    evidence = supervisor.root / 'actual-model-execution.json'
    campaign.atomic_json(evidence, {'actual_model': 'gpt-6-astra', 'reasoning_effort': 'xhigh', 'fixture': True})
    receipt = {'job_id': 'job-0', 'run_id': manifest['jobs'][0]['run_id'],
        'job_hash': campaign.canonical_hash(manifest['jobs'][0]), 'failure_count': 2,
        'error_signature': hold['error_signature'], 'actual_model': 'gpt-6-astra', 'reasoning_effort': 'xhigh',
        'original_model': 'gpt-6.1-sol', 'original_reasoning_effort': 'high',
        'escalation_reason': 'Two failures require stronger model analysis.',
        'retry_authorized': True, 'analysis_summary': 'Fixture error analyzed; one remaining recovery allowed.',
        'model_execution_receipt': {'path': str(evidence), 'sha256': campaign.file_hash(evidence)}}
    receipt_path = supervisor.root / 'analysis.json'
    campaign.atomic_json(receipt_path, {**receipt, 'actual_model': 'gpt-6.1-sol'})
    with pytest.raises(ValueError, match='stronger-model'):
        supervisor.authorize_recovery('job-0', receipt_path)
    campaign.atomic_json(receipt_path, receipt)
    assert supervisor.authorize_recovery('job-0', receipt_path)['jobs']['job-0']['failure_count'] == 2
    assert supervisor.authorize_recovery('job-0', receipt_path)['jobs']['job-0']['failure_count'] == 2
    state = supervisor.tick()
    assert state['jobs']['job-0']['attempts'] == 3
    wait_exit(supervisor)
    final = campaign.Supervisor(supervisor.data_root).tick()
    assert final['jobs']['job-0']['status'] == 'FAILED'
    assert final['jobs']['job-0']['failure_count'] == 3
    assert final['jobs']['job-0']['recovery_attempts'] == 2
    assert supervisor.tick()['jobs']['job-0']['attempts'] == 3


def test_pid_reuse_and_boot_change_are_not_same_worker():
    identity = campaign.process_identity()
    assert campaign.same_process(identity)
    assert not campaign.same_process({**identity, 'process_start_time': str(int(identity['process_start_time']) + 1)})
    assert not campaign.same_process({**identity, 'boot_identity': 'another-boot'})


def test_implementation_pending_never_becomes_failure(setup_campaign):
    supervisor, manifest = setup_campaign()
    # Register a separate pending queue with a dependent stage.
    pending = copy.deepcopy(manifest)
    pending['jobs'][0].update(implementation_pending=True, argv=[])
    dependent = {**pending['jobs'][0], 'job_id': 'later', 'cell_id': 'T2', 'run_id': 'later', 'dependencies': ['job-0']}
    pending['jobs'].append(dependent)
    other = campaign.Supervisor(supervisor.data_root / 'pending')
    other.initialize(pending)
    for _ in range(3):
        state = campaign.Supervisor(other.data_root).tick()
        assert not state['all_terminal']
        assert all(r['attempts'] == r['failure_count'] == r['recovery_attempts'] == 0 for r in state['jobs'].values())


def test_runtime_exit75_is_an_ordinary_bounded_failure(setup_campaign):
    supervisor, _ = setup_campaign('pending')
    supervisor.tick()
    assert wait_exit(supervisor)['status'] == 'EXITED'
    state = supervisor.tick()
    assert state['jobs']['job-0']['status'] == 'RUNNING'
    assert state['jobs']['job-0']['failure_count'] == 1
    assert state['jobs']['job-0']['attempts'] == 2
    wait_exit(supervisor)
    hold = supervisor.tick()['jobs']['job-0']
    assert hold['status'] == 'ANALYSIS_REQUIRED'
    assert hold['failure_count'] == 2


def test_exit_zero_without_artifacts_is_actual_failure(setup_campaign):
    supervisor, _ = setup_campaign('no-receipt')
    supervisor.tick()
    wait_exit(supervisor)
    state = supervisor.tick()
    assert state['jobs']['job-0']['failure_count'] == 1
    assert 'missing SUCCEEDED' in state['jobs']['job-0']['history'][0]['reason']
    # Let automatically allocated recovery fixture finish before temp cleanup.
    wait_exit(supervisor)


def test_changed_source_blocks_without_consuming_attempt(setup_campaign):
    supervisor, manifest = setup_campaign()
    Path(manifest['jobs'][0]['cwd'], 'fixture.py').write_text('print("changed")')
    record = supervisor.tick()['jobs']['job-0']
    assert record['status'] == 'IMPLEMENTATION_PENDING'
    assert record['attempts'] == record['failure_count'] == 0


def test_manifest_commands_cannot_change_after_registration(setup_campaign):
    supervisor, manifest = setup_campaign()
    changed = copy.deepcopy(manifest)
    changed['jobs'][0]['argv'].append('unapproved')
    with pytest.raises(ValueError, match='immutable'):
        supervisor.initialize(changed)


def test_legacy_queue_observation_requires_explicit_safe_handoff(setup_campaign):
    supervisor, manifest = setup_campaign()
    other = campaign.Supervisor(supervisor.data_root / 'legacy')
    other.initialize(manifest, campaign.process_identity())
    assert other.tick()['jobs']['job-0']['attempts'] == 0
    with pytest.raises(ValueError, match='still alive'):
        other.handoff()
    state = campaign.read_json(other.state_path)
    state['legacy_observer']['process_start_time'] = 'no-longer-same-process'
    campaign.atomic_json(other.state_path, state)
    assert other.tick()['jobs']['job-0']['attempts'] == 0
    assert other.handoff()['ownership_handoffs'] == 1
    assert other.handoff()['ownership_handoffs'] == 1


def test_gpu_lease_busy_leaves_queue_unspent(setup_campaign):
    supervisor, manifest = setup_campaign()
    gpu_manifest = copy.deepcopy(manifest)
    gpu_manifest['jobs'][0]['uses_gpu'] = True
    other = campaign.Supervisor(supervisor.data_root / 'gpu')
    other.initialize(gpu_manifest)
    with campaign.try_lock(other.data_root / 'runtime' / 'medvlm_gpu.lock'):
        state = other.tick()
    assert state['waiting_reason'] == 'GPU_LEASE_BUSY'
    assert state['jobs']['job-0']['attempts'] == 0


def test_artifact_tampering_and_identity_mismatch_rejected(setup_campaign):
    supervisor, manifest = setup_campaign()
    supervisor.tick()
    wait_exit(supervisor)
    job = manifest['jobs'][0]
    receipt = campaign.read_json(job['success_receipt'])
    receipt['model_hash'] = 'other-model'
    campaign.atomic_json(job['success_receipt'], receipt)
    with pytest.raises(ValueError, match='identity mismatch'):
        campaign.verify_success(job)
    receipt['model_hash'] = job['model_hash']
    campaign.atomic_json(job['success_receipt'], receipt)
    Path(receipt['artifacts'][0]['path']).write_text('tampered')
    with pytest.raises(ValueError, match='artifact hash mismatch'):
        campaign.verify_success(job)


def test_finalization_requires_report_and_ci_evidence(setup_campaign):
    supervisor, _ = setup_campaign()
    supervisor.tick()
    wait_exit(supervisor)
    supervisor.tick()
    receipt_path = supervisor.root / 'final.json'
    campaign.atomic_json(receipt_path, {'campaign_id': 'fixture-v6-v7', 'status': 'SUCCEEDED'})
    with pytest.raises(ValueError, match='report and CI'):
        supervisor.finalize(receipt_path)
    artifact = supervisor.root / 'report.txt'
    artifact.write_text('validated report and CI fixture')
    evidence = [{'path': str(artifact), 'sha256': campaign.file_hash(artifact)}]
    campaign.atomic_json(receipt_path, {'campaign_id': 'fixture-v6-v7', 'status': 'SUCCEEDED',
        'manifest_hash': supervisor.status()['manifest_hash'],
        'terminal_state_hash': supervisor.status()['terminal_state_hash'],
        'reports_verified': True, 'ci_verified': True, 'report_artifacts': evidence, 'ci_artifacts': evidence})
    assert supervisor.finalize(receipt_path)['ready_to_stop_scheduler']


def test_activation_binds_only_never_started_pending_execution(setup_campaign):
    supervisor, manifest = setup_campaign()
    registered = copy.deepcopy(manifest)
    registered['jobs'][0].update(implementation_pending=True, argv=[])
    other = campaign.Supervisor(supervisor.data_root / 'activate')
    other.initialize(registered)
    old_hash = other.status()['manifest_hash']
    activated = other.activate(manifest)
    assert activated['jobs']['job-0']['status'] == 'PENDING'
    assert activated['manifest_revisions'][0]['previous_hash'] == old_hash
    assert campaign.read_json(other.manifest_path) == registered
    assert other.load()[0] == manifest
    assert other.activate(manifest)['manifest_hash'] == activated['manifest_hash']
    other.tick()
    changed = copy.deepcopy(manifest)
    changed['jobs'][0]['config_hash'] = 'modified'
    with pytest.raises(ValueError, match='never-started'):
        other.activate(changed)
    wait_exit(other)
    assert other.tick()['jobs']['job-0']['status'] == 'SUCCEEDED'


def test_activation_preserves_scientific_identity_and_dependencies(setup_campaign):
    supervisor, manifest = setup_campaign()
    registered = copy.deepcopy(manifest)
    registered['jobs'][0].update(implementation_pending=True, argv=[])
    other = campaign.Supervisor(supervisor.data_root / 'identity-activate')
    other.initialize(registered)
    changed = copy.deepcopy(manifest)
    changed['jobs'][0]['run_id'] = 'changed-scientific-run'
    with pytest.raises(ValueError, match='scientific IDs'):
        other.activate(changed)
    assert other.status()['jobs']['job-0']['attempts'] == 0


def test_finalization_rejects_stale_manifest_and_terminal_outcomes(setup_campaign):
    supervisor, _ = setup_campaign()
    supervisor.tick()
    wait_exit(supervisor)
    state = supervisor.tick()
    artifact = supervisor.root / 'report.txt'
    artifact.write_text('report and CI fixture')
    evidence = [{'path': str(artifact), 'sha256': campaign.file_hash(artifact)}]
    receipt = {'campaign_id': state['campaign_id'], 'status': 'SUCCEEDED',
        'manifest_hash': state['manifest_hash'], 'terminal_state_hash': state['terminal_state_hash'],
        'reports_verified': True, 'ci_verified': True, 'report_artifacts': evidence, 'ci_artifacts': evidence}
    receipt_path = supervisor.root / 'final.json'
    campaign.atomic_json(receipt_path, {**receipt, 'manifest_hash': 'old-manifest'})
    with pytest.raises(ValueError, match='manifest hash is stale'):
        supervisor.finalize(receipt_path)
    campaign.atomic_json(receipt_path, {**receipt, 'terminal_state_hash': 'old-outcomes'})
    with pytest.raises(ValueError, match='terminal state hash is stale'):
        supervisor.finalize(receipt_path)
    campaign.atomic_json(receipt_path, receipt)
    current = campaign.read_json(supervisor.state_path)
    current['jobs']['job-0']['failure_count'] += 1
    campaign.atomic_json(supervisor.state_path, current)
    with pytest.raises(ValueError, match='terminal state hash is stale'):
        supervisor.finalize(receipt_path)


def test_failed_report_requires_verified_alternate_artifacts(setup_campaign):
    supervisor, manifest = setup_campaign()
    declared = copy.deepcopy(manifest)
    declared['jobs'][0]['phase'] = 'V7_REPORT'
    other = campaign.Supervisor(supervisor.data_root / 'failed-report')
    other.initialize(declared)
    state = campaign.read_json(other.state_path)
    state['jobs']['job-0'].update(status='FAILED', attempts=3, failure_count=3, recovery_attempts=2)
    campaign.atomic_json(other.state_path, state)
    artifact = other.root / 'alternate-report.txt'
    artifact.write_text('actual alternate report and CI fixture')
    evidence = [{'path': str(artifact), 'sha256': campaign.file_hash(artifact)}]
    receipt = {'campaign_id': state['campaign_id'], 'status': 'SUCCEEDED',
        'manifest_hash': state['manifest_hash'], 'terminal_state_hash': campaign.terminal_state_hash(state),
        'reports_verified': True, 'ci_verified': True, 'report_artifacts': evidence, 'ci_artifacts': evidence}
    receipt_path = other.root / 'final.json'
    campaign.atomic_json(receipt_path, receipt)
    with pytest.raises(ValueError, match='alternate report artifact'):
        other.finalize(receipt_path)
    receipt['alternate_report_artifacts'] = evidence
    campaign.atomic_json(receipt_path, receipt)
    artifact.write_text('changed after receipt')
    with pytest.raises(ValueError, match='artifact changed'):
        other.finalize(receipt_path)
    artifact.write_text('actual alternate report and CI fixture')
    assert other.finalize(receipt_path)['ready_to_stop_scheduler']


def source_recovery_candidate(supervisor, manifest):
    """Synthetic analysis evidence and a CPU-only replacement snapshot."""
    revised = copy.deepcopy(manifest)
    original = manifest['jobs'][0]
    snapshot = Path(original['cwd']).with_name('recovery-snapshot')
    shutil.copytree(original['cwd'], snapshot, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    script = snapshot / 'src/mediflow_medvlm/recovery_fixture.py'
    script.write_text(script.read_text().replace('SystemExit(7)', 'SystemExit(9)'))
    sources = {name: campaign.file_hash(snapshot / name) for name in original['source_files']}
    revised['jobs'][0].update(cwd=str(snapshot), source_files=sources,
                              source_hash=campaign.canonical_hash(sources))
    evidence = supervisor.root / 'actual-model-execution.json'
    campaign.atomic_json(evidence, {'actual_model': 'gpt-6-astra', 'reasoning_effort': 'xhigh', 'fixture': True})
    record = supervisor.status()['jobs']['job-0']
    receipt = {'job_id': 'job-0', 'run_id': original['run_id'],
        'job_hash': campaign.canonical_hash(original), 'failure_count': 2,
        'error_signature': record['error_signature'],
        'replacement_job_hash': campaign.canonical_hash(revised['jobs'][0]),
        'replacement_source_hash': revised['jobs'][0]['source_hash'],
        'actual_model': 'gpt-6-astra', 'reasoning_effort': 'xhigh',
        'original_model': 'gpt-6.1-sol', 'original_reasoning_effort': 'high',
        'escalation_reason': 'Two CPU fixture failures require recorded stronger analysis.',
        'retry_authorized': True, 'analysis_summary': 'Synthetic fixture analysis allows final retry.',
        'code_fix_explanation': 'Change fixture failure exit code to distinguish replacement execution.',
        'model_execution_receipt': {'path': str(evidence), 'sha256': campaign.file_hash(evidence)}}
    receipt_path = supervisor.root / 'analysis.json'
    campaign.atomic_json(receipt_path, receipt)
    return revised, receipt_path, receipt


@pytest.fixture
def recovery_case(setup_campaign):
    supervisor, manifest = setup_campaign('fail', jobs=2, module=True)
    # Seed only this temporary CPU fixture's durable hold; never start a worker.
    state = campaign.read_json(supervisor.state_path)
    record = state['jobs']['job-0']
    for attempt in (1, 2):
        record['attempts'] = attempt
        supervisor._failure(record, 'fixture failure', 7)
    record['recovery_attempts'] = 1
    campaign.atomic_json(supervisor.state_path, state)
    revised, receipt_path, receipt = source_recovery_candidate(supervisor, manifest)
    return supervisor, manifest, revised, receipt_path, receipt


def test_source_recovery_preserves_counters_history_and_registers_before_pointer(recovery_case):
    supervisor, manifest, revised, receipt_path, receipt = recovery_case
    before = campaign.read_json(supervisor.state_path)
    old_version = Path(before['manifest_path'])
    old_bytes = old_version.read_bytes()
    old_sources = {name: (Path(manifest['jobs'][0]['cwd']) / name).read_bytes()
                   for name in manifest['jobs'][0]['source_files']}
    writes = []
    original_write = campaign.atomic_json

    def observe_write(path, value):
        if Path(path) == supervisor.state_path:
            assert campaign.read_json(value['manifest_path']) == revised
            assert campaign.read_json(supervisor.state_path) == before
        writes.append(Path(path))
        original_write(path, value)

    with patch.object(campaign, 'atomic_json', side_effect=observe_write):
        result = supervisor.authorize_recovery('job-0', receipt_path, revised)
    record = result['jobs']['job-0']
    assert record['status'] == 'PENDING'
    assert record['attempts'] == record['failure_count'] == 2
    assert record['recovery_attempts'] == 1
    assert record['history'][:-1] == before['jobs']['job-0']['history']
    assert result['jobs']['job-1'] == before['jobs']['job-1']
    assert result['ownership_handoffs'] == before['ownership_handoffs']
    assert result['queue_owner'] == before['queue_owner']
    assert supervisor.load()[0] == revised
    assert campaign.read_json(supervisor.manifest_path) == manifest
    assert old_version.read_bytes() == old_bytes
    assert {name: (Path(manifest['jobs'][0]['cwd']) / name).read_bytes()
            for name in old_sources} == old_sources
    assert writes == [Path(result['manifest_path']), supervisor.state_path]
    revision = result['manifest_revisions'][-1]
    history = record['history'][-1]
    for audit in (revision, history):
        assert audit['original_job_hash'] == receipt['job_hash']
        assert audit['replacement_job_hash'] == receipt['replacement_job_hash']
        assert audit['original_source_hash'] == manifest['jobs'][0]['source_hash']
        assert audit['replacement_source_hash'] == receipt['replacement_source_hash']
        assert audit['analysis_receipt_hash'] == campaign.file_hash(receipt_path)
        assert audit['model_execution_receipt'] == receipt['model_execution_receipt']
        assert audit['previous_manifest_hash'] == before['manifest_hash']
        assert audit['manifest_hash'] == result['manifest_hash']
        assert audit['code_fix_explanation'] == receipt['code_fix_explanation']
    durable = supervisor.state_path.read_bytes()
    assert supervisor.authorize_recovery('job-0', receipt_path, revised)['manifest_hash'] == result['manifest_hash']
    assert supervisor.authorize_recovery('job-0', receipt_path)['manifest_hash'] == result['manifest_hash']
    assert supervisor.state_path.read_bytes() == durable


@pytest.mark.parametrize('field,value', [
    ('config_hash', 'changed-config'), ('data_hash', 'changed-data'), ('model_hash', 'changed-model'),
    ('argv', ['/bin/python', '-m', 'mediflow_medvlm.recovery_fixture', '--changed']),
    ('dependencies', ['job-1']), ('run_id', 'changed-run'), ('cell_id', 'T99'),
    ('phase', 'V7_TRAIN'), ('success_receipt', '/tmp/another-receipt.json'),
    ('uses_gpu', True), ('uses_gpu', 0), ('gpu_lock_mode', 'child'),
    ('implementation_pending', True), ('extra_protocol', None),
])
def test_source_recovery_rejects_non_source_fields_even_with_bound_receipt(recovery_case, field, value):
    supervisor, _, revised, receipt_path, receipt = recovery_case
    before = supervisor.state_path.read_bytes()
    revised['jobs'][0][field] = value
    receipt['replacement_job_hash'] = campaign.canonical_hash(revised['jobs'][0])
    campaign.atomic_json(receipt_path, receipt)
    with pytest.raises(ValueError, match='only cwd, source_files, and source_hash'):
        supervisor.authorize_recovery('job-0', receipt_path, revised)
    assert supervisor.state_path.read_bytes() == before
    assert len(list((supervisor.root / 'manifests').glob('*.json'))) == 1


@pytest.mark.parametrize('change', ['metadata', 'other-job', 'reorder', 'remove', 'add'])
def test_source_recovery_rejects_campaign_or_other_job_changes(recovery_case, change):
    supervisor, _, revised, receipt_path, _ = recovery_case
    before = supervisor.state_path.read_bytes()
    if change == 'metadata':
        revised['scheduler_identity']['id'] = 'another-scheduler'
    elif change == 'other-job':
        revised['jobs'][1]['data_hash'] = 'changed-other-data'
    elif change == 'reorder':
        revised['jobs'].reverse()
    elif change == 'remove':
        revised['jobs'].pop()
    else:
        revised['jobs'].append({**revised['jobs'][1], 'job_id': 'job-2'})
    with pytest.raises(ValueError, match='recovery cannot'):
        supervisor.authorize_recovery('job-0', receipt_path, revised)
    assert supervisor.state_path.read_bytes() == before


@pytest.mark.parametrize('field,value', [
    ('job_hash', 'stale-original'), ('error_signature', 'stale-failure'), ('failure_count', 1),
    ('replacement_job_hash', 'stale-replacement'), ('replacement_source_hash', 'stale-source'),
    ('code_fix_explanation', ''), ('code_fix_explanation', '   '), ('code_fix_explanation', None),
])
def test_source_recovery_rejects_stale_receipts_and_missing_fix_explanation(recovery_case, field, value):
    supervisor, _, revised, receipt_path, receipt = recovery_case
    before = supervisor.state_path.read_bytes()
    receipt[field] = value
    campaign.atomic_json(receipt_path, receipt)
    with pytest.raises(ValueError, match='does not bind|nonempty code_fix_explanation'):
        supervisor.authorize_recovery('job-0', receipt_path, revised)
    assert supervisor.state_path.read_bytes() == before


@pytest.mark.parametrize('status,failures', [('PENDING', 1), ('RUNNING', 2), ('FAILED', 3)])
def test_source_recovery_requires_exact_second_failure_hold(recovery_case, status, failures):
    supervisor, _, revised, receipt_path, _ = recovery_case
    state = campaign.read_json(supervisor.state_path)
    state['jobs']['job-0'].update(status=status, failure_count=failures)
    campaign.atomic_json(supervisor.state_path, state)
    before = supervisor.state_path.read_bytes()
    with pytest.raises(ValueError, match='second-failure analysis hold'):
        supervisor.authorize_recovery('job-0', receipt_path, revised)
    assert supervisor.state_path.read_bytes() == before


def test_source_recovery_rejects_unchanged_or_tampered_sources(recovery_case):
    supervisor, manifest, revised, receipt_path, _ = recovery_case
    unchanged = copy.deepcopy(revised)
    unchanged['jobs'][0]['source_files'] = manifest['jobs'][0]['source_files']
    unchanged['jobs'][0]['source_hash'] = manifest['jobs'][0]['source_hash']
    with pytest.raises(ValueError, match='changed source bytes'):
        supervisor.authorize_recovery('job-0', receipt_path, unchanged)
    source = Path(revised['jobs'][0]['cwd']) / 'src/mediflow_medvlm/recovery_fixture.py'
    source.write_text('raise RuntimeError("tampered after review")')
    with pytest.raises(ValueError, match='source snapshot changed'):
        supervisor.authorize_recovery('job-0', receipt_path, revised)
    assert supervisor.status()['jobs']['job-0']['status'] == 'ANALYSIS_REQUIRED'


def test_replacement_receipt_cannot_authorize_original_source_without_manifest(recovery_case):
    supervisor, _, _, receipt_path, _ = recovery_case
    with pytest.raises(ValueError, match='requires a revised manifest'):
        supervisor.authorize_recovery('job-0', receipt_path)
    assert supervisor.status()['jobs']['job-0']['status'] == 'ANALYSIS_REQUIRED'


def test_source_recovery_idempotency_rejects_conflicting_input_and_active_identity(recovery_case):
    supervisor, _, revised, receipt_path, _ = recovery_case
    supervisor.authorize_recovery('job-0', receipt_path, revised)
    durable = supervisor.state_path.read_bytes()
    conflict = copy.deepcopy(revised)
    conflict['jobs'][0]['config_hash'] = 'unapproved-config'
    with pytest.raises(ValueError, match='conflicting revised manifest'):
        supervisor.authorize_recovery('job-0', receipt_path, conflict)
    assert supervisor.state_path.read_bytes() == durable
    # Simulate a reverted state pointer: the same receipt must never silently pass.
    state = campaign.read_json(supervisor.state_path)
    original_hash = state['manifest_revisions'][-1]['previous_hash']
    state.update(manifest_hash=original_hash, manifest_path=str(supervisor.root / 'manifests' / (original_hash + '.json')))
    campaign.atomic_json(supervisor.state_path, state)
    with pytest.raises(ValueError, match='active authorized job'):
        supervisor.authorize_recovery('job-0', receipt_path, revised)


@pytest.mark.parametrize('tamper', ['source', 'model-evidence'])
def test_source_recovery_rechecks_bytes_and_model_evidence_on_repeated_receipt(recovery_case, tamper):
    supervisor, _, revised, receipt_path, receipt = recovery_case
    supervisor.authorize_recovery('job-0', receipt_path, revised)
    path = (Path(revised['jobs'][0]['cwd']) / 'src/mediflow_medvlm/recovery_fixture.py'
            if tamper == 'source' else Path(receipt['model_execution_receipt']['path']))
    path.write_text('changed after authorization')
    with pytest.raises(ValueError, match='source snapshot changed|execution evidence is missing or changed'):
        supervisor.authorize_recovery('job-0', receipt_path, revised)


def test_source_recovery_restores_after_interrupted_state_update(recovery_case):
    supervisor, _, revised, receipt_path, _ = recovery_case
    before = supervisor.state_path.read_bytes()
    original_write = campaign.atomic_json

    def interrupt_state_update(path, value):
        if Path(path) == supervisor.state_path:
            assert campaign.read_json(value['manifest_path']) == revised
            raise OSError('fixture crash before state pointer update')
        original_write(path, value)

    with patch.object(campaign, 'atomic_json', side_effect=interrupt_state_update):
        with pytest.raises(OSError, match='fixture crash'):
            supervisor.authorize_recovery('job-0', receipt_path, revised)
    assert supervisor.state_path.read_bytes() == before
    version = supervisor.root / 'manifests' / (campaign.canonical_hash(revised) + '.json')
    assert campaign.read_json(version) == revised
    original_time = version.stat().st_mtime_ns
    resumed = campaign.Supervisor(supervisor.data_root).authorize_recovery('job-0', receipt_path, revised)
    assert resumed['jobs']['job-0']['attempts'] == resumed['jobs']['job-0']['failure_count'] == 2
    assert len(resumed['manifest_revisions']) == 1
    assert version.stat().st_mtime_ns == original_time


def test_source_recovery_never_overwrites_corrupt_existing_manifest_version(recovery_case):
    supervisor, _, revised, receipt_path, _ = recovery_case
    version = supervisor.root / 'manifests' / (campaign.canonical_hash(revised) + '.json')
    campaign.atomic_json(version, {'corrupt': True})
    before = supervisor.state_path.read_bytes()
    with pytest.raises(ValueError, match='immutable recovery manifest version changed'):
        supervisor.authorize_recovery('job-0', receipt_path, revised)
    assert campaign.read_json(version) == {'corrupt': True}
    assert supervisor.state_path.read_bytes() == before


def test_source_recovery_uses_existing_supervisor_lock(recovery_case):
    supervisor, _, revised, receipt_path, _ = recovery_case
    with campaign.try_lock(supervisor.root / 'supervisor.lock'):
        assert supervisor.authorize_recovery('job-0', receipt_path, revised) == {'status': 'SUPERVISOR_BUSY'}
    assert supervisor.status()['jobs']['job-0']['status'] == 'ANALYSIS_REQUIRED'


def test_source_recovery_cli_accepts_optional_manifest(recovery_case, capsys):
    supervisor, _, revised, receipt_path, _ = recovery_case
    path = supervisor.root / 'revised-input.json'
    campaign.atomic_json(path, revised)
    assert campaign.main(['--data-root', str(supervisor.data_root), 'authorize-recovery',
        '--job-id', 'job-0', '--analysis-receipt', str(receipt_path), '--manifest', str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['manifest_hash'] == campaign.canonical_hash(revised)
    assert result['jobs']['job-0']['failure_count'] == 2


def test_source_recovery_final_third_failure_is_terminal_without_reset(setup_campaign):
    supervisor, manifest = setup_campaign('fail', module=True)
    supervisor.tick()
    assert wait_exit(supervisor)['exit_code'] == 7
    supervisor.tick()
    assert wait_exit(supervisor)['exit_code'] == 7
    held = supervisor.tick()
    assert held['jobs']['job-0']['status'] == 'ANALYSIS_REQUIRED'
    revised, receipt_path, _ = source_recovery_candidate(supervisor, manifest)
    supervisor.authorize_recovery('job-0', receipt_path, revised)
    launched = supervisor.tick()
    assert launched['jobs']['job-0']['attempts'] == 3
    launch = campaign.read_json(Path(launched['jobs']['job-0']['attempt_dir']) / 'launch.json')
    assert launch['job'] == revised['jobs'][0]
    assert launch['job']['argv'] == manifest['jobs'][0]['argv']
    assert wait_exit(supervisor)['exit_code'] == 9
    final = campaign.Supervisor(supervisor.data_root).tick()
    assert final['jobs']['job-0']['status'] == 'FAILED'
    assert final['jobs']['job-0']['attempts'] == final['jobs']['job-0']['failure_count'] == 3
    assert final['jobs']['job-0']['recovery_attempts'] == 2
    assert final['jobs']['job-0']['history'][:2] == held['jobs']['job-0']['history']
    assert supervisor.tick()['jobs']['job-0']['attempts'] == 3
    # An idempotent old authorization must also never reopen a terminal job.
    repeated = supervisor.authorize_recovery('job-0', receipt_path, revised)
    assert repeated['jobs']['job-0']['status'] == 'FAILED'


@pytest.mark.parametrize('attempts', [0, 1, 3])
def test_source_recovery_requires_exactly_two_started_attempts(recovery_case, attempts):
    supervisor, _, revised, receipt_path, _ = recovery_case
    state = campaign.read_json(supervisor.state_path)
    state['jobs']['job-0']['attempts'] = attempts
    campaign.atomic_json(supervisor.state_path, state)
    with pytest.raises(ValueError, match='second-failure analysis hold'):
        supervisor.authorize_recovery('job-0', receipt_path, revised)


def test_source_recovery_rejects_same_snapshot_and_resolved_alias(recovery_case):
    supervisor, manifest, revised, receipt_path, receipt = recovery_case
    original_root = Path(manifest['jobs'][0]['cwd'])
    alias = original_root.with_name('snapshot-alias')
    alias.symlink_to(original_root, target_is_directory=True)
    for root in (original_root, alias):
        candidate = copy.deepcopy(revised)
        candidate['jobs'][0]['cwd'] = str(root)
        receipt['replacement_job_hash'] = campaign.canonical_hash(candidate['jobs'][0])
        campaign.atomic_json(receipt_path, receipt)
        with pytest.raises(ValueError, match='different snapshot cwd'):
            supervisor.authorize_recovery('job-0', receipt_path, candidate)
    assert supervisor.status()['jobs']['job-0']['status'] == 'ANALYSIS_REQUIRED'



def test_source_recovery_rejects_mutated_failed_snapshot(recovery_case):
    supervisor, manifest, revised, receipt_path, _ = recovery_case
    source = Path(manifest['jobs'][0]['cwd']) / 'src/mediflow_medvlm/recovery_fixture.py'
    source.write_text('raise RuntimeError("old snapshot was overwritten")')
    with pytest.raises(ValueError, match='source snapshot changed'):
        supervisor.authorize_recovery('job-0', receipt_path, revised)
    assert supervisor.status()['jobs']['job-0']['status'] == 'ANALYSIS_REQUIRED'


@pytest.mark.parametrize('content', [None, 'null'])
def test_recovery_cli_never_treats_invalid_manifest_as_omitted(recovery_case, content):
    supervisor, _, _, receipt_path, _ = recovery_case
    path = supervisor.root / 'invalid-revision.json'
    if content is not None:
        path.write_text(content)
    with pytest.raises(ValueError, match='must contain a manifest object'):
        campaign.main(['--data-root', str(supervisor.data_root), 'authorize-recovery',
            '--job-id', 'job-0', '--analysis-receipt', str(receipt_path), '--manifest', str(path)])
    assert supervisor.status()['jobs']['job-0']['status'] == 'ANALYSIS_REQUIRED'
