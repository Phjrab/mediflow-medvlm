"""CPU-only manifest extension, legacy audit view and native ownership tests."""
import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from mediflow_medvlm import campaign_supervisor as engine

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('completion_owner', ROOT / 'scripts/campaign_control_v6_completion.py')
owner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(owner)


def write(path, value):
    engine.atomic_json(path, value)
    return path


class Legacy:
    def _manifest(self, value):
        owner.require(value == self.legacy_manifest, 'legacy manifest changed')

    def _check_original_records(self, state):
        owner.require(state['jobs'] == self.legacy_records, 'legacy records changed')


@pytest.fixture
def fixture(tmp_path):
    data = tmp_path / 'data'
    supervisor = engine.Supervisor(data)
    root = supervisor.root
    initial = {'campaign_id': 'cpu-fixture', 'jobs': []}
    for key in ['v6-family_selection', 'v6-additional_seeds', 'v6-test', 'v6-report', 'v7-P0', 'v7-P1', 'v7-R0', 'v7-R1', 'v7-report']:
        initial['jobs'].append({'job_id': key, 'run_id': key, 'phase': key.removeprefix('v6-'),
            'cell_id': 'ALL_V6' if key.startswith('v6') else 'ALL_V7', 'implementation_pending': True,
            'dependencies': [], 'dependency_policy': 'success'})
    supervisor.initialize(initial)
    state = engine.read_json(supervisor.state_path)
    source = root / 'snapshots' / ('1' * 40)
    wrapper = source / 'src/mediflow_medvlm/campaign_supervisor.py'
    module = source / 'src/mediflow_medvlm/campaign_family_selection.py'
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text('# pinned CPU fixture native wrapper\n')
    module.write_text('# pinned CPU fixture selection\n')
    files = {str(p.relative_to(source)): engine.file_hash(p) for p in [wrapper, module]}
    receipt = root / 'receipts/v6-family_selection.json'
    config = {'job_id': 'v6-family_selection', 'run_id': 'v6-family_selection', 'source_files': files,
              'source_hash': engine.canonical_hash(files), 'data': {'validation': 'fixture'},
              'data_hash': engine.canonical_hash({'validation': 'fixture'}), 'model_hash': 'fixed-model',
              'success_receipt': str(receipt)}
    config['config_hash'] = engine.canonical_hash(config)
    config_path = write(root / 'configs/selection.json', config)
    replacement = copy.deepcopy(initial)
    job = replacement['jobs'][0]
    job.update(implementation_pending=False, uses_gpu=False, requires_gpu=False, gpu_lock_mode='wrapper',
               argv=['/usr/bin/python3', '-m', 'mediflow_medvlm.campaign_family_selection', '--config', str(config_path), '--receipt', str(receipt)],
               cwd=str(source), success_receipt=str(receipt), source_files=files,
               **{k: config[k] for k in engine.HASH_FIELDS})
    replacement_path = write(root / 'manifests/proposal.json', replacement)
    parent_path = write(root / 'manifests/parent.json', initial)
    deployment = write(tmp_path / 'deployment.json', {'data_root': str(data)})
    proof = write(tmp_path / 'cpu-review.json', {'cpu_only': True, 'passed': True})
    binding = owner.prepare_binding(deployment, parent_path, replacement_path, supervisor.state_path,
        activated=['v6-family_selection'], inserted=[], config_paths=[config_path], review_paths=[proof])
    binding_path = write(root / 'binding.json', binding)
    class Control(owner.CompletionMixin, Legacy):
        def _verify_worker_job(self, job, config_ref):
            # Scientific worker tests use actual production configs separately.
            # These native-ownership fixtures isolate the manifest transaction.
            assert job['argv'][2] == 'mediflow_medvlm.campaign_family_selection'
    control = Control()
    control.root, control.data_root, control.engine, control.supervisor = root, data, engine, supervisor
    control.legacy_manifest, control.legacy_records = initial, copy.deepcopy(state['jobs'])
    control.completion_parent_deployment = owner.reference(deployment)
    control.completion_refs = [owner.reference(binding_path)]
    owner.install_supervisor(control)
    return SimpleNamespace(control=control, state=state, initial=initial, replacement=replacement,
        binding=binding, binding_path=binding_path, config=config, config_path=config_path,
        proof=proof, source=source, job=job)


def activate(f):
    return f.control.supervisor.refine_completion(f.control.completion_refs[0], f.binding, f.initial, f.replacement)


def test_native_registration_preserves_every_old_record_and_v7(fixture):
    f = fixture
    result = activate(f)
    assert result['jobs']['v6-family_selection']['status'] == 'PENDING'
    for key, original in f.state['jobs'].items():
        actual = result['jobs'][key]
        assert {k: v for k, v in actual.items() if k != 'status'} == {k: v for k, v in original.items() if k != 'status'}
        if key != 'v6-family_selection':
            assert actual == original
    assert len([e for e in result['manifest_revisions'] if e.get('kind') == owner.EVENT]) == 1
    actual_manifest, actual_state = engine.Supervisor.load(f.control.supervisor)
    assert actual_manifest == f.replacement
    assert actual_state['jobs']['v6-family_selection']['status'] == 'PENDING'
    f.control._check_original_records(actual_state)  # Explicit compatibility view only.
    assert actual_state['jobs']['v6-family_selection']['status'] == 'PENDING'


def test_native_registration_idempotent(fixture):
    f = fixture
    activate(f)
    raw = f.control.supervisor.state_path.read_bytes()
    activate(f)
    assert f.control.supervisor.state_path.read_bytes() == raw


def test_canonical_supervisor_busy_has_zero_mutations(fixture):
    f = fixture
    raw = f.control.supervisor.state_path.read_bytes()
    with engine.try_lock(f.control.root / 'supervisor.lock'):
        assert activate(f) == {'status': 'SUPERVISOR_BUSY'}
    assert f.control.supervisor.state_path.read_bytes() == raw


@pytest.mark.parametrize('field', ['campaign_id', 'jobs'])
def test_protected_campaign_or_v7_declarations_cannot_change(fixture, field):
    f = fixture
    replacement = copy.deepcopy(f.replacement)
    if field == 'jobs':
        replacement['jobs'][-1]['dependencies'] = ['v6-report']
    else:
        replacement[field] = 'foreign-campaign'
    with pytest.raises(ValueError):
        owner.validate_pair(engine, f.initial, replacement, activated=['v6-family_selection'], inserted=[])


def test_v7_activation_rejected(fixture):
    f = fixture
    with pytest.raises(ValueError, match='only explicit V6'):
        owner.validate_pair(engine, f.initial, f.replacement, activated=['v7-P0'], inserted=[])


@pytest.mark.parametrize('change', ['source', 'config', 'review'])
def test_pinned_evidence_tamper_rejected_before_state_mutation(fixture, change):
    f = fixture
    target = {'source': f.source / 'src/mediflow_medvlm/campaign_family_selection.py',
              'config': f.config_path, 'review': f.proof}[change]
    target.write_text(target.read_text() + '\n# changed\n')
    raw = f.control.supervisor.state_path.read_bytes()
    with pytest.raises((ValueError, OSError)):
        activate(f)
    assert f.control.supervisor.state_path.read_bytes() == raw


def test_counter_or_record_change_refuses_stale_prepared_registration(fixture):
    f = fixture
    state = engine.read_json(f.control.supervisor.state_path)
    state['jobs']['v6-family_selection']['attempts'] = 1
    write(f.control.supervisor.state_path, state)
    raw = f.control.supervisor.state_path.read_bytes()
    with pytest.raises(ValueError, match='parent state'):
        activate(f)
    assert f.control.supervisor.state_path.read_bytes() == raw


def test_extension_event_required_for_actual_registered_manifest(fixture):
    f = fixture
    activate(f)
    state = engine.read_json(f.control.supervisor.state_path)
    state['manifest_revisions'] = [e for e in state['manifest_revisions'] if e.get('kind') != owner.EVENT]
    write(f.control.supervisor.state_path, state)
    with pytest.raises(ValueError, match='registered completion audit'):
        f.control.supervisor.load()


def test_other_legacy_record_cannot_be_hidden_by_anchor_audit_view(fixture):
    f = fixture
    activate(f)
    state = engine.read_json(f.control.supervisor.state_path)
    state['jobs']['v7-P0']['attempts'] = 1
    write(f.control.supervisor.state_path, state)
    with pytest.raises(ValueError, match='legacy records'):
        f.control.supervisor.load()


def test_command_cannot_route_to_unapproved_worker(fixture):
    f = fixture
    replacement = copy.deepcopy(f.replacement)
    replacement['jobs'][0]['argv'][2] = 'arbitrary.module'
    with pytest.raises(ValueError):
        owner.validate_pair(engine, f.initial, replacement, activated=['v6-family_selection'], inserted=[])


def test_missing_existing_prerequisite_cannot_be_removed(fixture):
    f = fixture
    before, after = copy.deepcopy(f.initial), copy.deepcopy(f.replacement)
    before['jobs'][0]['dependencies'] = ['v6-additional_seeds']
    with pytest.raises(ValueError, match='prerequisite removed'):
        owner.validate_pair(engine, before, after, activated=['v6-family_selection'], inserted=[])


def test_scientific_verifier_uses_isolated_pinned_worker_and_exact_job(fixture):
    f = fixture
    job = copy.deepcopy(f.job)
    job['argv'][0] = sys.executable
    package = f.source / 'src/mediflow_medvlm'
    (package / '__init__.py').write_text('')
    (package / 'campaign_family_selection.py').write_text(
        'def job_from_config(path, *, repo, python):\n    return ' + repr(job) + '\n')
    owner.CompletionMixin._verify_worker_job(f.control, job, owner.reference(f.config_path))
    changed = copy.deepcopy(job)
    changed['uses_gpu'] = True
    with pytest.raises(ValueError, match='exact job closure'):
        owner.CompletionMixin._verify_worker_job(f.control, changed, owner.reference(f.config_path))


def test_scientific_verifier_refuses_other_interpreter(fixture):
    fixture.job['argv'][0] = '/definitely/not/the/pinned/python'
    with pytest.raises(ValueError, match='same pinned Python'):
        owner.CompletionMixin._verify_worker_job(fixture.control, fixture.job, owner.reference(fixture.config_path))


def extend_second_anchor(f):
    activate(f)
    parent, state = engine.Supervisor.load(f.control.supervisor)
    replacement = copy.deepcopy(parent)
    config = copy.deepcopy(f.config)
    key = 'v6-additional_seeds'
    config.update(job_id=key, run_id=key, success_receipt=str(f.control.root / 'receipts' / (key + '.json')))
    config['config_hash'] = engine.canonical_hash({k: v for k, v in config.items() if k != 'config_hash'})
    path = write(f.control.root / 'configs/additional.json', config)
    job = copy.deepcopy(f.job)
    job.update(job_id=key, run_id=key, phase='additional_seeds', config_hash=config['config_hash'],
               success_receipt=config['success_receipt'])
    job['argv'][4], job['argv'][6] = str(path), config['success_receipt']
    replacement['jobs'][1] = job
    proposal = write(f.control.root / 'manifests/second-proposal.json', replacement)
    deployment = Path(f.control.completion_parent_deployment['path'])
    parent_path = Path(state['manifest_path'])
    binding = owner.prepare_binding(deployment, parent_path, proposal, f.control.supervisor.state_path,
        activated=[key], inserted=[], config_paths=[path], review_paths=[f.proof],
        previous_binding=f.control.completion_refs[-1])
    binding_path = write(f.control.root / 'second-binding.json', binding)
    ref = owner.reference(binding_path)
    f.control.completion_refs.append(ref)
    f.control._completion_cache = None
    f.control.supervisor.refine_completion(ref, binding, parent, replacement)
    return replacement


def test_all_applied_refinement_events_remain_required(fixture):
    f = fixture
    extend_second_anchor(f)
    state = engine.read_json(f.control.supervisor.state_path)
    first = f.control.completion_refs[0]['sha256']
    state['manifest_revisions'] = [e for e in state['manifest_revisions'] if e.get('binding_sha256') != first]
    write(f.control.supervisor.state_path, state)
    with pytest.raises(ValueError, match='every applied prefix'):
        f.control.supervisor.load()


def test_new_source_failure_holds_pending_sibling_without_resetting_counts(fixture):
    f = fixture
    extend_second_anchor(f)
    state = engine.read_json(f.control.supervisor.state_path)
    record = state['jobs']['v6-family_selection']
    record.update(status='RUNNING', attempts=5, failure_count=4, recovery_attempts=4,
                  attempt_dir=str(f.control.root / 'attempts/v6-family_selection/0005'))
    prior_history = copy.deepcopy(record['history'])
    f.control.supervisor._ll_state = state
    f.control.supervisor._failure(record, 'known source defect', 1)
    assert record['failure_count'] == 5 and record['attempts'] == 5 and record['recovery_attempts'] == 4
    assert record['status'] == 'ANALYSIS_REQUIRED' and record['source_repair_required']
    assert record['history'][:len(prior_history)] == prior_history
    sibling = state['jobs']['v6-additional_seeds']
    assert sibling['status'] == 'ANALYSIS_REQUIRED' and sibling['source_repair_required']
    assert sibling['attempts'] == sibling['failure_count'] == sibling['recovery_attempts'] == 0
