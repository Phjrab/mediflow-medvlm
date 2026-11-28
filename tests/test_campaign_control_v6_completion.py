"""CPU-only manifest extension, legacy audit view and native ownership tests."""
import copy
from contextlib import contextmanager
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
    for index in range(88):
        key = 'protected-' + str(index)
        initial['jobs'].append({'job_id':key,'run_id':key,'phase':'cpu-fixture','cell_id':'PROTECTED',
            'implementation_pending':True,'dependencies':[],'dependency_policy':'success'})
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


def test_child_only_freeze_registration_preserves_every_existing_record(fixture):
    """The real freeze must finish before native evaluation configs can exist."""
    f = fixture
    activate(f)
    parent, state = engine.Supervisor.load(f.control.supervisor)
    before = copy.deepcopy(state['jobs'])
    replacement = copy.deepcopy(parent)
    key = 'v6-test-freeze'
    config = copy.deepcopy(f.config)
    config.update(job_id=key, run_id=key,
                  success_receipt=str(f.control.root / 'receipts' / (key + '.json')))
    config['config_hash'] = engine.canonical_hash({k: v for k, v in config.items() if k != 'config_hash'})
    path = write(f.control.root / 'configs/test-freeze.json', config)
    child = copy.deepcopy(f.job)
    child.update(job_id=key, run_id=key, phase='test_freeze',
                 config_hash=config['config_hash'], success_receipt=config['success_receipt'])
    child['argv'][4], child['argv'][6] = str(path), config['success_receipt']
    replacement['jobs'].insert(2, child)
    proposal = write(f.control.root / 'manifests/freeze-proposal.json', replacement)
    binding = owner.prepare_binding(Path(f.control.completion_parent_deployment['path']),
        Path(state['manifest_path']), proposal, f.control.supervisor.state_path,
        activated=[], inserted=[key], config_paths=[path], review_paths=[f.proof],
        previous_binding=f.control.completion_refs[-1])
    ref = owner.reference(write(f.control.root / 'freeze-binding.json', binding))
    f.control.completion_refs.append(ref)
    f.control._completion_cache = None
    result = f.control.supervisor.refine_completion(ref, binding, parent, replacement)
    assert all(result['jobs'][job] == record for job, record in before.items())
    assert result['jobs'][key] == {'status': 'PENDING', 'attempts': 0, 'failure_count': 0,
                                  'recovery_attempts': 0, 'history': []}
    assert binding['original_anchor_records'] == {}
    assert result['jobs']['v6-test']['status'] == 'IMPLEMENTATION_PENDING'
    assert result['jobs']['v6-report']['status'] == 'IMPLEMENTATION_PENDING'
    raw = f.control.supervisor.state_path.read_bytes()
    f.control.supervisor.refine_completion(ref, binding, parent, replacement)
    assert f.control.supervisor.state_path.read_bytes() == raw
    f.control._check_original_records(engine.read_json(f.control.supervisor.state_path))


def test_empty_refinement_cannot_create_an_audit_event(fixture):
    f = fixture
    with pytest.raises(ValueError, match='only explicit V6'):
        owner.validate_pair(engine, f.initial, f.initial, activated=[], inserted=[])


def test_child_only_refinement_cannot_change_existing_test_anchor(fixture):
    f = fixture
    proposal = copy.deepcopy(f.initial)
    proposal['jobs'][2]['dependencies'] = ['v6-family_selection']
    child = copy.deepcopy(f.initial['jobs'][0])
    child.update(job_id='v6-test-freeze', run_id='v6-test-freeze', phase='test_freeze')
    proposal['jobs'].insert(2, child)
    with pytest.raises(ValueError, match='existing protected job changed'):
        owner.validate_pair(engine, f.initial, proposal, activated=[], inserted=['v6-test-freeze'])


def test_interrupted_cpu_telemetry_preserves_failure_without_poisoning_gpu_science(fixture):
    f = fixture
    activate(f)
    parent, state = engine.Supervisor.load(f.control.supervisor)
    replacement = copy.deepcopy(parent)
    key = 'v6-test-telemetry-T2-tuned-20261114'
    config = copy.deepcopy(f.config)
    config.update(job_id=key, run_id=key,
                  success_receipt=str(f.control.root / 'receipts' / (key + '.json')))
    config['config_hash'] = engine.canonical_hash({k: v for k, v in config.items() if k != 'config_hash'})
    path = write(f.control.root / 'configs/telemetry.json', config)
    child = copy.deepcopy(f.job)
    child.update(job_id=key, run_id=key, phase='telemetry_import', kind='telemetry_import',
                 config_hash=config['config_hash'], success_receipt=config['success_receipt'])
    child['argv'][4], child['argv'][6] = str(path), config['success_receipt']
    replacement['jobs'].insert(2, child)
    proposal = write(f.control.root / 'manifests/telemetry-proposal.json', replacement)
    binding = owner.prepare_binding(Path(f.control.completion_parent_deployment['path']),
        Path(state['manifest_path']), proposal, f.control.supervisor.state_path,
        activated=[], inserted=[key], config_paths=[path], review_paths=[f.proof],
        previous_binding=f.control.completion_refs[-1])
    ref = owner.reference(write(f.control.root / 'telemetry-binding.json', binding))
    f.control.completion_refs.append(ref)
    f.control._completion_cache = None
    f.control.supervisor.refine_completion(ref, binding, parent, replacement)
    state = engine.read_json(f.control.supervisor.state_path)
    record = state['jobs'][key]
    record.update(status='RUNNING', attempts=1,
                  attempt_dir=str(f.control.root / ('attempts/' + key + '/0001')))
    sibling = state['jobs']['v6-family_selection']
    original = copy.deepcopy(sibling)
    f.control.supervisor._ll_state = state
    f.control.supervisor._failure(record, 'CPU importer interrupted after durable intent', 137)
    assert record['status'] == 'FAILED' and record['failure_count'] == record['attempts'] == 1
    assert record['recovery_attempts'] == 0 and record['exit_code'] == 137
    assert record['history'][0]['status'] == 'FAILED'
    assert record['history'][-1]['status'] == 'TELEMETRY_RECONSTRUCTION_FAILED'
    assert record['failure_domain'] == 'telemetry_reconstruction'
    assert record['telemetry_coverage'] == 'INDETERMINATE' and record['scientific_failure'] is False
    assert not record.get('source_repair_required')
    assert sibling == original and not sibling.get('source_repair_required')


@pytest.fixture
def repair_fixture(fixture):
    f = fixture; activate(f)
    parent,state = engine.Supervisor.load(f.control.supervisor)
    source = f.control.root/'snapshots'/('2'*40)
    names = ['src/mediflow_medvlm/campaign_supervisor.py',owner.REPAIR_SOURCE,
             *['src/mediflow_medvlm/cpu_fixture_'+str(i)+'.py' for i in range(34)]]
    for name in names:
        path = source/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_text('# original immutable CPU source\n')
    files = {name:owner.file_hash(source/name) for name in names}
    configs = {}; paths = []; jobs = []
    for key in owner.REPAIR_IDS:
        audit = key == owner.REPAIR_IDS[-1]
        data = {'original_base':True,'frozen_data':'same','recipe':'unchanged'}
        if audit: data['train_configs'] = [owner.reference(path) for path in paths]
        config = {'job_id':key,'run_id':key,'kind':'additional_audit' if audit else 'train',
            'source_files':files,'source_hash':owner.digest(files),'data':data,'data_hash':owner.digest(data),
            'model_hash':'fixed-model','success_receipt':str(f.control.root/'receipts'/(key+'.json')),
            'recipe':{'seed':20261114,'adapter_initialization':None}}
        config['config_hash'] = owner.digest(config)
        path = write(f.control.root/'configs/original'/ (key+'.json'),config); paths.append(path); configs[key] = config
        jobs.append({'job_id':key,'run_id':key,'kind':config['kind'],'phase':'additional_seeds' if audit else 'train',
            'cell_id':'ALL_V6' if audit else key.split('-')[2],'dependencies':['v6-family_selection'],'dependency_policy':'success',
            'implementation_pending':False,'uses_gpu':not audit,'requires_gpu':not audit,'gpu_lock_mode':'wrapper',
            'argv':[sys.executable,'-m','mediflow_medvlm.campaign_v6_completion','--config',str(path),'--receipt',config['success_receipt']],
            'cwd':str(source),'success_receipt':config['success_receipt'],'source_files':files,**{k:config[k] for k in engine.HASH_FIELDS}})
    replacement = copy.deepcopy(parent)
    replacement['jobs'][1] = jobs[-1]
    replacement['jobs'][1:1] = jobs[:-1]
    proposal = write(f.control.root/'manifests/additional-proposal.json',replacement)
    deployment = Path(f.control.completion_parent_deployment['path'])
    binding = owner.prepare_binding(deployment,Path(state['manifest_path']),proposal,f.control.supervisor.state_path,
        activated=['v6-additional_seeds'],inserted=owner.REPAIR_TRAIN_IDS,config_paths=paths,review_paths=[f.proof],
        previous_binding=f.control.completion_refs[-1])
    ref = owner.reference(write(f.control.root/'additional-binding.json',binding))
    f.control.completion_refs.append(ref); f.control._completion_cache = None
    f.control._verify_worker_job = lambda job,ref:None
    f.control.supervisor.refine_completion(ref,binding,parent,replacement)
    parent,state = engine.Supervisor.load(f.control.supervisor)
    for index,key in enumerate(owner.REPAIR_IDS):
        state['jobs'][key].update(status='ANALYSIS_REQUIRED',source_repair_required=True,failed_source_hash=jobs[index]['source_hash'])
        state['jobs'][key]['history'].append({'status':'SOURCE_REPAIR_REQUIRED','failed_job_id':owner.REPAIR_IDS[0]})
    state['jobs'][owner.REPAIR_IDS[0]].update(attempts=5,failure_count=5,recovery_attempts=4,exit_code=1)
    write(f.control.supervisor.state_path,state)
    new_source = f.control.root/'snapshots'/('3'*40)
    for name in names:
        path = new_source/name; path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('# minimal CLI source repair\n' if name == owner.REPAIR_SOURCE else (source/name).read_text())
    new_files = {name:owner.file_hash(new_source/name) for name in names}
    revised = copy.deepcopy(parent); new_paths = []
    for key in owner.REPAIR_IDS:
        config = copy.deepcopy(configs[key]); config.update(source_files=new_files,source_hash=owner.digest(new_files))
        if key == owner.REPAIR_IDS[-1]: config['data']['train_configs'] = [owner.reference(path) for path in new_paths]
        config['data_hash'] = owner.digest(config['data']); config['config_hash'] = owner.digest({k:v for k,v in config.items() if k != 'config_hash'})
        path = write(f.control.root/'configs/repair'/(key+'.json'),config); new_paths.append(path)
        job = next(j for j in revised['jobs'] if j['job_id'] == key)
        job.update(cwd=str(new_source),source_files=new_files,**{k:config[k] for k in engine.HASH_FIELDS}); job['argv'][4] = str(path)
    proposal = write(f.control.root/'manifests/repair-proposal.json',revised)
    review = write(f.control.root/'repair-review.json',{'schema_version':owner.REPAIR_REVIEW_SCHEMA,'status':'CPU_VALIDATED',
        'cpu_only':True,'cpu_tests_passed':True,'job_ids':owner.REPAIR_IDS,'parent_source_hash':owner.digest(files),
        'replacement_source_hash':owner.digest(new_files),'reason':'minimal reload CLI result field correction','evidence':[owner.reference(f.proof)]})
    binding = owner.prepare_repair_binding(deployment,Path(state['manifest_path']),proposal,f.control.supervisor.state_path,
        config_paths=new_paths,review_paths=[f.proof],previous_binding=f.control.completion_refs[-1],repair_review_path=review)
    ref = owner.reference(write(f.control.root/'repair-binding.json',binding))
    f.control.completion_refs.append(ref); f.control._completion_cache = None
    def quiescent(state,*,allow_dead_running):
        for record in state['jobs'].values():
            owner.require(not engine.same_process(record.get('worker_identity')),'live worker')
            owner.require(allow_dead_running or record.get('status') not in {'STARTING','RUNNING'},'reconcile first')
    f.control._quiescent = quiescent
    f.control._verify = lambda:f.control._completion_chain()
    return SimpleNamespace(control=f.control,parent=parent,revised=revised,state=copy.deepcopy(state),binding=binding,ref=ref,
        new_source=new_source,new_paths=new_paths,review=review,proof=f.proof)


def repair(f):
    return f.control.supervisor.repair_completion_source(f.ref,f.binding,f.parent,f.revised)


def refresh_repair_state_pin(f,state):
    write(f.control.supervisor.state_path,state)
    f.binding['parent_state_sha256'] = owner.file_hash(f.control.supervisor.state_path)
    f.binding['parent_record_hashes'] = {key:owner.digest(record) for key,record in state['jobs'].items()}
    f.binding['parent_repair_records'] = {key:copy.deepcopy(state['jobs'][key]) for key in owner.REPAIR_IDS}
    write(Path(f.ref['path']),f.binding)
    f.ref = owner.reference(f.ref['path']); f.control.completion_refs[-1] = f.ref; f.control._completion_cache = None


def test_source_only_repair_preserves_105_records_all_counters_history_and_reuses_canonical_owner(repair_fixture):
    f = repair_fixture; result = repair(f)
    assert len(result['jobs']) == 105
    for key,before in f.state['jobs'].items():
        after = result['jobs'][key]
        if key not in owner.REPAIR_IDS: assert after == before
        else:
            assert after['status'] == 'PENDING' and after['source_repair_required'] is False
            assert all(after[k] == before[k] for k in ('attempts','failure_count','recovery_attempts'))
            assert after['history'][:-1] == before['history'] and after['history'][-1]['status'] == 'SOURCE_REPAIR_AUTHORIZED'
    assert len([e for e in result['manifest_revisions'] if e.get('kind') == owner.REPAIR_EVENT]) == 1
    raw = f.control.supervisor.state_path.read_bytes(); repair(f)
    assert f.control.supervisor.state_path.read_bytes() == raw


def test_public_refinement_route_uses_same_routed_locks_and_never_dispatches(repair_fixture):
    f = repair_fixture; calls = []
    @contextmanager
    def routed():
        calls.append('enter'); yield; calls.append('exit')
    f.control.routed_locks = routed
    f.control._registered = lambda:None
    f.control.supervisor.tick = lambda:pytest.fail('registration cannot dispatch')
    result = f.control.call('refine-completion',manifest=f.revised)
    assert calls == ['enter','exit'] and all(result['jobs'][key]['status'] == 'PENDING' for key in owner.REPAIR_IDS)


@pytest.mark.parametrize('name',['supervisor','worker','gpu'])
def test_source_repair_busy_canonical_lease_never_mutates(repair_fixture,name):
    f = repair_fixture; raw = f.control.supervisor.state_path.read_bytes()
    path = {'supervisor':f.control.root/'supervisor.lock','worker':f.control.root/'worker.lock',
            'gpu':f.control.data_root/'runtime/medvlm_gpu.lock'}[name]
    with engine.try_lock(path):
        if name == 'supervisor': assert repair(f) == {'status':'SUPERVISOR_BUSY'}
        else:
            with pytest.raises(ValueError,match='lease busy'): repair(f)
    assert f.control.supervisor.state_path.read_bytes() == raw


@pytest.mark.parametrize('tamper',['protected','metadata','train_data','recipe','extra_source','audit_data','scope','dependency'])
def test_source_repair_rejects_scientific_or_scope_tampering(repair_fixture,tamper):
    f = repair_fixture; revised = copy.deepcopy(f.revised)
    job = next(j for j in revised['jobs'] if j['job_id'] == owner.REPAIR_IDS[0])
    if tamper == 'protected': revised['jobs'][-1]['cell_id'] = 'changed'
    elif tamper == 'metadata': revised['campaign_id'] = 'foreign'
    elif tamper == 'dependency': job['dependencies'] = []
    elif tamper == 'scope': job['kind'] = 'evaluation'
    elif tamper == 'extra_source': job['source_files'] = {**job['source_files'],'src/mediflow_medvlm/cpu_fixture_0.py':'bad'}
    else:
        if tamper == 'audit_data': job = next(j for j in revised['jobs'] if j['job_id'] == owner.REPAIR_IDS[-1])
        config = owner.read(job['argv'][4])
        if tamper == 'recipe': config['recipe']['seed'] = 42
        else: config['data']['original_base'] = False
        config['data_hash'] = owner.digest(config['data']); config['config_hash'] = owner.digest({k:v for k,v in config.items() if k != 'config_hash'})
        write(Path(job['argv'][4]),config); job.update(**{k:config[k] for k in engine.HASH_FIELDS})
    with pytest.raises(ValueError): owner.validate_source_pair(engine,f.parent,revised,repaired=owner.REPAIR_IDS)


@pytest.mark.parametrize('tamper',['success','receipt','unheld','counter','live'])
def test_source_repair_refuses_success_unheld_changed_or_live_parent(repair_fixture,tamper):
    f = repair_fixture; state = engine.read_json(f.control.supervisor.state_path); key = owner.REPAIR_IDS[0]
    if tamper == 'receipt':
        path = next(j for j in f.revised['jobs'] if j['job_id'] == key)['success_receipt']; write(Path(path),{'status':'SUCCEEDED'})
    else:
        if tamper == 'success': state['jobs'][key]['status'] = 'SUCCEEDED'
        elif tamper == 'unheld': state['jobs'][key]['source_repair_required'] = False
        elif tamper == 'counter': state['jobs'][key]['attempts'] += 1
        else: state['jobs'][key]['worker_identity'] = engine.process_identity()
        write(f.control.supervisor.state_path,state)
    raw = f.control.supervisor.state_path.read_bytes()
    with pytest.raises(ValueError): repair(f)
    assert f.control.supervisor.state_path.read_bytes() == raw


@pytest.mark.parametrize('change',['success','unheld','live','wrong_source'])
def test_correctly_pinned_unsafe_parent_still_cannot_bypass_repair_guards(repair_fixture,change):
    f = repair_fixture; state = copy.deepcopy(f.state); key = owner.REPAIR_IDS[0]
    if change == 'success': state['jobs'][key]['status'] = 'SUCCEEDED'
    elif change == 'unheld': state['jobs'][key]['source_repair_required'] = False
    elif change == 'wrong_source': state['jobs'][key]['failed_source_hash'] = 'foreign'
    else: state['jobs'][key]['worker_identity'] = engine.process_identity()
    refresh_repair_state_pin(f,state); raw = f.control.supervisor.state_path.read_bytes()
    with pytest.raises(ValueError,match='held prior-source|live worker'): repair(f)
    assert f.control.supervisor.state_path.read_bytes() == raw


def test_raw_patch_and_test_evidence_pins_are_supported_and_rehashed(repair_fixture):
    f = repair_fixture; path = f.control.root/'minimal-source.diff'; path.write_text('CPU-only literal patch evidence\n')
    review = owner.read(f.review); review['evidence'].append(owner.reference(path)); write(f.review,review)
    f.binding['repair_review'] = owner.reference(f.review)
    write(Path(f.ref['path']),f.binding); f.ref = owner.reference(f.ref['path'])
    f.control.completion_refs[-1] = f.ref; f.control._completion_cache = None
    f.control._completion_chain()
    path.write_text('tampered\n')
    with pytest.raises(ValueError,match='pinned completion evidence changed'): repair(f)


def test_repair_history_and_counter_audit_allows_real_later_attempts_but_never_reset(repair_fixture):
    f = repair_fixture; repair(f); state = engine.read_json(f.control.supervisor.state_path); key = owner.REPAIR_IDS[0]
    state['jobs'][key].update(status='RUNNING',attempts=6,recovery_attempts=5)
    write(f.control.supervisor.state_path,state); f.control.supervisor.load()
    state['jobs'][key]['failure_count'] = 0; write(f.control.supervisor.state_path,state)
    with pytest.raises(ValueError,match='cannot be reset'): f.control.supervisor.load()


@pytest.mark.parametrize('outcome',['interrupted','succeeded'])
def test_later_native_terminal_history_preserves_original_epoch_and_repair_authorization(repair_fixture,monkeypatch,outcome):
    f = repair_fixture; repair(f); state = engine.read_json(f.control.supervisor.state_path); key = owner.REPAIR_IDS[0]
    original = copy.deepcopy(f.state['jobs'][key]); record = state['jobs'][key]
    record.update(status='RUNNING',attempts=6,recovery_attempts=5)
    if outcome == 'interrupted':
        f.control.supervisor._ll_state = state
        f.control.supervisor._failure(record,'new native child interrupted',137)
        assert record['status'] == 'ANALYSIS_REQUIRED' and record['failure_count'] == 6
    else:
        # Exercise only owner history validation; scientific success still
        # requires the existing native receipt/artifact verifier at runtime.
        record.update(status='SUCCEEDED',exit_code=0)
        record['history'].append({'attempt':6,'status':'SUCCEEDED','at':engine.now()})
    write(f.control.supervisor.state_path,state); actual = f.control.supervisor.load()[1]['jobs'][key]
    assert actual['attempts'] == 6 and actual['recovery_attempts'] == 5
    assert actual['failure_count'] >= original['failure_count']
    assert actual['history'][:len(original['history'])] == original['history']
    assert actual['history'][len(original['history'])]['status'] == 'SOURCE_REPAIR_AUTHORIZED'
    if outcome == 'interrupted':
        monkeypatch.setattr(engine.Supervisor,'tick',lambda self:pytest.fail('known failed new source must remain held'))
        raw = f.control.supervisor.state_path.read_bytes()
        f.control.supervisor.tick()
        assert f.control.supervisor.state_path.read_bytes() == raw


@pytest.mark.parametrize('tamper',['old_event','repair_event','authorization','counter_proof','parent_history'])
def test_full_refinement_and_repair_chain_audits_remain_required(repair_fixture,tamper):
    f = repair_fixture; repair(f); state = engine.read_json(f.control.supervisor.state_path); key = owner.REPAIR_IDS[0]
    if tamper == 'old_event': state['manifest_revisions'] = [e for e in state['manifest_revisions'] if e.get('kind') != owner.EVENT]
    elif tamper == 'repair_event': state['manifest_revisions'] = [e for e in state['manifest_revisions'] if e.get('kind') != owner.REPAIR_EVENT]
    elif tamper == 'counter_proof': state['manifest_revisions'][-1]['preserved_counters'][key]['attempts'] = 0
    elif tamper == 'parent_history': state['jobs'][key]['history'][0]['status'] = 'rewritten'
    else: state['jobs'][key]['history'][-1]['replacement_source_hash'] = 'forged'
    write(f.control.supervisor.state_path,state)
    with pytest.raises(ValueError): f.control.supervisor.load()


@pytest.fixture
def scientific_cache_fixture(tmp_path):
    """An isolated native-style validator with real transitive file readers."""
    import json
    repo = tmp_path / 'scientific-source'
    package = repo / 'src/mediflow_medvlm'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text('')
    module = package / 'campaign_v6_completion.py'
    module.write_text('''import contextvars,json,pathlib
_TRANSACTION=contextvars.ContextVar('fixture_transaction',default=None)
def job_from_config(path, *, repo, python):
    config=json.loads(pathlib.Path(path).read_text())
    state=_TRANSACTION.get()
    assert state is not None
    if config['second']:
        assert state['cache']['shared']=='science'
    else:
        state['cache']['shared']=pathlib.Path(config['science']).read_text()
    pathlib.Path(config['science']).parent.iterdir().__iter__().__next__()
    assert not pathlib.Path(config['absent']).exists()
    native=json.loads(pathlib.Path(config['state']).read_text())
    assert native['jobs']['prior']['status']=='SUCCEEDED'
    return config['job']
''')
    data = tmp_path / 'private'
    supervisor = engine.Supervisor(data)
    state = {'campaign_id': 'science-fixture', 'queue_owner': 'supervisor',
             'manifest_hash': 'same-manifest', 'manifest_path': str(tmp_path / 'manifest.json'),
             'manifest_revisions': [], 'jobs': {'prior': {'status': 'SUCCEEDED', 'attempts': 1},
                                               'next': {'status': 'PENDING', 'attempts': 0}}}
    write(supervisor.state_path, state)
    science = tmp_path / 'immutable-science/result.txt'
    science.parent.mkdir()
    science.write_text('science')
    absent = tmp_path / 'missing-intent.json'
    jobs = []
    for index in range(2):
        config_path = tmp_path / ('config-' + str(index) + '.json')
        job = {'job_id': 'cpu-' + str(index), 'cwd': str(repo),
               'argv': [sys.executable, '-m', 'mediflow_medvlm.campaign_v6_completion', '--config', str(config_path)]}
        write(config_path, {'job_id': job['job_id'], 'config_hash': str(index), 'job': job,
                            'second': bool(index), 'science': str(science),
                            'absent': str(absent), 'state': str(supervisor.state_path)})
        jobs.append((job, owner.reference(config_path)))
    control = owner.CompletionMixin()
    control.supervisor = supervisor
    return SimpleNamespace(control=control, jobs=jobs, state=state, science=science,
                           absent=absent, module=module)


def test_batch_science_validation_shared_transaction_once_then_memory_proof(scientific_cache_fixture, monkeypatch):
    f = scientific_cache_fixture
    owner.CompletionMixin._verify_worker_jobs(f.control, f.jobs)
    assert len(f.control._scientific_proofs) == 1
    def forbidden(*args, **kwargs):
        raise AssertionError('successful immutable proof must not rerun native validator')
    monkeypatch.setattr(owner.subprocess, 'run', forbidden)
    owner.CompletionMixin._verify_worker_jobs(f.control, f.jobs)
    state = copy.deepcopy(f.state)
    state['jobs']['next'].update(status='SUCCEEDED', attempts=1)
    state['jobs']['new-child'] = {'status': 'RUNNING', 'attempts': 1}
    state['updated_at'] = 'native-progress'
    write(f.control.supervisor.state_path, state)
    owner.CompletionMixin._verify_worker_jobs(f.control, f.jobs)


@pytest.mark.parametrize('change', ['science', 'config', 'terminal', 'manifest', 'directory', 'absent', 'symlink'])
def test_cached_transitive_science_change_fails_closed(scientific_cache_fixture, change):
    f = scientific_cache_fixture
    owner.CompletionMixin._verify_worker_jobs(f.control, f.jobs)
    if change == 'science':
        original = f.science.stat()
        f.science.write_text('changed')
        # Same bytes/mtime size is insufficient: ctime still detects local edits.
        import os
        os.utime(f.science, ns=(original.st_atime_ns, original.st_mtime_ns))
    elif change == 'config':
        Path(f.jobs[0][1]['path']).write_text('{}')
    elif change == 'directory':
        (f.science.parent / 'unexpected.txt').write_text('extra')
    elif change == 'absent':
        f.absent.write_text('{}')
    elif change == 'symlink':
        target = f.science.parent.parent / 'replacement.txt'
        target.write_text('science')
        f.science.unlink()
        f.science.symlink_to(target)
    else:
        state = copy.deepcopy(f.state)
        if change == 'terminal':
            state['jobs']['prior']['attempts'] = 2
        else:
            state['manifest_hash'] = 'foreign-manifest'
        write(f.control.supervisor.state_path, state)
    with pytest.raises((ValueError, OSError), match='scientific|symlink'):
        owner.CompletionMixin._verify_worker_jobs(f.control, f.jobs)


def service_fixture_control(tmp_path, call):
    supervisor = engine.Supervisor(tmp_path / 'private')
    supervisor.root.mkdir(parents=True, exist_ok=True)
    physical = supervisor.root / 'canonical-supervisor.lock'
    with engine.try_lock(physical):
        pass
    class Control:
        def _physical_lock(self, path):
            return engine.try_lock(path)
        def call(self, command):
            assert command == 'tick'
            with self._physical_lock(physical) as lease:
                assert lease is not None
                with engine.try_lock(physical) as competitor:
                    assert competitor is None  # Existing one-shot owner stays fenced.
                return call(self)
    control = Control()
    control.root, control.data_root, control.supervisor = supervisor.root, supervisor.data_root, supervisor
    control.engine, control.physical = engine, {'supervisor': physical}
    return control


def test_service_uses_canonical_lock_and_tick_releases_lock_on_stop(tmp_path):
    import threading
    stop = threading.Event()
    count = []
    def tick(control):
        count.append(1)
        if len(count) == 2:
            stop.set()
        return {'jobs': {'v6-report': {'status': 'IMPLEMENTATION_PENDING'},
                         'v7-report': {'status': 'IMPLEMENTATION_PENDING'}}}
    control = service_fixture_control(tmp_path, tick)
    original = control._physical_lock
    result = owner.serve(control, interval_seconds=.01, heartbeat_seconds=.01, stop_event=stop)
    assert result['ticks_completed'] == 2 and result['status'] == 'STOPPED'
    assert result['held_v6_jobs'] == {'v6-report': 'IMPLEMENTATION_PENDING'}
    assert result['scope'] == 'V6_ONLY' and result['v7_policy'] == 'MANUAL'
    assert control._physical_lock == original
    with engine.try_lock(control.physical['supervisor']) as lock:
        assert lock is not None
    status = owner.service_status(control.root / 'runtime/persistent-supervisor.json')
    assert status['telemetry_only'] and status['process_alive']
    assert status['heartbeat_age_seconds'] < 2


def test_duplicate_service_returns_busy_without_tick_or_telemetry_mutation(tmp_path):
    control = service_fixture_control(tmp_path, lambda _: pytest.fail('duplicate service cannot dispatch'))
    path = control.root / 'runtime/persistent-supervisor.json'
    write(path, {'original': True})
    before = path.read_bytes()
    with engine.try_lock(control.physical['supervisor']):
        assert owner.serve(control) == {'status': 'SUPERVISOR_BUSY'}
    assert path.read_bytes() == before


def test_service_heartbeat_continues_during_long_tick_and_records_error(tmp_path):
    import threading
    import time
    ticks_entered = threading.Event()
    release = threading.Event()
    errors = []
    def tick(control):
        ticks_entered.set()
        assert release.wait(2)
        raise ValueError('finite scientific proof mismatch')
    control = service_fixture_control(tmp_path, tick)
    def runner():
        try:
            owner.serve(control, interval_seconds=.01, heartbeat_seconds=.01)
        except ValueError as exc:
            errors.append(str(exc))
    thread = threading.Thread(target=runner)
    thread.start()
    try:
        assert ticks_entered.wait(2)
        path = control.root / 'runtime/persistent-supervisor.json'
        first = owner.read(path)['heartbeat_unix']
        deadline = time.monotonic() + 2
        while owner.read(path)['heartbeat_unix'] <= first and time.monotonic() < deadline:
            time.sleep(.01)
        assert owner.read(path)['heartbeat_unix'] > first
    finally:
        release.set()
        thread.join(3)
    assert errors == ['finite scientific proof mismatch']
    assert owner.read(path)['status'] == 'ERROR'
    assert owner.read(path)['error']['type'] == 'ValueError'
    with engine.try_lock(control.physical['supervisor']) as lock:
        assert lock is not None


@pytest.mark.parametrize('interval', [0, -1, float('nan'), float('inf'), 61])
def test_service_rejects_unbounded_or_nonpositive_interval(tmp_path, interval):
    control = service_fixture_control(tmp_path, lambda _: {})
    with pytest.raises(ValueError, match='finite'):
        owner.serve(control, interval_seconds=interval)


def test_service_status_cli_never_builds_dispatch_owner(tmp_path, monkeypatch, capsys):
    import time
    path = write(tmp_path / 'status.json', {'heartbeat_unix': time.time(), 'status': 'IDLE'})
    monkeypatch.setattr(owner, 'build_control', lambda **_: pytest.fail('telemetry must be read only'))
    assert owner.main(['service-status', '--status-file', str(path)]) == 0
    assert '"telemetry_only": true' in capsys.readouterr().out


def test_service_advances_two_real_cpu_workers_with_cached_proof(scientific_cache_fixture, tmp_path):
    import os
    import shutil
    import threading
    import time
    f = scientific_cache_fixture
    snapshot = tmp_path / 'native-worker'
    package = snapshot / 'src/mediflow_medvlm'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text('')
    shutil.copy(engine.__file__, package / 'campaign_supervisor.py')
    fixture_script = snapshot / 'cpu.py'
    fixture_script.write_text('''import hashlib,json,os,pathlib,sys
root=pathlib.Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=True)
artifact=root/'actual.txt';artifact.write_text('actual CPU output')
receipt={'status':'SUCCEEDED','job_id':os.environ['MEDVLM_SUPERVISOR_JOB_ID'],'run_id':os.environ['MEDVLM_SUPERVISOR_RUN_ID'],
'artifacts':[{'path':str(artifact),'sha256':hashlib.sha256(artifact.read_bytes()).hexdigest()}]}
for key in ('source_hash','config_hash','data_hash','model_hash'):receipt[key]=os.environ['MEDVLM_SUPERVISOR_'+key.upper()]
(root/'success.json').write_text(json.dumps(receipt))
''')
    sources = {str(path.relative_to(snapshot)): engine.file_hash(path) for path in snapshot.rglob('*.py')}
    jobs = []
    for index in range(2):
        output = tmp_path / ('new-output-' + str(index))
        jobs.append({'job_id': 'v6-cpu-' + str(index), 'run_id': 'cpu-' + str(index),
                     'phase': 'CPU_FIXTURE', 'cell_id': 'T1', 'uses_gpu': False,
                     'argv': [sys.executable, str(fixture_script), str(output)], 'cwd': str(snapshot),
                     'source_files': sources, 'source_hash': engine.canonical_hash(sources),
                     'config_hash': 'fixed', 'data_hash': 'fixed', 'model_hash': 'fixed',
                     'success_receipt': str(output / 'success.json'),
                     'dependencies': [] if index == 0 else ['v6-cpu-0']})
    supervisor = engine.Supervisor(tmp_path / 'native-private')
    manifest = {'campaign_id': 'cpu-fixture', 'jobs': [
        {'job_id': 'prior', 'run_id': 'prior', 'phase': 'CPU', 'cell_id': 'ALL',
         'implementation_pending': True, 'dependencies': []}, *jobs]}
    supervisor.initialize(manifest)
    state = engine.read_json(supervisor.state_path)
    state['jobs']['prior']['status'] = 'SUCCEEDED'
    write(supervisor.state_path, state)
    f.control.supervisor = supervisor
    import json
    for _, ref in f.jobs:
        config = owner.read(ref['path'])
        config['state'] = str(supervisor.state_path)
        write(Path(ref['path']), config)
        ref['sha256'] = owner.file_hash(ref['path'])
    native_lock = engine.try_lock
    physical = supervisor.root / 'supervisor.lock'
    stop = threading.Event()
    deadline = time.monotonic() + 8
    results = []
    class Control:
        def _physical_lock(self, path):
            return native_lock(path)
        def call(self, command):
            assert command == 'tick'
            owner.CompletionMixin._verify_worker_jobs(f.control, f.jobs)
            with native_lock(physical) as competitor:
                assert competitor is None
            @contextmanager
            def route(path):
                if Path(path) == physical:
                    with self._physical_lock(path) as lease:
                        yield lease
                else:
                    with native_lock(path) as lease:
                        yield lease
            engine.try_lock = route
            try:
                result = supervisor.tick()
            finally:
                engine.try_lock = native_lock
            results.append(result)
            if result['all_terminal']:
                stop.set()
            assert time.monotonic() < deadline, 'finite CPU service integration timed out'
            return result
    control = Control()
    control.root, control.data_root, control.engine = supervisor.root, supervisor.data_root, engine
    control.physical = {'supervisor': physical}
    result = owner.serve(control, interval_seconds=.02, heartbeat_seconds=.01, stop_event=stop)
    assert result['status'] == 'STOPPED' and result['ticks_completed'] >= 3
    assert all(results[-1]['jobs'][job['job_id']]['status'] == 'SUCCEEDED' for job in jobs)
    assert all(results[-1]['jobs'][job['job_id']]['attempts'] == 1 for job in jobs)
    assert all(Path(job['success_receipt']).is_file() for job in jobs)
    assert len(f.control._scientific_proofs) == 1
    f.science.write_text('tampered terminal artifact')
    with pytest.raises(ValueError, match='scientific evidence changed'):
        owner.CompletionMixin._verify_worker_jobs(f.control, f.jobs)


def test_service_stop_cancels_and_reaps_only_cpu_verifier(scientific_cache_fixture):
    import threading
    import time
    f = scientific_cache_fixture
    f.module.write_text('''import pathlib,time
def job_from_config(path, *, repo, python):
    time.sleep(30)
    return {}
''')
    stop = threading.Event()
    f.control._service_stop = stop
    timer = threading.Timer(.1, stop.set)
    timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(owner.ServiceStopped):
            owner.CompletionMixin._verify_worker_job(f.control, *f.jobs[0])
    finally:
        timer.cancel()
        del f.control._service_stop
    assert time.monotonic() - started < 3
    assert not getattr(f.control, '_scientific_proofs', {})


def test_service_stop_prevents_new_launcher_after_tick_stop_request(tmp_path):
    import threading
    stop = threading.Event()
    def tick(control):
        stop.set()
        control.engine.subprocess.Popen([sys.executable, '-c', 'raise RuntimeError("must never execute")'])
        pytest.fail('stop must prevent launcher')
    control = service_fixture_control(tmp_path, tick)
    result = owner.serve(control, interval_seconds=.01, heartbeat_seconds=.01, stop_event=stop)
    assert result['status'] == 'STOPPED' and result['ticks_completed'] == 0


def test_heartbeat_failure_is_error_even_during_cooperative_stop(tmp_path, monkeypatch):
    import threading
    import time
    def tick(control):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            owner.check_service_stop(control)
            time.sleep(.005)
        pytest.fail('heartbeat failure must stop active owner')
    control = service_fixture_control(tmp_path, tick)
    native_atomic = engine.atomic_json
    def failing_heartbeat(path, value):
        if threading.current_thread().name == 'supervisor-heartbeat':
            raise OSError('finite telemetry write failure')
        return native_atomic(path, value)
    monkeypatch.setattr(engine, 'atomic_json', failing_heartbeat)
    with pytest.raises(RuntimeError, match='service heartbeat failed'):
        owner.serve(control, interval_seconds=.01, heartbeat_seconds=.01)
    status = owner.read(control.root / 'runtime/persistent-supervisor.json')
    assert status['status'] == 'ERROR' and status['error']['type'] == 'RuntimeError'
    with engine.try_lock(control.physical['supervisor']) as lease:
        assert lease is not None


@pytest.fixture
def hf_snapshot_fixture(scientific_cache_fixture, tmp_path):
    f = scientific_cache_fixture
    cache = tmp_path / 'models--fixture'
    blobs = cache / 'blobs'
    snapshot = cache / 'snapshots' / ('a' * 40)
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    blob = blobs / 'content-addressed-blob'
    blob.write_text('science')
    alias = snapshot / 'model-config.json'
    alias.symlink_to('../../blobs/' + blob.name)
    for _, ref in f.jobs:
        config = owner.read(ref['path'])
        config['science'] = str(alias)
        write(Path(ref['path']), config)
        ref['sha256'] = owner.file_hash(ref['path'])
    return SimpleNamespace(**vars(f), alias=alias, blob=blob, snapshot=snapshot, cache=cache)


def test_hf_snapshot_symlink_is_verified_then_cached(hf_snapshot_fixture, monkeypatch):
    f = hf_snapshot_fixture
    owner.CompletionMixin._verify_worker_jobs(f.control, f.jobs)
    proof = next(iter(f.control._scientific_proofs.values()))
    actual = proof['files'][str(f.alias)]
    assert actual['sha256'] == owner.file_hash(f.blob)
    assert actual['path_identity']['resolved_path'] == str(f.blob)
    assert actual['path_identity']['links'][0]['path'] == str(f.alias)
    assert actual['path_identity']['links'][0]['target'] == '../../blobs/' + f.blob.name
    monkeypatch.setattr(owner.subprocess, 'run', lambda *a, **k: pytest.fail('valid HF proof must stay cached'))
    owner.CompletionMixin._verify_worker_jobs(f.control, f.jobs)


@pytest.mark.parametrize('change', ['retarget', 'blob', 'parent-alias'])
def test_hf_snapshot_alias_or_blob_change_fails_closed(hf_snapshot_fixture, change):
    import shutil
    f = hf_snapshot_fixture
    owner.CompletionMixin._verify_worker_jobs(f.control, f.jobs)
    if change == 'blob':
        f.blob.write_text('changed model contents')
    elif change == 'retarget':
        other = f.blob.with_name('other-blob')
        other.write_text('science')  # Same bytes do not authorize a new alias.
        f.alias.unlink()
        f.alias.symlink_to('../../blobs/' + other.name)
    else:
        copied = f.snapshot.with_name('replacement-snapshot')
        shutil.copytree(f.snapshot, copied, symlinks=True)
        shutil.rmtree(f.snapshot)
        f.snapshot.symlink_to(copied, target_is_directory=True)
    with pytest.raises((ValueError, OSError), match='scientific'):
        owner.CompletionMixin._verify_worker_jobs(f.control, f.jobs)


def test_initial_model_parent_directory_alias_is_pinned(hf_snapshot_fixture):
    f = hf_snapshot_fixture
    directory_alias = f.cache / 'active-snapshot'
    directory_alias.symlink_to(f.snapshot, target_is_directory=True)
    for _, ref in f.jobs:
        config = owner.read(ref['path'])
        config['science'] = str(directory_alias / f.alias.name)
        write(Path(ref['path']), config)
        ref['sha256'] = owner.file_hash(ref['path'])
    owner.CompletionMixin._verify_worker_jobs(f.control, f.jobs)
    proof = next(iter(f.control._scientific_proofs.values()))
    links = proof['files'][str(directory_alias / f.alias.name)]['path_identity']['links']
    assert [item['path'] for item in links] == [str(directory_alias), str(f.alias)]
    owner.CompletionMixin._verify_worker_jobs(f.control, f.jobs)
    directory_alias.unlink()
    directory_alias.symlink_to(f.snapshot.parent, target_is_directory=True)
    with pytest.raises((ValueError, OSError)):
        owner.CompletionMixin._verify_worker_jobs(f.control, f.jobs)


def test_stat_only_model_shard_size_is_pinned_without_byte_hash(hf_snapshot_fixture):
    f = hf_snapshot_fixture
    # Native model preflight intentionally uses type + declared size for weights.
    # This checks that such unopened shards still belong to the cached proof.
    f.module.write_text(f.module.read_text().replace(
        "state['cache']['shared']=pathlib.Path(config['science']).read_text()",
        "assert pathlib.Path(config['science']).is_file()\n        assert pathlib.Path(config['science']).stat().st_size==7\n        state['cache']['shared']='science'"))
    owner.CompletionMixin._verify_worker_jobs(f.control, f.jobs)
    proof = next(iter(f.control._scientific_proofs.values()))
    assert str(f.alias) not in proof['files']
    assert proof['metadata'][str(f.alias)]['resolved_path'] == str(f.blob)
    f.blob.write_text('resized unopened native weight shard')
    with pytest.raises(ValueError, match='scientific file metadata changed'):
        owner.CompletionMixin._verify_worker_jobs(f.control, f.jobs)


def test_model_alias_support_keeps_config_and_source_symlink_rule(hf_snapshot_fixture):
    with pytest.raises(ValueError, match='nonsymlink'):
        owner.absolute(hf_snapshot_fixture.alias)


@pytest.fixture
def epoch_fixture(fixture, scientific_cache_fixture):
    from types import MethodType
    f, science = fixture, scientific_cache_fixture
    f.control._completion_chain()
    owner.CompletionMixin._verify_worker_jobs(science.control, science.jobs)
    f.control._scientific_proofs = science.control._scientific_proofs
    base_spec = importlib.util.spec_from_file_location('epoch_canonical_owner', ROOT / 'scripts/campaign_control.py')
    base = importlib.util.module_from_spec(base_spec)
    base_spec.loader.exec_module(base)
    native_lock = engine.try_lock
    f.control.physical = {name: f.control.root / ('physical-' + name + '.lock')
                          for name in ('supervisor', 'worker', 'gpu')}
    for path in f.control.physical.values():
        with native_lock(path):
            pass
    class Guard:
        def _verify(self, *, prepared=True):
            self.guard_entries += 1
            for _ in range(6):
                self._stage()
                self._completion_chain()
            return {'prepared': prepared}
        def _stage(self):
            self.binding_reads += 1
            owner.pin_bytes(self.binding_test_ref)
            return owner.read(self.binding_test_ref['path'])
        def _bootstrap(self):
            self.bootstrap_reads += 1
            return {'route': str(getattr(self, 'choice_likelihood_path', 'original'))}
    original = type(f.control)
    class Control(original, Guard):
        def _registered(self):
            self.registration_reads += 1
            for _ in range(6):
                self._stage()
                self._completion_chain()
    f.control.__class__ = Control
    f.control.guard_entries = f.control.registration_reads = f.control.binding_reads = f.control.bootstrap_reads = 0
    f.control.binding_test_ref = owner.reference(f.proof)
    f.control.base = base
    f.control.before_locked = None
    f.control.dispatches = 0
    @contextmanager
    def physical(path):
        with native_lock(path) as lease:
            if path == f.control.physical['supervisor'] and f.control.before_locked is not None:
                f.control.before_locked()
            yield lease
    f.control._physical_lock = physical
    f.control.routed_locks = MethodType(base.CampaignControl.routed_locks, f.control)
    state_path = f.control.supervisor.state_path
    def native_status():
        for _ in range(12):
            f.control._stage()
            f.control._completion_chain()
        return {'jobs': {}}
    def native_tick():
        with engine.try_lock(f.control.root / 'supervisor.lock') as lease:
            assert lease is not None
            native_status()
            f.control.dispatches += 1
            return {'jobs': {}}
    f.control.supervisor = SimpleNamespace(state_path=state_path, status=native_status, tick=native_tick)
    return SimpleNamespace(control=f.control, source=f, science=science, native_lock=native_lock)


def test_recursive_loads_share_epoch_but_native_lock_and_end_revalidate(epoch_fixture, monkeypatch):
    f = epoch_fixture
    checks = []
    actual = owner.verify_read_proof
    def counted(proof):
        checks.append(1)
        return actual(proof)
    monkeypatch.setattr(owner, 'verify_read_proof', counted)
    f.control.call('tick')
    # Initial facade guard, canonical base guard, physical-lock guard, end check.
    assert len(checks) == 4
    assert f.control.guard_entries == 3 and f.control.binding_reads == 3
    assert f.control.registration_reads == 2 and f.control.dispatches == 1
    assert f.control._completion_validation_epoch is None
    assert not f.control._completion_call_active
    f.control.call('tick')
    assert len(checks) == 8 and f.control.binding_reads == 6
    assert f.control.registration_reads == 4 and f.control.dispatches == 2


def test_mutation_between_preflight_and_canonical_lock_fails_before_dispatch(epoch_fixture):
    f = epoch_fixture
    before = f.control.supervisor.state_path.read_bytes()
    f.control.before_locked = lambda: f.science.science.write_text('changed before native lock guard')
    with pytest.raises(ValueError, match='scientific evidence changed'):
        f.control.call('tick')
    assert f.control.dispatches == 0
    assert f.control.supervisor.state_path.read_bytes() == before
    assert f.control._completion_validation_epoch is None
    assert not f.control._completion_call_active


def test_direct_chain_after_call_has_no_epoch_exemption(epoch_fixture):
    f = epoch_fixture
    f.control.call('status')
    f.science.science.write_text('changed after completed owner call')
    with pytest.raises(ValueError, match='scientific evidence changed'):
        f.control._completion_chain()


def test_failed_scientific_check_never_marks_epoch_verified(epoch_fixture, monkeypatch):
    f = epoch_fixture
    attempts = []
    def reject(proof):
        attempts.append(1)
        raise ValueError('finite rejected scientific proof')
    monkeypatch.setattr(owner, 'verify_read_proof', reject)
    f.control._completion_validation_epoch = {}
    try:
        for _ in range(2):
            with pytest.raises(ValueError, match='rejected scientific proof'):
                f.control._completion_chain()
        assert len(attempts) == 2
        assert 'checked' not in f.control._completion_validation_epoch
    finally:
        f.control._completion_validation_epoch = None


def test_legacy_static_binding_cache_resets_before_canonical_lock(epoch_fixture):
    f = epoch_fixture
    f.control.before_locked = lambda: f.source.proof.write_text('{"changed": true}')
    with pytest.raises(ValueError, match='pinned completion evidence changed'):
        f.control.call('tick')
    assert f.control.binding_reads == 3
    assert f.control.dispatches == 0


def test_legacy_static_binding_proxy_routes_and_return_values_are_independent(epoch_fixture):
    f = epoch_fixture
    f.control._completion_validation_epoch = {}
    try:
        first = f.control._bootstrap()
        first['route'] = 'caller-local mutation'
        assert f.control._bootstrap() == {'route': 'original'}
        proxy = copy.copy(f.control)
        proxy.choice_likelihood_path = Path('/different/pinned/source-repair-binding.json')
        assert proxy._bootstrap() == {'route': str(proxy.choice_likelihood_path)}
        assert f.control._bootstrap() == {'route': 'original'}
        assert f.control.bootstrap_reads == 1
    finally:
        f.control._completion_validation_epoch = None
    # Direct helper calls never inherit the previous transaction's exemptions.
    f.control._bootstrap()
    f.control._bootstrap()
    assert f.control.bootstrap_reads == 3


def test_direct_verify_does_not_leave_an_epoch_for_later_calls(epoch_fixture):
    f = epoch_fixture
    f.control._verify()
    assert f.control._completion_validation_epoch is None
    f.science.science.write_text('changed after direct verification')
    with pytest.raises(ValueError, match='scientific evidence changed'):
        f.control._completion_chain()
