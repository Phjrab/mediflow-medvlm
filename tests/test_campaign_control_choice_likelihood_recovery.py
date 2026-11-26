"""CPU-only source-repair facade tests using temporary native-owner fixtures."""
import copy
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

import test_campaign_control_choice_likelihood as prior

fd=prior.fd
ROOT=Path(__file__).resolve().parents[1]
operator=fd.load('choice_likelihood_recovery_owner_under_test',ROOT/'scripts/campaign_control_choice_likelihood_recovery.py')
engine,base,write,ref=prior.engine,prior.base,prior.write,prior.ref


class RecoveryFixtureControl(operator.ChoiceLikelihoodRecoveryMixin,prior.FixtureControl):
    base=base


class RecoveryOperatorTests(unittest.TestCase):
    executable=prior.ChoiceLikelihoodOperatorTests.executable
    success=prior.ChoiceLikelihoodOperatorTests.success

    def setUp(self):
        prior.ChoiceLikelihoodOperatorTests.setUp(self)
        self.control.call('refine-choice-likelihood',manifest=self.replacement)
        self.policy=write(self.f.data/'policy.json',{
            'schema_version':operator.POLICY_SCHEMA,'campaign_id':self.f.root.name,'campaign_root':str(self.f.root),
            'source_thread_id':operator.POLICY_USER_THREAD,'source_user_message_id':operator.POLICY_USER_MESSAGE,
            'exact_user_text':'재시도 제한 해제','retry_limit':None,'mandatory_astra_before_retry':False,
            'preserve_attempts_failures_history_gpu_ledger':True,'require_source_fix_and_cpu_tests_before_same_cause_retry':True,
            'sole_canonical_supervisor_owner':True,'single_hourly_automation_preserved':True})
        kwargs=dict(choice_likelihood_binding=self.binding_path,choice_likelihood_binding_sha256=engine.file_hash(self.binding_path),
            bootstrap_binding=self.prior_binding_path,bootstrap_binding_sha256=engine.file_hash(self.prior_binding_path),prior_manifest=self.parent,
            inherited=self.inherited,roots=self.f.roots,data_root=self.f.data,campaign_root=self.f.root,engine_root=self.f.base_engine,hold_receipt=self.f.hold)
        self.control=RecoveryFixtureControl(retry_policy_receipt=self.policy,retry_policy_sha256=engine.file_hash(self.policy),**kwargs)
        self.before=base.read_json(self.f.state_path)

    def running(self,key,*,code=1,live=False,token='token'):
        state=base.read_json(self.f.state_path); record=state['jobs'][key]
        attempt=self.f.root/'attempts'/key/'0001'
        record.update(status='RUNNING',attempts=1,failure_count=0,recovery_attempts=0,
                      launch_token=token,attempt_dir=str(attempt),history=[])
        write(attempt/'heartbeat.json',{'worker_identity':engine.process_identity() if live else None})
        write(attempt/'exit.json',{'launch_token':token,'exit_code':code,'error':'recorded worker failure'})
        write(self.f.state_path,state)
        return attempt

    def repair(self,*,reconcile=True):
        if reconcile: self.control.call('reconcile-only')
        source=self.f.root/'snapshots'/('9'*40); shutil.copytree(self.likelihood_source,source)
        (source/operator.WORKER_SOURCE).write_text('# CPU-validated likelihood evidence guard repair\n')
        fixed=copy.deepcopy(self.replacement); newjobs={job['job_id']:job for job in fixed['jobs']}; paths=[]
        sources={name:engine.file_hash(source/name) for name in self.jobs[0]['source_files']}
        for key in operator.EXECUTABLE_IDS:
            config=copy.deepcopy(self.configs[key]); path=self.f.root/'configs/choice_likelihood_recovery'/(key+'.json')
            config.update(source_files=sources,source_hash=engine.canonical_hash(sources))
            if key==operator.EXECUTABLE_IDS[-1]: config['data']['roles']=[ref(path) for path in paths]
            config['data_hash']=engine.canonical_hash(config['data'])
            config['config_hash']=engine.canonical_hash({k:v for k,v in config.items() if k!='config_hash'})
            write(path,config); paths.append(path)
            newjobs[key].update(cwd=str(source),source_files=sources,source_hash=config['source_hash'],
                                config_hash=config['config_hash'],data_hash=config['data_hash'])
            newjobs[key]['argv'][4]=str(path)
        manifest=write(self.f.root/'manifests/repair-proposal.json',fixed)
        choice=operator.prepare_choice_binding(self.binding_path,manifest,source,paths)
        choice_path=write(self.f.root/'operator/repaired-choice-binding.json',choice)
        tests=write(self.f.root/'operator/cpu-test-evidence.json',{'status':'PASS','cpu_only':True})
        repair=write(self.f.root/'operator/repair-evidence.json',{'schema_version':operator.REPAIR_SCHEMA,
            'campaign_root':str(self.f.root),'status':'CPU_VALIDATED','cpu_only':True,'cpu_tests_passed':True,
            'job_ids':operator.EXECUTABLE_IDS,'parent_source_hash':self.jobs[0]['source_hash'],
            'replacement_source_hash':newjobs[operator.ROLE_IDS[0]]['source_hash'],
            'reason':'Correct native input evidence comparison without changing scores or protocol','evidence':[ref(tests)]})
        binding=operator.prepare_binding(self.binding_path,choice_path,self.policy,repair,self.f.state_path)
        path=write(self.f.root/'operator/source-recovery-binding.json',binding)
        self.control.likelihood_recovery_path=path; self.control.likelihood_recovery_sha256=engine.file_hash(path)
        self.fixed,self.repair_binding,self.repair_binding_path,self.repaired_choice=fixed,binding,path,choice_path
        return fixed

    def reject(self,operation,match='.*'):
        before=self.f.state_path.read_bytes()
        with patch.object(engine,'atomic_json',side_effect=AssertionError('rejected operation wrote state')):
            with self.assertRaisesRegex((ValueError,OSError),match): operation()
        self.assertEqual(self.f.state_path.read_bytes(),before)

    def test_reconcile_only_dead_exit_preserves_counters_old_records_and_ledger(self):
        key=operator.ROLE_IDS[1]; self.running(key)
        state=base.read_json(self.f.state_path); first=state['jobs'][operator.ROLE_IDS[0]]
        first.update(status='ANALYSIS_REQUIRED',attempts=5,failure_count=5,recovery_attempts=4,
                     history=[{'status':'FAILED','reason':'preserved failure'}]); write(self.f.state_path,state)
        ledger=write(self.f.data/'runtime/gpu_budget.json',{'usage':123.4}); raw=ledger.read_bytes()
        with patch.object(engine.subprocess,'Popen',side_effect=AssertionError('CPU reconcile dispatched')):
            result=self.control.call('reconcile-only')
        self.assertEqual(result['jobs'][operator.ROLE_IDS[0]],first)
        self.assertEqual(result['jobs'][key]['failure_count'],1)
        self.assertEqual(result['jobs'][key]['attempts'],1)
        self.assertEqual(result['jobs'][key]['status'],'ANALYSIS_REQUIRED')
        self.assertTrue(result['jobs'][key]['source_repair_required'])
        for old in self.binding['parent_record_hashes']: self.assertEqual(result['jobs'][old],self.before['jobs'][old])
        self.assertEqual(ledger.read_bytes(),raw)
        after=self.f.state_path.read_bytes(); self.control.call('reconcile-only'); self.assertEqual(self.f.state_path.read_bytes(),after)

    def test_reconcile_refuses_live_child_and_every_canonical_busy_lease(self):
        key=operator.ROLE_IDS[0]; attempt=self.running(key,live=True)
        self.reject(lambda:self.control.call('reconcile-only'),'live worker')
        write(attempt/'heartbeat.json',{'child_identity':engine.process_identity()})
        self.reject(lambda:self.control.call('reconcile-only'),'live worker')
        write(attempt/'heartbeat.json',{})
        for lock in ('worker','gpu'):
            with self.control._physical_lock(self.control.physical[lock]):
                self.reject(lambda:self.control.call('reconcile-only'),'lease busy')
        with self.control._physical_lock(self.control.physical['supervisor']):
            self.assertEqual(self.control.call('reconcile-only'),{'status':'SUPERVISOR_BUSY'})
        self.assertFalse((self.f.root/'worker.lock').exists())

    def test_reconcile_zero_exit_uses_native_success_verification(self):
        key=operator.ROLE_IDS[0]; self.running(key,code=0)
        with patch.object(engine,'verify_success',return_value={'receipt':'synthetic','artifact_count':1}) as verified:
            result=self.control.call('reconcile-only')
        self.assertEqual(result['jobs'][key]['status'],'SUCCEEDED'); verified.assert_called_once()
        self.assertEqual(result['jobs'][key]['failure_count'],0)

    def test_reconcile_exit_token_mismatch_is_native_failure_not_success(self):
        key=operator.ROLE_IDS[0]; attempt=self.running(key,code=0)
        write(attempt/'exit.json',{'launch_token':'wrong-token','exit_code':0})
        with patch.object(engine,'verify_success',side_effect=AssertionError('mismatched exit verified')):
            result=self.control.call('reconcile-only')
        self.assertEqual(result['jobs'][key]['failure_count'],1)
        self.assertEqual(result['jobs'][key]['status'],'ANALYSIS_REQUIRED')
        self.assertEqual(result['jobs'][key]['error'],'exit receipt launch identity mismatch')

    def test_source_repair_requeues_thirteen_without_reset_and_is_idempotent(self):
        self.running(operator.ROLE_IDS[1]); state=base.read_json(self.f.state_path)
        first=state['jobs'][operator.ROLE_IDS[0]]; first.update(status='ANALYSIS_REQUIRED',attempts=7,failure_count=7,
            recovery_attempts=6,history=[{'status':'FAILED','detail':'unchanged'}]); write(self.f.state_path,state)
        fixed=self.repair(); before=base.read_json(self.f.state_path)
        ledger=write(self.f.data/'runtime/gpu_budget.json',{'usage':456.}); raw=ledger.read_bytes()
        with patch.object(engine.subprocess,'Popen',side_effect=AssertionError('registration dispatched')):
            result=self.control.call('repair-choice-likelihood',manifest=fixed)
        for key in operator.EXECUTABLE_IDS:
            self.assertEqual(result['jobs'][key]['status'],'PENDING')
            for counter in ('attempts','failure_count','recovery_attempts'):
                self.assertEqual(result['jobs'][key][counter],before['jobs'][key][counter])
            self.assertEqual(result['jobs'][key]['history'][:-1],before['jobs'][key]['history'])
        for old in self.binding['parent_record_hashes']: self.assertEqual(result['jobs'][old],before['jobs'][old])
        self.assertEqual(ledger.read_bytes(),raw)
        after=self.f.state_path.read_bytes(); self.control.call('repair-choice-likelihood',manifest=fixed)
        self.assertEqual(self.f.state_path.read_bytes(),after)
        self.assertEqual(len([e for e in result['manifest_revisions'] if e['kind']==operator.EVENT]),1)

    def test_source_repair_refuses_raw_dead_running_and_immutable_parent_drift(self):
        self.running(operator.ROLE_IDS[0]); fixed=self.repair(reconcile=False)
        self.reject(lambda:self.control.call('repair-choice-likelihood',manifest=fixed),'reconcile dead running')
        self.control.call('reconcile-only')
        self.reject(lambda:self.control.call('repair-choice-likelihood',manifest=fixed),'exact reconciled parent')

    def test_policy_pins_namespace_and_removed_astra_cap_are_explicit(self):
        original=base.read_json(self.policy)
        for key,value in (('campaign_root','/another/campaign'),('retry_limit',3),('mandatory_astra_before_retry',True),
                          ('require_source_fix_and_cpu_tests_before_same_cause_retry',False),('source_user_message_id','relay')):
            changed=copy.deepcopy(original); changed[key]=value; write(self.policy,changed)
            self.control.retry_policy_sha256=engine.file_hash(self.policy)
            with self.subTest(key=key): self.reject(lambda:self.control.call('reconcile-only'),'exact direct-user')
        write(self.policy,original); self.control.retry_policy_sha256='0'*64
        self.reject(lambda:self.control.call('status'),'policy bytes changed')
        self.reject(lambda:self.control.call('authorize-recovery'),'only status')

    def test_exact_source_only_pair_rejects_science_old_jobs_argv_or_aggregate_changes(self):
        fixed=self.repair(); first=next(j for j in fixed['jobs'] if j['job_id']==operator.ROLE_IDS[0])
        for kind in ('old','seed','model','output','argv','deps','source','aggregate','namespace'):
            changed=copy.deepcopy(fixed); jobs={j['job_id']:j for j in changed['jobs']}; job=jobs[operator.ROLE_IDS[0]]
            if kind=='old': changed['jobs'][0]['run_id']='changed'
            elif kind=='seed': job['seed']=2
            elif kind=='model': job['model_hash']='0'*64
            elif kind=='output': job['success_receipt']='/tmp/unrelated.json'
            elif kind=='argv': job['argv'][0]='/different/python'
            elif kind=='deps': job['dependencies']=[]
            elif kind=='source': job['source_files']['src/mediflow_medvlm/campaign_supervisor.py']='0'*64
            elif kind=='aggregate': jobs[operator.EXECUTABLE_IDS[-1]]['data_hash']='0'*64
            else: changed['campaign_root']='/other'
            with self.subTest(kind=kind),self.assertRaises(ValueError):operator.validate_pair(engine,self.replacement,changed)
        raw=Path(first['argv'][4]).read_bytes(); config=json.loads(raw); config['eval_context']=2048
        config['config_hash']=engine.canonical_hash({k:v for k,v in config.items() if k!='config_hash'}); write(first['argv'][4],config)
        self.reject(lambda:self.control.call('repair-choice-likelihood',manifest=fixed))

    def test_external_binding_evidence_archive_and_original_records_are_pinned(self):
        fixed=self.repair()
        paths=[self.repair_binding_path,self.repaired_choice,Path(self.repair_binding['repair_receipt']['path']),
               self.f.root/'snapshots'/('9'*40)/'README.md']
        for path in paths:
            raw=path.read_bytes(); path.write_bytes(raw+b' ')
            with self.subTest(path=path): self.reject(lambda:self.control.call('repair-choice-likelihood',manifest=fixed))
            path.write_bytes(raw)
        state=base.read_json(self.f.state_path); state['jobs'][next(iter(self.binding['parent_record_hashes']))]['history'].append({'forged':True}); write(self.f.state_path,state)
        self.reject(lambda:self.control.call('repair-choice-likelihood',manifest=fixed),'original state records')

    def test_tick_requires_repair_and_inherits_native_dispatch_with_canonical_leases(self):
        self.reject(lambda:self.control.call('tick'),'explicit source-repair binding')
        fixed=self.repair(); self.control.call('repair-choice-likelihood',manifest=fixed)
        checked=[]
        def spawned(argv,**kwargs):
            checked.append((argv,kwargs)); self.assertEqual(len(kwargs['pass_fds']),2)
            for key in ('supervisor','worker','gpu'):
                with self.control._physical_lock(self.control.physical[key]) as lock: self.assertIsNone(lock)
            return type('P',(),{'pid':99999999})()
        with patch.object(engine.subprocess,'Popen',side_effect=spawned),patch.object(engine,'process_identity',return_value={'pid':99999999}):
            result=self.control.call('tick')
        self.assertEqual(len(checked),1)
        self.assertEqual(result['jobs'][operator.ROLE_IDS[0]]['attempts'],1)
        self.assertEqual(result['jobs'][operator.ROLE_IDS[0]]['status'],'RUNNING')
        self.assertEqual(checked[0][1]['cwd'],str(self.f.root/'snapshots'/('9'*40)))

    def test_failure_beyond_old_limit_holds_all_same_source_before_next_native_dispatch(self):
        fixed=self.repair(); self.control.call('repair-choice-likelihood',manifest=fixed)
        key=operator.ROLE_IDS[0]; attempt=self.running(key)
        state=base.read_json(self.f.state_path); state['jobs'][key].update(attempts=9,failure_count=8,recovery_attempts=8); write(self.f.state_path,state)
        with patch.object(engine.subprocess,'Popen',side_effect=AssertionError('known bad source redispatched')):
            result=self.control.call('tick')
        self.assertEqual(result['jobs'][key]['attempts'],9)
        self.assertEqual(result['jobs'][key]['failure_count'],9)
        self.assertEqual(result['jobs'][key]['status'],'ANALYSIS_REQUIRED')
        self.assertTrue(all(result['jobs'][j].get('source_repair_required') for j in operator.EXECUTABLE_IDS))
        raw=self.f.state_path.read_bytes(); self.control.call('tick'); self.assertEqual(self.f.state_path.read_bytes(),raw)

    def test_completed_likelihood_receipt_cannot_be_rebound(self):
        fixed=self.repair(); state=base.read_json(self.f.state_path); state['jobs'][operator.ROLE_IDS[0]]['status']='SUCCEEDED'
        write(self.f.state_path,state)
        self.repair_binding['parent_state_sha256']=engine.file_hash(self.f.state_path)
        self.repair_binding['parent_record_hashes']={k:engine.canonical_hash(v) for k,v in state['jobs'].items()}
        write(self.repair_binding_path,self.repair_binding); self.control.likelihood_recovery_sha256=engine.file_hash(self.repair_binding_path)
        self.reject(lambda:self.control.call('repair-choice-likelihood',manifest=fixed),'successful likelihood')

    def test_exact_prior_operator_loader_and_public_factory_alias(self):
        self.assertIs(operator.make_control,operator.build_control)
        loaded=operator.load_prior(ROOT/'scripts/campaign_control_choice_likelihood.py')
        self.assertEqual(loaded.SCHEMA,prior.operator.SCHEMA)
        bad=self.f.root/'operator/modified.py'; bad.write_text('# changed\n')
        with self.assertRaisesRegex(ValueError,'operator changed'): operator.load_prior(bad)
