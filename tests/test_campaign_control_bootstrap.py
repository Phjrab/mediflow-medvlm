"""CPU-only temporary-fixture tests for one immutable bootstrap registration."""
import copy
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

import test_campaign_control_full_dev as fd

ROOT=Path(__file__).resolve().parents[1]
operator=fd.load('bootstrap_owner_under_test',ROOT/'scripts/campaign_control_bootstrap.py')
engine,base,write,ref=fd.engine,fd.base,fd.write,fd.ref


class FixtureControl(operator.BootstrapMixin,fd.FixtureControl):
    pass


class BootstrapOperatorTests(unittest.TestCase):
    executable=fd.FullDevOperatorTests.executable
    success=fd.FullDevOperatorTests.success

    def setUp(self):
        fd.FullDevOperatorTests.setUp(self)
        f=self.f
        # Give the preserved future declaration its real stage ID before either registration.
        for value in (self.before,self.after):
            next(j for j in value['jobs'] if j['job_id']=='fixture-future-1')['job_id']=operator.FAMILY_ID
            next(j for j in value['jobs'] if j['job_id']==operator.FAMILY_ID)['run_id']=operator.FAMILY_ID
        state=base.read_json(f.state_path);state['jobs'][operator.FAMILY_ID]=state['jobs'].pop('fixture-future-1')
        self.states=state['jobs'];write(self.parent_path,self.before);write(self.new_path,self.after)
        self.inherited['replacement']['manifest_hash']=engine.canonical_hash(self.before)
        state['manifest_hash']=engine.canonical_hash(self.before)
        self.binding['parent'].update(manifest_sha256=engine.file_hash(self.parent_path),manifest_hash=engine.canonical_hash(self.before))
        self.binding['replacement'].update(manifest_sha256=engine.file_hash(self.new_path),manifest_hash=engine.canonical_hash(self.after))
        fd.FullDevOperatorTests.rebind(self,state)
        self.control.call('refine-full-dev',manifest=self.after)
        state=base.read_json(f.state_path);self.states=state['jobs']
        for job in self.after['jobs']:
            if job['job_id'] in fd.operator.EXECUTABLE_IDS:self.success(job)
        # Native generation audit succeeded, explicitly without completing the R2 auxiliaries.
        aggregate_job=next(j for j in self.after['jobs'] if j['job_id']==operator.PARENT_ID)
        receipt=base.read_json(aggregate_job['success_receipt'])
        receipt.update(scope='VALIDATION_ONLY_FULL_DEV_TERMINAL_AUDIT',generation_evaluation_terminal=True,full_dev_stage_terminal=True,
            all_roles_succeeded=True,test_files_read=False,test_unlocked=False,family_selection_completed=False,
            clinical_quality_certified=False,full_dev_completed=False,r2_full_evaluation_completed=False,campaign_terminal=False)
        write(aggregate_job['success_receipt'],receipt);state['jobs'][operator.PARENT_ID]['verification']=engine.verify_success(aggregate_job)
        write(f.state_path,state)
        self.full_dev_binding_path=self.binding_path;self.full_dev_binding_sha=engine.file_hash(self.binding_path)
        self.parent=copy.deepcopy(self.after)
        self.bootstrap_parent_path=Path(state['manifest_path'])
        self.bootstrap_engine=f.root/'snapshots'/('7'*40)
        shutil.copytree(self.new_engine,self.bootstrap_engine)
        (self.bootstrap_engine/'src/mediflow_medvlm/campaign_bootstrap.py').write_text('# immutable bootstrap CPU source fixture\n')
        sources={p.relative_to(self.bootstrap_engine).as_posix():engine.file_hash(p) for p in self.bootstrap_engine.glob('src/mediflow_medvlm/**/*.py')}
        aggregate=base.read_json(aggregate_job['argv'][4]);initial=base.read_json(aggregate['data']['initial_protocol']['path'])
        role_ids=[f'v6-full-dev-{cell}-{role}' for cell in operator.CELL_IDS for role in operator.ROLES]
        jobs={j['job_id']:j for j in self.parent['jobs']}
        data={'aggregate_config':ref(aggregate_job['argv'][4]),'aggregate_receipt':ref(aggregate_job['success_receipt']),
            'native_source_repo':aggregate_job['cwd'],'roles':[{'job_id':key,'config':reference,'receipt':ref(jobs[key]['success_receipt']),
                'artifacts':base.read_json(jobs[key]['success_receipt'])['artifacts']} for key,reference in zip(role_ids,aggregate['data']['roles'])],
            'validation':{axis:{k:initial['validation_references'][axis][k] for k in ('path','sha256')} for axis in ('text','vqa')},'grouping':{}}
        for axis,n in (('text',1516),('vqa',512)):
            data['grouping'][axis]={'axis':axis,'scheduled':n,'group_count':n,'stratum_counts':{'fixture':n},'group_size_counts':{'1':n},
                'unavailable_reason_counts':{},'singleton_fallback_count':n,'source_evidence_sha256':'e'*64,'components_sha256':'d'*64,
                'profiles':[{'sources':['fixture'],'strata':['fixture'],'group_count':n}],'patient_or_study_independence_certified':False}
        self.config={'schema_version':operator.CONFIG_SCHEMA,'scope':operator.SCOPE,'kind':'bootstrap','phase':'bootstrap',
            'job_id':operator.JOB_ID,'run_id':operator.JOB_ID,'cell_id':'ALL_V6','seed':20261023,'uses_gpu':False,'device':'cpu',
            'state_path':str(f.state_path),'output_dir':str(f.root/'auxiliary/bootstrap/first_seed'),'success_receipt':str(f.root/'receipts/v6-bootstrap.json'),
            'data':data,'procedure':copy.deepcopy(operator.PROCEDURE),'model_hash':aggregate['model_hash'],'resource_policy':aggregate['resource_policy'],
            'source_files':sources,'source_hash':engine.canonical_hash(sources),'data_hash':engine.canonical_hash(data)}
        self.config['config_hash']=engine.canonical_hash(self.config)
        self.config_path=write(f.root/'configs/bootstrap/first-seed.json',self.config)
        self.job={k:self.config[k] for k in ('job_id','run_id','phase','cell_id','kind','seed','uses_gpu','source_files','success_receipt',*engine.HASH_FIELDS)}
        self.job.update(requires_gpu=False,gpu_lock_mode='wrapper',dependencies=[operator.PARENT_ID],dependency_policy='success',implementation_pending=False,
            argv=[sys.executable,'-m',operator.WORKER_MODULE,'--config',str(self.config_path),'--receipt',self.config['success_receipt']],cwd=str(self.bootstrap_engine))
        self.replacement=copy.deepcopy(self.parent);position=next(i for i,j in enumerate(self.replacement['jobs']) if j['job_id']==operator.PARENT_ID)+1
        self.replacement['jobs'].insert(position,self.job)
        self.replacement_path=write(f.root/'manifests/bootstrap-proposal.json',self.replacement)
        archive={p.relative_to(self.bootstrap_engine).as_posix():engine.file_hash(p) for p in sorted(self.bootstrap_engine.rglob('*')) if p.is_file()}
        source_tree={k:v for k,v in archive.items() if k.startswith('src/') and k.endswith('.py')}
        self.binding={'schema_version':operator.SCHEMA,'campaign_root':str(f.root),'data_root':str(f.data),
            'full_dev_operator_sha256':operator.FULL_DEV_OPERATOR_SHA256,'full_dev_binding_sha256':self.full_dev_binding_sha,
            'parent':{'manifest_path':str(self.bootstrap_parent_path),'manifest_sha256':engine.file_hash(self.bootstrap_parent_path),'manifest_hash':engine.canonical_hash(self.parent)},
            'replacement':{'manifest_path':str(self.replacement_path),'manifest_sha256':engine.file_hash(self.replacement_path),'manifest_hash':engine.canonical_hash(self.replacement)},
            'parent_state_sha256':engine.file_hash(f.state_path),'parent_record_hashes':{key:engine.canonical_hash(record) for key,record in state['jobs'].items()},
            'engine':{'root':str(self.bootstrap_engine),'commit':self.bootstrap_engine.name,'archive_files':archive,'archive_tree_hash':base.digest(archive),
                'source_files':source_tree,'source_tree_hash':base.digest(source_tree)},'configs':{operator.JOB_ID:ref(self.config_path)},
            'prerequisites':{key:{**ref(jobs[key]['success_receipt']),'job_hash':engine.canonical_hash(jobs[key]),'state_hash':engine.canonical_hash(state['jobs'][key])} for key in operator.PREREQUISITES}}
        self.binding_path=write(f.root/'operator/bootstrap-binding.json',self.binding)
        self.control=FixtureControl(bootstrap_binding=self.binding_path,bootstrap_binding_sha256=engine.file_hash(self.binding_path),
            full_dev_binding=self.full_dev_binding_path,full_dev_binding_sha256=self.full_dev_binding_sha,
            inherited=self.inherited,roots=f.roots,data_root=f.data,campaign_root=f.root,engine_root=f.base_engine,hold_receipt=f.hold)

    def rebind(self,state=None):
        if state is not None:
            write(self.f.state_path,state);self.binding['parent_state_sha256']=engine.file_hash(self.f.state_path)
            self.binding['parent_record_hashes']={key:engine.canonical_hash(record) for key,record in state['jobs'].items()}
        write(self.binding_path,self.binding);self.control.bootstrap_sha256=engine.file_hash(self.binding_path)

    def reject(self,operation=None,match='.*'):
        before=self.f.state_path.read_bytes()
        with patch.object(engine,'atomic_json',side_effect=AssertionError('rejected registration wrote')):
            with self.assertRaisesRegex((ValueError,OSError),match):(operation or (lambda:self.control.call('refine-bootstrap',manifest=self.replacement)))()
        self.assertEqual(self.f.state_path.read_bytes(),before)

    def test_registration_one_cpu_job_preserves_all_eighty_three_records_and_ledger(self):
        before=base.read_json(self.f.state_path);self.assertEqual(len(before['jobs']),83)
        ledger=write(self.f.data/'runtime/gpu_budget.json',{'total_gpu_hours':None,'cumulative_seconds':321.5});ledger_before=ledger.read_bytes()
        writes=[];atomic=engine.atomic_json
        def observe(path,value):
            if Path(path)==self.f.state_path:
                self.assertEqual(base.read_json(value['manifest_path']),self.replacement);self.assertEqual(base.read_json(path),before)
            writes.append(Path(path));atomic(path,value)
        with patch.object(engine,'atomic_json',side_effect=observe),patch.object(engine.subprocess,'Popen',side_effect=AssertionError('spawned')):
            result=self.control.call('refine-bootstrap',manifest=self.replacement)
        self.assertEqual(len(result['jobs']),84)
        for key,record in before['jobs'].items():self.assertEqual(result['jobs'][key],record)
        self.assertEqual(result['jobs'][operator.JOB_ID],{'status':'PENDING','attempts':0,'failure_count':0,'recovery_attempts':0,'history':[]})
        self.assertEqual(writes,[Path(result['manifest_path']),self.f.state_path]);self.assertEqual(ledger.read_bytes(),ledger_before)
        self.assertEqual(result['manifest_revisions'][:-1],before['manifest_revisions'])
        self.assertEqual(result['manifest_revisions'][-1]['original_record_hashes'],self.binding['parent_record_hashes'])
        self.assertFalse((self.f.root/'supervisor.lock').exists());self.assertFalse((self.f.root/'worker.lock').exists())

    def test_exact_pair_preserves_original_family_deps_and_science(self):
        for kind in ('metadata','old','family','missing','extra','position','gpu','requires','phase','scopefield','seed','boolseed','deps','policy','argv'):
            value=copy.deepcopy(self.replacement);jobs={j['job_id']:j for j in value['jobs']};new=jobs[operator.JOB_ID]
            if kind=='metadata':value['declared_seed']=42
            elif kind=='old':jobs[operator.PREREQUISITES[0]]['config_hash']='f'*64
            elif kind=='family':jobs[operator.FAMILY_ID]['dependencies']=[operator.JOB_ID]
            elif kind=='missing':value['jobs'].remove(new)
            elif kind=='extra':value['jobs'].append({**new,'job_id':'hidden','run_id':'hidden'})
            elif kind=='position':value['jobs'].remove(new);value['jobs'].append(new)
            elif kind=='gpu':new['uses_gpu']=True
            elif kind=='requires':new['requires_gpu']=True
            elif kind=='phase':new['phase']='family_selection'
            elif kind=='scopefield':new['hidden_queue']=True
            elif kind=='seed':new['seed']=2
            elif kind=='boolseed':new['seed']=True
            elif kind=='deps':new['dependencies']=[operator.PREREQUISITES[0]]
            elif kind=='policy':new['dependency_policy']='terminal'
            else:new['argv'][2]='another.module'
            with self.subTest(kind=kind),self.assertRaises(ValueError):operator.validate_pair(engine,self.parent,value)

    def test_byte_pins_full_archive_config_parent_and_prerequisite(self):
        for path in (self.binding_path,self.replacement_path,self.bootstrap_parent_path,self.config_path,
                Path(self.binding['prerequisites'][operator.PARENT_ID]['path']),self.bootstrap_engine/'README.md'):
            raw=path.read_bytes();path.write_bytes(raw+b' ')
            with self.subTest(path=path):self.reject()
            path.write_bytes(raw)

    def test_parent_state_counter_failure_orphan_and_future_progress_refused(self):
        original=base.read_json(self.f.state_path)
        for kind in ('counter','failure','orphan','missing','future','boolfuture','owner'):
            state=copy.deepcopy(original)
            if kind=='counter':state['jobs'][operator.PREREQUISITES[0]]['attempts']+=1
            elif kind=='failure':state['jobs'][operator.PARENT_ID]['status']='FAILED'
            elif kind=='orphan':state['jobs']['hidden']={'status':'PENDING'}
            elif kind=='missing':del state['jobs'][operator.PREREQUISITES[0]]
            elif kind=='future':state['jobs'][operator.FAMILY_ID]['attempts']=1
            elif kind=='boolfuture':state['jobs'][operator.FAMILY_ID]['attempts']=False
            else:state['queue_owner']='other'
            self.rebind(state)
            with self.subTest(kind=kind):self.reject()
        self.rebind(original);state=copy.deepcopy(original);state['updated_at']='unpinned';write(self.f.state_path,state)
        self.reject(match='parent bytes')

    def test_active_orphan_process_and_each_canonical_lease_refused(self):
        original=base.read_json(self.f.state_path)
        for kind in ('running','worker','heartbeat'):
            state=copy.deepcopy(original);record=state['jobs'][operator.FAMILY_ID]
            if kind=='running':record['status']='RUNNING'
            elif kind=='worker':record['worker_identity']=engine.process_identity()
            else:
                record['attempt_dir']=str(self.f.root/'attempts/orphan');write(Path(record['attempt_dir'])/'heartbeat.json',{'child_identity':engine.process_identity()})
            self.rebind(state)
            with self.subTest(kind=kind):self.reject(match='worker active')
        self.rebind(original)
        for key in ('worker','gpu'):
            with self.control._physical_lock(self.control.physical[key]):self.reject(match='lease busy')
        before=self.f.state_path.read_bytes()
        with self.control._physical_lock(self.control.physical['supervisor']):
            self.assertEqual(self.control.call('refine-bootstrap',manifest=self.replacement),{'status':'SUPERVISOR_BUSY'})
        self.assertEqual(self.f.state_path.read_bytes(),before)

    def test_idempotence_requires_exact_audit_and_never_rewrites_progress(self):
        self.control.call('refine-bootstrap',manifest=self.replacement);state=base.read_json(self.f.state_path)
        state['jobs'][operator.JOB_ID].update(status='FAILED',attempts=2,failure_count=2,recovery_attempts=1)
        write(self.f.state_path,state);before=self.f.state_path.read_bytes()
        with patch.object(engine,'atomic_json',side_effect=AssertionError('rewrote')):self.control.call('refine-bootstrap',manifest=self.replacement)
        self.assertEqual(self.f.state_path.read_bytes(),before)
        state['manifest_revisions'][-1]['original_record_hashes']={};write(self.f.state_path,state);self.reject(match='immutable bootstrap refinement audit')

    def test_atomic_pointer_interruption_keeps_parent_retryable(self):
        before=self.f.state_path.read_bytes();atomic=engine.atomic_json
        def interrupt(path,value):
            if Path(path)==self.f.state_path:raise OSError('CPU pointer interruption')
            atomic(path,value)
        with patch.object(engine,'atomic_json',side_effect=interrupt),self.assertRaises(OSError):self.control.call('refine-bootstrap',manifest=self.replacement)
        self.assertEqual(self.f.state_path.read_bytes(),before);self.control.call('refine-bootstrap',manifest=self.replacement)

    def test_native_runtime_identity_cpu_launch_and_no_gpu_lease(self):
        cls=type(self.control.supervisor)
        self.assertEqual({k for k,v in cls.__dict__.items() if callable(v)},{'refine_bootstrap'})
        for name in ('tick','status','load','authorize_recovery','_failure'):self.assertIs(getattr(cls,name),getattr(engine.Supervisor,name))
        self.control.call('refine-bootstrap',manifest=self.replacement)
        with self.control._physical_lock(self.control.physical['gpu']),patch.object(engine.subprocess,'Popen') as spawn:
            spawn.return_value.pid=999999999;result=self.control.call('tick')
        self.assertEqual(spawn.call_count,1);self.assertEqual(result['jobs'][operator.JOB_ID]['attempts'],1)
        self.assertEqual(len(spawn.call_args.kwargs['pass_fds']),1)
        self.assertTrue(all(result['jobs'][key]['attempts']==0 for key in (operator.FAMILY_ID,'fixture-future-2')))

    def test_hold_namespace_alias_and_lock_inode_protection(self):
        hold=base.read_json(self.f.hold);hold['dispatch_control']['tick_prohibited']=False;write(self.f.hold,hold);self.reject(match='MIGRATION_HOLD')
        hold['dispatch_control']['tick_prohibited']=True;write(self.f.hold,hold)
        for name in ('worker.lock','supervisor.lock'):
            path=self.f.root/name;path.touch();self.reject(match='namespace lock files');path.unlink()
        path=self.control.physical['worker'];replacement=path.with_name('new.lock');replacement.touch();os.replace(replacement,path);self.reject(match='physical lock inode')

    def test_scientific_config_overrides_even_self_hashed_refused(self):
        originals={p:p.read_bytes() for p in (self.config_path,self.replacement_path,self.binding_path)}
        binding=copy.deepcopy(self.binding)
        for kind in ('unknown','scope','procedure','ratio','device','seed','state','resource','aggregate','native','roles','role-artifacts','validation','group-count','certified','protected-output','old-output','receipt','sources'):
            config=copy.deepcopy(self.config);replacement=copy.deepcopy(self.replacement);self.binding=copy.deepcopy(binding)
            if kind=='unknown':config['max_time']=60
            elif kind=='scope':config['scope']='TEST_EVALUATION'
            elif kind=='procedure':config['procedure']['resamples']=20
            elif kind=='ratio':config['procedure']['denominator']='drop unavailable rows'
            elif kind=='device':config['device']='cuda'
            elif kind=='seed':config['seed']=True
            elif kind=='state':config['state_path']=str(self.f.old/'state.json')
            elif kind=='resource':config['resource_policy']={'total_gpu_hours':1}
            elif kind=='aggregate':config['data']['aggregate_receipt']['sha256']='f'*64
            elif kind=='native':config['data']['native_source_repo']=str(self.bootstrap_engine)
            elif kind=='roles':config['data']['roles'].reverse()
            elif kind=='role-artifacts':config['data']['roles'][0]['artifacts']=[]
            elif kind=='validation':config['data']['validation']['text']['path']=str(self.f.root/'test.jsonl')
            elif kind=='group-count':config['data']['grouping']['text']['scheduled']=1515
            elif kind=='certified':config['data']['grouping']['text']['patient_or_study_independence_certified']=True
            elif kind=='protected-output':config['output_dir']=str(self.f.root/'configs')
            elif kind=='old-output':config['output_dir']=base.read_json(self.parent['jobs'][0]['argv'][4])['output_dir']
            elif kind=='receipt':config['success_receipt']=str(Path(config['output_dir'])/'success.json')
            else:
                config['source_files'].pop('src/mediflow_medvlm/campaign_supervisor.py');config['source_hash']=engine.canonical_hash(config['source_files'])
            config['data_hash']=engine.canonical_hash(config['data']);config['config_hash']=engine.canonical_hash({k:v for k,v in config.items() if k!='config_hash'})
            write(self.config_path,config);job=next(j for j in replacement['jobs'] if j['job_id']==operator.JOB_ID)
            job.update(**{k:config[k] for k in engine.HASH_FIELDS},source_files=config['source_files'],success_receipt=config['success_receipt'])
            job['argv'][6]=config['success_receipt'];write(self.replacement_path,replacement)
            self.binding['configs'][operator.JOB_ID]=ref(self.config_path)
            self.binding['replacement'].update(manifest_sha256=engine.file_hash(self.replacement_path),manifest_hash=engine.canonical_hash(replacement));self.rebind()
            with self.subTest(kind=kind):self.reject(lambda:self.control.call('refine-bootstrap',manifest=replacement))
            for path,raw in originals.items():path.write_bytes(raw)
            self.control.bootstrap_sha256=engine.file_hash(self.binding_path)
        self.binding=binding

    def test_only_new_job_native_same_source_recovery_preserves_counters(self):
        self.control.call('refine-bootstrap',manifest=self.replacement);state=base.read_json(self.f.state_path);record=state['jobs'][operator.JOB_ID]
        record.update(status='ANALYSIS_REQUIRED',attempts=2,failure_count=2,recovery_attempts=1,error_signature='CPU_FIXTURE',
            history=[{'status':'FAILED','attempt':1},{'status':'FAILED','attempt':2}]);write(self.f.state_path,state)
        evidence=write(self.f.root/'operator/evidence.json',{'CPU_FIXTURE_ONLY':True,'actual_model_execution':False})
        receipt=write(self.f.root/'operator/analysis.json',{'job_id':operator.JOB_ID,'run_id':operator.JOB_ID,'job_hash':engine.canonical_hash(self.job),
            'failure_count':2,'error_signature':'CPU_FIXTURE','actual_model':'gpt-6-astra','reasoning_effort':'high','original_model':'CPU_FIXTURE',
            'original_reasoning_effort':'CPU_FIXTURE','escalation_reason':'Synthetic contract fixture','retry_authorized':True,
            'analysis_summary':'CPU fixture only; no actual model execution','model_execution_receipt':ref(evidence)})
        result=self.control.call('authorize-recovery',job_id=operator.JOB_ID,receipt=receipt);changed=result['jobs'][operator.JOB_ID]
        self.assertEqual(changed['status'],'PENDING')
        for key in ('attempts','failure_count','recovery_attempts'):self.assertEqual(changed[key],record[key])
        self.assertEqual(changed['history'][:2],record['history'])
        for key,old in state['jobs'].items():
            if key!=operator.JOB_ID:self.assertEqual(result['jobs'][key],old)
        for job in (operator.PARENT_ID,operator.PREREQUISITES[0],operator.FAMILY_ID):
            with self.subTest(job=job),self.assertRaisesRegex(ValueError,'same-source recovery'):self.control.call('authorize-recovery',job_id=job,receipt=receipt)

    def test_cli_scope_and_cold_import_full_parent_byte_pin(self):
        for command in ('init','prepare','refine-full-dev','activate','finalize','refine-caps'):
            with self.subTest(command=command),self.assertRaises(ValueError):self.control.call(command)
        self.assertTrue(hasattr(operator.load_full_dev(ROOT/'scripts/campaign_control_full_dev.py'),'FullDevMixin'))
        path=self.f.root/'changed-full-dev.py';path.write_bytes((ROOT/'scripts/campaign_control_full_dev.py').read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'operator changed'):operator.load_full_dev(path)
        script=("import importlib.util,sys;"+f"s=importlib.util.spec_from_file_location('cold',{str(ROOT/'scripts/campaign_control_bootstrap.py')!r});"+
            "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);assert 'torch' not in sys.modules;assert 'mediflow_medvlm' not in sys.modules")
        result=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'})
        self.assertEqual(result.returncode,0,result.stderr)
        flags=[]
        for name in ('full-dev-operator','cap-operator','validation-operator','stage-operator','recovery-operator','base-operator','data-root','campaign-root','engine-root',
            'hold-receipt','recovery-binding','stage-binding','validation-binding','cap-binding','full-dev-binding','bootstrap-binding'):
            flags.extend(['--'+name,str(self.f.root/name)])
        for name in ('recovery-binding-sha256','stage-binding-sha256','validation-binding-sha256','cap-binding-sha256','full-dev-binding-sha256','bootstrap-binding-sha256'):
            flags.extend(['--'+name,'d'*64])
        with patch.object(operator,'make_control') as make,redirect_stdout(StringIO()):
            make.return_value.call.return_value={};make.return_value.base.read_json.return_value=self.replacement
            operator.main(flags+['refine-bootstrap','--manifest',str(self.replacement_path)])
            make.return_value.call.assert_called_once_with('refine-bootstrap',manifest=self.replacement,job_id=None,receipt=None)


if __name__=='__main__':unittest.main()
