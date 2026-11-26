"""CPU-only likelihood queue registration with synthetic immutable native proofs."""
import copy
import json
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
operator=fd.load('choice_likelihood_owner_under_test',ROOT/'scripts/campaign_control_choice_likelihood.py')
bootstrap=fd.load('prior_bootstrap_for_likelihood_test',ROOT/'scripts/campaign_control_bootstrap.py')
engine,base,write,ref=fd.engine,fd.base,fd.write,fd.ref


class PriorBootstrapHarness(fd.HarnessCaps):
    """Native ownership guards; only already-reviewed older proof payloads mocked.

    This fixture does not re-run prior-stage registration per new-stage subtest.
    The exact previous facade bytes and complete loader chain are tested below.
    Every current-stage manifest/config/source/proof/lease check remains real.
    """
    def __init__(self,*,bootstrap_binding,bootstrap_binding_sha256,prior_manifest,**kwargs):
        self.bootstrap_path,self.bootstrap_sha256=Path(bootstrap_binding),bootstrap_binding_sha256
        self.prior_manifest=prior_manifest
        super().__init__(**kwargs)
        self.supervisor.__class__=bootstrap.make_supervisor_class(self.engine,self,type(self.supervisor))
    def _bootstrap(self):
        operator.require(base.file_hash(self.bootstrap_path)==self.bootstrap_sha256,'synthetic inherited binding changed')
        return base.read_json(self.bootstrap_path),None,self.prior_manifest


class FixtureControl(operator.ChoiceLikelihoodMixin,PriorBootstrapHarness):
    base=base


class ChoiceLikelihoodOperatorTests(unittest.TestCase):
    executable=fd.FullDevOperatorTests.executable
    success=fd.FullDevOperatorTests.success

    def setUp(self):
        fd.FullDevOperatorTests.setUp(self)
        f=self.f;self.parent=copy.deepcopy(self.after)
        family=next(j for j in self.parent['jobs'] if j['job_id']=='fixture-future-1')
        family.update(job_id=operator.FAMILY_ID,run_id=operator.FAMILY_ID)
        self.states[operator.FAMILY_ID]=self.states.pop('fixture-future-1')
        # Complete synthetic classification populations. Older-stage proofs are
        # opaque CPU fixtures; current-stage source/config/artifact pins are real.
        parents={j['job_id']:j for j in self.parent['jobs']}
        aggregate=base.read_json(parents[operator.GENERATION_ID]['argv'][4])
        initial=base.read_json(aggregate['data']['initial_protocol']['path'])
        rows=[]
        for i in range(1516):
            source,task=('aihub_71875','medical_mcq') if i<587 else (('pubmedqa_labeled','context_qa') if i<637 else ('medquad','medical_qa'))
            row={'sample_id':f'text-{i}','source_id':source,'task_type':task,'split':'validation','language':'en',
                'input':{'question':f'Q{i}'},'supervision':{'answer':'A' if i<587 else 'yes'},'metadata':{}}
            if i<587:row['input']['options']={'A':'one','B':'two'};row['supervision']['option_id']='A'
            rows.append(row)
        validation=f.root/'datasets/choice/validation.jsonl';validation.parent.mkdir(parents=True)
        validation.write_text(''.join(json.dumps(r)+'\n' for r in rows));self.validation_ref=ref(validation)
        initial['validation_references']['text']=self.validation_ref
        initial['protocol_hash']=engine.canonical_hash({k:v for k,v in initial.items() if k!='protocol_hash'})
        self.initial_ref=ref(write(f.root/'protocols/choice-initial-fixture.json',initial))
        for key in fd.operator.ADDED_IDS:
            if not key.startswith('v6-full-dev-T'):continue
            job=parents[key];config=base.read_json(job['argv'][4])
            config['data']['initial_protocol']=self.initial_ref;config['data']['validation']=self.validation_ref
            config['data_hash']=engine.canonical_hash(config['data']);config['config_hash']=engine.canonical_hash({k:v for k,v in config.items() if k!='config_hash'})
            write(job['argv'][4],config);job.update(**{k:config[k] for k in engine.HASH_FIELDS})
        aggregate['data']['initial_protocol']=self.initial_ref
        aggregate['data']['roles']=[ref(parents[key]['argv'][4]) for key in fd.operator.ADDED_IDS]
        aggregate['data_hash']=engine.canonical_hash(aggregate['data']);aggregate['config_hash']=engine.canonical_hash({k:v for k,v in aggregate.items() if k!='config_hash'})
        write(parents[operator.GENERATION_ID]['argv'][4],aggregate)
        parents[operator.GENERATION_ID].update(**{k:aggregate[k] for k in engine.HASH_FIELDS})
        for job in self.parent['jobs']:
            if job['job_id'] in fd.operator.EXECUTABLE_IDS:
                self.success(job)
                receipt=base.read_json(job['success_receipt'])
                if job['job_id']==operator.GENERATION_ID:
                    receipt.update(scope='VALIDATION_ONLY_FULL_DEV_TERMINAL_AUDIT',all_roles_succeeded=True,generation_evaluation_terminal=True,
                        test_files_read=False,test_unlocked=False,family_selection_completed=False,clinical_quality_certified=False,
                        full_dev_completed=False,r2_full_evaluation_completed=False,campaign_terminal=False)
                else:receipt['technical_complete']=job['job_id']!='v6-full-dev-T1-base'
                write(job['success_receipt'],receipt);self.states[job['job_id']]['verification']=engine.verify_success(job)
        prior_source=f.root/'snapshots'/('7'*40);shutil.copytree(self.new_engine,prior_source)
        (prior_source/'src/mediflow_medvlm/campaign_bootstrap.py').write_text('# immutable prior bootstrap CPU fixture\n')
        prior_path=f.root/'configs/bootstrap/first-seed.json'
        prior_job=self.executable(operator.PARENT_ID,prior_source,'mediflow_medvlm.campaign_bootstrap',prior_path)
        prior_job.update(phase='bootstrap',cell_id='ALL_V6',uses_gpu=False,requires_gpu=False,kind='bootstrap',seed=20261023,
            dependencies=[operator.GENERATION_ID],dependency_policy='success')
        prior_config={**{k:prior_job[k] for k in ('job_id','run_id','cell_id','phase','source_files','success_receipt',*engine.HASH_FIELDS)},
            'output_dir':str(f.root/'auxiliary/bootstrap/first_seed'),'resource_policy':{'total_gpu_hours':None}}
        write(prior_path,prior_config);self.success(prior_job)
        position=next(i for i,j in enumerate(self.parent['jobs']) if j['job_id']==operator.GENERATION_ID)+1
        self.parent['jobs'].insert(position,prior_job)
        self.parent_path=write(f.root/'manifests/likelihood-parent.json',self.parent)
        state={'campaign_id':f.root.name,'manifest_hash':engine.canonical_hash(self.parent),'manifest_path':str(self.parent_path),
            'manifest_revisions':[{'kind':'PRESERVED_OLDER_NATIVE_REGISTRATION'}],'queue_owner':'supervisor','jobs':self.states,
            'finalization':'NOT_DONE','updated_at':'preserved'}
        write(f.state_path,state)
        self.prior_binding_path=write(f.root/'operator/prior-bootstrap-binding.json',{'replacement':{'manifest_hash':engine.canonical_hash(self.parent)}})
        self.likelihood_source=f.root/'snapshots'/('8'*40);shutil.copytree(self.new_engine,self.likelihood_source)
        (self.likelihood_source/'src/mediflow_medvlm/campaign_choice_likelihood.py').write_text('# immutable new likelihood fixture\n')
        sources={p.relative_to(self.likelihood_source).as_posix():engine.file_hash(p) for p in self.likelihood_source.glob('src/mediflow_medvlm/**/*.py')}
        self.configs={};self.paths={};self.jobs=[]
        parents={j['job_id']:j for j in self.parent['jobs']}
        for key in operator.EXECUTABLE_IDS:
            role=key in operator.ROLE_IDS
            if role:
                cell,variant=key.removeprefix('v6-choice-likelihood-').split('-');native=parents[f'v6-full-dev-{cell}-{variant}']
            path=f.root/'configs/choice_likelihood'/(key+'.json')
            coverage={k:copy.deepcopy(operator.PROCEDURE[k]) for k in ('text_all_planned','classification_scheduled','classification_source_counts','unsupported_text','unsupported_vqa')}
            if role:
                original=base.read_json(native['argv'][4]);native_receipt=base.read_json(native['success_receipt'])
                proof={'config':ref(native['argv'][4]),'receipt':ref(native['success_receipt']),'artifacts':native_receipt['artifacts'],
                    'native_source_repo':native['cwd'],'aggregate_config':ref(parents[operator.GENERATION_ID]['argv'][4]),
                    'aggregate_receipt':ref(parents[operator.GENERATION_ID]['success_receipt']),'generation_technical_complete':native_receipt['technical_complete'],
                    **{k:original['data'][k] for k in ('initial_protocol','trained_cell','validation','cap_freeze')}}
                data={'native':proof,'ordered_sample_ids':[r['sample_id'] for r in rows],
                    'ordered_sample_ids_sha256':engine.canonical_hash([r['sample_id'] for r in rows]),'coverage':coverage}
            else:data={'roles':[ref(self.paths[k]) for k in operator.ROLE_IDS],'initial_protocol':self.initial_ref,'coverage':coverage}
            config={'schema_version':operator.CONFIG_SCHEMA,'scope':operator.SCOPES['role' if role else 'aggregate'],'kind':'role' if role else 'aggregate',
                'phase':'choice_likelihood_evaluation' if role else 'choice_likelihood','job_id':key,'run_id':key,'cell_id':cell if role else 'ALL_TEXT_V6',
                'seed':20261023,'uses_gpu':role,'device':'cuda' if role else 'cpu','state_path':str(f.state_path),
                'output_dir':str(f.root/'auxiliary/choice_likelihood'/key),'success_receipt':str(f.root/'receipts'/(key+'.json')),
                'data':data,'procedure':copy.deepcopy(operator.PROCEDURE),'resource_policy':aggregate['resource_policy'],
                'source_files':sources,'source_hash':engine.canonical_hash(sources),
                'model_hash':original['model_hash'] if role else engine.canonical_hash([self.configs[k]['model_hash'] for k in operator.ROLE_IDS])}
            if role:
                config.update(**{k:native[k] for k in ('model_id','task','family','role')},
                    **{k:original[k] for k in ('model','fixed_template_date','precision','gpu_ledger_path','supervisor_lock_paths')},
                    split='validation',eval_context=4096,local_files_only=True,trust_remote_code=False)
            config['data_hash']=engine.canonical_hash(config['data']);config['config_hash']=engine.canonical_hash(config);write(path,config)
            job={k:config[k] for k in ('job_id','run_id','phase','cell_id','kind','seed','uses_gpu','source_files','success_receipt',*engine.HASH_FIELDS)}
            job.update(requires_gpu=role,gpu_lock_mode='wrapper',implementation_pending=False,cwd=str(self.likelihood_source),
                argv=[sys.executable,'-m',operator.WORKER_MODULE,'--config',str(path),'--receipt',config['success_receipt']],
                dependencies=[operator.PARENT_ID,operator.GENERATION_ID,native['job_id']] if role else [operator.PARENT_ID,operator.GENERATION_ID,*operator.ROLE_IDS],
                dependency_policy='success' if role else 'terminal')
            if role:job.update(**{k:native[k] for k in ('model_id','task','family','role')})
            self.configs[key]=config;self.paths[key]=path;self.jobs.append(job)
        self.replacement=copy.deepcopy(self.parent)
        position=next(i for i,j in enumerate(self.replacement['jobs']) if j['job_id']==operator.PARENT_ID)+1
        self.replacement['jobs'][position:position]=self.jobs
        self.replacement_path=write(f.root/'manifests/likelihood-proposal.json',self.replacement)
        archive={p.relative_to(self.likelihood_source).as_posix():engine.file_hash(p) for p in sorted(self.likelihood_source.rglob('*')) if p.is_file()}
        source_tree={k:v for k,v in archive.items() if k.startswith('src/') and k.endswith('.py')}
        self.binding={'schema_version':operator.SCHEMA,'campaign_root':str(f.root),'data_root':str(f.data),
            'bootstrap_operator_sha256':operator.BOOTSTRAP_OPERATOR_SHA256,'bootstrap_binding_sha256':engine.file_hash(self.prior_binding_path),
            'parent':{'manifest_path':str(self.parent_path),'manifest_sha256':engine.file_hash(self.parent_path),'manifest_hash':engine.canonical_hash(self.parent)},
            'replacement':{'manifest_path':str(self.replacement_path),'manifest_sha256':engine.file_hash(self.replacement_path),'manifest_hash':engine.canonical_hash(self.replacement)},
            'parent_state_sha256':engine.file_hash(f.state_path),'parent_record_hashes':{key:engine.canonical_hash(record) for key,record in state['jobs'].items()},
            'engine':{'root':str(self.likelihood_source),'commit':self.likelihood_source.name,'archive_files':archive,'archive_tree_hash':base.digest(archive),
                'source_files':source_tree,'source_tree_hash':base.digest(source_tree)},'configs':{key:ref(path) for key,path in self.paths.items()},
            'prerequisites':{key:{**ref(parents[key]['success_receipt']),'job_hash':engine.canonical_hash(parents[key]),'state_hash':engine.canonical_hash(state['jobs'][key])} for key in operator.PREREQUISITES}}
        self.binding_path=write(f.root/'operator/likelihood-binding.json',self.binding)
        self.control=FixtureControl(choice_likelihood_binding=self.binding_path,choice_likelihood_binding_sha256=engine.file_hash(self.binding_path),
            bootstrap_binding=self.prior_binding_path,bootstrap_binding_sha256=engine.file_hash(self.prior_binding_path),prior_manifest=self.parent,
            inherited=self.inherited,roots=f.roots,data_root=f.data,campaign_root=f.root,engine_root=f.base_engine,hold_receipt=f.hold)

    def rebind(self,state=None):
        if state is not None:
            write(self.f.state_path,state);self.binding['parent_state_sha256']=engine.file_hash(self.f.state_path)
            self.binding['parent_record_hashes']={key:engine.canonical_hash(record) for key,record in state['jobs'].items()}
        write(self.binding_path,self.binding);self.control.choice_likelihood_sha256=engine.file_hash(self.binding_path)

    def reject(self,operation=None,match='.*'):
        before=self.f.state_path.read_bytes()
        with patch.object(engine,'atomic_json',side_effect=AssertionError('rejected operation wrote')):
            with self.assertRaisesRegex((ValueError,OSError),match):(operation or (lambda:self.control.call('refine-choice-likelihood',manifest=self.replacement)))()
        self.assertEqual(self.f.state_path.read_bytes(),before)

    def test_atomic_thirteen_job_registration_preserves_all_eighty_four_records_and_ledger(self):
        before=base.read_json(self.f.state_path);self.assertEqual(len(before['jobs']),84)
        ledger=write(self.f.data/'runtime/gpu_budget.json',{'total_gpu_hours':None,'cumulative_seconds':864.1});raw=ledger.read_bytes()
        with patch.object(engine.subprocess,'Popen',side_effect=AssertionError('spawned')):
            result=self.control.call('refine-choice-likelihood',manifest=self.replacement)
        self.assertEqual(len(result['jobs']),97)
        for key,record in before['jobs'].items():self.assertEqual(result['jobs'][key],record)
        for key in operator.EXECUTABLE_IDS:self.assertEqual(result['jobs'][key],{'status':'PENDING','attempts':0,'failure_count':0,'recovery_attempts':0,'history':[]})
        self.assertEqual(ledger.read_bytes(),raw);self.assertEqual(result['manifest_revisions'][:-1],before['manifest_revisions'])
        self.assertEqual(result['manifest_revisions'][-1]['added_jobs'],operator.EXECUTABLE_IDS)
        self.assertFalse((self.f.root/'supervisor.lock').exists());self.assertFalse((self.f.root/'worker.lock').exists())

    def test_exact_matrix_old_jobs_family_deps_and_auxiliary_scopes(self):
        for kind in ('metadata','old','family','missing','extra','position','gpu','requires','phase','kind','seed','boolseed','deps','policy','argv','unknown'):
            value=copy.deepcopy(self.replacement);jobs={j['job_id']:j for j in value['jobs']};new=jobs[operator.ROLE_IDS[0]]
            if kind=='metadata':value['declared_seed']=42
            elif kind=='old':jobs[operator.PARENT_ID]['config_hash']='f'*64
            elif kind=='family':jobs[operator.FAMILY_ID]['dependencies']=[operator.AGGREGATE_ID]
            elif kind=='missing':value['jobs'].remove(new)
            elif kind=='extra':value['jobs'].append({**new,'job_id':'hidden','run_id':'hidden'})
            elif kind=='position':value['jobs'].remove(new);value['jobs'].append(new)
            elif kind=='gpu':new['uses_gpu']=False
            elif kind=='requires':new['requires_gpu']=False
            elif kind=='phase':new['phase']='full_dev'
            elif kind=='kind':new['kind']='aggregate'
            elif kind=='seed':new['seed']=42
            elif kind=='boolseed':new['seed']=True
            elif kind=='deps':new['dependencies']=[operator.GENERATION_ID]
            elif kind=='policy':new['dependency_policy']='terminal'
            elif kind=='argv':new['argv'][2]='another.module'
            else:new['hidden_queue']=True
            with self.subTest(kind=kind),self.assertRaises(ValueError):operator.validate_pair(engine,self.parent,value)

    def test_external_pins_and_all_new_source_imports(self):
        prior=base.read_json(self.binding['prerequisites'][operator.PARENT_ID]['path'])
        for path in (self.binding_path,self.replacement_path,self.parent_path,self.paths[operator.ROLE_IDS[0]],
                Path(self.binding['prerequisites'][operator.PARENT_ID]['path']),Path(prior['artifacts'][0]['path']),self.likelihood_source/'README.md'):
            raw=path.read_bytes();path.write_bytes(raw+b' ')
            with self.subTest(path=path):self.reject()
            path.write_bytes(raw)

    def test_active_parent_pids_state_and_all_canonical_leases(self):
        original=base.read_json(self.f.state_path)
        for kind in ('running','worker','heartbeat','future','orphan','owner','counter','prerequisite'):
            state=copy.deepcopy(original);record=state['jobs'][operator.FAMILY_ID]
            if kind=='running':record['status']='RUNNING'
            elif kind=='worker':record['worker_identity']=engine.process_identity()
            elif kind=='heartbeat':
                record['attempt_dir']=str(self.f.root/'attempts/orphan');write(Path(record['attempt_dir'])/'heartbeat.json',{'child_identity':engine.process_identity()})
            elif kind=='future':record['attempts']=1
            elif kind=='orphan':state['jobs']['hidden']={'status':'PENDING'}
            elif kind=='owner':state['queue_owner']='other'
            elif kind=='counter':state['jobs'][operator.PREREQUISITES[0]]['failure_count']+=1
            else:state['jobs'][operator.PARENT_ID]['status']='FAILED'
            self.rebind(state)
            with self.subTest(kind=kind):self.reject()
        self.rebind(original)
        drift=copy.deepcopy(original);drift['updated_at']='unpinned';write(self.f.state_path,drift)
        self.reject(match='parent bytes');self.rebind(original)
        for key in ('worker','gpu'):
            with self.control._physical_lock(self.control.physical[key]):self.reject(match='lease busy')
        before=self.f.state_path.read_bytes()
        with self.control._physical_lock(self.control.physical['supervisor']):
            self.assertEqual(self.control.call('refine-choice-likelihood',manifest=self.replacement),{'status':'SUPERVISOR_BUSY'})
        self.assertEqual(self.f.state_path.read_bytes(),before)

    def test_inherited_migration_hold_and_lock_inode_aliases(self):
        hold=base.read_json(self.f.hold);hold['dispatch_control']['tick_prohibited']=False;write(self.f.hold,hold)
        self.reject(match='MIGRATION_HOLD');hold['dispatch_control']['tick_prohibited']=True;write(self.f.hold,hold)
        for name in ('worker.lock','supervisor.lock'):
            path=self.f.root/name;path.touch();self.reject(match='namespace lock files');path.unlink()
        path=self.control.physical['gpu'];new=path.with_name('replacement.lock');new.touch();os.replace(new,path)
        self.reject(match='physical lock inode')

    def test_idempotent_audit_and_atomic_interruption_preserve_records(self):
        before=self.f.state_path.read_bytes();atomic=engine.atomic_json
        def interrupt(path,value):
            if Path(path)==self.f.state_path:raise OSError('CPU pointer interruption')
            atomic(path,value)
        with patch.object(engine,'atomic_json',side_effect=interrupt),self.assertRaises(OSError):self.control.call('refine-choice-likelihood',manifest=self.replacement)
        self.assertEqual(self.f.state_path.read_bytes(),before)
        self.control.call('refine-choice-likelihood',manifest=self.replacement);state=base.read_json(self.f.state_path)
        state['jobs'][operator.ROLE_IDS[0]].update(status='FAILED',attempts=2,failure_count=2,recovery_attempts=1);write(self.f.state_path,state)
        before=self.f.state_path.read_bytes()
        with patch.object(engine,'atomic_json',side_effect=AssertionError('rewrote')):self.control.call('refine-choice-likelihood',manifest=self.replacement)
        self.assertEqual(self.f.state_path.read_bytes(),before)
        state['manifest_revisions'][-1]['original_record_hashes']={};write(self.f.state_path,state);self.reject(match='immutable choice_likelihood refinement audit')

    def test_native_runtime_gpu_role_cpu_gate_and_failure_sibling_behavior(self):
        cls=type(self.control.supervisor)
        self.assertEqual({k for k,v in cls.__dict__.items() if callable(v)},{'refine_choice_likelihood'})
        for name in ('tick','status','load','authorize_recovery','_failure'):self.assertIs(getattr(cls,name),getattr(engine.Supervisor,name))
        self.control.call('refine-choice-likelihood',manifest=self.replacement)
        original=copy.deepcopy(base.read_json(self.f.state_path)['jobs'])
        state=base.read_json(self.f.state_path);state['jobs'][operator.ROLE_IDS[0]].update(status='FAILED',attempts=3,failure_count=3,recovery_attempts=2);write(self.f.state_path,state)
        with patch.object(engine.subprocess,'Popen') as spawn:
            spawn.return_value.pid=999999999;result=self.control.call('tick')
        self.assertEqual(spawn.call_count,1);self.assertEqual(result['jobs'][operator.ROLE_IDS[1]]['attempts'],1)
        self.assertEqual(len(spawn.call_args.kwargs['pass_fds']),2)
        state=base.read_json(self.f.state_path)
        for key in operator.ROLE_IDS:state['jobs'][key].update(status='FAILED',attempts=3,failure_count=3,recovery_attempts=2)
        write(self.f.state_path,state)
        with self.control._physical_lock(self.control.physical['gpu']),patch.object(engine.subprocess,'Popen') as spawn:
            spawn.return_value.pid=999999999;result=self.control.call('tick')
        self.assertEqual(spawn.call_count,1);self.assertEqual(result['jobs'][operator.AGGREGATE_ID]['attempts'],1)
        self.assertEqual(len(spawn.call_args.kwargs['pass_fds']),1)
        self.assertEqual(result['jobs'][operator.FAMILY_ID]['attempts'],0)
        for key in self.binding['parent_record_hashes']:self.assertEqual(result['jobs'][key],original[key])
        self.assertEqual(result['jobs'][operator.GENERATION_ID]['status'],'SUCCEEDED')

    def test_self_hashed_scientific_overrides_and_proof_omissions_rejected(self):
        originals={p:p.read_bytes() for p in (self.binding_path,self.replacement_path,*self.paths.values())}
        binding=copy.deepcopy(self.binding)
        kinds=('unknown','scope','procedure','context','eos','length-normalization','unsupported','coverage','ids','native-config',
            'native-artifacts','native-source','generation-technical','checkpoint','cap','validation','model','ledger','leases','resource',
            'aggregate-roles','protected-output','old-output','duplicate-output','receipt','sources')
        for kind in kinds:
            key=operator.AGGREGATE_ID if kind=='aggregate-roles' else operator.ROLE_IDS[0]
            config=copy.deepcopy(self.configs[key]);replacement=copy.deepcopy(self.replacement);self.binding=copy.deepcopy(binding)
            if kind=='unknown':config['max_time']=60
            elif kind=='scope':config['scope']='NATIVE_GENERATION'
            elif kind=='procedure':config['procedure']['generation_used']=True
            elif kind=='context':config['eval_context']=2048
            elif kind=='eos':config['procedure']['eos_included']=True
            elif kind=='length-normalization':config['procedure']['score']='mean label-token log probability'
            elif kind=='unsupported':config['procedure']['unsupported_vqa']['model_roles']=0
            elif kind=='coverage':config['data']['coverage']['classification_scheduled']=636
            elif kind=='ids':config['data']['ordered_sample_ids'].reverse()
            elif kind=='native-config':config['data']['native']['config']['sha256']='f'*64
            elif kind=='native-artifacts':config['data']['native']['artifacts']=[]
            elif kind=='native-source':config['data']['native']['native_source_repo']=str(self.likelihood_source)
            elif kind=='generation-technical':config['data']['native']['generation_technical_complete']=True
            elif kind=='checkpoint':config['data']['native']['trained_cell']={}
            elif kind=='cap':config['data']['native']['cap_freeze']={}
            elif kind=='validation':config['data']['native']['validation']['path']=str(self.f.root/'test.jsonl')
            elif kind=='model':config['model']={'unreviewed':True}
            elif kind=='ledger':config['gpu_ledger_path']=str(self.f.root/'gpu_budget.json')
            elif kind=='leases':config['supervisor_lock_paths']=[str(self.f.root/'worker.lock')]
            elif kind=='resource':config['resource_policy']={'total_gpu_hours':1}
            elif kind=='aggregate-roles':config['data']['roles']=config['data']['roles'][:-1]
            elif kind=='protected-output':config['output_dir']=str(self.f.root/'configs')
            elif kind=='old-output':config['output_dir']=base.read_json(self.parent['jobs'][0]['argv'][4])['output_dir']
            elif kind=='duplicate-output':config['output_dir']=self.configs[operator.ROLE_IDS[1]]['output_dir']
            elif kind=='receipt':config['success_receipt']=str(Path(config['output_dir'])/'success.json')
            else:
                config['source_files'].pop('src/mediflow_medvlm/evaluation/metrics.py');config['source_hash']=engine.canonical_hash(config['source_files'])
            config['data_hash']=engine.canonical_hash(config['data']);config['config_hash']=engine.canonical_hash({k:v for k,v in config.items() if k!='config_hash'})
            write(self.paths[key],config);job=next(j for j in replacement['jobs'] if j['job_id']==key)
            job.update(**{k:config[k] for k in engine.HASH_FIELDS},source_files=config['source_files'],success_receipt=config['success_receipt'])
            job['argv'][6]=config['success_receipt'];self.binding['configs'][key]=ref(self.paths[key])
            if key!=operator.AGGREGATE_ID:
                aggregate=copy.deepcopy(self.configs[operator.AGGREGATE_ID]);aggregate['data']['roles'][0]=ref(self.paths[key])
                aggregate['data_hash']=engine.canonical_hash(aggregate['data']);aggregate['config_hash']=engine.canonical_hash({k:v for k,v in aggregate.items() if k!='config_hash'})
                write(self.paths[operator.AGGREGATE_ID],aggregate);self.binding['configs'][operator.AGGREGATE_ID]=ref(self.paths[operator.AGGREGATE_ID])
                next(j for j in replacement['jobs'] if j['job_id']==operator.AGGREGATE_ID).update(**{k:aggregate[k] for k in engine.HASH_FIELDS})
            write(self.replacement_path,replacement)
            self.binding['replacement'].update(manifest_sha256=engine.file_hash(self.replacement_path),manifest_hash=engine.canonical_hash(replacement));self.rebind()
            with self.subTest(kind=kind):self.reject(lambda:self.control.call('refine-choice-likelihood',manifest=replacement))
            for path,raw in originals.items():path.write_bytes(raw)
            self.control.choice_likelihood_sha256=engine.file_hash(self.binding_path)
        self.binding=binding

    def test_native_same_source_recovery_only_new_jobs_no_counter_reset(self):
        self.control.call('refine-choice-likelihood',manifest=self.replacement);state=base.read_json(self.f.state_path)
        key=operator.ROLE_IDS[0];job=self.jobs[0];record=state['jobs'][key]
        record.update(status='ANALYSIS_REQUIRED',attempts=2,failure_count=2,recovery_attempts=1,error_signature='CPU_FIXTURE',
            history=[{'status':'FAILED','attempt':1},{'status':'FAILED','attempt':2}]);write(self.f.state_path,state)
        evidence=write(self.f.root/'operator/evidence.json',{'CPU_FIXTURE_ONLY':True,'actual_model_execution':False})
        receipt=write(self.f.root/'operator/analysis.json',{'job_id':key,'run_id':key,'job_hash':engine.canonical_hash(job),'failure_count':2,
            'error_signature':'CPU_FIXTURE','actual_model':'gpt-6-astra','reasoning_effort':'high','original_model':'CPU_FIXTURE','original_reasoning_effort':'CPU_FIXTURE',
            'escalation_reason':'Synthetic contract fixture','retry_authorized':True,'analysis_summary':'CPU fixture only; no actual model execution','model_execution_receipt':ref(evidence)})
        result=self.control.call('authorize-recovery',job_id=key,receipt=receipt);changed=result['jobs'][key]
        self.assertEqual(changed['status'],'PENDING')
        for field in ('attempts','failure_count','recovery_attempts'):self.assertEqual(changed[field],record[field])
        self.assertEqual(changed['history'][:2],record['history'])
        for key in (operator.PARENT_ID,operator.GENERATION_ID,operator.FAMILY_ID):
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'same-source recovery'):self.control.call('authorize-recovery',job_id=key,receipt=receipt)

    def test_cli_scope_full_chain_pin_and_cold_cpu_import(self):
        for command in ('init','prepare','activate','refine-bootstrap','refine-full-dev','finalize'):
            with self.subTest(command=command),self.assertRaises(ValueError):self.control.call(command)
        prior=operator.load_bootstrap(ROOT/'scripts/campaign_control_bootstrap.py')
        full=prior.load_full_dev(ROOT/'scripts/campaign_control_full_dev.py');self.assertTrue(hasattr(full,'FullDevMixin'))
        path=self.f.root/'changed-parent.py';path.write_bytes((ROOT/'scripts/campaign_control_bootstrap.py').read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'operator changed'):operator.load_bootstrap(path)
        script=("import importlib.util,sys,os;"+f"s=importlib.util.spec_from_file_location('cold',{str(ROOT/'scripts/campaign_control_choice_likelihood.py')!r});"+
            "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);assert 'torch' not in sys.modules;assert 'mediflow_medvlm' not in sys.modules;assert os.environ['CUDA_VISIBLE_DEVICES']=='UNCHANGED_CPU_FIXTURE'")
        result=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'UNCHANGED_CPU_FIXTURE','PYTHONDONTWRITEBYTECODE':'1'})
        self.assertEqual(result.returncode,0,result.stderr)
        flags=[]
        for name in ('bootstrap-operator','full-dev-operator','cap-operator','validation-operator','stage-operator','recovery-operator','base-operator','data-root','campaign-root','engine-root',
            'hold-receipt','recovery-binding','stage-binding','validation-binding','cap-binding','full-dev-binding','bootstrap-binding','choice-likelihood-binding'):
            flags.extend(['--'+name,str(self.f.root/name)])
        for name in ('recovery-binding-sha256','stage-binding-sha256','validation-binding-sha256','cap-binding-sha256','full-dev-binding-sha256','bootstrap-binding-sha256','choice-likelihood-binding-sha256'):
            flags.extend(['--'+name,'d'*64])
        with patch.object(operator,'make_control') as make,redirect_stdout(StringIO()):
            make.return_value.call.return_value={};make.return_value.base.read_json.return_value=self.replacement
            operator.main(flags+['refine-choice-likelihood','--manifest',str(self.replacement_path)])
            make.return_value.call.assert_called_once_with('refine-choice-likelihood',manifest=self.replacement,job_id=None,receipt=None)


if __name__=='__main__':unittest.main()
