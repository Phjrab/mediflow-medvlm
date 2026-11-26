#!/usr/bin/env python3
"""Add twelve pinned GPU likelihood roles and one CPU aggregate through the existing sole native supervisor."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
BOOTSTRAP_OPERATOR_SHA256 = 'c700c31e2df911de22faaaec57fa3df7a982686f94149550da426fd615852ad9'
SCHEMA = 'mediflow.recorded-choice-likelihood-refinement.v1'
CONFIG_SCHEMA = 'medvlm.v6.choice-likelihood-config.v1'
SCOPES = {'role':'VALIDATION_ONLY_NATIVE_CHOICE_LIKELIHOOD_AUXILIARY','aggregate':'VALIDATION_ONLY_CHOICE_LIKELIHOOD_TERMINAL_AUDIT'}
WORKER_MODULE = 'mediflow_medvlm.campaign_choice_likelihood'
AGGREGATE_ID = 'v6-choice-likelihood'
ROLE_IDS = [f'v6-choice-likelihood-T{index}-{role}' for index in range(1,7) for role in ('base','tuned')]
EXECUTABLE_IDS = [*ROLE_IDS,AGGREGATE_ID]
PARENT_ID = 'v6-bootstrap'
GENERATION_ID = 'v6-full_dev'
FAMILY_ID = 'v6-family_selection'
CELL_IDS = ['T1','T2','T3','T4','T5','T6','V1','V2','V3','V4']
ROLES = ('base','tuned')
PREREQUISITES = ['v6-first-seed-'+cell for cell in CELL_IDS] + ['v6-first_seed_final'] + [
    f'v6-caps-initial-{cell}-{role}' for cell in CELL_IDS for role in ROLES] + ['v6-caps-decision'] + [
    f'v6-caps-bumped-{cell}-{role}' for cell in CELL_IDS for role in ROLES] + ['v6-validation_caps'] + [
    f'v6-full-dev-{cell}-{role}' for cell in CELL_IDS for role in ROLES] + [GENERATION_ID,PARENT_ID]
EVENT = 'RECORDED_CHOICE_LIKELIHOOD_REFINEMENT'
PROCEDURE = {'event': 'exact label text appended to native generation assistant prefix',
 'score': 'sum of all conditional label-token log probabilities; causal shift; no length '
          'normalization',
 'eos_included': False,
 'termination_tokens_included': False,
 'eos_reason': 'score exact label text before family-specific EOS/control suffix; fixed before '
               'execution',
 'boundary': 'candidate processor IDs have exact native prompt prefix; continuation decodes to '
             'exact label',
 'completed_assistant_whitespace': 'record native discrepancy; never silently insert or substitute '
                                   'whitespace',
 'label_order': 'medical_mcq native input options iteration order; context_qa yes,no,maybe',
 'tie_policy': 'exact numeric maximum; preserve all tied labels; choose first in declared label '
               'order',
 'context': 4096,
 'input_truncation': False,
 'generation_used': False,
 'supported': {'aihub_71875': 'medical_mcq', 'pubmedqa_labeled': 'context_qa'},
 'classification_scheduled': 637,
 'classification_source_counts': {'aihub_71875': 587, 'pubmedqa_labeled': 50},
 'text_all_planned': 1516,
 'unsupported_text': 879,
 'unsupported_vqa': {'status': 'UNSUPPORTED',
                     'reason': 'choice likelihood is undefined for freeform/image QA',
                     'validation_rows_per_role': 512,
                     'model_roles': 8},
 'denominator': 'all 637 planned classification rows, including technical failures; unsupported '
                'text rows reported separately',
 'interpretation': 'format-reduced auxiliary comparison; not pure clinical capability or clinical '
                   'truth'}
BASE_JOB_FIELDS = {'job_id','run_id','phase','cell_id','kind','seed','dependencies','dependency_policy','implementation_pending',
    'uses_gpu','requires_gpu','gpu_lock_mode','argv','cwd','success_receipt','source_files','source_hash','config_hash','data_hash','model_hash'}


def require(condition,message):
    if not condition:raise ValueError(message)


def load_bootstrap(path):
    path=Path(path)
    require(path.is_absolute() and '..' not in path.parts and not any(p.is_symlink() for p in (path,*path.parents)),
        'absolute nonsymlink bootstrap operator required')
    raw=path.read_bytes()
    require(hashlib.sha256(raw).hexdigest()==BOOTSTRAP_OPERATOR_SHA256,'original bootstrap operator changed')
    spec=importlib.util.spec_from_loader('_pinned_bootstrap_for_choice_likelihood',loader=None)
    module=importlib.util.module_from_spec(spec);module.__file__=str(path)
    exec(compile(raw,str(path),'exec'),module.__dict__)
    return module


def validate_pair(engine,parent,replacement):
    engine.validate_manifest(parent);engine.validate_manifest(replacement)
    same=lambda a,b:engine.canonical_hash(a)==engine.canonical_hash(b)
    require(same({k:v for k,v in parent.items() if k!='jobs'},{k:v for k,v in replacement.items() if k!='jobs'}),
        'campaign metadata must remain immutable')
    before={j['job_id']:j for j in parent['jobs']};after={j['job_id']:j for j in replacement['jobs']}
    require(len(before)==84 and set(PREREQUISITES+[FAMILY_ID]).issubset(before) and not set(EXECUTABLE_IDS).intersection(before),
        'exact eighty-four-job completed bootstrap parent required')
    order=[j['job_id'] for j in parent['jobs']];position=order.index(PARENT_ID)+1
    order[position:position]=EXECUTABLE_IDS
    require([j['job_id'] for j in replacement['jobs']]==order,'exact twelve likelihood roles and one aggregate insertion required')
    require(all(same(after[key],job) for key,job in before.items()),'every original job and family dependency must remain identical')
    for key in EXECUTABLE_IDS:
        job=after[key];role=key in ROLE_IDS;argv=job.get('argv',[])
        require(set(job)==BASE_JOB_FIELDS|({'model_id','task','family','role'} if role else set())
            and job.get('run_id')==key and job.get('phase')==('choice_likelihood_evaluation' if role else 'choice_likelihood')
            and job.get('kind')==('role' if role else 'aggregate') and type(job.get('seed')) is int and job['seed']==20261023
            and job.get('implementation_pending') is False and job.get('uses_gpu') is role and job.get('requires_gpu') is role
            and job.get('gpu_lock_mode')=='wrapper' and len(argv)==7 and argv[1:4]==['-m',WORKER_MODULE,'--config']
            and argv[5:]==['--receipt',job.get('success_receipt')],'exact likelihood auxiliary declaration and wrapper required')
        if role:
            cell,variant=key.removeprefix('v6-choice-likelihood-').split('-');native=before[f'v6-full-dev-{cell}-{variant}']
            require(all(same(job.get(k),native.get(k)) for k in ('cell_id','model_id','task','family','role','seed'))
                and job.get('task')=='text' and job.get('dependencies')==[PARENT_ID,GENERATION_ID,native['job_id']]
                and job.get('dependency_policy')=='success','exact independent text likelihood role required')
        else:
            require(job.get('cell_id')=='ALL_TEXT_V6' and job.get('dependencies')==[PARENT_ID,GENERATION_ID,*ROLE_IDS]
                and job.get('dependency_policy')=='terminal','likelihood aggregate must audit all terminal roles')
    return [after[key] for key in EXECUTABLE_IDS]


def make_supervisor_class(engine,owner,inherited):
    class ChoiceLikelihoodSupervisor(inherited):
        def refine_choice_likelihood(self,revised):
            with engine.try_lock(self.root/'supervisor.lock') as lock:
                if lock is None:return {'status':'SUPERVISOR_BUSY'}
                binding,parent,replacement=owner._choice_likelihood()
                require(engine.canonical_hash(revised)==engine.canonical_hash(replacement),'exact pinned choice_likelihood replacement required')
                current,state=self.load();digest=engine.canonical_hash(replacement)
                require(state.get('queue_owner')=='supervisor','sole Supervisor ownership required')
                if state['manifest_hash']==digest:
                    events=[e for e in state.get('manifest_revisions',[]) if e.get('kind')==EVENT
                        and e.get('manifest_hash')==digest and e.get('choice_likelihood_binding_sha256')==owner.choice_likelihood_sha256
                        and e.get('added_jobs')==EXECUTABLE_IDS and e.get('original_record_hashes')==binding['parent_record_hashes']]
                    require(len(events)==1 and engine.canonical_hash(current)==digest
                        and set(state['jobs'])==set(binding['parent_record_hashes'])|set(EXECUTABLE_IDS)
                        and all(engine.canonical_hash(state['jobs'][key])==sha for key,sha in binding['parent_record_hashes'].items()),
                        'exact immutable choice_likelihood refinement audit and old records required')
                    return self.status()
                require(engine.canonical_hash(current)==binding['parent']['manifest_hash']
                    and engine.canonical_hash(current)==engine.canonical_hash(parent),'choice_likelihood refinement parent is not active')
                require(set(state['jobs'])=={j['job_id'] for j in parent['jobs']}
                    and owner.base.file_hash(self.state_path)==binding['parent_state_sha256']
                    and all(engine.canonical_hash(state['jobs'][key])==sha for key,sha in binding['parent_record_hashes'].items()),
                    'exact original state records and parent bytes required')
                for key,record in state['jobs'].items():
                    health={}
                    if record.get('attempt_dir'):
                        path=owner.base.absolute(record['attempt_dir'])/'heartbeat.json'
                        if path.exists():health=owner.base.read_json(path)
                    require(record.get('status') not in {'STARTING','RUNNING'} and not any(engine.same_process(identity)
                        for identity in (record.get('worker_identity'),health.get('worker_identity'),health.get('child_identity'))),
                        'worker active; choice_likelihood refinement is registration only')
                    if key not in PREREQUISITES:
                        require(record.get('status')=='IMPLEMENTATION_PENDING'
                            and all(type(record.get(k)) is int and record[k]==0 for k in ('attempts','failure_count','recovery_attempts'))
                            and not record.get('history') and all(record.get(k) is None for k in ('attempt_dir','launch_token','worker_identity',
                                'verification','recovery_authorization_hash','recovery_authorization')),'future jobs must remain untouched and unstarted')
                with ExitStack() as leases:
                    for path in (self.root/'worker.lock',self.data_root/'runtime/medvlm_gpu.lock'):
                        require(leases.enter_context(engine.try_lock(path)) is not None,'worker/GPU lease busy')
                    jobs={j['job_id']:j for j in parent['jobs']};proofs={}
                    for key in PREREQUISITES:
                        record,pins=state['jobs'][key],binding['prerequisites'][key]
                        require(record.get('status')=='SUCCEEDED' and record.get('exit_code')==0
                            and type(record.get('attempts')) is int and record['attempts']>0
                            and engine.canonical_hash(record)==pins['state_hash'],'all seventy-five prerequisites must be verified successes')
                        proof=engine.verify_success(jobs[key]);verification=record.get('verification',{})
                        require(verification.get('receipt')==pins['path'] and verification.get('receipt_sha256')==pins['sha256']
                            and verification.get('artifact_count')==proof['artifact_count'],'original success verification changed')
                        proofs[key]=proof
                    owner._verify()
                    require(owner.base.file_hash(self.state_path)==binding['parent_state_sha256'],'parent state changed during receipt audit')
                    version=self.root/'manifests'/(digest+'.json')
                    if version.exists():require(engine.canonical_hash(owner.base.read_json(version))==digest,'immutable manifest version changed')
                    else:engine.atomic_json(version,replacement)
                    state.setdefault('manifest_revisions',[]).append({'kind':EVENT,'previous_hash':state['manifest_hash'],
                        'previous_manifest_path':state['manifest_path'],'manifest_hash':digest,'manifest_path':str(version),
                        'added_jobs':EXECUTABLE_IDS,'choice_likelihood_binding':str(owner.choice_likelihood_path),'choice_likelihood_binding_sha256':owner.choice_likelihood_sha256,
                        'original_record_hashes':binding['parent_record_hashes'],'prerequisite_proofs':proofs,'at':engine.now()})
                    for key in EXECUTABLE_IDS:
                        state['jobs'][key]={'status':'PENDING','attempts':0,'failure_count':0,'recovery_attempts':0,'history':[]}
                    state.update(manifest_hash=digest,manifest_path=str(version),updated_at=engine.now())
                    engine.atomic_json(self.state_path,state)
                return self.status()
    return ChoiceLikelihoodSupervisor


class ChoiceLikelihoodMixin:
    def __init__(self,*,choice_likelihood_binding,choice_likelihood_binding_sha256,**kwargs):
        require(isinstance(choice_likelihood_binding_sha256,str) and re.fullmatch(r'[0-9a-f]{64}',choice_likelihood_binding_sha256),
            'external choice_likelihood binding SHA256 required')
        self.choice_likelihood_path,self.choice_likelihood_sha256=Path(choice_likelihood_binding),choice_likelihood_binding_sha256
        super().__init__(**kwargs)
        self.choice_likelihood_path=self.base.absolute(self.choice_likelihood_path)
        self.supervisor.__class__=make_supervisor_class(self.engine,self,type(self.supervisor))

    def _choice_likelihood(self):
        raw=self.base.read_bytes(self.choice_likelihood_path)
        require(hashlib.sha256(raw).hexdigest()==self.choice_likelihood_sha256,'reviewed choice_likelihood binding changed')
        binding=json.loads(raw)
        require(binding.get('schema_version')==SCHEMA and binding.get('campaign_root')==str(self.root)
            and binding.get('data_root')==str(self.data_root) and binding.get('bootstrap_operator_sha256')==BOOTSTRAP_OPERATOR_SHA256
            and binding.get('bootstrap_binding_sha256')==self.bootstrap_sha256,'choice_likelihood namespace or inherited owner binding changed')
        inherited,_,native_parent=self._bootstrap()
        require(binding['parent']['manifest_hash']==inherited['replacement']['manifest_hash'],'choice_likelihood must extend exact bootstrap manifest')
        manifests=[]
        for pins in (binding['parent'],binding['replacement']):
            path=self.base.absolute(pins['manifest_path'])
            require(path.is_relative_to(self.root/'manifests') and self.base.file_hash(path)==pins['manifest_sha256'],'choice_likelihood manifest path or bytes changed')
            value=self.base.read_json(path)
            require(self.engine.canonical_hash(value)==pins['manifest_hash'],'choice_likelihood manifest identity changed');manifests.append(value)
        require(self.engine.canonical_hash(manifests[0])==self.engine.canonical_hash(native_parent),'choice_likelihood parent differs from inherited manifest')
        jobs=validate_pair(self.engine,*manifests);parents={j['job_id']:j for j in manifests[0]['jobs']}
        require(set(binding['parent_record_hashes'])==set(parents)
            and all(isinstance(v,str) and re.fullmatch(r'[0-9a-f]{64}',v) for v in binding['parent_record_hashes'].values())
            and set(binding['prerequisites'])==set(PREREQUISITES) and set(binding['configs'])==set(EXECUTABLE_IDS),'exact old records, seventy-five proofs and thirteen configs required')
        for key,pins in binding['prerequisites'].items():
            path=self.base.absolute(pins['path'])
            require(set(pins)=={'path','sha256','job_hash','state_hash'} and path.is_relative_to(self.root)
                and str(path)==parents[key]['success_receipt'] and self.base.file_hash(path)==pins['sha256']
                and self.engine.canonical_hash(parents[key])==pins['job_hash']
                and pins['state_hash']==binding['parent_record_hashes'][key],'original prerequisite receipt, job or state pins changed')
        pins=binding['engine'];source=self.base.absolute(pins['root'])
        require(source.parent==self.root/'snapshots' and source.name==pins['commit'] and re.fullmatch(r'[0-9a-f]{40}',pins['commit']),
            'new immutable source commit archive required')
        archive={}
        for path in sorted(source.rglob('*')):
            require(not path.is_symlink(),'source archive symlinks forbidden')
            if not path.is_dir():archive[path.relative_to(source).as_posix()]=self.base.file_hash(path)
        require(bool(archive) and archive==pins['archive_files'] and self.base.digest(archive)==pins['archive_tree_hash'],'complete choice_likelihood source archive changed')
        sources={k:v for k,v in archive.items() if k.startswith('src/') and k.endswith('.py')}
        require(sources==pins['source_files'] and self.base.digest(sources)==pins['source_tree_hash']
            and 'src/mediflow_medvlm/campaign_choice_likelihood.py' in sources,'complete choice_likelihood source changed')
        outputs=[];receipts=[]
        for job in jobs:
            require(job['cwd']==str(source),'new likelihood cwd differs from pinned archive');self.engine.verify_source(job)
            reference=binding['configs'][job['job_id']];path=self.base.absolute(job['argv'][4])
            require(path.is_relative_to(self.root/'configs') and str(path)==reference['path']
                and self.base.file_hash(path)==reference['sha256'],'reviewed likelihood config path or bytes changed')
            config=self.base.read_json(path)
            require(config.get('schema_version')==CONFIG_SCHEMA
                and config.get('config_hash')==self.base.digest({k:v for k,v in config.items() if k!='config_hash'})
                and all(self.base.digest(config.get(k))==self.base.digest(job.get(k)) for k in
                    ('job_id','run_id','cell_id','source_files','source_hash','config_hash','data_hash','model_hash','success_receipt')),
                'likelihood config identity differs from pinned job')
            expected_sources={**parents[GENERATION_ID]['source_files'],'src/mediflow_medvlm/campaign_choice_likelihood.py':sources['src/mediflow_medvlm/campaign_choice_likelihood.py']}
            require(config['source_files']==expected_sources and all(sources.get(k)==v for k,v in expected_sources.items()),
                'likelihood must capture unchanged native imports and its complete worker source')
            self._config_policy_choice_likelihood(config,job,binding,parents)
            outputs.append(self.base.absolute(config['output_dir']));receipts.append(self.base.absolute(config['success_receipt']))
        require(len(set(outputs))==13 and len(set(receipts))==13
            and all(not a.is_relative_to(b) for i,a in enumerate(outputs) for j,b in enumerate(outputs) if i!=j)
            and all(not r.is_relative_to(o) for r in receipts for o in outputs),'likelihood outputs and receipts must be distinct and nonoverlapping')
        return binding,manifests[0],manifests[1]

    def _config_policy_choice_likelihood(self,config,job,binding,parents):
        # Fixed scientific procedure and actual native proof structure are checked
        # without importing the new GPU worker into the old owner runtime.
        role=job['job_id'] in ROLE_IDS
        common={'schema_version','scope','kind','phase','job_id','run_id','cell_id','seed','uses_gpu','device','state_path',
            'output_dir','success_receipt','data','procedure','model_hash','resource_policy','source_files','source_hash','data_hash','config_hash'}
        gpu_fields={'model_id','task','family','role','split','eval_context','model','fixed_template_date','precision',
            'local_files_only','trust_remote_code','gpu_ledger_path','supervisor_lock_paths'}
        require(set(config)==common|(gpu_fields if role else set())
            and config.get('scope')==SCOPES['role' if role else 'aggregate'] and config.get('kind')==('role' if role else 'aggregate')
            and config.get('phase')==job['phase'] and config.get('uses_gpu') is role
            and type(config.get('seed')) is int and config['seed']==20261023
            and config.get('device')==('cuda' if role else 'cpu') and config.get('state_path')==str(self.supervisor.state_path)
            and self.base.digest(config['procedure'])==self.base.digest(PROCEDURE),'likelihood fixed procedure, keyset, scope or identity changed')
        output,receipt=self.base.absolute(config['output_dir']),self.base.absolute(config['success_receipt'])
        auxiliary=self.root/'auxiliary/choice_likelihood'
        require(output!=auxiliary and output.is_relative_to(auxiliary)
            and receipt==self.root/'receipts'/(job['job_id']+'.json'),'likelihood dedicated auxiliary output and receipt required')
        for old in parents.values():
            require(receipt!=Path(old.get('success_receipt','/')),'likelihood receipt overlaps original receipt')
            if old.get('argv') and not old.get('implementation_pending'):
                old_config=self.base.read_json(self.base.absolute(old['argv'][4]))
                if old_config.get('output_dir'):
                    old_output=self.base.absolute(old_config['output_dir'])
                    require(not output.is_relative_to(old_output) and not old_output.is_relative_to(output)
                        and not receipt.is_relative_to(old_output),'likelihood output or receipt overlaps immutable old results')
        data=config['data'];digest=self.base.digest
        aggregate_path=self.base.absolute(parents[GENERATION_ID]['argv'][4]);aggregate=self.base.read_json(aggregate_path)
        aggregate_receipt=self.base.read_json(self.base.absolute(parents[GENERATION_ID]['success_receipt']))
        require(config['data_hash']==digest(data) and digest(config['resource_policy'])==digest(aggregate['resource_policy']),
            'likelihood data or unlimited native resource identity changed')
        if not role:
            first=self.base.read_json(self.base.absolute(binding['configs'][ROLE_IDS[0]]['path']))
            require(set(data)=={'roles','initial_protocol','coverage'} and data['roles']==[binding['configs'][key] for key in ROLE_IDS]
                and data['initial_protocol']==first['data']['native']['initial_protocol'] and data['coverage']==first['data']['coverage']
                and config['model_hash']==digest([self.base.read_json(self.base.absolute(binding['configs'][key]['path']))['model_hash'] for key in ROLE_IDS]),
                'likelihood CPU audit must bind exact ordered twelve role configs and coverage')
            return
        native_id=f"v6-full-dev-{job['cell_id']}-{job['role']}";native_job=parents[native_id]
        native_path=self.base.absolute(native_job['argv'][4]);native=self.base.read_json(native_path)
        pins=binding['prerequisites'][native_id];receipt=self.base.read_json(self.base.absolute(pins['path']))
        proof=data.get('native',{})
        require(set(data)=={'native','ordered_sample_ids','ordered_sample_ids_sha256','coverage'}
            and set(proof)=={'config','receipt','artifacts','native_source_repo','aggregate_config','aggregate_receipt',
                'generation_technical_complete','initial_protocol','trained_cell','validation','cap_freeze'}
            and proof['config']=={'path':str(native_path),'sha256':self.base.file_hash(native_path)}
            and proof['receipt']=={k:pins[k] for k in ('path','sha256')} and digest(proof['artifacts'])==digest(receipt['artifacts'])
            and proof['native_source_repo']==native_job['cwd']
            and proof['aggregate_config']=={'path':str(aggregate_path),'sha256':self.base.file_hash(aggregate_path)}
            and proof['aggregate_receipt']=={k:binding['prerequisites'][GENERATION_ID][k] for k in ('path','sha256')}
            and type(proof['generation_technical_complete']) is bool
            and proof['generation_technical_complete'] is receipt.get('technical_complete')
            and all(digest(proof[key])==digest(native['data'][key]) for key in ('initial_protocol','trained_cell','validation','cap_freeze')),
            'likelihood original native role, checkpoint, cap or artifact proof changed')
        require(aggregate_receipt.get('scope')=='VALIDATION_ONLY_FULL_DEV_TERMINAL_AUDIT'
            and aggregate_receipt.get('all_roles_succeeded') is True and aggregate_receipt.get('generation_evaluation_terminal') is True
            and all(aggregate_receipt.get(key) is False for key in ('test_files_read','test_unlocked','family_selection_completed',
                'clinical_quality_certified','full_dev_completed','r2_full_evaluation_completed','campaign_terminal')),
            'actual generation-only aggregate success required; likelihood cannot unlock later science')
        require(all(digest(config.get(key))==digest(native.get(key)) for key in ('cell_id','model_id','task','family','role','seed','model','model_hash',
                'fixed_template_date','precision','resource_policy','gpu_ledger_path','supervisor_lock_paths'))
            and config['split']=='validation' and config['eval_context']==4096 and config['fixed_template_date']=='2026-11-14'
            and config['precision']=='nf4_double_quant_bfloat16' and config['local_files_only'] is True and config['trust_remote_code'] is False
            and config['gpu_ledger_path']==str(self.data_root/'runtime/gpu_budget.json')
            and config['supervisor_lock_paths']==[str(self.physical['worker']),str(self.physical['gpu'])],
            'likelihood native Base/Tuned parity, unlimited ledger or canonical leases changed')
        initial_path=self.base.absolute(proof['initial_protocol']['path'])
        require(self.base.file_hash(initial_path)==proof['initial_protocol']['sha256'],'native initial protocol bytes changed')
        initial=self.base.read_json(initial_path);reference={k:initial['validation_references']['text'][k] for k in ('path','sha256')}
        path=self.base.absolute(reference['path'])
        require(proof['validation']==reference and self.base.file_hash(path)==reference['sha256']
            and path.name=='validation.jsonl' and 'test' not in path.parts and 'tests' not in path.parts,
            'likelihood exact validation-only source bytes required')
        rows=[json.loads(line) for line in path.read_text().splitlines()];ids=[row['sample_id'] for row in rows]
        counts={};supported=0
        for row in rows:
            require(row.get('split')=='validation','likelihood test or nonvalidation rows forbidden')
            if PROCEDURE['supported'].get(row['source_id'])==row['task_type']:
                supported+=1;counts[row['source_id']]=counts.get(row['source_id'],0)+1
                if row['task_type']=='medical_mcq':
                    options=row['input'].get('options')
                    require(isinstance(options,(dict,list)) and len(options)>=2,'native MCQ options required')
                else:require(not row['input'].get('options'),'native context labels cannot override fixed yes/no/maybe')
        coverage={key:PROCEDURE[key] for key in ('text_all_planned','classification_scheduled','classification_source_counts','unsupported_text','unsupported_vqa')}
        require(len(ids)==1516 and len(set(ids))==1516 and supported==637 and counts==PROCEDURE['classification_source_counts']
            and data['ordered_sample_ids']==ids and data['ordered_sample_ids_sha256']==digest(ids) and data['coverage']==coverage,
            'likelihood full planned denominator, ordered native IDs or unsupported coverage changed')

    def _verify(self,*,prepared=True):
        result=super()._verify(prepared=prepared);self._choice_likelihood();return result

    def _manifest(self,value):
        binding,parent,replacement=self._choice_likelihood();actual=self.engine.canonical_hash(value)
        if actual==binding['parent']['manifest_hash']:return super()._manifest(value)
        require(actual==binding['replacement']['manifest_hash'] and actual==self.engine.canonical_hash(replacement),'manifest outside exact reviewed choice_likelihood pair')
        projected=copy.deepcopy(value);projected['jobs']=[j for j in projected['jobs'] if j['job_id'] not in EXECUTABLE_IDS]
        super()._manifest(projected)
        return value

    def call(self,command,*,manifest=None,job_id=None,receipt=None):
        require(command in {'status','tick','refine-choice-likelihood','authorize-recovery'},'choice_likelihood facade supports only native status/tick/exact refinement/same-source recovery')
        if command=='authorize-recovery':
            require(job_id in EXECUTABLE_IDS and manifest is None and receipt is not None,'same-source recovery requires exact choice_likelihood job, no revised manifest, and analysis receipt')
            return self.base.CampaignControl.call(self,command,manifest=None,job_id=job_id,receipt=receipt)
        if command!='refine-choice-likelihood':return self.base.CampaignControl.call(self,command,manifest=manifest,job_id=job_id,receipt=receipt)
        self._verify();self._registered();self._manifest(manifest)
        with self.routed_locks():return self.supervisor.refine_choice_likelihood(manifest)


def make_control(*,bootstrap_operator,full_dev_operator,cap_operator,validation_operator,stage_operator,recovery_operator,base_operator,**kwargs):
    bootstrap=load_bootstrap(bootstrap_operator);full=bootstrap.load_full_dev(full_dev_operator)
    caps=full.load_caps(cap_operator);validation=caps.load_validation(validation_operator)
    stage=validation.load_stage(stage_operator);recovery=stage.load_recovery(recovery_operator);original=recovery.load_base(base_operator)
    class ChoiceLikelihoodControl(ChoiceLikelihoodMixin,bootstrap.BootstrapMixin,full.FullDevMixin,caps.CapRefinementMixin,validation.ValidationMixin,stage.StageMixin,recovery.RecoveryMixin,original.CampaignControl):
        base=original
    return ChoiceLikelihoodControl(**kwargs)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('bootstrap-operator','full-dev-operator','cap-operator','validation-operator','stage-operator','recovery-operator','base-operator','data-root','campaign-root','engine-root',
        'hold-receipt','recovery-binding','stage-binding','validation-binding','cap-binding','full-dev-binding','bootstrap-binding','choice-likelihood-binding'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('recovery-binding-sha256','stage-binding-sha256','validation-binding-sha256','cap-binding-sha256','full-dev-binding-sha256','bootstrap-binding-sha256','choice-likelihood-binding-sha256'):
        parser.add_argument('--'+name,required=True)
    sub=parser.add_subparsers(dest='command',required=True);sub.add_parser('status');sub.add_parser('tick')
    refine=sub.add_parser('refine-choice-likelihood');refine.add_argument('--manifest',type=Path,required=True)
    recover=sub.add_parser('authorize-recovery');recover.add_argument('--job-id',required=True);recover.add_argument('--receipt',type=Path,required=True)
    args=vars(parser.parse_args(argv));command,manifest=args.pop('command'),args.pop('manifest',None)
    job_id,receipt=args.pop('job_id',None),args.pop('receipt',None);control=make_control(**args)
    value=control.base.read_json(manifest) if manifest is not None else None
    print(json.dumps(control.call(command,manifest=value,job_id=job_id,receipt=receipt),ensure_ascii=False,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
