#!/usr/bin/env python3
"""Add one pinned CPU bootstrap job through the existing sole native supervisor."""
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
FULL_DEV_OPERATOR_SHA256 = '3f9e5b1a557aa708f60cf9612f16f3bc48e614d1ccb9f54f6f110fd4ca23ff63'
SCHEMA = 'mediflow.recorded-bootstrap-refinement.v1'
CONFIG_SCHEMA = 'medvlm.v6.bootstrap-config.v1'
SCOPE = 'VALIDATION_ONLY_PAIRED_GROUP_BOOTSTRAP_AUXILIARY'
WORKER_MODULE = 'mediflow_medvlm.campaign_bootstrap'
JOB_ID = 'v6-bootstrap'
PARENT_ID = 'v6-full_dev'
FAMILY_ID = 'v6-family_selection'
CELL_IDS = ['T1','T2','T3','T4','T5','T6','V1','V2','V3','V4']
ROLES = ('base','tuned')
PREREQUISITES = ['v6-first-seed-'+cell for cell in CELL_IDS] + ['v6-first_seed_final'] + [
    f'v6-caps-initial-{cell}-{role}' for cell in CELL_IDS for role in ROLES] + ['v6-caps-decision'] + [
    f'v6-caps-bumped-{cell}-{role}' for cell in CELL_IDS for role in ROLES] + ['v6-validation_caps'] + [
    f'v6-full-dev-{cell}-{role}' for cell in CELL_IDS for role in ROLES] + [PARENT_ID]
EVENT = 'RECORDED_BOOTSTRAP_REFINEMENT'
PROCEDURE = {'seed': 20261023, 'resamples': 2000, 'confidence': 0.95, 'interval': 'percentile', 'quantile': 'linear interpolation at (n-1)*p', 'sampling': 'connected components, stratified by source and stratum-membership profile; draw K components with replacement within each profile', 'pairing': 'same component multiplicities for Base and Tuned and every metric', 'denominator': 'sampled-cluster ratio: sampled weighted numerator / sampled row count; equivalent calibration to original stratum N; raw draw N may vary; retain technical failures', 'empty_strata': 'impossible under stratum-membership profiles; assert positive sampled count', 'undefined_ratios': 'report undefined counts; withhold interval if any replicate is undefined', 'singleton_profiles': 'one-component profiles have degenerate cluster uncertainty', 'text_grouping': 'source-scoped document/question/duplicate IDs, XML/subset-qualified MedQuAD local IDs, source QA IDs, PMID, and MedQuAD source URL', 'vqa_grouping': 'image/duplicate IDs, image SHA256, decoded pixel hash, and source-scoped image path', 'missing_grouping': 'explicit sample-ID singleton fallback; no independence claim', 'uncertainty': 'within-training-seed item uncertainty; not independent-seed variance'}
JOB_FIELDS = {'job_id','run_id','phase','cell_id','kind','seed','dependencies','dependency_policy','implementation_pending',
    'uses_gpu','requires_gpu','gpu_lock_mode','argv','cwd','success_receipt','source_files','source_hash','config_hash','data_hash','model_hash'}


def require(condition,message):
    if not condition:raise ValueError(message)


def load_full_dev(path):
    path=Path(path)
    require(path.is_absolute() and '..' not in path.parts and not any(p.is_symlink() for p in (path,*path.parents)),
        'absolute nonsymlink full-dev operator required')
    raw=path.read_bytes()
    require(hashlib.sha256(raw).hexdigest()==FULL_DEV_OPERATOR_SHA256,'original full-dev operator changed')
    spec=importlib.util.spec_from_loader('_pinned_full_dev_for_bootstrap',loader=None)
    module=importlib.util.module_from_spec(spec);module.__file__=str(path)
    exec(compile(raw,str(path),'exec'),module.__dict__)
    return module


def validate_pair(engine,parent,replacement):
    engine.validate_manifest(parent);engine.validate_manifest(replacement)
    same=lambda a,b:engine.canonical_hash(a)==engine.canonical_hash(b)
    require(same({k:v for k,v in parent.items() if k!='jobs'},{k:v for k,v in replacement.items() if k!='jobs'}),
        'campaign metadata must remain immutable')
    before={j['job_id']:j for j in parent['jobs']};after={j['job_id']:j for j in replacement['jobs']}
    require(len(before)==83 and set(PREREQUISITES+[FAMILY_ID]).issubset(before) and JOB_ID not in before,
        'exact eighty-three-job full-dev parent required')
    order=[j['job_id'] for j in parent['jobs']];position=order.index(PARENT_ID)+1
    order[position:position]=[JOB_ID]
    require([j['job_id'] for j in replacement['jobs']]==order,'exact single bootstrap insertion after full-dev required')
    require(all(same(after[key],job) for key,job in before.items()),'every original job and family dependency must remain identical')
    job=after[JOB_ID];argv=job.get('argv',[])
    require(set(job)==JOB_FIELDS and job.get('run_id')==JOB_ID and job.get('phase')=='bootstrap'
        and job.get('cell_id')=='ALL_V6' and job.get('kind')=='bootstrap'
        and type(job.get('seed')) is int and job['seed']==20261023
        and job.get('implementation_pending') is False and job.get('uses_gpu') is False and job.get('requires_gpu') is False
        and job.get('gpu_lock_mode')=='wrapper' and job.get('dependencies')==[PARENT_ID] and job.get('dependency_policy')=='success'
        and len(argv)==7 and argv[1:4]==['-m',WORKER_MODULE,'--config'] and argv[5:]==['--receipt',job.get('success_receipt')],
        'exact CPU bootstrap declaration and wrapper required')
    return job


def make_supervisor_class(engine,owner,inherited):
    class BootstrapSupervisor(inherited):
        def refine_bootstrap(self,revised):
            with engine.try_lock(self.root/'supervisor.lock') as lock:
                if lock is None:return {'status':'SUPERVISOR_BUSY'}
                binding,parent,replacement=owner._bootstrap()
                require(engine.canonical_hash(revised)==engine.canonical_hash(replacement),'exact pinned bootstrap replacement required')
                current,state=self.load();digest=engine.canonical_hash(replacement)
                require(state.get('queue_owner')=='supervisor','sole Supervisor ownership required')
                if state['manifest_hash']==digest:
                    events=[e for e in state.get('manifest_revisions',[]) if e.get('kind')==EVENT
                        and e.get('manifest_hash')==digest and e.get('bootstrap_binding_sha256')==owner.bootstrap_sha256
                        and e.get('added_jobs')==[JOB_ID] and e.get('original_record_hashes')==binding['parent_record_hashes']]
                    require(len(events)==1 and engine.canonical_hash(current)==digest
                        and set(state['jobs'])==set(binding['parent_record_hashes'])|{JOB_ID}
                        and all(engine.canonical_hash(state['jobs'][key])==sha for key,sha in binding['parent_record_hashes'].items()),
                        'exact immutable bootstrap refinement audit and old records required')
                    return self.status()
                require(engine.canonical_hash(current)==binding['parent']['manifest_hash']
                    and engine.canonical_hash(current)==engine.canonical_hash(parent),'bootstrap refinement parent is not active')
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
                        'worker active; bootstrap refinement is registration only')
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
                            and engine.canonical_hash(record)==pins['state_hash'],'all seventy-four prerequisites must be verified successes')
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
                        'added_jobs':[JOB_ID],'bootstrap_binding':str(owner.bootstrap_path),'bootstrap_binding_sha256':owner.bootstrap_sha256,
                        'original_record_hashes':binding['parent_record_hashes'],'prerequisite_proofs':proofs,'at':engine.now()})
                    state['jobs'][JOB_ID]={'status':'PENDING','attempts':0,'failure_count':0,'recovery_attempts':0,'history':[]}
                    state.update(manifest_hash=digest,manifest_path=str(version),updated_at=engine.now())
                    engine.atomic_json(self.state_path,state)
                return self.status()
    return BootstrapSupervisor


class BootstrapMixin:
    def __init__(self,*,bootstrap_binding,bootstrap_binding_sha256,**kwargs):
        require(isinstance(bootstrap_binding_sha256,str) and re.fullmatch(r'[0-9a-f]{64}',bootstrap_binding_sha256),
            'external bootstrap binding SHA256 required')
        self.bootstrap_path,self.bootstrap_sha256=Path(bootstrap_binding),bootstrap_binding_sha256
        super().__init__(**kwargs)
        self.bootstrap_path=self.base.absolute(self.bootstrap_path)
        self.supervisor.__class__=make_supervisor_class(self.engine,self,type(self.supervisor))

    def _bootstrap(self):
        raw=self.base.read_bytes(self.bootstrap_path)
        require(hashlib.sha256(raw).hexdigest()==self.bootstrap_sha256,'reviewed bootstrap binding changed')
        binding=json.loads(raw)
        require(binding.get('schema_version')==SCHEMA and binding.get('campaign_root')==str(self.root)
            and binding.get('data_root')==str(self.data_root) and binding.get('full_dev_operator_sha256')==FULL_DEV_OPERATOR_SHA256
            and binding.get('full_dev_binding_sha256')==self.full_dev_sha256,'bootstrap namespace or inherited owner binding changed')
        inherited,_,full_dev_parent=self._full_dev()
        require(binding['parent']['manifest_hash']==inherited['replacement']['manifest_hash'],'bootstrap must extend exact full-dev manifest')
        manifests=[]
        for pins in (binding['parent'],binding['replacement']):
            path=self.base.absolute(pins['manifest_path'])
            require(path.is_relative_to(self.root/'manifests') and self.base.file_hash(path)==pins['manifest_sha256'],'bootstrap manifest path or bytes changed')
            value=self.base.read_json(path)
            require(self.engine.canonical_hash(value)==pins['manifest_hash'],'bootstrap manifest identity changed');manifests.append(value)
        require(self.engine.canonical_hash(manifests[0])==self.engine.canonical_hash(full_dev_parent),'bootstrap parent differs from inherited manifest')
        job=validate_pair(self.engine,*manifests);parents={j['job_id']:j for j in manifests[0]['jobs']}
        require(set(binding['parent_record_hashes'])==set(parents)
            and all(isinstance(v,str) and re.fullmatch(r'[0-9a-f]{64}',v) for v in binding['parent_record_hashes'].values())
            and set(binding['prerequisites'])==set(PREREQUISITES) and set(binding['configs'])=={JOB_ID},'exact old records, seventy-four proofs and one config required')
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
        require(bool(archive) and archive==pins['archive_files'] and self.base.digest(archive)==pins['archive_tree_hash'],'complete bootstrap source archive changed')
        sources={k:v for k,v in archive.items() if k.startswith('src/') and k.endswith('.py')}
        require(sources==pins['source_files'] and self.base.digest(sources)==pins['source_tree_hash']
            and 'src/mediflow_medvlm/campaign_bootstrap.py' in sources,'complete bootstrap source changed')
        require(job['cwd']==str(source),'new bootstrap cwd differs from pinned archive');self.engine.verify_source(job)
        reference=binding['configs'][JOB_ID];path=self.base.absolute(job['argv'][4])
        require(path.is_relative_to(self.root/'configs') and str(path)==reference['path']
            and self.base.file_hash(path)==reference['sha256'],'reviewed bootstrap config path or bytes changed')
        config=self.base.read_json(path)
        require(config.get('schema_version')==CONFIG_SCHEMA
            and config.get('config_hash')==self.base.digest({k:v for k,v in config.items() if k!='config_hash'})
            and all(self.base.digest(config.get(k))==self.base.digest(job.get(k)) for k in
                ('job_id','run_id','cell_id','source_files','source_hash','config_hash','data_hash','model_hash','success_receipt')),
            'bootstrap config identity differs from pinned job')
        expected_sources={**parents[PARENT_ID]['source_files'],'src/mediflow_medvlm/campaign_bootstrap.py':sources['src/mediflow_medvlm/campaign_bootstrap.py']}
        require(config['source_files']==expected_sources and all(sources.get(k)==v for k,v in expected_sources.items()),
            'bootstrap must capture unchanged full-dev imports and its complete worker source')
        self._config_policy_bootstrap(config,job,binding,parents)
        return binding,manifests[0],manifests[1]

    def _config_policy_bootstrap(self,config,job,binding,parents):
        # Scientific keysets and immutable native proof structure are checked here;
        # this controller never imports the new worker into the old runtime.
        require(set(config)=={'schema_version','scope','kind','phase','job_id','run_id','cell_id','seed','uses_gpu','device',
                'state_path','output_dir','success_receipt','data','procedure','model_hash','resource_policy','source_files','source_hash','data_hash','config_hash'}
            and config.get('scope')==SCOPE and config.get('kind')=='bootstrap' and config.get('phase')=='bootstrap'
            and config.get('uses_gpu') is False and config.get('device')=='cpu' and type(config.get('seed')) is int and config['seed']==20261023
            and config.get('state_path')==str(self.supervisor.state_path) and self.base.digest(config['procedure'])==self.base.digest(PROCEDURE),
            'bootstrap fixed procedure, keyset, scope or CPU identity changed')
        output,receipt=self.base.absolute(config['output_dir']),self.base.absolute(config['success_receipt'])
        protected=[self.root/name for name in ('operator','snapshots','configs','manifests','receipts','attempts')]
        auxiliary=self.root/'auxiliary/bootstrap'
        require(output!=auxiliary and output.is_relative_to(auxiliary) and receipt==self.root/'receipts'/(JOB_ID+'.json')
            and not any(output.is_relative_to(p) or p.is_relative_to(output) for p in protected)
            and not receipt.is_relative_to(output),'bootstrap output or receipt overlaps protected owner files')
        for old in parents.values():
            require(receipt!=Path(old.get('success_receipt','/')),'bootstrap receipt overlaps original receipt')
            if old.get('argv') and not old.get('implementation_pending'):
                old_config=self.base.read_json(self.base.absolute(old['argv'][4]))
                if old_config.get('output_dir'):
                    old_output=self.base.absolute(old_config['output_dir'])
                    require(not output.is_relative_to(old_output) and not old_output.is_relative_to(output)
                        and not receipt.is_relative_to(old_output),'bootstrap output or receipt overlaps immutable old results')
        data=config['data'];aggregate_job=parents[PARENT_ID]
        aggregate_path=self.base.absolute(aggregate_job['argv'][4]);aggregate=self.base.read_json(aggregate_path)
        aggregate_receipt=self.base.read_json(self.base.absolute(aggregate_job['success_receipt']))
        require(set(data)=={'aggregate_config','aggregate_receipt','native_source_repo','roles','validation','grouping'}
            and config['data_hash']==self.base.digest(data)
            and data['aggregate_config']=={'path':str(aggregate_path),'sha256':self.base.file_hash(aggregate_path)}
            and data['aggregate_receipt']=={k:binding['prerequisites'][PARENT_ID][k] for k in ('path','sha256')}
            and data['native_source_repo']==aggregate_job['cwd']
            and config['model_hash']==aggregate['model_hash'] and self.base.digest(config['resource_policy'])==self.base.digest(aggregate['resource_policy']),
            'bootstrap original aggregate, model, resource or data identity changed')
        require(aggregate_receipt.get('scope')=='VALIDATION_ONLY_FULL_DEV_TERMINAL_AUDIT'
            and aggregate_receipt.get('generation_evaluation_terminal') is True and aggregate_receipt.get('full_dev_stage_terminal') is True
            and aggregate_receipt.get('all_roles_succeeded') is True
            and all(aggregate_receipt.get(key) is False for key in ('test_files_read','test_unlocked','family_selection_completed',
                'clinical_quality_certified','full_dev_completed','r2_full_evaluation_completed','campaign_terminal')),
            'actual generation-only aggregate receipt required; bootstrap cannot unlock later science')
        role_ids=[f'v6-full-dev-{cell}-{role}' for cell in CELL_IDS for role in ROLES]
        require(isinstance(data['roles'],list) and [r.get('job_id') for r in data['roles']]==role_ids,'exact ordered twenty native roles required')
        for item,key,reference in zip(data['roles'],role_ids,aggregate['data']['roles']):
            pins=binding['prerequisites'][key];old=parents[key]
            old_receipt=self.base.read_json(self.base.absolute(pins['path']))
            require(set(item)=={'job_id','config','receipt','artifacts'} and item['config']==reference
                and item['config']['path']==old['argv'][4] and self.base.file_hash(self.base.absolute(item['config']['path']))==item['config']['sha256']
                and item['receipt']=={k:pins[k] for k in ('path','sha256')}
                and self.base.digest(item['artifacts'])==self.base.digest(old_receipt['artifacts']),
                'bootstrap role config, native receipt or complete artifact proof changed')
        original=self.base.read_json(self.base.absolute(aggregate['data']['initial_protocol']['path']))
        expected_validation={axis:{k:original['validation_references'][axis][k] for k in ('path','sha256')} for axis in ('text','vqa')}
        require(data['validation']==expected_validation and set(data['grouping'])=={'text','vqa'},'bootstrap exact frozen validation references and grouping required')
        grouping_keys={'axis','scheduled','group_count','stratum_counts','group_size_counts','unavailable_reason_counts',
            'singleton_fallback_count','source_evidence_sha256','components_sha256','profiles','patient_or_study_independence_certified'}
        for axis,count in (('text',1516),('vqa',512)):
            ref=data['validation'][axis];path=self.base.absolute(ref['path']);group=data['grouping'][axis]
            require(self.base.file_hash(path)==ref['sha256'] and path.name!='test.jsonl'
                and 'test' not in path.parts and 'tests' not in path.parts,'bootstrap test routes or changed validation bytes forbidden')
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            require(len(rows)==count and len({r['sample_id'] for r in rows})==count
                and all(r.get('split','validation')=='validation' for r in rows),'bootstrap full validation populations required')
            require(set(group)==grouping_keys and group['axis']==axis and type(group['scheduled']) is int and group['scheduled']==count
                and type(group['group_count']) is int and 1<=group['group_count']<=count
                and group['patient_or_study_independence_certified'] is False
                and all(isinstance(group[k],str) and re.fullmatch(r'[0-9a-f]{64}',group[k]) for k in ('source_evidence_sha256','components_sha256'))
                and type(group['singleton_fallback_count']) is int and 0<=group['singleton_fallback_count']<=count
                and isinstance(group['profiles'],list) and all(set(p)=={'sources','strata','group_count'}
                    and isinstance(p['sources'],list) and bool(p['sources']) and isinstance(p['strata'],list) and bool(p['strata'])
                    and type(p['group_count']) is int and p['group_count']>0 for p in group['profiles'])
                and sum(p['group_count'] for p in group['profiles'])==group['group_count']
                and isinstance(group['stratum_counts'],dict) and all(type(n) is int and n>0 for n in group['stratum_counts'].values())
                and sum(group['stratum_counts'].values())==count
                and isinstance(group['group_size_counts'],dict) and all(str(k).isdigit() and int(k)>0 and type(n) is int and n>0 for k,n in group['group_size_counts'].items())
                and sum(int(k)*n for k,n in group['group_size_counts'].items())==count
                and sum(group['group_size_counts'].values())==group['group_count']
                and isinstance(group['unavailable_reason_counts'],dict) and all(type(n) is int and n>=0 for n in group['unavailable_reason_counts'].values()),
                'bootstrap grouping metadata or full scheduled denominators changed')

    def _verify(self,*,prepared=True):
        result=super()._verify(prepared=prepared);self._bootstrap();return result

    def _manifest(self,value):
        binding,parent,replacement=self._bootstrap();actual=self.engine.canonical_hash(value)
        if actual==binding['parent']['manifest_hash']:return super()._manifest(value)
        require(actual==binding['replacement']['manifest_hash'] and actual==self.engine.canonical_hash(replacement),'manifest outside exact reviewed bootstrap pair')
        projected=copy.deepcopy(value);projected['jobs']=[j for j in projected['jobs'] if j['job_id']!=JOB_ID]
        super()._manifest(projected)
        return value

    def call(self,command,*,manifest=None,job_id=None,receipt=None):
        require(command in {'status','tick','refine-bootstrap','authorize-recovery'},'bootstrap facade supports only native status/tick/exact refinement/same-source recovery')
        if command=='authorize-recovery':
            require(job_id==JOB_ID and manifest is None and receipt is not None,'same-source recovery requires exact bootstrap job, no revised manifest, and analysis receipt')
            return self.base.CampaignControl.call(self,command,manifest=None,job_id=job_id,receipt=receipt)
        if command!='refine-bootstrap':return self.base.CampaignControl.call(self,command,manifest=manifest,job_id=job_id,receipt=receipt)
        self._verify();self._registered();self._manifest(manifest)
        with self.routed_locks():return self.supervisor.refine_bootstrap(manifest)


def make_control(*,full_dev_operator,cap_operator,validation_operator,stage_operator,recovery_operator,base_operator,**kwargs):
    full=load_full_dev(full_dev_operator);caps=full.load_caps(cap_operator);validation=caps.load_validation(validation_operator)
    stage=validation.load_stage(stage_operator);recovery=stage.load_recovery(recovery_operator);original=recovery.load_base(base_operator)
    class BootstrapControl(BootstrapMixin,full.FullDevMixin,caps.CapRefinementMixin,validation.ValidationMixin,stage.StageMixin,recovery.RecoveryMixin,original.CampaignControl):
        base=original
    return BootstrapControl(**kwargs)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('full-dev-operator','cap-operator','validation-operator','stage-operator','recovery-operator','base-operator','data-root','campaign-root','engine-root',
        'hold-receipt','recovery-binding','stage-binding','validation-binding','cap-binding','full-dev-binding','bootstrap-binding'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('recovery-binding-sha256','stage-binding-sha256','validation-binding-sha256','cap-binding-sha256','full-dev-binding-sha256','bootstrap-binding-sha256'):
        parser.add_argument('--'+name,required=True)
    sub=parser.add_subparsers(dest='command',required=True);sub.add_parser('status');sub.add_parser('tick')
    refine=sub.add_parser('refine-bootstrap');refine.add_argument('--manifest',type=Path,required=True)
    recover=sub.add_parser('authorize-recovery');recover.add_argument('--job-id',required=True);recover.add_argument('--receipt',type=Path,required=True)
    args=vars(parser.parse_args(argv));command,manifest=args.pop('command'),args.pop('manifest',None)
    job_id,receipt=args.pop('job_id',None),args.pop('receipt',None);control=make_control(**args)
    value=control.base.read_json(manifest) if manifest is not None else None
    print(json.dumps(control.call(command,manifest=value,job_id=job_id,receipt=receipt),ensure_ascii=False,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
