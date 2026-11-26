#!/usr/bin/env python3
"""Versioned source repair for likelihood jobs under the frozen sole Supervisor.

The user's unlimited retry policy removes count/Astra prerequisites, not source
repair evidence. Every new likelihood failure holds its source cohort until an
explicit, CPU-validated revision. Reconciliation never dispatches a worker.
"""
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
PRIOR_OPERATOR_SHA256 = '26d14b00f71a8970c86368f6a085aa8fe12feb89f2089d67d3b6830fdc0ee762'
SCHEMA = 'mediflow.recorded-choice-likelihood-source-recovery.v1'
REPAIR_SCHEMA = 'mediflow.choice-likelihood-source-fix.v1'
POLICY_SCHEMA = 'mediflow.owner-user-policy-change.v1'
EVENT = 'RECORDED_CHOICE_LIKELIHOOD_SOURCE_REPAIR'
ROLE_IDS = [f'v6-choice-likelihood-T{i}-{role}' for i in range(1,7) for role in ('base','tuned')]
EXECUTABLE_IDS = [*ROLE_IDS, 'v6-choice-likelihood']
WORKER_SOURCE = 'src/mediflow_medvlm/campaign_choice_likelihood.py'
POLICY_USER_THREAD = '01a1dd9c-d567-7c61-8c0f-ec859ba1497a'
POLICY_USER_MESSAGE = '01a2163e-f35e-7a12-8bac-badc4de0e28e'


def require(condition, message):
    if not condition: raise ValueError(message)


def absolute(path):
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts
            and not any(p.is_symlink() for p in (path,*path.parents)), 'absolute nonsymlink path required')
    return path


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def read(path): return json.loads(absolute(path).read_bytes())
def file_hash(path): return hashlib.sha256(absolute(path).read_bytes()).hexdigest()
def reference(path): return {'path':str(absolute(path)), 'sha256':file_hash(path)}


def load_prior(path):
    path=absolute(path); raw=path.read_bytes()
    require(hashlib.sha256(raw).hexdigest()==PRIOR_OPERATOR_SHA256,'original likelihood operator changed')
    spec=importlib.util.spec_from_loader('_pinned_choice_likelihood_for_source_recovery',loader=None)
    module=importlib.util.module_from_spec(spec); module.__file__=str(path)
    exec(compile(raw,str(path),'exec'),module.__dict__)
    return module


def archive_pins(root):
    root=absolute(root)
    require(re.fullmatch(r'[0-9a-f]{40}',root.name),'immutable commit archive required')
    files={}
    for path in sorted(root.rglob('*')):
        require(not path.is_symlink(),'archive symlinks forbidden')
        if path.is_file(): files[path.relative_to(root).as_posix()]=file_hash(path)
    sources={k:v for k,v in files.items() if k.startswith('src/') and k.endswith('.py')}
    return {'root':str(root),'commit':root.name,'archive_files':files,'archive_tree_hash':digest(files),
            'source_files':sources,'source_tree_hash':digest(sources)}


def prepare_choice_binding(original_choice_binding_path,replacement_manifest_path,new_engine_root,new_config_paths_ordered13):
    """Return a revised immutable prior-facade binding; never write files."""
    value=copy.deepcopy(read(original_choice_binding_path)); manifest=read(replacement_manifest_path)
    refs=[reference(path) for path in new_config_paths_ordered13]
    require([read(ref['path']).get('job_id') for ref in refs]==EXECUTABLE_IDS,'exact ordered thirteen repair configs required')
    value['replacement']={'manifest_path':str(absolute(replacement_manifest_path)),
        'manifest_sha256':file_hash(replacement_manifest_path),'manifest_hash':digest(manifest)}
    value['engine']=archive_pins(new_engine_root)
    value['configs']=dict(zip(EXECUTABLE_IDS,refs))
    return value


def prepare_binding(parent_choice_binding_path,replacement_choice_binding_path,user_policy_receipt_path,
                    repair_receipt_path,state_path,*,original_choice_binding_path=None):
    """Return external source-repair pins after a separate CPU reconciliation.

    A later manual repair, while this entire cohort remains incomplete, can use
    the preceding replacement choice binding as parent. Partial-success repair
    needs a separate reviewed scope: this facade never rebinds successful jobs.
    """
    parent,new=read(parent_choice_binding_path),read(replacement_choice_binding_path); state=read(state_path)
    original=original_choice_binding_path or parent_choice_binding_path
    require(parent['campaign_root']==new['campaign_root'] and parent['data_root']==new['data_root'],'repair namespace mismatch')
    return {'schema_version':SCHEMA,'campaign_root':parent['campaign_root'],'data_root':parent['data_root'],
        'prior_operator_sha256':PRIOR_OPERATOR_SHA256,'original_choice_binding':reference(original),
        'parent_choice_binding':reference(parent_choice_binding_path),'replacement_choice_binding':reference(replacement_choice_binding_path),
        'user_policy_receipt':reference(user_policy_receipt_path),'repair_receipt':reference(repair_receipt_path),
        'parent_state_sha256':file_hash(state_path),
        'parent_record_hashes':{key:digest(record) for key,record in state['jobs'].items()},
        'parent_manifest_hash':parent['replacement']['manifest_hash'],'replacement_manifest_hash':new['replacement']['manifest_hash']}


def validate_pair(engine,before,after):
    engine.validate_manifest(before); engine.validate_manifest(after)
    require(digest({k:v for k,v in before.items() if k!='jobs'})==digest({k:v for k,v in after.items() if k!='jobs'})
            and len(before['jobs'])==len(after['jobs'])==97
            and [j['job_id'] for j in before['jobs']]==[j['job_id'] for j in after['jobs']],
            'repair cannot change metadata, job membership or order')
    old={j['job_id']:j for j in before['jobs']}; new={j['job_id']:j for j in after['jobs']}
    require(set(EXECUTABLE_IDS).issubset(old),'exact thirteen likelihood repair jobs required')
    for key,job in old.items():
        if key not in EXECUTABLE_IDS:
            require(digest(job)==digest(new[key]),'all eighty-four original jobs must remain unchanged'); continue
        fixed=new[key]; aggregate=key==EXECUTABLE_IDS[-1]
        allowed={'cwd','source_files','source_hash','config_hash','argv'}|({'data_hash'} if aggregate else set())
        require(digest({k:v for k,v in job.items() if k not in allowed})==digest({k:v for k,v in fixed.items() if k not in allowed}),
                'repair scientific job identity changed')
        require(job['argv'][:4]+job['argv'][5:]==fixed['argv'][:4]+fixed['argv'][5:]
                and job['argv'][4]!=fixed['argv'][4] and job['cwd']!=fixed['cwd'], 'new immutable config and archive required; nonconfig argv immutable')
        require(set(job['source_files'])==set(fixed['source_files']) and fixed['source_files'].get(WORKER_SOURCE)!=job['source_files'].get(WORKER_SOURCE)
                and all(fixed['source_files'][name]==sha for name,sha in job['source_files'].items() if name!=WORKER_SOURCE),
                'only the likelihood worker source may change')
        prior,config=read(job['argv'][4]),read(fixed['argv'][4])
        identity={'source_files','source_hash','config_hash'}|({'data','data_hash'} if aggregate else set())
        require(digest({k:v for k,v in prior.items() if k not in identity})==digest({k:v for k,v in config.items() if k not in identity}),
                'repair scientific config identity changed')
        if aggregate:
            require(digest({k:v for k,v in prior['data'].items() if k!='roles'})==digest({k:v for k,v in config['data'].items() if k!='roles'})
                    and config['data']['roles']==[reference(new[role]['argv'][4]) for role in ROLE_IDS],
                    'aggregate may change only ordered role config references')
        require(config['config_hash']==digest({k:v for k,v in config.items() if k!='config_hash'})
                and config['data_hash']==digest(config['data']) and config['source_hash']==digest(config['source_files'])
                and all(config[k]==fixed[k] for k in ('source_files','source_hash','config_hash','data_hash','model_hash')),
                'repair config hashes changed')
        engine.verify_source(job); engine.verify_source(fixed)
    return old,new


def make_supervisor_class(engine,owner,inherited):
    class ChoiceLikelihoodRecoverySupervisor(inherited):
        def load(self):
            manifest,state=super().load()
            self._ll_manifest,self._ll_state=manifest,state
            return manifest,state

        def _failure(self,record,reason,exit_code=None):
            super()._failure(record,reason,exit_code)
            jobs={j['job_id']:j for j in getattr(self,'_ll_manifest',{}).get('jobs',[])}
            key=next((key for key,item in getattr(self,'_ll_state',{}).get('jobs',{}).items() if item is record),None)
            if key not in EXECUTABLE_IDS: return
            policy=owner._retry_policy(); policy_sha=owner.retry_policy_sha256
            # Keep the native failure increment, error signature and history.
            # Holding all pending roles prevents native tick from dispatching
            # another role of the same known-bad source in this same invocation.
            for target in EXECUTABLE_IDS:
                item=self._ll_state['jobs'][target]
                if target==key or item.get('status')=='PENDING':
                    item.update(status='ANALYSIS_REQUIRED',source_repair_required=True,
                        failed_source_hash=jobs[target]['source_hash'],retry_policy_receipt_sha256=policy_sha)
                    item.setdefault('history',[]).append({'status':'SOURCE_REPAIR_REQUIRED','failed_job_id':key,
                        'failed_source_hash':jobs[target]['source_hash'],'retry_limit':policy['retry_limit'],
                        'mandatory_astra_before_retry':False,'user_policy_receipt_sha256':policy_sha,'at':engine.now()})

        def tick(self):
            owner._retry_policy()
            _,state=self.load()
            if any(state['jobs'][key].get('source_repair_required') or state['jobs'][key].get('status') in {'ANALYSIS_REQUIRED','FAILED'} for key in EXECUTABLE_IDS):
                return self.status()
            return super().tick()

        def reconcile_choice_likelihood(self):
            with engine.try_lock(self.root/'supervisor.lock') as lock:
                if lock is None: return {'status':'SUPERVISOR_BUSY'}
                manifest,state=self.load(); owner._check_original_records(state)
                require(state.get('queue_owner')=='supervisor','sole Supervisor owner required')
                with ExitStack() as leases:
                    owner._quiescent(state,allow_dead_running=True)
                    for path in (self.root/'worker.lock',self.data_root/'runtime/medvlm_gpu.lock'):
                        require(leases.enter_context(engine.try_lock(path)) is not None,'worker/GPU lease busy')
                    owner._quiescent(state,allow_dead_running=True)
                    changed=[]
                    for job in manifest['jobs']:
                        record=state['jobs'][job['job_id']]
                        if record.get('status') not in {'STARTING','RUNNING'}: continue
                        require(job['job_id'] in EXECUTABLE_IDS,'reconciliation cannot alter an original job')
                        attempt=owner.base.absolute(record['attempt_dir']); exited=engine.read_json(attempt/'exit.json')
                        if exited and exited.get('launch_token')!=record.get('launch_token'):
                            self._failure(record,'exit receipt launch identity mismatch')
                        elif exited and exited.get('exit_code')==0:
                            try:
                                record['verification']=engine.verify_success(job); record.update(status='SUCCEEDED',exit_code=0)
                                record.setdefault('history',[]).append({'attempt':record['attempts'],'status':'SUCCEEDED','at':engine.now()})
                            except (ValueError,OSError,KeyError,TypeError) as exc: self._failure(record,str(exc),0)
                        else: self._failure(record,(exited or {}).get('error','worker exited without success receipt'),(exited or {}).get('exit_code'))
                        changed.append(job['job_id'])
                    if changed:
                        state.setdefault('manifest_revisions',[]).append({'kind':'CHOICE_LIKELIHOOD_RECONCILE_ONLY',
                            'jobs':changed,'dispatch_performed':False,'user_policy_receipt_sha256':owner.retry_policy_sha256,'at':engine.now()})
                        state['updated_at']=engine.now(); owner._check_original_records(state); engine.atomic_json(self.state_path,state)
                return self.status()

        def repair_choice_likelihood(self,revised):
            with engine.try_lock(self.root/'supervisor.lock') as lock:
                if lock is None: return {'status':'SUPERVISOR_BUSY'}
                binding,parent,replacement=owner._likelihood_recovery()
                require(digest(revised)==digest(replacement),'exact pinned source repair manifest required')
                current,state=self.load(); owner._check_original_records(state)
                require(state.get('queue_owner')=='supervisor','sole Supervisor owner required')
                target=digest(replacement)
                events=[event for event in state.get('manifest_revisions',[]) if event.get('kind')==EVENT
                    and event.get('binding_sha256')==owner.likelihood_recovery_sha256 and event.get('manifest_hash')==target]
                if state['manifest_hash']==target:
                    require(len(events)==1 and digest(current)==target,'exact source repair audit required')
                    return self.status()
                require(digest(current)==digest(parent)==binding['parent_manifest_hash']
                        and owner.base.file_hash(self.state_path)==binding['parent_state_sha256']
                        and {k:digest(v) for k,v in state['jobs'].items()}==binding['parent_record_hashes'],
                        'exact reconciled parent state required')
                owner._quiescent(state,allow_dead_running=False)
                with ExitStack() as leases:
                    for path in (self.root/'worker.lock',self.data_root/'runtime/medvlm_gpu.lock'):
                        require(leases.enter_context(engine.try_lock(path)) is not None,'worker/GPU lease busy')
                    owner._quiescent(state,allow_dead_running=False); owner._verify()
                    require(owner.base.file_hash(self.state_path)==binding['parent_state_sha256'],'parent changed during repair audit')
                    requeued=[]; old_jobs={j['job_id']:j for j in parent['jobs']}
                    for key in EXECUTABLE_IDS:
                        record=state['jobs'][key]
                        require(record.get('status')!='SUCCEEDED','successful likelihood outputs cannot be source-rebound')
                        require(all(type(record.get(k)) is int and record[k]>=0 for k in ('attempts','failure_count','recovery_attempts')),
                                'valid preserved retry counters required')
                        if record.get('attempts',0):
                            require(record['status'] in {'ANALYSIS_REQUIRED','FAILED','BLOCKED_PREREQUISITE'},'attempted role must be held for repair')
                        require(not Path(next(j for j in replacement['jobs'] if j['job_id']==key)['success_receipt']).exists(),
                                'existing likelihood success receipt cannot be overwritten')
                        record.update(status='PENDING',source_repair_required=False)
                        record.setdefault('history',[]).append({'status':'SOURCE_REPAIR_AUTHORIZED','binding_sha256':owner.likelihood_recovery_sha256,
                            'original_source_hash':old_jobs[key]['source_hash'],'replacement_source_hash':next(j for j in replacement['jobs'] if j['job_id']==key)['source_hash'],
                            'user_policy_receipt_sha256':owner.retry_policy_sha256,'retry_limit':None,'mandatory_astra_before_retry':False,'at':engine.now()})
                        requeued.append(key)
                    version=self.root/'manifests'/(target+'.json')
                    if version.exists(): require(digest(owner.base.read_json(version))==target,'immutable repair manifest changed')
                    else: engine.atomic_json(version,replacement)
                    state.setdefault('manifest_revisions',[]).append({'kind':EVENT,'previous_hash':state['manifest_hash'],
                        'manifest_hash':target,'manifest_path':str(version),'binding':str(owner.likelihood_recovery_path),
                        'binding_sha256':owner.likelihood_recovery_sha256,'user_policy_receipt_sha256':owner.retry_policy_sha256,
                        'original_record_hashes':owner._original_choice_binding()['parent_record_hashes'],
                        'parent_record_hashes':binding['parent_record_hashes'],'requeued_jobs':requeued,'at':engine.now()})
                    state.update(manifest_hash=target,manifest_path=str(version),updated_at=engine.now())
                    owner._check_original_records(state); engine.atomic_json(self.state_path,state)
                return self.status()
    return ChoiceLikelihoodRecoverySupervisor


class ChoiceLikelihoodRecoveryMixin:
    def __init__(self,*,retry_policy_receipt,retry_policy_sha256,likelihood_recovery_binding=None,likelihood_recovery_binding_sha256=None,**kwargs):
        require(re.fullmatch(r'[0-9a-f]{64}',retry_policy_sha256 or ''),'externally pinned user policy SHA required')
        require((likelihood_recovery_binding is None)==(likelihood_recovery_binding_sha256 is None),'recovery binding/path pins supplied together')
        if likelihood_recovery_binding_sha256 is not None:
            require(re.fullmatch(r'[0-9a-f]{64}',likelihood_recovery_binding_sha256),'externally pinned source repair SHA required')
        self.retry_policy_path,self.retry_policy_sha256=absolute(retry_policy_receipt),retry_policy_sha256
        self.likelihood_recovery_path=absolute(likelihood_recovery_binding) if likelihood_recovery_binding is not None else None
        self.likelihood_recovery_sha256=likelihood_recovery_binding_sha256
        super().__init__(**kwargs)
        self.supervisor.__class__=make_supervisor_class(self.engine,self,type(self.supervisor))

    def _retry_policy(self):
        require(self.base.file_hash(self.retry_policy_path)==self.retry_policy_sha256,'user retry policy bytes changed')
        policy=self.base.read_json(self.retry_policy_path)
        require(policy.get('schema_version')==POLICY_SCHEMA and policy.get('campaign_root')==str(self.root)
                and policy.get('campaign_id')==self.root.name and policy.get('source_thread_id')==POLICY_USER_THREAD
                and policy.get('source_user_message_id')==POLICY_USER_MESSAGE and policy.get('exact_user_text')=='재시도 제한 해제'
                and policy.get('retry_limit') is None and policy.get('mandatory_astra_before_retry') is False
                and all(policy.get(key) is True for key in ('preserve_attempts_failures_history_gpu_ledger',
                    'require_source_fix_and_cpu_tests_before_same_cause_retry','sole_canonical_supervisor_owner','single_hourly_automation_preserved')),
                'exact direct-user unlimited retry policy and source-repair guards required')
        return policy

    def _original_choice_binding(self):
        return super()._choice_likelihood()[0]

    def _check_original_records(self,state):
        pins=self._original_choice_binding()['parent_record_hashes']
        require(set(state['jobs'])==set(pins)|set(EXECUTABLE_IDS)
                and all(digest(state['jobs'][key])==sha for key,sha in pins.items()),'all eighty-four original state records must remain unchanged')

    def _quiescent(self,state,*,allow_dead_running):
        for key,record in state['jobs'].items():
            health={}
            if record.get('attempt_dir'):
                attempt=self.base.absolute(record['attempt_dir'])
                require(attempt.is_relative_to(self.root/'attempts'),'attempt escaped campaign')
                health=self.engine.read_json(attempt/'heartbeat.json',{}) or {}
            require(not any(self.engine.same_process(item) for item in (record.get('worker_identity'),health.get('worker_identity'),health.get('child_identity'))),
                    'live worker or child; refuse CPU reconciliation/source repair')
            if not allow_dead_running: require(record.get('status') not in {'STARTING','RUNNING'},'reconcile dead running workers first')

    def _likelihood_recovery(self):
        require(self.likelihood_recovery_path is not None,'explicit source-repair binding required')
        require(self.base.file_hash(self.likelihood_recovery_path)==self.likelihood_recovery_sha256,'source repair binding bytes changed')
        binding=self.base.read_json(self.likelihood_recovery_path); original=self._original_choice_binding()
        require(binding.get('schema_version')==SCHEMA and binding.get('campaign_root')==str(self.root)
                and binding.get('data_root')==str(self.data_root) and binding.get('prior_operator_sha256')==PRIOR_OPERATOR_SHA256
                and binding.get('original_choice_binding')=={'path':str(self.choice_likelihood_path),'sha256':self.choice_likelihood_sha256}
                and binding.get('user_policy_receipt')=={'path':str(self.retry_policy_path),'sha256':self.retry_policy_sha256},
                'source repair namespace/original binding/user policy changed')
        self._retry_policy(); values=[]
        for key in ('parent_choice_binding','replacement_choice_binding'):
            pins=binding[key]; path=self.base.absolute(pins['path'])
            require(reference(path)==pins,'pinned choice repair binding changed')
            proxy=copy.copy(self); proxy.choice_likelihood_path,proxy.choice_likelihood_sha256=path,pins['sha256']
            value=super(ChoiceLikelihoodRecoveryMixin,proxy)._choice_likelihood()
            require(value[0]['parent']==original['parent'] and value[0]['parent_record_hashes']==original['parent_record_hashes']
                    and value[0]['prerequisites']==original['prerequisites'],'source repair changed original scientific proofs')
            values.append(value)
        parent,new=values[0][2],values[1][2]; old_jobs,new_jobs=validate_pair(self.engine,parent,new)
        require(binding['parent_manifest_hash']==digest(parent) and binding['replacement_manifest_hash']==digest(new)
                and set(binding['parent_record_hashes'])==set(old_jobs),'source repair exact manifest/state pins required')
        pins=binding['repair_receipt']; require(reference(pins['path'])==pins,'CPU source repair evidence changed')
        repair=self.base.read_json(pins['path'])
        require(repair.get('schema_version')==REPAIR_SCHEMA and repair.get('campaign_root')==str(self.root)
                and repair.get('status')=='CPU_VALIDATED' and repair.get('cpu_only') is True and repair.get('cpu_tests_passed') is True
                and repair.get('job_ids')==EXECUTABLE_IDS and repair.get('parent_source_hash')==old_jobs[ROLE_IDS[0]]['source_hash']
                and repair.get('replacement_source_hash')==new_jobs[ROLE_IDS[0]]['source_hash']
                and isinstance(repair.get('reason'),str) and bool(repair['reason'].strip()) and bool(repair.get('evidence')),
                'source repair requires exact changed-source CPU test evidence')
        for pins in repair['evidence']: require(reference(pins['path'])==pins,'CPU repair test evidence bytes changed')
        return binding,parent,new

    def _verify(self,*,prepared=True):
        value=super()._verify(prepared=prepared); self._retry_policy()
        if self.likelihood_recovery_path is not None: self._likelihood_recovery()
        return value

    def _manifest(self,value):
        if self.likelihood_recovery_path is None: return super()._manifest(value)
        _,parent,replacement=self._likelihood_recovery()
        if digest(value) in {digest(parent),digest(replacement)}:
            # The complete parent/replacement bindings were checked above.
            # Reuse inherited namespace checks on its immutable original view.
            super()._manifest(super()._choice_likelihood()[2]); return value
        return super()._manifest(value)

    def _registered(self):
        super()._registered(); state=self.base.read_json(self.supervisor.state_path); self._check_original_records(state)
        if self.likelihood_recovery_path is not None:
            binding,_,replacement=self._likelihood_recovery()
            if state['manifest_hash']==digest(replacement):
                events=[event for event in state.get('manifest_revisions',[]) if event.get('kind')==EVENT
                    and event.get('binding_sha256')==self.likelihood_recovery_sha256 and event.get('manifest_hash')==digest(replacement)
                    and event.get('previous_hash')==binding['parent_manifest_hash']
                    and event.get('parent_record_hashes')==binding['parent_record_hashes']
                    and event.get('original_record_hashes')==self._original_choice_binding()['parent_record_hashes']
                    and event.get('requeued_jobs')==EXECUTABLE_IDS and event.get('user_policy_receipt_sha256')==self.retry_policy_sha256]
                require(len(events)==1,'exact registered source repair audit required')

    def call(self,command,*,manifest=None,job_id=None,receipt=None):
        require(command in {'status','tick','reconcile-only','repair-choice-likelihood'},'only status/tick/CPU reconciliation/exact source repair supported')
        require(job_id is None and receipt is None,'per-job recovery and Astra prerequisites are superseded by exact user-policy source repair')
        if command=='tick':
            _,_,replacement=self._likelihood_recovery()
            require(self.supervisor.load()[1]['manifest_hash']==digest(replacement),'register evidenced source repair before dispatch')
        if command in {'status','tick'}: return self.base.CampaignControl.call(self,command,manifest=manifest)
        self._verify(); self._registered()
        require(manifest is None if command=='reconcile-only' else manifest is not None,'exact command manifest routing required')
        if manifest is not None: self._manifest(manifest)
        with self.routed_locks():
            return self.supervisor.reconcile_choice_likelihood() if command=='reconcile-only' else self.supervisor.repair_choice_likelihood(manifest)


def build_control(*,choice_likelihood_operator,retry_policy_receipt,retry_policy_sha256,
                  likelihood_recovery_binding=None,likelihood_recovery_binding_sha256=None,**kwargs):
    prior=load_prior(choice_likelihood_operator); control=prior.make_control(**kwargs)
    # The prior constructor already built its exact native subclass. Add only
    # this versioned layer; no native module or frozen source is replaced.
    class RecoveryControl(ChoiceLikelihoodRecoveryMixin,type(control)): pass
    control.__class__=RecoveryControl
    require(re.fullmatch(r'[0-9a-f]{64}',retry_policy_sha256 or ''),'externally pinned user policy SHA required')
    require((likelihood_recovery_binding is None)==(likelihood_recovery_binding_sha256 is None),'recovery binding/path pins supplied together')
    if likelihood_recovery_binding_sha256 is not None:
        require(re.fullmatch(r'[0-9a-f]{64}',likelihood_recovery_binding_sha256),'externally pinned source repair SHA required')
    control.retry_policy_path,control.retry_policy_sha256=absolute(retry_policy_receipt),retry_policy_sha256
    control.likelihood_recovery_path=absolute(likelihood_recovery_binding) if likelihood_recovery_binding is not None else None
    control.likelihood_recovery_sha256=likelihood_recovery_binding_sha256
    control.supervisor.__class__=make_supervisor_class(control.engine,control,type(control.supervisor))
    return control


make_control=build_control


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('choice-likelihood-operator','bootstrap-operator','full-dev-operator','cap-operator','validation-operator','stage-operator',
                 'recovery-operator','base-operator','data-root','campaign-root','engine-root','hold-receipt','recovery-binding','stage-binding',
                 'validation-binding','cap-binding','full-dev-binding','bootstrap-binding','choice-likelihood-binding','retry-policy-receipt'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('recovery-binding-sha256','stage-binding-sha256','validation-binding-sha256','cap-binding-sha256',
                 'full-dev-binding-sha256','bootstrap-binding-sha256','choice-likelihood-binding-sha256','retry-policy-sha256'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--likelihood-recovery-binding',type=Path); parser.add_argument('--likelihood-recovery-binding-sha256')
    sub=parser.add_subparsers(dest='command',required=True)
    for command in ('status','tick','reconcile-only'): sub.add_parser(command)
    repair=sub.add_parser('repair-choice-likelihood'); repair.add_argument('--manifest',type=Path,required=True)
    args=vars(parser.parse_args(argv)); command=args.pop('command'); path=args.pop('manifest',None)
    control=build_control(**args)
    print(json.dumps(control.call(command,manifest=read(path) if path is not None else None),ensure_ascii=False,indent=2)); return 0


if __name__=='__main__': raise SystemExit(main())
