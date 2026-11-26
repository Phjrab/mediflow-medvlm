#!/usr/bin/env python3
"""Reviewed V6-only refinements over the existing canonical owner and V7 fence.

Legacy audits see the original immutable job declarations/never-started anchor
records. Native execution always sees the real extended manifest and records.
This explicit compatibility view never writes a fabricated state or receipt.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
GUARD_SHA = '9d6c6c8d596e0cef05c0b067469788f893d8d4c17c72e97a7ffb0bf39817ff8c'
SCHEMA = 'mediflow.recorded-v6-completion-refinement.v1'
EVENT = 'RECORDED_V6_COMPLETION_REFINEMENT'
REPAIR_SCHEMA = 'mediflow.recorded-v6-completion-source-repair.v1'
REPAIR_EVENT = 'RECORDED_V6_COMPLETION_SOURCE_REPAIR'
REPAIR_REVIEW_SCHEMA = 'mediflow.v6-completion-source-repair-review.v1'
REPAIR_TRAIN_IDS = [f'v6-additional-{cell}-{seed}' for cell in ('T2', 'T3', 'V1', 'V3') for seed in (20261114, 20261115)]
REPAIR_IDS = [*REPAIR_TRAIN_IDS, 'v6-additional_seeds']
REPAIR_SOURCE = 'src/mediflow_medvlm/campaign_v6_completion.py'
ANCHORS = {'v6-family_selection', 'v6-additional_seeds', 'v6-test', 'v6-report'}
MODULES = {'mediflow_medvlm.campaign_family_selection', 'mediflow_medvlm.campaign_v6_completion'}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def absolute(value):
    path = Path(value)
    require(path.is_absolute() and '..' not in path.parts
            and not any(p.is_symlink() for p in (path, *path.parents)), 'absolute nonsymlink path required')
    return path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with absolute(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(absolute(path).read_bytes())


def reference(path):
    return {'path': str(absolute(path)), 'sha256': file_hash(path)}


def pin_bytes(ref):
    require(set(ref) == {'path', 'sha256'} and re.fullmatch(r'[0-9a-f]{64}', ref['sha256']), 'exact external file pin required')
    require(reference(ref['path']) == ref, 'pinned completion evidence changed')


def pinned(ref):
    pin_bytes(ref)
    return read(ref['path'])


def validate_pair(engine, parent, replacement, *, activated, inserted):
    engine.validate_manifest(parent)
    engine.validate_manifest(replacement)
    require(digest({k: v for k, v in parent.items() if k != 'jobs'})
            == digest({k: v for k, v in replacement.items() if k != 'jobs'}), 'campaign science metadata immutable')
    old = {j['job_id']: j for j in parent['jobs']}
    new = {j['job_id']: j for j in replacement['jobs']}
    require((activated or inserted) and set(activated) <= ANCHORS
            and len(set(activated)) == len(activated),
            'only explicit V6 completion anchors or children may activate')
    require(len(set(inserted)) == len(inserted) and not set(inserted) & set(old)
            and set(new) == set(old) | set(inserted), 'exact declared child job insertion required')
    require([j['job_id'] for j in replacement['jobs'] if j['job_id'] in old]
            == [j['job_id'] for j in parent['jobs']], 'existing job order immutable')
    for key, before in old.items():
        after = new[key]
        if key not in activated:
            require(after == before, 'existing protected job changed')
            continue
        require(before.get('implementation_pending') and not before.get('argv'), 'only never-implemented anchors may activate')
        require(all(after.get(k) == before.get(k) for k in ('job_id', 'run_id', 'cell_id', 'phase')), 'anchor scientific identity changed')
        require(set(before.get('dependencies', [])) <= set(after.get('dependencies', [])), 'anchor prerequisite removed')
    for key in [*activated, *inserted]:
        job = new[key]
        argv = job.get('argv', [])
        require((key in ANCHORS or key.startswith(('v6-additional-', 'v6-test-')))
                and not key.startswith('v7-') and job.get('implementation_pending') is False,
                'executable V6-only completion work required')
        require(len(argv) == 7 and argv[1] == '-m' and argv[2] in MODULES
                and argv[3] == '--config' and argv[5:] == ['--receipt', job['success_receipt']],
                'exact native completion worker command required')
        require(job.get('gpu_lock_mode') == 'wrapper', 'same wrapper-owned physical lease required')
    return old, new


def validate_source_pair(engine, parent, replacement, *, repaired):
    engine.validate_manifest(parent); engine.validate_manifest(replacement)
    require(repaired == REPAIR_IDS and len(parent['jobs']) == len(replacement['jobs']) == 105
            and [j['job_id'] for j in parent['jobs']] == [j['job_id'] for j in replacement['jobs']]
            and digest({k:v for k,v in parent.items() if k != 'jobs'}) == digest({k:v for k,v in replacement.items() if k != 'jobs'}),
            'exact nine-job source-only repair of the same 105-job namespace required')
    old = {j['job_id']:j for j in parent['jobs']}; new = {j['job_id']:j for j in replacement['jobs']}
    require(set(REPAIR_IDS) <= set(old), 'registered additional-seed cohort required')
    for key, job in old.items():
        fixed = new[key]
        if key not in repaired:
            require(job == fixed, 'all ninety-six protected job declarations immutable'); continue
        audit = key == REPAIR_IDS[-1]
        require(job.get('kind') == ('additional_audit' if audit else 'train')
                and job.get('implementation_pending') is False
                and job.get('argv', [])[1:3] == ['-m','mediflow_medvlm.campaign_v6_completion'], 'only registered train/additional-audit source repair allowed')
        allowed = {'cwd','source_files','source_hash','config_hash','argv'} | ({'data_hash'} if audit else set())
        require(digest({k:v for k,v in job.items() if k not in allowed}) == digest({k:v for k,v in fixed.items() if k not in allowed}), 'source repair scientific job identity changed')
        require(len(job['argv']) == len(fixed['argv']) == 7
                and job['argv'][:4]+job['argv'][5:] == fixed['argv'][:4]+fixed['argv'][5:]
                and job['argv'][4] != fixed['argv'][4] and job['cwd'] != fixed['cwd'], 'source repair requires new archive/config and unchanged nonconfig argv')
        require(len(job['source_files']) == 36 and set(job['source_files']) == set(fixed['source_files'])
                and REPAIR_SOURCE in job['source_files'] and job['source_files'][REPAIR_SOURCE] != fixed['source_files'][REPAIR_SOURCE]
                and all(fixed['source_files'][name] == sha for name,sha in job['source_files'].items() if name != REPAIR_SOURCE), 'only completion worker may change in the same 36-source closure')
        before, after = read(job['argv'][4]), read(fixed['argv'][4])
        excluded = {'source_files','source_hash','config_hash'} | ({'data','data_hash'} if audit else set())
        require(digest({k:v for k,v in before.items() if k not in excluded}) == digest({k:v for k,v in after.items() if k not in excluded}), 'source repair scientific config identity changed')
        if audit:
            require(digest({k:v for k,v in before['data'].items() if k != 'train_configs'}) == digest({k:v for k,v in after['data'].items() if k != 'train_configs'})
                    and before['data']['train_configs'] == [reference(old[k]['argv'][4]) for k in REPAIR_TRAIN_IDS]
                    and after['data']['train_configs'] == [reference(new[k]['argv'][4]) for k in REPAIR_TRAIN_IDS], 'audit may change only ordered train-config references')
        for config, declaration in ((before,job),(after,fixed)):
            require(config.get('config_hash') == digest({k:v for k,v in config.items() if k != 'config_hash'})
                    and config.get('data_hash') == digest(config.get('data')) and config.get('source_hash') == digest(config.get('source_files'))
                    and all(config.get(k) == declaration.get(k) for k in ('job_id','run_id','source_files','source_hash','config_hash','data_hash','model_hash','success_receipt')), 'source repair config/job hashes changed')
            engine.verify_source(declaration)
    require(len({old[k]['source_hash'] for k in repaired}) == len({new[k]['source_hash'] for k in repaired}) == 1, 'one exact prior/replacement source cohort required')
    return old, new


class CompletionMixin:
    def _verify_worker_job(self, job, config_ref):
        """Run the pinned worker's own scientific validator in a fresh CPU process.

        The legacy owner refuses mixed package roots, so do not import a new
        worker into that process. This finite verifier cannot execute run_job.
        """
        require(job['argv'][0] == sys.executable, 'same pinned Python runtime required')
        code = ('import importlib,json,sys;'
                'spec=json.load(sys.stdin);sys.path.insert(0,spec["repo"]+"/src");'
                'm=importlib.import_module(spec["module"]);'
                'result=m.job_from_config(spec["config"],repo=spec["repo"],python=sys.executable);'
                'exec("if \\\"torch\\\" in sys.modules: raise RuntimeError(\\\"CPU verifier imported GPU runtime\\\")");'
                'print(json.dumps(result,sort_keys=True))')
        env = {key: os.environ[key] for key in ('HOME', 'PATH', 'LANG', 'LC_ALL', 'TMPDIR') if key in os.environ}
        env.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   PYTHONDONTWRITEBYTECODE='1', HF_HUB_DISABLE_TELEMETRY='1')
        process = subprocess.run([sys.executable, '-I', '-B', '-c', code],
            input=json.dumps({'module': job['argv'][2], 'repo': job['cwd'], 'config': config_ref['path']}),
            capture_output=True, text=True, env=env, cwd=job['cwd'], timeout=600)
        require(process.returncode == 0, 'pinned scientific worker validation failed: ' + process.stderr[-1200:])
        require(json.loads(process.stdout) == job, 'worker-generated exact job closure changed')

    def _completion_chain(self):
        cached = getattr(self, '_completion_cache', None)
        if cached is not None:
            # Full scientific validation runs at each outer _verify, including
            # the routed canonical lock entry. Repeated native load/status calls
            # within that transaction still rehash every pinned config/source.
            for ref, binding, _, _ in cached:
                pinned(ref)
                pinned(binding['parent'])
                replacement = pinned(binding['replacement'])
                jobs = {j['job_id']: j for j in replacement['jobs']}
                for key, config_ref in binding['configs'].items():
                    pinned(config_ref)
                    self.engine.verify_source(jobs[key])
                for proof in binding['review_evidence']:
                    pin_bytes(proof)
                if binding['schema_version'] == REPAIR_SCHEMA:
                    review = pinned(binding['repair_review'])
                    for proof in review['evidence']: pin_bytes(proof)
            return cached
        entries = []
        prior_replacement = None
        for ref in self.completion_refs:
            binding = pinned(ref)
            require(binding.get('schema_version') in {SCHEMA, REPAIR_SCHEMA}
                    and binding.get('campaign_root') == str(self.root)
                    and binding.get('data_root') == str(self.data_root)
                    and binding.get('parent_deployment') == self.completion_parent_deployment,
                    'completion owner namespace or parent route changed')
            parent, replacement = pinned(binding['parent']), pinned(binding['replacement'])
            require(binding.get('previous_binding') == (entries[-1][0] if entries else None), 'exact previous completion binding pin required')
            require(prior_replacement is None or digest(parent) == digest(prior_replacement), 'continuous immutable refinement chain required')
            repair = binding['schema_version'] == REPAIR_SCHEMA
            if repair:
                require(binding['activated'] == binding['inserted'] == [] and binding.get('previous_binding') is not None
                        and binding.get('original_anchor_records') == {}, 'repair cannot activate/insert or fabricate parent anchors')
                old, _ = validate_source_pair(self.engine,parent,replacement,repaired=binding['repaired'])
                require(set(binding['parent_record_hashes']) == set(old)
                        and set(binding['parent_repair_records']) == set(REPAIR_IDS), 'all 105 parent records and exact repair snapshots required')
                for key,record in binding['parent_repair_records'].items():
                    require(digest(record) == binding['parent_record_hashes'][key]
                            and record.get('status') in {'ANALYSIS_REQUIRED','FAILED','BLOCKED_PREREQUISITE'}
                            and record.get('source_repair_required') is True and record.get('failed_source_hash') == old[key]['source_hash']
                            and all(type(record.get(k)) is int and record[k] >= 0 for k in ('attempts','failure_count','recovery_attempts'))
                            and isinstance(record.get('history'),list), 'exact held prior-source record proof required')
                review = pinned(binding['repair_review'])
                new = {j['job_id']:j for j in replacement['jobs']}
                require(review.get('schema_version') == REPAIR_REVIEW_SCHEMA and review.get('status') == 'CPU_VALIDATED'
                        and review.get('cpu_only') is True and review.get('cpu_tests_passed') is True
                        and review.get('job_ids') == REPAIR_IDS
                        and review.get('parent_source_hash') == old[REPAIR_IDS[0]]['source_hash']
                        and review.get('replacement_source_hash') == new[REPAIR_IDS[0]]['source_hash']
                        and isinstance(review.get('reason'),str) and review['reason'].strip() and review.get('evidence'), 'typed exact-source CPU repair review required')
                for proof in review['evidence']: pin_bytes(proof)
            else:
                validate_pair(self.engine, parent, replacement, activated=binding['activated'], inserted=binding['inserted'])
            jobs = {j['job_id']: j for j in replacement['jobs']}
            changed = binding['repaired'] if repair else binding['activated'] + binding['inserted']
            require(set(binding['configs']) == set(changed), 'all changed workers need pinned configs')
            for key in changed:
                job = jobs[key]
                config = pinned(binding['configs'][key])
                require(job['argv'][4] == binding['configs'][key]['path']
                        and config.get('config_hash') == digest({k: v for k, v in config.items() if k != 'config_hash'})
                        and config.get('source_hash') == digest(config.get('source_files'))
                        and config.get('data_hash') == digest(config.get('data')), 'completion config identity changed')
                require(all(config.get(k) == job.get(k) for k in ('job_id', 'run_id', 'source_files',
                        'source_hash', 'config_hash', 'data_hash', 'model_hash', 'success_receipt')), 'completion config differs from job')
                source = absolute(job['cwd'])
                require(source.parent == self.root / 'snapshots' and re.fullmatch(r'[0-9a-f]{40}', source.name), 'immutable source archive required')
                require(absolute(job['argv'][4]).is_relative_to(self.root / 'configs')
                        and absolute(job['success_receipt']) == self.root / 'receipts' / (key + '.json'), 'completion config/receipt escaped owner namespace')
                self.engine.verify_source(job)
                self._verify_worker_job(job, binding['configs'][key])
            for proof in binding['review_evidence']:
                pin_bytes(proof)
            require(binding['review_evidence'], 'CPU review evidence required')
            entries.append((ref, binding, parent, replacement))
            prior_replacement = replacement
        require(entries, 'at least one pinned completion refinement required')
        self._completion_cache = entries
        return entries

    def _manifest(self, value):
        chain = self._completion_chain()
        known = [chain[0][2], *(entry[3] for entry in chain)]
        require(any(digest(value) == digest(item) for item in known), 'manifest outside exact completion chain')
        super()._manifest(chain[0][2])
        return value

    def _check_original_records(self, state):
        chain = self._completion_chain()
        active = [entry for entry in chain if digest(entry[3]) == state['manifest_hash']]
        if not active:
            return super()._check_original_records(state)
        index = chain.index(active[0])
        view = copy.deepcopy(state)
        for _, binding, _, _ in chain[:index + 1]:
            for key in binding['inserted']:
                view['jobs'].pop(key, None)
            for key in binding['activated']:
                view['jobs'][key] = copy.deepcopy(binding['original_anchor_records'][key])
        # Every noncompletion original record remains checked by the prior facade.
        return super()._check_original_records(view)

    def _verify(self, *, prepared=True):
        self._completion_cache = None
        result = super()._verify(prepared=prepared)
        self._completion_chain()
        return result

    def _registered(self):
        super()._registered()
        manifest, state = self.engine.Supervisor.load(self.supervisor)
        self._validate_live(manifest, state)
        repair, _, legacy = self._likelihood_recovery()
        require(digest(self._completion_chain()[0][2]) == digest(legacy), 'completion must extend the exact repaired legacy manifest')
        events = [event for event in state.get('manifest_revisions', [])
                  if event.get('kind') == 'RECORDED_CHOICE_LIKELIHOOD_SOURCE_REPAIR'
                  and event.get('binding_sha256') == self.likelihood_recovery_sha256
                  and event.get('manifest_hash') == digest(legacy)
                  and event.get('previous_hash') == repair['parent_manifest_hash']
                  and event.get('parent_record_hashes') == repair['parent_record_hashes']
                  and event.get('original_record_hashes') == self._original_choice_binding()['parent_record_hashes']
                  and event.get('requeued_jobs') == [f'v6-choice-likelihood-T{i}-{role}' for i in range(1, 7) for role in ('base', 'tuned')] + ['v6-choice-likelihood']
                  and event.get('user_policy_receipt_sha256') == self.retry_policy_sha256]
        require(len(events) == 1, 'prior source repair registration audit must remain verified')

    def _validate_live(self, manifest, state):
        chain = self._completion_chain()
        self._manifest(manifest)
        require(set(state['jobs']) == {j['job_id'] for j in manifest['jobs']}, 'actual manifest/state membership mismatch')
        self._check_original_records(state)
        matches = [i for i, entry in enumerate(chain) if digest(entry[3]) == state['manifest_hash']]
        if matches:
            for ref, binding, parent, replacement in chain[:matches[0] + 1]:
                repair = binding['schema_version'] == REPAIR_SCHEMA
                events = [e for e in state.get('manifest_revisions', []) if e.get('kind') == EVENT
                          and e.get('binding_sha256') == ref['sha256'] and e.get('manifest_hash') == digest(replacement)
                          and e.get('previous_hash') == digest(parent)
                          and e.get('activated_jobs') == binding['activated']
                          and e.get('inserted_jobs') == binding['inserted']
                          and e.get('parent_record_hashes') == binding['parent_record_hashes']]
                if repair:
                    counters = {key:{k:record[k] for k in ('attempts','failure_count','recovery_attempts')}
                                for key,record in binding['parent_repair_records'].items()}
                    events = [e for e in state.get('manifest_revisions', []) if e.get('kind') == REPAIR_EVENT
                              and e.get('binding_sha256') == ref['sha256'] and e.get('manifest_hash') == digest(replacement)
                              and e.get('previous_hash') == digest(parent) and e.get('repaired_jobs') == binding['repaired']
                              and e.get('parent_record_hashes') == binding['parent_record_hashes']
                              and e.get('preserved_counters') == counters
                              and e.get('repair_review_sha256') == binding['repair_review']['sha256']]
                    require(len(events) == 1, 'exact registered source repair audit required for every applied prefix')
                    event = events[0]
                    old,new = ({j['job_id']:j for j in value['jobs']} for value in (parent,replacement))
                    for key,before in binding['parent_repair_records'].items():
                        current = state['jobs'][key]; history = current.get('history',[]); prefix = before.get('history',[])
                        require(all(type(current.get(k)) is int and current[k] >= before[k] for k in counters[key])
                                and history[:len(prefix)] == prefix and len(history) > len(prefix), 'repair counters/history cannot be reset')
                        require(history[len(prefix)] == {'status':'SOURCE_REPAIR_AUTHORIZED','binding_sha256':ref['sha256'],
                            'original_source_hash':old[key]['source_hash'],'replacement_source_hash':new[key]['source_hash'],
                            'retry_limit':None,'mandatory_astra_before_retry':False,'at':event['at']}, 'pinned repair authorization history required')
                else:
                    require(len(events) == 1, 'exact registered completion audit required for every applied prefix')

    def call(self, command, *, manifest=None, job_id=None, receipt=None):
        require(command in {'status', 'tick', 'refine-completion'} and job_id is None and receipt is None,
                'only canonical status/tick/reviewed completion refinement supported')
        self._verify()
        self._registered()
        if command != 'refine-completion':
            require(manifest is None, 'unexpected manifest')
            # Bypass only the old facade's exact 97-job *routing* restriction.
            # Its verification, native Supervisor/recovery tick and V7 fence remain.
            return self.base.CampaignControl.call(self, command)
        chain = self._completion_chain()
        ref, binding, parent, replacement = chain[-1]
        require(manifest == replacement, 'exact reviewed replacement required')
        with self.routed_locks():
            if binding['schema_version'] == REPAIR_SCHEMA:
                return self.supervisor.repair_completion_source(ref,binding,parent,replacement)
            return self.supervisor.refine_completion(ref, binding, parent, replacement)


def install_supervisor(control):
    owner, engine, inherited = control, control.engine, type(control.supervisor)

    class CompletionSupervisor(inherited):
        def load(self):
            # Validate the full native view before inherited V7 dispatch projection.
            manifest, state = engine.Supervisor.load(self)
            owner._validate_live(manifest, state)
            return super().load()

        def _failure(self, record, reason, exit_code=None):
            super()._failure(record, reason, exit_code)
            # The inherited recovery load retains the exact native in-memory
            # records being transitioned by tick, not a separate disk reread.
            state = getattr(self, '_ll_state', None)
            if state is None:
                return
            current_keys = {key for _, binding, _, _ in owner._completion_chain()
                            for key in binding['activated'] + binding['inserted']}
            key = next((key for key in current_keys if state['jobs'].get(key) is record), None)
            if key is not None:
                manifest, _ = engine.Supervisor.load(self)
                jobs = {job['job_id']: job for job in manifest['jobs']}
                if (key.startswith('v6-test-telemetry-')
                        and jobs[key].get('kind') == 'telemetry_import'
                        and jobs[key].get('uses_gpu') is False):
                    # A killed CPU importer may have called the once-only import
                    # before losing its receipt. Keep the native failure and
                    # stop automatic import retries without holding GPU science.
                    record.update(status='FAILED', failure_domain='telemetry_reconstruction',
                                  telemetry_coverage='INDETERMINATE', scientific_failure=False)
                    record.setdefault('history', []).append({
                        'status': 'TELEMETRY_RECONSTRUCTION_FAILED',
                        'coverage': 'INDETERMINATE', 'automatic_reimport_allowed': False,
                        'at': engine.now()})
                    return
                for target in current_keys:
                    item = state['jobs'].get(target)
                    if item is not None and jobs[target]['source_hash'] == jobs[key]['source_hash'] \
                            and (target == key or item.get('status') == 'PENDING'):
                        item.update(status='ANALYSIS_REQUIRED', source_repair_required=True,
                                    failed_source_hash=jobs[key]['source_hash'])
                        item.setdefault('history', []).append({'status': 'SOURCE_REPAIR_REQUIRED',
                            'failed_job_id': key, 'failed_source_hash': jobs[key]['source_hash'],
                            'retry_limit': None, 'at': engine.now()})

        def tick(self):
            _, state = engine.Supervisor.load(self)
            keys = {key for _, binding, _, _ in owner._completion_chain()
                    for key in binding['activated'] + binding['inserted']}
            if any(state['jobs'].get(key, {}).get('source_repair_required') for key in keys):
                return self.status()
            return super().tick()

        def refine_completion(self, ref, binding, parent, replacement):
            with engine.try_lock(self.root / 'supervisor.lock') as lock:
                if lock is None:
                    return {'status': 'SUPERVISOR_BUSY'}
                current, state = engine.Supervisor.load(self)
                target = digest(replacement)
                if state['manifest_hash'] == target:
                    owner._validate_live(current, state)
                    return self.status()
                require(current == parent and file_hash(self.state_path) == binding['parent_state_sha256'], 'exact unchanged parent state required')
                require({key: digest(v) for key, v in state['jobs'].items()} == binding['parent_record_hashes'], 'preserve all parent records')
                old, new = validate_pair(engine, parent, replacement, activated=binding['activated'], inserted=binding['inserted'])
                for key in binding['activated']:
                    record = state['jobs'][key]
                    require(record == binding['original_anchor_records'][key]
                            and record.get('status') == 'IMPLEMENTATION_PENDING'
                            and all(record.get(k) == 0 for k in ('attempts', 'failure_count', 'recovery_attempts'))
                            and not record.get('history'), 'only untouched never-started anchor may activate')
                owner._completion_chain()
                version = self.root / 'manifests' / (target + '.json')
                if version.exists():
                    require(read(version) == replacement, 'immutable manifest version changed')
                else:
                    engine.atomic_json(version, replacement)
                for key in binding['activated']:
                    state['jobs'][key]['status'] = 'PENDING'
                for key in binding['inserted']:
                    state['jobs'][key] = {'status': 'PENDING', 'attempts': 0, 'failure_count': 0,
                                          'recovery_attempts': 0, 'history': []}
                state.setdefault('manifest_revisions', []).append({'kind': EVENT,
                    'previous_hash': digest(parent), 'manifest_hash': target, 'manifest_path': str(version),
                    'activated_jobs': binding['activated'], 'inserted_jobs': binding['inserted'],
                    'binding_sha256': ref['sha256'], 'parent_record_hashes': binding['parent_record_hashes'], 'at': engine.now()})
                state.update(manifest_hash=target, manifest_path=str(version), updated_at=engine.now())
                engine.atomic_json(self.state_path, state)
                return self.status()

        def repair_completion_source(self, ref, binding, parent, replacement):
            with engine.try_lock(self.root / 'supervisor.lock') as lock:
                if lock is None: return {'status':'SUPERVISOR_BUSY'}
                current,state = engine.Supervisor.load(self)
                require(state.get('queue_owner') == 'supervisor', 'sole canonical Supervisor owner required')
                if state['manifest_hash'] == digest(replacement):
                    owner._validate_live(current,state)
                    return self.status()
                require(current == parent and file_hash(self.state_path) == binding['parent_state_sha256']
                        and {key:digest(value) for key,value in state['jobs'].items()} == binding['parent_record_hashes'], 'exact unchanged reconciled repair parent state required')
                owner._validate_live(current,state)
                owner._quiescent(state,allow_dead_running=False)
                with ExitStack() as leases:
                    for path in (self.root/'worker.lock',owner.data_root/'runtime/medvlm_gpu.lock'):
                        require(leases.enter_context(engine.try_lock(path)) is not None, 'canonical worker/GPU lease busy')
                    owner._quiescent(state,allow_dead_running=False)
                    owner._verify()
                    require(file_hash(self.state_path) == binding['parent_state_sha256'], 'repair parent changed during locked verification')
                    old,new = validate_source_pair(engine,parent,replacement,repaired=binding['repaired'])
                    for key in binding['repaired']:
                        record = state['jobs'][key]
                        require(record.get('status') != 'SUCCEEDED', 'successful completion outputs cannot be source-rebound')
                        require(record == binding['parent_repair_records'][key]
                                and record.get('status') in {'ANALYSIS_REQUIRED','FAILED','BLOCKED_PREREQUISITE'}
                                and record.get('source_repair_required') is True
                                and record.get('failed_source_hash') == old[key]['source_hash'], 'all nine jobs must be held for the exact prior source')
                        require(all(type(record.get(k)) is int and record[k] >= 0 for k in ('attempts','failure_count','recovery_attempts'))
                                and isinstance(record.get('history'),list), 'valid preserved counters/history required')
                        require(not absolute(new[key]['success_receipt']).exists(), 'existing completion success receipt cannot be overwritten')
                    target = digest(replacement); version = self.root/'manifests'/(target+'.json'); at = engine.now()
                    if version.exists(): require(read(version) == replacement, 'immutable repair manifest changed')
                    else: engine.atomic_json(version,replacement)
                    counters = {key:{k:state['jobs'][key][k] for k in ('attempts','failure_count','recovery_attempts')} for key in binding['repaired']}
                    for key in binding['repaired']:
                        state['jobs'][key].update(status='PENDING',source_repair_required=False)
                        state['jobs'][key]['history'].append({'status':'SOURCE_REPAIR_AUTHORIZED','binding_sha256':ref['sha256'],
                            'original_source_hash':old[key]['source_hash'],'replacement_source_hash':new[key]['source_hash'],
                            'retry_limit':None,'mandatory_astra_before_retry':False,'at':at})
                    state.setdefault('manifest_revisions',[]).append({'kind':REPAIR_EVENT,
                        'previous_hash':digest(parent),'manifest_hash':target,'manifest_path':str(version),
                        'repaired_jobs':binding['repaired'],'binding_sha256':ref['sha256'],
                        'parent_record_hashes':binding['parent_record_hashes'],'preserved_counters':counters,
                        'repair_review_sha256':binding['repair_review']['sha256'],'at':at})
                    state.update(manifest_hash=target,manifest_path=str(version),updated_at=at)
                    engine.atomic_json(self.state_path,state)
                return self.status()

    control.supervisor.__class__ = CompletionSupervisor


def build_control(*, deployment, deployment_sha256, completion_binding, completion_binding_sha256, scope='auto'):
    require(scope == 'auto', 'completion owner retains V6-only automatic scope')
    saved_ref = {'path': str(absolute(deployment)), 'sha256': deployment_sha256}
    saved = pinned(saved_ref)
    require(saved['status_argv'][-1] == 'status' and saved['tick_argv'][-1] == 'tick'
            and saved['status_argv'][:-1] == saved['tick_argv'][:-1], 'same native status/tick route required')
    argv = saved['status_argv']
    require(file_hash(argv[1]) == GUARD_SHA and argv[-3:-1] == ['--scope', 'auto'], 'exact automatic V7 boundary required')
    spec = importlib.util.spec_from_file_location('_completion_prior_guard', absolute(argv[1]))
    prior = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prior)
    kwargs = {key[2:].replace('-', '_'): value for key, value in zip(argv[2:-1:2], argv[3:-1:2])}
    require(len(kwargs) * 2 == len(argv[2:-1]) and kwargs.get('scope') == 'auto', 'unique saved guard keywords required')
    control = prior.build_control(**kwargs)
    refs = []
    current = {'path': str(absolute(completion_binding)), 'sha256': completion_binding_sha256}
    seen = set()
    while current:
        require(current['sha256'] not in seen, 'finite acyclic refinement chain required')
        seen.add(current['sha256'])
        value = pinned(current)
        refs.append(current)
        current = value.get('previous_binding')
    class CompletionControl(CompletionMixin, type(control)):
        pass
    control.__class__ = CompletionControl
    control.completion_refs = list(reversed(refs))
    control.completion_parent_deployment = saved_ref
    install_supervisor(control)
    return control


def prepare_binding(deployment, parent_manifest, replacement_manifest, state_path, *, activated,
                    inserted, config_paths, review_paths, previous_binding=None):
    state = read(state_path)
    return {'schema_version': SCHEMA, 'campaign_root': str(absolute(state_path).parent),
            'data_root': read(deployment)['data_root'], 'parent_deployment': reference(deployment),
            'parent': reference(parent_manifest), 'replacement': reference(replacement_manifest),
            'previous_binding': previous_binding, 'activated': activated, 'inserted': inserted,
            'parent_state_sha256': file_hash(state_path),
            'parent_record_hashes': {key: digest(v) for key, v in state['jobs'].items()},
            'original_anchor_records': {key: state['jobs'][key] for key in activated},
            'configs': {read(path)['job_id']: reference(path) for path in config_paths},
            'review_evidence': [reference(path) for path in review_paths]}


def prepare_repair_binding(deployment,parent_manifest,replacement_manifest,state_path,*,config_paths,
                           review_paths,previous_binding,repair_review_path):
    state = read(state_path)
    require(previous_binding is not None, 'source repair must extend the previous completion binding')
    require(state['manifest_hash'] == digest(read(parent_manifest)), 'repair binding must pin the current parent manifest')
    value = prepare_binding(deployment,parent_manifest,replacement_manifest,state_path,activated=[],inserted=[],
        config_paths=config_paths,review_paths=review_paths,previous_binding=previous_binding)
    value.update(schema_version=REPAIR_SCHEMA,repaired=copy.deepcopy(REPAIR_IDS),
        parent_repair_records={key:copy.deepcopy(state['jobs'][key]) for key in REPAIR_IDS},
        repair_review=reference(repair_review_path))
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deployment', required=True)
    parser.add_argument('--deployment-sha256', required=True)
    parser.add_argument('--completion-binding', required=True)
    parser.add_argument('--completion-binding-sha256', required=True)
    parser.add_argument('--scope', choices=['auto'], default='auto')
    parser.add_argument('command', choices=['status', 'tick', 'refine-completion'])
    parser.add_argument('--manifest')
    args = vars(parser.parse_args(argv))
    command, manifest = args.pop('command'), args.pop('manifest')
    control = build_control(**args)
    result = control.call(command, manifest=read(manifest) if manifest else None)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
