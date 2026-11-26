#!/usr/bin/env python3
"""Fence automatic V7 dispatch; foreground terminal uses the same Supervisor.

Only a verified in-memory eligibility view changes during tick. The immutable
manifest, original records, native transition/verification code and physical
supervisor/worker/GPU locks remain owned by the existing control object.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
PRIOR_SHA256 = 'cecc18abe1e7350e9d1305f3b4850f64d17d7902ad2196ba1ef80a7069009f25'
CONDITIONS = ['P0', 'P1', 'R0', 'R1']
AUTOMATIC_V7_IDS = {*(f'v7-{condition}' for condition in CONDITIONS), 'v7-report'}
TERMINAL = {'SUCCEEDED', 'FAILED', 'BLOCKED_PREREQUISITE'}


def require(value, reason):
    if not value:
        raise ValueError(reason)


def absolute(value):
    path = Path(value)
    require(path.is_absolute() and '..' not in path.parts
            and not any(p.is_symlink() for p in (path, *path.parents)),
            'absolute nonsymlink path required')
    return path


def file_hash(path):
    h = hashlib.sha256()
    with absolute(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(absolute(path).read_bytes())


def pinned(path, sha):
    require(re.fullmatch(r'[0-9a-f]{64}', sha or ''), 'external SHA256 required')
    require(file_hash(path) == sha, 'pinned file changed')
    return read(path)


def v7_job(job):
    return (job['job_id'].startswith('v7-') or job.get('cell_id') in CONDITIONS + ['ALL_V7']
            or str(job.get('phase', '')).lower().startswith('v7')
            or str(job.get('run_id', '')).startswith('v7-'))


class Boundary:
    def __init__(self, control, *, scope, deployment, deployment_sha256,
                 handoff=None, handoff_sha256=None):
        require(scope in {'auto', 'terminal'}, 'unknown execution scope')
        self.control, self.scope = control, scope
        self.deployment, self.deployment_sha256 = absolute(deployment), deployment_sha256
        self.handoff, self.handoff_sha256 = handoff, handoff_sha256
        self.dispatching = False

    def authorize(self, manifest, state, *, token_required):
        h = pinned(self.handoff, self.handoff_sha256)
        require(h.get('schema_version') == 'mediflow.v7-terminal-handoff.v1'
                and h.get('campaign_root') == str(self.control.root)
                and h.get('data_root') == str(self.control.data_root)
                and h.get('campaign_id') == state['campaign_id']
                and h.get('model_id') == 'Qwen/Qwen3-1.7B'
                and h.get('conditions') == CONDITIONS, 'V7 handoff identity mismatch')
        require(h.get('parent_deployment') == {'path': str(self.deployment),
                'sha256': self.deployment_sha256}, 'parent deployment mismatch')
        require(h.get('guard_operator') == {'path': str(absolute(__file__)),
                'sha256': file_hash(__file__)}, 'guard source identity mismatch')
        ref = h['manifest']
        bound = pinned(ref['path'], ref['sha256'])
        require(bound == manifest and ref['manifest_hash'] == state['manifest_hash']
                and self.control.engine.canonical_hash(bound) == state['manifest_hash'],
                'handoff requires exact current immutable manifest')
        jobs = {j['job_id']: j for j in manifest['jobs']}
        allowed = []
        purposes = {}
        for condition in CONDITIONS:
            entry = h['condition_jobs'][condition]
            require(entry.get('train') and entry.get('evaluation'), 'V7 train/evaluation implementation missing')
            for kind in ('train', 'evaluation'):
                require(isinstance(entry[kind], list), 'ordered job ID list required')
                allowed.extend(entry[kind])
                purposes.update({key: (condition, kind) for key in entry[kind]})
        require(h.get('report_job_ids'), 'V7 report implementation missing')
        allowed.extend(h['report_job_ids'])
        purposes.update({key: ('ALL_V7', 'report') for key in h['report_job_ids']})
        require(allowed == h.get('allowed_job_ids') and len(set(allowed)) == len(allowed)
                and set(allowed) == {key for key, job in jobs.items() if v7_job(job)},
                'exact complete V7 scope required')
        require(all(not job.get('implementation_pending') and job.get('argv')
                    for key, job in jobs.items() if key in allowed), 'V7 executable workers missing')
        bindings = h.get('job_bindings', {})
        require(set(bindings) == set(allowed), 'every V7 executable requires a science binding')
        for key in allowed:
            job, binding = jobs[key], bindings[key]
            condition, kind = purposes[key]
            require(binding.get('job_hash') == self.control.engine.canonical_hash(job)
                    and binding.get('condition') == condition and binding.get('kind') == kind,
                    'V7 job/condition identity mismatch')
            argv = job['argv']
            require(argv.count('--config') == 1, 'V7 native job requires one frozen --config')
            config_path = argv[argv.index('--config') + 1]
            ref = binding['config']
            require(config_path == ref['path'], 'V7 frozen config routing mismatch')
            config = pinned(config_path, ref['sha256'])
            canonical = self.control.engine.canonical_hash
            require(config.get('config_hash') == canonical({k: v for k, v in config.items() if k != 'config_hash'})
                    and config.get('data_hash') == canonical(config.get('data'))
                    and config.get('model_hash') == canonical(config.get('model'))
                    and config.get('source_hash') == canonical(config.get('source_files')),
                    'V7 config science hashes changed')
            require(all(config.get(field) == job.get(field) for field in
                    ('job_id', 'run_id', 'source_files', 'source_hash', 'config_hash', 'data_hash', 'model_hash'))
                    and config.get('model_id') == 'Qwen/Qwen3-1.7B'
                    and config.get('condition') == condition
                    and job.get('cell_id') == condition,
                    'V7 model/condition/config identity mismatch')
        require(all(r['status'] in TERMINAL for key, r in state['jobs'].items() if key not in allowed),
                'V6 must be terminal before terminal V7 takeover')
        report_id = h.get('v6_report_job_id')
        require(report_id == 'v6-report' and state['jobs'][report_id]['status'] == 'SUCCEEDED',
                'verified V6 report required')
        for key, job in jobs.items():
            if key in allowed:
                self.control.engine.verify_source(job)
            if state['jobs'][key]['status'] == 'SUCCEEDED':
                actual = self.control.engine.verify_success(job)
                require(actual.get('receipt_sha256') == state['jobs'][key].get('verification', {}).get('receipt_sha256'),
                        'successful receipt identity changed')
        require(h.get('readiness'), 'V7 readiness evidence missing')
        for ref in h['readiness']:
            evidence = pinned(ref['path'], ref['sha256'])
            require(evidence.get('status') == 'READY' and evidence.get('cpu_tests_passed') is True
                    and evidence.get('v7_workers_implemented') is True
                    and evidence.get('campaign_root') == str(self.control.root)
                    and evidence.get('manifest_hash') == state['manifest_hash']
                    and evidence.get('job_bindings_hash') == self.control.engine.canonical_hash(bindings),
                    'V7 readiness is incomplete or belongs to another science closure')
        if token_required:
            token = os.environ.get('MEDVLM_V7_SCOPE_TOKEN', '')
            require(token and hashlib.sha256(token.encode()).hexdigest() == h.get('manual_scope_token_sha256'),
                    'explicit foreground terminal scope token required')
        return allowed

    def install(self):
        boundary, inherited = self, type(self.control.supervisor)

        class ScopedSupervisor(inherited):
            def load(self):
                manifest, state = super().load()
                if boundary.dispatching:
                    # Native tick calls load after acquiring its routed canonical
                    # supervisor lock. Revalidate scope there, not just at CLI preflight.
                    if boundary.scope == 'auto':
                        require({job['job_id'] for job in manifest['jobs'] if v7_job(job)} == AUTOMATIC_V7_IDS,
                                'V7 cohort changed: review a new pinned boundary before dispatch')
                        allowed = None
                    else:
                        allowed = boundary.authorize(manifest, state, token_required=True)
                    manifest = {**manifest, 'jobs': [job for job in manifest['jobs']
                        if (job['job_id'] in allowed if allowed is not None else not v7_job(job))]}
                return manifest, state

            def tick(self):
                require(not boundary.dispatching, 'recursive tick forbidden')
                boundary.dispatching = True
                try:
                    return super().tick()
                finally:
                    boundary.dispatching = False

            def status(self):
                active = boundary.dispatching
                boundary.dispatching = False
                try:
                    result = super().status()
                    manifest, state = super().load()
                    verified, reasons = False, []
                    if boundary.scope == 'terminal':
                        try:
                            boundary.authorize(manifest, state, token_required=True)
                            verified = True
                        except (ValueError, OSError, KeyError, TypeError) as exc:
                            reasons.append(str(exc))
                    result.update(terminal_scope_v7=boundary.scope == 'terminal',
                        manual_handoff_verified=verified, expected_manifest_hash=state['manifest_hash'],
                        v7_manual_start_required=True, terminal_readiness_reasons=reasons,
                        automatic_dispatch_scope='V6_ONLY', guard_operator_sha256=file_hash(__file__))
                    return result
                finally:
                    boundary.dispatching = active

        self.control.supervisor.__class__ = ScopedSupervisor
        return self.control


def build_control(deployment, deployment_sha256, *, scope='auto', handoff=None, handoff_sha256=None):
    saved = pinned(deployment, deployment_sha256)
    a, tick = saved['status_argv'], saved['tick_argv']
    require(a[-1] == 'status' and tick[-1] == 'tick' and a[:-1] == tick[:-1], 'saved status/tick routing mismatch')
    require(len(a[2:-1]) % 2 == 0 and saved['operator'] == a[1], 'saved operator arguments invalid')
    require(file_hash(a[1]) == PRIOR_SHA256, 'prior recovery operator changed')
    pairs = list(zip(a[2:-1:2], a[3:-1:2]))
    require(all(re.fullmatch(r'--[a-z][a-z0-9-]*', key) for key, _ in pairs)
            and len({key for key, _ in pairs}) == len(pairs), 'invalid saved control keywords')
    kwargs = {key[2:].replace('-', '_'): value for key, value in pairs}
    require(kwargs.get('campaign_root') == saved['campaign_root'] and kwargs.get('data_root') == saved['data_root'],
            'saved namespace routing mismatch')
    spec = importlib.util.spec_from_file_location('_terminal_guard_prior', absolute(a[1]))
    prior = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prior)
    control = prior.build_control(**kwargs)
    return Boundary(control, scope=scope, deployment=deployment, deployment_sha256=deployment_sha256,
                    handoff=handoff, handoff_sha256=handoff_sha256).install()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deployment', type=Path, required=True)
    parser.add_argument('--deployment-sha256', required=True)
    parser.add_argument('--scope', choices=['auto', 'terminal'], default='auto')
    parser.add_argument('--handoff', type=Path)
    parser.add_argument('--handoff-sha256')
    parser.add_argument('command', choices=['status', 'tick'])
    args = vars(parser.parse_args(argv)); command = args.pop('command')
    control = build_control(**args)
    print(json.dumps(control.call(command), ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
