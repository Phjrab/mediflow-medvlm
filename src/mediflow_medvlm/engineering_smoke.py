"""Preliminary T1 engineering smoke through the unchanged native training CLI.

This is neither the common V6 freeze nor a V6 main training condition. Its
adapter must never initialize main training. GPU work occurs only when this
module is explicitly launched by the authorized independent supervisor.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .runtime import atomic_json, canonical_hash, file_hash, utcnow
from .training.config import load_config, load_frozen_data, verified_checkpoint


def _read(path):
    return json.loads(Path(path).read_text())


def _training_code_hash():
    package = Path(__file__).resolve().parent
    names = {"src/mediflow_medvlm/" + path.relative_to(package).as_posix(): file_hash(path)
             for path in sorted((package / 'training').rglob('*'))
             if path.is_file() and path.suffix in {'.py', '.json', '.yaml', '.yml'}}
    names['src/mediflow_medvlm/runtime.py'] = file_hash(package / 'runtime.py')
    return canonical_hash(names)


def _bound_identity(config_path, config):
    values = {'run_id': config['run_id'], 'config_hash': file_hash(config_path),
              'data_hash': file_hash(config['freeze_manifest']),
              'model_hash': canonical_hash({'model_id': config['model_id'], 'revision': config['revision']})}
    for key, actual in values.items():
        expected = os.environ.get('MEDVLM_SUPERVISOR_' + key.upper())
        if expected is not None and expected != actual:
            raise ValueError('supervisor identity mismatch: ' + key)
    source_hash = os.environ.get('MEDVLM_SUPERVISOR_SOURCE_HASH')
    values['source_hash'] = _training_code_hash() if source_hash is None else source_hash
    return values


def _check_manifest_identity(manifest, config, data):
    if (manifest.get('run_id') != config['run_id'] or manifest.get('model_id') != config['model_id']
            or manifest.get('revision') != config['revision']
            or manifest.get('config_hash') != canonical_hash(config)):
        raise ValueError('training manifest config/run/model mismatch')
    identity = manifest.get('identity', {})
    required = {'config_sha256': canonical_hash(config), 'data': data,
                'model_id': config['model_id'], 'revision': config['revision'],
                'training_code_sha256': _training_code_hash()}
    if any(identity.get(key) != value for key, value in required.items()):
        raise ValueError('training checkpoint source/config/data/model identity mismatch')
    return identity


def _latest(output, manifest, expected_identity):
    pointer = output / 'latest.json'
    candidate = _read(pointer).get('checkpoint') if pointer.exists() else manifest.get('latest_checkpoint')
    if not candidate:
        raise ValueError('no complete latest checkpoint available')
    path = Path(candidate).resolve()
    checkpoints = (output / 'checkpoints').resolve()
    if not path.is_relative_to(checkpoints) or path.parent != checkpoints or path.name.startswith('.'):
        raise ValueError('latest checkpoint outside this run')
    checkpoint = verified_checkpoint(path)
    if checkpoint.get('identity') != expected_identity:
        raise ValueError('latest checkpoint identity mismatch')
    files = checkpoint.get('artifacts', {}).get('files', {})
    if not any(name.startswith('adapter_model') and name.endswith('.safetensors') for name in files):
        raise ValueError('latest checkpoint has no verified adapter weights')
    if 'training_state.pt' not in files:
        raise ValueError('adapter-only checkpoint cannot resume exact optimizer state')
    if not 0 <= checkpoint.get('optimizer_steps', -1) <= 20:
        raise ValueError('checkpoint optimizer count outside finite smoke')
    return path, checkpoint


def _verified_final(output, manifest, config, data):
    identity = _check_manifest_identity(manifest, config, data)
    if (manifest.get('status') != 'SUCCEEDED' or manifest.get('optimizer_steps') != 20
            or manifest.get('samples_processed') != 320 or manifest.get('adapter_changed') is not True):
        raise ValueError('engineering smoke requires success, 20 updates, 320 samples and changed adapter')
    selection = manifest.get('training_selection', {})
    if selection.get('usable') != 320 or selection.get('excluded') != 0 or selection.get('offered') != 320:
        raise ValueError('engineering smoke requires exactly 320 usable intact samples without exclusions')
    final, checkpoint = _latest(output, manifest, identity)
    if checkpoint.get('optimizer_steps') != 20 or checkpoint.get('samples_processed') != 320:
        raise ValueError('latest checkpoint is not the complete final smoke checkpoint')
    if str(final) != str(Path(manifest.get('latest_checkpoint', '')).resolve()):
        raise ValueError('successful manifest latest differs from final pointer')
    return final


def run_smoke(config_path, receipt_path, *, invoke=None):
    config_path, receipt_path = Path(config_path).resolve(), Path(receipt_path).resolve()
    config = load_config(config_path)
    if (config.get('model_id') != 'Qwen/Qwen3-1.7B' or config.get('mode') != 'smoke'
            or config.get('max_optimizer_steps') != 20 or config.get('train_samples_cap') != 320
            or config.get('gradient_accumulation_steps', 16) != 16
            or config.get('synthetic_fixture') or config.get('device', 'cuda') != 'cuda'):
        raise ValueError('only the real preliminary T1 20-update/320-sample smoke is supported')
    identity = _bound_identity(config_path, config)
    _, data = load_frozen_data(config)
    output = Path(config['output_dir']).resolve()
    snapshot = output / 'config.json'
    if snapshot.exists() and canonical_hash(_read(snapshot)) != canonical_hash(config):
        raise ValueError('existing run config is immutable')
    progress_path = Path(os.environ.get('MEDVLM_SUPERVISOR_PROGRESS_PATH', str(receipt_path.with_suffix('.progress.json'))))
    progress = {**identity, 'phase': 'preliminary_engineering_smoke', 'optimizer_step': 0,
                'total_updates': 20, 'last_complete_checkpoint': None, 'stage': 'verify_config'}
    def update(stage, manifest=None):
        progress.update(stage=stage, heartbeat_at=utcnow(), last_progress_at=utcnow())
        if manifest:
            progress.update(optimizer_step=manifest.get('optimizer_steps', 0),
                            last_complete_checkpoint=manifest.get('latest_checkpoint'))
        atomic_json(progress_path, progress)
    def command(stage, extra=()):
        update(stage)
        args = [sys.executable, '-m', 'mediflow_medvlm.training', stage, '--config', str(config_path), *extra]
        result = (invoke or subprocess.run)(args, check=False)
        if result.returncode:
            raise RuntimeError('training subprocess failed: ' + stage)
        if file_hash(config_path) != identity['config_hash'] or file_hash(config['freeze_manifest']) != identity['data_hash']:
            raise ValueError('immutable config/freeze changed during smoke')
    update('preflight')
    manifest_path = output / 'manifest.json'
    manifest = _read(manifest_path) if manifest_path.exists() else None
    if manifest and manifest.get('status') == 'SUCCEEDED':
        final = _verified_final(output, manifest, config, data)
    else:
        resume = None
        if manifest:
            # A failure before any update can safely rerun the same immutable job.
            if manifest.get('optimizer_steps', 0) or (output / 'latest.json').exists():
                expected = _check_manifest_identity(manifest, config, data)
                resume, _ = _latest(output, manifest, expected)
            elif (manifest.get('run_id') != config['run_id']
                  or manifest.get('config_hash') != canonical_hash(config)):
                raise ValueError('prior failed run identity mismatch')
        command('train', ['--resume', str(resume)] if resume else [])
        manifest = _read(manifest_path)
        final = _verified_final(output, manifest, config, data)
    update('reload_final', manifest)
    command('reload', ['--checkpoint', str(final)])
    reload_path = output / 'reload.json'
    reload = _read(reload_path)
    if (reload.get('status') != 'SUCCEEDED' or reload.get('run_id') != config['run_id']
            or Path(reload.get('checkpoint', '')).resolve() != final
            or not reload.get('fresh_process_required') or reload.get('generated_tokens', 0) < 1
            or not reload.get('pid') or reload['pid'] in {manifest.get('training_pid'), os.getpid()}):
        raise ValueError('final checkpoint fresh-process reload was not verified')
    # Reload may update the historical selected-best flag; final proof is separate.
    manifest = _read(manifest_path)
    _verified_final(output, manifest, config, data)
    artifacts = [{'path': str(path), 'sha256': file_hash(path)} for path in
                 (config_path, manifest_path, reload_path, final / 'checkpoint.json')]
    result = {'schema_version': 'medvlm.engineering-smoke.receipt.v1', 'status': 'SUCCEEDED',
              **identity, 'job_id': os.environ.get('MEDVLM_SUPERVISOR_JOB_ID'),
              'job_hash': os.environ.get('MEDVLM_SUPERVISOR_JOB_HASH'),
              'scope': 'PRELIMINARY_T1_ENGINEERING_SMOKE_ONLY', 'cell_id': 'T1',
              'optimizer_steps': 20, 'samples_processed': 320, 'final_checkpoint': str(final),
              'fresh_process_reload_verified': True, 'usable_as_main_initialization': False,
              'v6_common_freeze_completed': False, 'v6_main_training_completed': False,
              'artifacts': artifacts, 'finished_at': utcnow()}
    atomic_json(receipt_path, result)
    update('SUCCEEDED', manifest)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--receipt', required=True)
    args = parser.parse_args(argv)
    try:
        result = run_smoke(args.config, args.receipt)
        print(json.dumps({'status': result['status'], 'scope': result['scope']}))
        return 0
    except Exception as exc:
        atomic_json(Path(args.receipt).with_suffix('.failure.json'),
                    {'status': 'FAILED', 'exception_type': type(exc).__name__, 'finished_at': utcnow(),
                     'scope': 'PRELIMINARY_T1_ENGINEERING_SMOKE_ONLY'})
        print(json.dumps({'status': 'FAILED', 'exception_type': type(exc).__name__}), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
