#!/usr/bin/env python3
"""Finite, isolated 71762 category-random experiment; resumes verified successes."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys

from complete_v1 import train_stage, reload_stage, verify_training_cache, verify_reload_cache
from mediflow_medvlm.runtime import atomic_json, canonical_hash, file_hash, resolve_resource_policy, utcnow
from mediflow_medvlm.evaluation.runner import compare, freeze_protocol, load_protocol
from mediflow_medvlm.evaluation.v2 import promote_protocol


def read(path):
    return json.loads(Path(path).read_text())


def immutable(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError('IMMUTABLE_DECLARATION_CHANGED: ' + path.name)
    else:
        atomic_json(path, value)


def training_config(original, corpus, root, model, phase):
    data = corpus['splits']
    name = 'aihub71762_category_random_v4/'
    config = {k: v for k, v in original.items()
              if k not in {'total_gpu_hours', 'per_model_gpu_hours', 'wall_seconds'}}
    run_id = f'{model}-category-random-{phase}-v4-20261112'
    config.update(run_id=run_id, pair_id=f'{model}-category-random-v4-20261112',
                  experiment_id='aihub71762-category-random-v4-20261112',
                  output_dir=str(root / 'v4/runs' / run_id), training_schema_version='medvlm.training.v2',
                  training_recipe='one_pass', mode='smoke' if phase == 'smoke' else 'v2',
                  train_file=data[name+'train']['snapshot_path'],
                  validation_file=data[name+'validation']['snapshot_path'],
                  freeze_manifest=corpus['freeze_manifest'], train_samples_cap=4000,
                  max_optimizer_steps=20 if phase == 'smoke' else 250,
                  validation_samples=48, checkpoint_every_steps=100, max_length=2048,
                  max_epochs=1, seed=20261023, resource_policy=resolve_resource_policy().as_dict(),
                  parent_condition='original_base', initial_adapter=None)
    return config


def prepare(root):
    out = root / 'v4'
    corpus = read(out / 'corpus.json')
    if corpus['status'] != 'SUCCEEDED' or corpus['splits']['aihub71762_category_random_v4/train']['count'] != 4000:
        raise ValueError('SCREENED_FROZEN_4000_ROW_CORPUS_REQUIRED')
    if file_hash(corpus['freeze_manifest']) != corpus['freeze_manifest_sha256']:
        raise ValueError('CORPUS_MANIFEST_CHANGED')
    configs = out / 'configs'
    configs.mkdir(exist_ok=True)
    entries = {}
    for model in ('qwen3_17b', 'qwen3_4b'):
        original = read(root / f'runs/{model}-v1-20261023/config.json')
        for phase in ('smoke', 'full'):
            config = training_config(original, corpus, root, model, phase)
            path = configs / (config['run_id'] + '.json')
            immutable(path, config)
            entries[config['run_id']] = {'model': model, 'phase': phase, 'config': str(path),
                                        'config_sha256': file_hash(path),
                                        'supervised_tokens_full_pass': corpus['token_counts_by_model'][model]['train']['supervised_tokens']}
    selection = freeze_protocol([corpus['splits']['aihub71762_category_random_v4/test']['snapshot_path']],
                                out / 'random-heldout-selection.json', cap=0)
    selection = {k: v for k, v in selection.items() if k != 'protocol_hash'}
    selection['max_new_tokens'] = {**selection['max_new_tokens'], 'medical_qa': 512}
    selection['protocol_hash'] = canonical_hash(selection)
    protocol = promote_protocol(selection, out / 'random-heldout-test.json',
                                prompt_policy='native-task-instructions.v1',
                                validation_evidence={'policy': '512-token diagnostic budget fixed before predictions; reference is category-random, not a verified per-question gold answer'},
                                previously_observed_test=False)
    common_protocol, common = load_protocol(root / 'v2/text/text-test.json')
    plan = {'schema': 'medvlm.aihub71762.campaign.v4', 'corpus_sha256': file_hash(out / 'corpus.json'),
            'runner_sha256': file_hash(__file__), 'runs': entries, 'models': ['qwen3_17b', 'qwen3_4b'],
            'sample_count': 4000, 'effective_batch': 16, 'full_pass_updates': 250,
            'smoke_updates': 20, 'epochs': 1, 'learning_rate': .0002,
            'lora': {'rank': 8, 'alpha': 16, 'dropout': .05},
            'seed': 20261023, 'max_train_length': 2048, 'max_eval_input_length': 4096,
            'resource_policy': resolve_resource_policy().as_dict(),
            'random_protocol_sha256': file_hash(out / 'random-heldout-test.json'),
            'random_heldout_count': len(protocol['sample_ids']), 'random_max_new_tokens': 512,
            'common_protocol_sha256': file_hash(root / 'v2/text/text-test.json'),
            'common_heldout_count': len(common), 'common_generation_budgets': common_protocol['max_new_tokens'],
            'checkpoint_selection': '48 deterministically selected validation rows; no test selection',
            'original_base_independent_sft': True, 'curriculum_on_existing_adapter': False,
            'evaluation_cells': ['each model random-heldout base and tuned', 'each model common-v2 tuned'],
            'common_base': 'reuse original verified v2 base evaluations with identical config/protocol',
            'additional_seeds': 'not_selected',
            'limitations': ['Category-random targets are not verified per-question gold answers.',
                           'Augmentation parents unavailable; split lineage guarantees impossible.',
                           'Sampled corpus, one seed, no clinical reviewer.',
                           'Common external benchmarks were previously observed in v1/v2.']}
    immutable(out / 'plan.json', plan)
    return plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--phase', choices=('prepare', 'train', 'evaluate', 'all'), default='all')
    args = parser.parse_args()
    root = args.data_root.resolve()
    out = root / 'v4'
    (root / 'runs').mkdir(exist_ok=True)
    with (out / 'campaign.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = prepare(root)
        path = out / 'state.json'
        state = read(path) if path.exists() else {'plan_sha256': canonical_hash(plan), 'training': {}, 'evaluation': {}, 'errors': {}}
        if state['plan_sha256'] != canonical_hash(plan):
            raise ValueError('CAMPAIGN_IDENTITY_CHANGED')
        def save():
            state['updated_at'] = utcnow()
            atomic_json(path, state)
        def failure(stage, error):
            state['errors'][stage] = {'type': type(error).__name__, 'message': str(error)}
            print(json.dumps({'failed': stage, 'error': type(error).__name__}), flush=True)
            save()
        save()
        if args.phase in ('all', 'train'):
            for model in plan['models']:
                for phase in ('smoke', 'full'):
                    run_id = f'{model}-category-random-{phase}-v4-20261112'
                    entry = plan['runs'][run_id]
                    state['active'] = run_id
                    save()
                    try:
                        if not (train_stage(entry['config'], root) and reload_stage(entry['config'], root)):
                            raise RuntimeError('TRAIN_OR_FRESH_PROCESS_RELOAD_FAILED')
                        config = read(entry['config'])
                        manifest, selection = verify_training_cache(config)
                        verify_reload_cache(config, manifest, selection)
                        state['training'][run_id] = {'status': 'SUCCEEDED', 'optimizer_steps': manifest['optimizer_steps'],
                            'samples_processed': manifest['samples_processed'], 'supervised_tokens': manifest['supervised_tokens'],
                            'reload_verified': True, 'selected_checkpoint': selection['checkpoint']}
                        state['errors'].pop(run_id, None)
                        print(json.dumps({'training_finished': run_id, **state['training'][run_id]}), flush=True)
                        save()
                    except Exception as error:
                        failure(run_id, error)
                        break
        if args.phase in ('all', 'evaluate'):
            for model in plan['models']:
                config_path = root / f'v2/text/configs/{model}-evaluation.json'
                full_id = f'{model}-category-random-full-v4-20261112'
                adapter = None
                try:
                    config = read(plan['runs'][full_id]['config'])
                    manifest, selection = verify_training_cache(config)
                    verify_reload_cache(config, manifest, selection)
                    adapter = selection['checkpoint']
                except Exception as error:
                    failure(model + '-adapter', error)
                outputs = {}
                for dataset, conditions in (('random', ('base', 'tuned')), ('common', ('tuned',))):
                    protocol_path = out / 'random-heldout-test.json' if dataset == 'random' else root / 'v2/text/text-test.json'
                    pair_id = f'{model}-category-random-v4-20261112' if dataset == 'random' else f'{model}-text-v2-20261026'
                    for condition in conditions:
                        if condition == 'tuned' and adapter is None:
                            continue
                        key = f'{model}-{dataset}-{condition}-v4-20261112'
                        destination = out / 'evaluations' / key
                        state['active'] = key
                        save()
                        argv = [sys.executable, '-m', 'mediflow_medvlm.evaluation', 'evaluate', '--config', str(config_path),
                                '--protocol', str(protocol_path), '--output-dir', str(destination), '--pair-id', pair_id]
                        if condition == 'tuned':
                            argv += ['--adapter', adapter]
                        try:
                            with (out / (key + '.log')).open('a') as log:
                                process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
                                state['active_pid'] = process.pid
                                save()
                                print(json.dumps({'evaluation_started': key, 'pid': process.pid}), flush=True)
                                code = process.wait()
                            manifest = read(destination / 'run_manifest.json')
                            if code or manifest['status'] != 'SUCCEEDED':
                                raise RuntimeError('EVALUATION_FAILED')
                            state['evaluation'][key] = {'status': 'SUCCEEDED', 'output_dir': str(destination),
                                                       'predictions': manifest['predictions']}
                            outputs[(dataset, condition)] = destination
                            state['errors'].pop(key, None)
                            print(json.dumps({'evaluation_finished': key, **state['evaluation'][key]}), flush=True)
                            save()
                        except Exception as error:
                            failure(key, error)
                    if (dataset, 'tuned') in outputs:
                        base = outputs.get((dataset, 'base')) if dataset == 'random' else root / f'v2/text/evaluations/{model}-base-text-v2-20261026'
                        if base:
                            try:
                                compare(base, outputs[(dataset, 'tuned')], out / f'{model}-{dataset}-base-vs-tuned.json')
                                state['errors'].pop(model + '-' + dataset + '-comparison', None)
                            except Exception as error:
                                failure(model + '-' + dataset + '-comparison', error)
        state['active'] = None
        state['active_pid'] = None
        state['status'] = ('SUCCEEDED' if len(state['training']) == 4 and len(state['evaluation']) == 6
                           and not state['errors'] else 'PREPARED_NOT_RUN' if args.phase == 'prepare' else 'PARTIAL')
        save()
        print(json.dumps({'status': state['status'], 'training': len(state['training']), 'evaluations': len(state['evaluation'])}), flush=True)


if __name__ == '__main__':
    main()
