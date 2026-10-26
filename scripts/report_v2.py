#!/usr/bin/env python3
"""Read-only v2 evidence report. No inference, training, raw-text export or v1 writes."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import io
import json
import math
from pathlib import Path
import statistics
import time

from mediflow_medvlm.runtime import canonical_hash, directory_digest, file_hash, resolve_resource_policy, utcnow
from mediflow_medvlm.evaluation.runner import load_protocol, validated_predictions, verify_completed_evaluation
from mediflow_medvlm.evaluation.metrics import aggregate, korean_summary, paired_compare, score_rows
from mediflow_medvlm.evaluation.v2 import normalize_vqa

REPO = Path(__file__).resolve().parents[1]
VLM = ('qwen35_2b', 'medgemma15_4b')
TEXT = ('qwen3_17b', 'qwen3_4b')
SUFFIX = 'v2-20261026'
PUBLIC_FILES = ('RESULTS_KO.md', 'stage_status.json', 'run_summary.json', 'comparison.csv', 'metrics.csv', 'verification.json')
STATUS = {'SUCCEEDED', 'RUNNING', 'FAILED_REPRODUCIBLE', 'SKIPPED_BUDGET', 'PREPARED_NOT_RUN',
          'NOT_PREPARED', 'NOT_RUN', 'FAILED_VERIFICATION', 'PARTIAL_WITH_FAILURES', 'VALIDATION_COMPLETE',
          'TRAINED_AWAITING_RELOAD'}


def read(path, default=None):
    p = Path(path)
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else default


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def safe_status(value, default='NOT_RUN'):
    return value if value in STATUS else default


def distribution(values):
    values = sorted(v for v in values if isinstance(v, (int, float)) and math.isfinite(v))
    return {'n': len(values), 'mean': statistics.mean(values) if values else None,
            'median': statistics.median(values) if values else None,
            'p95': values[max(0, math.ceil(.95 * len(values)) - 1)] if values else None,
            'max': max(values) if values else None}


def aggregate_safe(rows):
    # Scores contain references internally; only aggregate metrics cross the public boundary.
    return [{k: v for k, v in item.items() if k != 'label_support'} for item in aggregate(rows)]


def metric_bundle(samples, predictions):
    strict = score_rows(samples, predictions)
    normalized = score_rows(samples, predictions, vqa_normalizer=normalize_vqa)
    cohorts = aggregate_safe(normalized)
    korean = korean_summary(cohorts)
    if korean:
        subset = [r for r in normalized if r['source_id'] == 'kormedmcqa']
        korean['invalid_format_rate'] = sum(r['status'] == 'invalid_format' for r in subset) / len(subset)
        korean['all_invalid_or_unanswered_rate'] = 1 - korean['coverage']
        korean['subject_metrics'] = [c for c in cohorts if c['source_id'] == 'kormedmcqa']
    by_id = {p['sample_id']: p for p in predictions}
    score_by_id = {r['sample_id']: r for r in normalized}
    strict_by_id = {r['sample_id']: r for r in strict}
    vqa = []
    for source in sorted({s['source_id'] for s in samples if s['task_type'] == 'medical_vqa'}):
        for group in ('ALL', 'OPEN', 'CLOSED'):
            selected = [s for s in samples if s['source_id'] == source and
                        (group == 'ALL' or str(s.get('metadata', {}).get('answer_type', '')).upper() == group)]
            if not selected:
                continue
            ids = [s['sample_id'] for s in selected]; pp = [by_id[sid] for sid in ids]
            rr = [score_by_id[sid] for sid in ids]; ss = [strict_by_id[sid] for sid in ids]
            vqa.append({'source_id': source, 'answer_type': group, 'scheduled': len(ids),
                        'observed_group_count': len({r['group_id'] for r in rr}),
                        'strict_lexical_em': sum(r['exact_match'] for r in ss) / len(ids),
                        'strict_lexical_token_f1': sum(r['token_f1'] for r in ss) / len(ids),
                        'v2_normalized_lexical_em': sum(r['exact_match'] for r in rr) / len(ids),
                        'v2_normalized_lexical_token_f1': sum(r['token_f1'] for r in rr) / len(ids),
                        'valid_answers': sum(r['valid'] for r in rr),
                        'truncated_count': sum(p.get('finish_reason') == 'length' for p in pp),
                        'output_characters': distribution([len(p.get('raw_output') or '') for p in pp]),
                        'output_whitespace_words': distribution([len((p.get('raw_output') or '').split()) for p in pp]),
                        'generated_tokens': distribution([p.get('output_tokens') for p in pp]),
                        'e2e_latency_seconds': distribution([r['latency_seconds'] for r in rr]),
                        'vision_ingested_count': sum(p.get('vision_ingested') is True for p in pp),
                        'pixel_tensor_evidence_count': sum(bool(p.get('pixel_tensor_keys')) for p in pp),
                        'clinical_accuracy': None})
    aihub = []
    for q_type in ('1', '2', '3'):
        ids = {s['sample_id'] for s in samples if s['source_id'] == 'aihub_71875' and
               str(s.get('metadata', {}).get('q_type')) == q_type}
        if ids:
            for item in aggregate_safe([r for r in normalized if r['sample_id'] in ids]):
                aihub.append({'q_type': q_type, 'type': {'1': 'MCQ', '2': 'short', '3': 'long'}[q_type], **item})
    return {'cohorts': cohorts, 'kormedmcqa': korean, 'aihub_q_types': aihub, 'vqa': vqa,
            'e2e_latency_seconds': distribution([r['latency_seconds'] for r in normalized]),
            'clinical_accuracy': None, 'human_review': 'NOT_PERFORMED'}, normalized, strict


class Evidence:
    """Cache repeated reads of immutable protocols/adapters, without rehashing base weights."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.protocols = {}
        self.adapters = {}
        self.receipts = []
        self.errors = []
        self.private = {}

    def artifact(self, path):
        path = Path(path).resolve()
        if path.is_relative_to(self.root):
            return '${MEDVLM_DATA_ROOT}/' + str(path.relative_to(self.root))
        if path.is_relative_to(REPO):
            return str(path.relative_to(REPO))
        return '<external-private-artifact>'

    def receipt(self, path, kind):
        value = {'kind': kind, 'artifact': self.artifact(path), 'sha256': file_hash(path)}
        self.receipts.append(value)
        return value['sha256']

    def protocol(self, path):
        key = str(Path(path).resolve()), file_hash(path)
        if key not in self.protocols:
            self.protocols[key] = load_protocol(path)
            self.receipt(path, 'frozen_protocol_inputs_references_images_verified')
        return self.protocols[key]

    def adapter(self, path):
        key = str(Path(path).resolve())
        if key not in self.adapters:
            self.adapters[key] = directory_digest(path)['sha256']
        return self.adapters[key]

    def evaluation(self, model, condition, phase, folder, protocol_path, config_path, *, expected_count,
                   expected_sources=None, state=None, selected_training=None):
        run_id = folder.name
        cell = {'run_id': run_id, 'model': model, 'condition': condition, 'phase': phase,
                'scheduled_planned': expected_count, 'status': 'NOT_RUN', 'verified': False}
        state = state or {}
        entries = state.get('runs', state.get('evaluation', {}))
        state_cell = entries.get(run_id, {})
        cell['status'] = 'RUNNING' if state.get('active') == run_id else safe_status(state_cell.get('status'))
        manifest_path = folder / 'run_manifest.json'
        manifest = read(manifest_path)
        if not manifest:
            return cell
        cell['status'] = safe_status(manifest.get('status'))
        cell['manifest_sha256'] = self.receipt(manifest_path, 'evaluation_manifest_snapshot')
        if cell['status'] != 'SUCCEEDED':
            # An in-flight file can change. Never publish interim predictions as completed metrics.
            cell['metrics'] = None
            return cell
        try:
            before = file_hash(manifest_path)
            config = read(config_path)
            protocol, samples = self.protocol(protocol_path)
            require(manifest['run_id'] == run_id, 'RUN_ID_MISMATCH')
            require(manifest['config_hash'] == canonical_hash(config), 'CONFIG_IDENTITY_MISMATCH')
            require(manifest['protocol_hash'] == protocol['protocol_hash'], 'PROTOCOL_IDENTITY_MISMATCH')
            require(manifest['model_id'] == config['model_id'] and manifest['model_revision'] == config['revision'], 'MODEL_IDENTITY_MISMATCH')
            require(len(samples) == expected_count == manifest['scheduled'], 'SCHEDULE_SIZE_MISMATCH')
            if expected_sources:
                require(dict(Counter(s['source_id'] for s in samples)) == expected_sources, 'SOURCE_SCHEDULE_MISMATCH')
            origin_path = self.root / 'runs' / f'{model}-v1-20261023' / 'manifest.json'
            origin = read(origin_path)
            native = manifest['model']['native_files']['sha256']
            require(native == origin['model']['native_files']['sha256'], 'RECORDED_NATIVE_MODEL_MISMATCH')
            require(manifest['model']['chat_template_sha256'] == origin['model']['chat_template_sha256'], 'CHAT_TEMPLATE_MISMATCH')
            require(manifest['model_id'] == origin['model_id'] and manifest['model_revision'] == origin['revision'], 'ORIGINAL_MODEL_REVISION_MISMATCH')
            keys = ('config_hash', 'protocol_hash', 'model_id', 'model_revision', 'adapter_path',
                    'pair_id', 'renderer_version', 'image_condition')
            identity = {key: manifest[key] for key in keys}
            if manifest.get('adapter_path'):
                require(condition not in ('base', 'base_text_only'), 'UNEXPECTED_BASE_ADAPTER')
                identity['adapter_digest'] = manifest['adapter_digest']
                require(self.adapter(manifest['adapter_path']) == manifest['adapter_digest'], 'ADAPTER_BYTES_MISMATCH')
                train_dir = selected_training or self.root / 'runs' / f'{model}-v1-20261023'
                selection = read(train_dir / 'selection.json')
                require(Path(selection['checkpoint']).resolve() == Path(manifest['adapter_path']).resolve(), 'SELECTED_ADAPTER_MISMATCH')
                require(selection['adapter']['digest'] == manifest['adapter_digest'] and selection['reload_verified'], 'ADAPTER_SELECTION_RECEIPT_MISMATCH')
                self.receipt(train_dir / 'selection.json', 'selected_adapter_link')
            else:
                require(condition in ('base', 'base_text_only'), 'MISSING_TUNED_ADAPTER')
            require(canonical_hash(identity) == manifest['identity_hash'], 'RUN_IDENTITY_MISMATCH')
            verify_completed_evaluation(manifest, protocol, folder, native)
            predictions = validated_predictions(folder / 'predictions.jsonl', protocol, manifest)
            require(len(predictions) == expected_count == manifest['predictions'], 'PREDICTION_COUNT_MISMATCH')
            if phase == 'noimage':
                require(manifest['image_condition'] == 'no_image', 'NO_IMAGE_PROTOCOL_MISMATCH')
                require(all(not p.get('vision_ingested') and not p.get('pixel_tensor_keys') for p in predictions), 'NO_IMAGE_PIXEL_EVIDENCE')
            elif phase == 'text':
                require(all(s['task_type'] != 'medical_vqa' and not s['input'].get('images') for s in samples), 'TEXT_PROTOCOL_HAS_IMAGES')
                require(all(p.get('vision_ingested') is False and not p.get('pixel_tensor_keys') for p in predictions), 'TEXT_RUN_PIXEL_EVIDENCE')
            elif samples[0]['task_type'] == 'medical_vqa':
                require(manifest['image_condition'] == 'image', 'IMAGE_PROTOCOL_MISMATCH')
                require(all(p.get('vision_ingested') and p.get('pixel_tensor_keys') for p in predictions
                            if p.get('status') == 'answered'), 'MISSING_NATIVE_PIXEL_EVIDENCE')
            bundle, normalized, strict = metric_bundle(samples, predictions)
            stored = read(folder / 'metrics.json')
            require(canonical_hash(stored['scores']) == canonical_hash(normalized), 'STORED_METRICS_RECOMPUTE_MISMATCH')
            require(before == file_hash(manifest_path), 'MANIFEST_CHANGED_DURING_READ')
            cell.update(status='SUCCEEDED', verified=True, scheduled=len(samples), metrics=bundle,
                        protocol_hash=protocol['protocol_hash'], identity_hash=manifest['identity_hash'],
                        model_revision=manifest['model_revision'], native_model_recorded_digest=native,
                        adapter_digest=manifest.get('adapter_digest'), precision=manifest['precision'],
                        generation=manifest['generation'], image_condition=manifest['image_condition'],
                        peak_allocated_bytes=manifest.get('peak_allocated_bytes'), peak_reserved_bytes=manifest.get('peak_reserved_bytes'),
                        wall_seconds=manifest.get('wall_seconds'), load_seconds=manifest.get('load_seconds'),
                        recorded_evaluation_code_sha256=manifest.get('code', {}).get('source_tree_sha256'))
            if phase == 'text':
                cell['text_modality_evidence'] = {'verified_predictions': len(predictions),
                    'vision_ingested_records': 0, 'native_pixel_tensor_records': 0,
                    'source_inputs_have_images': False}
            self.receipt(config_path, 'evaluation_config_verified')
            self.receipt(origin_path, 'native_model_recorded_training_identity')
            for name in ('predictions.jsonl', 'metrics.json'):
                self.receipt(folder / name, 'completed_evaluation_artifact_verified')
            self.private[run_id] = (manifest, protocol, samples, predictions, normalized, strict)
        except (OSError, ValueError, KeyError, TypeError, AssertionError) as error:
            cell.update(status='FAILED_VERIFICATION', metrics=None, verification_error=type(error).__name__)
            self.errors.append({'run_id': run_id, 'error_type': type(error).__name__,
                                'reason_code': str(error) if str(error).isupper() and ' ' not in str(error) else 'ARTIFACT_VERIFICATION_FAILED'})
        return cell

    def training(self, model, condition, phase, state):
        run_id = f'{model}-{condition}-{phase}-{SUFFIX}'
        folder = self.root / 'v2/text/runs' / run_id
        entry = state.get('training', {}).get(run_id, {})
        result = {'run_id': run_id, 'model': model, 'condition': condition, 'phase': phase,
                  'planned_train_count': 4000 if condition == 'ko' else 8000,
                  'planned_optimizer_steps': 20 if phase == 'smoke' else 250 if condition == 'ko' else 500,
                  'status': 'RUNNING' if state.get('active') == run_id else safe_status(entry.get('status')),
                  'verified': False, 'reload_verified': False}
        manifest_path = folder / 'manifest.json'
        manifest = read(manifest_path)
        if not manifest:
            return result
        result['status'] = safe_status(manifest.get('status'))
        for key in ('optimizer_steps', 'samples_processed', 'supervised_tokens', 'actual_epochs', 'gpu_seconds'):
            result[key] = manifest.get(key)
        if result['status'] != 'SUCCEEDED':
            return result
        reload_status = read(folder / 'reload.json', {}).get('status')
        if reload_status not in (None, 'RUNNING', 'SUCCEEDED'):
            result['status'] = safe_status(reload_status, 'FAILED_REPRODUCIBLE')
            return result
        if reload_status in (None, 'RUNNING') or not manifest.get('reload_verified'):
            result['status'] = 'TRAINED_AWAITING_RELOAD'
            return result
        try:
            config_path = folder / 'config.json'
            config = read(config_path)
            require(canonical_hash(config) == manifest['config_hash'], 'TRAIN_CONFIG_MISMATCH')
            plan = read(self.root / 'v2/text/plan.json', {})
            if run_id in plan.get('runs', {}):
                declaration = plan['runs'][run_id]
                require(file_hash(declaration['config']) == declaration['config_sha256'] and
                        canonical_hash(read(declaration['config'])) == manifest['config_hash'], 'TRAIN_PLAN_CONFIG_MISMATCH')
            require(file_hash(config['freeze_manifest']) == manifest['data']['freeze_manifest_sha256'], 'TRAIN_FREEZE_MISMATCH')
            for split in ('train', 'validation'):
                require(file_hash(config[split + '_file']) == manifest['data'][split]['sha256'], 'TRAIN_DATA_MISMATCH')
            require(manifest['model_id'] == config['model_id'] and manifest['revision'] == config['revision'], 'TRAIN_MODEL_MISMATCH')
            selection = read(folder / 'selection.json'); reload = read(folder / 'reload.json')
            require(selection['run_id'] == run_id and selection['identity'] == manifest['identity'], 'TRAIN_SELECTION_IDENTITY_MISMATCH')
            require(selection['test_used'] is False and manifest['test_used_for_selection'] is False, 'TEST_USED_FOR_SELECTION')
            require(selection['adapter']['digest'] == self.adapter(selection['checkpoint']), 'TRAIN_ADAPTER_BYTES_MISMATCH')
            require(reload['status'] == 'SUCCEEDED' and reload['fresh_process_required'] is True and
                    reload['checkpoint'] == selection['checkpoint'] and selection['reload_verified'] and manifest['reload_verified'], 'RELOAD_NOT_VERIFIED')
            require(reload['model']['native_files']['sha256'] == manifest['model']['native_files']['sha256'], 'RELOAD_NATIVE_IDENTITY_MISMATCH')
            cp = read(Path(selection['checkpoint']) / 'checkpoint.json')
            require(cp['identity'] == manifest['identity'], 'CHECKPOINT_IDENTITY_MISMATCH')
            result.update(verified=True, reload_verified=True, adapter_digest=selection['adapter']['digest'],
                          native_model_recorded_digest=manifest['model']['native_files']['sha256'],
                          selected_checkpoint_steps=cp.get('optimizer_steps'),
                          selected_checkpoint_samples=cp.get('samples_processed'),
                          selected_checkpoint_supervised_tokens=cp.get('supervised_tokens'),
                          selected_checkpoint_validation_loss=cp.get('best_validation_loss'),
                          config_hash=manifest['config_hash'], freeze_manifest_sha256=manifest['data']['freeze_manifest_sha256'])
            exclusions = read(folder / 'excluded.json', [])
            result['preflight_excluded_count'] = len(exclusions)
            result['preflight_exclusion_reasons'] = dict(Counter(item['reason'] for item in exclusions))
            for name in ('config.json', 'manifest.json', 'selection.json', 'reload.json'):
                self.receipt(folder / name, 'training_reload_artifact_verified')
        except (OSError, ValueError, KeyError, TypeError) as error:
            result.update(status='FAILED_VERIFICATION', verification_error=type(error).__name__)
            self.errors.append({'run_id': run_id, 'error_type': type(error).__name__, 'reason_code': 'TRAINING_VERIFICATION_FAILED'})
        return result


def comparisons(evidence, cells):
    result = []
    lookup = {(c['model'], c['phase'], c['condition']): c for c in cells if c.get('verified')}
    for model in VLM + TEXT:
        phases = ('slake', 'vqarad', 'noimage') if model in VLM else ('text',)
        for phase in phases:
            base = lookup.get((model, phase, 'base'))
            if not base:
                continue
            for condition in (('tuned_v1',) if model in VLM else ('en_v1', 'ko', 'mix')):
                target = lookup.get((model, phase, condition))
                if not target:
                    continue
                left = evidence.private[base['run_id']]; right = evidence.private[target['run_id']]
                if any(left[0][k] != right[0][k] for k in ('protocol_hash', 'config_hash', 'model_id', 'model_revision', 'precision', 'generation', 'pair_id')):
                    evidence.errors.append({'run_id': target['run_id'], 'reason_code': 'COMPARISON_CONDITIONS_MISMATCH'})
                    continue
                for metric_name, index in (('v2_normalized', 4), ('strict', 5)):
                    if phase == 'text' and metric_name == 'strict':
                        continue
                    for item in paired_compare(left[index], right[index], resamples=2000, seed=20261023):
                        result.append({'model': model, 'phase': phase, 'left': 'base', 'right': condition,
                                       'normalization': metric_name, **item})
    # Same selected IDs, prompt, generation and model/adapter; image absence is diagnostic only.
    for model in VLM:
        for condition in ('base', 'tuned_v1'):
            image = lookup.get((model, 'slake', condition)); no_image = lookup.get((model, 'noimage', condition))
            if not image or not no_image:
                continue
            a = evidence.private[image['run_id']]; b = evidence.private[no_image['run_id']]
            parent = b[1].get('parent_protocol_hash')
            if parent != a[1]['protocol_hash'] or any(a[0].get(k) != b[0].get(k) for k in ('model_id', 'model_revision', 'adapter_digest', 'precision', 'generation')):
                evidence.errors.append({'run_id': no_image['run_id'], 'reason_code': 'IMAGE_ABLATION_IDENTITY_MISMATCH'})
                continue
            ids = {r['sample_id'] for r in b[4]}
            aa = [r for r in a[4] if r['sample_id'] in ids]
            for item in paired_compare(aa, b[4], resamples=2000, seed=20261023):
                result.append({'model': model, 'phase': 'image_diagnostic', 'left': condition + '_image',
                               'right': condition + '_no_image', 'normalization': 'v2_normalized',
                               'diagnostic_not_training_data': True, **item})
    return result


def data_summary(root, evidence):
    corpus_path = root / 'v2/manifests/text-corpora-v2.json'
    corpus = read(corpus_path)
    result = {'text_corpora': {'status': 'NOT_PREPARED'}}
    if corpus:
        evidence.receipt(corpus_path, 'text_corpus_manifest')
        result['text_corpora'] = {'status': 'SUCCEEDED', 'scope': 'corpus preparation only; training is reported separately',
                                 'seed': corpus['seed'], 'max_length': corpus['max_length'], 'corpora': {}}
        for condition, item in corpus['corpora'].items():
            freeze = read(item['freeze_manifest'])
            require(canonical_hash({k:v for k,v in freeze.items() if k != 'freeze_id'}) == freeze['freeze_id'], 'CORPUS_FREEZE_ID_MISMATCH')
            for split in ('train', 'validation'):
                records = jsonl(item[split + '_file'])
                matched = [v for k,v in freeze['splits'].items() if k.endswith('/' + split)]
                require(len(matched) == 1 and file_hash(item[split + '_file']) == matched[0]['sha256'], 'CORPUS_FROZEN_BYTES_MISMATCH')
                require(len(records) == item[split + '_count'] == matched[0]['count'], 'CORPUS_FROZEN_COUNT_MISMATCH')
                require([r['sample_id'] for r in records] == matched[0]['ordered_sample_ids'], 'CORPUS_FROZEN_IDS_MISMATCH')
            public = {k: item[k] for k in ('train_count', 'validation_count', 'supervised_tokens', 'optimizer_steps_at_batch16') if k in item}
            public['artifacts'] = {key: evidence.receipt(item[key], 'text_corpus_' + key)
                                   for key in ('train_file', 'validation_file', 'freeze_manifest')}
            result['text_corpora']['corpora'][condition] = public
        result['text_corpora']['identical_common_validation'] = file_hash(corpus['corpora']['ko']['validation_file']) == file_hash(corpus['corpora']['mix']['validation_file'])
        require(result['text_corpora']['identical_common_validation'], 'KO_MIX_VALIDATION_DIFFER')
        audit_path = Path(corpus['source_audit_path'])
        selection = read(corpus['selection_manifest_path'])
        result['text_corpora']['selection'] = {k: selection[k] for k in ('common_validation_count', 'ko_validation_count',
            'corpus_selection_used_test_performance', 'semantic_quarantine_rows', 'semantic_quarantine_groups',
            'full_aihub_heldout_preserved', 'pre_freeze_decision_history') if k in selection}
        for key in ('ko', 'en', 'en_validation'):
            result['text_corpora']['selection'][key + '_selected_counts'] = selection[key]['selected_counts']
        for field in ('token_audit_path', 'semantic_audit_path', 'selection_manifest_path', 'source_import_path'):
            if corpus.get(field) and Path(corpus[field]).is_file():
                evidence.receipt(corpus[field], field)
    else:
        candidates = sorted((root / 'v2/prepared/aihub_71875').glob('*/audit.json'), key=lambda p: p.stat().st_mtime)
        audit_path = candidates[-1] if candidates else None
    if audit_path and audit_path.exists():
        audit = read(audit_path)
        evidence.receipt(audit_path, 'aihub_actual_source_audit')
        result['aihub_71875'] = {k: audit[k] for k in ('status', 'source_id', 'source_version', 'canonical_counts',
            'counts', 'excluded_counts', 'q_type_split_counts', 'original_official_test_supplied',
            'per_record_human_review', 'patient_group_verified') if k in audit}
        result['aihub_71875']['counts_are_candidate_pool'] = not bool(corpus)
        result['aihub_71875']['canonical_before_semantic_quarantine'] = True
        result['aihub_71875']['lexical_contamination_screen'] = {k:audit['leakage_audit'][k] for k in
            ('protected_count','confirmed_overlap_rows','near_candidate_rows','near_threshold','translation_semantic_equivalence_verified')}
    semantic = read(root / 'v2/manifests/semantic_kormed_v2/audit.json')
    progress = read(root / 'v2/manifests/semantic_kormed_v2/progress.json', {})
    result['semantic_audit'] = {'status': semantic.get('status', 'RECORDED') if semantic else progress.get('status', 'NOT_COMPLETED')}
    if semantic:
        evidence.receipt(root / 'v2/manifests/semantic_kormed_v2/audit.json', 'semantic_audit')
        result['semantic_audit'].update({k:semantic[k] for k in ('threshold', 'top_k', 'protected_count',
            'candidate_count', 'candidate_rows', 'semantic_equivalence_verified', 'translation_equivalence_verified', 'gpu_used')})
        result['semantic_audit']['candidate_pair_count'] = len(semantic['candidate_pairs'])
        candidate_ids = {p['sample_id'] for p in semantic['candidate_pairs']}
        if audit_path:
            split_counts = {}
            for split, item in audit['canonical_files'].items():
                require(file_hash(item['path']) == item['sha256'], 'SOURCE_POOL_BYTES_MISMATCH')
                rows = jsonl(item['path'])
                flagged = sum(r['sample_id'] in candidate_ids for r in rows)
                split_counts[split] = {'canonical': len(rows), 'semantic_candidate_rows': flagged,
                    'eligible_after_quarantine': len(rows) if split == 'test' else len(rows) - flagged,
                    'test_preserved_candidates_only_flagged': split == 'test'}
            result['semantic_audit']['source_split_counts'] = split_counts
    med = read(REPO / 'reports/public_summary/v2/medmcqa_source.json')
    if med:
        result['medmcqa'] = {'status': 'PREPARED_NOT_RUN', 'selected_for_ko_mix': med['core_ko_mix_included'],
                            'canonical_counts': {k:v['count'] for k,v in med['canonical_files'].items()},
                            'evidence_sha256': evidence.receipt(REPO / 'reports/public_summary/v2/medmcqa_source.json', 'optional_medmcqa_source')}
    blocked_path = root / 'v2/manifests/input-status-20261026.json'
    blocked = read(blocked_path)
    result['aihub_71762'] = {'status': 'BLOCKED_INPUT', 'actual_acquired_bytes': 0,
                           'reason_code': 'CONNECTOR_SIZE_LIMIT_AND_BROWSER_BLOCK', 'local_path_pending': True}
    if blocked:
        result['aihub_71762']['evidence_sha256'] = evidence.receipt(blocked_path, 'input_availability_receipt')
        source = blocked['sources']['aihub_71762']
        result['aihub_71762'].update({k:source[k] for k in ('status', 'expected_bytes', 'actual_acquired_bytes', 'reader_validation', 'pairing_verified', 'platform_block_bypassed')})
    else:
        result['aihub_71762']['status'] = 'BLOCKED_INPUT_RECORDED_IN_HANDOFF'
    result['jetson'] = {'status': 'USER_DEFERRED', 'attempted_in_v2': False}
    handoff = REPO / 'docs/agent/HANDOFF.md'
    if handoff.exists():
        result['jetson']['evidence_sha256'] = evidence.receipt(handoff, 'user_deferred_handoff')
    return result


def usage_summary(root, evidence):
    path = root / 'runtime/gpu_budget.json'
    ledger = read(path, {})
    previous_path = REPO / 'reports/public_summary/run_summary.json'
    previous = read(previous_path, {})
    old = previous.get('budget', {}).get('models', {})
    completed = ledger.get('models', {})
    active = ledger.get('active')
    result = {'policy': resolve_resource_policy().as_dict(), 'completed_accounted_seconds': sum(completed.values()),
              'completed_by_model_seconds': completed, 'v1_baseline_seconds': sum(old.values()) if old else None,
              'v1_baseline_by_model_seconds': old, 'completed_delta_from_v1_seconds': sum(completed.values()) - sum(old.values()) if old else None,
              'active_unsettled_elapsed_seconds': max(0, time.time() - active['started_epoch']) if active else 0,
              'active_session_included_in_completed': False,
              'measurement': 'wall-clock duration of actual GPU ledger sessions; not device utilization or energy',
              'v1_historical_limits_hours': {'total': 24, 'per_model': 6}}
    phases = defaultdict(float)
    for run, values in ledger.get('run_phases', {}).items():
        if '-v2-' in run:
            for phase, seconds in values.items():
                phases[phase] += seconds
    result['v2_completed_by_phase_seconds'] = dict(phases)
    if path.exists():
        evidence.receipt(path, 'readonly_usage_ledger_snapshot')
    if previous_path.exists():
        evidence.receipt(previous_path, 'unchanged_v1_usage_baseline')
    return result


def review_summary(root, evidence):
    """Verify four local unscored packets; export counts and hashes, never review text/key."""
    from mediflow_medvlm.evaluation.expert_review import REVIEW_FIELDS, _review_runs
    packets = []
    plan = root / 'v2/review/plan.json'
    if plan.exists():
        evidence.receipt(plan, 'review_preparation_plan')
    for model in TEXT:
        for condition in ('ko', 'mix'):
            packet_id = f'{model}-base-vs-{condition}'
            folder = root / 'v2/review' / packet_id
            path = folder / 'manifest.json'
            item = {'packet_id': packet_id, 'model': model, 'comparison': 'base-vs-' + condition,
                    'status': 'NOT_PREPARED', 'verified': False, 'prepared_count': 0,
                    'human_review': 'not_performed', 'human_review_count': 0, 'clinical_accuracy': None}
            packets.append(item)
            if not path.exists():
                continue
            try:
                manifest_sha = file_hash(path); manifest = read(path)
                require(manifest['review_version'] == 'medvlm.aihub-expert-review-packet.v2' and
                        manifest['status'] == 'PREPARED_NOT_RUN' and manifest['source_id'] == 'aihub_71875', 'REVIEW_PACKET_SCHEMA_MISMATCH')
                require(manifest['human_review'] == 'not_performed' and manifest['human_review_count'] == 0 and
                        manifest['review_scores_generated'] is False and manifest['clinical_accuracy'] is None and
                        manifest['clinical_scores'] is None, 'REVIEW_PACKET_NOT_UNSCORED')
                require(manifest['selected_count'] == manifest['cap'] == 16 and manifest['selected_by_q_type'] == {'2': 8, '3': 8}
                        and manifest['seed'] == 20261023 and manifest['test_review_must_not_select_training'] is True, 'REVIEW_SELECTION_CONTRACT_MISMATCH')
                require(set(manifest['files_sha256']) == {'review.csv', 'private_key.json', 'instructions.txt'}, 'REVIEW_FILES_RECEIPTS_MISSING')
                for name, digest in manifest['files_sha256'].items():
                    asset = folder / name
                    require(not asset.is_symlink() and asset.resolve().is_relative_to(folder.resolve()) and
                            file_hash(asset) == digest, 'REVIEW_PACKET_FILE_HASH_MISMATCH')
                # Inspect only rating cells and q_type counts; never copy source text or the A/B key.
                with (folder / 'review.csv').open(encoding='utf-8', newline='') as stream:
                    table = list(csv.DictReader(stream))
                require(len(table) == 16 and dict(Counter(row['q_type'] for row in table)) == {'2': 8, '3': 8}, 'REVIEW_TABLE_STRATUM_MISMATCH')
                require(all(row[field] == '' for row in table for field in REVIEW_FIELDS), 'REVIEW_RATINGS_NOT_BLANK')
                pp = root / 'v2/text/text-test.json'
                require(Path(manifest['protocol_path']).resolve() == pp.resolve() and
                        file_hash(pp) == manifest['protocol_file_sha256'], 'REVIEW_PROTOCOL_RECEIPT_MISMATCH')
                protocol, _ = evidence.protocol(pp)
                require(protocol['protocol_hash'] == manifest['protocol_hash'], 'REVIEW_PROTOCOL_HASH_MISMATCH')
                paths = [root / 'v2/text/evaluations' / f'{model}-{role}-text-{SUFFIX}' / 'predictions.jsonl'
                         for role in ('base', condition)]
                require(len(manifest['source_runs']) == 2, 'REVIEW_SOURCE_RUN_COUNT_MISMATCH')
                runs, _ = _review_runs(paths, protocol)
                origin = read(root / 'runs' / f'{model}-v1-20261023/manifest.json')
                for source, pred_path, run in zip(manifest['source_runs'], paths, runs):
                    mp = pred_path.parent / 'run_manifest.json'
                    require(Path(source['predictions_path']).resolve() == pred_path.resolve() and
                            Path(source['manifest_path']).resolve() == mp.resolve() and
                            file_hash(mp) == source['manifest_sha256'], 'REVIEW_SOURCE_MANIFEST_CHANGED')
                    require(source['run_id'] == run['run_id'] == pred_path.parent.name and
                            source['identity_hash'] == run['identity_hash'] and source['status'] == run['status'], 'REVIEW_SOURCE_IDENTITY_MISMATCH')
                    require(run['status'] in {'SUCCEEDED', 'FAILED_REPRODUCIBLE', 'BLOCKED_HARDWARE', 'SKIPPED_BUDGET', 'INTERRUPTED'}, 'REVIEW_SOURCE_NOT_TERMINAL')
                    require(source['artifact_hashes'] == run['artifact_hashes'] and set(source['artifact_hashes']) == {'predictions.jsonl','metrics.json'}, 'REVIEW_SOURCE_ARTIFACT_RECEIPT_MISMATCH')
                    for name, digest in source['artifact_hashes'].items():
                        require(file_hash(pred_path.parent / name) == digest, 'REVIEW_SOURCE_PREDICTIONS_CHANGED')
                    native = run['model']['native_files']['sha256']
                    require(native == origin['model']['native_files']['sha256'] == manifest['matched_native_files_sha256'] and
                            run['model']['chat_template_sha256'] == manifest['matched_chat_template_sha256'] and
                            run['model']['dtype'] == manifest['matched_dtype'], 'REVIEW_NATIVE_MODEL_MISMATCH')
                    if run['status'] == 'SUCCEEDED':
                        verify_completed_evaluation(run, protocol, pred_path.parent, native)
                require(runs[0].get('adapter_path') is None and runs[1].get('adapter_path') and
                        evidence.adapter(runs[1]['adapter_path']) == runs[1]['adapter_digest'], 'REVIEW_ADAPTER_BYTES_MISMATCH')
                require(file_hash(path) == manifest_sha, 'REVIEW_MANIFEST_CHANGED_DURING_READ')
                item.update(status='PREPARED_NOT_RUN', verified=True, prepared_count=16,
                            prepared_by_q_type={'2': 8, '3': 8}, protocol_hash=manifest['protocol_hash'],
                            manifest_sha256=manifest_sha, packet_file_hashes=manifest['files_sha256'],
                            source_prediction_receipts=[{'run_id':r['run_id'], 'manifest_sha256':r['manifest_sha256'],
                                'predictions_sha256':r['artifact_hashes']['predictions.jsonl']} for r in manifest['source_runs']])
                evidence.receipt(path, 'unscored_review_packet_verified')
            except (OSError, ValueError, KeyError, TypeError) as error:
                item.update(status='FAILED_VERIFICATION', verification_error=type(error).__name__)
                evidence.errors.append({'run_id': packet_id, 'reason_code': 'REVIEW_PACKET_VERIFICATION_FAILED', 'error_type': type(error).__name__})
    return {'planned_packets': 4, 'verified_prepared_packets': sum(p['verified'] for p in packets),
            'prepared_review_pairs': sum(p['prepared_count'] for p in packets), 'packets': packets,
            'human_review': 'not_performed', 'human_review_count': 0, 'clinical_accuracy': None,
            'private_tables_published': False, 'status': 'PREPARED_NOT_RUN' if all(p['verified'] for p in packets) else 'INCOMPLETE_PREPARATION'}


def v1_intersections(root, evidence, cells):
    """Descriptive protocol-bundle comparison on exactly matched prior test IDs."""
    from analyze_v1_followup import verified_pair
    result = []
    lookup = {(c['model'], c['phase'], c['condition']): c for c in cells if c.get('verified')}
    for model in VLM:
        for phase, scope, name in (('slake', 'test', 'slake-test-v1.json'), ('vqarad', 'vqarad', 'vqarad-test-v1.json')):
            if not any((model, phase, mode) in lookup for mode in ('base', 'tuned_v1')):
                continue
            try:
                old_samples, old_runs, receipts = verified_pair(root, model, scope, name)
                for mode, (old_manifest, old_preds) in zip(('base', 'tuned_v1'), old_runs):
                    new = lookup.get((model, phase, mode))
                    if not new:
                        continue
                    current = evidence.private[new['run_id']]
                    require(old_manifest['model_revision'] == current[0]['model_revision'] and
                            old_manifest.get('adapter_digest') == current[0].get('adapter_digest'), 'V1_V2_MODEL_ADAPTER_CHANGED')
                    old = {s['sample_id']:s for s in old_samples}; recent = {s['sample_id']:s for s in current[2]}
                    ids = set(old) & set(recent)
                    require(all(canonical_hash(old[s]['input']) == canonical_hash(recent[s]['input']) and
                                old[s]['supervision'] == recent[s]['supervision'] for s in ids), 'V1_V2_SHARED_INPUT_REFERENCE_CHANGED')
                    a = [r for r in score_rows(old_samples, old_preds) if r['sample_id'] in ids]
                    b = [r for r in current[5] if r['sample_id'] in ids]
                    c = [r for r in current[4] if r['sample_id'] in ids]
                    groups = []
                    for group in ('ALL', 'OPEN', 'CLOSED'):
                        selected = {r['sample_id'] for r in a if group == 'ALL' or str(r.get('answer_type')).upper() == group}
                        if not selected:
                            continue
                        score = lambda rows: sum(r['exact_match'] for r in rows if r['sample_id'] in selected) / len(selected)
                        groups.append({'answer_type': group, 'support': len(selected), 'v1_strict_em': score(a),
                                       'v2_strict_em': score(b), 'v2_normalized_em': score(c)})
                    result.append({'model': model, 'phase': phase, 'condition': mode, 'intersection_count': len(ids),
                        'v1_full_scheduled': len(old_samples), 'v2_full_scheduled': len(current[2]), 'groups': groups,
                        'interpretation': 'protocol-bundle change (prompt, generation budget and normalization); not a causal isolation',
                        'test_used_for_selection': False, 'v1_run_manifest_sha256': receipts[0 if mode == 'base' else 1]['run_manifest_sha256'],
                        'v1_predictions_sha256': receipts[0 if mode == 'base' else 1]['predictions_sha256']})
            except (OSError, ValueError, KeyError, TypeError) as error:
                evidence.errors.append({'run_id': model + '-' + phase + '-v1-intersection',
                                        'reason_code': 'V1_INTERSECTION_VERIFICATION_FAILED', 'error_type': type(error).__name__})
    return result


def collect(root):
    root = Path(root).resolve(); evidence = Evidence(root)
    try:
        data = data_summary(root, evidence)
    except (OSError, ValueError, KeyError, TypeError) as error:
        evidence.errors.append({'run_id': 'v2-data-preparation', 'reason_code': 'DATA_PREPARATION_VERIFICATION_FAILED',
                                'error_type': type(error).__name__})
        data = {'text_corpora': {'status': 'FAILED_VERIFICATION'},
                'aihub_71762': {'status': 'UNVERIFIED_INPUT_RECEIPT', 'actual_acquired_bytes': None},
                'jetson': {'status': 'USER_DEFERRED', 'attempted_in_v2': False}}
    vqa_state = read(root / 'v2/vqa/state.json', {}); text_state = read(root / 'v2/text/state.json', {})
    campaigns = {}
    for name, state in (('vqa', vqa_state), ('text', text_state)):
        plan_path = root / f'v2/{name}/plan.json'; plan = read(plan_path)
        campaigns[name] = {'status': safe_status(state.get('status'), 'NOT_PREPARED' if not plan else 'PREPARED_NOT_RUN'),
                           'plan_present': bool(plan), 'state_present': bool(state), 'active_run': state.get('active')}
        if plan:
            evidence.receipt(plan_path, 'campaign_plan')
            if state:
                require(state['plan_sha256'] == canonical_hash(plan), 'CAMPAIGN_STATE_PLAN_MISMATCH')
                evidence.receipt(root / f'v2/{name}/state.json', 'campaign_state_snapshot')
    decision_path = root / 'v2/vqa/validation-decision.json'; decision = read(decision_path)
    validation = {'status': 'NOT_RUN', 'chosen_max_vqa_tokens': None, 'test_used_for_selection': False}
    if decision:
        try:
            from run_vqa_v2 import verify_validation_decision
            ready = verify_validation_decision(read(root / 'v2/vqa/plan.json'), decision,
                {n: root / f'v2/vqa/protocols/validation-{n}.json' for n in (128, 256)}, run_root=root / 'v2/vqa/runs')
            validation.update(status='SUCCEEDED', ready_models=ready, chosen_max_vqa_tokens=decision['chosen_max_vqa_tokens'],
                              observations=[{k: x[k] for k in ('model', 'mode', 'status', 'length_count', 'predictions') if k in x} for x in decision['observations']])
            evidence.receipt(decision_path, 'validation_only_decision_verified')
        except (OSError, ValueError, KeyError, TypeError) as error:
            validation.update(status='FAILED_VERIFICATION', error_type=type(error).__name__)
            evidence.errors.append({'run_id': 'vqa-validation-decision', 'reason_code': 'VALIDATION_DECISION_VERIFICATION_FAILED'})
    cells = []
    for model in VLM:
        config_path = root / f'runs/{model}-v1-20261023/config.json'
        for phase, count, protocol in [('validation128', 32, 'validation-128.json'), ('validation256', 32, 'validation-256.json'),
                                       ('slake', 1061, 'slake-test.json'), ('vqarad', 451, 'vqarad-test.json'), ('noimage', 32, 'slake-no-image.json')]:
            for mode in ('base', 'tuned'):
                folder = root / 'v2/vqa/runs' / f'{model}-{phase}-{mode}-{SUFFIX}'
                cells.append(evidence.evaluation(model, 'base' if mode == 'base' else 'tuned_v1', phase, folder,
                    root / 'v2/vqa/protocols' / protocol, config_path, expected_count=count, state=vqa_state))
    training = [evidence.training(model, condition, phase, text_state)
                for model in TEXT for condition in ('ko', 'mix') for phase in ('smoke', 'full')]
    for model in TEXT + ('medgemma15_4b',):
        for condition in (('base',) if model == 'medgemma15_4b' else ('base', 'en_v1', 'ko', 'mix')):
            folder = root / 'v2/text/evaluations' / f'{model}-{condition}-text-{SUFFIX}'
            cells.append(evidence.evaluation(model, condition, 'text', folder, root / 'v2/text/text-test.json',
                root / 'v2/text/configs' / f'{model}-evaluation.json', expected_count=4530,
                expected_sources={'kormedmcqa': 3009, 'pubmedqa_labeled': 500, 'medquad': 64, 'aihub_71875': 957},
                state=text_state, selected_training=root / 'v2/text/runs' / f'{model}-{condition}-full-{SUFFIX}' if condition in ('ko', 'mix') else None))
    pairs = comparisons(evidence, cells)
    intersections = v1_intersections(root, evidence, cells)
    review = review_summary(root, evidence)
    reanalysis_path = REPO / 'reports/public_summary/v2/v1_reanalysis.json'
    reanalysis = read(reanalysis_path)
    v1_link = {'status': reanalysis.get('status', 'NOT_RUN') if reanalysis else 'NOT_RUN',
               'report': 'V1_REANALYSIS_KO.md', 'new_inference': False}
    if reanalysis:
        v1_link['sha256'] = evidence.receipt(reanalysis_path, 'v1_reanalysis_aggregate')
    preflight_path = root / 'v2/text/medgemma-native-text-preflight.json'
    preflight = read(preflight_path)
    preflight_public = {'status': 'NOT_RUN', 'gpu_inference': False}
    if preflight:
        preflight_public.update({k:preflight[k] for k in ('status','evidence_level','model_id','revision','protocol_hash')})
        preflight_public['receipt_sha256'] = evidence.receipt(preflight_path, 'cpu_processor_preflight_not_gpu_inference')
    snapshots = []
    for path in sorted((root / 'v2/source_snapshots').glob('*/receipt.json')):
        declaration = read(path)
        try:
            require(all((path.parent / relative).resolve().is_relative_to(path.parent.resolve()) and
                        file_hash(path.parent / relative) == digest for relative, digest in declaration['files'].items()), 'CODE_SNAPSHOT_BYTES_MISMATCH')
            snapshots.append({'snapshot_id': path.parent.name, 'files': len(declaration['files']),
                              'receipt_sha256': evidence.receipt(path, 'frozen_v2_execution_sources_verified')})
        except (OSError, ValueError, KeyError, TypeError) as error:
            evidence.errors.append({'run_id': 'v2-source-snapshot', 'reason_code': 'CODE_SNAPSHOT_VERIFICATION_FAILED', 'error_type': type(error).__name__})
    summary = {'schema_version': 'medvlm.public-report.v2', 'generated_at': utcnow(),
        'scope': 'read-only evidence snapshot; in-flight work may progress after this timestamp',
        'data': data, 'campaigns': campaigns, 'vqa_validation': validation, 'training': training, 'evaluations': cells,
        'comparisons': pairs, 'usage': usage_summary(root, evidence), 'v1_reanalysis': v1_link,
        'v1_v2_matched_intersections': intersections,
        'medgemma_text_native_cpu_preflight': preflight_public, 'execution_source_snapshots': snapshots,
        'expert_review_preparation': review,
        'clinical_accuracy': None, 'human_review': 'NOT_PERFORMED',
        'recipe': {'seed': 20261023, 'epochs': 1, 'learning_rate': 0.0002,
                   'effective_batch': 16, 'lora': {'rank': 8, 'alpha': 16, 'dropout': .05},
                   'checkpoint_validation': 'same common dev file; 48 source-balanced samples; seed 20261023; not q_type-stratified',
                   'medgemma_text_condition': 'original base, no VQA adapter', 'additional_seeds': 'NOT_SELECTED'},
        'limitations': ['MedGemma has known dataset-level SLAKE pretraining exposure; exact test exposure is unknown. Past exposure of the general models is also unknown.',
            'Previously observed v1 test items are included; this is not a new blind benchmark.',
            'One seed does not measure training-seed variance; group bootstrap only resamples observed examples.',
            'KO4000 and MIX8000 differ in total examples, tokens and optimizer updates; effects are not attributable solely to language mix.',
            'Free-text EM/F1 and reference phrase inclusion are not clinical accuracy. No expert review was performed.',
            'VQA-RAD is an external source under this experiment, not proof of clinically valid generalization.',
            'No-image results are a matched diagnostic, not training data or causal proof of image understanding.',
            'Recorded native model digests are linked to prior training receipts; base weight bytes are not rehashed by this report.']}
    stage = {'schema_version': 'medvlm.stage-status.v2', 'generated_at': summary['generated_at'],
        'vqa_validation': validation['status'],
        'vqa_main': stage_counts([c for c in cells if c['phase'] in ('slake', 'vqarad', 'noimage')]),
        'text_training_and_reload': stage_counts(training), 'text_evaluation': stage_counts([c for c in cells if c['phase'] == 'text']),
        'aihub_71762': data['aihub_71762']['status'], 'jetson': 'USER_DEFERRED', 'human_review': 'NOT_PERFORMED',
        'review_preparation': {k:review[k] for k in ('status','planned_packets','verified_prepared_packets','prepared_review_pairs','human_review_count')},
        'verification_errors': len(evidence.errors)}
    stage['execution_matrix_status'] = 'SUCCEEDED' if all(stage[k]['verified_succeeded'] == stage[k]['planned'] for k in
        ('vqa_main', 'text_training_and_reload', 'text_evaluation')) and validation['status'] == 'SUCCEEDED' and review['verified_prepared_packets'] == 4 and not evidence.errors else 'IN_PROGRESS_OR_PARTIAL'
    stage['status'] = stage['execution_matrix_status']
    if stage['status'] == 'SUCCEEDED' and stage['aihub_71762'] != 'SUCCEEDED':
        stage['status'] = 'COMPLETED_AVAILABLE_WORK_WITH_EXTERNAL_BLOCKER'
    verification = {'schema_version': 'medvlm.public-verification.v2', 'generated_at': summary['generated_at'],
        'status': 'PASS_FOR_COMPLETED_ARTIFACTS' if not evidence.errors else 'FAILED_VERIFICATION',
        'base_weight_bytes_rehashed': False, 'new_model_load_or_gpu_work': False, 'v1_modified': False,
        'verified_evaluation_count': sum(c['verified'] for c in cells), 'verified_training_reload_count': sum(c['verified'] for c in training),
        'checks': ['frozen protocol/file/input/reference/image digests', 'run/config/model revision and recorded native model identity',
                   'prediction run identity, exact schedule and artifact hashes', 'stored metrics match independent recomputation',
                   'selected adapter bytes and reload/selection receipts', 'same paired protocol/config/precision/generation'],
        'errors': evidence.errors, 'receipts': list({(x['kind'], x['artifact'], x['sha256']): x for x in evidence.receipts}.values())}
    return summary, stage, verification


def stage_counts(cells):
    counts = dict(Counter(c['status'] for c in cells))
    return {'planned': len(cells), 'verified_succeeded': sum(c.get('verified', False) for c in cells), 'statuses': counts}


def pct(value):
    return '미실행' if value is None else f'{100 * value:.2f}%'


def markdown(summary, stages):
    lines = ['# v2 후속 결과', '', '## 현재 단계', '',
        f"기준 시각: {summary['generated_at']}. 실제 파일을 읽은 시점의 상태다. 실행 중 항목의 중간 예측은 완료 점수에 포함하지 않았다.", '',
        '| 단계 | 완료·무결성 검증 | 예정 | 상태 |', '|---|---:|---:|---|']
    for key, label in [('vqa_main', 'VQA 본평가·이미지 진단'), ('text_training_and_reload', 'KO/MIX smoke·본학습·재로드'), ('text_evaluation', '텍스트 평가 조건')]:
        s = stages[key]; lines.append(f"| {label} | {s['verified_succeeded']} | {s['planned']} | {json.dumps(s['statuses'], ensure_ascii=False)} |")
    lines += ['', f"VQA validation 결정: **{stages['vqa_validation']}**, 선택된 출력 상한 {summary['vqa_validation']['chosen_max_vqa_tokens']} tokens. Validation의 길이 종료 규칙으로 128→256을 한 번 검토했다. 테스트로 상한을 선택하지 않았다.",
        '', '## 이번에 한 일', '',
        '원본 v1 가중치·어댑터를 재사용한 VQA 평가와 새 KO/MIX 텍스트 조건을 분리했다. 아래 수치는 완료 파일의 예측 SHA, protocol·입력·정답·이미지 SHA, config·model revision, 기록된 native 모델 digest, adapter byte hash를 검증한 뒤 다시 집계한 값이다. base 가중치 전체 재해시는 하지 않았다.', '',
        '| 데이터/작업 | 실제 확보·준비 | 상태 |', '|---|---|---|']
    data = summary['data']; ai = data.get('aihub_71875', {})
    lines += [f"| AI-Hub 71875 | {json.dumps(ai.get('canonical_counts', {}), ensure_ascii=False)} | {ai.get('status', 'NOT_PREPARED')} |",
              f"| KO/MIX corpus | {json.dumps({k:v.get('train_count') for k,v in data['text_corpora'].get('corpora',{}).items()})} | {data['text_corpora']['status']} |",
              f"| MedMCQA | {json.dumps(data.get('medmcqa', {}).get('canonical_counts', {}))}; KO/MIX 미포함 | {data.get('medmcqa', {}).get('status', 'NOT_PREPARED')} |",
              f"| AI-Hub 71762 | 확인된 다운로드 {data['aihub_71762']['actual_acquired_bytes']} bytes; 입력 receipt 기준 | {data['aihub_71762']['status']} |",
              '| Jetson | 사용자 보류; v2 접속·배포·benchmark 미실행 | USER_DEFERRED |',
              f"| 전문가 검토표 준비 | {summary['expert_review_preparation']['verified_prepared_packets']}/4 packet, {summary['expert_review_preparation']['prepared_review_pairs']}개 답변 쌍; q_type 2/3 각 8개/packet | {summary['expert_review_preparation']['status']} |",
              '| 전문가 검토 수행 | 검토 0건·점수 미수집 | NOT_PERFORMED |', '']
    lines += ['검토표 준비와 사람의 검토는 별도다. 준비된 표는 파일·프로토콜·출처 예측 해시와 점수 칸 공란을 확인한다. 원문이 있는 CSV 및 A/B 대응 private key는 로컬 데이터 경로에만 두며 공개 보고서에는 싣지 않는다.', '']
    if ai:
        lines += [f"71875 수집/제외 수: {json.dumps(ai.get('counts', {}), ensure_ascii=False)}; 제외: {json.dumps(ai.get('excluded_counts', {}), ensure_ascii=False)}. "
                  + ('후보 pool 기준이며 최종 학습 선택은 아직 완료되지 않았다.' if ai.get('counts_are_candidate_pool') else '최종 corpus manifest가 가리키는 source audit 기준이다.'), '']
    if data['aihub_71762'].get('actual_acquired_bytes') == 0:
        lines += ['71762는 connector 256 MiB 다운로드 제한과 정상 브라우저의 ERR_BLOCKED_BY_CLIENT로 막혔다. 약 4.25 GB 원본은 확보되지 않았으며 QA pairing도 확인되지 않았다. 수집된 원본이 없는 reader 구현을 데이터 준비 완료로 계산하지 않는다.', '']
    semantic = data.get('semantic_audit', {})
    if 'candidate_rows' in semantic:
        lines += [f"의미 유사도 검사: KorMedMCQA {semantic['protected_count']}문항을 보호하고 E5 cosine ≥{semantic['threshold']}, top-{semantic['top_k']}에서 {semantic['candidate_rows']}행/{semantic['candidate_pair_count']}쌍을 후보로 찾았다. 이는 확정 중복이나 번역 동등성 판정이 아니다. lexical exact/near 검사와 별도다.",
            f"split별 후보 처리: {json.dumps(semantic.get('source_split_counts', {}), ensure_ascii=False)}. train/dev 후보는 보수적으로 제외하고 test 전체 957개는 보존하며 후보 표시만 남겼다.", '']
    selected = data['text_corpora'].get('selection')
    if selected:
        lines += [f"최종 KO q_type 선택은 {json.dumps(selected['ko_selected_counts'])}; MIX는 같은 KO에 {json.dumps(selected['en_selected_counts'])}를 더한다. 공통 dev는 {selected['common_validation_count']}개(AI-Hub {selected['ko_validation_count']} + EN {sum(selected['en_validation_selected_counts'].values())})다. 최초 균등 q_type quota와 dev 1,912개에서 바뀐 이유는 고정 threshold의 후보 격리로 long/train과 dev가 줄었기 때문이다. 평가 성능으로 선택하거나 threshold를 바꾸지 않았다.", '']
    lines += ['KO4,000(250 update), MIX8,000(500 update)는 seed 20261023 한 pass를 계획한다. 각각 독립 smoke 20 update와 새 프로세스 재로드를 먼저 확인한다. LR 2e-4, LoRA rank 8/alpha 16/dropout 0.05, effective batch 16이다. KO/MIX checkpoint 비교에는 동일 공통 dev 파일에서 source별 균형으로 고른 48개를 쓴다. q_type별 층화 표본을 보장하지 않는다.', '',
              '| 학습 run | 상태 | 실제 update | 실제 처리 문항 | supervised tokens | 선택 update | 재로드 확인 |', '|---|---|---:|---:|---:|---:|---|']
    for c in summary['training']:
        lines.append(f"| {c['run_id']} | {c['status']} | {c.get('optimizer_steps', '미실행')} | {c.get('samples_processed', '미실행')} | {c.get('supervised_tokens', '미실행')} | {c.get('selected_checkpoint_steps', '미확정')} | {c['reload_verified']} |")
    lines += ['', '실제 처리량은 전체 run의 실행량이며, 평가는 validation으로 선택한 checkpoint를 사용한다. 선택 시점의 문항·token·loss는 run_summary.json에 별도로 기록한다. 각 run의 최신·선택 checkpoint와 검증 이벤트를 보존하며, 선택되지 않은 중간 checkpoint 파일은 기존 보관 정책에 따라 정리한다. 이전 v1 실험은 변경하지 않는다.']
    lines += ['', '## 확인 결과', '', 'KorMedMCQA는 동일 전체 3,009문항의 micro/macro subject 정답률, coverage, invalid, 유효 응답 정답률을 함께 표시한다. 무응답·실패는 전체 분모에 남긴다. [직군별 세부 지표](metrics.csv)는 source_id=kormedmcqa와 subject 열로 확인할 수 있다.', '',
              '| 모델/조건 | n | 전체 micro | subject macro | coverage | invalid 형식 | 유효 응답 정답률 |', '|---|---:|---:|---:|---:|---:|---:|']
    for c in summary['evaluations']:
        if c.get('verified') and c['phase'] == 'text' and c['metrics']['kormedmcqa']:
            k = c['metrics']['kormedmcqa']
            lines.append(f"| {c['model']}/{c['condition']} | {k['scheduled']} | {pct(k['micro_end_to_end_correct_fraction'])} | {pct(k['macro_subject_end_to_end_correct_fraction'])} | {pct(k['coverage'])} | {pct(k['invalid_format_rate'])} | {pct(k['micro_answered_accuracy'])} |")
    lines += ['', '완료된 텍스트 평가가 없으면 위 표는 비어 있다. MedGemma 텍스트 조건은 원본 Base이며 VQA adapter를 사용하지 않는다. 검증된 text 실행은 원문 입력과 native pixel tensor 기록에서 영상 입력이 없음을 확인했다.', '',
              '| 모델/조건 | AI-Hub q_type | n | 전체 MCQ 정답률 또는 lexical EM | lexical F1 | coverage | 잘림 |', '|---|---|---:|---:|---:|---:|---:|']
    for c in summary['evaluations']:
        if c.get('verified'):
            for a in c['metrics']['aihub_q_types']:
                f1 = '해당 없음' if a['task_type'] == 'medical_mcq' else pct(a.get('lexical_token_f1'))
                lines.append(f"| {c['model']}/{c['condition']} | {a['type']} | {a['scheduled']} | {pct(a.get('end_to_end_correct_fraction', a.get('end_to_end_lexical_exact_match')))} | {f1} | {pct(a.get('coverage'))} | {a.get('truncated_count', 0)} |")
    lines += ['', 'short/long의 EM/F1는 문자열 일치도이며 임상 정확도가 아니다. MCQ q_type 1, short 2, long 3을 따로 집계했다. 잘림은 사전에 고정한 생성 상한에 도달한 건수이며 기술 실패와 구분한다. 잘린 응답도 전체 평가 분모에 유지하므로, 특히 자유서술 점수는 답변 길이 제한과 함께 해석한다.', '',
              '| 모델/조건 | 영어 retention source | n | 전체 정답률/lexical EM | lexical F1 | coverage | 잘림 |', '|---|---|---:|---:|---:|---:|---:|']
    for c in summary['evaluations']:
        if c.get('verified') and c['phase'] == 'text':
            for a in c['metrics']['cohorts']:
                if a['source_id'] in ('pubmedqa_labeled', 'medquad'):
                    f1 = '해당 없음' if a['source_id'] == 'pubmedqa_labeled' else pct(a.get('lexical_token_f1'))
                    lines.append(f"| {c['model']}/{c['condition']} | {a['source_id']} | {a['scheduled']} | {pct(a.get('end_to_end_correct_fraction', a.get('end_to_end_lexical_exact_match')))} | {f1} | {pct(a.get('coverage'))} | {a.get('truncated_count', 0)} |")
    lines += ['', 'VQA: 두 모델의 Base/Tuned는 같은 짧은 답변 prompt와 선택된 생성 상한을 쓴다. SLAKE 1,061개, VQA-RAD 451개 전체와 고정 32개 no-image 진단을 구분한다.', '',
              '| 모델/조건 | source/이미지 | OPEN/CLOSED | n | strict EM | v2 EM | v2 F1 | 잘림 | 출력 단어 중앙값 |', '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for c in summary['evaluations']:
        if c.get('verified') and not c['phase'].startswith('validation'):
            for v in c['metrics']['vqa']:
                lines.append(f"| {c['model']}/{c['condition']} | {v['source_id']}/{c['image_condition']} | {v['answer_type']} | {v['scheduled']} | {pct(v['strict_lexical_em'])} | {pct(v['v2_normalized_lexical_em'])} | {pct(v['v2_normalized_lexical_token_f1'])} | {v['truncated_count']} | {v['output_whitespace_words']['median']} |")
    lines += ['', 'strict는 기존 NFKC·casefold·공백 정규화이고, v2는 문장 끝 .!?。！？만 추가 제거한다. 숫자·단위·부정과 내부 구두점은 유지한다. 정답 구절 포함률을 정확도로 사용하지 않았다.', '',
              '| 모델 | 비교 | source/단계 | n/groups | 정답률 또는 lexical EM 차이 | 관측 group bootstrap 95% CI |', '|---|---|---|---:|---:|---|']
    for pair in summary['comparisons']:
        if pair['normalization'] == 'v2_normalized':
            interval = '[' + ', '.join(f'{100*v:+.2f}' for v in pair['ci95']) + ']%p' if pair['ci95'] else '미계산'
            lines.append(f"| {pair['model']} | {pair['right']} − {pair['left']} | {pair['source_id']}/{pair['phase']} | {pair['n']}/{pair['group_count']} | {100*pair['delta']:+.2f}%p | {interval} |")
    lines += ['', '이미지 진단은 같은 32개 ID의 image/no-image 결과만 비교한다. native no-image 실행 실패는 지원을 가정하거나 blank image로 바꾸지 않는다. VQA-RAD 외부 source 성능은 임상 일반화의 증명이 아니다.', '',
              '| 평가 run | 상태 | 전체 E2E latency 중앙값(초) | peak allocated/reserved(GiB) |', '|---|---|---:|---:|']
    for c in summary['evaluations']:
        if c['phase'].startswith('validation'):
            continue
        latency = None
        if c.get('verified'):
            latency = c['metrics']['e2e_latency_seconds']['median']
        mem = '/'.join(f"{(c.get(k) or 0)/1024**3:.3f}" for k in ('peak_allocated_bytes','peak_reserved_bytes')) if c.get('verified') else '미측정'
        latency_display = f'{latency:.4f}' if latency is not None else '미측정'
        lines.append(f"| {c['run_id']} | {c['status']} | {latency_display} | {mem} |")
    lines += ['', '위 latency는 실제 측정된 모든 예측의 E2E 중앙값이다. 전처리·생성을 포함하고 모델 load·warmup을 제외한다. 정확한 source/subject별 측정 분모와 분포는 JSON에 있다. Peak memory는 전체 run의 측정값이며 decode-only 처리량이나 에너지를 뜻하지 않는다.', '',
              '[v1 예측 재분석](V1_REANALYSIS_KO.md): KorMedMCQA 개선에는 invalid→valid 전환이 포함된다. both-valid 변화와 분리해 봐야 한다. MedGemma Base의 v1 SLAKE EM 0%는 출력 형태·32-token 상한의 영향을 받으며 의료 능력 0%를 뜻하지 않는다. 정답 구절 포함은 모순·부정에도 생길 수 있다. v1의 준비 8,450→처리 8,400은 50개 CONTEXT_OVERFLOW 제외였으며 시간·step 상한 때문이 아니다.', '']
    lines += ['| 동일 문항 교집합 | 조건 | OPEN/CLOSED | n | v1 strict EM | v2 strict EM | v2 normalized EM |', '|---|---|---|---:|---:|---:|---:|']
    for item in summary['v1_v2_matched_intersections']:
        for g in item['groups']:
            lines.append(f"| {item['model']}/{item['phase']} | {item['condition']} | {g['answer_type']} | {g['support']} | {pct(g['v1_strict_em'])} | {pct(g['v2_strict_em'])} | {pct(g['v2_normalized_em'])} |")
    lines += ['', '위 표는 v1과 v2의 동일 ID·입력·정답 교집합만 재집계한 것이다. v1 부분집합과 v2 전체집합을 직접 빼지 않는다. prompt·생성 상한·정규화가 함께 바뀐 프로토콜 묶음 효과이며 특정 변수 하나의 원인으로 단정하지 않는다.', '']
    usage = summary['usage']; policy = usage['policy']
    policy_label = '누적 GPU 시간 상한 없음' if policy['budget_mode'] == 'unlimited' else '유한 GPU 시간 상한 적용'
    resume_note = ('선언된 GPU 실행 행렬과 검토표 준비를 완료했다. 아래 명령은 복구·무결성 확인을 위한 재개 절차이며, 검증된 완료 학습·평가는 재사용한다.'
                   if stages.get('execution_matrix_status') == 'SUCCEEDED' else
                   '미완료 조건은 동일 선언과 artifact를 사용해 재개하고, 모두 끝난 뒤 이 보고서를 다시 생성한다. 완료 조건을 새로 학습하는 명령이 아니다.')
    lines += [f"현재 resource policy: **{policy['budget_mode']}** — {policy_label}. total/model/run/job/campaign 시간 상한은 {json.dumps({k: v for k,v in policy.items() if k not in ('schema_version','budget_mode')})}. v1의 24 GPU-hour/모델별 6-hour는 과거 실행 당시 설정이다.", '',
              f"실제 누적 정산 사용 {usage['completed_accounted_seconds']/3600:.4f}시간; v1 기준 "
              + (f"{usage['v1_baseline_seconds']/3600:.4f}시간; 정산 증가 {usage['completed_delta_from_v1_seconds']/3600:.4f}시간." if usage['v1_baseline_seconds'] is not None else '미확인.')
              + f" 진행 중 미정산 세션의 읽기 시점 경과 {usage['active_unsettled_elapsed_seconds']/3600:.4f}시간은 누적 정산값과 분리했다. 사용량 ledger를 초기화하거나 수정하지 않았다.", '',
              '학습 seed는 하나다. KO/MIX는 총 문항·token·update 수가 달라 언어 구성만의 효과를 분리할 수 없다. MedGemma의 SLAKE dataset-level 사전학습 노출은 알려져 있으나 정확한 test 노출 여부는 미상이다. 일반 모델들의 과거 노출도 미상으로 남긴다. v1에서 본 test를 포함하므로 새로운 blind 결과가 아니다. 자유서술 결과에 전문가 정확도 점수는 없다.', '',
              '## 다음 단계', '', resume_note, '', '```bash',
              'export MEDVLM_DATA_ROOT="$(realpath ../medvlm-data)"',
              'PYTHONPATH=src ../.venv-medvlm/bin/python scripts/run_vqa_v2.py --data-root "$MEDVLM_DATA_ROOT" --validation-only',
              'PYTHONPATH=src ../.venv-medvlm/bin/python scripts/run_vqa_v2.py --data-root "$MEDVLM_DATA_ROOT"',
              'PYTHONPATH=src ../.venv-medvlm/bin/python scripts/run_text_v2.py --data-root "$MEDVLM_DATA_ROOT" --phase all',
              'PYTHONPATH=src ../.venv-medvlm/bin/python scripts/report_v2.py --data-root "$MEDVLM_DATA_ROOT" --output reports/public_summary/v2', '```', '',
              '현재 캠페인이 실행 중이면 중복 실행하지 않는다. v2/vqa/state.json과 v2/text/state.json의 완료·실패 상태를 확인한 뒤 재개한다. verification.json의 실패가 있으면 해당 artifact를 조사하고 점수를 사용하지 않는다.', '',
              '## 사용자가 할 일', '', '현재 허용된 나머지 작업은 계속 진행할 수 있다. AI-Hub 71762를 이어가려면 정상적으로 확보한 원본의 로컬 경로가 필요하다. 접근 제한을 우회하지 않았고 71875 준비에 새 API key는 필요하지 않았다. Jetson은 사용자 보류 상태로 유지한다.', '']
    return '\n'.join(lines)


def csv_text(rows, fields):
    out = io.StringIO(); writer = csv.DictWriter(out, fieldnames=fields, extrasaction='ignore', lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
    return out.getvalue()


def validate_public(value):
    """Fail closed before writing if private paths or prediction bodies escaped the whitelist."""
    forbidden = {'raw_output', 'parsed_answer', 'question', 'supervision', 'input_digests', 'sample_ids', 'access_token', 'refresh_token'}
    if isinstance(value, dict):
        require(not forbidden.intersection(value), 'PRIVATE_FIELD_IN_PUBLIC_REPORT')
        for item in value.values(): validate_public(item)
    elif isinstance(value, (list, tuple)):
        for item in value: validate_public(item)
    elif isinstance(value, str):
        require(not any(x in value for x in ('/home/', '/Users/', 'Bearer ', 'drive.google.com', 'googleusercontent.com')), 'PRIVATE_PATH_OR_AUTH_IN_PUBLIC_REPORT')


def write_reports(summary, stages, verification, output):
    output = Path(output).resolve()
    # Refuse the legacy public_summary root and every other v1 subdirectory.
    require(output.name == 'v2', 'OUTPUT_MUST_BE_V2_DIRECTORY')
    public_root = REPO / 'reports/public_summary'
    require(not output.is_relative_to(public_root) or output == public_root / 'v2', 'PROTECTED_V1_OUTPUT')
    payloads = {'run_summary.json': summary, 'stage_status.json': stages, 'verification.json': verification}
    for value in payloads.values(): validate_public(value)
    text = markdown(summary, stages); validate_public(text)
    rows = []
    for c in summary['evaluations']:
        if c.get('verified'):
            for item in c['metrics']['cohorts']:
                rows.append({'run_id': c['run_id'], 'model': c['model'], 'condition': c['condition'], 'phase': c['phase'], **item})
    comparison_fields = ['model','phase','left','right','normalization','source_id','task_type','language','n','group_count','delta','ci95','bootstrap_unit','bootstrap_resamples','metric']
    metric_fields = ['run_id','model','condition','phase','source_id','task_type','language','subject','scheduled','attempted','valid_answers','correct','exact_matches','coverage','answered_accuracy','end_to_end_correct_fraction','answered_lexical_exact_match','end_to_end_lexical_exact_match','lexical_token_f1','invalid_format_rate','technical_failure_rate','truncated_count']
    output.mkdir(parents=True, exist_ok=True)
    documents = {**{name: json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n' for name,value in payloads.items()},
                 'RESULTS_KO.md': text, 'comparison.csv': csv_text(summary['comparisons'], comparison_fields), 'metrics.csv': csv_text(rows, metric_fields)}
    for name, body in documents.items():
        validate_public(body)
        destination = output / name
        require(not destination.is_symlink(), 'PUBLIC_OUTPUT_SYMLINK_FORBIDDEN')
        temporary = output / ('.' + name + '.tmp')
        require(not temporary.is_symlink(), 'PUBLIC_TEMP_SYMLINK_FORBIDDEN')
        temporary.write_text(body, encoding='utf-8'); temporary.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('reports/public_summary/v2'))
    args = parser.parse_args()
    summary, stages, verification = collect(args.data_root)
    write_reports(summary, stages, verification, args.output)
    print(json.dumps({'status': stages['status'], 'verification': verification['status'],
                      'verified_evaluations': verification['verified_evaluation_count'],
                      'verified_training_reload': verification['verified_training_reload_count']}))
    return 0 if not verification['errors'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
