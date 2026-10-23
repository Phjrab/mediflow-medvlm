"""Canonical records keep inference inputs physically separate from labels."""
from __future__ import annotations
import re
import unicodedata
from .io import digest

VERSION = 'medvlm.dataset.v1'
TRANSFORM_VERSION = 'medvlm.data.v1.0'
INPUT_FIELDS = {'question', 'context', 'options', 'images'}


def normalized(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', str(text))).strip()


def input_digest(row):
    """Preserve case, numbers, punctuation, units and negation in duplicate keys."""
    def norm(x):
        if isinstance(x, str): return normalized(x)
        if isinstance(x, list): return [norm(v) for v in x]
        if isinstance(x, dict): return {k: norm(v) for k, v in x.items()}
        return x
    identity = {k: norm(v) for k, v in row['input'].items() if k != 'images' and v not in (None, [], {})}
    if row['input'].get('images'):
        meta = row.get('metadata', {})
        identity['image'] = meta.get('decoded_pixel_hash') or meta.get('image_sha256') or row['group_ids'].get('image') or row['input']['images']
    return digest(identity)


def sample(source, revision, original_id, task, question, answer, split, *, language='en', source_url=None, document=None, original_split=None, context=None, options=None, option_id=None, metadata=None, license_ref=None, fetched_at=None, raw_sha256=None, annotation='source_provided'):
    inp = {'question': question, 'images': []}
    if context is not None: inp['context'] = context
    if options is not None: inp['options'] = options
    return {
        'schema_version': VERSION,
        'sample_id': f'{source}/{revision[:12]}/{digest(str(original_id))[:24]}',
        'source_id': source, 'source_revision': revision, 'original_id': str(original_id),
        'task_type': task, 'language': language, 'original_split': original_split,
        'split': split, 'input': inp,
        'supervision': {'answer': answer, 'option_id': option_id, 'rationale': None, 'annotation_provenance': annotation},
        'group_ids': {'patient': None, 'study': None, 'document': document, 'image': None, 'parent': None, 'duplicate_cluster': None},
        'grouping_quality': {'status': 'verified' if document else 'proxy', 'level': 'document' if document else 'original_question', 'patient_group_verified': False},
        'provenance': {'source_url': source_url, 'local_asset_sha256': raw_sha256, 'transformation_version': TRANSFORM_VERSION, 'fetched_at': fetched_at, 'license_ref': license_ref},
        'derivation': 'original', 'review_status': 'machine_checked',
        'restrictions': {'raw_redistribution': False, 'research_only': True, 'sft_allowed': source != 'kormedmcqa'},
        'metadata': metadata or {},
    }


def validate_record(row):
    required = {'schema_version','sample_id','source_id','source_revision','original_id','task_type','language','original_split','split','input','supervision','group_ids','grouping_quality','provenance','derivation','review_status','restrictions'}
    errors = []
    if required - row.keys(): errors.append('missing_required_fields')
    if row.get('schema_version') != VERSION: errors.append('schema_version')
    if not row.get('source_revision'): errors.append('unresolved_revision')
    if row.get('split') not in {'train','validation','test'}: errors.append('split')
    inp = row.get('input', {})
    if set(inp) - INPUT_FIELDS: errors.append('forbidden_input_field')
    if not isinstance(inp.get('question'), str) or not inp.get('question','').strip(): errors.append('empty_question')
    if not row.get('supervision', {}).get('answer'): errors.append('empty_answer')
    if row.get('task_type') == 'medical_mcq':
        opts = inp.get('options', {})
        if row['supervision'].get('option_id') not in opts: errors.append('invalid_option_id')
    if row.get('task_type') == 'medical_vqa' and not inp.get('images'): errors.append('missing_image')
    if row.get('source_id') == 'kormedmcqa' and row.get('split') == 'train': errors.append('eval_source_in_train')
    return errors
