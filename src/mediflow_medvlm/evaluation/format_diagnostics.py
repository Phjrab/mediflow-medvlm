"""Validation-only classification diagnostics; historical parsers stay unchanged."""
from collections import Counter
import re

from ..runtime import canonical_hash
from ..training.rendering import user_text

SEED = 20261023
TRANSLATIONS = {
    'medical_mcq': ('Return only the option identifier.', '정답의 선택지 식별자만 출력하세요.'),
    'context_qa': ('Answer using the provided context. Return only yes, no, or maybe.',
                   '제공된 문맥을 이용해 답하세요. yes, no, maybe 중 하나만 출력하세요.'),
}


def diagnostic_text(row, language):
    restricted = {k: row[k] for k in ('sample_id', 'task_type', 'input')}
    if row['task_type'] not in TRANSLATIONS or row['input'].get('images'):
        raise ValueError('CLASSIFICATION_TEXT_ONLY')
    text = user_text(restricted)
    old, new = TRANSLATIONS[row['task_type']]
    if not text.endswith(old):
        raise ValueError('UNEXPECTED_ORIGINAL_INSTRUCTION')
    if language == 'original':
        return text
    if language == 'ko':
        return text[:-len(old)] + new
    raise ValueError('UNKNOWN_INSTRUCTION_LANGUAGE')


def select_validation(rows, test_rows):
    """Eight per source, KorMed split equally by four professions, no gold use."""
    test_ids = {r['sample_id'] for r in test_rows}
    test_inputs = {canonical_hash(r['input']) for r in test_rows}
    buckets = {}
    for row in rows:
        if row['split'] != 'validation':
            raise ValueError('VALIDATION_ONLY')
        if (row['sample_id'] in test_ids or canonical_hash(row['input']) in test_inputs or
                row['input'].get('images') or row['task_type'] not in TRANSLATIONS):
            continue
        source = row['source_id']
        if source not in {'kormedmcqa', 'aihub_71875', 'pubmedqa_labeled'}:
            continue
        subject = row.get('metadata', {}).get('subject') if source == 'kormedmcqa' else None
        if subject == 'pharmacy':
            subject = 'pharm'
        buckets.setdefault((source, subject), []).append(row)
    expected = {('kormedmcqa', s): 2 for s in ('doctor','dentist','nurse','pharm')}
    expected.update({('aihub_71875', None): 8, ('pubmedqa_labeled', None): 8})
    result = []
    for key, count in expected.items():
        candidates = sorted(buckets.get(key, []), key=lambda r: canonical_hash([SEED,r['sample_id']]))
        if len(candidates) < count:
            raise ValueError('INSUFFICIENT_VALIDATION_COHORT')
        result.extend(candidates[:count])
    if len({r['sample_id'] for r in result}) != 24:
        raise ValueError('DUPLICATE_VALIDATION_ID')
    return sorted(result, key=lambda r: canonical_hash([SEED,r['sample_id']]))


def descriptive_style(predictions):
    """No relaxed parsing, label extraction, or alternative correctness scores."""
    counts = Counter(p['status'] for p in predictions)
    return {'n': len(predictions), 'statuses': dict(counts),
            'length_terminated': sum(p.get('finish_reason') == 'length' for p in predictions),
            'empty_outputs': sum(not p.get('raw_output','').strip() for p in predictions),
            'unicode_replacement_outputs': sum('\ufffd' in p.get('raw_output','') for p in predictions),
            'answer_declaration_prose': sum(bool(re.match(r'(?:정답은|정답\s*[:：]|답은|the answer is)',
                                          p.get('raw_output','').strip(), re.I)) for p in predictions),
            'leading_yes_no_maybe_with_extra_text': sum(bool(re.match(r'^(yes|no|maybe)[,\s]+\S',
                                          p.get('raw_output','').strip(), re.I)) for p in predictions),
            'clinical_accuracy': None, 'rescored_with_relaxed_parser': False}
