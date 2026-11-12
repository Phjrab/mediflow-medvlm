import json
import sqlite3
import unittest

from mediflow_medvlm.data.aihub71762_experiment import (
    check_authorization, category_key, component_split, component_pools,
    pair_components, canonical_pair,
)
from mediflow_medvlm.data.io import digest


class CategoryPairingTests(unittest.TestCase):
    def database(self, rows):
        c = sqlite3.connect(':memory:')
        c.executescript('''CREATE TABLE components(archive,member,split,kind,
            text_sha256,domain,intention,content_json);
            CREATE TABLE assets(archive,member,sha256);''')
        for archive, sid, kind, text, disease in rows:
            member = f'{"1.질문" if kind == "question" else "2.답변"}/분류/{disease}/치료/{sid}.json'
            obj = {'question': text} if kind == 'question' else {'answer': text}
            text_hash = digest(text)
            c.execute('INSERT INTO components VALUES(?,?,?,?,?,?,?,?)',
                      (archive, member, 'train' if archive == 'TL.zip' else 'validation',
                       kind, text_hash, '분류', '치료', json.dumps(obj)))
            c.execute('INSERT INTO assets VALUES(?,?,?)', (archive, member, digest(obj)))
        return c

    def test_explicit_user_exception_required(self):
        with self.assertRaisesRegex(ValueError, 'AUTHORIZATION'):
            check_authorization({'provider_says_random_pairing_is_ok': True})
        check_authorization({'dataset_id': '71762',
                             'decision': 'ALLOW_SEPARATE_CATEGORY_RANDOM_EXPERIMENT',
                             'human_instruction': '71762는 제공처 방식으로 별도 실험 진행'})

    def test_serials_are_irrelevant_and_no_cross_category_or_cartesian(self):
        c = self.database([
            ('TL.zip', 'HC-Q-1', 'question', '치료 질문', '질환A'),
            ('TL.zip', 'HC-A-99', 'answer', {'body': '다른 번호의 답변'}, '질환A'),
            ('TL.zip', 'HC-A-100', 'answer', {'body': '추가 답변'}, '질환A'),
            ('TL.zip', 'HC-A-1', 'answer', {'body': '같은 번호 다른 질환'}, '질환B')])
        pools, _ = component_pools(c)
        pairs, missing = pair_components(pools)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][1]['category'][1], '질환A')
        self.assertEqual(missing, {})
        self.assertEqual(pair_components(pools), (pairs, missing))

    def test_cross_split_exact_questions_and_answers_excluded(self):
        rows = []
        for archive in ('TL.zip', 'VL.zip'):
            rows.extend([(archive, 'HC-Q-1', 'question', '동일 질문', '질환'),
                         (archive, 'HC-A-2', 'answer', {'body': '동일 답변'}, '질환')])
        pools, receipt = component_pools(self.database(rows))
        self.assertEqual(pools, {})
        self.assertEqual(receipt['cross_split_exact_component_groups'], 2)

    def test_sections_numbers_negation_and_no_fake_parent(self):
        sections = {'intro': '  안내 ', 'body': '2 mg을 복용하지 마세요.\n', 'conclusion': None}
        c = self.database([('TL.zip', 'HC-Q-1', 'question', '  약물 질문? ', '질환'),
                           ('TL.zip', 'HC-A-99', 'answer', sections, '질환')])
        pools, _ = component_pools(c)
        q, a = pair_components(pools)[0][0]
        row = canonical_pair(c, q, a, revision='revision', authorization_sha256='auth', fetched_at='date')
        self.assertEqual(row['input']['question'], '  약물 질문? ')
        self.assertEqual(row['metadata']['answer_sections'], sections)
        self.assertEqual(row['supervision']['answer'], '  안내 \n\n2 mg을 복용하지 마세요.\n\n\n')
        self.assertIsNone(row['group_ids']['parent'])
        self.assertFalse(row['metadata']['true_qa_correspondence_verified'])

    def test_tags_must_match_paths_and_text_group_split_is_stable(self):
        with self.assertRaisesRegex(ValueError, 'PATH_TAG'):
            category_key('1.질문/분류/질환/치료/Q.json', '다른분류', '치료')
        with self.assertRaises(ValueError):
            category_key('../분류/질환/치료/Q.json', '분류', '치료')
        self.assertEqual(component_split('validation', 'same'), component_split('validation', 'same'))
        self.assertEqual(component_split('train', 'same'), 'train')


if __name__ == '__main__':
    unittest.main()
