import copy
import unittest

from mediflow_medvlm.evaluation.format_diagnostics import diagnostic_text,select_validation,descriptive_style
from mediflow_medvlm.training.rendering import user_text


def fixture(source,index,subject=None):
    return {'sample_id':source+str(subject)+str(index),'source_id':source,'split':'validation',
            'task_type':'context_qa' if source=='pubmedqa_labeled' else 'medical_mcq',
            'input':{'question':'q'+str(index)+str(subject),'images':[],
                     **({'context':'ctx'} if source=='pubmedqa_labeled' else {'options':{'A':'one','B':'two'}})},
            'supervision':{'answer':'SECRET','option_id':'B'},'metadata':{'subject':subject,'hidden':'SECRET'}}


class FormatDiagnosticTests(unittest.TestCase):
    def test_only_terminal_instruction_translated_no_supervision_or_metadata(self):
        for source in ['aihub_71875','pubmedqa_labeled']:
            r=fixture(source,1);before=copy.deepcopy(r)
            self.assertEqual(diagnostic_text(r,'original'),user_text(r))
            text=diagnostic_text(r,'ko')
            self.assertNotIn('SECRET',text);self.assertNotEqual(text,user_text(r))
            self.assertIn(r['input']['question'],text)
            self.assertEqual(before,r)
        r['input']['answer']='SECRET'
        with self.assertRaisesRegex(ValueError,'unsupported'):diagnostic_text(r,'ko')

    def test_validation_selection_is_gold_blind_stratified_and_excludes_test(self):
        rows=[fixture('kormedmcqa',i,s) for s in ['doctor','dentist','nurse','pharm'] for i in range(6)]
        rows += [fixture(s,i) for s in ['aihub_71875','pubmedqa_labeled'] for i in range(20)]
        blocked=copy.deepcopy(rows[0]);blocked['split']='test'
        picked=select_validation(rows,[blocked]);self.assertEqual(len(picked),24)
        self.assertNotIn(blocked['sample_id'],{r['sample_id'] for r in picked})
        changed=copy.deepcopy(rows)
        for r in changed:r['supervision']={'answer':'DIFFERENT','option_id':'A'}
        self.assertEqual([r['sample_id'] for r in picked],[r['sample_id'] for r in select_validation(changed,[blocked])])
        changed[0]['split']='test'
        with self.assertRaisesRegex(ValueError,'VALIDATION_ONLY'):select_validation(changed,[])

    def test_style_counts_never_rescore_extracted_labels_as_accuracy(self):
        x=descriptive_style([{'status':'invalid_format','raw_output':'정답은 B입니다.','finish_reason':'length'},
                             {'status':'invalid_format','raw_output':'Yes, the study found a difference.'}])
        self.assertEqual(x['answer_declaration_prose'],1)
        self.assertEqual(x['leading_yes_no_maybe_with_extra_text'],1)
        self.assertEqual(x['length_terminated'],1)
        self.assertIsNone(x['clinical_accuracy']);self.assertFalse(x['rescored_with_relaxed_parser'])


if __name__=='__main__':unittest.main()
