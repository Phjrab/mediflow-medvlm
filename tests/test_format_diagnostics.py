import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from mediflow_medvlm.evaluation.format_diagnostics import diagnostic_text,select_validation,descriptive_style
from mediflow_medvlm.training.rendering import user_text
from mediflow_medvlm.data.io import write_jsonl
from mediflow_medvlm.runtime import canonical_hash,file_hash


def fixture(source,index,subject=None):
    return {'sample_id':source+str(subject)+str(index),'source_id':source,'split':'validation',
            'task_type':'context_qa' if source=='pubmedqa_labeled' else 'medical_mcq',
            'input':{'question':'q'+str(index)+str(subject),'images':[],
                     **({'context':'ctx'} if source=='pubmedqa_labeled' else {'options':{'A':'one','B':'two'}})},
            'supervision':{'answer':'SECRET','option_id':'B'},'metadata':{'subject':subject,'hidden':'SECRET'}}


class FormatDiagnosticTests(unittest.TestCase):
    def test_saved_diagnostic_rejects_parser_changes_and_counts_prefix_differences(self):
        path=Path(__file__).resolve().parents[1]/'scripts/run_format_diagnostic_v5.py'
        spec=importlib.util.spec_from_file_location('format_diagnostic_test',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);folder=root/'runs/model-base';folder.mkdir(parents=True)
            rows=[fixture('aihub_71875',i) for i in range(24)]
            write_jsonl(root/'validation.jsonl',rows)
            plan={'instruction_languages':['original','ko']};predictions=[]
            for language in plan['instruction_languages']:
                for maximum in [16,256]:
                    for row in rows:
                        predictions.append({'sample_id':row['sample_id'],'instruction_language':language,
                            'maximum':maximum,'input_hash':canonical_hash(row['input']),
                            'instruction_text_sha256':canonical_hash(diagnostic_text(row,language)),
                            'prompt_token_sha256':language,'generated_token_ids':[1],'output_tokens':1,
                            'raw_output':'A','status':'answered','parsed_answer':'A'})
            def save():
                write_jsonl(folder/'predictions.jsonl',predictions)
                (folder/'manifest.json').write_text(json.dumps({'plan_sha256':canonical_hash(plan),
                    'status':'SUCCEEDED','predictions_sha256':file_hash(folder/'predictions.jsonl')}))
            save();receipt,_=module.verify_worker(root,plan,'model','base')
            self.assertEqual(receipt['budget_pair_prefix_matches'],48)
            predictions[0]['generated_token_ids']=[9];save()
            receipt,_=module.verify_worker(root,plan,'model','base')
            self.assertEqual(receipt['budget_pair_prefix_matches'],47)
            predictions[0]['raw_output']='A with explanation';save()
            with self.assertRaisesRegex(ValueError,'STRICT_PARSER_CHANGED'):
                module.verify_worker(root,plan,'model','base')

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
