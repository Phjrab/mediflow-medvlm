import copy
import json
from pathlib import Path
import tempfile
import unittest
from mediflow_medvlm.evaluation.metrics import parse_answer,score_rows,aggregate,paired_compare
from mediflow_medvlm.evaluation.runner import freeze_protocol,load_protocol


def sample(i=0):
    return {"sample_id":str(i),"source_id":"fixture","task_type":"medical_mcq","language":"en","split":"test",
            "input":{"question":"Pick one","options":{"A":"x","B":"y"},"images":[]},"supervision":{"option_id":"B","answer":"y"},"group_ids":{"document":"doc"}}


class EvaluationTests(unittest.TestCase):
    def test_parser_never_uses_first_letter_from_reasoning(self):
        self.assertEqual(parse_answer("A appears possible, but the final answer is B.","medical_mcq"),(None,"invalid_format"))
        self.assertEqual(parse_answer("A/B","medical_mcq"),(None,"invalid_format"))
        self.assertEqual(parse_answer('{"final_answer":"B"}',"medical_mcq"),('B','answered'))
        self.assertEqual(parse_answer("maybe","context_qa"),('maybe','answered'))

    def test_failure_denominators_and_zero_answer_null(self):
        samples=[sample(i) for i in range(4)]
        preds=[{"sample_id":"0","parsed_answer":"B","status":"answered"},
               {"sample_id":"1","status":"oom"},{"sample_id":"2","status":"invalid_format"}]
        result=aggregate(score_rows(samples,preds))[0]
        self.assertEqual((result['scheduled'],result['attempted'],result['completed'],result['valid_answers']),(4,3,2,1))
        self.assertEqual(result['coverage'],.25);self.assertEqual(result['answered_accuracy'],1)
        self.assertEqual(result['end_to_end_correct_fraction'],.25)
        self.assertIsNone(aggregate(score_rows(samples,[]))[0]['answered_accuracy'])

    def test_pair_rejects_different_ids_and_resamples_groups(self):
        rows=score_rows([sample(0),sample(1)],[]);tuned=copy.deepcopy(rows)
        for r in tuned:r['exact_match']=1
        result=paired_compare(rows,tuned,resamples=50)[0]
        self.assertEqual(result['group_count'],1);self.assertEqual(result['ci95'],[1,1])
        with self.assertRaises(ValueError):paired_compare(rows,tuned[:1])

    def test_frozen_input_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'test.jsonl';output=Path(temp)/'protocol.json'
            path.write_text(json.dumps(sample())+'\n');freeze_protocol([path],output)
            load_protocol(output)
            changed=sample();changed['input']['question']='Changed'
            path.write_text(json.dumps(changed)+'\n')
            with self.assertRaises(ValueError):load_protocol(output)

    def test_gold_does_not_change_eval_messages(self):
        from mediflow_medvlm.training.rendering import render_messages
        a=sample();b=copy.deepcopy(a);b['supervision']={'answer':'SECRET','option_id':'A','rationale':'SECRET'}
        self.assertEqual(render_messages(a),render_messages(b))


if __name__=='__main__':unittest.main()
