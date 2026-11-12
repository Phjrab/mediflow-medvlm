import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

from mediflow_medvlm.data.io import write_jsonl, sha256
from mediflow_medvlm.data.schema import sample
from mediflow_medvlm.evaluation.runner import freeze_protocol

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name+'.py'))
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


campaign = load_script('run_71762_v4')
report = load_script('report_71762_v4')


class V4CampaignTests(unittest.TestCase):
    def test_pair_verification_rejects_different_native_prompt_tokens(self):
        with tempfile.TemporaryDirectory() as temporary:
            left=Path(temporary)/'base';right=Path(temporary)/'tuned'
            left.mkdir();right.mkdir()
            row={'sample_id':'q','status':'answered','input_hash':'same',
                 'prompt_token_sha256':'native-A','generation_settings':{'max_new_tokens':512,'do_sample':False}}
            write_jsonl(left/'predictions.jsonl',[row]);write_jsonl(right/'predictions.jsonl',[row])
            self.assertEqual(report.verify_pair_prompts(left,right)['matched_native_prompt_hashes'],1)
            row['prompt_token_sha256']='different';write_jsonl(right/'predictions.jsonl',[row])
            with self.assertRaisesRegex(ValueError,'NATIVE_PROMPT_TOKENS'):
                report.verify_pair_prompts(left,right)
            row.update(status='model_error',prompt_token_sha256=None)
            write_jsonl(right/'predictions.jsonl',[row])
            self.assertEqual(report.verify_pair_prompts(left,right)['unverifiable_due_to_generation_failure'],1)

    def test_gpu_report_excludes_other_runs_and_preserves_completion_floor(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/'runtime').mkdir();(root/'v4').mkdir()
            cfg=root/'config.json';cfg.write_text(json.dumps({'model_id':'M'}))
            eval_cfg=root/'v2/text/configs/model-evaluation.json';eval_cfg.parent.mkdir(parents=True)
            eval_cfg.write_text(cfg.read_text())
            plan={'runs':{'train':{'config':str(cfg)}},'models':['model']}
            state={'evaluation':{'model-eval':{}}}
            ledger={'active':{'run_id':'unrelated'},'models':{'M':5015},
                    'run_phases':{'train':{'train':10,'reload':2},'model-eval':{'evaluation':3},
                                  'unrelated':{'evaluation':5000}}}
            path=root/'runtime/gpu_budget.json';path.write_text(json.dumps(ledger))
            _,delta,completion=report.gpu_accounting(root,plan,state)
            self.assertEqual(delta,{'M':15})
            ledger['models']['M']+=100;path.write_text(json.dumps(ledger))
            _,delta,second=report.gpu_accounting(root,plan,state)
            self.assertEqual(delta,{'M':15});self.assertEqual(completion,second)
            ledger['models']['M']=1;path.write_text(json.dumps(ledger))
            with self.assertRaisesRegex(ValueError,'REWOUND'):
                report.gpu_accounting(root,plan,state)

    def test_metric_verification_only_tolerates_float_roundoff(self):
        self.assertTrue(report.equivalent({'n':4000,'f1':.24168677508061542}, {'n':4000,'f1':.2416867750806154}))
        self.assertFalse(report.equivalent({'n':4000}, {'n':4001}))
        self.assertFalse(report.equivalent({'f1':.24}, {'f1':.25}))
        self.assertFalse(report.equivalent(float('nan'),float('nan')))

    def test_independent_base_one_pass_has_no_resurrected_v1_time_cap(self):
        original = {'model_id':'model', 'total_gpu_hours':24, 'per_model_gpu_hours':6,
                    'wall_seconds':18000, 'max_length':999}
        corpus = {'freeze_manifest':'new-freeze', 'splits':{
            'aihub71762_category_random_v4/train':{'snapshot_path':'train'},
            'aihub71762_category_random_v4/validation':{'snapshot_path':'dev'}}}
        for phase, steps in [('smoke',20),('full',250)]:
            config = campaign.training_config(original, corpus, Path('/private'), 'model', phase)
            self.assertEqual(config['max_optimizer_steps'], steps)
            self.assertEqual(config['train_samples_cap'],4000)
            self.assertIsNone(config['initial_adapter'])
            self.assertEqual(config['parent_condition'],'original_base')
            self.assertIsNone(config['resource_policy']['total_gpu_hours'])
            self.assertFalse({'total_gpu_hours','per_model_gpu_hours','wall_seconds'} & config.keys())
        self.assertEqual(original['total_gpu_hours'],24)

    def test_review_uses_native_raw_output_and_preserves_blank_human_scores(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = sample('fixture','rev','q','medical_qa','질문','무작위 참조','test')
            data = root/'data.jsonl'; write_jsonl(data,[row])
            protocol = root/'protocol.json'; freeze_protocol([data],protocol,cap=0)
            plan_path=root/'v4/review/plan.json';plan_path.parent.mkdir(parents=True)
            plan_path.write_text(json.dumps({'sample_ids':[row['sample_id']], 'protocol_sha256':sha256(protocol)}))
            folders=[]
            for label in ('base','tuned'):
                folder=root/label;folder.mkdir();folders.append(folder)
                write_jsonl(folder/'predictions.jsonl',[{'sample_id':row['sample_id'],
                            'status':'answered','raw_output':label+' response'}])
                (folder/'run_manifest.json').write_text(json.dumps({'status':'SUCCEEDED'}))
            report.review_packet(root,'model',protocol,*folders)
            path=root/'v4/review/model/review.csv'
            with path.open(encoding='utf-8-sig') as stream: exported=list(csv.DictReader(stream))
            self.assertEqual({exported[0]['output_A'],exported[0]['output_B']},{'base response','tuned response'})
            self.assertEqual(exported[0]['correctness_A'],'')
            before=path.read_bytes()
            report.review_packet(root,'model',protocol,*folders)
            self.assertEqual(before,path.read_bytes())
            predictions=folders[0]/'predictions.jsonl'
            write_jsonl(predictions,[{'sample_id':row['sample_id'],'status':'answered','output':'wrong field'}])
            with self.assertRaisesRegex(ValueError,'OUTPUT_FIELD'):
                report.review_packet(root,'model',protocol,*folders)
            plan_path.write_text(json.dumps({'sample_ids':['changed'], 'protocol_sha256':sha256(protocol)}))
            with self.assertRaisesRegex(ValueError,'PROSPECTIVE_REVIEW'):
                report.review_packet(root,'model',protocol,*folders)


if __name__=='__main__':
    unittest.main()
