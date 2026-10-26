import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('v1_reanalysis',Path(__file__).parents[1]/'scripts/analyze_v1_followup.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def sample(sid='a'):
    return {'sample_id':sid,'source_id':'kormedmcqa','task_type':'medical_mcq','language':'ko',
            'supervision':{'answer':'answer','option_id':'A'},'group_ids':{}}


def pred(sid,status,answer):
    return {'sample_id':sid,'status':status,'parsed_answer':answer}


def test_validity_partitions_reconcile_exact_delta():
    samples=[sample(x) for x in 'abcd']
    a=[pred('a','answered','B'),pred('b','invalid_format',None),pred('c','answered','A'),pred('d','oom',None)]
    b=[pred('a','answered','A'),pred('b','answered','A'),pred('c','invalid_format',None),pred('d','oom',None)]
    parts=module.correctness_partition(samples,a,b)
    assert [p['support'] for p in parts]==[1,1,1,1]
    assert sum(p['net_correct_delta'] for p in parts)==1
    assert sum(p['contribution_to_total_delta'] for p in parts)==.25


def test_reference_phrase_does_not_match_inside_word():
    s=sample();s.update(source_id='fixture',task_type='medical_vqa',metadata={'answer_type':'CLOSED'})
    s['supervision']={'answer':'no'}
    p=pred('a','answered','normal');p.update(raw_output='normal',finish_reason='length')
    out=module.vqa_diagnostics([s],[p])[0]
    assert out['reference_phrase_contains_count']==0 and out['truncated_count']==1
    p.update(raw_output='No, but this is just a fixture.')
    assert module.vqa_diagnostics([s],[p])[0]['reference_phrase_contains_count']==1
