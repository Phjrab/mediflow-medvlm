"""Only completed verified aggregate cells may become plotted numbers."""
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('plot_v2',Path(__file__).parents[1]/'scripts/plot_v2.py')
plot=importlib.util.module_from_spec(spec);spec.loader.exec_module(plot)


def summary(cells):
    return {'schema_version':'medvlm.public-report.v2','evaluations':cells}


def vqa(condition='base',status='SUCCEEDED',verified=True,phase='slake'):
    return {'model':'qwen35_2b','condition':condition,'phase':phase,'image_condition':'image',
            'status':status,'verified':verified,'protocol_hash':'a'*64,
            'metrics':{'vqa':[{'source_id':'slake','answer_type':'ALL','scheduled':10,'v2_normalized_lexical_em':.4}]}}


def test_vqa_incomplete_and_diagnostics_are_absent_not_zero():
    selected=plot.select_vqa(summary([vqa(),vqa('tuned_v1','RUNNING'),vqa('tuned_v1',verified=False),vqa(phase='noimage')]))
    assert selected=={('qwen35_2b','slake','ALL','base'):{'value':.4,'support':10,'protocol_hash':'a'*64}}
    assert ('qwen35_2b','slake','ALL','tuned_v1') not in selected
    assert plot.select_vqa(summary([]))=={}


def test_text_label_accuracy_coverage_and_only_mcq_are_selected():
    cell={'model':'qwen3_17b','condition':'ko','phase':'text','verified':True,'status':'SUCCEEDED','protocol_hash':'b'*64,
          'metrics':{'kormedmcqa':{'scheduled':3009,'micro_end_to_end_correct_fraction':.3,'coverage':.8},
                     'aihub_q_types':[{'q_type':'1','task_type':'medical_mcq','scheduled':777,'end_to_end_correct_fraction':.2},
                                      {'q_type':'2','task_type':'medical_qa','scheduled':83,'end_to_end_lexical_exact_match':.9}]}}
    selected=plot.select_text(summary([cell]));assert len(selected)==3
    assert selected[('qwen3_17b','ko','aihub_mcq_accuracy')]['support']==777
    cell['status']='RUNNING';assert plot.select_text(summary([cell]))=={}


def test_invalid_or_duplicate_verified_metric_is_rejected():
    with pytest.raises(ValueError,match='duplicate'):plot.select_vqa(summary([vqa(),vqa()]))
    broken=vqa();broken['metrics']['vqa'][0]['v2_normalized_lexical_em']=float('nan')
    with pytest.raises(ValueError,match='finite'):plot.select_vqa(summary([broken]))
