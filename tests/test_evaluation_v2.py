import copy,json
from types import SimpleNamespace
import pytest
from mediflow_medvlm.evaluation.v2 import (VQA_PROMPT,evaluation_text,encode_evaluation_sample,normalize_vqa,promote_protocol,generation_finish)
from mediflow_medvlm.evaluation.runner import freeze_protocol,load_protocol
from mediflow_medvlm.evaluation.metrics import score_rows
from mediflow_medvlm.runtime import file_hash


def fixture():
 return {'sample_id':'fixture-vqa','source_id':'fixture','split':'validation','task_type':'medical_vqa','language':'en',
         'input':{'question':'Is the measured value 1.5 mg?','images':[]},'supervision':{'answer':'No.'},
         'metadata':{'answer_type':'SECRET_METADATA','rationale':'SECRET_RATIONALE'}}

class Processor:
 def __init__(self,extra=None):self.extra=extra or {}
 def apply_chat_template(self,messages,**kw):self.messages=messages;return json.dumps(messages)
 def __call__(self,**kw):self.kwargs=kw;return {'input_ids':SimpleNamespace(shape=(1,8)),**self.extra}


def test_v2_no_gold_or_metadata_and_no_image_tensors():
 a=fixture();b=copy.deepcopy(a);b['supervision']['answer']='SECRET';b['metadata']['answer_type']='different'
 assert evaluation_text(a,VQA_PROMPT)==evaluation_text(b,VQA_PROMPT)
 proc=Processor();encoded,meta=encode_evaluation_sample(a,proc,image_condition='no_image')
 assert meta['images']==[] and 'images' not in proc.kwargs and 'SECRET' not in proc.kwargs['text'][0]
 with pytest.raises(ValueError,match='UNEXPECTED_NO_IMAGE'):
  encode_evaluation_sample(a,Processor({'pixel_values':object()}),image_condition='no_image')
 with pytest.raises(ValueError,match='CONTEXT_OVERFLOW'):
  encode_evaluation_sample(a,Processor(),image_condition='no_image',max_length=4)
 with pytest.raises(ValueError,match='training labels'):
  encode_evaluation_sample(a,Processor(),image_condition='no_image',include_labels=True)


def test_v2_normalization_keeps_internal_numbers_units_and_negation():
 assert normalize_vqa(' NO.  ')=='no'
 assert normalize_vqa('not 1.5 mg.')=='not 1.5 mg'
 assert normalize_vqa('yes, but no.')=='yes, but no'
 assert normalize_vqa('1.5')!='15'
 a=fixture();p=[{'sample_id':a['sample_id'],'status':'answered','parsed_answer':'No'}]
 assert score_rows([a],p)[0]['exact_match']==0
 assert score_rows([a],p,vqa_normalizer=normalize_vqa)[0]['exact_match']==1
 p[0]['parsed_answer']='No, because this is a fixture.'
 assert score_rows([a],p,vqa_normalizer=normalize_vqa)[0]['exact_match']==0


def test_eos_at_generation_budget_is_not_reported_as_truncation():
 assert generation_finish(128,2,128,[2,3])=='stop'
 assert generation_finish(128,3,128,3)=='stop'
 assert generation_finish(128,8,128,[2,3])=='length'
 assert generation_finish(4,8,128,None)=='stop'


def test_v2_protocol_cannot_replace_v1_or_mutate_generation(tmp_path):
 a=fixture();a['task_type']='medical_qa'
 data=tmp_path/'validation.jsonl';data.write_text(json.dumps(a)+'\n')
 oldpath=tmp_path/'v1.json';old=freeze_protocol([data],oldpath,split='validation',cap=1);before=file_hash(oldpath)
 newpath=tmp_path/'v2.json';new=promote_protocol(old,newpath,prompt_policy=VQA_PROMPT,previously_observed_test=False)
 assert old['max_new_tokens']['medical_vqa']==32 and new['max_new_tokens']['medical_vqa']==128
 assert file_hash(oldpath)==before and new['protocol_hash']!=old['protocol_hash']
 assert load_protocol(newpath)[0]==new
 with pytest.raises(ValueError,match='overwrite'):
  promote_protocol(old,newpath,prompt_policy=VQA_PROMPT,max_vqa_tokens=256)
 with pytest.raises(ValueError,match='overwrite'):
  promote_protocol(old,oldpath,prompt_policy=VQA_PROMPT)
