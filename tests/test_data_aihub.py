"""Synthetic safety/parser fixtures; these tests are not real source evidence."""
import io
import json
from pathlib import Path
import zipfile

import pytest

from mediflow_medvlm.data import aihub


def test_file_bom_and_key_bom_are_distinct_and_collisions_rejected():
    row,events=aihub._safe_json(b'\xef\xbb\xbf{"\\ufeffqa_id":1,"answer":"fixture"}')
    assert row['qa_id']==1 and events=={'file_bom':1,'key_bom':1}
    with pytest.raises(ValueError,match='COLLISION'):
        aihub._safe_json('{"qa_id":1,"\\ufeffqa_id":2}')
    with pytest.raises(ValueError,match='COLLISION'):
        aihub._safe_json('{"qa_id":1,"qa_id":2}')


def test_mcq_line_boundary_numbers_units_and_answer_mismatch():
    question='40세의 조건에서 2 mg이 아닌 것은?\n1) 1 mg\n2) 2 mg\n3) 3 mg'
    stem,options,label=aihub.parse_71875_mcq(question,'2) 2 mg')
    assert stem=='40세의 조건에서 2 mg이 아닌 것은?'
    assert options=={'A':'1 mg','B':'2 mg','C':'3 mg'} and label=='B'
    for answer in ['2) 3 mg','1, 2','2) 2 mg 또는 3 mg']:
        with pytest.raises(ValueError):aihub.parse_71875_mcq(question,answer)
    with pytest.raises(ValueError):aihub.parse_71875_mcq('1) 20세\n2) 40세','1) 20세')


def test_mcq_ambiguous_numbered_stem_is_not_guessed():
    with pytest.raises(ValueError):
        aihub.parse_71875_mcq('조건?\n1) 병력\n2) 용량\n선택:\n1) A\n2) B','1) A')


def _records(question='조건?\n1) A\n2) B',answer='1) A',qid=1,split='train',asset='file.json'):
    return aihub.records_71875([({'qa_id':qid,'domain':1,'q_type':1,'question':question,'answer':answer},{'relative_path':asset,'sha256':'0'*64,'original_split':split})],revision='v2fixture',license_ref='fixture_only',fetched_at='2026-10-26')


def test_integer_qa_id_not_global_document_id_and_bad_row_quarantined():
    a,_=_records(asset='one.json');b,_=_records(asset='two.json')
    assert a[0]['sample_id']!=b[0]['sample_id']
    assert a[0]['group_ids']['document'] is None
    assert a[0]['task_type']=='medical_mcq' and a[0]['supervision']['option_id']=='A'
    bad,stats=_records(answer='2) A')
    assert not bad and stats['excluded_counts']=={'MCQ_ANSWER_TEXT_MISMATCH':1}


def test_official_heldout_wins_and_group_split_is_repeatable():
    train,_=_records(asset='train.json');valid,_=_records(split='validation',asset='valid.json')
    rows,excluded=aihub.deduplicate_and_split(train+valid)
    assert len(rows)==1 and rows[0]['original_split']=='validation'
    assert rows[0]['split'] in {'validation','test'} and excluded
    assert aihub.deduplicate_and_split(valid)[0][0]['split']==rows[0]['split']


def test_full_question_options_permutation_and_medical_values_preserved():
    protected,_=_records(question='2 mg이 아닌 조건?\n1) A\n2) B',split='test')
    reordered,_=_records(question='2 mg이 아닌 조건?\n1) B\n2) A',answer='2) A',asset='reordered.json')
    changed,_=_records(question='3 mg인 조건?\n1) A\n2) B',asset='changed.json')
    kept,report=aihub.leakage_audit(reordered+changed,protected)
    assert [x['sample_id'] for x in kept]==[changed[0]['sample_id']]
    assert any(m['kind']=='option_order_permutation' for m in report['confirmed_overlaps'][0]['matches'])
    assert protected[0]['split']=='test'


def test_near_candidates_are_quarantined_not_called_confirmed():
    q='긴 조건을 구성하기 위해 임상 문맥을 그대로 보존하며 숫자 20 mg과 부정 표현 아닌 것을 포함한 합성 문장입니다'
    protected,_=_records(question=q+'?\n1) A\n2) B',split='test')
    candidate,_=_records(question=q+'!\n1) A\n2) B',asset='near.json')
    kept,report=aihub.leakage_audit(candidate,protected)
    assert not kept and report['near_candidate_rows']==1 and report['confirmed_overlap_rows']==0
    assert report['near_candidates_are_confirmed_duplicates'] is False


def test_archive_traversal_rejected_and_root_marker_streaming_only(tmp_path,monkeypatch):
    monkeypatch.setattr(aihub,'disk_guard',lambda *args:None)
    source=tmp_path/'source.zip'
    with zipfile.ZipFile(source,'w') as z:z.writestr('../escaped.json','{}')
    with pytest.raises(ValueError,match='UNSAFE_ARCHIVE_PATH'):
        aihub.import_aihub([source],tmp_path/'data','71875',version='fixture')
    assert not (tmp_path/'escaped.json').exists()
    with pytest.raises(ValueError):aihub._safe_member('/rooted.json')
    assert str(aihub._safe_member('/rooted.json',streaming_root_marker=True))=='rooted.json'


def test_zip_import_preserves_original_and_excludes_model_eval_paths(tmp_path,monkeypatch):
    monkeypatch.setattr(aihub,'disk_guard',lambda *args:None)
    source=tmp_path/'source.zip'
    with zipfile.ZipFile(source,'w') as z:
        z.writestr('Training/TL_fixture/one.json',json.dumps({'qa_id':1}))
        z.writestr('models/KorMedMCQA/test.json','{}')
    original=source.read_bytes()
    result=aihub.import_aihub([source],tmp_path/'data','71875',version='fixture')
    docs=list(aihub.json_documents(result))
    assert len(docs)==1 and docs[0][1]['original_split']=='train'
    assert source.read_bytes()==original and result['originals_preserved']
    assert not (tmp_path/'data'/'Training').exists()


def test_split_archive_order_missing_parts_and_originals(tmp_path,monkeypatch):
    monkeypatch.setattr(aihub,'disk_guard',lambda *args:None)
    data=io.BytesIO()
    with zipfile.ZipFile(data,'w') as z:z.writestr('fixture.json','{"value":1}')
    payload=data.getvalue();parts=[]
    for i,(a,b) in enumerate([(0,len(payload)//3),(len(payload)//3,2*len(payload)//3),(2*len(payload)//3,len(payload))],1):
        p=tmp_path/f'data.zip.part{i:03}';p.write_bytes(payload[a:b]);parts.append(p)
    with pytest.raises(ValueError,match='MISSING'):aihub.merge_parts([parts[0],parts[2]],tmp_path/'missing')
    target,receipt=aihub.merge_parts(list(reversed(parts)),tmp_path/'merged')
    assert target.read_bytes()==payload and all(p.exists() for p in parts)
    assert receipt['original_parts_preserved']


def test_import_directory_of_parts_and_complete_zip_never_double_merges(tmp_path,monkeypatch):
    monkeypatch.setattr(aihub,'disk_guard',lambda *args:None)
    inputs=tmp_path/'inputs';inputs.mkdir();data=io.BytesIO()
    with zipfile.ZipFile(data,'w') as z:z.writestr('Training/TL_fixture/one.json','{"qa_id":1}')
    payload=data.getvalue();half=len(payload)//2
    (inputs/'data.zip.part001').write_bytes(payload[:half]);(inputs/'data.zip.part002').write_bytes(payload[half:])
    imported=aihub.import_aihub([inputs],tmp_path/'data','71875',version='fixture')
    assert len(list(aihub.json_documents(imported)))==1 and len(imported['split_archives'])==1
    (inputs/'data.zip').write_bytes(payload)
    imported=aihub.import_aihub([inputs],tmp_path/'data','71875',version='fixture')
    assert len(list(aihub.json_documents(imported)))==1 and not imported['split_archives']


def test_71762_explicit_one_to_many_preserves_parent_and_rejects_cartesian():
    contract={'question_id':'qid','answer_parent_key':'qid','answer_id':'aid','cardinality':'one_to_many','evidence_sha256':'fixture_evidence','augmentation_parent_key':'original_parent'}
    rows,unresolved=aihub.join_71762([{'qid':'q1','original_parent':'p1'}],[{'qid':'q1','aid':'a1'},{'qid':'q1','aid':'a2'}],contract=contract)
    assert len(rows)==2 and not unresolved and all(r['parent_group']=='p1' for r in rows)
    with pytest.raises(ValueError,match='cardinality'):
        aihub.join_71762([{'qid':'q1'}],[{'qid':'q1','aid':'a1'},{'qid':'q1','aid':'a2'}],contract={**contract,'cardinality':'one_to_one'})
    with pytest.raises(ValueError,match='explicit'):
        aihub.join_71762([],[],contract={**contract,'question_id':'disease'})
    shared,missing=aihub.join_71762([{'qid':'q1','answer_ref':'a1'},{'qid':'q2','answer_ref':'a1'}],[{'aid':'a1'}],contract={**contract,'cardinality':'many_to_one','question_answer_key':'answer_ref'})
    assert len(shared)==2 and not missing and shared[0]['parent_group']==shared[1]['parent_group']=='a1'


def test_balanced_selection_and_freeze_are_immutable_and_group_safe(tmp_path):
    records=[]
    for i in range(6):
        rows,_=_records(question=f'합성 조건 {i}?\n1) A\n2) B',asset=f'{i}.json')
        rows[0]['metadata']['domain']=i%2;records+=rows
    chosen,selection=aihub.balanced_select(records,{'1':4})
    assert len(chosen)==4 and selection['domain_counts']=={'0':2,'1':2}
    assert aihub.balanced_select(list(reversed(records)),{'1':4})[1]==selection
    frozen=aihub.freeze_corpus(tmp_path,'ko_fixture',{'train':chosen,'validation':[],'test':[]},provenance={'synthetic_only':True})
    assert aihub.freeze_corpus(tmp_path,'ko_fixture',{'train':chosen,'validation':[],'test':[]},provenance={'synthetic_only':True})==frozen
    Path(frozen['splits']['ko_fixture/train']['path']).write_text('tampered')
    with pytest.raises(ValueError,match='FROZEN_FILE_CHANGED'):
        aihub.freeze_corpus(tmp_path,'ko_fixture',{'train':chosen,'validation':[],'test':[]},provenance={'synthetic_only':True})


def test_token_audit_matches_native_text_encoding_not_batchencoding_len(monkeypatch):
    import sys
    from types import SimpleNamespace
    class Tokenizer:
        def get_chat_template(self):return 'fixture_template'
        def apply_chat_template(self,messages,*,tokenize,add_generation_prompt,enable_thinking):
            assert tokenize is False and enable_thinking is False
            return 'abc' if add_generation_prompt else 'abcde'
        def __call__(self,text,*,add_special_tokens):
            assert add_special_tokens is False
            return {'input_ids':list(text),'attention_mask':[1]*len(text)}
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a,**k:Tokenizer())))
    monkeypatch.setitem(sys.modules,'huggingface_hub',SimpleNamespace(try_to_load_from_cache=lambda *a,**k:None))
    rows,_=_records()
    result=aihub.token_audit(rows,{'fixture':{'model_id':'fixture','revision':'fixture'}},max_length=5)
    assert result['common_eligible_count']==1
    measured=next(iter(result['encodings'].values()))['samples'][rows[0]['sample_id']]
    assert measured['sequence_tokens']==5 and measured['supervised_tokens']==2
