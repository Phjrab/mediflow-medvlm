import json
from pathlib import Path
import pytest
from mediflow_medvlm.data.schema import sample, validate_record, input_digest
from mediflow_medvlm.data.pipeline import audit, freeze, write_source
from mediflow_medvlm.data.pubmedqa import official_folds, build_pubmedqa
from mediflow_medvlm.data.medquad import build_medquad
from mediflow_medvlm.data.io import download


def row(oid='one',split='train',question='A medical question',answer='yes'):
    return sample('fixture','abc123',oid,'medical_qa',question,answer,split,document='doc-'+oid)


def test_audit_blocks_known_document_leakage(tmp_path):
    a,b=row(),row('two','test',question='Different question')
    b['group_ids']['document']=a['group_ids']['document']
    write_source(tmp_path,'fixture',[a,b],{})
    report=audit(tmp_path)
    assert any(e['reason']=='known_group_cross_split' for e in report['critical_errors'])
    with pytest.raises(ValueError,match='Freeze refused'):freeze(tmp_path)


def test_input_duplicate_key_keeps_medical_distinctions_and_image_identity():
    a,b=row(question='Dose 1.5 mg left eye'),row(question='Dose 15 mg right eye')
    assert input_digest(a)!=input_digest(b)
    a['input']['images']=['one.jpg'];b['input']={**a['input'],'images':['two.jpg']}
    a['group_ids']['image']='one';b['group_ids']['image']='two'
    assert input_digest(a)!=input_digest(b)


def test_gold_must_stay_outside_input_and_korean_source_not_train():
    a=row();a['input']['rationale']='sentinel secret gold'
    assert 'forbidden_input_field' in validate_record(a)
    a=row();a['source_id']='kormedmcqa'
    assert 'eval_source_in_train' in validate_record(a)


def test_official_fold_reproducible_disjoint():
    data={str(i):{'final_decision':['yes','no','maybe'][i%3]} for i in range(1000)}
    folds=official_folds(data)
    assert {k:len(v) for k,v in folds.items()}=={'train':450,'validation':50,'test':500}
    assert set(folds['train']).isdisjoint(folds['test'])
    assert set(folds['validation']).isdisjoint(folds['test'])
    assert official_folds(data)==folds
    assert official_folds(data,1)['test']==folds['test']


def test_pubmedqa_excludes_label_conclusion_and_classifier_prediction(tmp_path):
    (tmp_path/'data').mkdir();(tmp_path/'preprocess').mkdir()
    (tmp_path/'preprocess'/'split_dataset.py').write_text('fixture')
    data={str(i):{'QUESTION':f'Question {i}?','CONTEXTS':['Allowed context','Conclusion sentinel'], 'LABELS':['RESULTS','CONCLUSIONS'],'LONG_ANSWER':'Conclusion sentinel','reasoning_required_pred':'GOLD LEAK','final_decision':['yes','no','maybe'][i%3]} for i in range(1000)}
    (tmp_path/'data'/'ori_pqal.json').write_text(json.dumps(data))
    rows,stats=build_pubmedqa(tmp_path,'revision','2026-10-23')
    assert len(rows)==1000
    assert all(r['input']['context']==['Allowed context'] for r in rows)
    assert all('GOLD LEAK' not in json.dumps(r['input']) for r in rows)
    assert stats['counts']['conclusion_or_question_context_segments']==1000


def test_medquad_removed_subsets_and_answer_conflicts_quarantined(tmp_path):
    for directory,answer in [('10_MPlus_ADAM_QA','Do not reuse'),('1_CancerGov_QA','yes'),('2_GARD_QA','no')]:
        d=tmp_path/directory;d.mkdir()
        (d/'1.xml').write_text(f'<Document id="1" url="https://example.org/{directory}"><QAPairs><QAPair><Question qid="q1">Same input?</Question><Answer>{answer}</Answer></QAPair></QAPairs></Document>')
    rows,stats=build_medquad(tmp_path,'revision','2026-10-23')
    assert rows==[]
    assert stats['counts']['copyright_removed_subset']==1
    assert stats['counts']['label_conflict']==2


def test_download_rejects_nonofficial_and_secret_urls(tmp_path):
    with pytest.raises(ValueError,match='allowlist'):download('https://example.com/data',tmp_path/'x')
    with pytest.raises(ValueError,match='credential'):download('https://huggingface.co/data?token=secret',tmp_path/'x')


def test_reused_medquad_document_qid_keeps_unique_sample_ids_same_group(tmp_path):
    d=tmp_path/'1_CancerGov_QA';d.mkdir()
    for index in range(2):
        (d/f'fragment{index}.xml').write_text(f'<Document id="same_doc" url="https://example.org/article"><QAPairs><QAPair><Question qid="same_qid">Distinct question {index}?</Question><Answer>Answer {index}</Answer></QAPair></QAPairs></Document>')
    rows,_=build_medquad(tmp_path,'revision','2026-10-23')
    assert len(rows)==2
    assert len({r['sample_id'] for r in rows})==2
    assert len({r['group_ids']['document'] for r in rows})==1
    assert len({r['split'] for r in rows})==1


def test_audit_checks_image_checksum(tmp_path):
    image=tmp_path/'image.png';image.write_bytes(b'changed')
    a=row();a['input']['images']=[str(image)];a['metadata']['image_sha256']='wrong'
    write_source(tmp_path,'fixture',[a],{})
    assert any(e['reason']=='image_checksum_mismatch' for e in audit(tmp_path)['critical_errors'])


def test_empty_dataset_cannot_be_declared_data_ready(tmp_path):
    with pytest.raises(ValueError,match='Freeze refused'):freeze(tmp_path)


def test_public_freeze_summary_redacts_data_root_without_changing_evidence(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_root=tmp_path/'private-user'/'data'
    original=row(question='UNIQUE RAW QUESTION MUST REMAIN PRIVATE')
    write_source(data_root,'fixture',[original],{})
    frozen=freeze(data_root)
    public=json.loads((tmp_path/'reports/public_summary/data_freeze.json').read_text())
    local=json.loads((data_root/'manifests/current_freeze.json').read_text())
    assert str(data_root) not in json.dumps(public)
    assert 'UNIQUE RAW QUESTION' not in json.dumps(public)
    assert public['splits']['fixture/train']['path'].startswith('${MEDVLM_DATA_ROOT}')
    assert local['splits']['fixture/train']['path'].startswith(str(data_root))
    assert local['freeze_id']==frozen['freeze_id']


def test_freeze_cli_logs_counts_and_digest_without_ordered_ids(tmp_path,monkeypatch,capsys):
    from mediflow_medvlm.data import __main__ as cli
    monkeypatch.setattr(cli,'freeze',lambda _: {'status':'SUCCEEDED','evidence_level':'CPU_TESTED','freeze_id':'digest','record_count':10000,'critical_error_count':0,'manifest_path':'local/manifest.json','splits':{'fixture/train':{'count':10000,'ordered_sample_ids':['PRIVATE-ID']*10000}}})
    assert cli.main(['freeze','--data-root',str(tmp_path)])==0
    output=capsys.readouterr().out
    assert 'PRIVATE-ID' not in output and len(output)<500
    assert json.loads(output)['counts']['fixture/train']==10000
