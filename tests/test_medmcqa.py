import json
import zipfile

import pytest

from mediflow_medvlm.data.medmcqa import option_key, prepare
from mediflow_medvlm.data.io import sha256


def record(sid, question='Fixture 10 mg, not 20 mg?', cop=1):
    return {'id':sid,'question':question,'opa':'A choice','opb':'B choice','opc':'C choice','opd':'D choice',
            'cop':cop,'choice_type':'single','exp':None,'subject_name':'Fixture'}


def archive(path, train, dev, test):
    with zipfile.ZipFile(path,'w') as z:
        for name,rows in [('train.json',train),('dev.json',dev),('test.json',test)]:
            z.writestr(name,''.join(json.dumps(r)+'\n' for r in rows))
    return sha256(path)


def test_option_order_key_keeps_numbers_and_negation():
    a=record('a');b=record('b');b['opa'],b['opb']=b['opb'],b['opa']
    assert option_key(a)==option_key(b)
    assert option_key(a)!=option_key(record('c','Fixture 20 mg, not 10 mg?'))
    assert option_key(a)!=option_key(record('d','Fixture 10 mg, 20 mg?'))


def test_optional_cohort_excludes_heldout_conflict_and_unlabeled_test(tmp_path,monkeypatch):
    monkeypatch.setattr('mediflow_medvlm.data.medmcqa.disk_guard',lambda *a,**k:None)
    p=tmp_path/'data.zip'
    train=[record('good','Good?'),record('dev_overlap','Dev?'),record('test_overlap','Test?'),
           record('conflict1','Conflict?',1),record('conflict2','Conflict?',2),
           {**record('multi','Multi?'),'choice_type':'multi'},record('duplicate','Good?')]
    dev=[record('dev','Dev?')];test=[record('test','Test?')];test[0].pop('cop')
    digest=archive(p,train,dev,test)
    report=prepare(tmp_path,p,expected_archive_sha256=digest)
    assert {k:v['count'] for k,v in report['canonical_files'].items()}=={'train':1,'validation':1,'test':0}
    assert report['excluded_counts']['train']['official_heldout_input_overlap']==2
    assert report['excluded_counts']['train']['conflicting_answer_text']==2
    assert report['excluded_counts']['train']['not_single_choice']==1
    assert report['excluded_counts']['train']['within_split_exact_input_duplicate']==1
    assert not report['core_ko_mix_included'] and not report['training_executed']
    assert prepare(tmp_path,p,expected_archive_sha256=digest)==report
    with pytest.raises(ValueError,match='identity'):
        prepare(tmp_path,p,expected_archive_sha256='wrong')


def test_archive_extra_files_are_not_silently_imported(tmp_path,monkeypatch):
    monkeypatch.setattr('mediflow_medvlm.data.medmcqa.disk_guard',lambda *a,**k:None)
    p=tmp_path/'data.zip';archive(p,[],[],[])
    with zipfile.ZipFile(p,'a') as z:z.writestr('../unsafe.json','{}')
    with pytest.raises(ValueError,match='unexpected'):
        prepare(tmp_path,p,expected_archive_sha256=sha256(p))
