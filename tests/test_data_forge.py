import json
import pytest
from mediflow_medvlm.data.forge_export import export_forge
from mediflow_medvlm.data.io import digest,sha256,write_jsonl
from mediflow_medvlm.data.schema import sample


def fixture(tmp_path,extra=False):
    source=tmp_path/'train.jsonl'
    rows=[sample('fixture','revision',str(i),'medical_qa','Question '+str(i),'Source answer '+str(i),'train') for i in range(4)]
    if extra:
        disallowed=sample('kormedmcqa','revision','excluded','medical_mcq','Question','A','train',options={'A':'x'},option_id='A')
        rows.insert(0,disallowed)
    write_jsonl(source,rows)
    f={'schema_version':'medvlm.freeze.v1','splits':{'fixture/train':{'path':str(source),'sha256':sha256(source)}}}
    f['freeze_id']=digest(f);f['status']='SUCCEEDED';f['evidence_level']='CPU_TESTED'
    path=tmp_path/'freeze.json';path.write_text(json.dumps(f));return path,source


def test_export_preserves_provenance_and_never_includes_eval_only_source(tmp_path):
    f,_=fixture(tmp_path,True);result=export_forge(tmp_path,f)
    assert result['rows']==4 and result['bytes']<=128*1024
    exported=[json.loads(x) for x in open(result['data_path'])]
    assert all(set(x)=={'messages'} for x in exported)
    sidecar=json.load(open(result['sidecar_path']))
    assert len(sidecar['records'])==4
    assert all(x['source_id']=='fixture' and x['provenance'] for x in sidecar['records'])
    assert export_forge(tmp_path,f)==result


def test_export_rejects_changed_snapshot_and_row_limit(tmp_path):
    f,source=fixture(tmp_path)
    with pytest.raises(ValueError,match='1..1000'):export_forge(tmp_path,f,1001)
    source.write_text(source.read_text()+'\n')
    with pytest.raises(ValueError,match='checksum'):export_forge(tmp_path,f)


def test_export_does_not_overwrite_existing_mismatched_result(tmp_path):
    f,_=fixture(tmp_path);result=export_forge(tmp_path,f)
    from pathlib import Path
    Path(result['data_path']).write_text('changed')
    with pytest.raises(ValueError,match='immutable export'):export_forge(tmp_path,f)
