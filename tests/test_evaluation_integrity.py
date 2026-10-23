"""Failures that would silently invalidate frozen paired model comparisons."""
import base64
import copy
import json

import pytest

from mediflow_medvlm.evaluation.metrics import aggregate, score_rows
from mediflow_medvlm.evaluation.runner import freeze_protocol, load_protocol, validated_predictions
from mediflow_medvlm.evaluation.runner import verify_adapter_parent
from mediflow_medvlm.evaluation.runner import verify_completed_evaluation
from mediflow_medvlm.runtime import file_hash


def canonical(sid='one'):
    return {
        'sample_id': sid, 'source_id': 'fixture', 'task_type': 'medical_mcq',
        'language': 'en', 'split': 'test',
        'input': {'question': 'Choose the fixture value.', 'options': {'A': 'x', 'B': 'y'}, 'images': []},
        'supervision': {'answer': 'B', 'option_id': 'B'},
        'group_ids': {'document': 'fixture-document-'+sid},
    }


def write_rows(path,rows):
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf-8')


def protocol_fixture(tmp_path, rows=None):
    rows=rows or [canonical()]
    dataset=tmp_path/'test.jsonl';write_rows(dataset,rows)
    output=tmp_path/'protocol.json'
    protocol=freeze_protocol([dataset],output,cap=0)
    return protocol, output


def prediction(protocol):
    return {'sample_id':'one','input_hash':protocol['input_digests']['one'],
            'run_id':'run-one','pair_id':'pair-one','model_identity':'identity-one',
            'status':'answered','parsed_answer':'B','attempted':True}


IDENTITY={'run_id':'run-one','pair_id':'pair-one','identity_hash':'identity-one'}


def test_frozen_dataset_paths_survive_image_path_collection(tmp_path):
    """A local `paths` variable for images must not replace dataset filenames."""
    protocol,output=protocol_fixture(tmp_path)
    assert list(protocol['files'])==[str((tmp_path/'test.jsonl').resolve())]
    _,rows=load_protocol(output)
    assert [r['sample_id'] for r in rows]==['one']


def test_mutated_image_bytes_rejected_without_jsonl_change(tmp_path):
    image=tmp_path/'one.png'
    image.write_bytes(base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aV7cAAAAASUVORK5CYII='))
    row=canonical();row['task_type']='medical_vqa';row['input']={'question':'Fixture image question','images':[str(image)]}
    protocol,output=protocol_fixture(tmp_path,[row]);load_protocol(output)
    assert protocol['image_files']['one'][0]['path']==str(image.resolve())
    image.write_bytes(image.read_bytes()+b'changed-image-bytes')
    with pytest.raises(ValueError,match='frozen image bytes changed'):load_protocol(output)


@pytest.mark.parametrize('field,value',[('run_id','another-run'),('pair_id','another-pair'),('model_identity','another-model')])
def test_stale_prediction_run_and_model_identity_rejected(tmp_path,field,value):
    protocol,_=protocol_fixture(tmp_path)
    pred=prediction(protocol);pred[field]=value
    path=tmp_path/'predictions.jsonl';write_rows(path,[pred])
    with pytest.raises(ValueError,match='another run'):validated_predictions(path,protocol,IDENTITY)


def test_prediction_for_changed_input_rejected(tmp_path):
    protocol,_=protocol_fixture(tmp_path)
    pred=prediction(protocol);pred['input_hash']='stale-input-hash'
    path=tmp_path/'predictions.jsonl';write_rows(path,[pred])
    with pytest.raises(ValueError,match='input identity'):validated_predictions(path,protocol,IDENTITY)


@pytest.mark.parametrize('duplicate',[True,False])
def test_duplicate_or_unscheduled_prediction_ids_rejected(tmp_path,duplicate):
    protocol,_=protocol_fixture(tmp_path)
    pred=prediction(protocol)
    rows=[pred,copy.deepcopy(pred)] if duplicate else [{**pred,'sample_id':'unplanned-id'}]
    path=tmp_path/'predictions.jsonl';write_rows(path,rows)
    with pytest.raises(ValueError,match='duplicate or unscheduled'):validated_predictions(path,protocol,IDENTITY)


def test_missing_predictions_remain_in_scheduled_denominator(tmp_path):
    protocol,output=protocol_fixture(tmp_path,[canonical('one'),canonical('two')])
    path=tmp_path/'predictions.jsonl';write_rows(path,[prediction(protocol)])
    preds=validated_predictions(path,protocol,IDENTITY)
    _,rows=load_protocol(output)
    cohort=aggregate(score_rows(rows,preds))[0]
    assert cohort['scheduled']==2
    assert cohort['attempted']==1
    assert cohort['valid_answers']==1
    assert cohort['coverage']==.5
    assert cohort['end_to_end_correct_fraction']==.5
    assert cohort['statuses']['not_attempted']==1


def test_duplicate_canonical_ids_refused_before_freeze(tmp_path):
    with pytest.raises(ValueError,match='duplicate sample IDs'):protocol_fixture(tmp_path,[canonical(),canonical()])


@pytest.mark.parametrize('mutation',['weights','template','adapter','revision'])
def test_eval_refuses_changed_training_parent_or_adapter(tmp_path,mutation):
    checkpoint=tmp_path/'checkpoints'/'step-1';checkpoint.mkdir(parents=True)
    asset=checkpoint/'adapter_model.safetensors';asset.write_bytes(b'engineering-fixture')
    parent={'model_id':'fixture/model','revision':'fixed','quantization':'none','template_sha256':'template'}
    (checkpoint/'checkpoint.json').write_text(json.dumps({'identity':parent,'artifacts':{'files':{asset.name:file_hash(asset)}}}))
    (tmp_path/'manifest.json').write_text(json.dumps({'model':{'native_files':{'sha256':'native'}}}))
    config={k:parent[k] for k in ('model_id','revision','quantization')}
    meta={'native_files':{'sha256':'native'},'chat_template_sha256':'template'}
    verify_adapter_parent(checkpoint,config,meta)
    if mutation=='weights':meta['native_files']['sha256']='changed'
    elif mutation=='template':meta['chat_template_sha256']='changed'
    elif mutation=='revision':config['revision']='changed'
    else:asset.write_bytes(b'changed')
    with pytest.raises(ValueError):verify_adapter_parent(checkpoint,config,meta)


@pytest.mark.parametrize('mutation',['prediction','metrics','base','missing'])
def test_completed_eval_cache_refuses_artifact_tampering(tmp_path,mutation):
    protocol,_=protocol_fixture(tmp_path)
    preds=tmp_path/'predictions.jsonl';write_rows(preds,[prediction(protocol)])
    metrics=tmp_path/'metrics.json';metrics.write_text('{}')
    manifest={**IDENTITY,'model':{'native_files':{'sha256':'base'}},
              'artifact_hashes':{p.name:file_hash(p) for p in (preds,metrics)}}
    verify_completed_evaluation(manifest,protocol,tmp_path,'base')
    digest='base'
    if mutation=='prediction':preds.write_text('')
    elif mutation=='metrics':metrics.write_text('{"changed":true}')
    elif mutation=='base':digest='changed'
    else:
        preds.write_text('');manifest['artifact_hashes'][preds.name]=file_hash(preds)
    with pytest.raises(ValueError):verify_completed_evaluation(manifest,protocol,tmp_path,digest)
