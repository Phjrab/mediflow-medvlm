import json
import math
from pathlib import Path

import pytest
from mediflow_medvlm import telemetry as t
from mediflow_medvlm import telemetry_import as imp


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value)); return path


def metrics(size):
    counts = [16] * (size // 16) + ([size % 16] if size % 16 else [])
    return {"example_losses": [float(i % 11) / 10 for i in range(size)], "update_example_counts": counts,
        "optimizer_steps": len(counts), "samples_processed": size, "loss_mean": sum(float(i % 11) / 10 for i in range(size)) / size,
        "example_supervised_tokens": [i % 5 + 1 for i in range(size)], "supervised_tokens": sum(i % 5 + 1 for i in range(size)),
        "sequence_tokens": size * 10, "learning_rates": [1. - i / len(counts) for i in range(len(counts))],
        "gradient_norms_before_clip": [.2] * len(counts), "final_learning_rate": 0., "resource": {"wall_seconds": 123., "peak_allocated_bytes": 999}}


@pytest.mark.parametrize('size,last,updates',[(7997,13,500),(1979,11,124)])
def test_actual_final_counts_exact_loss_and_lr_boundaries_no_invented_history(size,last,updates):
    source=metrics(size); rows=imp.reconstruct_v6(source)
    assert len(rows)==updates and rows[-1]['batch_examples']==last
    assert rows[-1]['samples_processed']==size
    assert rows[-1]['train_loss']==sum(source['example_losses'][-last:])/last
    assert rows[-1]['supervised_tokens_cumulative']==source['supervised_tokens']
    assert rows[0]['lr_used']==1. and rows[0]['lr_next']==source['learning_rates'][1]
    assert rows[-1]['lr_next']==0.
    assert all(x['step_wall_seconds'] is None and x['cuda_allocated_bytes'] is None and x['sequence_tokens'] is None for x in rows)
    assert math.isclose(sum(x['train_loss']*x['batch_examples'] for x in rows)/size,source['loss_mean'])


@pytest.mark.parametrize('kind',['loss_length','counts','rates','norms','inf_lr','inf_norm','nan_last','negative_tokens','float_tokens'])
def test_invalid_boundaries_and_nonfinite_scalars_rejected(kind):
    source=metrics(29)
    if kind=='loss_length':source['example_losses'].pop()
    elif kind=='counts':source['update_example_counts'][-1]=16
    elif kind=='rates':source['learning_rates'].pop()
    elif kind=='norms':source['gradient_norms_before_clip'].pop()
    elif kind=='inf_lr':source['learning_rates'][0]=float('inf')
    elif kind=='inf_norm':source['gradient_norms_before_clip'][0]=float('nan')
    elif kind=='nan_last':source['final_learning_rate']=float('nan')
    elif kind=='negative_tokens':source['example_supervised_tokens'][0]=-1
    elif kind=='float_tokens':source['example_supervised_tokens'][0]=1.5
    with pytest.raises(ValueError):imp.reconstruct_v6(source)


def test_legacy_scheduler_next_and_timestamp_are_not_relabelled():
    event={'event':'optimizer_update','step':1,'samples':13,'loss':2.,'lr':.01,'gradient_norm':.4,'target_tokens':40,'seconds':2.,'at':'2026-01-01T00:00:00+00:00'}
    row,at=imp.legacy_steps([event],1,13)[0]
    assert row['lr_used'] is None and row['lr_next']==.01 and row['samples_per_second']==6.5 and at==event['at']
    assert row['availability']['lr_used']['reason']=='legacy_lr_recorded_after_scheduler_step'


def test_native_import_hash_proof_idempotency_private_text_and_missing_times(tmp_path):
    source=tmp_path/'source'; source.mkdir()
    write(source/'config.json',{'run_id':'real-v6','job_id':'real-v6','task':'text','model_id':'native/model','recipe':{'seed':1}})
    write(source/'train.metrics.json',metrics(29))
    write(source/'completion.json',{'status':'SUCCEEDED','artifacts':[{'path':'/private/train.metrics.json','sha256':t.file_hash(source/'train.metrics.json')}]})
    expected={'train.metrics.json':t.file_hash(source/'train.metrics.json')}
    before={p.name:p.read_bytes() for p in source.iterdir()}
    result=imp.import_directory(source,tmp_path/'out','medvlm-v6',expected_hashes=expected)
    assert result['steps']==2 and result['reconciliation']=='passed'
    assert imp.import_directory(source,tmp_path/'out','medvlm-v6',expected_hashes=expected)['imported'] is False
    assert before=={p.name:p.read_bytes() for p in source.iterdir()}
    root=tmp_path/'out'/'medvlm-v6'/'real-v6'/result['attempt_id']
    rows=[json.loads(x) for x in (root/'steps.jsonl').read_text().splitlines()]
    assert rows[0]['emitted_at'] is None and rows[0]['monotonic_ns'] is None
    assert rows[0]['provenance']=='historical_reconstructed'
    assert rows[0]['source_artifact']['sha256']==expected['train.metrics.json']
    with pytest.raises(ValueError,match='SOURCE_RECEIPT_HASH_MISMATCH'):
        imp.import_directory(source,tmp_path/'other','medvlm-v6',expected_hashes={'train.metrics.json':'0'*64})


def test_real_evaluator_aggregate_allowlist_does_not_export_scores_answers(tmp_path):
    source=tmp_path/'eval'; source.mkdir()
    write(source/'run_manifest.json',{'run_id':'native-tuned','status':'SUCCEEDED','model_id':'native/model','split':'validation','protocol_hash':'a'*64})
    write(source/'metrics.json',{'protocol_hash':'a'*64,'scores':[{'gold':'PRIVATE-GOLD','prediction':'PRIVATE-PRED'}],
        'cohorts':[{'source_id':'source','task_type':'medical_mcq','language':'ko','scheduled':8,'attempted':7,'valid_answers':5,
            'coverage':.625,'answered_accuracy':.8,'end_to_end_correct_fraction':.5,'statuses':{'answered':5,'invalid_format':2,'not_attempted':1},'not_attempted_count':1,
            'raw_answer':'PRIVATE','e2e_latency_seconds':{'n':7,'median':.3,'min':.2,'max':1.,'scope':'input_and_generation'}}]})
    result=imp.import_directory(source,tmp_path/'out','medvlm-v1')
    root=tmp_path/'out'/'medvlm-v1'/'native-tuned'/result['attempt_id']
    exported=(root/'evaluations.json').read_text()
    assert 'PRIVATE' not in exported and '/private' not in exported
    evaluation=json.loads(exported)['evaluations'][0]
    assert evaluation['scheduled']==8 and evaluation['valid']==5 and evaluation['invalid']==2 and evaluation['unattempted']==1
    assert evaluation['metrics']['answered_accuracy']==.8 and evaluation['receipt_verified'] is None
    assert evaluation['status']=='final' and evaluation['split']=='validation'


def test_actual_image_specific_hash_preserved_and_text_has_no_image_hash():
    row={'sample_id':'id','input':{'images':['/private/image.jpg']},'provenance':{'local_asset_sha256':'a'*64,'image_sha256':'b'*64},'metadata':{'image_sha256':'b'*64}}
    assert imp.metadata_row(row,'freeze')['image_sha256']=='b'*64
    row['input']['images']=[]
    assert imp.metadata_row(row,'freeze')['image_sha256'] is None


def test_freeze_metadata_only_preserves_native_identity_and_size(tmp_path):
    source=tmp_path/'freeze';source.mkdir()
    freeze={'axes':{}}
    for task,size in [('text',1101),('vqa',17)]:
        p=source/(task+'.train.jsonl')
        with p.open('w') as f:
            for i in range(size):
                f.write(json.dumps({'sample_id':str(i),'source_id':'native','source_revision':'pinned','split':'train','task_type':task,
                    'group_ids':{'document':'group'},'input':{'text':'PRIVATE QUESTION','images':[]},'supervision':{'answer':'PRIVATE ANSWER'},
                    'provenance':{'license_ref':'restricted-license','local_asset_sha256':'a'*64},'derivation':'native','review_status':'not_reviewed','restrictions':['no-redistribution']})+'\n')
        freeze['axes'][task]={'common_count':size,'train_sha256':t.file_hash(p)}
    freeze['freeze_id']=t.digest(freeze);write(source/'freeze.json',freeze)
    result=imp.export_freeze(source,tmp_path/'out')
    assert result[0]['record_count']==1101
    dataset=tmp_path/'out/datasets/v6-common-text'
    metadata=(dataset/'records.metadata.jsonl').read_text()
    assert len(metadata.splitlines())==1101 and len(metadata)>128*1024 and 'PRIVATE' not in metadata
    manifest=json.loads((dataset/'manifest.json').read_text())
    assert manifest['frozen'] and manifest['provenance_preserved'] and manifest['split_counts']=={'train':1101,'validation':0,'test':0}
    assert imp.export_freeze(source,tmp_path/'out')[0]['imported'] is False


def test_protocol_hash_match_supplies_original_split_version_and_paired_ci(tmp_path):
    root=tmp_path/'data'; root.mkdir()
    protocol={'protocol_version':'medvlm.eval.v2','split':'validation','sample_ids':['id'],'input_digests':{'id':'a'*64},'reference_digest':'b'*64,'files':{'/private/data.jsonl':'c'*64}}
    protocol['protocol_hash']=t.digest(protocol)
    write(root/'manifests/evaluation/protocol.json',protocol)
    catalog=imp.load_protocol_catalog(root)
    assert catalog[protocol['protocol_hash']]['split']=='validation'
    assert '/private' not in json.dumps(catalog)
    source=root/'v2/text/evaluations/eval';source.mkdir(parents=True)
    write(source/'run_manifest.json',{'run_id':'tuned-eval','status':'SUCCEEDED','model_id':'native/model'})
    write(source/'metrics.json',{'protocol_hash':protocol['protocol_hash'],'cohorts':[{'source_id':'source','task_type':'medical_mcq','language':'ko','scheduled':1,'attempted':1,'valid_answers':1}]})
    result=imp.import_directory(source,tmp_path/'out','medvlm-v2',protocol_catalog=catalog)
    row=json.loads((tmp_path/'out/medvlm-v2/tuned-eval'/result['attempt_id']/'evaluations.json').read_text())['evaluations'][0]
    assert row['split']=='validation' and row['protocol_version']=='medvlm.eval.v2'
    assert row['data_hash']==catalog[protocol['protocol_hash']]['data_hash']
    write(root/'v2/text/pair.json',{'base_run':'base','tuned_run':str(source),'comparison':[{'source_id':'source','task_type':'medical_mcq','language':'ko','n':1,'delta':.1,'ci95':[-.1,.2],'bootstrap_unit':'observed_source_group','bootstrap_resamples':2000,'training_seed_uncertainty_measured':False,'metric':'end_to_end_correct_fraction'}]})
    imported=imp.import_comparisons(root,tmp_path/'out')
    pair=tmp_path/'out/medvlm-v2'/imported[0]['run_id']/imported[0]['attempt_id']/'evaluations.json'
    assert json.loads(pair.read_text())['evaluations'][0]['metrics']['ci95']==[-.1,.2]


def test_native_image_condition_and_renderer_are_preserved_in_cohort_identity():
    metrics_source={'protocol_hash':'a'*64,'cohorts':[{'source_id':'source','task_type':'medical_vqa','language':'en','scheduled':2,'attempted':2,'valid_answers':1}]}
    manifest={'run_id':'same-run','model':'native/model','precision':'nf4','data_hash':'b'*64}
    image=imp.evaluations_from(metrics_source,{'image_condition':'image','renderer_version':'native.v1'},manifest,None)[0]
    ablation=imp.evaluations_from(metrics_source,{'image_condition':'no_image','renderer_version':'medvlm.no-image-diagnostic.v1'},manifest,None)[0]
    assert image['cohort_id']!=ablation['cohort_id']
    assert ablation['image_condition']=='no_image' and ablation['renderer_version']=='medvlm.no-image-diagnostic.v1'
