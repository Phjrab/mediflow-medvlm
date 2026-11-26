"""CPU-only completion contract tests; synthetic rows cannot establish native GPU fit."""
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from mediflow_medvlm import campaign_v6_completion as w


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path


def jsonl(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(''.join(json.dumps(r)+'\n' for r in rows));return path


def sample(source,index,*,task='medical_qa',qtype=None,answer_type=None):
    row={'sample_id':f'{source}/{index}','source_id':source,'split':'test','task_type':task,'language':'ko' if source in {'kormedmcqa','aihub_71875'} else 'en',
         'input':{'question':'synthetic question','context':'','options':[],'images':[]},'supervision':{'answer':'synthetic answer'},
         'metadata':{},'group_ids':{'document':f'{source}/document/{index//2}'}}
    if qtype:row['metadata']['q_type']=str(qtype)
    if answer_type:row['metadata']['answer_type']=answer_type
    if task=='medical_mcq':row['input']['options']=[{'label':'A','text':'one'},{'label':'B','text':'two'}];row['supervision']={'option_id':'A','answer':'one'}
    if task=='medical_vqa':row['input']['images']=[f'{source}/image{index//2}.png'];row['group_ids']={'image':f'{source}/image{index//2}'}
    return row


@pytest.fixture
def planning(tmp_path,monkeypatch):
    root=tmp_path/'campaign';selection_path=write(root/'family/config.json',{'job_id':'v6-family_selection'})
    proof={'config':w._reference(selection_path),'receipt':{'path':'private-native-receipt','sha256':'receipt'}}
    cells={'text':['T2','T3'],'vqa':['V1','V3']};chosen={'selected_cells_by_axis':cells,
        'selected_models':[w.dev.cell_spec(c) for axis in cells for c in cells[axis]],
        'additional_seed_plan':[{**w.dev.cell_spec(c),'seed':seed} for axis in cells for c in cells[axis] for seed in w.SEEDS]}
    records=[{'sample_id':str(i),'native':{'supervised_tokens_after_shift':i%4+1,'sequence_tokens':i%4+6}} for i in range(29)]
    models={}
    for cell,*_ in w.dev.CELLS:
        recipe={'seed':20261023,'examples':29,'optimizer_steps':2,'learning_rate':.01,'effective_batch_size':16,'passes':1,'checkpoint':'one_pass_final_only'}
        template={'cell_id':cell,'model':{'revision':'r'*40,'model_id':'original/base','snapshot_path':'original-private-snapshot'},'model_hash':'original-base-hash',
            'recipe':recipe,'adapter_initialization':None,'max_length':2048,'fixed_template_date':'2026-11-14','local_files_only':True,
            'trust_remote_code':False,'runtime_versions':{},'gpu_ledger_path':str(tmp_path/'ledger.json'),'supervisor_lock_paths':['canonical-a','canonical-b'],
            'output_dir':str(root/'first'/cell),'data':{'common_freeze':{'path':'common-freeze'},'image_root':str(tmp_path/'images'),
            'target_tokens':sum(r['native']['supervised_tokens_after_shift'] for r in records),'sequence_tokens':sum(r['native']['sequence_tokens'] for r in records)}}
        path=write(root/'first'/cell/'config.json',template);metric=write(root/'first'/cell/'metrics.json',{'native_contract':{}})
        models[cell]={'config':w._reference(path),'metrics':w._reference(metric),'checkpoint':{'path':'original-checkpoint','sha256':'original'},'adapter_sha256':'first-adapter',
                      'reload':{'path':'reload','sha256':'reload'},'model_hash':'original-base-hash','revision':'r'*40}
    initial={'campaign_root':str(root),'models':models}
    selection={'state_path':str(root/'state.json'),'data':{'initial_protocol':{'path':'initial'},'final_protocol':{'path':'final'}}}
    monkeypatch.setattr(w,'_selection',lambda path:(copy.deepcopy(selection),copy.deepcopy(proof),copy.deepcopy(chosen),copy.deepcopy(initial)))
    def inputs(template,seed):
        order=w.train.common._orders([r['sample_id'] for r in records])[str(seed)]
        ordered=[copy.deepcopy(records[i]) for i in order['indices']]
        return {},ordered,order
    monkeypatch.setattr(w,'_seed_inputs',inputs)
    return SimpleNamespace(root=root,selection_path=selection_path,chosen=chosen,initial=initial,records=records,proof=proof,inputs=inputs)


def test_import_is_cpu_only_and_cli_unset_cuda_works(tmp_path):
    code="import sys; from mediflow_medvlm import campaign_v6_completion as w; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules; assert w.main(['--config',sys.argv[1],'--receipt','unused'])==1"
    path=write(tmp_path/'bad.json',{'uses_gpu':False})
    result=subprocess.run([sys.executable,'-B','-c',code,str(path)],env={k:v for k,v in os.environ.items() if k!='CUDA_VISIBLE_DEVICES'},capture_output=True,text=True)
    assert result.returncode==0,result.stderr


@pytest.mark.parametrize('seed',[20261114,20261115])
def test_fresh_selected_seed_preserves_original_base_recipe_full_targets_and_native_math(planning,seed):
    f=planning;cfg=w.prepare_train_config(f.selection_path,'T2',seed,f.root/'completion/train'/str(seed))
    assert cfg['adapter_initialization'] is None and cfg['recipe']['seed']==seed and cfg['seed']==seed
    template=w._ref(f.initial['models']['T2']['config'])
    assert {k:v for k,v in cfg['recipe'].items() if k!='seed'}=={k:v for k,v in template['recipe'].items() if k!='seed'}
    assert cfg['data']['first_seed_template']==f.initial['models']['T2']['config']
    order=w.train.common._orders([r['sample_id'] for r in f.records])[str(seed)]
    assert cfg['data']['sample_order_sha256']==order['sample_order_sha256']
    assert cfg['data']['update_example_counts']==[16,13]
    assert cfg['data']['probe_sample_id']==f.records[order['indices'][-1]]['sample_id']
    assert cfg['data']['target_tokens']==template['data']['target_tokens']
    assert w.validate_config(json.loads(json.dumps(cfg)))==cfg
    runtime=w.training_runtime(cfg)
    for name in w.CLONED:
        assert getattr(runtime,name).__code__ is getattr(w.train,name).__code__
        assert getattr(runtime,name).__globals__ is not getattr(w.train,name).__globals__
    assert runtime._train.__globals__['_validate'] is w._training_validate
    assert w.train._train.__globals__['_validate'] is not w._training_validate
    assert w.train.SCOPE=='NATIVE_FIRST_SEED_ONE_PASS_FINAL_ONLY'


@pytest.mark.parametrize('mutate',[lambda c:c.update(adapter_initialization='first-seed-adapter'),lambda c:c['recipe'].update(passes=2),
    lambda c:c['recipe'].update(seed=20261023),lambda c:c['data'].update(update_example_counts=[16,16]),
    lambda c:c['data'].update(sample_order_sha256='old-first-seed-order'),lambda c:c['data']['native_function_contract'].update(_train='modified')])
def test_rehashed_seed_or_recipe_or_adapter_tampering_fails(planning,mutate):
    f=planning;cfg=w.prepare_train_config(f.selection_path,'T2',20261114,f.root/'completion/train/s')
    mutate(cfg);w._hash_config(cfg)
    with pytest.raises(ValueError):w.validate_config(cfg)


def test_unselected_seed_and_other_seed_cannot_train(planning):
    f=planning
    for cell,seed in [('T1',20261114),('T2',20261023),('T2',20261116)]:
        with pytest.raises(ValueError):w.prepare_train_config(f.selection_path,cell,seed,f.root/'completion/train/s')


def test_seed_order_uses_original_frozen_all_three_orders_not_first_order(monkeypatch):
    ids=[str(i) for i in range(29)];axis={'sample_ids':ids,'training_orders':w.train.common._orders(ids)}
    first=[{'sample_id':ids[i]} for i in axis['training_orders']['20261023']['indices']]
    monkeypatch.setattr(w.train,'_freeze',lambda path:({}, {}, {}))
    monkeypatch.setattr(w.train,'_inputs',lambda *args:({}, {}, first,axis,{}))
    template={'data':{'common_freeze':{'path':'immutable'}},'cell_id':'T2'}
    # Use the actual implementation, not the planning fixture seam.
    _,records,order=w._seed_inputs(template,20261115)
    assert [r['sample_id'] for r in records]==[ids[i] for i in axis['training_orders']['20261115']['indices']]
    assert [r['sample_id'] for r in records]!=[r['sample_id'] for r in first]
    axis['training_orders']['20261115']['indices'].reverse()
    with pytest.raises(ValueError,match='FROZEN_EXTRA_ORDER'):w._seed_inputs(template,20261115)


@pytest.mark.parametrize('size',[27,29])
def test_cloned_partial_update_uses_actual_example_divisor_against_scalar_gradient_oracle(planning,monkeypatch,size):
    cfg=w.prepare_train_config(planning.selection_path,'T2',20261114,planning.root/'completion/train/math')
    gradients=[];state={'gradient':0.0}
    class Loss:
        def __init__(self,value):self.value=value
        def __truediv__(self,n):return Loss(self.value/n)
        def backward(self):state['gradient']+=self.value
        def detach(self):return self
        def __float__(self):return float(self.value)
    class Optimizer:
        param_groups=[{'lr':.1}]
        def zero_grad(self,**kw):state['gradient']=0.0
        def step(self):gradients.append(state['gradient'])
    optimizer=Optimizer()
    fake=SimpleNamespace(isfinite=lambda loss:True,nn=SimpleNamespace(utils=SimpleNamespace(clip_grad_norm_=lambda params,norm:abs(state['gradient']))))
    monkeypatch.setitem(sys.modules,'torch',fake)
    model=SimpleNamespace(parameters=lambda:[]);scheduler=SimpleNamespace(step=lambda:optimizer.param_groups[0].update(lr=0.0))
    records=[{'sample_id':str(i)} for i in range(size)]
    actual=w.training_runtime(cfg)._updates(model,optimizer,scheduler,records,{},lambda r:(Loss(int(r['sample_id'])+1),2,4),lambda:{})
    expected=[sum(range(start+1,min(start+17,size+1)))/min(16,size-start) for start in range(0,size,16)]
    assert gradients==pytest.approx(expected,abs=1e-13)
    assert actual['micro_loss_divisor_per_update']==[16,size-16] and actual['samples_processed']==size
    assert actual['supervised_tokens']==2*size and actual['sample_ids']==[str(i) for i in range(size)]


@pytest.fixture
def full_test(tmp_path):
    by_source={
        'kormedmcqa':[sample('kormedmcqa',i,task='medical_mcq') for i in range(3009)],
        'text':[sample('aihub_71875',i,task='medical_mcq' if i<777 else 'medical_qa',qtype=1 if i<777 else 2 if i<860 else 3) for i in range(957)],
        'pubmedqa':[sample('pubmedqa_labeled',i,task='context_qa') for i in range(500)],
        'medquad':[sample('medquad',i) for i in range(1301)],
        'slake':[sample('slake',i,task='medical_vqa',answer_type='OPEN' if i%2 else 'CLOSED') for i in range(1061)],
        'vqa_rad':[sample('vqa_rad',i,task='medical_vqa',answer_type='OPEN' if i%2 else 'CLOSED') for i in range(451)]}
    entries={}
    for source,rows in by_source.items():
        path=jsonl(tmp_path/source/'test.jsonl',rows);entries[source]={'path':str(path),'sha256':w.file_hash(path),'count':len(rows)}
    ids=[r['sample_id'] for r in by_source['medquad']][19:83]
    old=write(tmp_path/'original-fixed-selection.json',{'sample_ids':['kormedmcqa/old-256',*ids]})
    entries['medquad'].update(selected_ids=ids,selection_metadata=w._reference(old))
    return entries


def test_full_population_includes_original_medquad_subset_not_all_1301_or_old_820(full_test):
    selected,refs=w._test_rows(full_test)
    assert {axis:len(rows) for axis,rows in selected.items()}=={'text':4530,'vqa':1512}
    assert [r['sample_id'] for r,_ in selected['text'] if r['source_id']=='medquad']==full_test['medquad']['selected_ids']
    assert {s for r,s in selected['vqa']}=={'slake_open','slake_closed','vqa_rad_open','vqa_rad_closed'}
    assert len(w._diagnostic_ids(selected)['text'])==32 and len(w._diagnostic_ids(selected)['vqa'])==32
    modified=copy.deepcopy(full_test);modified['kormedmcqa']['count']=256
    with pytest.raises(ValueError):w._test_rows(modified)


def test_test_row_bytes_and_medquad_selection_tampering_fail(full_test):
    modified=copy.deepcopy(full_test);modified['medquad']['selected_ids'].reverse()
    with pytest.raises(ValueError,match='MEDQUAD_SELECTION'):w._test_rows(modified)
    path=Path(full_test['text']['path']);path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError,match='SOURCE_CHANGED'):w._test_rows(full_test)


def test_selection_gate_precedes_any_test_manifest_or_row_read(monkeypatch):
    def blocked(path):raise ValueError('selection still pending')
    monkeypatch.setattr(w,'_selection',blocked)
    monkeypatch.setattr(w.train,'_freeze',lambda path:pytest.fail('test lineage touched before selection'))
    with pytest.raises(ValueError,match='selection still pending'):w._test_lineage('/no-selection')


def test_test_grouping_uses_real_source_row_digest_and_pairing_with_cross_strata_clusters():
    rows=[sample('slake',i,task='medical_vqa',answer_type='OPEN' if i%2 else 'CLOSED') for i in range(12)]
    selected=[(row,w._test_stratum(row)) for row in rows]
    report,units,profiles,evidence=w._test_grouping(selected,'vqa')
    assert report['group_count']==6 and sorted(map(len,units))==[2]*6
    assert all(e['source_row_sha256']==w.canonical_hash(row) for e,row in zip(evidence,rows))
    assert report==json.loads(json.dumps(report)) and len(profiles)==1
    base=[{'sample_id':r['sample_id'],'exact_match':0.,'raw_exact_match':0.,'valid':True,'status':'valid','finish_reason':'stop','token_f1':0.} for r in rows]
    tuned=[{**r,'exact_match':1.,'raw_exact_match':1.,'token_f1':1.} for r in base]
    result=w.bootstrap.bootstrap_pair(selected,base,tuned,units,profiles)
    for stratum in result.values():
        interval=stratum['metrics']['normalized_em']['tuned_minus_base']
        assert (interval['estimate'],interval['lower'],interval['upper'],interval['defined_resamples'])==(1.,1.,1.,2000)


def test_three_seed_missing_failure_is_explicit_and_never_averaged_or_best_selected():
    result=w.seed_statistics({20261023:.8,20261115:.2})
    assert result['values']['20261114'] is None and result['mean'] is None and result['sample_std'] is None
    assert not result['three_seed_complete'] and not result['best_seed_selected']
    complete=w.seed_statistics({20261023:.2,20261114:.4,20261115:.6})
    assert complete['mean']==pytest.approx(.4) and complete['sample_std']==pytest.approx(.2)
    with pytest.raises(ValueError):w.seed_statistics({20261023:math.nan})


def test_test_matrix_does_not_replace_failed_extra_seed(planning):
    audit={'conditions':{f"v6-additional-{r['cell_id']}-{r['seed']}":{'status':'FAILED'} for r in planning.chosen['additional_seed_plan']}}
    matrix=w._test_matrix(planning.chosen,audit)
    assert len(matrix)==20 and all(seed==20261023 for _,_,seed in matrix)
    audit['conditions']['v6-additional-T2-20261114']['status']='SUCCEEDED'
    matrix=w._test_matrix(planning.chosen,audit)
    assert len(matrix)==21 and ('T2','tuned',20261114) in matrix
    assert ('T1','tuned',20261114) not in matrix


def test_arbitrary_success_receipt_cannot_substitute_for_ci_or_preservation(tmp_path):
    ref=w._reference(write(tmp_path/'family-success.json',{'status':'SUCCEEDED','family_selection_completed':True}))
    with pytest.raises(ValueError,match='TYPED_OWNER'):
        w._report_external({'source_hash':'source','data':{'evidence':{'code_report_ci':ref}}})
    proof,missing=w._report_external({'source_hash':'source','data':{'evidence':{}}})
    assert proof=={} and set(missing)=={'code_report_ci','existing_result_preservation'}


def test_audit_success_receipt_semantic_flag_tampering_rejected(tmp_path,monkeypatch):
    cfg={'kind':'additional_audit','uses_gpu':False,'scope':w.SCOPES['additional_audit'],'output_dir':str(tmp_path/'out'),
         'success_receipt':str(tmp_path/'receipt.json'),**{k:'test-'+k for k in w.IDENTITY_KEYS}}
    path=write(tmp_path/'config.json',cfg)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','');monkeypatch.setattr(w,'validate_config',lambda config,**kw:config)
    monkeypatch.setattr(w.dev.caps,'_env',lambda config:None)
    summary={'additional_seeds_terminal':True,'planned':0,'conditions':{},'replacement_after_failure_allowed':False,'test_inference_performed':False}
    monkeypatch.setattr(w,'_additional_summary',lambda config:copy.deepcopy(summary))
    result=w.run_job(path,cfg['success_receipt']);assert result['status']=='SUCCEEDED' and not result['v6_completed']
    assert w.run_job(path,cfg['success_receipt'])==result
    marker=Path(cfg['output_dir'])/'completion.json';value=w._read(marker);value['test_files_read']=True;write(marker,value)
    with pytest.raises(ValueError,match='RECEIPT_CHANGED'):w.verify_result(cfg)


def test_fatal_native_diagnostic_preserves_attempt_no_success_and_finishes_ledger(tmp_path,monkeypatch):
    cfg={'kind':'diagnostic','uses_gpu':True,'scope':w.SCOPES['diagnostic'],'output_dir':str(tmp_path/'out'),
         'success_receipt':str(tmp_path/'receipt.json'),'gpu_ledger_path':str(tmp_path/'ledger.json'),'model_id':'native/model',
         **{k:'test-'+k for k in w.IDENTITY_KEYS}}
    path=write(tmp_path/'config.json',cfg);calls=[]
    monkeypatch.setattr(w,'validate_config',lambda config,**kw:config);monkeypatch.setattr(w.dev.caps,'_env',lambda config:None)
    monkeypatch.setattr(w.train,'_inherited_supervisor_fds',lambda config:())
    class Ledger:
        def __init__(self,*a,**kw):pass
        def start(self,*a):calls.append('start')
        def finish(self):calls.append('finish')
    monkeypatch.setattr(w,'BudgetLedger',Ledger)
    def fail(config,stream):stream.write('{"actual_attempt":"fatal synthetic forward"}\n');raise RuntimeError('synthetic OOM')
    monkeypatch.setattr(w,'_diagnostic',fail)
    with pytest.raises(RuntimeError,match='synthetic OOM'):w.run_job(path,cfg['success_receipt'])
    assert calls==['start','finish'] and not Path(cfg['output_dir']).exists() and not Path(cfg['success_receipt']).exists()
    attempt=next(tmp_path.glob('out.attempt-*'));assert (attempt/'records.private.jsonl').is_file() and (attempt/'failure.private.json').is_file()


def test_historical_train_verify_uses_pinned_repo_and_does_not_claim_parent_job_identity(planning,monkeypatch):
    f=planning;cfg=w.prepare_train_config(f.selection_path,'T2',20261114,f.root/'completion/train/verified')
    seen=[]
    monkeypatch.setattr(w,'validate_config',lambda config,**kw:seen.append(kw.get('repo')) or config)
    monkeypatch.setattr(w,'training_runtime',lambda config:SimpleNamespace(_completed_output=lambda config,records:{'status':'SUCCEEDED','records':len(records)}))
    monkeypatch.setenv('MEDVLM_SUPERVISOR_RUN_ID','different-parent-audit')
    result=w.verify_result(cfg,repo='/immutable-original-snapshot')
    assert result=={'status':'SUCCEEDED','records':29} and seen==['/immutable-original-snapshot']
    with pytest.raises(ValueError,match='SUPERVISOR_IDENTITY_CHANGED'):w._training_validate(cfg)


def test_read_transaction_memoizes_only_inside_call_and_rehashes_all_refs(tmp_path):
    path=write(tmp_path/'pin.json',{'value':1});ref=w._reference(path);calls=[]
    @w._memo
    def expensive():calls.append('called');return w._ref(ref)
    @w._transaction
    def transaction(mutate=False):
        assert expensive()==expensive()
        if mutate:write(path,{'value':2})
        return 'audited'
    assert transaction()=='audited' and len(calls)==1
    assert transaction()=='audited' and len(calls)==2
    with pytest.raises(ValueError,match='REFERENCE_CHANGED_DURING_AUDIT'):transaction(True)


def test_native_no_image_context_or_internal_pixel_failure_is_not_unsupported(monkeypatch):
    from mediflow_medvlm.evaluation import v2
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace())
    row=sample('slake',0,task='medical_vqa',answer_type='OPEN');bundle=SimpleNamespace(processor=None,architecture=None)
    config={'eval_context':4096,'data':{'image_root':'private'}}
    def context(*a,**kw):raise ValueError('CONTEXT_OVERFLOW')
    monkeypatch.setattr(v2,'encode_evaluation_sample',context)
    result=w._no_image(bundle,row,'slake_open',128,config)
    assert result['status']=='TECHNICAL_FAILURE' and 'CONTEXT_OVERFLOW' in result['reason_code']
    def unsupported(*a,**kw):raise ValueError('native processor requires an image')
    monkeypatch.setattr(v2,'encode_evaluation_sample',unsupported)
    result=w._no_image(bundle,row,'slake_open',128,config)
    assert result['status']=='UNSUPPORTED_NATIVE_NO_IMAGE'


def test_diagnostic_executor_writes_real_jsonl_lines_and_warmup_measurement_matrix(tmp_path,monkeypatch):
    from mediflow_medvlm.evaluation import v6_native
    rows=[sample('medquad',i) for i in range(32)];selected=[(row,'medquad') for row in rows]
    trained=write(tmp_path/'trained.json',{'output_dir':'original'});metrics=write(tmp_path/'train-metrics.json',{'native_contract':{}})
    protocol=write(tmp_path/'protocol.json',{'caps':{'medical_qa':128}})
    cfg={'cell_id':'T2','task':'text','role':'base','model':{},'data':{'model_ref':{'config':w._reference(trained),'metrics':w._reference(metrics),'adapter_sha256':'original'},
        'protocol':w._reference(protocol),'diagnostic_ids':[row['sample_id'] for row in rows],'procedure':copy.deepcopy(w.DIAGNOSTIC_PROCEDURE)}}
    cuda=SimpleNamespace(synchronize=lambda:None,reset_peak_memory_stats=lambda:None,max_memory_allocated=lambda:100,max_memory_reserved=lambda:200,
        get_device_properties=lambda i:SimpleNamespace(name='syntheticCPU',total_memory=1000,uuid='synthetic-device'),get_device_capability=lambda i:(0,0))
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=cuda));monkeypatch.setattr(v6_native,'load_evaluation_bundle',lambda *a,**kw:SimpleNamespace(contract={'synthetic':True}))
    monkeypatch.setattr(w,'_rows_for',lambda config:{'text':selected})
    def prediction(bundle,row,stratum,cap,config):
        return {**w.dev.caps.initial._failure(row,stratum,cap,'synthetic-only'),'status':'SUCCEEDED','reason_code':None,
            'generated_tokens':1,'input_tokens':10,'generated_text':'answer','finish_reason':'stop'}
    monkeypatch.setattr(w.dev.caps.initial,'_predict_one',prediction)
    path=tmp_path/'records.jsonl'
    with path.open('w') as stream:summary,native=w._diagnostic(cfg,stream)
    records=[json.loads(line) for line in path.read_text().splitlines()]
    assert len(records)==98 and len([r for r in records if r['stage']=='warmup'])==2
    assert summary['sample_count']==96 and summary['successful_requests']==96 and summary['by_task']['medical_qa']['scheduled']==96
    assert native['gpu_identity']['uuid']=='synthetic-device' and summary['ttft_seconds'] is None
    records[-1]['repeat']=0
    with pytest.raises(ValueError,match='MEASURED_MATRIX_CHANGED'):w._diagnostic_summary(cfg,records)


def test_plot_height_preserves_every_open_closed_condition():
    svg=w._plot('OPEN/CLOSED',[(str(i),float(i),.5) for i in range(32)])
    assert 'height="686"' in svg and 'y="623"' in svg and 'y="666"' in svg


@pytest.fixture
def native_planning(planning,full_test,monkeypatch):
    f=planning;selected,sources=w._test_rows(full_test)
    protocol={'protocol_hash':'sealed-protocol','ordered_test_ids':{a:[r['sample_id'] for r,_ in rows] for a,rows in selected.items()},
              'caps':{'medical_mcq':16,'context_qa':16,'medical_qa':512,'medical_vqa':128},
              'decode':{'do_sample':False,'num_beams':1,'use_cache':True,'max_time':None,'warmup_requests':0},
              'input_digests':{axis:{row['sample_id']:w.canonical_hash(row['input']) for row,_ in rows} for axis,rows in selected.items()},
              'reference_digest':{axis:w.canonical_hash([(row['sample_id'],row['supervision']) for row,_ in rows]) for axis,rows in selected.items()},
              'fixed_template_date':'2026-11-14','diagnostic_ids':w._diagnostic_ids(selected),'grouping':{axis:w._test_grouping(rows,axis)[0] for axis,rows in selected.items()}}
    audit_path=write(f.root/'extra-audit/config.json',{'job_id':'v6-additional_seeds','output_dir':str(f.root/'extra-audit'),'data':{'train_configs':[]}})
    summary={'conditions':{f"v6-additional-{r['cell_id']}-{r['seed']}":{'status':'FAILED','error':'synthetic terminal failure'} for r in f.chosen['additional_seed_plan']}}
    write(f.root/'extra-audit/summary.json',summary)
    freeze={'kind':'test_freeze','job_id':'v6-test-freeze','config_hash':'synthetic-freeze-config','output_dir':str(f.root/'freeze'),'data':{'selection':f.proof,'test_entries':full_test,
        'additional_audit':{'config':w._reference(audit_path)},'test_sources':sources,'forge':{'data_root':str(f.root/'data'),'output_root':str(f.root.parent/'forge'),'campaign_id':f.root.name}}}
    freeze_path=write(f.root/'freeze/config.json',freeze);write(f.root/'freeze/protocol.json',protocol)
    for axis in ('text','vqa'):w.atomic_json(f.root/'freeze'/f'evaluation_protocol.{axis}.json',w._axis_protocol(freeze,protocol,axis))
    freeze_proof={'config':w._reference(freeze_path),'receipt':{'path':'synthetic-receipt','sha256':'synthetic'}}
    registered={str(freeze_path):(freeze,freeze_proof,{'status':'SUCCEEDED'})}
    monkeypatch.setattr(w,'_registered',lambda path,*a:copy.deepcopy(registered[str(path)]))
    return SimpleNamespace(**vars(f),selected=selected,protocol=protocol,freeze=freeze,freeze_path=freeze_path,freeze_proof=freeze_proof,registered=registered)


def test_native_test_conditions_pin_all_rows_caps_native_original_models_and_selected_first_seed(native_planning):
    f=native_planning
    cfg=w.prepare_evaluation_config(f.freeze_path,'T2','base',20261023,f.root/'completion/evaluation/T2/base')
    assert cfg['split']=='test' and cfg['data']['model_ref']==f.initial['models']['T2']
    assert len(cfg['data']['ordered_sample_ids'])==4530 and cfg['decode']==f.protocol['decode']
    assert cfg['eval_context']==4096 and cfg['fixed_template_date']=='2026-11-14'
    assert w.validate_config(json.loads(json.dumps(cfg)))==cfg
    for key,value in [('ordered_sample_ids',['test-subset']),('image_root','different')]:
        modified=copy.deepcopy(cfg);modified['data'][key]=value;w._hash_config(modified)
        with pytest.raises(ValueError,match='SCIENTIFIC_IDENTITY_CHANGED'):w.validate_config(modified)
    with pytest.raises(ValueError):w.prepare_evaluation_config(f.freeze_path,'T2','base',20261114,f.root/'completion/evaluation/invalid')
    with pytest.raises(ValueError):w.prepare_likelihood_config(f.freeze_path,'V1','base',20261023,f.root/'completion/likelihood/invalid')


def test_diagnostic_fixed_32_balances_all_task_types_and_cannot_change_after_results(native_planning):
    f=native_planning
    cfg=w.prepare_diagnostic_config(f.freeze_path,'T2','tuned',f.root/'completion/diagnostic/T2/tuned')
    selected={r['sample_id']:r for r,_ in f.selected['text']}
    counts={task:sum(selected[sid]['task_type']==task for sid in cfg['data']['diagnostic_ids']) for task in ('medical_mcq','context_qa','medical_qa')}
    assert counts=={'medical_mcq':12,'context_qa':8,'medical_qa':12}
    assert cfg['data']['procedure']['warmup_requests']==2 and cfg['data']['procedure']['measured_repeats']==3
    modified=copy.deepcopy(cfg);modified['data']['diagnostic_ids'].reverse();w._hash_config(modified)
    with pytest.raises(ValueError):w.validate_config(modified)
    vqa=w.prepare_diagnostic_config(f.freeze_path,'V1','base',f.root/'completion/diagnostic/V1/base')
    lookup={r['sample_id']:s for r,s in f.selected['vqa']}
    assert dict(w.Counter(lookup[sid] for sid in vqa['data']['diagnostic_ids']))=={'slake_open':8,'slake_closed':8,'vqa_rad_open':8,'vqa_rad_closed':8}


def test_full_test_matrix_and_anchor_dependencies_are_exact(native_planning):
    f=native_planning;paths={kind:[] for kind in ('evaluation','diagnostic','likelihood')}
    for cell,*_ in w.dev.CELLS:
        for role in ('base','tuned'):
            for kind in paths:
                if kind=='likelihood' and w.dev.cell_spec(cell)['task']!='text':continue
                directory=f.root/'completion'/kind/cell/role
                cfg=(w.prepare_diagnostic_config(f.freeze_path,cell,role,directory) if kind=='diagnostic' else
                    (w.prepare_likelihood_config if kind=='likelihood' else w.prepare_evaluation_config)(f.freeze_path,cell,role,20261023,directory))
                paths[kind].append(write(f.root/'configs'/f'{kind}-{cell}-{role}.json',cfg))
    kwargs={'diagnostic_config_paths':paths['diagnostic'],'likelihood_config_paths':paths['likelihood']}
    cfg=w.prepare_test_audit_config(f.selection_path,f.freeze_path,paths['evaluation'],f.root/'completion/test_audit/all',**kwargs)
    assert len(cfg['data']['evaluations'])==20 and len(cfg['data']['diagnostics'])==20 and len(cfg['data']['likelihoods'])==12
    path=write(f.root/'configs/test-audit.json',cfg);job=w.job_from_config(path)
    assert job['job_id']=='v6-test' and job['phase']=='test' and job['requires_gpu'] is False
    assert job['dependency_policy']=='terminal' and 'v6-additional_seeds' in job['dependencies'] and 'v6-test-freeze' in job['dependencies']
    assert len(job['dependencies'])==55  # selection + freeze + original direct additional anchor +52conditions
    with pytest.raises(ValueError,match='EXACT_EVALUATION_MATRIX'):w.prepare_test_audit_config(f.selection_path,f.freeze_path,paths['evaluation'][:-1],f.root/'completion/test_audit/short',**kwargs)
    with pytest.raises(ValueError,match='EXACT_DIAGNOSTIC_MATRIX'):w.prepare_test_audit_config(f.selection_path,f.freeze_path,paths['evaluation'],f.root/'completion/test_audit/no-speed',likelihood_config_paths=paths['likelihood'])


def test_owner_ci_evidence_binds_test_protocol_report_table_and_actual_head(native_planning):
    f=native_planning;table=[{'cell_id':'T2','normalized_em':.2}];commit='a'*40
    proof={'schema_version':'medvlm.v6.owner-evidence.v1','kind':'code_report_ci','status':'SUCCEEDED','source_hash':'source',
           'source_commit':commit,'pushed_commit':commit,'test_protocol_hash':f.protocol['protocol_hash'],'published_report_table_sha256':w.canonical_hash(table),
           'cpu_tests':{'passed':100,'failed':0},'ci_runs':[{'head_sha':commit,'conclusion':'success','url':f'https://github.com/example/repo/actions/runs/{i}'} for i in range(3)]}
    path=write(f.root/'owner-proof.json',proof);cfg={'source_hash':'source','data':{'test_freeze':f.freeze_proof,'evidence':{'code_report_ci':w._reference(path)}}}
    assert w._report_external(cfg,table)[1]==['existing_result_preservation']
    for field,value in [('test_protocol_hash','old-protocol'),('published_report_table_sha256','old-table'),('pushed_commit','b'*40)]:
        modified=copy.deepcopy(proof);modified[field]=value;write(path,modified);cfg['data']['evidence']['code_report_ci']=w._reference(path)
        with pytest.raises(ValueError):w._report_external(cfg,table)


def test_no_image_encoder_oom_is_fatal_not_unavailable_capability(monkeypatch):
    from mediflow_medvlm.evaluation import v2
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace())
    class OutOfMemoryError(RuntimeError):pass
    def fail(*a,**kw):raise OutOfMemoryError('synthetic cuda allocator')
    monkeypatch.setattr(v2,'encode_evaluation_sample',fail)
    row=sample('slake',0,task='medical_vqa',answer_type='OPEN')
    with pytest.raises(OutOfMemoryError):w._no_image(SimpleNamespace(processor=None,architecture=None),row,'slake_open',128,{'eval_context':4096,'data':{'image_root':'private'}})


def test_signed_plot_retains_negative_quality_regression():
    import re
    svg=w._plot('delta',[('worse',0.,-.8),('better',1.,.1)],ylabel='Tuned minus Base')
    assert '-0.8' in svg
    points=[float(value) for value in re.findall(r'cy="([0-9.]+)"',svg)]
    assert len(points)==2 and all(80<=point<=420 for point in points)


@pytest.mark.parametrize('mutate',[
    lambda p:p.update(generated_ids=[],generated_tokens=0,generation_scores_checked_steps=0),
    lambda p:p.update(no_thinking_output={'explicit_nonempty_thinking_detected':True}),
    lambda p:p.update(finish_reason='length'),lambda p:p.update(generation_scores_checked_steps=2),
    lambda p:p.update(input_tokens=4090),lambda p:p.update(raw_forward_logits_finite_verified=True)])
def test_no_image_success_uses_same_native_proof_guards_as_image(tmp_path,monkeypatch,mutate):
    rows=[sample('slake',i,task='medical_vqa',answer_type='OPEN') for i in range(32)]
    selected=[(row,'slake_open') for row in rows];ids=[r['sample_id'] for r in rows]
    protocol=write(tmp_path/'protocol.json',{'caps':{'medical_vqa':128}})
    cfg={'kind':'diagnostic','task':'vqa','cell_id':'V1','data':{'model_ref':{},'diagnostic_ids':ids,'procedure':copy.deepcopy(w.DIAGNOSTIC_PROCEDURE),'protocol':w._reference(protocol)}}
    monkeypatch.setattr(w,'_rows_for',lambda config:{'vqa':selected})
    monkeypatch.setattr(w.dev.caps,'_native_proof',lambda *a:None)
    def prediction(row):
        return {**w.dev.caps.initial._failure(row,'slake_open',128,None),'status':'SUCCEEDED','generated_ids':[42],'generated_tokens':1,
            'generated_ids_sha256':w.canonical_hash([42]),'generated_text':'synthetic answer','generated_raw_text':'synthetic answer','input_tokens':10,
            'finish_reason':'stop','eos_token_ids':42,'no_thinking_output':{'explicit_nonempty_thinking_detected':False},
            'generation_scores_valid':True,'generation_scores_checked_steps':1,'raw_forward_logits_finite_verified':False,'pixel_tensor_keys':[],
            'input_metadata':{'images':[]}}
    schedule=[('warmup',0,row) for row in rows[:2]]+[('measured',repeat,row) for repeat in range(3) for row in rows]+[('no_image',0,row) for row in rows]
    records=[{'stage':stage,'repeat':repeat,'prediction':prediction(row),'synchronized_e2e_seconds':.1,'peak_allocated_bytes':100,'peak_reserved_bytes':200} for stage,repeat,row in schedule]
    metrics=w._diagnostic_summary(cfg,records);metrics['gpu_identity']={'name':'syntheticCPU','total_memory':1000}
    w._gpu_metrics_proof(cfg,metrics,records)
    mutate(records[-1]['prediction']);metrics.update(w._diagnostic_summary(cfg,records))
    with pytest.raises(ValueError,match='CALIBRATION_'):w._gpu_metrics_proof(cfg,metrics,records)


def test_report_failures_keep_all_rows_selected_seed_missing_and_ll_aux_separate(native_planning,monkeypatch):
    f=native_planning
    monkeypatch.setattr(w,'bootstrap',SimpleNamespace(**{**vars(w.bootstrap),'bootstrap_pair':lambda *a:{'synthetic_interval_fixture':True}}))
    refs={'evaluations':[],'diagnostics':[],'likelihoods':[]}
    for cell,*_ in w.dev.CELLS:
        spec=w.dev.cell_spec(cell)
        for role in ('base','tuned'):
            for key,kind in [('evaluations','evaluation'),('diagnostics','diagnostic'),('likelihoods','likelihood')]:
                if key=='likelihoods' and spec['task']!='text':continue
                job=f"v6-test-{kind+'-' if kind!='evaluation' else ''}{cell}-{role}-20261023"
                cfg={**spec,'kind':kind,'role':role,'seed':20261023,'job_id':job,'output_dir':str(f.root/'empty'/job)}
                refs[key].append(w._reference(write(f.root/'report-inputs'/f'{job}.json',cfg)))
    def terminal(config,ref):
        child=w._ref(ref)
        return child,{'status':'FAILED','error':'synthetic documented failure','error_signature':'synthetic','receipt':None},None
    monkeypatch.setattr(w,'_terminal',terminal)
    config={'scope':w.SCOPES['report'],'source_hash':'source','data':{'selection':f.proof,'test_freeze':f.freeze_proof,'evidence':{},**refs}}
    monkeypatch.setattr(w,'_experiment_manifest',lambda *args:{'synthetic_only':True})
    summary=w._report(config)
    assert len(summary['evaluation_conditions'])==20 and len(summary['first_seed_table'])==104
    assert all(row['technical_failure_rate']==1. for row in summary['first_seed_table'])
    assert {r['cell_id'] for r in summary['selected_three_seed_confirmation']}=={'T2','T3','V1','V3'}
    assert all(r['mean'] is None and r['values']['20261114'] is None and not r['all_three_evaluations_technical_complete'] for r in summary['selected_three_seed_confirmation'])
    likelihood=next(iter(summary['likelihood_conditions'].values()))
    assert likelihood['all_rows_unavailable'] and likelihood['metrics']['classification_scheduled']==4286
    assert likelihood['metrics']['technical_failure_count']==4286 and likelihood['metrics']['unsupported_count']==244
    assert not summary['v6_completed'] and not summary['clinical_quality_certified'] and summary['human_review_performed']==0
    assert len(summary['paired_group_intervals'])==10 and summary['bootstrap_procedure']['resamples']==2000


def test_test_likelihood_adapter_preserves_native_math_and_never_reads_gold(native_planning):
    f=native_planning;cfg=w.prepare_likelihood_config(f.freeze_path,'T2','base',20261023,f.root/'completion/likelihood/T2/base')
    runtime=w.likelihood_runtime(cfg)
    assert cfg['data']['procedure']['classification_scheduled']==4286 and cfg['data']['procedure']['text_all_planned']==4530
    for name in w.CHOICE_CLONED:
        assert getattr(runtime,name).__code__ is getattr(w.choice,name).__code__
        assert getattr(runtime,name).__globals__ is not getattr(w.choice,name).__globals__
    assert w.choice.labels_for.__globals__['labels_for'] is w.choice.labels_for
    class GoldForbidden(dict):
        def __getitem__(self,key):
            if key=='supervision':raise AssertionError('gold was read while constructing candidates')
            return super().__getitem__(key)
    row=GoldForbidden(sample('kormedmcqa',0,task='medical_mcq'));row['input']['options']={'A':'one','B':'two'}
    assert w._test_labels_for(row)==['A','B']
    with pytest.raises(ValueError,match='VALIDATION_ONLY'):w.choice.labels_for(row)
    logits=[[0.,0.,0.],[0.,math.log(2.),0.],[0.,0.,math.log(4.)],[0.,0.,0.]]
    result=runtime.conditional_log_likelihood(logits,[1,2],2)
    assert result['conditional_log_likelihood']==pytest.approx(math.log(2/4)+math.log(4/6),abs=1e-12)
    assert result==w.choice.conditional_log_likelihood(logits,[1,2],2)
    assert w._test_choice_coverage(f.selected['text'])['classification_scheduled']==4286
    with pytest.raises(ValueError,match='EXACT_TEST_DENOMINATORS'):w._test_choice_coverage(f.selected['text'][:820])
    modified=copy.deepcopy(cfg);modified['data']['choice_function_contract']['_score_one']='changed';w._hash_config(modified)
    with pytest.raises(ValueError):w.validate_config(modified)


def test_experiment_manifest_counts_original_native_smokes_and_failed_new_seed(planning):
    f=planning
    smoke={f'v6-native-smoke-{cell[0]}':{'status':'SUCCEEDED','attempts':2,'failure_count':0,'recovery_attempts':0} for cell in w.dev.CELLS}
    common_cfg=write(f.root/'common/config.json',{'prerequisites':{'states':smoke}})
    common_receipt=write(f.root/'common-receipt.json',{'status':'SUCCEEDED','artifacts':[w._reference(common_cfg)]})
    for cell,ref in f.initial['models'].items():
        cfg=w._ref(ref['config']);cfg['data']['common_freeze']=w._reference(common_receipt);write(Path(ref['config']['path']),cfg)
        f.initial['models'][cell]['config']=w._reference(ref['config']['path'])
    first_summary=write(f.root/'first-aggregate/summary.json',{'cells':[{'cell_id':cell[0],'status':'SUCCEEDED','attempts':1,'failure_count':0,'recovery_attempts':0} for cell in w.dev.CELLS]})
    first_receipt=write(f.root/'first-receipt.json',{'status':'SUCCEEDED','artifacts':[w._reference(first_summary)]})
    f.initial['first_seed_receipt']=w._reference(first_receipt)
    extra=w.prepare_train_config(f.selection_path,'T2',20261114,f.root/'completion/train/failed-extra')
    path=write(f.root/'extra-config.json',extra)
    additional={extra['job_id']:{'config':w._reference(path),'status':'FAILED','attempts':3,'failure_count':3,'recovery_attempts':1,'error':'native load OOM','error_signature':'stable'}}
    manifest=w._experiment_manifest(f.initial,additional,{'data':{'evaluations':[],'diagnostics':[],'likelihoods':[]}})
    assert manifest['native_smoke_conditions']==10 and manifest['native_smoke_attempts']==20
    assert manifest['verified_original_base_main_training_conditions']==10 and manifest['first_seed_training_attempts']==10
    assert manifest['additional_seed_training_attempts']==3 and not manifest['report_performed_new_training']
    assert manifest['conditions'][-1]['error']=='native load OOM' and manifest['conditions'][-1]['origin']=='selected_validation_frozen_fresh_original_base'


def test_fatal_test_likelihood_persists_allplanned_attempt_metrics_ledger_and_no_success(tmp_path,monkeypatch):
    cfg={'kind':'likelihood','uses_gpu':True,'scope':w.SCOPES['likelihood'],'output_dir':str(tmp_path/'out'),
         'success_receipt':str(tmp_path/'receipt.json'),'gpu_ledger_path':str(tmp_path/'ledger.json'),'model_id':'native/model','protocol_hash':'fixed',
         **{k:'test-'+k for k in w.IDENTITY_KEYS}}
    path=write(tmp_path/'config.json',cfg);calls=[]
    monkeypatch.setattr(w,'validate_config',lambda config,**kw:config);monkeypatch.setattr(w.dev.caps,'_env',lambda config:None)
    monkeypatch.setattr(w.train,'_inherited_supervisor_fds',lambda config:())
    class Ledger:
        session_seconds=.5
        def __init__(self,*a,**kw):pass
        def start(self,*a):calls.append('start')
        def finish(self):calls.append('finish')
        def snapshot(self):return {'resource_policy':w.ResourcePolicy().as_dict()}
    monkeypatch.setattr(w,'BudgetLedger',Ledger)
    def fatal(config,stream):
        stream.write(json.dumps({'all_planned_synthetic_rows':4530})+'\n')
        return {'classification_scheduled':4286,'technical_failure_count':4286,'unsupported_count':244}, {'native':{'fatal_reason':'CHOICE_MODEL_LOAD_OutOfMemoryError','model_load_succeeded':False,'native_contract':None}},[]
    monkeypatch.setattr(w,'_likelihood',fatal)
    with pytest.raises(w.dev.caps.initial.CalibrationFatal):w.run_job(path,cfg['success_receipt'])
    assert calls==['start','finish'] and not Path(cfg['success_receipt']).exists() and not Path(cfg['output_dir']).exists()
    attempt=next(tmp_path.glob('out.attempt-*'));metrics=w._read(attempt/'metrics.attempt.private.json')
    assert metrics['classification_scheduled']==4286 and metrics['technical_failure_count']==4286 and metrics['unsupported_count']==244
    assert metrics['gpu_accounting']['session_seconds']==.5 and not metrics['native']['model_load_succeeded']


def test_final_immutable_report_cannot_register_before_required_evidence(tmp_path,monkeypatch):
    monkeypatch.setattr(w,'_registered',lambda *a:pytest.fail('premature report must reject before reading inputs'))
    with pytest.raises(ValueError,match='FINAL_REPORT_OWNER_EVIDENCE_REQUIRED'):
        w.prepare_report_config('test-audit',tmp_path/'out')
    cfg={'data':{},'kind':'report'}
    monkeypatch.setattr(w,'_report_config',lambda *a,**kw:cfg)
    monkeypatch.setattr(w,'_report',lambda config:{'v6_completed':False,'missing_required_evidence':['speed_unavailable']})
    with pytest.raises(ValueError,match='FINAL_REPORT_REQUIRED_EVIDENCE_MISSING'):
        w.prepare_report_config('test-audit',tmp_path/'out',evidence={'code_report_ci':'ci','existing_result_preservation':'preserved'})


def test_task_resource_summaries_keep_native_units_failures_eos_and_utf8_separate():
    selected={};records=[]
    for i,(task,text,status,ids,eos) in enumerate([
        ('medical_mcq','가','SUCCEEDED',[7],7),('medical_mcq','','TECHNICAL_FAILURE',[],7),
        ('medical_qa','English','SUCCEEDED',[8],[7])]):
        row=sample('synthetic',i,task=task);selected[row['sample_id']]=(row,'synthetic')
        records.append({'prediction':{'sample_id':row['sample_id'],'task_type':task,'status':status,'generated_text':text,
            'generated_ids':ids,'eos_token_ids':eos,'generated_tokens':len(ids),'input_tokens':10+i,
            'finish_reason':'length' if i==2 else 'stop'},'synchronized_e2e_seconds':i+1.,
            'peak_allocated_bytes':100+i,'peak_reserved_bytes':200+i})
    mcq=w._resource_summary(records[:2],selected);qa=w._resource_summary(records[2:],selected)
    assert mcq['scheduled']==2 and mcq['successful_requests']==1 and mcq['technical_failures']==1
    assert mcq['eos_stops']==1 and mcq['eos_stop_rate']==.5 and mcq['output_characters']==1 and mcq['output_utf8_bytes']==3
    assert mcq['input_native_tokens']==21 and mcq['output_native_tokens']==1 and mcq['sample_count']==2 and mcq['latency_p50_seconds']==1.5
    assert mcq['peak_allocated_bytes']==101 and mcq['peak_reserved_bytes']==201
    assert qa['truncation_rate']==1. and qa['eos_stops']==0 and qa['peak_reserved_bytes']==202
    assert mcq['native_input_token_measurement_count']==2 and mcq['ttft_seconds'] is None and mcq['decode_only_seconds'] is None
    assert mcq['input_serialized_utf8_bytes']==sum(len(json.dumps(selected[r['prediction']['sample_id']][0]['input'],ensure_ascii=False).encode()) for r in records[:2])


def test_generation_summary_truthfully_points_to_separate_native_likelihood_job():
    row=sample('medquad',0);selected=[(row,'medquad')]
    predictions=[w.dev.caps.initial._failure(row,'medquad',128,'synthetic-only')]
    original_scored,original=w.dev.score_predictions(selected,predictions)
    scored,summary=w._generation_scores(selected,predictions)
    assert scored==original_scored and {k:v for k,v in summary.items() if k!='likelihood_auxiliary'}=={k:v for k,v in original.items() if k!='likelihood_auxiliary'}
    assert summary['likelihood_auxiliary']['status']=='SEPARATE_NATIVE_AUXILIARY_JOB' and not summary['likelihood_auxiliary']['generation_validity_affected']


def test_report_completion_accepts_failed_auxiliary_ll_but_rejects_diagnostic_failure_and_unknown_gpu(native_planning,monkeypatch):
    f=native_planning;refs={'evaluations':[],'diagnostics':[],'likelihoods':[]};table=[];diag_paths=[]
    monkeypatch.setattr(w,'bootstrap',SimpleNamespace(**{**vars(w.bootstrap),'bootstrap_pair':lambda *a:{'synthetic_interval_fixture':True}}))
    monkeypatch.setattr(w,'_experiment_manifest',lambda *a:{'synthetic_only':True})
    for cell,*_ in w.dev.CELLS:
        spec=w.dev.cell_spec(cell)
        for role in ('base','tuned'):
            for key,kind in [('evaluations','evaluation'),('diagnostics','diagnostic'),('likelihoods','likelihood')]:
                if key=='likelihoods' and spec['task']!='text':continue
                job=f"v6-test-{kind+'-' if kind!='evaluation' else ''}{cell}-{role}-20261023";out=f.root/'successful-report'/job
                receipt=write(out/'receipt.json',{'synthetic_only':True})
                child={**spec,'kind':kind,'role':role,'seed':20261023,'job_id':job,'output_dir':str(out),'success_receipt':str(receipt)}
                refs[key].append(w._reference(write(out/'config.json',child)))
                if kind=='evaluation':
                    selected=f.selected[spec['task']];predictions=[]
                    for row,stratum in selected:
                        gold=row['supervision'].get('option_id') or row['supervision']['answer']
                        predictions.append({**w.dev.caps.initial._failure(row,stratum,512,'synthetic-only'),'status':'SUCCEEDED',
                            'generated_text':str(gold),'finish_reason':'stop'})
                    scored,summary=w._generation_scores(selected,predictions)
                    write(out/'metrics.json',summary);jsonl(out/'scored.private.jsonl',scored);jsonl(out/'predictions.private.jsonl',predictions)
                    for name,value in summary['strata'].items():table.append({**{k:child[k] for k in ('cell_id','model_id','family','task','role','seed')},'stratum':name,'job_status':'SUCCEEDED',**value})
                elif kind=='diagnostic':
                    by_task={task:{'technical_failures':0,'latency_p50_seconds':.1,'peak_reserved_bytes':200,'sample_count':12}
                             for task in {row['task_type'] for row,_ in f.selected[spec['task']]}}
                    path=write(out/'metrics.json',{'technical_failures':0,'gpu_identity':{'uuid':'GPU-12345678-1234-1234-1234-123456789abc'},
                        'by_task':by_task,'latency_p50_seconds':.1,'peak_reserved_bytes':200,'image_ablation_32':{'technical_failures':0} if spec['task']=='vqa' else None})
                    diag_paths.append(path)
    def terminal(config,ref):
        child=w._ref(ref);failed=child['kind']=='likelihood'
        return child,{'status':'FAILED' if failed else 'SUCCEEDED','error':'synthetic LL load failure' if failed else None},None if failed else {'synthetic_only':True}
    monkeypatch.setattr(w,'_terminal',terminal)
    commit='a'*40;common={'schema_version':'medvlm.v6.owner-evidence.v1','status':'SUCCEEDED','source_hash':'source','test_protocol_hash':f.protocol['protocol_hash']}
    ci={**common,'kind':'code_report_ci','source_commit':commit,'pushed_commit':commit,'published_report_table_sha256':w.canonical_hash(table),
        'cpu_tests':{'passed':1,'failed':0},'ci_runs':[{'head_sha':commit,'conclusion':'success','url':f'https://github.com/example/repo/actions/runs/{i}'} for i in range(3)]}
    protected=write(f.root/'protected.json',{'synthetic_only':True})
    preserve={**common,'kind':'existing_result_preservation','existing_v1_v5_preserved':True,'approved_release_preserved':True,'protected_artifacts':[w._reference(protected)]}
    evidence={name:w._reference(write(f.root/(name+'.json'),proof)) for name,proof in [('code_report_ci',ci),('existing_result_preservation',preserve)]}
    config={'scope':w.SCOPES['report'],'source_hash':'source','data':{'selection':f.proof,'test_freeze':f.freeze_proof,'evidence':evidence,**refs}}
    monkeypatch.setattr(w,'_telemetry_dispositions',lambda *a:{'conditions':{},'all_required_cpu_dispositions_recorded':True,'telemetry_all_evaluations_imported':True,'user_attention_required':False})
    summary=w._report(config)
    assert summary['v6_completed'] and not summary['missing_required_evidence']
    monkeypatch.setattr(w,'_packet',lambda *a:{'synthetic_only':True})
    rendered=w._render_report(config,summary,{})['RESULTS_KO.md']
    assert all(label in rendered for label in ('한국어(ko)','영어(en)','kormedmcqa','medquad','slake_open','vqa_rad_closed','관측 Pareto'))
    assert len(summary['source_task_findings'])==52 and {r['language'] for r in summary['source_task_findings']}=={'ko','en'}
    assert all(r['candidates'] and not r['test_based_family_or_seed_reselection'] for r in summary['usage_candidates'])
    assert {r['metric'] for r in summary['source_task_findings']}=={'strict_all_row_accuracy','lexical_f1','normalized_em'}
    assert all(c['metrics']['technical_failure_count']==4286 and c['metrics']['unsupported_count']==244 for c in summary['likelihood_conditions'].values())
    monkeypatch.setattr(w,'_telemetry_dispositions',lambda *a:{'conditions':{'synthetic':{'status':'UNAVAILABLE','reason':'SYNTHETIC_IMPORT_FAILURE'}},'all_required_cpu_dispositions_recorded':True,'telemetry_all_evaluations_imported':False,'user_attention_required':True})
    unavailable=w._report(config)
    assert unavailable['scientific_v6_completed'] and unavailable['v6_completed'] and not unavailable['telemetry_all_evaluations_imported'] and unavailable['user_attention_required']
    assert '중요: telemetry 가져오기 미완료/불명확' in w._render_report(config,unavailable,{})['RESULTS_KO.md']
    monkeypatch.setattr(w,'_telemetry_dispositions',lambda *a:{'conditions':{},'all_required_cpu_dispositions_recorded':False,'telemetry_all_evaluations_imported':False,'user_attention_required':True})
    incomplete=w._report(config)
    assert incomplete['scientific_v6_completed'] and not incomplete['v6_completed']
    original=w._read(diag_paths[0])
    for altered,reason in [({**original,'technical_failures':1},'speed_technical_failures'),
        ({**original,'gpu_identity':{'uuid':'GPU-87654321-1234-1234-1234-123456789abc'}},'speed_gpu_identity_mismatch'),
        ({**original,'gpu_identity':{'uuid':None}},'same_physical_gpu_identity_unverified'),
        ({**original,'gpu_identity':{'uuid':'None'}},'same_physical_gpu_identity_unverified')]:
        write(diag_paths[0],altered);negative=w._report(config)
        assert not negative['v6_completed'] and any(reason in item for item in negative['missing_required_evidence'])
    write(diag_paths[0],original)


@pytest.fixture
def telemetry_native(native_planning,monkeypatch):
    f=native_planning
    protocols={axis:w._axis_protocol(f.freeze,f.protocol,axis) for axis in ('text','vqa')}
    publication=w._catalog_publication(f.freeze,protocols,publish=True)
    w.atomic_json(f.root/'freeze/catalog_publication.json',publication)
    cfg=w.prepare_evaluation_config(f.freeze_path,'T2','tuned',20261023,f.root/'completion/evaluation/T2/tuned')
    out=Path(cfg['output_dir']);out.mkdir(parents=True,exist_ok=True);w.atomic_json(out/'config.json',cfg)
    predictions=[]
    for row,stratum in f.selected['text']:
        predictions.append({**w.dev.caps.initial._failure(row,stratum,512,'synthetic-only'),'status':'SUCCEEDED',
            'generated_text':str(row['supervision'].get('option_id') or row['supervision']['answer']),'finish_reason':'stop'})
    scored,metrics=w._generation_scores(f.selected['text'],predictions);metrics['protocol_hash']=cfg['protocol_hash']
    jsonl(out/'predictions.private.jsonl',predictions);jsonl(out/'scored.private.jsonl',scored);w.atomic_json(out/'metrics.json',metrics)
    w.atomic_json(out/'run_manifest.json',w._native_run_manifest(cfg,out));w.atomic_json(out/'completion.json',{'status':'SUCCEEDED','synthetic_only':True})
    receipt=write(Path(cfg['success_receipt']),{'synthetic_only':True,'artifacts':[w._reference(out/name) for name in (*w._artifact_names('evaluation'),'completion.json')]})
    cfgpath=write(f.root/'evaluation-input.json',cfg)
    proof={'config':w._reference(cfgpath),'receipt':w._reference(receipt)};f.registered[str(cfgpath)]=(cfg,proof,w._read(receipt))
    original_registered=w._registered
    def verified_registered(path,*args):
        value=original_registered(path,*args)
        if str(path)==str(cfgpath):
            assert all(w._reference(item['path'])==item for item in value[2]['artifacts']), 'REGISTERED_NATIVE_ARTIFACT_CHANGED'
        return value
    monkeypatch.setattr(w,'_registered',verified_registered)
    import_cfg=w.prepare_telemetry_import_config(cfgpath,f.root/'completion/telemetry_import/T2/tuned')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    return SimpleNamespace(**vars(f),native_config=cfg,native_path=cfgpath,import_config=import_cfg,protocols=protocols,publication=publication)


def test_native_axis_catalog_is_discovered_exact4530_1512_and_immutable(telemetry_native):
    f=telemetry_native;catalog=w.telemetry_import.load_protocol_catalog(f.freeze['data']['forge']['data_root'])
    for axis,n in [('text',4530),('vqa',1512)]:
        protocol=f.protocols[axis];entry=catalog[protocol['protocol_hash']]
        assert len(protocol['sample_ids'])==n and len(protocol['input_digests'])==n
        assert entry=={'split':'test','protocol_version':w.AXIS_PROTOCOL_VERSION,'data_hash':w.dev._dataset_hash(protocol),'sha256':f.publication['protocols'][axis]['sha256']}
    assert f.native_config['protocol_hash']==f.protocols['text']['protocol_hash']!=f.protocols['vqa']['protocol_hash']
    assert w._catalog_publication(f.freeze,f.protocols,publish=True)==f.publication
    path=Path(f.publication['protocols']['text']['path']);before=path.read_bytes();path.write_bytes(before+b' ')
    with pytest.raises(ValueError,match='IMMUTABLE_CATALOG'):w._catalog_publication(f.freeze,f.protocols,publish=True)
    assert path.read_bytes()==before+b' '


def test_actual_importer_once_native_hashes_historical_provenance_and_training_null_unchanged(telemetry_native,monkeypatch):
    f=telemetry_native;cfg=f.import_config;native=f.native_config;native_out=Path(native['output_dir']);calls=[]
    training=Path(f.freeze['data']['forge']['output_root'])/'training-before-evaluation';training.mkdir(parents=True)
    write(training/'config.json',{'run_id':'v6-additional-T2-20261114','status':'SUCCEEDED','model_id':'original/base'})
    write(training/'metrics.json',{'example_losses':[1.], 'update_example_counts':[1],'optimizer_steps':1,'samples_processed':1,'loss_mean':1.,
        'example_supervised_tokens':[2],'supervised_tokens':2,'learning_rates':[.1],'gradient_norms_before_clip':[.2],'final_learning_rate':0.})
    train_import=w.telemetry_import.import_directory(training,f.freeze['data']['forge']['output_root'],f.root.name)
    train_status=Path(f.freeze['data']['forge']['output_root'])/train_import['campaign_id']/train_import['run_id']/train_import['attempt_id']/'status.json'
    before_training=train_status.read_bytes();assert json.loads(before_training)['evaluation_completed'] is None
    original=w.telemetry_import.import_directory
    def importing(*args,**kwargs):
        assert 'torch' not in sys.modules and os.environ['CUDA_VISIBLE_DEVICES']==''
        assert kwargs['protocol_catalog']==w.telemetry_import.load_protocol_catalog(f.freeze['data']['forge']['data_root'])
        assert set(kwargs['expected_hashes'])=={'config.json','metrics.json','run_manifest.json','completion.json'}
        calls.append(args[0]);return original(*args,**kwargs)
    monkeypatch.setattr(w.telemetry_import,'import_directory',importing)
    hashes={p.name:w.file_hash(p) for p in native_out.iterdir()};receipt_before=Path(native['success_receipt']).read_bytes()
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as executor:first,second=list(executor.map(w._telemetry_import_once,[cfg,cfg]))
    assert first==second and first['status']=='IMPORTED' and len(calls)==1
    assert first['evaluation_receipt']['path']==cfg['data']['evaluation']['success_receipt'] and not first['training_evaluation_state_modified']
    imported=w._ref(first['imported_artifacts'][1])['evaluations']
    assert imported and all(row['provenance']=='historical_reconstructed' and row['receipt_verified'] is True and row['status']=='final' for row in imported)
    assert w._ref(first['imported_artifacts'][3])['evaluation_completed'] is True
    assert w._ref(first['imported_artifacts'][3])['training_succeeded'] is None
    assert hashes=={p.name:w.file_hash(p) for p in native_out.iterdir()} and receipt_before==Path(native['success_receipt']).read_bytes()
    assert before_training==train_status.read_bytes() and json.loads(train_status.read_text())['evaluation_completed'] is None
    manifest=w._read(native_out/'run_manifest.json')
    assert manifest['scheduled']==4530 and manifest['ordered_sample_ids_sha256']==w.canonical_hash(f.protocols['text']['sample_ids'])
    assert manifest['artifact_hashes']['metrics.json']==w.file_hash(native_out/'metrics.json')
    path=write(f.root/'telemetry-config.json',cfg);job=w.job_from_config(path)
    assert job['requires_gpu'] is False and job['dependency_policy']=='success' and native['job_id'] in job['dependencies']
    assert w.validate_config(json.loads(json.dumps(cfg)))==cfg


@pytest.mark.parametrize('name',['metrics.json','run_manifest.json','config.json','completion.json'])
def test_import_native_source_tamper_fails_before_sole_call(telemetry_native,monkeypatch,name):
    f=telemetry_native;path=Path(f.native_config['output_dir'])/name;path.write_text(path.read_text()+' ')
    monkeypatch.setattr(w.telemetry_import,'import_directory',lambda *a,**kw:pytest.fail('tampered native output reached importer'))
    with pytest.raises(AssertionError,match='REGISTERED_NATIVE_ARTIFACT_CHANGED'):w._telemetry_import_once(f.import_config)
    assert not w._import_intent_path(f.import_config).exists()


def test_durable_started_or_failed_import_never_retries_and_does_not_change_native_success(telemetry_native,monkeypatch):
    f=telemetry_native;cfg=f.import_config;calls=[]
    def interrupted(*a,**kw):calls.append('called');raise KeyboardInterrupt('synthetic process interruption')
    monkeypatch.setattr(w.telemetry_import,'import_directory',interrupted)
    with pytest.raises(KeyboardInterrupt):w._telemetry_import_once(cfg)
    assert w._read(w._import_intent_path(cfg))['status']=='STARTED'
    second=w._telemetry_import_once(cfg)
    assert calls==['called'] and second['status']=='INDETERMINATE' and second['user_attention_required']
    assert second['scientific_evaluation_status']=='SUCCEEDED' and Path(f.native_config['success_receipt']).exists()
    # An independent native CPU failure record is audited, never reset or claimed successful.
    w._import_intent_path(cfg).unlink();calls.clear()
    def unavailable(*a,**kw):calls.append('called');raise OSError('synthetic import-only failure')
    monkeypatch.setattr(w.telemetry_import,'import_directory',unavailable)
    first=w._telemetry_import_once(cfg);second=w._telemetry_import_once(cfg)
    assert first==second and calls==['called'] and first['status']=='UNAVAILABLE' and first['user_attention_required']
    assert first['scientific_evaluation_status']=='SUCCEEDED' and not first['telemetry_all_evaluations_imported']


def test_terminal_failed_cpu_job_reports_started_intent_without_mutation_or_import(telemetry_native,monkeypatch):
    f=telemetry_native;cfg=f.import_config;child,base=w._import_base(cfg);intent=w._import_intent_path(cfg)
    w.atomic_json(intent,{**base,'status':'STARTED','reason':'SOLE_CPU_IMPORT_CALL_INTENT'});before=intent.read_bytes()
    ref=w._reference(write(f.root/'telemetry-config.json',cfg))
    monkeypatch.setattr(w,'_terminal',lambda *a:(cfg,{'status':'FAILED','error':'synthetic CPU process death','attempts':3},None))
    monkeypatch.setattr(w.telemetry_import,'import_directory',lambda *a,**kw:pytest.fail('read-only failed audit cannot import'))
    coverage=w._telemetry_dispositions({'data':{'telemetry_imports':[ref]}},{child['job_id']:{'status':'SUCCEEDED'}})
    condition=coverage['conditions'][child['job_id']]
    assert condition['status']=='INDETERMINATE' and condition['cpu_completion']['status']=='FAILED' and condition['cpu_completion']['attempts']==3
    assert coverage['all_required_cpu_dispositions_recorded'] and not coverage['telemetry_all_evaluations_imported'] and coverage['user_attention_required']
    assert before==intent.read_bytes()
    missing=w._telemetry_dispositions({'data':{}},{child['job_id']:{'status':'SUCCEEDED'}})
    assert not missing['all_required_cpu_dispositions_recorded'] and missing['conditions'][child['job_id']]['status']=='NOT_RECORDED'
    empty=w._telemetry_dispositions({'data':{}},{child['job_id']:{'status':'FAILED'}})
    assert empty['required_evaluations']==0 and empty['all_required_cpu_dispositions_recorded'] and not empty['telemetry_all_evaluations_imported']


def test_import_rejects_torch_loaded_process_without_intent(telemetry_native,monkeypatch):
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace())
    with pytest.raises(ValueError,match='FRESH_CPU_REQUIRED'):w._telemetry_import_once(telemetry_native.import_config)
    assert not w._import_intent_path(telemetry_native.import_config).exists()


def test_prepared_import_pins_future_native_config_without_fabricated_receipt(telemetry_native,monkeypatch):
    f=telemetry_native;native=f.native_config;Path(native['success_receipt']).unlink()
    original=w._registered
    def only_frozen_prerequisites(path,*a):
        if str(path)==str(f.native_path):raise ValueError('SYNTHETIC_NATIVE_NOT_YET_SUCCEEDED')
        return original(path,*a)
    monkeypatch.setattr(w,'_registered',only_frozen_prerequisites)
    cfg=w.prepare_telemetry_import_config(f.native_path,f.root/'completion/telemetry_import/prepared-before-native')
    assert cfg['data']['evaluation']=={'config':w._reference(f.native_path),'success_receipt':native['success_receipt']}
    assert 'native_source_hashes' not in cfg['data'] and 'receipt' not in cfg['data']['evaluation']
    assert w.validate_config(cfg)==cfg
    monkeypatch.setattr(w.telemetry_import,'import_directory',lambda *a,**kw:pytest.fail('future native proof reached importer'))
    with pytest.raises(ValueError,match='NOT_YET_SUCCEEDED'):w._telemetry_import_once(cfg)
    assert not w._import_intent_path(cfg).exists()


def test_imported_telemetry_artifact_tamper_is_rejected_without_duplicate_call(telemetry_native,monkeypatch):
    f=telemetry_native;calls=[];original=w.telemetry_import.import_directory
    def importing(*a,**kw):calls.append(1);return original(*a,**kw)
    monkeypatch.setattr(w.telemetry_import,'import_directory',importing)
    first=w._telemetry_import_once(f.import_config);path=Path(first['imported_artifacts'][1]['path']);path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError,match='IMPORTED_ARTIFACT_CHANGED'):w._telemetry_import_once(f.import_config)
    assert calls==[1]


def test_cpu_telemetry_worker_lifecycle_receipt_exposes_honest_coverage_and_reuse(telemetry_native,monkeypatch):
    f=telemetry_native;cfg=f.import_config;path=write(f.root/'telemetry-config.json',cfg)
    monkeypatch.setattr(w.dev.caps,'_env',lambda config:None)
    calls=[];original=w.telemetry_import.import_directory
    def importing(*a,**kw):calls.append(1);return original(*a,**kw)
    monkeypatch.setattr(w.telemetry_import,'import_directory',importing)
    first=w.run_job(path,cfg['success_receipt']);second=w.run_job(path,cfg['success_receipt'])
    assert first==second and calls==[1] and first['status']=='SUCCEEDED' and first['telemetry_coverage']=='IMPORTED'
    assert first['scientific_evaluation_status']=='SUCCEEDED' and first['telemetry_all_evaluations_imported'] and not first['user_attention_required']
    assert not first['v6_completed'] and not first['clinical_quality_certified']
    assert w.verify_result(cfg)==first


def test_failed_native_eval_with_blocked_prepared_import_is_explicit_not_importable(telemetry_native,monkeypatch):
    f=telemetry_native;cfg=f.import_config;ref=w._reference(write(f.root/'telemetry-config.json',cfg))
    condition={'status':'BLOCKED_PREREQUISITE','error':'required dependency failed','attempts':0,'failure_count':0,'recovery_attempts':0}
    monkeypatch.setattr(w,'_terminal',lambda *a:(cfg,condition,None))
    monkeypatch.setattr(w,'_import_base',lambda *a:pytest.fail('failed native eval cannot require a fabricated success proof'))
    monkeypatch.setattr(w.telemetry_import,'import_directory',lambda *a,**kw:pytest.fail('blocked import cannot execute'))
    result=w._telemetry_dispositions({'data':{'telemetry_imports':[ref]}},{f.native_config['job_id']:{'status':'FAILED'}})
    row=result['conditions'][f.native_config['job_id']]
    assert row['status']=='NOT_APPLICABLE_NATIVE_EVALUATION_UNSUCCESSFUL' and row['scientific_evaluation_status']=='FAILED'
    assert row['cpu_completion']==condition and result['required_evaluations']==0 and result['all_required_cpu_dispositions_recorded']
    assert not result['telemetry_all_evaluations_imported'] and not w._import_intent_path(cfg).exists()


def test_real_terminal_projection_preserves_owner_telemetry_failure_domain_history(telemetry_native,monkeypatch):
    f=telemetry_native;cfg=f.import_config;ref=w._reference(write(f.root/'telemetry-config.json',cfg))
    record={'status':'FAILED','attempts':1,'failure_count':1,'recovery_attempts':0,'error':'worker exited137','error_signature':'native137',
        'failure_domain':'telemetry_reconstruction','telemetry_coverage':'INDETERMINATE','scientific_failure':False,
        'history':[{'event':'worker_failed','returncode':137,'preserved':True}]}
    manifest_path=write(f.root/'active-manifest.json',{'synthetic_only':True})
    state={'jobs':{cfg['job_id']:record},'manifest_path':str(manifest_path)};write(Path(cfg['state_path']),state)
    manifest={'jobs':[{'job_id':cfg['job_id'],'success_receipt':cfg['success_receipt']}]}
    monkeypatch.setattr(w.dev.caps,'_active',lambda path:(manifest,state))
    monkeypatch.setattr(w.dev.caps,'_child_config',lambda *a:(cfg,ref))
    child,condition,result=w._terminal(cfg,ref)
    assert result is None and condition['status']=='FAILED' and condition['native_telemetry_record']==record
    assert condition['failure_domain']=='telemetry_reconstruction' and condition['telemetry_coverage']=='INDETERMINATE' and condition['scientific_failure'] is False
    coverage=w._telemetry_dispositions({'state_path':cfg['state_path'],'data':{'telemetry_imports':[ref]}},{f.native_config['job_id']:{'status':'SUCCEEDED'}})
    exported=coverage['conditions'][f.native_config['job_id']]
    assert exported['status']=='UNAVAILABLE' and exported['cpu_completion']['native_telemetry_record']['history']==record['history']
    assert exported['scientific_evaluation_status']=='SUCCEEDED' and exported['user_attention_required'] and state['jobs'][cfg['job_id']]==record


def test_cpu_test_audit_shares_bounded_proofs_reaudits_next_call_and_rehashes_mutation(tmp_path,monkeypatch):
    shared=write(tmp_path/'shared-native-proof.json',{'version':1});calls=[];mutate={'enabled':False}
    @w._memo
    def selection(path):
        calls.append(path)
        return w._ref(w._reference(shared))
    monkeypatch.setattr(w,'_selection',selection)
    refs={kind:[w._reference(write(tmp_path/(kind+'.json'),{'job_id':kind}))] for kind in ('evaluations','diagnostics','likelihoods')}
    def terminal(config,ref):
        proof=w._selection('same-immutable-prerequisite');child=w._ref(ref)
        if mutate['enabled'] and child['job_id']=='diagnostics':write(shared,{'version':3})
        return child,{'status':'FAILED','synthetic_verified_version':proof['version']},None
    monkeypatch.setattr(w,'_terminal',terminal)
    config={'data':{**refs,'telemetry_imports':[]}}
    first=w._test_audit_summary(config)
    assert calls==['same-immutable-prerequisite'] and all(v['synthetic_verified_version']==1 for group in first['conditions'].values() for v in group.values())
    write(shared,{'version':2});second=w._test_audit_summary(config)
    assert len(calls)==2 and all(v['synthetic_verified_version']==2 for group in second['conditions'].values() for v in group.values())
    mutate['enabled']=True
    with pytest.raises(ValueError,match='REFERENCE_CHANGED_DURING_AUDIT'):w._test_audit_summary(config)
    assert len(calls)==3


@pytest.mark.parametrize('mode',['reload-child','train-child','main-train','main-cpu','reload-proof-error','reload-non-success'])
def test_actual_cli_subprocess_projects_declared_scope_without_mutating_native_receipts(tmp_path,mode):
    cfg={key:'a'*64 for key in w.IDENTITY_KEYS}
    cfg.update(kind='additional_audit' if mode=='main-cpu' else 'train',uses_gpu=mode!='main-cpu',
        scope=w.SCOPES['additional_audit'] if mode=='main-cpu' else w.TRAIN_SCOPE)
    path=write(tmp_path/'config.json',cfg);receipt=tmp_path/'native-receipt.json';directory=tmp_path/'stage';directory.mkdir()
    code=r'''
import json,os,sys
from pathlib import Path
from types import SimpleNamespace
from mediflow_medvlm import campaign_v6_completion as w
mode,config_path,receipt_path,directory=sys.argv[1:]
config=json.loads(Path(config_path).read_text())
identity={key:config[key] for key in w.IDENTITY_KEYS}
# Native reload_checkpoint returns probe_scope and identity fields, never scope.
reload_result={"status":"SUCCEEDED",**identity,"pid":123,"process_identity":{"pid":123,"boot_id":"synthetic","start_ticks":1},
    "probe_scope":"TRAINING_EXAMPLE_SANITY_CHECK","fresh_process_required":True,"adapter_weights_verified":True,
    "adapter_sha256":"b"*64,"trained_example_sample_id":"synthetic/training-only","trained_example_loss":.5,"target_tokens":4,
    "finite_forward":True,"base_dtype_signature":[],"generated_tokens":1,"generation_max_new_tokens":32,
    "generated_ids_sha256":"c"*64,"no_thinking_output":{"explicit_nonempty_thinking_detected":False},
    "dev_test_read":False,"runtime_versions":{},"finished_at":"synthetic"}
assert "scope" not in reload_result
train_result={"status":"SUCCEEDED","scope":"NATIVE_MAIN_ONE_PASS_TRAIN_ONLY",**identity,"training_pid":123,
    "process_identity":{"pid":123,"boot_id":"synthetic","start_ticks":1},"optimizer_steps":500,"samples_processed":7997,
    "metrics_sha256":"d"*64,"retry_start":"original_base","finished_at":"synthetic"}
def child(which,*args):
    assert [str(value) for value in args]==[config_path,directory,receipt_path]
    if mode=="reload-proof-error":raise w.dev.caps.initial.gate.GateError("SYNTHETIC_RELOAD_PROOF_FAILED")
    assert which==("train" if mode=="train-child" else "reload")
    result=dict(train_result if which=="train" else reload_result)
    if mode=="reload-non-success":result["status"]="FAILED"
    Path(receipt_path).write_text(json.dumps(result))
    return result
w.training_runtime=lambda actual:(config==actual and SimpleNamespace(train_checkpoint=lambda *a:child("train",*a),reload_checkpoint=lambda *a:child("reload",*a)))
def run_job(*args):
    assert list(args)==[config_path,receipt_path]
    if mode=="main-cpu":assert os.environ["CUDA_VISIBLE_DEVICES"]=="" and "torch" not in sys.modules
    result={"status":"SUCCEEDED","scope":config["scope"],**identity,"clinical_quality_certified":False}
    Path(receipt_path).write_text(json.dumps(result))
    return result
w.run_job=run_job
argv=["--config",config_path,"--receipt",receipt_path]
if mode in {"reload-child","reload-proof-error","reload-non-success"}:argv += ["--reload-checkpoint",directory]
elif mode=="train-child":argv += ["--train-checkpoint",directory]
raise SystemExit(w.main(argv))
'''
    result=subprocess.run([sys.executable,'-B','-c',code,mode,str(path),str(receipt),str(directory)],capture_output=True,text=True,
        env={key:value for key,value in os.environ.items() if key!='CUDA_VISIBLE_DEVICES'})
    if mode in {'reload-proof-error','reload-non-success'}:
        assert result.returncode==1 and not result.stdout and 'Traceback' not in result.stderr
        error=json.loads(result.stderr)
        assert error['reason_code']==('SYNTHETIC_RELOAD_PROOF_FAILED' if mode=='reload-proof-error' else 'COMPLETION_CLI_RESULT_NOT_SUCCEEDED')
        assert receipt.exists() is (mode=='reload-non-success')
    else:
        assert result.returncode==0,result.stderr
        assert json.loads(result.stdout)=={'status':'SUCCEEDED','scope':cfg['scope']}
        original=json.loads(receipt.read_text())
        if mode=='reload-child':assert 'scope' not in original and original['probe_scope']=='TRAINING_EXAMPLE_SANITY_CHECK' and original['dev_test_read'] is False
        if mode=='train-child':assert original['scope']=='NATIVE_MAIN_ONE_PASS_TRAIN_ONLY' and original['samples_processed']==7997 and original['optimizer_steps']==500
        assert not result.stderr
