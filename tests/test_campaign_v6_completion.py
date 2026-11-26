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
