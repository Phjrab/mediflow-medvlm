"""Full common validation: immutable CPU contracts, no GPU execution."""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from mediflow_medvlm import campaign_full_dev as d
from mediflow_medvlm import campaign_supervisor as supervisor
from test_campaign_validation_caps import planned as initial_planned, proposal
from test_campaign_validation_cap_refinement import predictions, native_metrics, state_record

REPO=Path(__file__).resolve().parents[1]


def write(path,value):d.atomic_json(path,value);return Path(path)


def active(path):
    state=d._read(path);manifest=d._read(state['manifest_path'])
    assert d.canonical_hash(manifest)==state['manifest_hash']
    supervisor.validate_manifest(manifest)
    return manifest,state


def update(p,jobs,records):
    manifest,state=active(p.state_path)
    manifest['jobs'].extend(jobs);state['jobs'].update(records);state['manifest_hash']=d.canonical_hash(manifest)
    write(state['manifest_path'],manifest);write(p.state_path,state)


@pytest.fixture
def planned(initial_planned,monkeypatch):
    p=initial_planned;original=copy.deepcopy(p.protocol)
    # Complete synthetic validation populations, with private references never in inputs.
    for axis,reference in original['validation_references'].items():
        rows=[json.loads(line) for line in Path(reference['path']).read_text().splitlines()]
        for row in rows:
            row['language']='en'
            if row['task_type']=='medical_mcq':row['input']['options']={'A':'x','B':'y'};row['supervision']['option_id']='B'
            elif row['task_type']=='context_qa':row['supervision']['answer']='yes'
            else:row['supervision']['answer']='Private synthetic target'
        Path(reference['path']).write_text(''.join(json.dumps(row)+'\n' for row in rows))
        reference['sha256']=d.file_hash(reference['path'])
    original['protocol_hash']=d.canonical_hash({k:v for k,v in original.items() if k!='protocol_hash'})
    write(p.protocol_path,original)
    final={'schema_version':'medvlm.v6.validation-caps-frozen.v1','cap_protocol_frozen':True,'further_bump_allowed':False,
        'test_files_read':False,'test_unlocked':False,'eval_context':4096,'train_context':2048,'fixed_template_date':'2026-11-14',
        'decode':original['decode'],'caps':d.caps.initial.MAXIMUM_CAPS,'initial_protocol':d._reference(p.protocol_path),
        'initial_protocol_hash':original['protocol_hash']}
    final['protocol_hash']=d.canonical_hash(final)
    output=p.root/'cap-final';files,source=d.caps.source_identity(REPO)
    cap={'kind':'freeze','scope':d.caps.SCOPES['freeze'],'job_id':d.caps.FREEZE_ID,'run_id':d.caps.FREEZE_ID,
        'source_files':files,'source_hash':source,'data_hash':d.canonical_hash(final),'model_hash':d.canonical_hash(original['models']),
        'data':{'protocol':d._reference(p.protocol_path)},'output_dir':str(output),'success_receipt':str(p.root/'receipts/caps.json')}
    cap['config_hash']=d.canonical_hash(cap);cap_path=write(p.root/'cap-config.json',cap)
    write(output/'config.json',cap);write(output/'protocol.final.json',final)
    write(output/'summary.json',{'final_protocol':final,'cap_freeze_audit_succeeded':True})
    job={**{k:cap[k] for k in d.IDENTITY_KEYS},'source_files':files,'cwd':str(REPO),'phase':'validation_caps','cell_id':'ALL_V6',
        'dependencies':[],'implementation_pending':False,'uses_gpu':False,'success_receipt':cap['success_receipt'],
        'argv':[d.sys.executable,'-m','mediflow_medvlm.campaign_validation_cap_refinement','--config',str(cap_path),'--receipt',cap['success_receipt']]}
    write(cap['success_receipt'],{'status':'SUCCEEDED','scope':cap['scope'],'cap_protocol_frozen':True,
        **{k:cap[k] for k in d.IDENTITY_KEYS},'artifacts':[d._reference(output/name) for name in ('config.json','protocol.final.json','summary.json')]})
    manifest={'campaign_id':'recorded','campaign_root':str(p.root),'jobs':[job]}
    state={'campaign_id':'recorded','manifest_path':str(p.root/'manifest.json'),'manifest_hash':d.canonical_hash(manifest),
        'jobs':{job['job_id']:state_record(job)}}
    write(state['manifest_path'],manifest);state_path=write(p.root/'state.json',state)
    monkeypatch.setattr(d.caps,'_active',active)
    real_source=d.source_identity
    monkeypatch.setattr(d,'source_identity',lambda repo=None:real_source(repo or REPO))
    value=SimpleNamespace(root=p.root,data_root=p.data_root,state_path=state_path,original=original,final=final,cap=cap,
        telemetry=p.data_root.parent/'forge-telemetry')
    value.config=d.build_config(cap['success_receipt'],state_path,'T1','base',p.root/'dev/T1/base',repo=REPO,forge_output_root=value.telemetry)
    value.path=write(p.root/'dev-configs/T1base.json',value.config)
    return value


def mock_gpu(monkeypatch,*,technical=False):
    calls=[]
    monkeypatch.setattr(d.caps.initial.first.main_train.smoke,'_inherited_supervisor_fds',lambda _: (10,11))
    class Ledger:
        session_seconds=.001
        def __init__(self,*a,**k):pass
        def start(self,*args):calls.append(('gpu',args))
        def finish(self):calls.append(('finish',))
        def snapshot(self):return {'resource_policy':d.ResourcePolicy().as_dict()}
    monkeypatch.setattr(d,'BudgetLedger',Ledger)
    def execute(config,protocol,selected,limits,stream):
        rows=predictions(config,protocol,selected,limits)
        for result,(row,_) in zip(rows,selected):result['generated_text']=row['supervision'].get('option_id',row['supervision']['answer'])
        if technical:rows[0]=d.caps.initial._failure(selected[0][0],selected[0][1],limits[selected[0][0]['task_type']],'INPUT_PLUS_CAP_EXCEEDS_EVAL_CONTEXT')
        stream.write(''.join(json.dumps(row)+'\n' for row in rows))
        return rows,native_metrics(config['role'],protocol,config['cell_id'])
    monkeypatch.setattr(d,'_execute',execute)
    monkeypatch.setattr(d,'_launch_forge_import',lambda *args:calls.append(('cpu_import',)))
    return calls


def register(p,config,path):
    job=d.job_from_config(path,repo=REPO)
    update(p,[job],{job['job_id']:{'status':'PENDING','attempts':0,'failure_count':0,'recovery_attempts':0}})
    return job


def complete(p,config,path):
    job=register(p,config,path);result=d.run_job(path,config['success_receipt'])
    manifest,state=active(p.state_path);state['jobs'][job['job_id']]=state_record(job);write(p.state_path,state)
    return result


def test_full_populations_cap_and_source_identity(planned):
    p=planned;c=p.config
    assert c['data']['count']==1516 and len(d._rows(p.original,'vqa'))==512
    assert c['decode']=={'do_sample':False,'num_beams':1,'use_cache':True,'max_time':None,'warmup_requests':0}
    assert c['eval_context']==4096 and c['data']['evaluation_protocol']['caps']==d.caps.initial.MAXIMUM_CAPS
    assert c['data']['evaluation_protocol']['further_bump_allowed'] is False
    assert 'Private synthetic target' not in json.dumps(c['data']['evaluation_protocol'])
    assert all('src/mediflow_medvlm/'+name in c['source_files'] for name in ('telemetry.py','telemetry_import.py','evaluation/metrics.py','evaluation/v2.py'))
    assert c['supervisor_lock_paths']==[str(p.data_root/'v6_v7/worker.lock'),str(p.data_root/'runtime/medvlm_gpu.lock')]
    assert c['forge']['output_root']==str(p.telemetry)
    assert d.validate_config(c,repo=REPO)==c


@pytest.mark.parametrize('field,value',[('eval_context',2048),('decode',{'do_sample':True}),('seed',1),('fixed_template_date','today'),('role','tuned'),('uses_gpu',False)])
def test_self_hashed_scientific_override_rejected(planned,field,value):
    c=copy.deepcopy(planned.config);c[field]=value;c['config_hash']=d.canonical_hash({k:v for k,v in c.items() if k!='config_hash'})
    with pytest.raises(ValueError):d.validate_config(c,repo=REPO)


@pytest.mark.parametrize('changed',['cap-artifact','cap-state','validation','source','test-path'])
def test_provenance_tamper_rejected(planned,changed):
    p=planned
    if changed=='cap-artifact':
        path=Path(p.cap['output_dir'])/'summary.json';path.write_text(path.read_text()+' ')
    elif changed=='cap-state':
        state=d._read(p.state_path);state['jobs'][d.caps.FREEZE_ID]['attempts']=2;write(p.state_path,state)
    elif changed=='validation':
        path=Path(p.config['data']['validation']['path']);path.write_text(path.read_text()+'\n')
    elif changed=='source':p.config['source_files']['src/mediflow_medvlm/telemetry_import.py']='0'*64
    else:
        protocol=copy.deepcopy(p.original);protocol['validation_references']['text']['path']=str(p.root/'test.jsonl')
        with pytest.raises(ValueError,match='ONLY_EXACT_FROZEN_VALIDATION_FILE_ALLOWED'):d._rows(protocol,'text')
        return
    with pytest.raises(ValueError):d.validate_config(p.config,repo=REPO)


def sample(sid,task='medical_mcq',gold='B',stratum='71875_mcq'):
    return ({'sample_id':sid,'source_id':'aihub_71875','task_type':task,'language':'ko','input':{'question':'Q','options':{'A':'x','B':'y'}},
        'supervision':{'option_id':gold,'answer':gold},'metadata':{}},stratum)


def test_strict_parser_denominators_raw_normalized_and_failures():
    selected=[sample(str(i)) for i in range(5)]
    texts=['B','The answer is B','abstain','{"final_answer":"B"}','B']
    predictions=[{'sample_id':str(i),'status':'SUCCEEDED','generated_text':text,'finish_reason':'stop'} for i,text in enumerate(texts)]
    predictions[-1].update(status='TECHNICAL_FAILURE',generated_text='',finish_reason='technical_failure')
    scored,metrics=d.score_predictions(selected,predictions);s=metrics['strata']['71875_mcq']
    assert s['scheduled']==5 and s['strict_all_row_accuracy']==.4 and s['valid_format_rate']==.4 and s['answered_accuracy']==1
    assert s['raw_em']==.2 and s['invalid_format_rate']==.2 and s['abstention_rate']==.2 and s['technical_failure_rate']==.2
    assert [r['status'] for r in scored]==['answered','invalid_format','abstained','answered','technical_failure']
    assert metrics['likelihood_auxiliary']['generation_validity_affected'] is False


def test_freeform_and_vqa_normalization_is_lexical_only():
    selected=[sample('q','medical_qa','No 5 mg','71875_short'),sample('v','medical_vqa','yes','slake_closed')]
    selected[1][0]['source_id']='slake';selected[1][0]['metadata']['answer_type']='CLOSED'
    rows=[{'sample_id':'q','status':'SUCCEEDED','generated_text':' no 5 MG ','finish_reason':'length'},
          {'sample_id':'v','status':'SUCCEEDED','generated_text':'Yes.','finish_reason':'stop'}]
    _,metrics=d.score_predictions(selected,rows)
    assert metrics['strata']['71875_short']['normalized_em']==1 and metrics['strata']['71875_short']['raw_em']==0
    assert metrics['strata']['71875_short']['lexical_f1']==1 and metrics['strata']['71875_short']['truncated_rate']==1
    assert metrics['strata']['slake_closed']['normalized_em']==1
    assert metrics['clinical_quality_certified'] is False and metrics['strata']['slake_closed']['clinical_accuracy'] is None


def test_real_role_lifecycle_technical_failure_denominator_and_ledger_order(planned,monkeypatch):
    p=planned;calls=mock_gpu(monkeypatch,technical=True)
    result=complete(p,p.config,p.path)
    assert result['status']=='SUCCEEDED' and result['evaluation_completed'] is True and result['technical_complete'] is False
    metrics=d._read(Path(p.config['output_dir'])/'metrics.json')
    assert metrics['scheduled']==1516 and metrics['technical_failure_count']==1
    assert metrics['strata']['71875_mcq']['scheduled']==587 and metrics['strata']['71875_mcq']['technical_failure_count']==1
    assert calls[-2:]==[('finish',),('cpu_import',)]
    assert d.run_job(p.path,p.config['success_receipt'])==result


@pytest.mark.parametrize('field,value',[('test_unlocked',True),('evaluation_completed',False),('test_files_read',True),('technical_complete',False)])
def test_native_marker_semantics_not_just_hashes(planned,monkeypatch,field,value):
    p=planned;mock_gpu(monkeypatch);complete(p,p.config,p.path)
    path=Path(p.config['output_dir'])/'completion.json';marker=d._read(path);marker[field]=value;write(path,marker)
    with pytest.raises(ValueError):d.verify_result(p.config,repo=REPO)


def test_run_manifest_and_scored_artifact_tamper(planned,monkeypatch):
    p=planned;mock_gpu(monkeypatch);complete(p,p.config,p.path)
    output=Path(p.config['output_dir']);run=d._read(output/'run_manifest.json');run['split']='test';write(output/'run_manifest.json',run)
    marker=d._read(output/'completion.json');marker['artifacts']=[d._reference(output/name) for name in ('config.json','evaluation_protocol.json','metrics.json','predictions.private.jsonl','run_manifest.json','scored.private.jsonl')];write(output/'completion.json',marker)
    with pytest.raises(ValueError,match='FULL_DEV_RUN_MANIFEST_CHANGED'):d.verify_result(p.config,repo=REPO)


def test_lease_guard_precedes_all_gpu_work(planned,monkeypatch):
    def reject(_):raise ValueError('NO_OWNER_LEASE')
    monkeypatch.setattr(d.caps.initial.first.main_train.smoke,'_inherited_supervisor_fds',reject)
    monkeypatch.setattr(d,'BudgetLedger',lambda *a,**k:pytest.fail('ledger before physical lease'))
    with pytest.raises(ValueError,match='NO_OWNER_LEASE'):d.run_job(planned.path,planned.config['success_receipt'])


def test_cpu_child_explicit_cuda_hidden_and_no_inherited_lease(planned,monkeypatch):
    calls=[];monkeypatch.setattr(d.subprocess,'run',lambda *a,**k:calls.append((a,k)))
    d._launch_forge_import(planned.path,planned.config)
    args,kwargs=calls[0]
    assert '--forge-import-only' in args[0] and kwargs['env']['CUDA_VISIBLE_DEVICES']=='' and kwargs['close_fds'] is True
    assert 'timeout' not in kwargs


def test_import_real_supported_api_once_expected_hashes_and_catalog(planned,monkeypatch):
    p=planned;mock_gpu(monkeypatch);complete(p,p.config,p.path)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    if 'torch' in d.sys.modules:pytest.skip('fresh-child no-torch assertion is verified separately')
    calls=[];real=d.telemetry_import.import_directory
    def tracked(*args,**kwargs):calls.append(kwargs);return real(*args,**kwargs)
    monkeypatch.setattr(d.telemetry_import,'import_directory',tracked)
    receipt=d._forge_import_once(p.config)
    assert receipt['status']=='SUCCEEDED' and len(calls)==1
    assert set(calls[0]['expected_hashes'])=={'run_manifest.json','metrics.json','completion.json'}
    assert p.config['protocol_hash'] in calls[0]['protocol_catalog']
    assert calls[0]['protocol_catalog'][p.config['protocol_hash']]['data_hash']==d._dataset_hash(p.config['data']['evaluation_protocol'])
    assert d._forge_import_once(p.config)==receipt and len(calls)==1
    assert d._import_coverage(p.config,True)['status']=='SUCCEEDED'
    imported=receipt['imported_artifacts'][0]
    path=Path(imported['path']);path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError,match='FORGE_IMPORTED_ARTIFACT_CHANGED'):d._import_coverage(p.config,True)


def test_import_failure_and_started_intent_never_repeat_or_change_native(planned,monkeypatch):
    p=planned;mock_gpu(monkeypatch);result=complete(p,p.config,p.path);monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    if 'torch' in d.sys.modules:pytest.skip('fresh-child no-torch assertion is verified separately')
    calls=[]
    def fail(*a,**k):calls.append(1);raise OSError('catalog unavailable')
    monkeypatch.setattr(d.telemetry_import,'import_directory',fail)
    receipt=d._forge_import_once(p.config)
    assert receipt['status']=='FAILED' and d.verify_result(p.config,repo=REPO)==result
    assert d._forge_import_once(p.config)==receipt and len(calls)==1
    receipt['status']='STARTED';write(p.config['forge']['receipt_path'],receipt)
    assert d._forge_import_once(p.config)['status']=='STARTED' and len(calls)==1


def full_plan(p):
    paths=[];configs=[];jobs=[]
    for cell in d.CELLS:
        for role in d.caps.initial.ROLES:
            config=d.build_config(p.cap['success_receipt'],p.state_path,cell[0],role,p.root/'dev'/cell[0]/role,
                repo=REPO,forge_output_root=p.telemetry)
            path=write(p.root/'dev-configs'/(cell[0]+role+'.json'),config)
            paths.append(path);configs.append(config);jobs.append(d.job_from_config(path,repo=REPO))
    update(p,jobs,{j['job_id']:state_record(j,'FAILED') for j in jobs})
    aggregate=d.build_aggregate_config(p.cap['success_receipt'],p.state_path,paths,p.root/'dev-summary',repo=REPO)
    path=write(p.root/'dev-aggregate-config.json',aggregate)
    return configs,paths,jobs,aggregate,path


def test_exact_twenty_gpu_roles_and_original_cpu_gate(planned):
    p=planned;configs,paths,jobs,aggregate,path=full_plan(p)
    assert [j['job_id'] for j in jobs]==d.ROLE_IDS
    assert all(j['requires_gpu'] and j['uses_gpu'] and j['dependencies']==[d.caps.FREEZE_ID] and j['dependency_policy']=='success' for j in jobs)
    assert [c['data']['count'] for c in configs]==[1516]*12+[512]*8
    job=d.job_from_config(path,repo=REPO)
    assert job['job_id']=='v6-full_dev' and job['requires_gpu'] is False and job['uses_gpu'] is False
    assert job['dependencies']==[d.caps.FREEZE_ID,*d.ROLE_IDS] and job['dependency_policy']=='terminal'
    invalid=paths[:-1]
    with pytest.raises(ValueError,match='EXACT_ORDERED_TWENTY'):d.build_aggregate_config(p.cap['success_receipt'],p.state_path,invalid,p.root/'bad',repo=REPO)


def test_aggregate_retains_terminal_failures_denominators_and_pending_auxiliaries(planned,monkeypatch):
    p=planned;mock_gpu(monkeypatch);configs,paths,jobs,aggregate,path=full_plan(p)
    result=d.run_job(paths[0],configs[0]['success_receipt'])
    manifest,state=active(p.state_path);state['jobs'][jobs[0]['job_id']]=state_record(jobs[0]);write(p.state_path,state)
    receipt=d.run_job(path,aggregate['success_receipt']);summary=d._read(Path(aggregate['output_dir'])/'summary.json')
    assert receipt['generation_evaluation_terminal'] is True and receipt['full_dev_completed'] is False and receipt['r2_full_evaluation_completed'] is False
    assert summary['all_roles_succeeded'] is False and summary['all_roles_technical_complete'] is False
    assert summary['conditions'][jobs[1]['job_id']]['unavailable_count']==1516
    assert summary['conditions'][jobs[-1]['job_id']]['unavailable_count']==512
    assert sum(s['scheduled'] for s in summary['conditions'][jobs[1]['job_id']]['strata'].values())==1516
    assert summary['paired_transitions']['T1']['include_unavailable_as_incorrect'] is True
    assert summary['paired_transitions']['T1']['both_roles_technical_complete'] is False
    assert summary['paired_transitions']['T1']['strata']['71875_mcq']['correct_transitions']=={'1->0':587}
    assert summary['forge_import_complete'] is False and summary['forge_imports'][jobs[0]['job_id']]['status']=='NOT_RECORDED'
    assert summary['likelihood_auxiliary']['status']=='PENDING_NOT_IMPLEMENTED'
    assert summary['separate_speed_protocol']['status']=='PENDING_NOT_EXECUTED'
    assert summary['bootstrap_confidence_intervals']['status']=='PENDING_NOT_COMPUTED'
    assert all(summary[k] is False for k in ('family_selection_completed','test_unlocked','test_files_read','clinical_quality_certified','v6_completed','campaign_terminal'))
    marker_path=Path(aggregate['output_dir'])/'completion.json';marker=d._read(marker_path);marker['test_files_read']=True;write(marker_path,marker)
    with pytest.raises(ValueError,match='FULL_DEV_AGGREGATE_CHANGED'):d.verify_result(aggregate,repo=REPO)


def test_aggregate_refuses_nonterminal_role(planned):
    p=planned;configs,paths,jobs,aggregate,path=full_plan(p)
    manifest,state=active(p.state_path);state['jobs'][jobs[-1]['job_id']]['status']='PENDING';write(p.state_path,state)
    with pytest.raises(ValueError,match='CAP_CHILD_NOT_TERMINAL'):d.run_job(path,aggregate['success_receipt'])
    assert not Path(aggregate['success_receipt']).exists()


@pytest.mark.parametrize('fault',['load','postprocess'])
def test_fatal_native_call_preserves_full_planned_failure_rows(planned,monkeypatch,fault):
    from mediflow_medvlm.evaluation import v6_native
    p=planned;selected=d._rows(p.original,'text')[:3]
    cuda=SimpleNamespace(reset_peak_memory_stats=lambda:None)
    monkeypatch.setitem(d.sys.modules,'torch',SimpleNamespace(cuda=cuda))
    def load(*a,**k):
        if fault=='load':raise RuntimeError('synthetic load failure')
        return object()
    monkeypatch.setattr(v6_native,'load_evaluation_bundle',load)
    monkeypatch.setattr(d.caps.initial,'_predict_one',lambda *a,**k:(_ for _ in ()).throw(ValueError('synthetic decode failure')))
    file=p.root/'fatal.private.jsonl'
    with file.open('w') as stream:
        with pytest.raises(d.caps.initial.CalibrationFatal):d._execute(p.config,p.original,selected,p.final['caps'],stream)
    rows=[json.loads(line) for line in file.read_text().splitlines()]
    assert len(rows)==3 and all(row['status']=='TECHNICAL_FAILURE' and row['model_fatal'] for row in rows)
    assert [row['sample_id'] for row in rows]==[row['sample_id'] for row,_ in selected]


def test_all_actual_import_sidecars_allow_complete_coverage_without_full_r2_claim(planned,monkeypatch):
    p=planned;mock_gpu(monkeypatch);monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    if 'torch' in d.sys.modules:pytest.skip('fresh CPU importer tested with no torch')
    configs,paths,jobs,aggregate,path=full_plan(p)
    manifest,state=active(p.state_path)
    for config,config_path,job in zip(configs,paths,jobs):
        d.run_job(config_path,config['success_receipt'])
        assert d._forge_import_once(config)['status']=='SUCCEEDED'
        state['jobs'][job['job_id']]=state_record(job)
    write(p.state_path,state)
    summary=d._summary(aggregate)
    assert summary['all_roles_succeeded'] is True and summary['all_roles_technical_complete'] is True
    assert summary['forge_import_complete'] is True and len(summary['forge_imports'])==20
    assert summary['full_dev_completed'] is False and summary['r2_full_evaluation_completed'] is False
    assert summary['human_review']=={'status':'NOT_PERFORMED','performed':0}
    hashes=[d._read(Path(config['forge']['receipt_path']))['result'] for config in configs]
    assert len({result['run_id'] for result in hashes})==20
    for axis in ('text','vqa'):
        assert len({d._dataset_hash(config['data']['evaluation_protocol']) for config in configs if config['task']==axis})==1
    assert len({config['data_hash'] for config in configs})==10
    assert len({config['config_hash'] for config in configs})==20


def test_forge_namespace_must_not_overlap_scientific_artifacts(planned):
    p=planned
    for output_root in (p.root,p.data_root,p.data_root.parent):
        with pytest.raises(ValueError,match='FORGE_OUTPUT_MUST_BE_DISJOINT'):
            d.build_config(p.cap['success_receipt'],p.state_path,'T1','base',p.root/'other',repo=REPO,forge_output_root=output_root)
