"""Finite CPU selection math, immutable lifecycle, and native-proof tampering."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from mediflow_medvlm import campaign_family_selection as s

REPO = Path(__file__).resolve().parents[1]


def strata(axis, score=.5):
    keys = ("71875_mcq","71875_short","71875_long","pubmedqa_labeled","medquad") if axis == "text" else ("slake_open","slake_closed")
    return {k:{"scheduled":10+i,"technical_failure_count":0,"strict_all_row_accuracy":score,"normalized_em":score,"lexical_f1":score} for i,k in enumerate(keys)}


def candidates():
    return [{**s.dev.cell_spec(cell[0]),**s.score_components(strata(cell[2]),cell[2]),
             "eligible":True,"exclusion_reason":None,"trained_cell":{"original":"unchanged"}} for cell in s.dev.CELLS]


def test_cpu_import_and_exact_source_closure():
    result = subprocess.run([sys.executable,"-c","import sys; from mediflow_medvlm import campaign_family_selection; assert 'torch' not in sys.modules"],cwd=REPO,capture_output=True,text=True)
    assert result.returncode == 0,result.stderr
    files,source = s.source_identity(REPO)
    assert set(files) == set(s.dev.source_identity(REPO)[0])|{"src/mediflow_medvlm/"+name for name in ("campaign_bootstrap.py","campaign_choice_likelihood.py","campaign_family_selection.py")}
    assert source == s.canonical_hash(files)


def test_text_macro_equal_stratum_weight_and_not_pooled_or_answered_accuracy():
    x = strata("text")
    for key,value in zip(x,[.9,.3,.6,.8,.2]): x[key]["strict_all_row_accuracy"] = x[key]["lexical_f1"] = value
    x["71875_mcq"].update(scheduled=587,answered_accuracy=1.)
    result = s.score_components(x,"text")
    assert result["score_components"]["K"] == pytest.approx(.6)
    assert result["score_components"]["E"] == .5
    assert result["macro_score"] == pytest.approx(.55) and result["tie_break_score"] == pytest.approx(.85)
    assert result["scheduled"] == sum(v["scheduled"] for v in x.values())


def test_vqa_equal_open_closed_primary_but_weighted_overall_tie():
    x = strata("vqa"); x["slake_open"].update(scheduled=9,normalized_em=1.); x["slake_closed"].update(scheduled=1,normalized_em=0.)
    result = s.score_components(x,"vqa")
    assert result["macro_score"] == .5 and result["tie_break_score"] == .9


def test_family_representative_then_top_two_distinct_families_and_eight_fixed_seeds():
    cs = candidates()
    for c in cs: c["macro_score"] = {"T1":.99,"T2":.98,"T3":.97,"T4":.96,"T5":.95,"T6":.94,"V1":.9,"V2":.8,"V3":.85,"V4":.7}[c["cell_id"]]
    result = s.select_candidates(cs)
    assert result["selected_cells_by_axis"] == {"text":["T1","T3"],"vqa":["V1","V3"]}
    assert result["family_representatives"]["text"]["qwen"] == "T1"
    assert result["additional_seed_count"] == 8
    assert [p["seed"] for p in result["additional_seed_plan"]] == [20261114,20261115]*4
    assert all(result[k] == value for k,value in s.FLAGS.items())


def test_absolute_tuned_scores_and_ties_use_unrounded_secondary_then_model_id():
    cs = candidates()
    # Same primary within Qwen: classification tie wins; huge Base gain is ignored.
    cs[0].update(tie_break_score=.2,base_delta=1000)
    cs[1].update(tie_break_score=.3,base_delta=-1000)
    result = s.select_candidates(cs)
    assert result["family_representatives"]["text"]["qwen"] == "T2"
    # Exact primary+secondary tie within Gemma resolves model ID, not cell index.
    assert result["family_representatives"]["text"]["gemma"] == "T3"
    cs[0]["macro_score"] = .500000000000001
    assert s.select_candidates(cs)["family_representatives"]["text"]["qwen"] == "T1"


def test_failed_cells_excluded_and_missing_families_reported_without_replacement():
    cs = candidates()
    for c in cs:
        c["eligible"] = c["family"] in {"qwen","gemma_vlm"}
        c["exclusion_reason"] = None if c["eligible"] else "TUNED_TECHNICALLY_INCOMPLETE"
    out = s.select_candidates(cs)
    assert len(out["selected_cells_by_axis"]["text"]) == len(out["selected_cells_by_axis"]["vqa"]) == 1
    assert out["additional_seed_count"] == 4
    assert all(v["reason"] == "FEWER_THAN_TWO_TECHNICALLY_COMPLETE_FAMILIES" for v in out["shortfalls"].values())
    for c in cs: c["eligible"] = False
    assert s.select_candidates(cs)["additional_seed_plan"] == []


@pytest.mark.parametrize("bad",[float("nan"),float("inf"),-.1,1.1,True,None])
def test_nonfinite_or_invalid_score_rejected(bad):
    cs = candidates(); cs[0]["macro_score"] = bad
    with pytest.raises(ValueError,match="INVALID_RATE"): s.select_candidates(cs)


def test_matrix_family_and_denominator_tampering_rejected():
    cs = candidates(); cs[1]["family"] = "qwen2"
    with pytest.raises(ValueError,match="DECLARED_MODEL_FAMILY"): s.select_candidates(cs)
    with pytest.raises(ValueError,match="EXACT_TEN"): s.select_candidates(candidates()[:-1])
    x = strata("text"); x["71875_mcq"]["scheduled"] = 0
    with pytest.raises(ValueError,match="DENOMINATOR"): s.score_components(x,"text")
    x = strata("vqa"); x["test_metrics"] = {}
    with pytest.raises(ValueError,match="EXACT_STRATA"): s.score_components(x,"vqa")


@pytest.fixture
def lifecycle(tmp_path,monkeypatch):
    root = tmp_path/"v6-forge-recorded-synthetic"; root.mkdir()
    inputs = {}
    for name in ("full_dev","bootstrap","choice_likelihood"):
        path = root/(name+".json"); s.atomic_json(path,{"synthetic":True}); inputs[name] = {"config":s._reference(path)}
    data = {**inputs,"initial_protocol":{"original":"pinned"},"final_protocol":{"caps":"pinned"},"validation":{"text":{"ordered_sample_ids":["a"]},"vqa":{"ordered_sample_ids":["b"]}},"candidates":candidates()}
    calls = []
    def collect(*paths):
        calls.append(paths)
        for key,path in zip(inputs,paths): assert s._reference(path) == inputs[key]["config"]
        return copy.deepcopy(data),{"state_path":str(root/"state.json"),"model_hash":"original_models"}
    monkeypatch.setattr(s,"_collect",collect); monkeypatch.setenv("CUDA_VISIBLE_DEVICES","")
    config = s.prepare_config(*(inputs[k]["config"]["path"] for k in inputs),root/"family_selection/frozen",repo=REPO)
    path = root/"selection-config.json"; s.atomic_json(path,config)
    return root,config,path,data,calls


def test_config_rehash_cannot_change_scientific_procedure_or_sources(lifecycle):
    _,config,_,_,_ = lifecycle
    assert s.validate_config(config,repo=REPO) == config
    for key,value in (("seed",5),("procedure",{**s.PROCEDURE,"additional_seeds":[7]}),("device","cuda"),("data",{**config["data"],"candidates":list(reversed(config["data"]["candidates"]))})):
        changed = copy.deepcopy(config); changed[key] = value
        changed["config_hash"] = s.canonical_hash({k:v for k,v in changed.items() if k!="config_hash"})
        with pytest.raises(ValueError,match="FIXED_PROCEDURE_OR_PROVENANCE"): s.validate_config(changed,repo=REPO)
    changed = copy.deepcopy(config); changed["source_hash"] = "bad"
    changed["config_hash"] = s.canonical_hash({k:v for k,v in changed.items() if k!="config_hash"})
    with pytest.raises(ValueError,match="HASH_OR_SOURCE"): s.validate_config(changed,repo=REPO)


def test_immutable_lifecycle_receipt_verification_and_no_test_access(lifecycle,monkeypatch):
    root,config,path,_,_ = lifecycle
    result = s.run_job(path,config["success_receipt"])
    assert result == s.verify_result(config,repo=REPO) == s.run_job(path,config["success_receipt"])
    assert result["test_files_read"] is False and result["test_unlocked"] is False
    assert len(result["artifacts"]) == 3 and result["additional_seed_count"] == 8
    output = Path(config["output_dir"])
    selection = s._read(output/"selection.json"); selection["selected_cells_by_axis"]["text"] = ["T5"]
    s.atomic_json(output/"selection.json",selection)
    marker = s._read(output/"completion.json"); marker["selection_sha256"] = s.file_hash(output/"selection.json")
    marker["artifacts"] = [s._reference(output/name) for name in ("config.json","selection.json")]
    s.atomic_json(output/"completion.json",marker)
    with pytest.raises(ValueError,match="RESULT_CHANGED"): s.verify_result(config,repo=REPO)
    denied = root/"test"/"metrics.json"
    with pytest.raises(ValueError,match="TEST_PATH_FORBIDDEN"): s._read(denied)
    denied = root/"test.jsonl"
    with pytest.raises(ValueError,match="TEST_PATH_FORBIDDEN"): s._reference(denied)


def test_cpu_job_fixed_dependencies_destinations_and_environment(lifecycle,monkeypatch):
    root,config,path,_,calls = lifecycle
    job = s.job_from_config(path,repo=REPO)
    assert job["dependencies"] == ["v6-full_dev","v6-bootstrap","v6-choice-likelihood"]
    assert job["dependency_policy"] == "success" and job["uses_gpu"] is False and job["requires_gpu"] is False
    assert job["argv"][1:3] == ["-m","mediflow_medvlm.campaign_family_selection"]
    with pytest.raises(ValueError,match="DEDICATED_DESTINATION"):
        s.prepare_config(*(config["data"][k]["config"]["path"] for k in ("full_dev","bootstrap","choice_likelihood")),root/"full_dev/aggregate")
    n = len(calls); monkeypatch.setenv("CUDA_VISIBLE_DEVICES","0")
    with pytest.raises(ValueError,match="FRESH_CPU_PROCESS"): s.run_job(path,config["success_receipt"])
    assert len(calls) == n
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES",""); monkeypatch.setitem(sys.modules,"torch",object())
    with pytest.raises(ValueError,match="FRESH_CPU_PROCESS"): s.run_job(path,config["success_receipt"])


@pytest.fixture
def native(tmp_path):
    root = tmp_path/"v6-forge-recorded-native"; root.mkdir()
    config = {"job_id":"native","run_id":"native","source_files":{"src/mediflow_medvlm/campaign_family_selection.py":s.file_hash(Path(s.__file__))},"data_hash":"data","model_hash":"model","state_path":str(root/"state.json"),"success_receipt":str(root/"receipts/native.json")}
    config["source_hash"] = s.canonical_hash(config["source_files"]); config["config_hash"] = s.canonical_hash(config)
    path = root/"native-config.json"; s.atomic_json(path,config)
    job = {"job_id":"native","run_id":"native","cwd":str(REPO),"source_files":config["source_files"],"success_receipt":config["success_receipt"],
           **{k:config[k] for k in s.HASH_FIELDS},"argv":[sys.executable,"-m","mediflow_medvlm.campaign_full_dev","--config",str(path),"--receipt",config["success_receipt"]]}
    artifact = root/"native-metrics.json"; s.atomic_json(artifact,{"score":"audited"})
    receipt = {"status":"SUCCEEDED",**{k:config[k] for k in s.IDENTITY_KEYS},"test_files_read":False,"test_unlocked":False,"artifacts":[s._reference(artifact)]}
    Path(config["success_receipt"]).parent.mkdir(); s.atomic_json(config["success_receipt"],receipt)
    attempt = root/"attempts/native/0001"; attempt.mkdir(parents=True)
    s.atomic_json(attempt/"launch.json",{"launch_token":"token","attempt":1,"campaign_id":root.name,"job":job,"job_hash":s.canonical_hash(job)})
    s.atomic_json(attempt/"exit.json",{"launch_token":"token","exit_code":0})
    record = {"status":"SUCCEEDED","attempts":1,"failure_count":0,"recovery_attempts":0,"attempt_dir":str(attempt),"launch_token":"token","exit_code":0,
              "verification":{"receipt":config["success_receipt"],"receipt_sha256":s.file_hash(config["success_receipt"]),"artifact_count":1}}
    return root,job,record,config


def test_real_native_hash_attempt_artifact_audit(native):
    _,job,record,config = native
    actual,receipt,proof = s._audit_job(job,record,"campaign_full_dev")
    assert actual == config and receipt["status"] == "SUCCEEDED" and proof["terminal_record"]["attempts"] == 1


def test_native_test_path_rejected_before_native_reader(native,monkeypatch):
    root,job,record,_ = native
    job["argv"][4] = str(root/"test"/"metrics.json")
    monkeypatch.setattr(s.dev.caps,"_child_config",lambda *args:pytest.fail("forbidden test path reached native reader"))
    with pytest.raises(ValueError,match="TEST_PATH_FORBIDDEN"): s._audit_job(job,record,"campaign_full_dev")


@pytest.mark.parametrize("tamper",["failed","artifact","source","exit_token","exit_code","launch_job","launch_attempt","receipt_identity","receipt_hash"])
def test_native_tampering_fails_before_selection(native,tamper):
    _,job,record,_ = native; attempt = Path(record["attempt_dir"])
    if tamper == "failed": record["status"] = "FAILED"
    elif tamper == "artifact": Path(s._read(job["success_receipt"])["artifacts"][0]["path"]).write_text("changed")
    elif tamper == "source": job["source_files"] = {**job["source_files"],"src/mediflow_medvlm/runtime.py":"wrong"}
    elif tamper.startswith("exit_"):
        data = s._read(attempt/"exit.json"); data["launch_token" if tamper == "exit_token" else "exit_code"] = "forged" if tamper == "exit_token" else 1; s.atomic_json(attempt/"exit.json",data)
    elif tamper.startswith("launch_"):
        data = s._read(attempt/"launch.json"); data["job_hash" if tamper == "launch_job" else "attempt"] = "forged"; s.atomic_json(attempt/"launch.json",data)
    elif tamper == "receipt_identity":
        data = s._read(job["success_receipt"]); data["model_hash"] = "wrong"; s.atomic_json(job["success_receipt"],data)
        record["verification"]["receipt_sha256"] = s.file_hash(job["success_receipt"])
    else: record["verification"]["receipt_sha256"] = "wrong"
    with pytest.raises(ValueError): s._audit_job(job,record,"campaign_full_dev")


@pytest.fixture
def collector(tmp_path,monkeypatch):
    """Real collector joins and eligibility; native verifier separately tested above."""
    root = tmp_path/"v6-forge-recorded-collector"; root.mkdir()
    anchor = root/"original-training-proof.json"; s.atomic_json(anchor,{"original":True})
    selected = {}
    for axis in ("text","vqa"):
        selected[axis] = [({"sample_id":axis+"-"+key+"-"+str(i),"split":"validation"},key)
                          for key,value in strata(axis).items() for i in range(value["scheduled"])]
    validation_refs = {}
    for axis,rows in selected.items():
        path = root/(axis+"-validation.json"); s.atomic_json(path,{"ids":[r["sample_id"] for r,_ in rows]}); validation_refs[axis] = s._reference(path)
    original = {"campaign_id":root.name,"campaign_root":str(root),"validation_references":validation_refs,
                "models":{cell[0]:{key:s._reference(anchor) for key in ("config","receipt","checkpoint","metrics","reload","final_selection")} for cell in s.dev.CELLS}}
    original_path = root/"initial-protocol.json"; s.atomic_json(original_path,original); original_ref = s._reference(original_path)
    configs = {}; refs = []
    conditions = {}; records = {}
    for cell in s.dev.CELLS:
        for role in ("base","tuned"):
            key = f"v6-full-dev-{cell[0]}-{role}"
            conf = {"job_id":key,"task":cell[2],"role":role,"data":{"initial_protocol":original_ref,
                    "ordered_sample_ids":[r["sample_id"] for r,_ in selected[cell[2]]],"evaluation_protocol":{"split":"validation","fixed":True}}}
            path = root/(key+".json"); s.atomic_json(path,conf); configs[key] = conf; refs.append(s._reference(path))
            conditions[key] = {"status":"SUCCEEDED","technical_complete":True,"strata":strata(cell[2]),"receipt":s._reference(anchor)}
            records[key] = {"status":"SUCCEEDED","attempts":1}
    aggregate = {"job_id":s.dev.AGGREGATE_ID,"kind":"aggregate","state_path":str(root/"state.json"),"output_dir":str(root/"full_dev"),
                 "data":{"roles":refs,"initial_protocol":original_ref,"protocol":s._reference(anchor)}}
    full_path = root/"full-dev-config.json"; s.atomic_json(full_path,aggregate); full_ref = s._reference(full_path); configs[s.dev.AGGREGATE_ID] = aggregate
    summary = {"aggregate_audit_succeeded":True,"conditions":conditions,"terminal_records":records}
    def write_summary(): s.atomic_json(Path(aggregate["output_dir"])/"summary.json",summary)
    write_summary()
    auxiliary_paths = []
    for key,conf in ((s.bootstrap.JOB_ID,{"procedure":s.bootstrap.PROCEDURE,"data":{"aggregate_config":full_ref}}),
                     (s.choice.AGGREGATE_ID,{"kind":"aggregate","procedure":s.choice.PROCEDURE,"data":{"initial_protocol":original_ref}})):
        conf = {**conf,"job_id":key,"state_path":aggregate["state_path"]}; configs[key] = conf
        path = root/(key+"-config.json"); s.atomic_json(path,conf); auxiliary_paths.append(path)
    jobs = [{"job_id":key,"cwd":str(REPO)} for key in configs]
    state = {"jobs":{key:{"status":"SUCCEEDED"} for key in configs}}
    monkeypatch.setattr(s.dev.caps,"_active",lambda path:({"campaign_id":root.name,"jobs":jobs},state))
    monkeypatch.setattr(s.dev,"verify_result",lambda *args,**kw:{})
    monkeypatch.setattr(s.dev,"_rows",lambda original,axis:copy.deepcopy(selected[axis]))
    def audit(job,record,module):
        s._require(record["status"] == "SUCCEEDED","SELECTION_NATIVE_SUCCESS_REQUIRED")
        key = job["job_id"]; conf = configs[key]
        path = full_path if key == s.dev.AGGREGATE_ID else auxiliary_paths[0] if key == s.bootstrap.JOB_ID else auxiliary_paths[1] if key == s.choice.AGGREGATE_ID else root/(key+".json")
        receipt = {} if key == s.dev.AGGREGATE_ID else {"technical_complete":conditions[key]["technical_complete"]} if key in conditions else {"bootstrap_auxiliary_completed":True,"scope":s.choice.SCOPES["aggregate"],"all_roles_technical_complete":False,"arbitrary_nonranking_auxiliary_score":999.}
        return copy.deepcopy(conf),receipt,{"config":s._reference(path)}
    monkeypatch.setattr(s,"_audit_job",audit)
    return (full_path,*auxiliary_paths),summary,configs,state,write_summary


def test_real_collector_base_failure_does_not_exclude_tuned_and_auxiliary_scores_do_not_rank(collector):
    paths,summary,_,_,write = collector
    summary["conditions"]["v6-full-dev-T1-base"]["technical_complete"] = False
    summary["conditions"]["v6-full-dev-T1-base"]["strata"]["71875_mcq"]["technical_failure_count"] = 6
    write(); data,_ = s._collect(*paths)
    assert next(c for c in data["candidates"] if c["cell_id"] == "T1")["eligible"] is True
    assert s.select_candidates(data["candidates"])["additional_seed_count"] == 8


def test_real_collector_excludes_incomplete_tuned_and_retains_failed_candidate(collector):
    paths,summary,_,_,write = collector
    summary["conditions"]["v6-full-dev-T2-tuned"]["technical_complete"] = False
    summary["conditions"]["v6-full-dev-T2-tuned"]["strata"]["71875_mcq"]["technical_failure_count"] = 1
    summary["conditions"]["v6-full-dev-T3-tuned"]["status"] = "FAILED"
    summary["conditions"]["v6-full-dev-T3-tuned"]["technical_complete"] = False
    write(); data,_ = s._collect(*paths); by_id = {c["cell_id"]:c for c in data["candidates"]}
    assert by_id["T2"]["eligible"] is False and by_id["T2"]["exclusion_reason"] == "TUNED_TECHNICALLY_INCOMPLETE"
    assert by_id["T3"]["eligible"] is False and by_id["T3"]["exclusion_reason"] == "TUNED_NATIVE_ROLE_UNAVAILABLE"
    assert len(data["candidates"]) == 10 and s.select_candidates(data["candidates"])["family_representatives"]["text"]["gemma"] == "T4"


@pytest.mark.parametrize("tamper",["denominator","ids","auxiliary_pending"])
def test_real_collector_rejects_validation_changes_or_missing_native_auxiliary_receipt(collector,tamper):
    paths,summary,configs,state,write = collector
    if tamper == "denominator":
        summary["conditions"]["v6-full-dev-T1-tuned"]["strata"]["71875_mcq"]["scheduled"] += 1; write()
    elif tamper == "ids":
        config = configs["v6-full-dev-T1-tuned"]; config["data"]["ordered_sample_ids"] = list(reversed(config["data"]["ordered_sample_ids"]))
        s.atomic_json(Path(paths[0]).parent/"v6-full-dev-T1-tuned.json",config)
        # Rebind the parent list to exercise semantic ID equality after raw hashes agree.
        aggregate = s._read(paths[0]); aggregate["data"]["roles"][1] = s._reference(Path(paths[0]).parent/"v6-full-dev-T1-tuned.json")
        s.atomic_json(paths[0],aggregate); configs[s.dev.AGGREGATE_ID] = aggregate
    else: state["jobs"][s.choice.AGGREGATE_ID]["status"] = "RUNNING"
    with pytest.raises(ValueError,match="DENOMINATORS_CHANGED|COMMON_VALIDATION_CHANGED|NATIVE_SUCCESS_REQUIRED"): s._collect(*paths)
