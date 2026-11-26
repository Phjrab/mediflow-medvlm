"""CPU synthetic likelihood oracle and immutable auxiliary lifecycle tests."""
import copy
from contextlib import nullcontext
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from mediflow_medvlm import campaign_choice_likelihood as c

REPO = Path(__file__).resolve().parents[1]


def row(sample_id="x", *, task="medical_mcq", source="aihub_71875", options=None):
    return {"sample_id":sample_id, "source_id":source, "split":"validation", "task_type":task,
            "input":{"question":"Which?", "images":[], **({"options":options or {"AA":"one","B":"two"}} if task == "medical_mcq" else {})},
            "supervision":{"option_id":"B", "answer":"yes", "rationale":"private gold rationale"},
            "metadata":{"q_type":"1"}, "provenance":{"private":"not model input"}}


def selected_rows():
    return ([(row("m"+str(i)), "71875_mcq") for i in range(587)]
            +[(row("p"+str(i), task="context_qa", source="pubmedqa_labeled"), "pubmedqa_labeled") for i in range(50)]
            +[(row("u"+str(i), task="medical_qa", source="medquad"), "medquad") for i in range(879)])


class Tensor:
    """Only CPU list operations required by the processor contract."""
    dtype = "torch.int64"
    def __init__(self, value): self.value = value
    @property
    def shape(self):
        return (len(self.value),len(self.value[0])) if self.value and isinstance(self.value[0],list) else (len(self.value),)
    def __getitem__(self, index): return Tensor(self.value[index])
    def tolist(self): return copy.deepcopy(self.value)
    def to(self, device): return self


class Processor:
    eos_token_id = 99999
    all_special_ids = [99999]
    chat_template = "synthetic frozen template"
    def __init__(self, *, boundary=False, key_drift=False, mask_drift=False, special=False, overflow=0, leading=""):
        self.boundary, self.key_drift, self.mask_drift, self.special = boundary,key_drift,mask_drift,special
        self.overflow,self.leading = overflow,leading
        self.calls = []
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is False
        prefix = "u"*self.overflow+messages[0]["content"]+"<|assistant|>"
        return prefix if add_generation_prompt else prefix+self.leading+messages[-1]["content"]+"<END>"
    def __call__(self, *, text, return_tensors, padding, add_special_tokens):
        assert return_tensors == "pt" and padding is False and add_special_tokens is False
        text = text[0]; self.calls.append(text)
        candidate = not text.endswith("<|assistant|>")
        ids = list(map(ord,text))
        boundary = text.index("<|assistant|>")+len("<|assistant|>")
        if candidate and self.boundary: ids[boundary-1] += 1
        if candidate and self.special: ids[-1] = self.eos_token_id
        result = {"input_ids":Tensor([ids]),"attention_mask":Tensor([[1]*len(ids)]), "mm_token_type_ids":Tensor([[0]*len(ids)])}
        if candidate and self.mask_drift: result["mm_token_type_ids"].value[0][0] = 1
        if candidate and self.key_drift: result["token_type_ids"] = Tensor([[0]*len(ids)])
        return result
    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert skip_special_tokens is False and clean_up_tokenization_spaces is False
        return "".join(map(chr,ids))


def bundle(processor): return SimpleNamespace(processor=processor, architecture=SimpleNamespace(model_type="phi3"))


def success(sample, stratum="71875_mcq", *, tie=False):
    result = c._empty(sample, stratum, attempted=True)
    scores = []
    for index,label in enumerate(result["labels"]):
        values = [-.5]*len(label) if tie else [-float(index+1)]*len(label)
        scores.append({"label":label,"token_ids":list(map(ord,label)),"token_count":len(label),
                       "token_log_probabilities":values,"conditional_log_likelihood":math.fsum(values),
                       "prefix_verified":True,"label_decode_verified":True,"eos_included":False})
    selected,ties = c.select_label(scores)
    gold = sample["supervision"]["option_id"] if sample["task_type"] == "medical_mcq" else sample["supervision"]["answer"]
    result.update(status="SUCCEEDED",reason_code=None,label_scores=scores,selected_label=selected,tied_labels=ties,
                  exact_match=float(selected == gold),input_tokens=10,input_metadata={"input_hash":c.canonical_hash(sample["input"])},
                  native_prefix={"assistant_prefix_verified":True})
    return result


def test_import_has_no_torch_and_exact_source_closure():
    files,digest = c.source_identity(REPO)
    assert set(files) == set(c.dev.source_identity(REPO)[0])|{"src/mediflow_medvlm/campaign_choice_likelihood.py"}
    assert c.relevant_source_files() == c.SOURCE_FILES and digest == c.canonical_hash(files)
    result = subprocess.run([sys.executable,"-c","import sys; from mediflow_medvlm import campaign_choice_likelihood; assert 'torch' not in sys.modules"],cwd=REPO,text=True,capture_output=True)
    assert result.returncode == 0,result.stderr


def test_causal_shift_full_vocab_and_all_tokens_reverse_first_token_ranking():
    # Unused initial positions contain NaNs: only causal label positions matter.
    logits = [[math.nan,0.], [math.log(.9),math.log(.1)], [math.log(.01),math.log(.99)], [999.,-999.]]
    a = c.conditional_log_likelihood(logits,[0,0],2)
    b = c.conditional_log_likelihood(logits,[1],2)
    assert a["token_log_probabilities"] == pytest.approx([math.log(.9),math.log(.01)])
    assert a["conditional_log_likelihood"] == pytest.approx(math.log(.009))
    assert b["conditional_log_likelihood"] == pytest.approx(math.log(.1))
    assert c.select_label([{"label":"AA",**a},{"label":"B",**b}]) == ("B",["B"])


def test_full_vocab_normalizer_stability_shared_prefix_and_exact_ties():
    result = c.conditional_log_likelihood([[10000.,10000.,10000.]], [1],1)
    assert result["conditional_log_likelihood"] == pytest.approx(-math.log(3))
    a = c.conditional_log_likelihood([[0.,0.],[0.,0.]], [0,1],1)
    b = c.conditional_log_likelihood([[0.,0.],[0.,0.]], [0,0],1)
    assert a == b
    assert c.select_label([{"label":"second",**a},{"label":"first",**b}]) == ("second",["second","first"])
    assert c.conditional_log_likelihood([[-math.inf,0.]], [1],1)["conditional_log_likelihood"] == 0.


@pytest.mark.parametrize("logits,ids,start,reason", [
    ([[math.nan,0.]],[0],1,"INVALID_LOGIT"), ([[math.inf,0.]],[0],1,"INVALID_LOGIT"),
    ([[-math.inf,-math.inf]],[0],1,"INVALID_LOGIT"), ([[-math.inf,0.]],[0],1,"NONFINITE_TARGET"),
    ([[0.,0.]],[2],1,"OUTSIDE_VOCAB"), ([[0.,0.]],[0,1],1,"LOGIT_SHAPE"),
    ([[0.,0.]],[0],0,"INVALID_SCORE_BOUNDARY"), ([[0.,0.]],[],1,"INVALID_SCORE_BOUNDARY"),
])
def test_bad_distributions_and_boundaries_fail_closed(logits,ids,start,reason):
    with pytest.raises(ValueError,match=reason): c.conditional_log_likelihood(logits,ids,start)


def test_native_exact_prefix_multitoken_labels_all_tensors_and_gold_invariance():
    sample = row(); processor = Processor(leading=" ")
    candidates,metadata = c.encode_candidates(sample,bundle(processor),{"eval_context":4096})
    assert [e["label"] for _,e in candidates] == ["AA","B"]
    assert [e["token_count"] for _,e in candidates] == [2,1]
    assert all(e["native_completed_leading_whitespace"] == " " and e["eos_included"] is False for _,e in candidates)
    assert all(set(encoded) == {"input_ids","attention_mask","mm_token_type_ids"} for encoded,_ in candidates)
    assert "private gold rationale" not in "".join(processor.calls)
    changed = copy.deepcopy(sample); changed["supervision"] = {"option_id":"AA","answer":"private changed","rationale":"changed"}; changed["metadata"]["new"]="private"
    other,meta2 = c.encode_candidates(changed,bundle(Processor(leading=" ")),{"eval_context":4096})
    assert metadata == meta2
    assert [encoded["input_ids"].tolist() for encoded,_ in candidates] == [encoded["input_ids"].tolist() for encoded,_ in other]


@pytest.mark.parametrize("kwargs,reason", [({"boundary":True},"BPE_PROMPT_BOUNDARY"),({"key_drift":True},"BPE_PROMPT_BOUNDARY"),
    ({"mask_drift":True},"NON_TEXT_MODALITY"),({"special":True},"LABEL_CONTINUATION_DECODE"),({"overflow":4096},"CONTEXT_OVERFLOW")])
def test_processor_boundary_special_mask_key_and_context_failures(kwargs,reason):
    with pytest.raises(ValueError,match=reason): c.encode_candidates(row(),bundle(Processor(**kwargs)),{"eval_context":4096})


def test_context_reserves_label_tokens_not_generation_cap():
    sample = row(); baseline = c.encode_candidates(sample,bundle(Processor()),{"eval_context":4096})[1]["input_tokens"]
    extra = 4096-baseline-2
    candidates,metadata = c.encode_candidates(sample,bundle(Processor(overflow=extra)),{"eval_context":4096})
    assert metadata["input_tokens"]+2 == 4096
    assert candidates[0][1]["token_count"] == 2
    with pytest.raises(ValueError,match="CONTEXT_OVERFLOW"):
        c.encode_candidates(sample,bundle(Processor(overflow=extra+1)),{"eval_context":4096})


def test_score_one_teacher_forces_all_candidates_without_generation_or_gold(monkeypatch):
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(inference_mode=nullcontext,
        autocast=lambda **kwargs:nullcontext(),bfloat16="synthetic-bfloat16"))
    calls = []
    class Model:
        device = "synthetic-cpu"
        def __call__(self, **kwargs):
            assert kwargs.pop("use_cache") is False and kwargs.pop("return_dict") is True
            assert set(kwargs) == {"input_ids","attention_mask","mm_token_type_ids"}
            calls.append(kwargs["input_ids"].tolist())
            return SimpleNamespace(logits=[[[0.]*256 for _ in range(kwargs["input_ids"].shape[-1])]])
    processor=Processor(); b=bundle(processor); b.model=Model()
    result=c._score_one(b,row(),"71875_mcq",{"eval_context":4096})
    assert result["status"] == "SUCCEEDED" and result["selected_label"] == "B" and result["exact_match"] == 1.
    assert len(calls) == 2 and [len(s["token_log_probabilities"]) for s in result["label_scores"]] == [2,1]
    assert "private gold rationale" not in "".join(processor.calls)
    b.processor=Processor(boundary=True)
    failed=c._score_one(b,row(),"71875_mcq",{"eval_context":4096})
    assert failed["status"] == "TECHNICAL_FAILURE" and failed["model_fatal"] is False and len(calls) == 2


def test_labels_only_from_input_and_fixed_pubmed_order_no_test_or_vqa():
    assert c.labels_for(row(options={"B":"second","AA":"first"})) == ["B","AA"]
    assert c.labels_for(row(task="context_qa",source="pubmedqa_labeled")) == ["yes","no","maybe"]
    assert c.labels_for(row(task="medical_vqa",source="slake")) is None
    sample = row(); sample["split"]="test"
    with pytest.raises(ValueError,match="VALIDATION_ONLY"): c.labels_for(sample)


def test_all_planned_failure_denominators_and_semantic_tamper():
    selected = selected_rows(); rows = [c._empty(r,s,"SYNTHETIC_FAILURE") for r,s in selected]
    rows[0] = success(selected[0][0])
    metrics = c._metrics(selected,rows)
    assert metrics["classification_scheduled"] == 637 and metrics["technical_failure_count"] == 636 and metrics["unsupported_count"] == 879
    assert metrics["strata"]["71875_mcq"]["classification_scheduled"] == 587
    assert metrics["strata"]["pubmedqa_labeled"]["technical_failure_rate"] == 1.
    rows[0]["label_scores"][0]["token_log_probabilities"][0] -= .1
    with pytest.raises(ValueError,match="TOKEN_SUM_OR_BOUNDARY"): c._metrics(selected,rows)


@pytest.fixture
def auxiliary(tmp_path,monkeypatch):
    root = tmp_path/"campaign"; root.mkdir()
    original = {"campaign_root":str(root)}; original_path = root/"initial.json"; c.atomic_json(original_path,original)
    selected = selected_rows(); configs = {}
    contract = {"architecture":{},"template":{},"thinking":{},"loader":"Synthetic", "revision":"synthetic",
        "snapshot_path":str(root/"weights"),"processor_class":"Processor","quantization":{},
        "parameter_storage":{"base_dtypes":[],"trainable_parameters":0,"adapter_parameters":0},
        "weights_loaded":True,"cuda_residency_verified":True}
    metrics_path=root/"training-metrics.json"; c.atomic_json(metrics_path,{"native_contract":contract})
    for cell in c.TEXT_CELLS:
        for role in ("base","tuned"):
            path = root/f"native-{cell}-{role}.json"
            child = {**c.dev.cell_spec(cell),"job_id":f"v6-full-dev-{cell}-{role}","kind":"role","role":role,
                     "state_path":str(root/"state.json"),"model":{"revision":"synthetic"},"model_hash":f"{cell}-{role}",
                     "resource_policy":c.ResourcePolicy().as_dict(),"gpu_ledger_path":str(root/"gpu.json"),
                     "supervisor_lock_paths":[str(root/"worker.lock"),str(root/"gpu.lock")],
                     "fixed_template_date":"2026-11-14","precision":"nf4_double_quant_bfloat16"}
            c.atomic_json(path,child); configs[str(path)] = child
    def native(path):
        child = configs[str(path)]
        proof = {"config":c._reference(path),"initial_protocol":c._reference(original_path),"generation_technical_complete":False,
                 "aggregate_config":{"path":"synthetic", "sha256":"synthetic"},
                 "trained_cell":{"metrics":c._reference(metrics_path),"adapter_sha256":"synthetic"}}
        return copy.deepcopy(child),proof,copy.deepcopy(original),copy.deepcopy(selected),{}
    monkeypatch.setattr(c,"_native_inputs",native)
    guards = []; monkeypatch.setattr(c.dev.caps.initial.first.main_train.smoke,"_inherited_supervisor_fds",lambda config:guards.append(config["job_id"]))
    ledger_calls = []
    class Ledger:
        session_seconds = 2.
        def __init__(self,path,*,policy): ledger_calls.append((path,policy.as_dict()))
        def start(self,*args): ledger_calls.append(args)
        def finish(self): ledger_calls.append("finish")
        def snapshot(self): return {"resource_policy":c.ResourcePolicy().as_dict()}
    monkeypatch.setattr(c,"BudgetLedger",Ledger)
    def execute(config,original,selected,stream):
        rows = [c._empty(row,stratum,"CHOICE_CONTEXT_OVERFLOW",attempted=c.labels_for(row) is not None) for row,stratum in selected]
        for result in rows: stream.write(json.dumps(result)+"\n")
        loaded = {**contract,"evaluation_role":config["role"],"evaluation_adapter_sha256":"synthetic" if config["role"] == "tuned" else None}
        return rows,{"native_contract":loaded,"model_load_succeeded":True,"fatal_reason":None}
    monkeypatch.setattr(c,"_execute",execute)
    return root,configs,selected,guards,ledger_calls


def role_config(auxiliary,cell="T1",role="base"):
    root = auxiliary[0]
    return c.prepare_config(root/f"native-{cell}-{role}.json",root/f"auxiliary/choice_likelihood/{cell}-{role}",repo=REPO)


def test_config_namespace_immutable_fields_and_job_contract(auxiliary):
    root = auxiliary[0]; config = role_config(auxiliary)
    assert c.validate_config(config,repo=REPO) == config
    path = root/"choice.json"; c.atomic_json(path,config); job = c.job_from_config(path,repo=REPO)
    assert job["dependencies"] == ["v6-bootstrap","v6-full_dev","v6-full-dev-T1-base"]
    assert job["uses_gpu"] is True and job["requires_gpu"] is True and job["gpu_lock_mode"] == "wrapper"
    for key,value in (("procedure",{**c.PROCEDURE,"eos_included":True}),("gpu_ledger_path",str(root/"other.json")),("uses_gpu",False)):
        changed = copy.deepcopy(config); changed[key]=value; changed["config_hash"]=c.canonical_hash({k:v for k,v in changed.items() if k!="config_hash"})
        with pytest.raises(ValueError,match="FIXED_PROCEDURE_OR_NATIVE"): c.validate_config(changed,repo=REPO)
    with pytest.raises(ValueError,match="DEDICATED_OUTPUT"):
        c.prepare_config(root/"native-T1-base.json",root/"full_dev/T1/base",repo=REPO)


def test_role_lifecycle_separate_failures_ledger_idempotence_and_artifact_semantics(auxiliary):
    root,_,_,guards,ledger_calls = auxiliary; config = role_config(auxiliary)
    path = root/"choice.json"; c.atomic_json(path,config)
    result = c.run_job(path,config["success_receipt"])
    assert result["status"] == "SUCCEEDED" and result["technical_complete"] is False
    assert result["classification_scheduled"] == 637 and all(result[k] is v for k,v in c.FLAGS.items())
    assert guards == [config["job_id"]]
    assert ledger_calls == [(config["gpu_ledger_path"],config["resource_policy"]),(config["model_id"],config["run_id"],"native_choice_likelihood"),"finish"]
    assert c.verify_result(config,repo=REPO) == result
    assert c.run_job(path,config["success_receipt"]) == result
    assert len(ledger_calls) == 3
    output = Path(config["output_dir"]); metrics = c._read(output/"metrics.json"); metrics["clinical_quality_certified"]=True; c.atomic_json(output/"metrics.json",metrics)
    marker = c._read(output/"completion.json"); marker["artifacts"]=[c._reference(output/name) for name in ("config.json","metrics.json","scores.private.jsonl")]; c.atomic_json(output/"completion.json",marker)
    with pytest.raises(ValueError,match="METRICS_CHANGED"): c.verify_result(config,repo=REPO)


def test_aggregate_config_exact_twelve_cpu_job_and_missing_roles_full_denominator(auxiliary,monkeypatch):
    root = auxiliary[0]; refs = []; jobs = []; records = {}
    for cell in c.TEXT_CELLS:
        for role in ("base","tuned"):
            config = role_config(auxiliary,cell,role); path = root/(config["job_id"]+".json"); c.atomic_json(path,config); refs.append(path)
            jobs.append({"job_id":config["job_id"],"cwd":str(REPO),"config":str(path)})
            records[config["job_id"]] = {"status":"FAILED","error":"synthetic","attempts":1,"failure_count":1,"recovery_attempts":0}
    config = c.prepare_aggregate_config(refs,root/"auxiliary/choice_likelihood/aggregate",repo=REPO)
    path = root/"aggregate.json"; c.atomic_json(path,config); job = c.job_from_config(path,repo=REPO)
    assert job["uses_gpu"] is False and job["dependencies"] == ["v6-bootstrap","v6-full_dev",*c.ROLE_IDS] and job["dependency_policy"] == "terminal"
    monkeypatch.setattr(c.dev.caps,"_active",lambda path:({"jobs":jobs},{"jobs":records}))
    monkeypatch.setattr(c.dev,"_rows",lambda original,axis:copy.deepcopy(auxiliary[2]))
    monkeypatch.setattr(c.dev.caps,"_child_config",lambda job,module:(c._read(job["config"]),c._reference(job["config"])))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","")
    result = c.run_job(path,config["success_receipt"])
    assert result["all_roles_succeeded"] is False and result["all_roles_technical_complete"] is False
    summary = c._read(Path(config["output_dir"])/"summary.json")
    assert len(summary["conditions"]) == 12
    assert summary["conditions"][c.ROLE_IDS[0]]["strata"]["71875_mcq"]["technical_failures"] == 587
    assert summary["paired_transitions"]["T1"]["pubmedqa_labeled"]["classification_scheduled"] == 50
    assert c.verify_result(config,repo=REPO) == result


@pytest.mark.parametrize("failure",["CHOICE_MODEL_LOAD_OutOfMemoryError","CHOICE_FORWARD_OutOfMemoryError"])
def test_fatal_attempt_keeps_all_rows_accounting_but_never_success(auxiliary,monkeypatch,failure):
    root=auxiliary[0]; config=role_config(auxiliary); path=root/"choice.json"; c.atomic_json(path,config)
    def execute(config,original,selected,stream):
        rows=[c._empty(row,stratum,failure) for row,stratum in selected]
        for result in rows: stream.write(json.dumps(result)+"\n")
        stream.flush()
        loaded = not failure.startswith("CHOICE_MODEL_LOAD_")
        contract = c._ref(config["data"]["native"]["trained_cell"]["metrics"])["native_contract"]
        contract.update(evaluation_role=config["role"],evaluation_adapter_sha256=None)
        return rows,{"native_contract":contract if loaded else None,"model_load_succeeded":loaded,"fatal_reason":failure}
    monkeypatch.setattr(c,"_execute",execute)
    with pytest.raises(c.dev.caps.initial.CalibrationFatal,match="OutOfMemoryError"):
        c.run_job(path,config["success_receipt"])
    assert not Path(config["output_dir"]).exists() and not Path(config["success_receipt"]).exists()
    attempts=list(Path(config["output_dir"]).parent.glob(Path(config["output_dir"]).name+".attempt-*"))
    assert len(attempts) == 1
    assert len((attempts[0]/"scores.private.jsonl").read_text().splitlines()) == 1516
    assert c._read(attempts[0]/"metrics.json")["classification_scheduled"] == 637
    assert c._read(attempts[0]/"metrics.json")["gpu_accounting"]["session_seconds"] == 2.
    assert c._read(attempts[0]/"failure.private.json")["status"] == "FAILED"
    assert auxiliary[4][-1] == "finish"


def test_fresh_process_guard_before_model_or_ledger(auxiliary,monkeypatch):
    root = auxiliary[0]; config = role_config(auxiliary); path=root/"choice.json"; c.atomic_json(path,config)
    monkeypatch.setitem(sys.modules,"torch",object())
    with pytest.raises(ValueError,match="FRESH_PROCESS"): c.run_job(path,config["success_receipt"])
    assert auxiliary[3] == [] and auxiliary[4] == []


@pytest.mark.parametrize("controlled",[True,False])
def test_cli_reports_controlled_fatal_reason_without_third_party_error_text(tmp_path,monkeypatch,capsys,controlled):
    path=tmp_path/"config.json"; c.atomic_json(path,{"kind":"role"})
    def failure(*args):
        if controlled: raise c.dev.caps.initial.CalibrationFatal("CHOICE_MODEL_LOAD_OutOfMemoryError")
        raise RuntimeError("third party private detail")
    monkeypatch.setattr(c,"run_job",failure)
    assert c.main(["--config",str(path),"--receipt",str(tmp_path/"receipt.json")]) == 1
    output=json.loads(capsys.readouterr().out)
    assert output["status"] == "FAILED"
    assert output.get("reason_code") == ("CHOICE_MODEL_LOAD_OutOfMemoryError" if controlled else None)
