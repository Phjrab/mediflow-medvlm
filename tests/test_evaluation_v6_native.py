"""Native evaluation loader wiring and Base/Tuned equivalence; CPU mocks only."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from mediflow_medvlm.evaluation import v6_native as e


@pytest.fixture
def native_fixture(monkeypatch):
    import sys
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    bitsandbytes = SimpleNamespace(nn=SimpleNamespace())
    calls = []
    class Linear4bit: pass
    class FixtureNative:
        config = SimpleNamespace(use_cache=True)
        is_loaded_in_4bit = True
        def __init__(self): self.parameter = torch.nn.Parameter(torch.zeros(1))
        @classmethod
        def from_pretrained(cls,*args,**kwargs):
            calls.append(("load",args,kwargs))
            return cls()
        def requires_grad_(self,value):
            self.parameter.requires_grad_(value)
            return self
        def named_modules(self):return [("model.layers.0.q_proj",Linear4bit())]
        def named_parameters(self):return [("model.layers.0.q_proj.weight",self.parameter)]
        def eval(self):calls.append(("eval",));return self
    class Processor:pass
    architecture=SimpleNamespace(model_type="qwen3")
    audit={"architecture":"FixtureNative","excluded_roots":[],
           "exact_target_modules":["model.layers.0.q_proj"],"parameter_counts":{"total":1}}
    storage={"trainable_parameters":0,"adapter_parameters":0,
             "base_dtypes":[{"path":"model.layers.0.q_proj.weight","dtype":"torch.float32","packed_shape":[1]}]}
    expected={"architecture":audit,"template":{"date":"2026-11-14"},"thinking":{"disabled":True},
        "loader":"FixtureNative","revision":"a"*40,"snapshot_path":"/private/exact-model",
        "processor_class":"Processor","quantization":{"method":"nf4","double_quant":True,
        "compute_dtype":"bfloat16","excluded_native_roots":[]},"parameter_storage":storage.copy()}
    model_config={"revision":"a"*40,"snapshot_path":"/private/exact-model","cache_dir":"/private/hub"}
    monkeypatch.setitem(sys.modules,"transformers",SimpleNamespace(
        FixtureNative=FixtureNative, AutoConfig=SimpleNamespace(from_pretrained=lambda *a,**k:architecture),
        BitsAndBytesConfig=lambda **kwargs:kwargs))
    monkeypatch.setitem(sys.modules,"bitsandbytes",bitsandbytes)
    monkeypatch.setattr(e.native,"_pinned_files",lambda config:{"all_prepared_files_rehashed":True})
    monkeypatch.setattr(e,"audit_native_architecture",lambda architecture,cell:audit)
    monkeypatch.setattr(e,"load_processor",lambda *a:Processor())
    monkeypatch.setattr(e,"freeze_native_template",lambda processor,fixed_date:expected["template"])
    monkeypatch.setattr(e,"verify_native_no_thinking",lambda *a:expected["thinking"])
    monkeypatch.setattr(e.native,"_cuda_guard",lambda _:calls.append(("cuda_guard",)))
    monkeypatch.setattr(e.native,"_cuda_residency",lambda model:calls.append(("residency",)))
    monkeypatch.setattr(e.native,"_install_smolvlm_inputs_merger",lambda _:None)
    monkeypatch.setattr(e.native,"parameter_storage_report",lambda *args:storage.copy())
    monkeypatch.setattr(bitsandbytes.nn,"Linear4bit",Linear4bit,raising=False)
    def prepare(model,**kwargs):
        calls.append(("prepare",kwargs));return model
    monkeypatch.setattr(peft,"prepare_model_for_kbit_training",prepare)
    monkeypatch.setattr(peft,"get_peft_model",lambda *a,**k:pytest.fail("Base must not create an adapter"))
    return SimpleNamespace(calls=calls,model_config=model_config,expected=expected,storage=storage,Model=FixtureNative)


def test_base_native_no_adapter_fp32_policy_and_exact_quantization(native_fixture):
    p=native_fixture
    bundle=e.load_evaluation_bundle(p.model_config,"T1",role="base",expected_contract=p.expected)
    assert not hasattr(bundle.model,"peft_config")
    assert all(not parameter.requires_grad for _,parameter in bundle.model.named_parameters())
    assert ("prepare",{"use_gradient_checkpointing":False}) in p.calls
    kwargs=next(call[2] for call in p.calls if call[0]=="load")
    assert kwargs["trust_remote_code"] is False and kwargs["local_files_only"] is True
    assert kwargs["attn_implementation"]=="sdpa"
    assert kwargs["quantization_config"]["bnb_4bit_quant_type"]=="nf4"
    assert kwargs["quantization_config"]["bnb_4bit_use_double_quant"] is True
    assert kwargs["quantization_config"]["llm_int8_skip_modules"]==["lm_head"]
    assert bundle.model.config.use_cache is False
    assert bundle.contract["parameter_storage"]["adapter_parameters"]==0


def test_tuned_reuses_final_adapter_and_processor_and_checks_digest(native_fixture,tmp_path,monkeypatch):
    p=native_fixture
    contract={**p.expected,"parameter_storage":{**p.storage,"adapter_parameters":16}}
    bundle=SimpleNamespace(model=SimpleNamespace(peft_config={"default":True}),contract=contract)
    calls=[]
    def load(*args,**kwargs):calls.append(kwargs);return bundle
    monkeypatch.setattr(e.native,"load_v6_gpu_bundle",load)
    monkeypatch.setattr(e.native,"adapter_digest",lambda _:"final-sha")
    e.load_evaluation_bundle(p.model_config,"T1",role="tuned",expected_contract=p.expected,
        final_directory=tmp_path,expected_adapter_sha256="final-sha")
    assert calls[0]["training"] is False
    assert calls[0]["adapter_path"]==tmp_path/"adapter"
    assert calls[0]["processor_path"]==tmp_path/"processor"
    with pytest.raises(ValueError,match="EVAL_FINAL_ADAPTER_WEIGHTS_CHANGED"):
        e.load_evaluation_bundle(p.model_config,"T1",role="tuned",expected_contract=p.expected,
            final_directory=tmp_path,expected_adapter_sha256="wrong-sha")


def test_base_dtype_comparison_normalizes_only_peft_wrapper():
    a={"base_dtypes":[{"path":"layer.q.base_layer.weight","dtype":"float32","packed_shape":[2]}]}
    b={"base_dtypes":[{"path":"layer.q.weight","dtype":"float32","packed_shape":[2]}]}
    assert e.normalized_base_dtype_signature(a)==e.normalized_base_dtype_signature(b)
    b["base_dtypes"][0]["dtype"]="bfloat16"
    assert e.normalized_base_dtype_signature(a)!=e.normalized_base_dtype_signature(b)
