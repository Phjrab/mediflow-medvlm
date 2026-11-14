"""CPU contracts for the opt-in worker; mocks never establish GPU capability."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from mediflow_medvlm import campaign_gpu_smoke as worker
from mediflow_medvlm.campaign_supervisor import verify_success


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.fixture
def candidate(tmp_path, monkeypatch):
    cpu = worker.cpu
    model_id, revision = "Qwen/Qwen3-1.7B", "a" * 40
    cache = tmp_path / "models/hub"
    snapshot = cache / "models--Qwen--Qwen3-1.7B/snapshots" / revision
    asset = write(snapshot / "config.json", {"model_type": "qwen3"})
    files = {"config.json": {"sha256": worker.file_hash(asset), "bytes": asset.stat().st_size}}
    paths = [write(tmp_path / "preparation/pin.json", {"model_id": model_id, "revision": revision}),
             write(tmp_path / "preparation/native_assets.json", {"model_id": model_id, "revision": revision,
                   "snapshot_path": str(snapshot), "cache_dir": str(cache)}),
             write(tmp_path / "preparation/provenance.json", {"model_id": model_id, "revision": revision, "files": files})]
    prepared = write(tmp_path / "prepared.json", {"status": "SUCCEEDED", "scope": "MODEL_PREPARATION_ONLY",
        "model_id": model_id, "model_revision": revision, "model_files_hash": worker.canonical_hash(files),
        "artifacts": [{"path": str(p), "sha256": worker.file_hash(p)} for p in paths]})
    rows = [{"sample_id": f"train-{i}", "split": "train", "source_id": "engineering_fixture", "language": "en",
             "task_type": "medical_qa", "input": {"question": "private question", "images": []},
             "supervision": {"answer": "private answer"}} for i in range(324)]
    train = tmp_path / "train.jsonl"
    train.write_text("".join(json.dumps(row) + "\n" for row in rows))
    preconfig = cpu.build_config("T1", train, prepared, tmp_path / "v6_v7/preflight/T1", expected_count=len(rows))
    @contextmanager
    def cpu_only():
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
        yield SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False))
    monkeypatch.setattr(cpu, "_cpu_only", cpu_only)
    monkeypatch.setattr(cpu, "_load_native", lambda c: (None, None, {"architecture": {}, "template": {}, "thinking": {}}))
    monkeypatch.setattr(cpu, "_fixtures", lambda *args: {})
    def encode(row, *args):
        if row["sample_id"] == "train-3":
            raise ValueError("CONTEXT_OVERFLOW")
        return {}, {"sequence_tokens": 3, "supervised_tokens_after_shift": 2, "full_target_preserved": True}
    monkeypatch.setattr(cpu, "_checked_encode", encode)
    prepath = write(tmp_path / "cpu-config.json", preconfig)
    prereceipt = tmp_path / "v6_v7/receipts/preflight-T1.json"
    cpu.run_preflight(prepath, prereceipt)
    config = worker.build_config("T1", prereceipt, tmp_path / "v6_v7/gpu_smoke/T1")
    config_path = write(tmp_path / "config.json", config)
    return SimpleNamespace(root=tmp_path, config=config, config_path=config_path, preconfig=preconfig,
        prereceipt=prereceipt, rows=rows, train=train, receipt=tmp_path / "v6_v7/receipts/smoke-T1.json", asset=asset)


def mock_metrics():
    return {"optimizer_steps": 20, "samples_processed": 320, "excluded_count": 0, "adapter_changed": True,
            "training_pid": os.getpid(), "adapter_sha256": "f" * 64,
            "native_contract": {"parameter_storage": {"base_dtype_signature": "dtypes"}}}


@pytest.fixture
def gpu_seams(monkeypatch):
    def train(config, rows, staging):
        write(staging / "adapter/adapter_config.json", {"revision": config["model"]["revision"]})
        (staging / "adapter/adapter_model.safetensors").write_bytes(b"PRIVATE_FAKE_ADAPTER_CPU_TEST")
        write(staging / "processor/tokenizer_config.json", {"chat_template": "frozen"})
        return mock_metrics()
    monkeypatch.setattr(worker, "_train", train)
    monkeypatch.setattr(worker, "_inherited_supervisor_fds", lambda config: ())
    def invoke(args, **kwargs):
        config = json.loads(Path(args[args.index("--config") + 1]).read_text())
        if "--train-checkpoint" in args:
            staging = Path(args[args.index("--train-checkpoint") + 1])
            metrics = worker._train(config, {}, staging)
            metrics["training_pid"] = os.getpid() + 200000
            write(staging / "train.metrics.json", metrics)
            value = {"status": "SUCCEEDED", "scope": "NATIVE_GPU_SMOKE_TRAIN_ONLY",
                **{k: config[k] for k in worker.IDENTITY_KEYS}, "training_pid": metrics["training_pid"],
                "optimizer_steps": 20, "samples_processed": 320, "retry_start": "original_base",
                "metrics_sha256": worker.file_hash(staging / "train.metrics.json")}
            write(Path(args[args.index("--receipt") + 1]), value)
            return SimpleNamespace(returncode=0)
        value = {"status": "SUCCEEDED", **{k: config[k] for k in worker.IDENTITY_KEYS}, "pid": os.getpid() + 100000,
                 "adapter_sha256": "f" * 64, "held_train_sample_id": config["data"]["selection"]["held_train_record"]["sample_id"],
                 "finite_forward": True, "adapter_weights_verified": True, "fresh_process_required": True,
                 "dev_test_read": False, "runtime_versions": config["runtime_versions"], "held_train_loss": 1.0,
                 "target_tokens": 2, "generated_tokens": 1, "base_dtype_signature": "dtypes"}
        write(Path(args[args.index("--receipt") + 1]), value)
        return SimpleNamespace(returncode=0)
    return invoke


def test_build_is_train_only_deterministic_and_no_gpu_import(candidate):
    c = candidate
    selected = c.config["data"]["selection"]
    ids = [r["sample_id"] for r in selected["train_records"]]
    assert len(ids) == len(set(ids)) == 320
    assert "train-3" not in ids and selected["held_train_record"]["sample_id"] not in ids
    assert c.config["job_id"] == c.config["run_id"] == "v6-native-smoke-T1"
    assert c.config["recipe"]["betas"] == [.9, .999] and c.config["recipe"]["eps"] == 1e-8
    assert c.config["recipe"]["loss"] == "mean_example_target_token_mean"
    assert all(v is None for k, v in c.config["resource_policy"].items() if k not in {"schema_version", "budget_mode"})
    assert c.config == worker.build_config("T1", c.prereceipt, c.config["output_dir"])
    assert "private question" not in json.dumps(c.config) and "private answer" not in json.dumps(c.config)


def test_complete_receipt_artifacts_reuse_and_accounting(candidate, gpu_seams, monkeypatch):
    c = candidate
    result = worker.run_smoke(c.config_path, c.receipt, invoke=gpu_seams)
    assert result["optimizer_steps"] == 20 and result["samples_processed"] == 320
    assert result["fresh_process_reload_verified"] and result["excluded_count"] == 0
    assert result["usable_as_main_initialization"] is False
    assert result["v6_main_training_completed"] is False and result["campaign_terminal"] is False
    assert verify_success({**c.config, "success_receipt": str(c.receipt)})["artifact_count"] >= 8
    ledger = json.loads(Path(c.config["gpu_ledger_path"]).read_text())
    assert [s["phase"] for s in ledger["sessions"]] == ["native_smoke_train", "native_smoke_reload"]
    assert ledger["active"] is None
    assert ledger["current_policy"]["budget_mode"] == "unlimited"
    for path in Path(c.config["output_dir"]).rglob("*"):
        assert path.stat().st_mode & 0o077 == 0
    monkeypatch.setattr(worker, "_train", lambda *args: pytest.fail("completed output must not retrain"))
    assert worker.run_smoke(c.config_path, c.receipt) == result
    assert len(json.loads(Path(c.config["gpu_ledger_path"]).read_text())["sessions"]) == 2


def test_atomic_completion_recovers_rename_receipt_crash(candidate, gpu_seams, monkeypatch):
    original = worker.atomic_json
    def crash(path, value):
        if Path(path) == candidate.receipt:
            raise OSError("synthetic receipt gap")
        return original(path, value)
    monkeypatch.setattr(worker, "atomic_json", crash)
    with pytest.raises(OSError):
        worker.run_smoke(candidate.config_path, candidate.receipt, invoke=gpu_seams)
    marker = Path(candidate.config["output_dir"]) / "completion.json"
    digest = worker.file_hash(marker)
    monkeypatch.setattr(worker, "atomic_json", original)
    monkeypatch.setattr(worker, "_train", lambda *args: pytest.fail("must reuse completion"))
    assert worker.run_smoke(candidate.config_path, candidate.receipt)["status"] == "SUCCEEDED"
    assert worker.file_hash(marker) == digest


@pytest.mark.parametrize("change", ["source", "runtime", "train", "preflight", "model", "supervisor", "recipe", "context", "selection"])
def test_identity_failures_before_gpu(candidate, monkeypatch, change):
    c = candidate
    if change == "source": monkeypatch.setattr(worker, "source_identity", lambda repo=None: ({}, "bad"))
    elif change == "runtime": monkeypatch.setattr(worker, "runtime_versions", lambda: {})
    elif change == "train": c.train.write_text(c.train.read_text() + "\n")
    elif change == "preflight": c.prereceipt.write_text(c.prereceipt.read_text() + "\n")
    elif change == "model": c.asset.write_text("changed")
    elif change == "supervisor": monkeypatch.setenv("MEDVLM_SUPERVISOR_RUN_ID", "wrong")
    else:
        if change == "recipe": c.config["recipe"]["optimizer_steps"] = 19
        elif change == "context": c.config["max_length"] = 4096
        else: c.config["data"]["selection"]["train_records"].reverse()
        c.config["config_hash"] = worker.canonical_hash({k: v for k, v in c.config.items() if k != "config_hash"})
        write(c.config_path, c.config)
    monkeypatch.setattr(worker, "_train", lambda *a: pytest.fail("must fail before GPU"))
    with pytest.raises((worker.SmokeError, worker.cpu.PreflightError)):
        worker.run_smoke(c.config_path, c.receipt)
    assert not Path(c.config["output_dir"]).exists() and not c.receipt.exists()


def test_selected_encoding_cannot_drop_or_change_images_or_target(candidate, monkeypatch):
    record = candidate.config["data"]["selection"]["train_records"][0]
    row = next(r for r in candidate.rows if r["sample_id"] == record["sample_id"])
    bundle = SimpleNamespace(processor=None, architecture=None)
    monkeypatch.setattr(worker.cpu, "_checked_encode", lambda *args: ({}, {**record["native"], "images": [{"sha256": "changed"}]}))
    with pytest.raises(worker.SmokeError, match="ENCODING_CHANGED"):
        worker._encode(row, record, bundle, candidate.config)
    def fail(*args): raise ValueError("private native target error")
    monkeypatch.setattr(worker.cpu, "_checked_encode", fail)
    with pytest.raises(worker.SmokeError, match="NO_DROP_ALLOWED"):
        worker._encode(row, record, bundle, candidate.config)


def test_failed_attempt_preserved_same_identity_retry_starts_fresh(candidate, gpu_seams, monkeypatch):
    good = worker._train
    def fail(config, rows, staging):
        write(staging / "partial.json", {"step": 7})
        raise RuntimeError("private example")
    monkeypatch.setattr(worker, "_train", fail)
    with pytest.raises(RuntimeError):
        worker.run_smoke(candidate.config_path, candidate.receipt, invoke=gpu_seams)
    attempts = list(Path(candidate.config["output_dir"]).parent.glob("T1.attempt-*"))
    assert len(attempts) == 1 and (attempts[0] / "partial.json").exists()
    assert "private example" not in (attempts[0] / "failure.json").read_text()
    assert not candidate.receipt.exists()
    monkeypatch.setattr(worker, "_train", good)
    assert worker.run_smoke(candidate.config_path, candidate.receipt, invoke=gpu_seams)["status"] == "SUCCEEDED"
    assert (attempts[0] / "partial.json").exists()


@pytest.mark.parametrize("failure", ["exit", "same_pid", "weights", "nonfinite", "no_generation", "dtype"])
def test_fresh_reload_proof_failures(candidate, gpu_seams, failure):
    def invoke(args, **kwargs):
        result = gpu_seams(args, **kwargs)
        if "--train-checkpoint" in args: return result
        if failure == "exit": return SimpleNamespace(returncode=1)
        path = Path(args[args.index("--receipt") + 1])
        value = json.loads(path.read_text())
        if failure == "same_pid": value["pid"] = os.getpid()
        elif failure == "weights": value["adapter_sha256"] = "wrong"
        elif failure == "nonfinite": value["held_train_loss"] = float("nan")
        elif failure == "no_generation": value["generated_tokens"] = 0
        else: value["base_dtype_signature"] = "wrong"
        write(path, value)
        return result
    with pytest.raises(worker.SmokeError, match="RELOAD"):
        worker.run_smoke(candidate.config_path, candidate.receipt, invoke=invoke)
    assert not candidate.receipt.exists() and not Path(candidate.config["output_dir"]).exists()
    assert list(Path(candidate.config["output_dir"]).parent.glob("T1.attempt-*/failure.json"))


@pytest.mark.parametrize("artifact", ["metrics.json", "adapter/adapter_model.safetensors", "processor/tokenizer_config.json"])
def test_completed_artifact_tamper_rejected(candidate, gpu_seams, artifact):
    worker.run_smoke(candidate.config_path, candidate.receipt, invoke=gpu_seams)
    path = Path(candidate.config["output_dir"]) / artifact
    path.write_text("tamper")
    with pytest.raises(worker.SmokeError, match="ARTIFACT_CHANGED"):
        worker.run_smoke(candidate.config_path, candidate.receipt, invoke=gpu_seams)


def test_no_public_raw_errors(candidate, monkeypatch, capsys):
    monkeypatch.setattr(worker, "run_smoke", lambda *a: (_ for _ in ()).throw(ValueError("secret private answer")))
    assert worker.main(["--config", str(candidate.config_path), "--receipt", str(candidate.receipt)]) == 1
    text = capsys.readouterr().err
    assert "secret" not in text and "private answer" not in text
    assert json.loads(text)["exception_type"] == "ValueError"


def test_target_loss_ignores_prompt_and_means_target_tokens():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[[90., -90., 0.], [.2, .3, .4], [.7, .1, .2], [0., 0., 0.]]], requires_grad=True)
    model = SimpleNamespace(device=torch.device("cpu"))
    class Model:
        device = model.device
        def __call__(self, **kwargs):
            assert "labels" not in kwargs
            return SimpleNamespace(logits=logits)
    batch = {"input_ids": torch.tensor([[0, 0, 1, 2]]), "attention_mask": torch.ones(1, 4),
             "labels": torch.tensor([[-100, -100, 1, 2]])}
    actual, count = worker._target_loss(Model(), batch)
    expected = torch.nn.functional.cross_entropy(logits[0, 1:3], torch.tensor([1, 2]))
    assert count == 2 and torch.equal(actual, expected)
    actual.backward()
    assert not bool(logits.grad[0, 0].any())


def test_train_loop_twenty_updates_320_examples_and_each_adapter_changes(candidate, monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    from mediflow_medvlm.training import v6_gpu as native
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.A = torch.nn.Parameter(torch.randn(3, 2) * .1)
            self.B = torch.nn.Parameter(torch.zeros(2, 3))
            self.device = torch.device("cpu")
        def forward(self, input_ids, attention_mask):
            return SimpleNamespace(logits=(self.A @ self.B).unsqueeze(0))
        def save_pretrained(self, directory, **kwargs):
            write(directory / "adapter_config.json", {})
            (directory / "adapter_model.safetensors").write_bytes(b"tiny_cpu_test")
    tiny = Tiny()
    bundle = SimpleNamespace(model=tiny, processor=SimpleNamespace(save_pretrained=lambda p: write(p / "tokenizer.json", {})),
        architecture=None, contract={"architecture": {}})
    monkeypatch.setattr(native, "load_v6_gpu_bundle", lambda *a, **k: bundle)
    monkeypatch.setattr(native, "assert_frozen_base", lambda m: None)
    def digests(m, audit=None):
        return {"projection": {k: worker.canonical_hash(getattr(m, k).detach().tolist()) for k in ("A", "B")}}
    monkeypatch.setattr(native, "adapter_digest", lambda m: worker.canonical_hash(digests(m)))
    monkeypatch.setattr(native, "projection_adapter_digests", digests)
    def gradients(m, audit, **kwargs):
        tensors = {k: {"finite": bool(torch.isfinite(getattr(m, k).grad).all()),
                        "nonzero": bool(getattr(m, k).grad.any())} for k in ("A", "B")}
        assert all(t["finite"] for t in tensors.values())
        return {"all_finite": True, "projections": [{"path": "projection", "tensors": tensors}]}
    monkeypatch.setattr(native, "projection_gradient_report", gradients)
    monkeypatch.setattr(worker, "_inherited_supervisor_fds", lambda c: ())
    monkeypatch.setattr(worker, "_verify_native_contract", lambda *a: None)
    seen = []
    def encode(row, record, bundle, config):
        seen.append(row["sample_id"])
        return {"input_ids": torch.tensor([[0, 1, 2]]), "attention_mask": torch.ones(1, 3),
                "labels": torch.tensor([[-100, 1, 2]])}, record["native"]
    monkeypatch.setattr(worker, "_encode", encode)
    for name in ("manual_seed_all", "reset_peak_memory_stats", "synchronize", "empty_cache"):
        monkeypatch.setattr(torch.cuda, name, lambda *a: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: SimpleNamespace(name="MOCK_CPU", total_memory=0))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (0, 0))
    monkeypatch.setattr(torch.backends.cudnn, "version", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)
    monkeypatch.setattr(torch, "set_num_threads", lambda *a: None)
    # Explicitly patch CUDA graph guard; optimizer otherwise asks the mocked
    # availability API about the real driver even though every tensor is CPU.
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.optim.Optimizer, "_accelerator_graph_capture_health_check", lambda self: None)
    rows = {r["sample_id"]: r for r in candidate.rows}
    result = worker._train(candidate.config, rows, tmp_path)
    assert len(seen) == len(set(seen)) == 320
    assert result["optimizer_steps"] == 20 and result["samples_processed"] == 320
    assert result["supervised_tokens"] == 640 and result["final_learning_rate"] == 0
    assert result["learning_rates"][0] == 2e-4 and result["learning_rates"][-1] == pytest.approx(1e-5)
    assert result["gradient_reports"][0]["projections"][0]["tensors"]["A"]["nonzero"] is False
    assert all(result["nonzero_gradients_observed"]["projection"].values())
    assert result["all_projection_tensors_changed"] is True


def test_inherited_leases_stay_locked_through_real_child(candidate, monkeypatch):
    import fcntl
    c = candidate
    for key in worker.IDENTITY_KEYS:
        monkeypatch.setenv("MEDVLM_SUPERVISOR_" + key.upper(), c.config[key])
    locks = []
    try:
        for name in c.config["supervisor_lock_paths"]:
            path = Path(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = path.open("a")
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks.append(stream)
        fds = worker._inherited_supervisor_fds(c.config)
        child = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read(1)"], stdin=subprocess.PIPE, pass_fds=fds)
        for stream in locks: stream.close()
        for name in c.config["supervisor_lock_paths"]:
            with Path(name).open("a") as probe:
                with pytest.raises(BlockingIOError): fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        child.communicate(b"x", timeout=10)
        for name in c.config["supervisor_lock_paths"]:
            with Path(name).open("a") as probe: fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        for stream in locks: stream.close()
        if "child" in locals() and child.poll() is None:
            child.kill()
            child.wait()


def test_open_but_unlocked_descriptors_are_not_supervisor_leases(candidate, monkeypatch):
    for key in worker.IDENTITY_KEYS:
        monkeypatch.setenv("MEDVLM_SUPERVISOR_" + key.upper(), candidate.config[key])
    streams = []
    try:
        for value in candidate.config["supervisor_lock_paths"]:
            Path(value).parent.mkdir(parents=True, exist_ok=True)
            streams.append(Path(value).open("a"))
        with pytest.raises(worker.SmokeError, match="LEASE_NOT_HELD"):
            worker._inherited_supervisor_fds(candidate.config)
    finally:
        for stream in streams: stream.close()


def test_actual_fresh_process_reload_path_with_cpu_model(candidate, gpu_seams):
    """Execute the reload function in a real child, with explicit CPU seams."""
    pytest.importorskip("torch")
    staging = candidate.root / "reload_cpu_fixture"
    staging.mkdir()
    metrics = worker._train(candidate.config, {}, staging)
    write(staging / "metrics.json", metrics)
    worker._checkpoint(candidate.config, staging, metrics)
    program = r'''
from contextlib import nullcontext
import json, sys
from pathlib import Path
from types import SimpleNamespace
import torch
from mediflow_medvlm import campaign_gpu_smoke as w
from mediflow_medvlm.training import v6_gpu as native, rendering
assert not torch.cuda.is_initialized()
w._inherited_supervisor_fds = lambda c: ()
w._verify_native_contract = lambda *a: None
torch.autocast = lambda *a, **k: nullcontext()
class Tiny:
    device = torch.device('cpu')
    def eval(self): pass
    def __call__(self, **kwargs):
        assert 'labels' not in kwargs
        return SimpleNamespace(logits=torch.zeros(1, 3, 3))
    def generate(self, **kwargs):
        assert 'labels' not in kwargs and not kwargs['do_sample']
        return torch.cat([kwargs['input_ids'], torch.tensor([[2]])], dim=-1)
bundle = SimpleNamespace(model=Tiny(), processor=SimpleNamespace(decode=lambda *a, **k: 'ready'),
                         architecture=None, contract={'parameter_storage': {'base_dtype_signature': 'dtypes'}})
def load(*args, **kwargs):
    assert kwargs['training'] is False and Path(kwargs['processor_path']).is_dir()
    return bundle
native.load_v6_gpu_bundle = load
native.adapter_digest = lambda m: 'f' * 64
w._encode = lambda *a: ({'input_ids': torch.tensor([[0,1,2]]),
    'attention_mask': torch.ones(1,3), 'labels': torch.tensor([[-100,1,2]])}, {})
def prompt(*args, **kwargs):
    assert kwargs['include_labels'] is False
    return {'input_ids': torch.tensor([[0,1]]), 'attention_mask': torch.ones(1,2)}, {}
rendering.encode_sample = prompt
w.reload_checkpoint(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
assert not torch.cuda.is_initialized()
'''
    def invoke(args, **kwargs):
        return subprocess.run([sys.executable, "-c", program, str(candidate.config_path), str(staging),
                               args[args.index("--receipt") + 1]], **kwargs)
    result = worker._reload(candidate.config_path, staging, candidate.config, metrics, invoke)
    assert result["pid"] != os.getpid() and result["generated_tokens"] == 1
    assert result["finite_forward"] and result["adapter_weights_verified"]
    generated = json.loads((staging / "reload.generated.private.json").read_text())
    assert generated["generated_ids"] == [2] and generated["generated_text"] == "ready"


@pytest.fixture
def tiny_native_gemma():
    """Actual Transformers/PEFT modules on CPU; no prepared weights or CUDA."""
    torch = pytest.importorskip('torch')
    pytest.importorskip('transformers')
    pytest.importorskip('peft')
    from peft import LoraConfig, get_peft_model
    from transformers import Gemma3Config, Gemma3ForConditionalGeneration
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(215)
    config = Gemma3Config(text_config=dict(vocab_size=97, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=4,
        max_position_embeddings=2048, sliding_window=16, layer_types=['sliding_attention', 'full_attention'],
        pad_token_id=0, bos_token_id=1, eos_token_id=2, use_cache=False),
        vision_config=dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=4, image_size=8, patch_size=2), mm_tokens_per_image=4, image_token_id=96)
    config._attn_implementation = 'sdpa'
    native = Gemma3ForConditionalGeneration(config)
    # Transformers initializes this projector to zero: use synthetic nonzero
    # weights to make vision-path loss/gradient parity an informative test.
    with torch.no_grad():
        native.model.multi_modal_projector.mm_input_projection_weight.normal_(std=.02)
    native.requires_grad_(False)
    native.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    native.enable_input_require_grads()
    targets = [f'model.language_model.layers.{layer}.{part}.{projection}'
        for layer in range(2) for part, projections in [('self_attn', ('q_proj', 'k_proj', 'v_proj', 'o_proj')),
            ('mlp', ('gate_proj', 'up_proj', 'down_proj'))] for projection in projections]
    model = get_peft_model(native, LoraConfig(r=8, lora_alpha=16, lora_dropout=.05,
        target_modules=targets, task_type='CAUSAL_LM', bias='none'))
    with torch.no_grad():
        # Both adapter sides must have nonzero gradients, beyond B's initial zero.
        for name, parameter in model.named_parameters():
            if '.lora_B.' in name: parameter.normal_(std=.02)
    model.train()
    try:
        yield model
    finally:
        torch.set_num_threads(previous_threads)


def gemma_batch(torch, length, *, vision=False, dense=False):
    ids = (torch.arange(length).reshape(1, -1) % 90) + 3
    mask = torch.ones_like(ids)
    labels = ids.clone()
    labels[:, :31 if length == 1123 else 6] = -100
    if not dense:
        labels[:, 7::3] = -100
        mask[:, -1] = 0
        labels[:, -1] = -100
    batch = {'input_ids': ids, 'attention_mask': mask, 'labels': labels,
             'position_ids': torch.arange(length).reshape(1, -1)}
    if vision:
        ids[:, 1:5] = 96
        labels[:, 1:5] = -100
        batch['token_type_ids'] = torch.zeros_like(ids)
        batch['token_type_ids'][:, 1:5] = 1
        batch['pixel_values'] = torch.linspace(-1, 1, 3 * 8 * 8).reshape(1, 3, 8, 8).requires_grad_()
    return batch


def dense_target_reference(model, batch):
    import torch
    output = model(**{k: v for k, v in batch.items() if k != 'labels'})
    shifted = batch['labels'][:, 1:]
    selected = shifted != -100
    return torch.nn.functional.cross_entropy(output.logits[:, :-1, :][selected].float(), shifted[selected])


@pytest.mark.parametrize('vision,length,dense', [(False, 17, False), (False, 1123, True),
    (True, 19, False), (True, 130, True)])
def test_native_gemma_chunk_loss_adapter_input_image_grad_and_rng_parity(tiny_native_gemma, vision, length, dense):
    import torch
    model = tiny_native_gemma
    native = model.get_base_model()
    assert native.is_gradient_checkpointing and native.lm_head.weight.dtype == torch.float32
    batch = gemma_batch(torch, length, vision=vision, dense=dense)
    embedded = []
    def retain_embedding(module, args, result):
        result.retain_grad()
        embedded.append(result)
    hook = native.get_input_embeddings().register_forward_hook(retain_embedding)
    torch.manual_seed(1045)
    expected = dense_target_reference(model, batch)
    (expected / 16).backward()
    expected_rng = torch.get_rng_state()
    expected_grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.requires_grad}
    expected_input_grad = embedded[-1].grad.clone()
    expected_pixel_grad = batch['pixel_values'].grad.clone() if vision else None
    model.zero_grad(set_to_none=True)
    if vision: batch['pixel_values'].grad = None
    embedded.clear()
    torch.manual_seed(1045)
    actual, count = worker._target_loss(model, batch)
    (actual / 16).backward()
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    assert count == int((batch['labels'][:, 1:] != -100).sum())
    if length == 1123: assert count == 1092
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(parameter.grad, expected_grads[name], rtol=5e-5, atol=2e-6)
    torch.testing.assert_close(embedded[-1].grad, expected_input_grad, rtol=5e-5, atol=2e-6)
    if vision:
        torch.testing.assert_close(batch['pixel_values'].grad, expected_pixel_grad, rtol=5e-5, atol=2e-6)
        assert bool(expected_pixel_grad.any())
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert all(parameter.grad is None for parameter in native.lm_head.parameters())
    hook.remove()


def test_gemma_long_dense_targets_bound_head_rows_and_saved_logits(tiny_native_gemma, monkeypatch):
    import torch
    from torch.utils import checkpoint as checkpoint_module
    model = tiny_native_gemma
    head = model.get_base_model().lm_head
    original_linear = torch.nn.functional.linear
    rows = []
    def linear(features, weight, bias=None):
        if weight is head.weight:
            rows.append(features.numel() // features.shape[-1])
        return original_linear(features, weight, bias)
    monkeypatch.setattr(torch.nn.functional, 'linear', linear)
    original_checkpoint = checkpoint_module.checkpoint
    checkpoints = []
    def checkpoint(function, *args, **kwargs):
        assert kwargs['use_reentrant'] is False
        assert kwargs.get('preserve_rng_state', True) is True
        checkpoints.append(args[0].shape[0])
        return original_checkpoint(function, *args, **kwargs)
    monkeypatch.setattr(checkpoint_module, 'checkpoint', checkpoint)
    saved = []
    def pack(tensor):
        saved.append(tuple(tensor.shape))
        return tensor
    batch = gemma_batch(torch, 1123, dense=True)
    backbone_calls = []
    hook = model.get_base_model().model.register_forward_pre_hook(lambda *args: backbone_calls.append(True))
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        loss, count = worker._target_loss(model, batch)
        loss.backward()
    hook.remove()
    assert count == 1092
    assert sum(checkpoints) == count and max(checkpoints) <= worker._GEMMA_TARGET_CHUNK
    assert len(rows) >= 2 * len(checkpoints)  # Frozen head/CE recomputed in backward.
    assert max(rows) <= worker._GEMMA_TARGET_CHUNK
    assert len(backbone_calls) == 1  # No repeated backbone/LoRA-dropout forward per chunk.
    assert not any(len(shape) == 2 and shape[-1] == 97 and shape[0] > worker._GEMMA_TARGET_CHUNK for shape in saved)


def test_gemma_no_grad_chunks_without_checkpoint_or_retained_graph(tiny_native_gemma, monkeypatch):
    import torch
    from torch.utils import checkpoint as checkpoint_module
    model = tiny_native_gemma.eval()
    batch = gemma_batch(torch, 130, vision=True, dense=True)
    def forbidden(*args, **kwargs): pytest.fail('no checkpoint in no_grad reload forward')
    monkeypatch.setattr(checkpoint_module, 'checkpoint', forbidden)
    with torch.no_grad():
        expected = dense_target_reference(model, batch)
        actual, count = worker._target_loss(model, batch)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    assert count == int((batch['labels'][:, 1:] != -100).sum())
    assert not actual.requires_grad and actual.grad_fn is None


@pytest.mark.parametrize('failure', ['empty', 'batch-two', 'trainable-head', 'hooked-head', 'unsupported-class'])
def test_gemma_specialized_loss_fails_closed(tiny_native_gemma, failure):
    import torch
    model = tiny_native_gemma
    batch = gemma_batch(torch, 17)
    if failure == 'empty': batch['labels'].fill_(-100)
    elif failure == 'batch-two': batch = {key: value.repeat(2, 1) for key, value in batch.items()}
    elif failure == 'trainable-head': model.get_base_model().lm_head.weight.requires_grad_(True)
    elif failure == 'hooked-head': model.get_base_model().lm_head.register_forward_hook(lambda *args: None)
    else:
        class Unsupported(type(model.get_base_model())): pass
        model.get_base_model().__class__ = Unsupported
    with pytest.raises(worker.SmokeError): worker._target_loss(model, batch)


def test_non_gemma_target_loss_retains_full_forward_and_empty_rejects():
    torch = pytest.importorskip('torch')
    class Ordinary:
        device = torch.device('cpu')
        config = SimpleNamespace(model_type='qwen3')
        calls = 0
        def __call__(self, **kwargs):
            assert 'labels' not in kwargs
            self.calls += 1
            return SimpleNamespace(logits=torch.tensor([[[.2, .3, .4], [.7, .1, .2], [0., 0., 0.]]], requires_grad=True))
    model = Ordinary()
    batch = {'input_ids': torch.tensor([[0, 1, 2]]), 'labels': torch.tensor([[-100, 1, 2]])}
    loss, count = worker._target_loss(model, batch)
    assert count == 2 and model.calls == 1
    torch.testing.assert_close(loss, dense_target_reference(model, batch), rtol=0, atol=0)
    batch['labels'].fill_(-100)
    with pytest.raises(worker.SmokeError, match='NO_SUPERVISED_TARGET_TOKENS'): worker._target_loss(model, batch)



@pytest.mark.parametrize('vision', [False, True])
def test_native_gemma_checkpoint_recomputation_preserves_bf16_autocast(tiny_native_gemma, vision):
    import torch
    model = tiny_native_gemma
    batch = gemma_batch(torch, 130, vision=vision, dense=True)
    torch.manual_seed(580)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        expected = dense_target_reference(model, batch)
    expected.backward()
    expected_grads = {name: parameter.grad.clone() for name, parameter in model.named_parameters() if parameter.requires_grad}
    model.zero_grad(set_to_none=True)
    if vision: batch['pixel_values'].grad = None
    torch.manual_seed(580)
    mask = batch['labels'][:, 1:] != -100
    with torch.autocast('cpu', dtype=torch.bfloat16):
        actual = worker._gemma_target_loss(model.get_base_model(), batch, mask)
    # Backward deliberately runs outside autocast; checkpoint must restore the
    # original forward's autocast state when it recomputes the frozen head/CE.
    actual.backward()
    assert actual.dtype == expected.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            # BF16 matrix shapes/reduction order can change rounding (eps=.0078).
            torch.testing.assert_close(parameter.grad, expected_grads[name], rtol=2e-2, atol=2e-5)
    assert model.get_base_model().lm_head.weight.dtype == torch.float32
    assert not model.get_base_model().lm_head.weight.requires_grad


def test_chunked_loss_keeps_equal_example_weight_for_different_target_counts(tiny_native_gemma):
    import torch
    model = tiny_native_gemma.eval()
    batches = [gemma_batch(torch, 17), gemma_batch(torch, 130, dense=True)]
    expected = sum(dense_target_reference(model, batch) for batch in batches) / 2
    expected.backward()
    expected_grads = {name: parameter.grad.clone() for name, parameter in model.named_parameters() if parameter.requires_grad}
    model.zero_grad(set_to_none=True)
    losses_counts = [worker._target_loss(model, batch) for batch in batches]
    assert losses_counts[0][1] != losses_counts[1][1]
    actual = sum(loss for loss, count in losses_counts) / 2
    actual.backward()
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(parameter.grad, expected_grads[name], rtol=5e-5, atol=2e-6)



@pytest.mark.parametrize('failure', ['exit', 'same-pid', 'source', 'steps', 'samples', 'restart',
    'metrics-hash', 'metrics-pid', 'missing-receipt', 'missing-metrics'])
def test_train_child_failure_never_starts_reload_and_keeps_one_budget_session(candidate, gpu_seams, failure):
    calls = []
    def invoke(args, **kwargs):
        assert '--train-checkpoint' in args
        calls.append('train')
        result = gpu_seams(args, **kwargs)
        path = Path(args[args.index('--receipt') + 1])
        metrics_path = path.parent / 'train.metrics.json'
        value = json.loads(path.read_text())
        if failure == 'exit': return SimpleNamespace(returncode=1, stderr=b'PRIVATE_TRAIN_FAILURE')
        if failure == 'same-pid': value['training_pid'] = os.getpid()
        elif failure == 'source': value['source_hash'] = 'stale'
        elif failure == 'steps': value['optimizer_steps'] = 19
        elif failure == 'samples': value['samples_processed'] = 319
        elif failure == 'restart': value['retry_start'] = 'previous_adapter'
        elif failure == 'metrics-hash': metrics_path.write_text(metrics_path.read_text() + '\n')
        elif failure == 'metrics-pid':
            metrics = json.loads(metrics_path.read_text()); metrics['training_pid'] += 1
            write(metrics_path, metrics); value['metrics_sha256'] = worker.file_hash(metrics_path)
        elif failure == 'missing-receipt': path.unlink(); return result
        elif failure == 'missing-metrics': metrics_path.unlink(); return result
        write(path, value)
        return result
    with pytest.raises(worker.SmokeError, match='TRAIN_PROCESS'):
        worker.run_smoke(candidate.config_path, candidate.receipt, invoke=invoke)
    assert calls == ['train']
    ledger = json.loads(Path(candidate.config['gpu_ledger_path']).read_text())
    assert ledger['active'] is None
    assert [entry['phase'] for entry in ledger['sessions']] == ['native_smoke_train']
    assert not candidate.receipt.exists() and not Path(candidate.config['output_dir']).exists()
    attempts = list(Path(candidate.config['output_dir']).parent.glob('T1.attempt-*'))
    assert len(attempts) == 1
    assert (attempts[0] / 'failure.json').is_file()
    assert not (attempts[0] / 'reload.json').exists()
    assert (attempts[0] / 'adapter/adapter_model.safetensors').is_file()
    if failure == 'exit':
        assert b'PRIVATE_TRAIN_FAILURE' in (attempts[0] / 'train.stderr.private.log').read_bytes()
        assert 'PRIVATE_TRAIN_FAILURE' not in (attempts[0] / 'failure.json').read_text()


def test_stage_coordinator_rejects_initialized_cuda_before_spawning(candidate, monkeypatch):
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: True))
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    def forbidden(*args, **kwargs): pytest.fail('coordinator CUDA context must fail before child spawn')
    with pytest.raises(worker.SmokeError, match='COORDINATOR_CUDA_CONTEXT_FORBIDDEN'):
        worker._train_process(candidate.config_path, candidate.root, candidate.config, forbidden)
    with pytest.raises(worker.SmokeError, match='COORDINATOR_CUDA_CONTEXT_FORBIDDEN'):
        worker._reload(candidate.config_path, candidate.root, candidate.config, mock_metrics(), forbidden)


def test_train_stage_rejects_partial_restart_and_stale_config_before_model_load(candidate, gpu_seams, monkeypatch):
    staging = candidate.root / 'new-train-stage'
    staging.mkdir()
    write(staging / 'config.json', candidate.config)
    write(staging / 'attempt.json', {'status': 'STARTED', **{k: candidate.config[k] for k in worker.IDENTITY_KEYS}})
    monkeypatch.setattr(worker, '_train', lambda *args: pytest.fail('partial/stale stage must not load model'))
    write(staging / 'metrics.json', {'partial': True})
    with pytest.raises(worker.SmokeError, match='FRESH_TRAIN_STAGE_IDENTITY_REQUIRED'):
        worker.train_checkpoint(candidate.config_path, staging, staging / 'train.json')
    (staging / 'metrics.json').unlink()
    write(staging / 'config.json', {**candidate.config, 'source_hash': 'stale'})
    with pytest.raises(worker.SmokeError, match='FRESH_TRAIN_STAGE_IDENTITY_REQUIRED'):
        worker.train_checkpoint(candidate.config_path, staging, staging / 'train.json')


def test_internal_train_cli_keeps_private_exception_and_disallows_both_stages(candidate, monkeypatch, capsys):
    staging = candidate.root / 'failed-train-stage'
    staging.mkdir()
    def fail(*args): raise RuntimeError('private target text')
    monkeypatch.setattr(worker, 'train_checkpoint', fail)
    assert worker.main(['--config', str(candidate.config_path), '--receipt', str(staging / 'train.json'),
                        '--train-checkpoint', str(staging)]) == 1
    assert 'private target text' not in capsys.readouterr().err
    assert 'private target text' in (staging / 'train.failure.private.json').read_text()
    with pytest.raises(SystemExit):
        worker.main(['--config', str(candidate.config_path), '--receipt', str(staging / 'train.json'),
                     '--train-checkpoint', str(staging), '--reload-checkpoint', str(staging)])


def test_real_sequential_train_and_reload_children_keep_same_leases_and_parent_without_cuda(candidate, monkeypatch):
    """Real CPU child exits release even deliberately retained model cycles.

    Native GPU training/reload are replaced here; separate actual Gemma tests
    cover loss/gradient behavior. No CUDA context is initialized in any process.
    """
    import fcntl
    pytest.importorskip('torch')
    for key in worker.IDENTITY_KEYS:
        monkeypatch.setenv('MEDVLM_SUPERVISOR_' + key.upper(), candidate.config[key])
    train_program = r'''
import json, os, sys
from pathlib import Path
import torch
from mediflow_medvlm import campaign_gpu_smoke as w
assert not torch.cuda.is_initialized()
retained_models = []
def train(config, rows, staging):
    assert len(rows) == 324
    layer = torch.nn.Linear(2, 2)
    layer(torch.ones(1, 2)).sum().backward()
    layer.cycle = {'model': layer}
    retained_models.append(layer)  # Deliberately kept until process exit.
    (staging/'adapter').mkdir()
    (staging/'adapter'/'adapter_model.safetensors').write_bytes(b'CPU_FAKE_ADAPTER')
    (staging/'processor').mkdir()
    w.atomic_json(staging/'processor'/'tokenizer.json', {})
    return {'optimizer_steps':20,'samples_processed':320,'excluded_count':0,'adapter_changed':True,
        'training_pid':os.getpid(),'adapter_sha256':'f'*64,
        'native_contract':{'parameter_storage':{'base_dtype_signature':'dtypes'}}}
w._train = train
code = w.main(['--config',sys.argv[1],'--train-checkpoint',sys.argv[2],'--receipt',sys.argv[3]])
assert not torch.cuda.is_initialized()
raise SystemExit(code)
'''
    reload_program = r'''
import json, os, sys
from pathlib import Path
import torch
from mediflow_medvlm import campaign_gpu_smoke as w
config = json.loads(Path(sys.argv[1]).read_text())
w._inherited_supervisor_fds(config)
assert not Path('/proc/'+sys.argv[3]).exists(), 'training child must already be reaped'
assert not torch.cuda.is_initialized()
value = {'status':'SUCCEEDED',**{k:config[k] for k in w.IDENTITY_KEYS},'pid':os.getpid(),
    'adapter_sha256':'f'*64,'held_train_sample_id':config['data']['selection']['held_train_record']['sample_id'],
    'finite_forward':True,'adapter_weights_verified':True,'fresh_process_required':True,'dev_test_read':False,
    'runtime_versions':config['runtime_versions'],'held_train_loss':1.0,'target_tokens':2,'generated_tokens':1,
    'base_dtype_signature':'dtypes'}
w.atomic_json(Path(sys.argv[2]),value)
'''
    streams = []
    seen = []
    training_pid = None
    def invoke(args, **kwargs):
        nonlocal training_pid
        assert set(kwargs['pass_fds']) == {stream.fileno() for stream in streams}
        receipt = args[args.index('--receipt') + 1]
        if '--train-checkpoint' in args:
            seen.append('train')
            staging = args[args.index('--train-checkpoint') + 1]
            result = subprocess.run([sys.executable, '-c', train_program, str(candidate.config_path), staging, receipt], **kwargs)
            if result.returncode == 0:
                training_pid = json.loads(Path(receipt).read_text())['training_pid']
                assert not Path('/proc/' + str(training_pid)).exists()
            return result
        seen.append('reload')
        assert training_pid is not None
        return subprocess.run([sys.executable, '-c', reload_program, str(candidate.config_path), receipt, str(training_pid)], **kwargs)
    try:
        for name in candidate.config['supervisor_lock_paths']:
            path = Path(name); path.parent.mkdir(parents=True, exist_ok=True)
            stream = path.open('a'); fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB); streams.append(stream)
        result = worker.run_smoke(candidate.config_path, candidate.receipt, invoke=invoke)
        assert result['status'] == 'SUCCEEDED' and seen == ['train', 'reload']
        metrics = json.loads((Path(candidate.config['output_dir']) / 'metrics.json').read_text())
        isolation = metrics['process_isolation']
        assert isolation['training_pid'] == training_pid != os.getpid()
        assert isolation['training_child_exit_verified'] and isolation['training_exited_before_reload']
        assert isolation['coordinator_cuda_initialized'] is False
        proof = json.loads((Path(candidate.config['output_dir']) / 'train.json').read_text())
        assert proof['metrics_sha256'] == worker.file_hash(Path(candidate.config['output_dir']) / 'train.metrics.json')
        for name in candidate.config['supervisor_lock_paths']:
            with Path(name).open('a') as probe:
                with pytest.raises(BlockingIOError): fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        for stream in streams: stream.close()
