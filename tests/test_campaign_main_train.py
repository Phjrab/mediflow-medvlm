"""CPU-only worker proofs. Fake CUDA seams never establish actual training fit."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from mediflow_medvlm import campaign_main_train as w

EXECUTION_VALIDATE = w._validate


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def candidate(tmp_path, monkeypatch):
    size, steps = 29, 2
    records = [{"sample_id": f"train-{i}", "native": {"supervised_tokens_after_shift": i % 3 + 1,
        "sequence_tokens": i % 3 + 5}} for i in range(size)]
    architecture = {"architecture": {"exact_target_modules": ["a", "b"], "target_count": 2}, "template": {}, "thinking": {}}
    prepath = write(tmp_path / "pre/config.json", {"output_dir": str(tmp_path / "pre")})
    write(tmp_path / "pre/architecture.json", architecture)
    model = {"model_id": "native/base", "snapshot_path": "immutable-snapshot", "revision": "pinned", "model_files_hash": "weights"}
    config = {"schema_version": w.SCHEMA, "scope": w.SCOPE, "cell_id": "T1", "model_id": "native/base",
        "output_dir": str(tmp_path / "out/T1"), "success_receipt": str(tmp_path / "receipt.json"),
        "recipe": {"examples": size, "optimizer_steps": steps, "learning_rate": .05}, "model": model,
        "runtime_versions": {}, "gpu_ledger_path": str(tmp_path / "ledger.json"),
        "data": {"sample_order_sha256": w.canonical_hash([r["sample_id"] for r in records]),
            "target_tokens": sum(r["native"]["supervised_tokens_after_shift"] for r in records),
            "sequence_tokens": sum(r["native"]["sequence_tokens"] for r in records),
            "update_example_counts": [16, 13], "preflight_config": {"path": str(prepath)},
            "probe_sample_id": records[-1]["sample_id"]}, **{k:"identity-" + k for k in w.IDENTITY_KEYS}}
    path = write(tmp_path / "config.json", config)
    monkeypatch.setattr(w, "_validate", lambda cfg: ({}, copy.deepcopy(records)))
    monkeypatch.setattr(w, "_inherited_supervisor_fds", lambda cfg: ())
    contract = {**architecture, "weights_loaded": True, "cuda_residency_verified": True,
        "files": {"all_prepared_files_rehashed": True, "model_files_hash": "weights"},
        "revision": "pinned", "snapshot_path": "immutable-snapshot", "gradient_checkpointing": True,
        "quantization": {"method": "nf4", "double_quant": True, "compute_dtype": "bfloat16"},
        "coverage": {"base_frozen": True, "vision_projector_embeddings_head_frozen": True,
            "quantized_targets_verified": True, "exact_target_modules": ["a", "b"], "target_count": 2,
            "target_modules_sha256": w.canonical_hash(["a", "b"])}, "parameter_storage": {"base_dtype_signature": "dtype"}}
    reports = [{"all_finite": True, "projections": [{"path": p, "tensors": {side: {"finite": True, "nonzero": True, "l2_norm": .1}
        for side in ("A", "B")}} for p in ("a", "b")]} for _ in range(steps)]
    metrics = {"optimizer_steps": steps, "samples_processed": size, "excluded_count": 0,
        "sample_ids": [r["sample_id"] for r in records], "sample_order_sha256": config["data"]["sample_order_sha256"],
        "update_example_counts": [16, 13], "micro_loss_divisor_per_update": [16, 13],
        "example_losses": [1.] * size, "loss_mean": 1.,
        "example_supervised_tokens": [r["native"]["supervised_tokens_after_shift"] for r in records],
        "supervised_tokens": config["data"]["target_tokens"], "sequence_tokens": config["data"]["sequence_tokens"],
        "learning_rates": [.05, .025], "final_learning_rate": 0., "gradient_reports": reports,
        "gradient_norms_before_clip": [.1, .1], "initial_adapter_sha256": "before", "adapter_sha256": "after",
        "adapter_changed": True, "initial_projection_sha256": {p:{s:"old" for s in ("A", "B")} for p in ("a", "b")},
        "final_projection_sha256": {p:{s:"new" for s in ("A", "B")} for p in ("a", "b")},
        "all_projection_tensors_changed": True, "base_frozen_verified": True, "retry_start": "original_base",
        "optimizer_rng_resume_supported": False, "native_contract": contract, "training_pid": os.getpid() + 100000}
    return SimpleNamespace(config=config, path=path, records=records, metrics=metrics, root=tmp_path)


@pytest.fixture
def invoke(candidate):
    c = candidate
    def run(args, **kwargs):
        config = w._read(args[args.index("--config") + 1])
        stage = "train" if "--train-checkpoint" in args else "reload"
        directory = Path(args[args.index("--" + stage + "-checkpoint") + 1])
        path = Path(args[args.index("--receipt") + 1])
        pid = c.metrics["training_pid"] if stage == "train" else c.metrics["training_pid"] + 1
        identity = {"worker_pid": pid, "boot_identity": "boot", "process_start_time": str(pid)}
        if stage == "train":
            write(directory / "adapter/adapter_config.json", {"base": "pinned"})
            (directory / "adapter/adapter_model.safetensors").write_bytes(b"CPU_FAKE_PRIVATE_ADAPTER")
            write(directory / "processor/config.json", {"processor": "native"})
            write(directory / "train.metrics.json", c.metrics)
            value = {"status": "SUCCEEDED", "scope": "NATIVE_MAIN_ONE_PASS_TRAIN_ONLY",
                **{k:config[k] for k in w.IDENTITY_KEYS}, "training_pid": pid, "process_identity": identity,
                "metrics_sha256": w.file_hash(directory / "train.metrics.json"), "retry_start": "original_base",
                "optimizer_steps": config["recipe"]["optimizer_steps"], "samples_processed": config["recipe"]["examples"]}
        else:
            from mediflow_medvlm.training.v6_native import verify_native_no_thinking_output
            write(directory / "reload.generated.private.json", {"trained_example_sample_id": c.records[-1]["sample_id"],
                "generated_ids": [42], "generated_text": "answer"})
            value = {"status": "SUCCEEDED", **{k:config[k] for k in w.IDENTITY_KEYS}, "pid": pid, "process_identity": identity,
                "probe_scope": "TRAINING_EXAMPLE_SANITY_CHECK", "trained_example_sample_id": c.records[-1]["sample_id"],
                "adapter_sha256": "after", "finite_forward": True, "adapter_weights_verified": True,
                "fresh_process_required": True, "dev_test_read": False, "trained_example_loss": 1.,
                "target_tokens": c.records[-1]["native"]["supervised_tokens_after_shift"], "base_dtype_signature": "dtype",
                "runtime_versions": {}, "generated_tokens": 1, "generated_ids_sha256": w.canonical_hash([42]),
                "generation_max_new_tokens": 32, "no_thinking_output": verify_native_no_thinking_output("answer")}
        write(path, value)
        return SimpleNamespace(returncode=0, pid=pid, process_identity=identity)
    return run


def test_cpu_import_does_not_load_gpu_runtime():
    result = subprocess.run([sys.executable, "-c", "import sys; from mediflow_medvlm import campaign_main_train; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"], capture_output=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("size", [17, 29, 27])
def test_partial_update_exact_example_mean_gradient_optimizer_and_schedule(size):
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(1, 1, bias=False).double()
    with torch.no_grad(): model.weight.fill_(.3)
    reference = copy.deepcopy(model)
    records = [{"sample_id": str(i), "n": i % 5 + 1} for i in range(size)]
    steps = (size + 15) // 16
    optimizer = torch.optim.AdamW(model.parameters(), lr=.05, betas=(.9, .999), eps=1e-8, weight_decay=0.)
    expected_optimizer = torch.optim.AdamW(reference.parameters(), lr=.05, betas=(.9, .999), eps=1e-8, weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(0., 1. - step / steps))
    expected_scheduler = torch.optim.lr_scheduler.LambdaLR(expected_optimizer, lambda step: max(0., 1. - step / steps))
    def loss(m, r):
        x = torch.tensor([[.1 * (int(r["sample_id"]) % 3 + 1)]], dtype=torch.float64)
        targets = torch.arange(r["n"], dtype=torch.float64) * .05
        return (m(x).reshape(()) - targets).square().mean()
    observed_gradients = []
    result = w._updates(model, optimizer, scheduler, records, {},
        lambda r: (loss(model, r), r["n"], r["n"] + 2),
        lambda: observed_gradients.append(model.weight.grad.clone()) or {})
    for step, start in enumerate(range(0, size, 16)):
        batch = records[start:start + 16]
        expected_optimizer.zero_grad(set_to_none=True)
        torch.stack([loss(reference, r) for r in batch]).mean().backward()
        torch.testing.assert_close(observed_gradients[step], reference.weight.grad, rtol=1e-13, atol=1e-13)
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.)
        expected_optimizer.step(); expected_scheduler.step()
    torch.testing.assert_close(model.weight, reference.weight, rtol=1e-13, atol=1e-13)
    assert result["sample_ids"] == [str(i) for i in range(size)]
    assert result["samples_processed"] == size and result["optimizer_steps"] == steps
    assert result["micro_loss_divisor_per_update"] == [16, size - 16]
    assert result["supervised_tokens"] == sum(r["n"] for r in records)
    assert result["learning_rates"] == [.05, .025] and result["final_learning_rate"] == 0


def test_metrics_valid(candidate):
    assert w._metrics_proof(candidate.config, candidate.metrics, candidate.records) == candidate.metrics


@pytest.mark.parametrize("kind", ["truncated", "extra", "reordered", "duplicated", "wrong_divisor", "target", "sequence", "lr", "final_lr", "loss", "projection", "gradient", "gradient_bool", "base", "loaded", "quantization", "model", "files"])
def test_false_one_pass_or_native_evidence_rejected(candidate, kind):
    m = candidate.metrics
    if kind == "truncated": m["samples_processed"] -= 1
    elif kind == "extra": m["optimizer_steps"] += 1
    elif kind == "reordered": m["sample_ids"].reverse()
    elif kind == "duplicated": m["sample_ids"][-1] = m["sample_ids"][0]
    elif kind == "wrong_divisor": m["micro_loss_divisor_per_update"][-1] = 16
    elif kind == "target": m["example_supervised_tokens"][-1] += 1
    elif kind == "sequence": m["sequence_tokens"] += 1
    elif kind == "lr": m["learning_rates"][-1] = .05
    elif kind == "final_lr": m["final_learning_rate"] = .001
    elif kind == "loss": m["example_losses"][-1] = float("nan")
    elif kind == "projection":
        del m["initial_projection_sha256"]["b"]; del m["final_projection_sha256"]["b"]
        for report in m["gradient_reports"]: report["projections"].pop()
    elif kind == "gradient": m["gradient_reports"][-1]["all_finite"] = False
    elif kind == "gradient_bool": m["gradient_reports"][0]["projections"][0]["tensors"]["A"]["nonzero"] = 1
    elif kind == "base": m["base_frozen_verified"] = False
    elif kind == "loaded": m["native_contract"]["weights_loaded"] = False
    elif kind == "quantization": m["native_contract"]["quantization"]["compute_dtype"] = "float32"
    elif kind == "model": m["native_contract"]["revision"] = "other"
    else: m["native_contract"]["files"]["all_prepared_files_rehashed"] = False
    with pytest.raises(w.MainTrainError): w._metrics_proof(candidate.config, m, candidate.records)


def test_success_final_selection_exact_artifacts_idempotency_and_budget(candidate, invoke):
    c = candidate
    result = w.run_train(c.path, Path(c.config["success_receipt"]), invoke=invoke)
    assert result["samples_processed"] == 29 and result["optimizer_steps"] == 2
    assert result["full_one_pass_verified"] and result["final_checkpoint_verified"]
    out = Path(c.config["output_dir"])
    assert w._read(out / "final_selection.json")["selection_rule"] == "ONE_PASS_FINAL_NOT_MINIMUM_LOSS"
    assert w._read(out / "reload.json")["probe_scope"] == "TRAINING_EXAMPLE_SANITY_CHECK"
    assert w.run_train(c.path, Path(c.config["success_receipt"])) == result
    Path(c.config["success_receipt"]).unlink()
    assert w.run_train(c.path, Path(c.config["success_receipt"])) == result
    ledger = w._read(c.config["gpu_ledger_path"])
    assert [r["phase"] for r in ledger["sessions"]] == ["native_main_train", "native_main_reload"]
    assert ledger["current_policy"]["budget_mode"] == "unlimited" and ledger["active"] is None
    assert all(p.stat().st_mode & 0o077 == 0 for p in out.rglob("*"))


@pytest.mark.parametrize("artifact", ["adapter/adapter_model.safetensors", "processor/config.json", "metrics.json", "final_selection.json", "reload.json"])
def test_final_artifact_corruption_rejected(candidate, invoke, artifact):
    c = candidate; w.run_train(c.path, Path(c.config["success_receipt"]), invoke=invoke)
    path = Path(c.config["output_dir"]) / artifact; path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(w.MainTrainError): w.run_train(c.path, Path(c.config["success_receipt"]))


@pytest.mark.parametrize("kind", ["train_exit", "reload_exit", "same_pid", "actual_pid", "heldout", "generation", "target", "weights"])
def test_child_failure_or_false_reload_never_publishes_success(candidate, invoke, kind):
    c = candidate; seen = []
    def altered(args, **kwargs):
        stage = "train" if "--train-checkpoint" in args else "reload"; seen.append(stage)
        result = invoke(args, **kwargs)
        if kind == stage + "_exit": result.returncode = 2
        path = Path(args[args.index("--receipt") + 1]); value = w._read(path)
        if stage == "reload":
            if kind == "same_pid": value["pid"] = c.metrics["training_pid"]
            if kind == "actual_pid": result.pid += 1
            if kind == "heldout": value["probe_scope"] = "HELD_OUT_EVALUATION"
            if kind == "generation": value["generated_tokens"] = 2
            if kind == "target": value["target_tokens"] += 1
            if kind == "weights": value["adapter_sha256"] = "wrong"
            write(path, value)
        return result
    with pytest.raises(w.MainTrainError): w.run_train(c.path, Path(c.config["success_receipt"]), invoke=altered)
    assert not Path(c.config["output_dir"]).exists() and not Path(c.config["success_receipt"]).exists()
    attempts = list(Path(c.config["output_dir"]).parent.glob("T1.attempt-*"))
    assert len(attempts) == 1 and (attempts[0] / "failure.private.json").exists()
    assert not (attempts[0] / "completion.json").exists()
    if kind == "train_exit": assert seen == ["train"] and len(w._read(c.config["gpu_ledger_path"])["sessions"]) == 1


def test_failed_attempt_preserved_and_retry_original_base(candidate, invoke):
    c = candidate
    def fail(args, **kwargs):
        directory = Path(args[args.index("--train-checkpoint") + 1]); write(directory / "partial.json", {"step": 1})
        return SimpleNamespace(returncode=2)
    with pytest.raises(w.MainTrainError): w.run_train(c.path, Path(c.config["success_receipt"]), invoke=fail)
    previous = next(Path(c.config["output_dir"]).parent.glob("T1.attempt-*"))
    assert w.run_train(c.path, Path(c.config["success_receipt"]), invoke=invoke)["status"] == "SUCCEEDED"
    assert (previous / "partial.json").exists() and w._read(previous / "attempt.json")["retry_start"] == "original_base"


def test_incomplete_output_cannot_resume(candidate):
    out = Path(candidate.config["output_dir"]); write(out / "adapter/adapter_config.json", {})
    with pytest.raises(w.MainTrainError, match="INCOMPLETE_OUTPUT"): w.run_train(candidate.path, Path(candidate.config["success_receipt"]))


def test_resealed_metrics_still_need_exact_semantics(candidate, invoke):
    c = candidate; w.run_train(c.path, Path(c.config["success_receipt"]), invoke=invoke)
    out = Path(c.config["output_dir"]); value = w._read(out / "metrics.json"); value["optimizer_steps"] = 3
    write(out / "metrics.json", value); marker = w._read(out / "completion.json")
    for item in marker["artifacts"]:
        if item["path"] == str(out / "metrics.json"): item["sha256"] = w.file_hash(out / "metrics.json")
    write(out / "completion.json", marker); Path(c.config["success_receipt"]).unlink()
    with pytest.raises(w.MainTrainError, match="INCOMPLETE_OR_REORDERED"): w.run_train(c.path, Path(c.config["success_receipt"]))


def test_strict_rebuild_rejects_changed_scientific_config(candidate, monkeypatch):
    expected = copy.deepcopy(candidate.config)
    expected["config_hash"] = w.canonical_hash({k:v for k,v in expected.items() if k != "config_hash"})
    expected["data"]["common_freeze"] = {"path": "/immutable/freeze.json"}
    expected["config_hash"] = w.canonical_hash({k:v for k,v in expected.items() if k != "config_hash"})
    monkeypatch.setattr(w, "build_config", lambda *a, **kw: copy.deepcopy(expected))
    assert w.validate_config(expected) == expected
    booleans = copy.deepcopy(expected)
    booleans["recipe"]["passes"] = 1
    monkeypatch.setattr(w, "build_config", lambda *a, **kw: copy.deepcopy(booleans))
    booleans["config_hash"] = w.canonical_hash({k:v for k,v in booleans.items() if k != "config_hash"})
    masquerade = copy.deepcopy(booleans); masquerade["recipe"]["passes"] = True
    masquerade["config_hash"] = w.canonical_hash({k:v for k,v in masquerade.items() if k != "config_hash"})
    with pytest.raises(w.MainTrainError, match="FROZEN_CONFIG_IDENTITY"): w.validate_config(masquerade)
    monkeypatch.setattr(w, "build_config", lambda *a, **kw: copy.deepcopy(expected))
    changed = copy.deepcopy(expected); changed["recipe"]["learning_rate"] *= 2
    changed["config_hash"] = w.canonical_hash({k:v for k,v in changed.items() if k != "config_hash"})
    with pytest.raises(w.MainTrainError, match="FROZEN_CONFIG_IDENTITY"): w.validate_config(changed)


def test_real_child_pid_identity_exit_and_inherited_leases(candidate, monkeypatch):
    import fcntl
    from mediflow_medvlm.campaign_supervisor import same_process
    c = candidate
    paths = [c.root / "worker.lock", c.root / "gpu.lock"]
    c.config["supervisor_lock_paths"] = list(map(str, paths)); write(c.path, c.config)
    locks = [p.open("a+") for p in paths]
    for stream in locks: fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for key in w.IDENTITY_KEYS: monkeypatch.setenv("MEDVLM_SUPERVISOR_" + key.upper(), c.config[key])
    monkeypatch.setattr(w, "_inherited_supervisor_fds", w.smoke._inherited_supervisor_fds)
    script = c.root / "child.py"
    script.write_text('''import json,os,sys
from pathlib import Path
from mediflow_medvlm import campaign_main_train as w
c=w._read(sys.argv[1]); w._inherited_supervisor_fds(c)
v={"status":"SUCCEEDED",**{k:c[k] for k in w.IDENTITY_KEYS},"training_pid":os.getpid(),"process_identity":w._child_identity()}
w.atomic_json(Path(sys.argv[2]),v)
''')
    original = subprocess.Popen
    def popen(args, **kwargs):
        assert len(kwargs["pass_fds"]) == 2
        return original([sys.executable, str(script), str(c.path), args[args.index("--receipt") + 1]], **kwargs)
    monkeypatch.setattr(w.subprocess, "Popen", popen)
    try:
        value, exit = w._process("train", c.path, c.root, c.config)
        assert exit["pid"] == value["training_pid"] != os.getpid() and exit["exit_verified"]
        assert not same_process(exit["process_identity"])
        for p in paths:
            with p.open("a+") as stream:
                with pytest.raises(BlockingIOError): fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        for stream in locks: stream.close()


def test_mid_update_failure_cannot_save_or_publish_partial_checkpoint():
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.05)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    records = [{"sample_id": str(i)} for i in range(17)]
    calls = []
    def example(record):
        calls.append(record["sample_id"])
        if len(calls) == 9: raise ValueError("private encoding failure")
        return model(torch.ones(1, 1)).square().mean(), 1, 2
    with pytest.raises(ValueError): w._updates(model, optimizer, scheduler, records, {}, example, lambda: {})
    assert optimizer.state == {} and scheduler.last_epoch == 0


def test_historical_freeze_requires_terminal_success_even_with_success_verification(tmp_path, monkeypatch):
    path = tmp_path / "receipt.json"
    identity = {k:"id-" + k for k in w.IDENTITY_KEYS}
    config = {"schema_version": w.common.SCHEMA, "scope": w.common.SCOPE, "state_path": str(tmp_path / "state.json"),
        "success_receipt": str(path), "source_files": {}, **identity}
    config["source_hash"] = w.canonical_hash({})
    config["job_id"] = config["run_id"] = w.common.JOB_ID
    config["config_hash"] = w.canonical_hash({k:v for k,v in config.items() if k != "config_hash"})
    cfgpath = write(tmp_path / "config.json", config)
    receipt = {"status":"SUCCEEDED", "scope":w.common.SCOPE, "common_train_freeze_decided":True,
        "v6_main_training_completed":False, "artifacts":[{"path":str(cfgpath),"sha256":w.file_hash(cfgpath)}]}
    write(path, receipt)
    monkeypatch.setattr(w.common, "_completed", lambda _: receipt)
    job = {**config, "job_id":w.common.JOB_ID, "phase":"common_train_freeze", "cell_id":"ALL_V6"}
    monkeypatch.setattr(w.common.gate, "_active", lambda _: ({"jobs":[job]}, {"jobs":{w.common.JOB_ID:{"status":"RUNNING"}}}))
    with pytest.raises(w.MainTrainError, match="COMMON_FREEZE_TERMINAL_SOURCE_IDENTITY"): w._freeze(path)


def test_changed_proven_gpu_runtime_rejected_before_loading(tmp_path, monkeypatch):
    spec = w.cell_spec("T1")
    nativepath = write(tmp_path / "native.json", {})
    gpupath = write(tmp_path / "gpu.json", {"runtime_versions":{"torch":"old"},"model":{}})
    run = {**spec, "adapter_initialization":None,"initialization_source":"original_pinned_base",
        "smoke_adapter_reuse":False,"historical_training_reused":False,"model":{},
        "native_preflight_config":{"path":str(nativepath),"sha256":w.file_hash(nativepath)},
        "gpu_smoke_config":{"path":str(gpupath),"sha256":w.file_hash(gpupath)}}
    plan = {"supported_cells":["T1"],"per_cell_run_plan":{"T1":run},"axes":{"text":{"cells":["T1"],"common_count":7997}}}
    monkeypatch.setattr(w,"runtime_versions",lambda:{"torch":"new"})
    with pytest.raises(w.MainTrainError,match="PROVEN_GPU_RUNTIME_OR_MODEL_CHANGED"): w._inputs({},plan,"T1")


def test_aggregate_environment_pure_verification_and_execution_separation(candidate, invoke, monkeypatch):
    c = candidate
    c.config["data"]["common_freeze"] = {"path": "/immutable/freeze.json"}
    c.config["config_hash"] = w.canonical_hash({k:v for k,v in c.config.items() if k != "config_hash"})
    write(c.path, c.config)
    result = w.run_train(c.path, Path(c.config["success_receipt"]), invoke=invoke)
    expected = copy.deepcopy(c.config)
    monkeypatch.setattr(w, "build_config", lambda *a, **kw: copy.deepcopy(expected))
    monkeypatch.setattr(w, "_freeze", lambda _: ({}, {}, {}))
    monkeypatch.setattr(w, "_inputs", lambda *a: ({}, {}, copy.deepcopy(c.records), {}, {}))
    for key in w.IDENTITY_KEYS: monkeypatch.setenv("MEDVLM_SUPERVISOR_" + key.upper(), "aggregate-" + key)
    assert w.validate_config(c.config) == c.config
    assert w.verify_result(c.config) == result
    with pytest.raises(w.MainTrainError, match="SUPERVISOR_IDENTITY_MISMATCH"): EXECUTION_VALIDATE(c.config)


def test_static_failed_cell_validation_never_loads_missing_model_or_data(candidate, monkeypatch):
    c = copy.deepcopy(candidate.config)
    c.update(w.cell_spec("T1"), phase="native_main_train", job_id="v6-first-seed-T1", run_id="v6-first-seed-T1",
        adapter_initialization=None, device="cuda", max_length=2048, fixed_template_date="2026-11-14",
        local_files_only=True, trust_remote_code=False, resource_policy=w.ResourcePolicy().as_dict())
    c["recipe"].update(seed=20261023, examples=7997, optimizer_steps=500)
    c["data"].update(count=7997, partition="train", probe_scope="TRAINING_EXAMPLE_SANITY_CHECK", update_example_counts=[16]*499+[13],
        train_file="/unavailable/train.jsonl")
    c["model"]["snapshot_path"] = "/unavailable/model"
    c["source_files"], c["source_hash"] = w.source_identity()
    c["data_hash"], c["model_hash"] = w.canonical_hash(c["data"]), w.canonical_hash(c["model"])
    c["config_hash"] = w.canonical_hash({k:v for k,v in c.items() if k != "config_hash"})
    monkeypatch.setattr(w, "_recipe", lambda plan, task: copy.deepcopy(plan["recipe"]))
    monkeypatch.setattr(w, "_freeze", lambda *a: pytest.fail("static failed-cell validation must not read live inputs"))
    assert w.validate_config_identity(c) == c
    c["recipe"]["examples"] = 7996
    c["config_hash"] = w.canonical_hash({k:v for k,v in c.items() if k != "config_hash"})
    with pytest.raises(w.MainTrainError): w.validate_config_identity(c)
