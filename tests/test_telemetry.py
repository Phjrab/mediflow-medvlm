"""Telemetry can observe computed scalars without changing training science."""
import copy
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import pytest
from mediflow_medvlm import telemetry as t
from mediflow_medvlm import campaign_main_train as main


def manifest():
    return t.manifest_for({"job_id": "job", "run_id": "run", "task": "text", "model_id": "native/base",
        "recipe": {"optimizer_steps": 2, "examples": 29, "seed": 17}}, "attempt", "campaign")


def records(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()]


def test_import_has_no_gpu_runtime():
    code = "import sys; from mediflow_medvlm import telemetry,telemetry_import,telemetry_probe; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
    assert subprocess.run([sys.executable, "-c", code], capture_output=True).returncode == 0


@pytest.mark.parametrize("size", [29, 27])
def test_on_off_exact_weights_optimizer_rng_order_losses_and_partial_batch(tmp_path, size):
    torch = pytest.importorskip("torch")
    rows = [{"sample_id": str(i)} for i in range(size)]
    def run(enabled):
        random.seed(73); torch.manual_seed(73)
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(.1), torch.nn.Linear(2, 1)).double()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.02, weight_decay=0.)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(0., 1. - step / 2))
        lifecycle = t.TelemetryWriter(tmp_path / "out", manifest()) if enabled else None
        writer = t.TelemetryWriter(tmp_path / "out", manifest(), role="training") if enabled else None
        def example(row):
            x = torch.tensor([[float(int(row["sample_id"]) % 7) / 10, .5]], dtype=torch.float64)
            return model(x).square().mean(), int(row["sample_id"]) % 3 + 1, 9
        metrics = main._updates(model, optimizer, scheduler, rows, {}, example, lambda: {}, telemetry=writer)
        if writer: writer.close()
        return copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict()), torch.get_rng_state(), random.getstate(), metrics
    off, on = run(False), run(True)
    for key in off[0]: assert torch.equal(off[0][key], on[0][key])
    for parameter, state in off[1]["state"].items():
        for key, value in state.items():
            assert torch.equal(value, on[1]["state"][parameter][key]) if torch.is_tensor(value) else value == on[1]["state"][parameter][key]
    assert torch.equal(off[2], on[2]) and off[3] == on[3] and off[4] == on[4]
    steps = records(tmp_path / "out/steps.jsonl")
    assert [x["batch_examples"] for x in steps] == [16, size - 16]
    assert [x["lr_used"] for x in steps] == [.02, .01]
    assert [x["lr_next"] for x in steps] == [.01, 0.]
    assert steps[-1]["train_loss"] == sum(on[4]["example_losses"][16:]) / (size - 16)
    assert t.reconcile(steps, on[4])["status"] == "passed"
    assert steps[0]["cuda_allocated_bytes"] is None
    assert steps[0]["availability"]["cuda_allocated_bytes"]["available"] is False


def test_flush_observer_artifacts_finish_and_private_text_exclusion(tmp_path):
    root = tmp_path / "attempt"
    lifecycle = t.TelemetryWriter(root, manifest())
    child = t.TelemetryWriter(root, manifest(), role="training")
    lifecycle.event("model_loading", details={"prompt": "PRIVATE", "path": "/private", "samples_processed": 0})
    lifecycle.observe_steps()
    metrics = {"optimizer_steps": 2, "samples_processed": 29, "loss_mean": (16 + 26) / 29,
        "supervised_tokens": 58, "sequence_tokens": 145, "learning_rates": [.1, .05],
        "gradient_norms_before_clip": [.2, .3], "final_learning_rate": 0.}
    for i, n in enumerate((16, 13)):
        row = t.update_payload(step=i + 1, total=2, losses=[i + 1.] * n, tokens=[2] * n, sequences=[5] * n,
            samples=16 if i == 0 else 29, samples_total=29, lr_used=.1 if i == 0 else .05,
            lr_next=.05 if i == 0 else 0., norm=.2 if i == 0 else .3, wall=.2)
        child.step(row)
        assert len(records(root / "steps.jsonl")) == i + 1  # visible before close
    child.close(); lifecycle.stop_observer()
    t.atomic_json(tmp_path / "checkpoint.json", {"private_path": "/never/export"})
    t.atomic_json(tmp_path / "reload.json", {"status": "SUCCEEDED", "generated_text": "PRIVATE"})
    lifecycle.record_artifacts(tmp_path, reload_verified=True)
    lifecycle.finish(metrics, reload_succeeded=True); lifecycle.close()
    assert json.loads((root / "status.json").read_text())["telemetry_complete"] is True
    assert json.loads((root / "summary.json").read_text())["reconciliation"]["status"] == "passed"
    events = records(root / "events.jsonl")
    assert [r["seq"] for r in events] == list(range(1, len(events) + 1))
    assert sum(r["event"] == "optimizer_update" for r in events) == 2
    assert all(a["reload_verified"] for a in json.loads((root / "manifest.json").read_text())["artifacts"])
    assert "PRIVATE" not in "".join(p.read_text() for p in root.glob("*.json*"))
    assert "/never/export" not in (root / "manifest.json").read_text()


def test_logging_failure_isolated_and_durable_gap(tmp_path, monkeypatch):
    root = tmp_path / "attempt"; t.TelemetryWriter(root, manifest())
    writer = t.TelemetryWriter(root, manifest(), role="training")
    original = Path.open
    def fail(self, *args, **kwargs):
        if self.name == "steps.jsonl": raise OSError("SECRET /private/path")
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, "open", fail)
    writer.step({"optimizer_step": 1, "train_loss": math.nan})
    writer.close()
    health = json.loads((root / "training.health.json").read_text())
    assert health["telemetry_complete"] is False and health["writer_health"] == "degraded"
    assert health["gaps"][0]["from_seq"] == 1
    assert "SECRET" not in json.dumps(health)


def test_resource_sensor_failure_and_shared_device_units(monkeypatch):
    def failed(*args, **kwargs): raise FileNotFoundError("secret")
    monkeypatch.setattr(t.subprocess, "run", failed)
    row = t.sample_resources()
    assert row["gpu_utilization_percent"] is None
    assert row["availability"]["gpu_utilization_percent"]["reason"] == "nvidia_smi_unavailable"
    assert row["attribution"] == "unavailable" and row["process_rss_bytes"] > 0
    from types import SimpleNamespace
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(t.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="GPU-safe, 0, 30, 10, 100, 42, 18.5, 900, 1200\n"))
    row = t.sample_resources()
    assert row["attribution"] == "device_shared" and row["gpu_memory_used_bytes"] == 10 * 1024 * 1024
    assert type(row["gpu_sm_clock_mhz"]) is int and row["gpu_uuid"] == "GPU-safe"


def test_attempt_stream_is_exclusive_and_bad_root_health_is_private(tmp_path, monkeypatch):
    root=tmp_path/'out'; t.TelemetryWriter(root, manifest())
    one=t.TelemetryWriter(root, manifest(), role='training'); one.step({'optimizer_step':1})
    two=t.TelemetryWriter(root, manifest(), role='training'); two.step({'optimizer_step':2})
    assert two.error == 'append_or_flush_failed' and len(records(root/'steps.jsonl')) == 1
    one.close(); two.close()
    monkeypatch.setenv('MEDVLM_TELEMETRY_ROOT','relative')
    result=t.attempt_writer({},tmp_path)
    assert result.error == 'initialization_failed'
    assert (tmp_path/'telemetry.health.json').exists()
