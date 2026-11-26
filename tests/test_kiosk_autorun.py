"""Orchestration checks with real short subprocesses; never load ML runtimes."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from mediflow_medvlm.kiosk import autorun


@pytest.fixture
def flow(tmp_path, monkeypatch):
    photos = tmp_path / "photos"
    photos.mkdir()
    adapter = tmp_path / "previous-model"
    adapter.mkdir()
    template = tmp_path / "previous-config.json"
    template.write_text("{}")
    config = {"schema_version": "medvlm.kiosk-terminal.v1", "data_root": str(tmp_path / "data"),
              "work_dir": str(tmp_path / "work"), "training_template": str(template),
              "initialization_adapter": str(adapter), "start_from_base": False,
              "sources": [{"source_id": "eye", "image_dir": str(photos), "modality": "external_eye_rgb"}],
              "teachers": {}, "pseudo_labels": {"allow_train": True, "allow_validation": False,
                                               "min_confidence": .95, "min_margin": .15}}
    config_path = tmp_path / "kiosk.json"
    config_path.write_text(json.dumps(config))
    calls = tmp_path / "calls.jsonl"
    monkeypatch.setenv("KIOSK_AUTORUN_TEST_CALLS", str(calls))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    script = "#!" + sys.executable + "\n" + r'''
import json, os, signal, sys, time
from pathlib import Path
stage = "preflight" if sys.argv[1] == "-c" else sys.argv[sys.argv.index("--config") + 2]
with open(os.environ["KIOSK_AUTORUN_TEST_CALLS"], "a") as stream:
    stream.write(json.dumps({"stage": stage, "executable": sys.argv[0],
                             "cuda": os.environ.get("CUDA_VISIBLE_DEVICES"), "argv": sys.argv[1:]}) + "\n")
print("fixture stage", stage, flush=True)
if stage == os.environ.get("KIOSK_AUTORUN_TEST_FAIL"):
    print("fixture nonzero / partial result", flush=True)
    sys.exit(2)
if stage == os.environ.get("KIOSK_AUTORUN_TEST_TAMPER"):
    snapshot = Path(sys.argv[sys.argv.index("--config") + 1])
    snapshot.write_text(snapshot.read_text() + " ")
if stage == "train" and os.environ.get("KIOSK_AUTORUN_TEST_WAIT"):
    marker = Path(os.environ["KIOSK_AUTORUN_TEST_WAIT"])
    def stop(signum, frame):
        marker.with_suffix(".stopped").write_text(str(signum))
        sys.exit(0)
    signal.signal(signal.SIGINT, stop)
    marker.write_text(str(os.getpid()))
    while True:
        time.sleep(.02)
'''
    label, train = tmp_path / "label python", tmp_path / "train python"
    for executable in (label, train):
        executable.write_text(script)
        executable.chmod(0o700)
    argv = ["--config", str(config_path), "--label-python", str(label), "--train-python", str(train),
            "--name", "night-01", "--allow-pseudo-validation"]
    return {"argv": argv, "config_path": config_path, "config": config, "calls": calls,
            "root": tmp_path / "work" / "autoruns" / "night-01", "label": label, "train": train}


def read_calls(flow):
    return [json.loads(row) for row in flow["calls"].read_text().splitlines()] if flow["calls"].exists() else []


def test_sequence_uses_separate_envs_and_never_evaluates_pseudo_truth(flow):
    original = flow["config_path"].read_bytes()
    assert autorun.main(flow["argv"]) == 0
    calls = read_calls(flow)
    assert [row["stage"] for row in calls] == ["preflight", "scan", "label", "review", "export", "prepare", "train", "reload", "results"]
    for row in calls:
        cpu = row["stage"] in {"scan", "label", "review", "export"}
        assert row["executable"] == str(flow["label"] if cpu else flow["train"])
        assert row["cuda"] == ("-1" if cpu else "3")
    assert flow["config_path"].read_bytes() == original
    snapshot = json.loads((flow["root"] / "config.json").read_text())
    assert snapshot["pseudo_labels"]["allow_validation"] is True
    assert snapshot["pseudo_labels"]["min_confidence"] == .95
    assert snapshot["initialization_adapter"] == flow["config"]["initialization_adapter"]
    assert snapshot["start_from_base"] is False
    state = json.loads((flow["root"] / "state.json").read_text())
    assert state["status"] == "SUCCEEDED" and state["clinical_accuracy_evaluated"] is False
    assert len(list(flow["root"].glob("*.log"))) == 9
    assert calls[-1]["argv"][-2:] == ["--save", "--plot"]


@pytest.mark.parametrize("failure", ["preflight", "label", "export", "train", "reload", "results"])
def test_failure_never_launches_later_stages(flow, monkeypatch, failure):
    monkeypatch.setenv("KIOSK_AUTORUN_TEST_FAIL", failure)
    assert autorun.main(flow["argv"]) == 2
    calls = read_calls(flow)
    assert calls[-1]["stage"] == failure
    state = json.loads((flow["root"] / "state.json").read_text())
    assert state["status"] == "FAILED"
    assert state["stages"][-1]["status"] == "FAILED"
    if failure in {"preflight", "label", "export"}:
        assert "train" not in [row["stage"] for row in calls]


def test_existing_name_cannot_replace_snapshot_or_state(flow):
    assert autorun.main(flow["argv"]) == 0
    state = (flow["root"] / "state.json").read_bytes()
    calls = flow["calls"].read_bytes()
    assert autorun.main(flow["argv"]) == 2
    assert (flow["root"] / "state.json").read_bytes() == state
    assert flow["calls"].read_bytes() == calls


def test_spawn_failure_marks_stage_failed(flow, monkeypatch):
    def fail_spawn(*args, **kwargs):
        raise OSError("fixture executable vanished")
    monkeypatch.setattr(autorun.subprocess, "Popen", fail_spawn)
    assert autorun.main(flow["argv"]) == 2
    state = json.loads((flow["root"] / "state.json").read_text())
    assert state["status"] == "FAILED"
    assert state["stages"][-1]["status"] == "FAILED"
    assert state["stages"][-1]["finished_at"]
    assert "vanished" in state["stages"][-1]["reason"]


@pytest.mark.parametrize("field", ["training_template", "initialization_adapter"])
def test_missing_existing_vlm_stops_before_labeling(flow, field):
    config = flow["config"]
    config[field] = None
    flow["config_path"].write_text(json.dumps(config))
    assert autorun.main(flow["argv"]) == 2
    assert not read_calls(flow)


def test_snapshot_tampering_stops_before_later_commands(flow, monkeypatch):
    monkeypatch.setenv("KIOSK_AUTORUN_TEST_TAMPER", "label")
    assert autorun.main(flow["argv"]) == 2
    assert [row["stage"] for row in read_calls(flow)] == ["preflight", "scan", "label"]
    assert "snapshot" in json.loads((flow["root"] / "state.json").read_text())["reason"]


def test_pseudo_validation_requires_explicit_flag(flow):
    config = flow["config"]
    config["pseudo_labels"]["allow_validation"] = True
    flow["config_path"].write_text(json.dumps(config))
    assert autorun.main(flow["argv"][:-1]) == 2
    assert not read_calls(flow)


def test_venv_python_symlink_is_preserved(tmp_path):
    alias = tmp_path / "venv" / "bin" / "python"
    alias.parent.mkdir(parents=True)
    alias.symlink_to(sys.executable)
    assert autorun._python(str(alias)) == str(alias)
    assert autorun._python(str(alias)) != str(alias.resolve())


@pytest.mark.skipif(os.name != "posix", reason="Linux/WSL signal workflow")
def test_sigterm_becomes_child_sigint_and_waits_for_cleanup(flow, tmp_path):
    marker = tmp_path / "worker.ready"
    env = dict(os.environ, KIOSK_AUTORUN_TEST_WAIT=str(marker))
    runner = Path(autorun.REPO) / "scripts" / "kiosk_autorun.py"
    process = subprocess.Popen([sys.executable, str(runner), *flow["argv"]], env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 15
        while not marker.exists() and time.monotonic() < deadline and process.poll() is None:
            time.sleep(.02)
        assert marker.exists()
        child_pid = int(marker.read_text())
        process.send_signal(signal.SIGTERM)
        process.communicate(timeout=10)
        assert process.returncode == 143
        assert marker.with_suffix(".stopped").read_text() == str(signal.SIGINT)
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
        state = json.loads((flow["root"] / "state.json").read_text())
        assert state["status"] == "CANCELLED"
        assert state["stages"][-1]["name"] == "train"
        assert state["stages"][-1]["status"] == "CANCELLED"
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            process.communicate(timeout=10)
