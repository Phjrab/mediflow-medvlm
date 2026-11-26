"""CPU checks for user-facing commands, truthful status, and interruption handling."""
import csv
import json
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest

from mediflow_medvlm.kiosk import common, terminal
from mediflow_medvlm.runtime import atomic_json


def configuration(tmp_path):
    return {
        "schema_version": common.SCHEMA,
        "work_dir": str(tmp_path / "work"), "data_root": str(tmp_path / "shared"),
        "sources": [{"source_id": "eye", "image_dir": str(tmp_path / "photos"),
                     "modality": "external_eye_rgb"}],
        "teachers": {}, "split": {"train": 0.8, "validation": 0.1, "test": 0.1, "seed": 7},
        "pseudo_labels": {"allow_train": True, "allow_validation": False,
                          "min_confidence": 0.95, "min_margin": 0.15},
    }


def test_config_resolves_relative_paths_against_file(tmp_path):
    cfg = configuration(tmp_path)
    cfg["sources"][0]["image_dir"] = "photos"
    cfg["teachers"]["eye"] = {"checkpoint": "weights/model.keras"}
    path = tmp_path / "kiosk.local.json"
    atomic_json(path, cfg)
    loaded = common.load_config(path)
    assert loaded["sources"][0]["image_dir"] == str(tmp_path / "photos")
    assert loaded["teachers"]["eye"]["checkpoint"] == str(tmp_path / "weights/model.keras")


@pytest.mark.parametrize("field", ["work_dir", "images", "config"])
def test_local_material_stays_outside_git_checkout(tmp_path, monkeypatch, field):
    repo = tmp_path / "repo"
    monkeypatch.setattr(common, "REPO", repo)
    cfg = configuration(tmp_path)
    path = tmp_path / "kiosk.local.json"
    if field == "work_dir":
        cfg["work_dir"] = str(repo / "run")
    elif field == "images":
        cfg["sources"][0]["image_dir"] = str(repo / "photos")
    else:
        path = repo / "kiosk.local.json"
    with pytest.raises(ValueError, match="git repository"):
        common.normalize_config(cfg, path)


@pytest.mark.parametrize("change", ["overlap", "duplicate_source", "nan_threshold", "boolean_seed"])
def test_bad_data_contract_is_rejected(tmp_path, change):
    cfg = configuration(tmp_path)
    if change == "overlap":
        cfg["sources"][0]["image_dir"] = str(tmp_path)
    elif change == "duplicate_source":
        cfg["sources"].append(dict(cfg["sources"][0]))
    elif change == "nan_threshold":
        cfg["pseudo_labels"]["min_confidence"] = float("nan")
    else:
        cfg["split"]["seed"] = True
    with pytest.raises(ValueError):
        common.normalize_config(cfg, tmp_path / "kiosk.local.json")


def test_init_validates_before_writing_and_preserves_existing_file(tmp_path):
    path = tmp_path / "kiosk.local.json"
    args = terminal.build_parser().parse_args([
        "--config", str(path), "init", "--image-dir", str(tmp_path / "images"),
        "--data-root", str(tmp_path / "shared"), "--source-id", "../escape",
    ])
    with pytest.raises(ValueError, match="source_id"):
        terminal.initialize(args)
    assert not path.exists()
    args.source_id = "eye"
    terminal.initialize(args)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="이미 있습니다"):
        terminal.initialize(args)
    assert path.read_bytes() == before


def test_results_imports_no_training_runtime_and_starts_no_model(tmp_path):
    cfg_path = tmp_path / "kiosk.local.json"
    atomic_json(cfg_path, configuration(tmp_path))
    code = """import sys
from mediflow_medvlm.kiosk.terminal import main
assert main(['--config', sys.argv[1], '--json', 'results']) == 0
assert 'torch' not in sys.modules and 'tensorflow' not in sys.modules
"""
    env = dict(terminal.os.environ, PYTHONPATH=str(terminal.REPO / "src"))
    result = subprocess.run([sys.executable, "-c", code, str(cfg_path)], env=env,
                            text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["runs"] == []
    assert not (tmp_path / "work").exists()


def test_running_receipt_does_not_claim_liveness_and_csv_has_real_event_fields(tmp_path):
    cfg = configuration(tmp_path)
    run = tmp_path / "work/runs/pilot"
    output = run / "training"
    training_config = run / "train.json"
    atomic_json(training_config, {"max_optimizer_steps": 20})
    atomic_json(run / "prepared.json", {"output_dir": str(output), "training_config": str(training_config)})
    atomic_json(output / "manifest.json", {"status": "RUNNING", "optimizer_steps": 0})
    events = [{"event": "optimizer_update", "step": 3, "loss": 0.72, "lr": 0.0001,
               "at": "2026-11-26T00:00:00+00:00"}]
    common.write_jsonl(output / "events.jsonl", events)
    state = terminal.run_status(cfg, "pilot")
    assert state["steps"] == 3 and state["loss"] == 0.72
    assert state["live_worker_verified"] is False
    assert "저장된 상태" in state["note"]
    result = terminal.results(cfg, "pilot", save=True)
    with (Path(result["report_dir"]) / "training_loss.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    assert rows == [{"step": "3", "loss": "0.72", "learning_rate": "0.0001",
                     "timestamp": events[0]["at"]}]


def test_no_observed_loss_cannot_produce_plot(tmp_path):
    cfg = configuration(tmp_path)
    run = tmp_path / "work/runs/pilot"
    atomic_json(run / "prepared.json", {"output_dir": str(run / "training"), "training_config": str(run / "train.json")})
    with pytest.raises(ValueError, match="기록이 없어"):
        terminal.results(cfg, "pilot", plot=True)
    assert not (run / "report/training_loss.png").exists()


def test_waits_for_own_controller_after_repeated_ctrl_c():
    class Child:
        def __init__(self):
            self.waits, self.signals = 0, []
        def wait(self):
            self.waits += 1
            if self.waits < 3:
                raise KeyboardInterrupt
            return 130
        def poll(self):
            return None
        def send_signal(self, value):
            self.signals.append(value)
    child = Child()
    assert terminal._wait_foreground(child) == 130
    assert child.waits == 3 and child.signals == [signal.SIGINT]


def test_zero_exit_code_with_failed_manifest_is_not_success(tmp_path, monkeypatch):
    cfg = configuration(tmp_path)
    receipt = tmp_path / "failed.json"
    atomic_json(receipt, {"status": "FAILED_REPRODUCIBLE"})
    seen = {}
    class Child:
        pid, returncode = 123, 0
        def poll(self): return 0
        def wait(self): return 0
    def popen(argv, **kwargs):
        seen.update(argv=argv, **kwargs)
        return Child()
    monkeypatch.setattr(terminal.subprocess, "Popen", popen)
    result = terminal._execute(cfg, "pilot", ["python", "dummy"], success_file=receipt, phase="evaluate")
    assert result["success"] is False and result["exit_code"] == 0
    assert seen["shell"] is False and seen["start_new_session"] is True
    assert seen["argv"][0] == sys.executable


def test_event_tail_ignores_only_incomplete_last_line(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text('{"step": 1}\n{"step":', encoding="utf-8")
    assert terminal._last_events(path) == [{"step": 1}]
    path.write_text('{broken}\n{"step": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        terminal._last_events(path)
    path.write_text('{broken}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        terminal._last_events(path)


@pytest.mark.parametrize("source_id,expected", [("kiosk_eye", (8, 12)), ("historical_vqa", (12, 8))])
def test_kiosk_exif_orientation_matches_teacher_without_changing_legacy(tmp_path, monkeypatch, source_id, expected):
    from PIL import Image
    from mediflow_medvlm.training import rendering
    path = tmp_path / "rotated.jpg"
    photo = Image.new("RGB", (12, 8), "red")
    exif = Image.Exif()
    exif[274] = 6
    photo.save(path, exif=exif)
    seen = []
    def encode(processor, messages, *, generation_prompt, images):
        seen.extend(image.size for image in images)
        assert "SECRET" not in json.dumps(messages)
        return {"input_ids": SimpleNamespace(shape=(1, 4))}
    monkeypatch.setattr(rendering, "native_encode", encode)
    rendering.encode_sample({"sample_id": "fixture", "source_id": source_id, "task_type": "medical_vqa",
                             "input": {"question": "classify", "images": [str(path)]},
                             "supervision": {"answer": "SECRET"}}, object(), {}, include_labels=False)
    assert seen == [expected]
