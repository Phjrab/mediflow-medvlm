"""Model setup contracts using tiny bytes; never download or load medical models."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import urllib.error

import pytest

from mediflow_medvlm.kiosk import setup_models as setup


@pytest.fixture
def tiny_models(monkeypatch):
    payload = b"non-model fixture bytes"
    models = deepcopy(setup.MODELS)
    for spec in models.values():
        spec["bytes"] = len(payload)
        spec["sha256"] = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(setup, "MODELS", models)
    return payload


def test_download_verifies_and_reuses_only_exact_bytes(tmp_path, monkeypatch, tiny_models):
    calls = []
    def open_request(request, *, timeout):
        calls.append(request.full_url)
        assert timeout == 60
        return io.BytesIO(tiny_models)
    monkeypatch.setattr(setup.urllib.request, "urlopen", open_request)
    downloaded = setup.download_model("eye", tmp_path)
    assert downloaded["status"] == "DOWNLOADED_VERIFIED"
    assert Path(downloaded["path"]).read_bytes() == tiny_models
    assert len(calls) == 1 and setup.EYE_REVISION in calls[0]
    assert setup.download_model("eye", tmp_path)["status"] == "REUSED_VERIFIED"
    assert len(calls) == 1
    Path(downloaded["path"]).write_bytes(b"user's different model")
    with pytest.raises(ValueError, match="differs"):
        setup.download_model("eye", tmp_path)
    assert Path(downloaded["path"]).read_bytes() == b"user's different model"
    assert len(calls) == 1


@pytest.mark.parametrize("failure", ["wrong_hash", "too_large", "interrupted_transfer"])
def test_bad_or_interrupted_download_never_publishes_partial_file(tmp_path, monkeypatch, tiny_models, failure):
    class BrokenStream(io.BytesIO):
        def read(self, _size):
            if self.tell():
                raise urllib.error.URLError("fixture disconnected")
            return super().read(3)
    response = (BrokenStream(tiny_models) if failure == "interrupted_transfer" else
                io.BytesIO(tiny_models + b"extra" if failure == "too_large" else b"x" * len(tiny_models)))
    monkeypatch.setattr(setup.urllib.request, "urlopen", lambda *args, **kwargs: response)
    with pytest.raises((ValueError, urllib.error.URLError)):
        setup.download_model("eye", tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_racing_destination_is_not_replaced(tmp_path, monkeypatch, tiny_models):
    monkeypatch.setattr(setup.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(tiny_models))
    def race(source, destination):
        Path(destination).write_bytes(b"do not overwrite")
        raise FileExistsError("fixture race")
    monkeypatch.setattr(setup.os, "link", race)
    with pytest.raises(ValueError, match="differs"):
        setup.download_model("eye", tmp_path)
    assert (tmp_path / "eye_effnet_b0.pth").read_bytes() == b"do not overwrite"
    assert not list(tmp_path.glob(".*download*"))


def config_args(tmp_path, tiny_models, *, scalp="public6", eye=True):
    assets = tmp_path / "models"
    assets.mkdir(exist_ok=True)
    for spec in setup.MODELS.values():
        (assets / spec["filename"]).write_bytes(tiny_models)
    eye_dir, scalp_dir = tmp_path / "eye-photos", tmp_path / "scalp-photos"
    eye_dir.mkdir(exist_ok=True)
    scalp_dir.mkdir(exist_ok=True)
    argv = ["configure", "--config", str(tmp_path / "kiosk.local.json"), "--assets", str(assets),
            "--data-root", str(tmp_path / "data"), "--scalp-dir", str(scalp_dir), "--scalp", scalp]
    if eye:
        argv.extend(["--eye-dir", str(eye_dir), "--eye-input", "single_eye_crop"])
    return setup.build_parser().parse_args(argv)


def test_config_uses_verified_models_exact_classes_and_camera_contracts(tmp_path, tiny_models):
    args = config_args(tmp_path, tiny_models)
    result = setup.configure(args)
    config = json.loads(Path(result["config"]).read_text())
    assert result["training_paths_configured"] is False
    assert config["initialization_adapter"] is None and config["start_from_base"] is False
    assert [source["modality"] for source in config["sources"]] == ["external_eye_rgb", "scalp_microscope"]
    eye, scalp = config["teachers"]["eye"], config["teachers"]["scalp"]
    assert eye["backend"] == "mediflow_eye" and eye["input_scope"] == "single_eye_crop"
    assert eye["input_size"] == [224, 224] and eye["preprocessing"] == "imagenet"
    assert eye["output_type"] == "logits" and eye["class_names"] == setup.EYE_CLASSES
    assert scalp["backend"] == "keras" and scalp["image_processor"] == "tensorflow_legacy"
    assert scalp["input_size"] == [384, 384] and scalp["preprocessing"] == "raw_0_255"
    assert scalp["class_names"] == setup.SCALP_CLASSES + ["양호"]
    assert scalp["expected_checkpoint_sha256"] == setup.MODELS["public6"]["sha256"]
    assert not Path(config["work_dir"]).exists()
    before = Path(result["config"]).read_bytes()
    with pytest.raises(ValueError, match="이미 있습니다"):
        setup.configure(args)
    assert Path(result["config"]).read_bytes() == before


def test_camera_five_class_never_invents_normal_class(tmp_path, tiny_models):
    args = config_args(tmp_path, tiny_models, scalp="camera5", eye=False)
    result = setup.configure(args)
    teacher = json.loads(Path(result["config"]).read_text())["teachers"]["scalp"]
    assert teacher["class_names"] == setup.SCALP_CLASSES
    assert "정상 없음" in teacher["model_note"]


@pytest.mark.parametrize("failure", ["eye_scope_missing", "missing_photo", "changed_weight", "unknown_scalp_choice"])
def test_invalid_configuration_does_not_create_file(tmp_path, tiny_models, failure):
    args = config_args(tmp_path, tiny_models)
    if failure == "eye_scope_missing":
        args.eye_input = None
    elif failure == "missing_photo":
        args.eye_dir = str(tmp_path / "missing")
    elif failure == "changed_weight":
        (Path(args.assets) / "eye_effnet_b0.pth").write_bytes(b"changed")
    else:
        args.scalp = None
    with pytest.raises(ValueError):
        setup.configure(args)
    assert not Path(args.config).exists()


def test_assets_cannot_be_downloaded_inside_repository(tmp_path, monkeypatch):
    monkeypatch.setattr(setup, "REPO", tmp_path)
    with pytest.raises(ValueError, match="outside"):
        setup.download_model("eye", tmp_path / "models")
    assert not (tmp_path / "models").exists()


def test_list_and_help_do_not_import_model_runtimes():
    repo = Path(__file__).resolve().parents[1]
    code = """import sys
sys.path.insert(0, sys.argv[1])
from mediflow_medvlm.kiosk.setup_models import main
assert main(['list']) == 0
assert not {'torch', 'torchvision', 'tensorflow', 'keras'} & set(sys.modules)
"""
    process = subprocess.run([sys.executable, "-c", code, str(repo / "src")],
                             text=True, capture_output=True, timeout=10)
    assert process.returncode == 0, process.stderr
    assert len(json.loads(process.stdout)["models"]) == 3
