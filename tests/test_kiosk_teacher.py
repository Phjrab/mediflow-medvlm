"""Classifier contracts and provenance using real tiny image inventories.

Model adapters are exercised with runtime stubs; these tests do not claim that
private checkpoints or TensorFlow/PyTorch packages have been validated here.
"""
from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from PIL import Image
import pytest

from mediflow_medvlm.kiosk import dataset, teacher
from mediflow_medvlm.kiosk.common import read_jsonl, write_jsonl
from mediflow_medvlm.runtime import canonical_hash, file_hash


@pytest.fixture
def environment(tmp_path):
    image_dir = tmp_path / "photos"
    image_dir.mkdir()
    Image.new("RGB", (3, 2), (255, 64, 0)).save(image_dir / "first.png")
    Image.new("RGB", (4, 3), (0, 64, 255)).save(image_dir / "second.png")
    checkpoint = tmp_path / "model.keras"
    checkpoint.write_bytes(b"test checkpoint placeholder; runtime is stubbed")
    config = {
        "work_dir": str(tmp_path / "run"),
        "sources": [{"source_id": "eye", "modality": "external_eye_rgb", "image_dir": str(image_dir)}],
        "split": {"train": 0.8, "validation": 0.1, "test": 0.1, "seed": 1},
        "teachers": {"eye": {
            "backend": "keras", "modality": "external_eye_rgb", "checkpoint": str(checkpoint),
            "class_names": ["class_a", "class_b"], "input_size": [2, 3],
            "preprocessing": "raw_0_255", "color_order": "rgb", "output_type": "probabilities",
            "device": "cpu",
        }},
    }
    dataset.scan(config)
    return config, dataset.load_inventory(config, verify_images=True)


def imported_config(environment, tmp_path, *, missing_last=False):
    config, images = deepcopy(environment)
    path = tmp_path / "external_predictions.jsonl"
    records = [{"image_id": image["image_id"], "image_sha256": image["sha256"],
                "scores": {"class_a": 0.98, "class_b": 0.02}} for image in images]
    write_jsonl(path, records[:-1] if missing_last else records)
    config["teachers"]["eye"] = {
        "backend": "predictions", "modality": "external_eye_rgb", "predictions_file": str(path),
        "class_names": ["class_a", "class_b"], "output_type": "probabilities",
    }
    return config, images


def test_inspection_does_not_import_model_runtimes(environment, monkeypatch):
    config, _ = environment
    monkeypatch.setitem(sys.modules, "tensorflow", None)
    monkeypatch.setitem(sys.modules, "torch", None)
    result = teacher.inspect_teachers(config)
    assert result["status"] == "READY"
    assert result["model_runtime_loaded"] is False
    signature = result["teachers"][0]["teacher"]["signature"]
    Path(config["teachers"]["eye"]["checkpoint"]).write_bytes(b"different model")
    assert teacher.inspect_teachers(config)["teachers"][0]["teacher"]["signature"] != signature


@pytest.mark.parametrize("change", [
    {"modality": "fundus"}, {"class_names": ["same", "same"]}, {"class_names": ["one"]},
    {"class_names": {"0": "a", "1": "b"}}, {"preprocessing": None}, {"color_order": None},
    {"input_size": [True, 3]}, {"output_type": "sigmoid"}, {"device": "cuda"},
    {"input_layout": "nchw"},
])
def test_incomplete_or_ambiguous_teacher_is_blocked(environment, change):
    config, _ = deepcopy(environment)
    config["teachers"]["eye"].update(change)
    result = teacher.inspect_teachers(config)
    assert result["status"] == "BLOCKED_CONFIGURATION"
    assert result["teachers"][0]["errors"]


def test_class_map_file_is_explicit_and_hashed(environment, tmp_path):
    config, _ = deepcopy(environment)
    names = tmp_path / "classes.json"
    names.write_text('["class_a", "class_b"]')
    spec = config["teachers"]["eye"]
    spec["class_names_file"] = str(names)
    assert teacher.inspect_teachers(config)["status"] == "BLOCKED_CONFIGURATION"
    spec.pop("class_names")
    identity = teacher.inspect_teachers(config)["teachers"][0]["teacher"]
    assert identity["class_names_file_sha256"] == file_hash(names)


@pytest.mark.parametrize("scores", [[float("nan"), 0.5], [float("inf"), 0.1], [0.5],
                                       [0.8, 0.8], [-0.2, 1.2], [False, 1.0],
                                       {"class_a": 1.0, "wrong_class": 0.0}])
def test_invalid_probability_vectors_are_rejected(scores):
    with pytest.raises(ValueError):
        teacher._probabilities(scores, ["class_a", "class_b"], "probabilities")


def test_logits_stable_and_probabilities_not_softmaxed():
    logits = teacher._probabilities([10000, 9999], ["a", "b"], "logits")
    assert logits["a"] == pytest.approx(0.7310585786)
    probabilities = teacher._probabilities([0.9, 0.1], ["a", "b"], "probabilities")
    assert probabilities["a"] == pytest.approx(0.9)
    assert sum(logits.values()) == pytest.approx(1.0)


@pytest.mark.parametrize("method,order,expected", [
    ("raw_0_255", "rgb", [255, 64, 0]), ("zero_one", "bgr", [0, 64 / 255, 1]),
    ("minus_one_one", "rgb", [1, 64 / 127.5 - 1, -1]),
    ("imagenet", "rgb", [(1 - 0.485) / 0.229, (64 / 255 - 0.456) / 0.224, -0.406 / 0.225]),
])
def test_preprocessing_is_explicit(environment, method, order, expected):
    config, _ = environment
    spec = dict(config["teachers"]["eye"], preprocessing=method, color_order=order, input_size=[5, 7])
    path = Path(config["sources"][0]["image_dir"]) / "first.png"
    values = teacher._prepare_image(path, spec, file_hash(path))
    assert values.shape == (5, 7, 3)
    assert values.dtype == np.float32 and values.flags.c_contiguous
    np.testing.assert_allclose(values[0, 0], expected, atol=1e-6)
    with pytest.raises(ValueError, match="IMAGE_BYTES_CHANGED"):
        teacher._prepare_image(path, spec, "not-the-input-hash")


def test_keras_contract_uses_cpu_safe_load_and_nhwc(environment, monkeypatch):
    config, images = environment
    spec, _ = teacher._specification(config["sources"][0], config["teachers"]["eye"])
    model = Mock(return_value=np.asarray([[0.8, 0.2]], dtype=np.float32))
    model.input_shape = (None, 2, 3, 3)
    model.output_shape = (None, 2)
    load = Mock(return_value=model)
    visible = Mock()
    tf = SimpleNamespace(config=SimpleNamespace(set_visible_devices=visible),
                         keras=SimpleNamespace(models=SimpleNamespace(load_model=load)))
    monkeypatch.setitem(sys.modules, "tensorflow", tf)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    predictor = teacher._load_predictor(spec)
    scores = predictor(images[0]["path"], images[0]["sha256"])
    load.assert_called_once_with(spec["checkpoint"], compile=False, safe_mode=True)
    visible.assert_called_once_with([], "GPU")
    assert model.call_args.args[0].shape == (1, 2, 3, 3)
    assert model.call_args.kwargs == {"training": False}
    assert scores["class_a"] == pytest.approx(0.8)
    import os
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"


@pytest.mark.parametrize("input_shape,output_shape", [
    ((None, 3, 2, 3), (None, 2)), ((None, 2, 3, 1), (None, 2)),
    ([(None, 2, 3, 3), (None, 1)], (None, 2)), ((None, 2, 3, 3), (None, 1)),
])
def test_keras_wrong_model_contract_is_rejected(environment, input_shape, output_shape):
    config, _ = environment
    with pytest.raises(ValueError):
        teacher._validate_keras_shape(SimpleNamespace(input_shape=input_shape, output_shape=output_shape),
                                     config["teachers"]["eye"])


def test_legacy_keras_loader_without_safe_mode_can_load_trusted_model(environment, monkeypatch):
    config, images = environment
    spec = config["teachers"]["eye"]
    loaded = []
    model = Mock(return_value=np.asarray([[0.8, 0.2]]))
    model.input_shape, model.output_shape = (None, 2, 3, 3), (None, 2)

    def legacy_load(filepath, custom_objects=None, compile=True):
        loaded.append((filepath, custom_objects, compile))
        return model

    tf = SimpleNamespace(config=SimpleNamespace(set_visible_devices=Mock()),
                         keras=SimpleNamespace(models=SimpleNamespace(load_model=legacy_load)))
    monkeypatch.setitem(sys.modules, "tensorflow", tf)
    scores = teacher._load_predictor(spec)(images[0]["path"], images[0]["sha256"])
    assert loaded == [(spec["checkpoint"], None, False)]
    assert scores["class_a"] == pytest.approx(0.8)


def test_torchscript_adapter_uses_jit_cpu_and_nchw(environment, monkeypatch):
    config, images = environment
    spec = dict(config["teachers"]["eye"], backend="torchscript", output_type="logits")

    class Tensor:
        def __init__(self, data):
            self.data = np.asarray(data)
            self.shape = self.data.shape

        def unsqueeze(self, axis):
            return Tensor(np.expand_dims(self.data, axis))

        def detach(self):
            return self

        float = cpu = detach

        def numpy(self):
            return self.data

    model = Mock(return_value=Tensor([[10000, 9999]]))
    load = Mock(return_value=model)
    torch = SimpleNamespace(jit=SimpleNamespace(load=load), Tensor=Tensor,
                            from_numpy=Tensor, inference_mode=nullcontext)
    monkeypatch.setitem(sys.modules, "torch", torch)
    predictor = teacher._load_predictor(spec)
    scores = predictor(images[0]["path"], images[0]["sha256"])
    load.assert_called_once_with(spec["checkpoint"], map_location="cpu")
    model.eval.assert_called_once_with()
    assert model.call_args.args[0].shape == (1, 3, 2, 3)
    assert scores["class_a"] == pytest.approx(0.7310585786)


def test_import_end_to_end_preserves_provenance_and_can_reuse(environment, tmp_path):
    config, images = imported_config(environment, tmp_path)
    updates = []
    result = teacher.label(config, progress=updates.append)
    assert result["status"] == "SUCCEEDED"
    assert result["counts"] == {"total": 2, "predicted": 2, "failed": 0}
    assert result["teacher_config_sha256"] == canonical_hash(config["teachers"])
    assert result["inventory_sha256"] == file_hash(Path(config["work_dir"]) / "inventory.jsonl")
    assert result["predictions_sha256"] == file_hash(result["predictions_file"])
    assert result["clinical_truth_created"] is False
    rows = read_jsonl(result["predictions_file"])
    assert {row["image_id"] for row in rows} == {image["image_id"] for image in images}
    assert all(row["teacher"]["signature"] == result["teacher_signatures"]["eye"] for row in rows)
    assert rows[0]["confidence"] == pytest.approx(0.98)
    assert rows[0]["margin"] == pytest.approx(0.96)
    assert updates[0]["status"] == "RUNNING" and updates[-1]["status"] == "SUCCEEDED"
    assert teacher.label(config)["reused"] is True
    predictions, _ = dataset._predictions(config, images)
    assert len(predictions) == 2


def test_missing_import_scores_are_visible_partial_failures(environment, tmp_path):
    config, _ = imported_config(environment, tmp_path, missing_last=True)
    result = teacher.label(config)
    assert result["status"] == "PARTIAL"
    assert result["counts"]["failed"] == 1
    rows = read_jsonl(result["predictions_file"])
    failed = next(row for row in rows if row["status"] == "failed")
    assert failed["error_code"] == "MISSING_IMPORTED_PREDICTION"
    assert failed["scores"] == {} and failed["predicted_label"] is None
    teacher.label(config)
    assert len(list((Path(config["work_dir"]) / "label_attempts").iterdir())) == 1


@pytest.mark.parametrize("mutation", ["hash", "unknown_id", "duplicate", "label_only", "wrong_source"])
def test_imported_scores_need_image_and_class_identity(environment, tmp_path, mutation):
    config, images = imported_config(environment, tmp_path)
    spec = config["teachers"]["eye"]
    rows = read_jsonl(spec["predictions_file"])
    if mutation == "hash":
        rows[0]["image_sha256"] = "stale"
    elif mutation == "unknown_id":
        rows[0]["image_id"] = "unknown"
    elif mutation == "duplicate":
        rows.append(rows[0])
    elif mutation == "wrong_source":
        rows[0]["source_id"] = "another_camera"
    else:
        rows[0].pop("scores")
        rows[0]["predicted_label"] = "class_a"
    write_jsonl(spec["predictions_file"], rows)
    with pytest.raises(ValueError):
        teacher._import_predictions(spec, images)
    assert teacher.label(config)["status"] == "FAILED"


def test_replaced_input_or_model_cannot_reuse_old_predictions(environment, tmp_path):
    config, images = imported_config(environment, tmp_path)
    teacher.label(config)
    path = Path(config["teachers"]["eye"]["predictions_file"])
    rows = read_jsonl(path)
    rows[0]["scores"] = {"class_a": 0.1, "class_b": 0.9}
    write_jsonl(path, rows)
    with pytest.raises(ValueError, match="identity changed"):
        teacher.label(config)
    Path(images[0]["path"]).write_bytes(b"changed image")
    with pytest.raises(ValueError, match="Image bytes changed"):
        teacher.label(config)


def test_tampered_generated_output_is_rejected(environment, tmp_path):
    config, _ = imported_config(environment, tmp_path)
    result = teacher.label(config)
    with Path(result["predictions_file"]).open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="modified"):
        teacher.label(config)


def test_missing_runtime_does_not_report_success(environment, monkeypatch):
    config, _ = environment
    monkeypatch.setattr(teacher, "_load_predictor", Mock(side_effect=ImportError("missing runtime")))
    result = teacher.label(config)
    assert result["status"] == "FAILED"
    assert result["counts"]["predicted"] == 0
    assert all(row["error_code"] == "DEPENDENCY_MISSING" for row in read_jsonl(result["predictions_file"]))


def test_model_replacement_during_inference_is_not_published(environment, monkeypatch):
    config, _ = environment

    def predict(path, digest):
        Path(config["teachers"]["eye"]["checkpoint"]).write_bytes(b"changed mid-run")
        return {"class_a": 0.8, "class_b": 0.2}

    monkeypatch.setattr(teacher, "_load_predictor", lambda spec: predict)
    with pytest.raises(ValueError, match="changed while labeling"):
        teacher.label(config)
    assert not (Path(config["work_dir"]) / "predictions.jsonl").exists()
    progress = json.loads((Path(config["work_dir"]) / "label_progress.json").read_text())
    assert progress["status"] == "INTERRUPTED_OR_FAILED"


def test_label_workspace_lock_prevents_competing_processes(tmp_path):
    with teacher._label_lock(tmp_path / "label.lock"):
        with pytest.raises(RuntimeError, match="Another label process"):
            with teacher._label_lock(tmp_path / "label.lock"):
                pytest.fail("second label process acquired the same workspace")
