"""CPU contract checks; actual VLM generation is stubbed, never claimed as GPU evidence."""
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

from mediflow_medvlm.kiosk import inference
from mediflow_medvlm.kiosk.dataset import classification_question
from mediflow_medvlm.runtime import atomic_json, canonical_hash, directory_digest, file_hash


@pytest.fixture
def saved_run(tmp_path):
    work, data = tmp_path / "kiosk", tmp_path / "shared-data"
    run = work / "runs" / "pilot"
    output = run / "output"
    checkpoint = output / "checkpoints" / "step-001"
    checkpoint.mkdir(parents=True)
    config = {"run_id": "kiosk-pilot", "pair_id": "pair", "experiment_id": "fixture",
              "model_id": "fixture/vlm", "revision": "a" * 40,
              "train_file": str(tmp_path / "absent-train.jsonl"),
              "validation_file": str(tmp_path / "absent-validation.jsonl"),
              "freeze_manifest": str(tmp_path / "absent-freeze.json"),
              "output_dir": str(output), "data_root": str(data),
              "device": "cuda", "quantization": "none", "native_snapshot_path": "/fixture/base",
              "resource_policy": {"budget_mode": "unlimited"}}
    config_hash = canonical_hash(config)
    identity = {"config_sha256": config_hash, "model_id": config["model_id"], "revision": config["revision"],
                "quantization": "none", "template_sha256": canonical_hash("saved-template")}
    weight = checkpoint / "adapter_model.safetensors"
    weight.write_bytes(b"fixture-not-a-real-model")
    atomic_json(checkpoint / "checkpoint.json", {"complete": True, "identity": identity,
                                                "artifacts": {"files": {weight.name: file_hash(weight)}}})
    atomic_json(run / "training.json", config)
    atomic_json(output / "manifest.json", {"status": "SUCCEEDED", "config_hash": config_hash,
                                          "identity": identity, "model": {"native_files": {"sha256": "base-sha"}}})
    atomic_json(output / "selection.json", {"checkpoint": str(checkpoint), "reload_verified": True,
                                           "identity": identity, "adapter": {"digest": directory_digest(checkpoint)["sha256"]}})
    atomic_json(run / "prepared.json", {"training_config": str(run / "training.json"),
                                       "training_config_sha256": config_hash, "output_dir": str(output)})
    # This photo has never appeared in a scan, review CSV, or labeled dataset.
    image = tmp_path / "new-photo.png"
    Image.new("RGB", (3, 2), (42, 128, 200)).save(image)
    kiosk = {"work_dir": str(work), "data_root": str(data), "sources": [
        {"source_id": "eye", "modality": "external_eye_rgb", "image_dir": str(tmp_path / "unused-inventory")}]}
    return kiosk, image, config


def runtime_stub(monkeypatch, *, with_pixels=True, cuda_available=True):
    from mediflow_medvlm.training import models, rendering
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=9,
                                decode=Mock(return_value="class_a"), chat_template="saved-template")
    processor = SimpleNamespace(tokenizer=tokenizer, chat_template="saved-template")
    model = SimpleNamespace(device="cuda", config=SimpleNamespace(), eval=Mock(),
                            generation_config=SimpleNamespace(eos_token_id=9),
                            generate=Mock(return_value=np.asarray([[1, 2, 7, 9]])))
    metadata = {"native_files": {"sha256": "base-sha"}, "chat_template_sha256": canonical_hash("saved-template")}
    load = Mock(return_value=(model, processor, metadata))
    saved_processor = Mock(return_value=processor)
    seen = []

    def encode(sample, *args, **kwargs):
        seen.append(deepcopy(sample))
        assert kwargs["include_labels"] is False
        encoded = {"input_ids": np.asarray([[1, 2]])}
        if with_pixels:
            encoded["pixel_values"] = np.ones((1, 3, 2, 3))
        return encoded, {"images": [{"sha256": "fixture-pixels"}]}

    cuda = SimpleNamespace(is_available=lambda: cuda_available, synchronize=lambda: None,
                           get_device_name=lambda: "CPU test stub", get_device_properties=lambda _: SimpleNamespace(total_memory=1),
                           reset_peak_memory_stats=lambda: None, max_memory_allocated=lambda: 0,
                           max_memory_reserved=lambda: 0, empty_cache=lambda: None)
    torch = SimpleNamespace(cuda=cuda, version=SimpleNamespace(cuda="stub"),
                            inference_mode=nullcontext, OutOfMemoryError=type("StubOOM", (Exception,), {}))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(models, "load_model", load)
    monkeypatch.setattr(models, "load_processor", saved_processor)
    monkeypatch.setattr(rendering, "encode_sample", encode)
    return SimpleNamespace(model=model, processor=processor, load=load, saved_processor=saved_processor, seen=seen)


def test_prepare_needs_no_gold_dataset_or_model_runtime(saved_run, monkeypatch):
    config, image, training = saved_run
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    result = inference.prepare_prediction(config, "pilot", image, "eye")
    request = inference._request(result["request_file"])
    assert result["gpu_started"] is False
    assert result["argv"][-2:] == ["--request", result["request_file"]]
    assert request["question"] == classification_question("external_eye_rgb")
    assert request["image"]["sha256"] == file_hash(image)
    assert not Path(training["train_file"]).exists()
    assert not Path(training["validation_file"]).exists()
    assert "supervision" not in inference._inference_sample(request)


@pytest.mark.parametrize("field,value", [("source", "unknown"), ("question", "  "), ("name", "../escape")])
def test_prepare_rejects_invalid_user_inputs(saved_run, field, value):
    config, image, _ = saved_run
    with pytest.raises(ValueError):
        inference.prepare_prediction(config, value if field == "name" else "pilot", image,
                                     value if field == "source" else "eye", value if field == "question" else None)


def test_new_image_prediction_uses_saved_adapter_processor_and_shared_ledger(saved_run, monkeypatch):
    config, image, _ = saved_run
    prepared = inference.prepare_prediction(config, "pilot", image, "eye")
    stub = runtime_stub(monkeypatch)
    result = inference.predict_request(prepared["request_file"])
    assert result["status"] == "SUCCEEDED"
    assert result["answer"] == "class_a" and result["generated_tokens"] == 2
    assert result["clinical_truth_created"] is False and result["reference_label_used"] is False
    assert result["review_status"] == "unreviewed_model_output"
    assert result["vision_ingested"] is True
    assert stub.load.call_args.args[0]["local_files_only"] is True
    assert stub.load.call_args.kwargs["training"] is False
    saved = stub.saved_processor.call_args.args[0]
    assert saved["native_snapshot_path"] is None
    assert saved["model_id"] == result["checkpoint"]
    assert stub.seen[0]["source_id"] == "kiosk_eye"
    assert set(stub.seen[0]) == {"sample_id", "source_id", "task_type", "input"}
    assert set(stub.seen[0]["input"]) == {"question", "images"}
    assert stub.model.generate.call_args.kwargs["do_sample"] is False
    ledger = json.loads((Path(config["data_root"]) / "runtime" / "gpu_budget.json").read_text())
    assert ledger["active"] is None and ledger["sessions"][-1]["phase"] == "prediction"
    assert json.loads(Path(prepared["result_file"]).read_text())["answer"] == "class_a"
    with pytest.raises(ValueError, match="already exists"):
        inference.predict_request(prepared["request_file"])


@pytest.mark.parametrize("mutation", ["image", "checkpoint", "config", "selection"])
def test_changed_inputs_are_rejected_before_loading(saved_run, monkeypatch, mutation):
    config, image, _ = saved_run
    prepared = inference.prepare_prediction(config, "pilot", image, "eye")
    request = inference._request(prepared["request_file"])
    stub = runtime_stub(monkeypatch)
    if mutation == "image":
        image.write_bytes(b"different input")
    elif mutation == "checkpoint":
        (Path(request["checkpoint"]) / "adapter_model.safetensors").write_bytes(b"changed model")
    else:
        path = Path(request["run_dir"]) / ("training.json" if mutation == "config" else "output/selection.json")
        content = json.loads(path.read_text())
        content["changed"] = True
        atomic_json(path, content)
    result = inference.predict_request(prepared["request_file"])
    assert result["status"] == "FAILED_REPRODUCIBLE" and result["answer"] is None
    stub.load.assert_not_called()


@pytest.mark.parametrize("kind", ["no_cuda", "no_pixels", "midrun_image_change", "template_mismatch"])
def test_inference_failures_never_claim_an_answer(saved_run, monkeypatch, kind):
    config, image, _ = saved_run
    prepared = inference.prepare_prediction(config, "pilot", image, "eye")
    stub = runtime_stub(monkeypatch, with_pixels=kind != "no_pixels", cuda_available=kind != "no_cuda")
    if kind == "midrun_image_change":
        def change_image(**kwargs):
            image.write_bytes(b"modified while generating")
            return np.asarray([[1, 2, 7, 9]])
        stub.model.generate.side_effect = change_image
    elif kind == "template_mismatch":
        stub.processor.chat_template = "changed-template"
    result = inference.predict_request(prepared["request_file"])
    assert result["status"] != "SUCCEEDED" and result["answer"] is None
    assert result["clinical_truth_created"] is False
    if kind == "no_cuda":
        stub.load.assert_not_called()


def test_request_tampering_is_rejected(saved_run):
    config, image, _ = saved_run
    prepared = inference.prepare_prediction(config, "pilot", image, "eye")
    path = Path(prepared["request_file"])
    request = json.loads(path.read_text())
    request["question"] = "changed after preparation"
    atomic_json(path, request)
    with pytest.raises(ValueError, match="identity changed"):
        inference.predict_request(path)
    assert not Path(prepared["result_file"]).exists()


def test_evaluation_preserves_kiosk_orientation_source_without_gold_leak(saved_run, monkeypatch, tmp_path):
    """Exercise both real evaluate() projection sites, using stubbed model tensors."""
    from mediflow_medvlm.evaluation import runner
    from mediflow_medvlm.training import config as training_config, runner as training_runner
    _, image, config = saved_run
    stub = runtime_stub(monkeypatch)
    row = {"sample_id": "test-photo", "source_id": "kiosk_eye", "task_type": "medical_vqa", "split": "test",
           "language": "ko", "input": {"question": "Classify this photo.", "images": [str(image)]},
           "supervision": {"answer": "DO_NOT_PASS_GOLD_TO_ENCODER"}, "group_ids": {"patient": "fixture"}}
    validation = {**row, "sample_id": "validation-photo", "split": "validation"}
    monkeypatch.setattr(training_config, "load_frozen_data", lambda config: ({"validation": [validation]}, {}))
    monkeypatch.setattr(training_runner, "_git_identity", lambda: {})
    data_path, protocol_path, config_path = tmp_path / "test.jsonl", tmp_path / "protocol.json", tmp_path / "evaluation.json"
    data_path.write_text(json.dumps(row) + "\n")
    runner.freeze_protocol([data_path], protocol_path)
    atomic_json(config_path, config)
    result = runner.evaluate(config_path, protocol_path, tmp_path / "evaluation")
    assert result["status"] == "SUCCEEDED"
    assert len(stub.seen) == 2
    assert {sample["sample_id"] for sample in stub.seen} == {"test-photo", "validation-photo"}
    assert all(sample["source_id"] == "kiosk_eye" and "supervision" not in sample for sample in stub.seen)
