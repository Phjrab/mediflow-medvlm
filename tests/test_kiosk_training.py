"""CPU-only preparation/receipt checks; fixtures are not medical model weights."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import subprocess
import sys
import types
import urllib.request

import pytest
from PIL import Image

from mediflow_medvlm.kiosk.common import load_config, read_jsonl
from mediflow_medvlm.kiosk.dataset import REVIEW_COLUMNS, export_dataset, load_inventory, scan, write_review_template
from mediflow_medvlm.kiosk.training import prepare_evaluation, prepare_training
from mediflow_medvlm.runtime import atomic_json, canonical_hash, directory_digest, file_hash
from mediflow_medvlm.training.config import assert_resume_identity, load_frozen_data, validate_kiosk_supervision
from mediflow_medvlm.training.initialization import (V6_IDENTITY, V6_SCHEMA, V6_SCOPE,
                                                    assert_initialization_model, verify_initialization)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dataset_fixture(tmp_path, train_count=4, test_reviewed=True):
    images = tmp_path / "photos"
    images.mkdir()
    groups = tmp_path / "groups.csv"
    with groups.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["relative_path", "subject_id", "split"])
        writer.writeheader()
        for i, split in enumerate(["train"] * train_count + ["validation", "test"]):
            name = f"engineering_{i}.png"
            Image.new("RGB", (12 + i, 15), (i, 30, 90)).save(images / name)
            writer.writerow({"relative_path": name, "subject_id": str(i), "split": split})
    path = tmp_path / "kiosk.json"
    atomic_json(path, {"schema_version": "medvlm.kiosk-terminal.v1", "work_dir": str(tmp_path / "work"), "start_from_base": True,
                      "data_root": str(tmp_path / "shared"), "sources": [{"source_id": "eye", "image_dir": str(images),
                      "groups_csv": str(groups), "modality": "external_eye_rgb"}]})
    config = load_config(path)
    scan(config)
    write_review_template(config)
    reviews = Path(config["work_dir"]) / "reviews.csv"
    with reviews.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        if test_reviewed or row["split"] != "test":
            row.update(decision="approved", label="synthetic_class", label_source="human_reviewed", reviewer="fixture_reviewer")
    with reviews.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    export_dataset(config)
    template_path = tmp_path / "template.json"
    template = {"run_id": "previous-vqa", "pair_id": "previous-pair", "experiment_id": "old",
                "model_id": "engineering/native-vlm", "revision": "a" * 40,
                "train_file": "original/train.jsonl", "validation_file": "original/validation.jsonl",
                "freeze_manifest": "original/freeze.json", "image_root": "original/images",
                "output_dir": "original/output", "data_root": "shared", "cache_dir": "shared/hf-cache",
                "quantization": "none", "dtype": "bfloat16", "device": "cuda", "local_files_only": True,
                "gradient_accumulation_steps": 2, "learning_rate": .0001, "lora_rank": 8, "lora_alpha": 16,
                "lora_dropout": .05, "max_optimizer_steps": 100, "resource_policy": {"budget_mode": "unlimited"}}
    atomic_json(template_path, template)
    config["training_template"] = str(template_path)
    return config, template


def standard_parent(tmp_path, kiosk, template):
    run = tmp_path / "parent"
    checkpoint = run / "checkpoints" / "step-000010-fixture"
    checkpoint.mkdir(parents=True)
    train_file = run / "train.jsonl"
    train_file.write_text(json.dumps({"task_type": "medical_vqa", "input": {"images": ["old-photo.png"]}}) + "\n", encoding="utf-8")
    source = {**template, "data_root": kiosk["data_root"], "train_file": str(train_file), "output_dir": str(run)}
    atomic_json(run / "config.json", source)
    identity = {"identity_schema_version": "medvlm.training-identity.v2", "model_id": source["model_id"],
                "revision": source["revision"], "config_sha256": canonical_hash(source),
                "data": {"train": {"sha256": file_hash(train_file)}}, "base_files_sha256": "base-fixture-digest",
                "template_sha256": "template-fixture-digest", "target_modules": ["model.layers.0.self_attn.q_proj"],
                "quantization": source["quantization"], "training_code_sha256": "prior-code", "renderer_version": "prior-renderer"}
    atomic_json(checkpoint / "adapter_config.json", {"peft_type": "LORA", "task_type": "CAUSAL_LM", "bias": "none",
                "base_model_name_or_path": source["model_id"], "revision": source["revision"],
                "target_modules": identity["target_modules"], "r": 8, "lora_alpha": 16, "lora_dropout": .05})
    (checkpoint / "adapter_model.safetensors").write_bytes(b"engineering receipt fixture; not loadable tensors")
    (checkpoint / "training_state.pt").write_bytes(b"must never be unpickled by initialization")
    atomic_json(checkpoint / "checkpoint.json", {"complete": True, "identity": identity, "artifacts": directory_digest(checkpoint)})
    atomic_json(run / "manifest.json", {"identity": identity, "model": {"native_files": {"sha256": identity["base_files_sha256"]}}})
    kiosk["training_template"] = str(run / "config.json")
    kiosk["initialization_adapter"] = str(checkpoint)
    kiosk["start_from_base"] = False
    return checkpoint, source, identity


def forbid_execution(*args, **kwargs):
    raise AssertionError("prepare must not run a subprocess or access the network")


def test_prepare_preserves_native_recipe_shared_root_and_no_worker(tmp_path, monkeypatch):
    config, original = dataset_fixture(tmp_path)
    monkeypatch.setattr(subprocess, "Popen", forbid_execution)
    monkeypatch.setattr(subprocess, "run", forbid_execution)
    monkeypatch.setattr(urllib.request, "urlopen", forbid_execution)
    receipt = prepare_training(config, "eye_stage_1")
    training = read(receipt["training_config"])
    assert receipt["status"] == "PREPARED_NOT_RUN"
    assert receipt["gpu_started"] is False
    assert receipt["train_argv"] == [sys.executable, "-m", "mediflow_medvlm.training", "train", "--config", receipt["training_config"]]
    assert training["data_root"] == config["data_root"]
    assert training["cache_dir"] == str(tmp_path / "shared" / "hf-cache")
    assert not (Path(config["data_root"]) / "runtime").exists()
    assert not Path(receipt["output_dir"]).exists()
    for key in ("model_id", "revision", "device", "dtype", "quantization", "learning_rate", "resource_policy"):
        assert training[key] == original[key]
    assert training["training_recipe"] == "one_pass"
    assert (training["train_samples_cap"], training["max_optimizer_steps"]) == (4, 2)
    assert prepare_training(config, "eye_stage_1") == receipt


def test_new_name_required_after_changed_template_or_review(tmp_path):
    config, _ = dataset_fixture(tmp_path)
    receipt = prepare_training(config, "stable")
    before = file_hash(receipt["training_config"])
    template = read(config["training_template"])
    template["learning_rate"] = .0002
    atomic_json(config["training_template"], template)
    with pytest.raises(ValueError, match="immutable"):
        prepare_training(config, "stable")
    assert file_hash(receipt["training_config"]) == before
    assert prepare_training(config, "new_recipe")["training_config"] != receipt["training_config"]
    review = Path(config["work_dir"]) / "reviews.csv"
    review.write_text(review.read_text(encoding="utf-8-sig").replace("synthetic_class", "corrected_class"), encoding="utf-8-sig")
    with pytest.raises(ValueError, match="reviews_sha256"):
        prepare_training(config, "stale_review")
    assert not (Path(config["work_dir"]) / "runs" / "stale_review").exists()


def test_fresh_base_requires_explicit_choice(tmp_path):
    config, template = dataset_fixture(tmp_path)
    config.pop("start_from_base", None)
    with pytest.raises(ValueError, match="existing VQA checkpoint"):
        prepare_training(config, "implicit_base")
    standard_parent(tmp_path, config, template)
    config["start_from_base"] = True
    with pytest.raises(ValueError, match="not both"):
        prepare_training(config, "ambiguous_parent")


def test_smoke_budget_is_exact_and_requires_real_minimum(tmp_path):
    config, _ = dataset_fixture(tmp_path, train_count=25)
    receipt = prepare_training(config, "smoke10", max_steps=10)
    training = read(receipt["training_config"])
    assert (training["mode"], training["train_samples_cap"], training["max_optimizer_steps"]) == ("smoke", 20, 10)
    with pytest.raises(ValueError, match="at least 10"):
        prepare_training(config, "too_short", max_steps=9)
    with pytest.raises(ValueError, match="eligible training rows"):
        prepare_training(config, "too_large", max_steps=14)
    with pytest.raises(ValueError, match="positive integer"):
        prepare_training(config, "bool_steps", max_steps=True)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "name with space", "", ".pending"])
def test_run_names_cannot_escape_workspace(tmp_path, name):
    config, _ = dataset_fixture(tmp_path)
    with pytest.raises(ValueError, match="run_name"):
        prepare_training(config, name)


def test_bad_revision_and_shared_data_root_fail_before_run_creation(tmp_path):
    config, template = dataset_fixture(tmp_path)
    config["data_root"] = str(tmp_path / "separate_gpu_ledger")
    with pytest.raises(ValueError, match="original MEDVLM_DATA_ROOT"):
        prepare_training(config, "bad_root")
    config["data_root"] = str(tmp_path / "shared")
    template["revision"] = "main"
    atomic_json(config["training_template"], template)
    with pytest.raises(ValueError, match="immutable Hub commit"):
        prepare_training(config, "bad_revision")
    assert not (Path(config["work_dir"]) / "runs").exists()


def test_parent_adapter_starts_new_identity_and_never_reuses_optimizer(tmp_path, monkeypatch):
    config, template = dataset_fixture(tmp_path)
    parent, source, original_identity = standard_parent(tmp_path, config, template)
    receipt = prepare_training(config, "continued")
    training = read(receipt["training_config"])
    provenance = training["initialization_provenance"]
    assert provenance["adapter_path"] == str(parent)
    assert provenance["source_task"] == "medical_vqa"
    assert provenance["optimizer_scheduler_rng_restored"] is False
    assert training["run_id"] != source["run_id"]
    assert training["train_file"] != source["train_file"]
    assert "--resume" not in receipt["train_argv"]
    with pytest.raises(ValueError, match="resume identity mismatch"):
        assert_resume_identity({**original_identity, "config_sha256": canonical_hash(training)}, parent)
    metadata = {"native_files": {"sha256": provenance["base_files_sha256"]},
                "chat_template_sha256": provenance["template_sha256"], "exact_target_modules": provenance["target_modules"]}
    assert_initialization_model(provenance, metadata)
    with pytest.raises(ValueError, match="base_files_sha256"):
        assert_initialization_model(provenance, {**metadata, "native_files": {"sha256": "changed"}})
    # Stop at load_model: this verifies adapter dispatch before an optimizer can
    # be instantiated, with no torch/transformers installation or tensor load.
    from mediflow_medvlm.training import runner
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(get_linear_schedule_with_warmup=forbid_execution))
    loaded = []
    def stop_at_model(*args, **kwargs):
        loaded.append(kwargs)
        raise RuntimeError("STOP_BEFORE_OPTIMIZER")
    monkeypatch.setattr(runner, "load_model", stop_at_model)
    with pytest.raises(RuntimeError, match="STOP_BEFORE_OPTIMIZER"):
        runner._train_locked(training, Path(receipt["output_dir"]), {"initialization": provenance}, {}, {}, None, None,
                             stop_control={"requested": False, "deadline": None}, attempt_started=0, prior_wall=0)
    assert loaded == [{"training": True, "adapter_path": str(parent)}]


def test_parent_artifact_revision_task_and_provenance_mismatch(tmp_path):
    config, template = dataset_fixture(tmp_path)
    parent, source, _ = standard_parent(tmp_path, config, template)
    receipt = prepare_training(config, "good")
    training = read(receipt["training_config"])
    with pytest.raises(ValueError, match="parent mismatch: revision"):
        verify_initialization({**training, "revision": "b" * 40})
    with pytest.raises(ValueError, match="provenance changed"):
        verify_initialization({**training, "initialization_provenance": {"wrong_parent": True}})
    Path(source["train_file"]).write_text(json.dumps({"task_type": "medical_qa", "input": {"images": []}}) + "\n")
    with pytest.raises(ValueError, match="VQA training data digest"):
        verify_initialization(training)
    (parent / "adapter_model.safetensors").write_bytes(b"modified adapter")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        verify_initialization(training)


def test_unbound_adapter_weights_are_rejected(tmp_path):
    config, template = dataset_fixture(tmp_path)
    bare = tmp_path / "bare-adapter"
    bare.mkdir()
    (bare / "adapter_model.safetensors").write_bytes(b"not a receipt")
    with pytest.raises(ValueError, match="naked adapter weights"):
        verify_initialization({**template, "initialization_adapter": str(bare)})


def test_kiosk_gate_rejects_unapproved_or_pseudo_test_rows(tmp_path):
    config, _ = dataset_fixture(tmp_path)
    exported = read(Path(config["work_dir"]) / "latest_dataset.json")
    freeze = read(exported["freeze_manifest"])
    row = read_jsonl(exported["train_file"])[0]
    row["restrictions"]["sft_allowed"] = False
    with pytest.raises(ValueError, match="sft_allowed"):
        validate_kiosk_supervision(row, "train", {}, freeze)
    row = read_jsonl(exported["test_file"])[0]
    row["supervision"]["supervision_kind"] = row["supervision"]["annotation_provenance"] = "teacher_pseudo"
    row["metadata"]["supervision_kind"] = "teacher_pseudo"
    row["review_status"] = "policy_accepted_pseudo"
    with pytest.raises(ValueError, match="cannot be test references"):
        validate_kiosk_supervision(row, "test", {"allow_pseudo_validation": True}, freeze)
    # Historical sources keep their separate admission policies.
    validate_kiosk_supervision({"source_id": "legacy_research"}, "train", {}, {})


def test_pseudo_validation_needs_explicit_native_policy(tmp_path):
    config, _ = dataset_fixture(tmp_path)
    exported = read(Path(config["work_dir"]) / "latest_dataset.json")
    row = read_jsonl(exported["validation_file"])[0]
    freeze = read(exported["freeze_manifest"])
    freeze["identity"]["selection_policy"]["allow_validation"] = True
    policy = freeze["identity"]["selection_policy"]
    row["metadata"]["kiosk"].update(selection_policy_sha256=canonical_hash(policy),
                  teacher={"signature": "fixture", "class_names": ["synthetic_class", "other"], "modality": "external_eye_rgb"},
                  teacher_confidence=.99, teacher_margin=.98)
    row["supervision"]["supervision_kind"] = row["supervision"]["annotation_provenance"] = "teacher_pseudo"
    row["metadata"]["supervision_kind"] = "teacher_pseudo"
    row["review_status"] = "policy_accepted_pseudo"
    with pytest.raises(ValueError, match="allow_pseudo_validation"):
        validate_kiosk_supervision(row, "validation", {}, freeze)
    validate_kiosk_supervision(row, "validation", {"allow_pseudo_validation": True}, freeze)
    row["metadata"]["kiosk"]["teacher_margin"] = .01
    with pytest.raises(ValueError, match="margin"):
        validate_kiosk_supervision(row, "validation", {"allow_pseudo_validation": True}, freeze)


def test_v6_loss_receives_intact_native_processor_tensors(monkeypatch):
    from mediflow_medvlm import campaign_gpu_smoke
    from mediflow_medvlm.training.rendering import native_encode
    from mediflow_medvlm.training.runner import _example_loss
    tensors = {name: object() for name in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw",
                                         "pixel_attention_mask", "mm_token_type_ids", "token_type_ids")}
    calls = []
    class NativeProcessor:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {"tokenize": False, "add_generation_prompt": False, "enable_thinking": False}
            return "fixture native prompt"
        def __call__(self, **kwargs):
            calls.append(kwargs)
            return tensors
    image = object()
    batch = native_encode(NativeProcessor(), [{"role": "user", "content": "fixture"}], generation_prompt=False, images=[image])
    batch["labels"] = object()
    sentinel, model = object(), object()
    def target_loss(found_model, found_batch):
        assert found_model is model and found_batch is batch
        assert all(found_batch[key] is value for key, value in tensors.items())
        return sentinel, 7
    monkeypatch.setattr(campaign_gpu_smoke, "_target_loss", target_loss)
    assert _example_loss(model, batch, {"v6_native": {"cell_id": "V4"}}) is sentinel
    assert calls[0]["images"] == [image]
    assert calls[0]["add_special_tokens"] is False


def test_evaluation_prepares_reviewed_protocol_without_starting_gpu(tmp_path, monkeypatch):
    config, _ = dataset_fixture(tmp_path)
    receipt = prepare_training(config, "evaluation")
    monkeypatch.setattr(subprocess, "Popen", forbid_execution)
    result = prepare_evaluation(config, "evaluation", "base")
    assert result["output_dir"] == str(Path(receipt["run_dir"]) / "evaluation" / "base")
    assert result["test_samples"] == 1
    assert result["test_truth"] == "reviewed_only"
    assert "--adapter" not in result["argv"]
    assert prepare_evaluation(config, "evaluation", "base") == result
    with pytest.raises(ValueError, match="validation-selected"):
        prepare_evaluation(config, "evaluation", "tuned")
    with pytest.raises(ValueError, match="held-out"):
        prepare_evaluation(config, "evaluation", "base", split="validation")


def test_empty_reviewed_test_cannot_be_presented_as_evaluation(tmp_path):
    config, _ = dataset_fixture(tmp_path, test_reviewed=False)
    prepare_training(config, "no_test")
    with pytest.raises(ValueError, match="test set is empty"):
        prepare_evaluation(config, "no_test", "base")


def v6_parent(tmp_path, kiosk):
    root = tmp_path / "v6-final"
    (root / "adapter").mkdir(parents=True)
    (root / "processor").mkdir()
    snapshot = tmp_path / "original-snapshot"
    snapshot.mkdir()
    atomic_json(snapshot / "config.json", {"model_type": "smolvlm", "vision_config": {"fixture": True}})
    (snapshot / "model.safetensors").write_bytes(b"engineering base receipt fixture, not tensors")
    (snapshot / "tokenizer.json").write_bytes(b"engineering tokenizer fixture")
    files = {p.name: {"sha256": file_hash(p), "bytes": p.stat().st_size, "upstream_lfs_sha256": None} for p in snapshot.iterdir()}
    preparation = tmp_path / "model-preparation"
    preparation.mkdir()
    model_id, revision = "engineering/smolvlm-vqa", "b" * 40
    for name, extra in (("pin.json", {}), ("native_assets.json", {"snapshot_path": str(snapshot)}),
                        ("provenance.json", {"files": files})):
        atomic_json(preparation / name, {"model_id": model_id, "revision": revision, **extra})
    prep_receipt = preparation / "receipt.json"
    atomic_json(prep_receipt, {"status": "SUCCEEDED", "scope": "MODEL_PREPARATION_ONLY", "model_id": model_id,
                "model_revision": revision, "model_files_hash": canonical_hash(files),
                "artifacts": [{"path": str(preparation / name), "sha256": file_hash(preparation / name)}
                              for name in ("pin.json", "native_assets.json", "provenance.json")]})
    model = {"model_id": model_id, "revision": revision, "snapshot_path": str(snapshot),
             "cache_dir": str(Path(kiosk["data_root"]) / "huggingface" / "hub"),
             "preparation_receipt": str(prep_receipt), "preparation_receipt_sha256": file_hash(prep_receipt),
             "model_files_hash": canonical_hash(files),
             "native_files": {name: item["sha256"] for name, item in files.items() if not name.endswith(".safetensors")}}
    recipe = {"passes": 1, "microbatch": 1, "effective_batch_size": 16,
              "quantization": "nf4_double_quant", "compute_dtype": "bfloat16", "optimizer": "AdamW",
              "betas": [.9, .999], "eps": 1e-8, "scheduler": "linear_decay", "warmup_steps": 0,
              "lora_rank": 8, "lora_alpha": 16, "lora_dropout": .05, "gradient_checkpointing": True,
              "partial_update_divisor": "actual_example_count", "max_length": 2048,
              "learning_rate": .0001, "weight_decay": 0., "seed": 20261023, "gradient_clip_norm": 1.,
              "examples": 1979, "optimizer_steps": 124}
    source = {"schema_version": V6_SCHEMA, "scope": V6_SCOPE, "task": "vqa", "cell_id": "V4",
              "model_id": model_id, "model": model, "recipe": recipe, "run_id": "v6-first-seed-V4",
              "job_id": "v6-first-seed-V4", "source_hash": "old-code", "data_hash": "old-data",
              "model_hash": canonical_hash(model), "output_dir": str(root), "device": "cuda",
              "local_files_only": True, "fixed_template_date": "2026-11-14",
              "data": {"sample_order_sha256": "old-order"}, "resource_policy": {"budget_mode": "unlimited"},
              "gpu_ledger_path": str(Path(kiosk["data_root"]) / "runtime" / "gpu_budget.json")}
    source["config_hash"] = canonical_hash(source)
    atomic_json(root / "config.json", source)
    targets = ["model.text_model.layers.0.self_attn.q_proj"]
    atomic_json(root / "adapter" / "adapter_config.json", {"base_model_name_or_path": str(snapshot),
                "revision": revision, "peft_type": "LORA", "task_type": "CAUSAL_LM", "bias": "none",
                "target_modules": targets, "r": 8, "lora_alpha": 16, "lora_dropout": .05})
    (root / "adapter" / "adapter_model.safetensors").write_bytes(b"engineering V6 receipt fixture, not tensors")
    atomic_json(root / "processor" / "processor_config.json", {"synthetic_fixture": True})
    identity = {key: source[key] for key in V6_IDENTITY}
    atomic_json(root / "checkpoint.json", {**identity, "complete": True, "final_checkpoint": True,
                "selection_rule": "ONE_PASS_FINAL_NOT_MINIMUM_LOSS", "resume_supported": False,
                "optimizer_rng_saved": False, "optimizer_steps": 124, "samples_processed": 1979,
                "sample_order_sha256": "old-order", "adapter_files": directory_digest(root / "adapter")["files"],
                "processor_files": directory_digest(root / "processor")["files"]})
    atomic_json(root / "metrics.json", {"native_contract": {"revision": revision, "snapshot_path": str(snapshot),
                "files": {"model_files_hash": model["model_files_hash"]}, "architecture": {"exact_target_modules": targets},
                "template": {"fixed_date": "2026-11-14", "templates": {"processor": {"frozen_sha256": "frozen-template"}}}}})
    atomic_json(root / "reload.json", {"status": "SUCCEEDED"})
    atomic_json(root / "final_selection.json", {"selection_rule": "ONE_PASS_FINAL_NOT_MINIMUM_LOSS"})
    atomic_json(root / "completion.json", {**identity, "schema_version": "medvlm.v6.main-train-receipt.v1",
                "status": "SUCCEEDED", "scope": V6_SCOPE, "full_one_pass_verified": True,
                "final_checkpoint_verified": True, "fresh_process_reload_verified": True,
                "v6_main_cell_training_completed": True, "excluded_count": 0, "dev_test_read": False,
                "optimizer_steps": 124, "samples_processed": 1979,
                "artifacts": [{"path": str(p), "sha256": file_hash(p)} for p in sorted(root.rglob("*")) if p.is_file()]})
    kiosk["training_template"] = str(root / "config.json")
    kiosk["initialization_adapter"] = str(root)
    kiosk["start_from_base"] = False
    return root, snapshot


def test_v6_final_maps_actual_native_recipe_and_binds_completion(tmp_path, monkeypatch):
    config, _ = dataset_fixture(tmp_path)
    parent, snapshot = v6_parent(tmp_path, config)
    monkeypatch.setattr(urllib.request, "urlopen", forbid_execution)
    receipt = prepare_training(config, "v6_continued")
    training = read(receipt["training_config"])
    assert training["model_id"] == "engineering/smolvlm-vqa"
    assert training["native_snapshot_path"] == str(snapshot)
    assert training["revision"] == "b" * 40
    assert training["initialization_adapter"] == str(parent / "adapter")
    assert training["v6_native"] == {"cell_id": "V4", "fixed_template_date": "2026-11-14"}
    assert training["quantization"] == "nf4"
    assert training["dtype"] == "bfloat16"
    assert training["gradient_accumulation_steps"] == 16
    assert training["data_root"] == config["data_root"]
    assert training["initialization_provenance"]["format"] == "v6_completed_vqa_final"
    assert verify_initialization(training) == training["initialization_provenance"]
    (snapshot / "model.safetensors").write_bytes(b"changed native base")
    with pytest.raises(ValueError, match="parent artifact changed"):
        verify_initialization(training)


def test_v6_unfinished_output_cannot_initialize(tmp_path):
    config, _ = dataset_fixture(tmp_path)
    parent, _ = v6_parent(tmp_path, config)
    completion = read(parent / "completion.json")
    completion["fresh_process_reload_verified"] = False
    atomic_json(parent / "completion.json", completion)
    with pytest.raises(ValueError, match="verified reload"):
        prepare_training(config, "unfinished_v6")
