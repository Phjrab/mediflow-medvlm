"""Verified adapter initialization for a new run, distinct from strict resume.

Only repository checkpoint receipts are accepted. This module reads local JSON
and hashes bytes; it never loads tensors, imports torch, or accesses the Hub.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..runtime import canonical_hash, file_hash
from .config import verified_checkpoint

V6_SCHEMA = "medvlm.v6.main-train-config.v1"
V6_SCOPE = "NATIVE_FIRST_SEED_ONE_PASS_FINAL_ONLY"
V6_IDENTITY = ("job_id", "run_id", "source_hash", "config_hash", "data_hash", "model_hash")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _path(value, base):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _tree(root):
    return {p.relative_to(root).as_posix(): file_hash(p)
            for p in sorted(root.rglob("*")) if p.is_file()}


def _bound_file(root, name, digest):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("unsafe parent artifact path: " + str(name))
    path = root / relative
    if not path.is_file() or file_hash(path) != digest:
        raise ValueError("initialization parent artifact changed: " + str(name))
    return path


def _adapter_config(adapter, files, parent_model, parent_revision, targets, config):
    if "adapter_config.json" not in files or not any(
            name.startswith("adapter_model") and name.endswith(".safetensors") for name in files):
        raise ValueError("initialization requires receipt-bound adapter_config and safetensors weights")
    saved = _read(adapter / "adapter_config.json")
    if (saved.get("peft_type") != "LORA" or saved.get("task_type") != "CAUSAL_LM"
            or saved.get("modules_to_save") or saved.get("bias", "none") != "none"):
        raise ValueError("initialization must be a language-only repository LoRA adapter")
    if saved.get("base_model_name_or_path") != parent_model or saved.get("revision") != parent_revision:
        raise ValueError("adapter_config base model or revision differs from its parent receipt")
    actual_targets = saved.get("target_modules")
    if (not isinstance(actual_targets, list) or not actual_targets
            or set(actual_targets) != set(targets)):
        raise ValueError("adapter target modules differ from the verified parent identity")
    for recipe_key, adapter_key in (("lora_rank", "r"), ("lora_alpha", "lora_alpha"), ("lora_dropout", "lora_dropout")):
        if recipe_key in config and config[recipe_key] != saved.get(adapter_key):
            raise ValueError("initialization adapter conflicts with training template: " + recipe_key)
    return saved


def _standard_parent(root, config):
    checkpoint = verified_checkpoint(root)
    identity = checkpoint.get("identity", {})
    files = checkpoint.get("artifacts", {}).get("files", {})
    # Reject additional unrecorded model files that a loader could prefer.
    if {name: sha for name, sha in _tree(root).items() if name != "checkpoint.json"} != files:
        raise ValueError("initialization checkpoint contains unbound or changed artifacts")
    run = root.parent.parent
    source_path = run / "config.json"
    source = _read(source_path)
    if canonical_hash(source) != identity.get("config_sha256"):
        raise ValueError("initialization original training config is not bound to the checkpoint")
    receipt_path = Path(config.get("initialization_manifest") or run / "manifest.json")
    receipt = _read(receipt_path)
    if receipt.get("identity") != identity:
        raise ValueError("initialization manifest identity differs from the checkpoint")
    for key, expected in (("model_id", config["model_id"]), ("revision", config["revision"]),
                          ("quantization", config.get("quantization", "none"))):
        if identity.get(key) != expected:
            raise ValueError("initialization parent mismatch: " + key)
    source_train = _path(source["train_file"], source_path.parent)
    train_hash = identity.get("data", {}).get("train", {}).get("sha256")
    if not train_hash or file_hash(source_train) != train_hash:
        raise ValueError("initialization original VQA training data digest is unavailable or changed")
    with source_train.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows or any(row.get("task_type") != "medical_vqa" or not row.get("input", {}).get("images") for row in rows):
        raise ValueError("initialization source must be a verified multimodal VQA run, not a text adapter")
    base_hash = identity.get("base_files_sha256") or receipt.get("model", {}).get("native_files", {}).get("sha256")
    if not base_hash or not identity.get("template_sha256"):
        raise ValueError("initialization parent needs native base and template digests")
    saved = _adapter_config(root, files, source.get("native_snapshot_path") or source["model_id"],
                            source["revision"], identity.get("target_modules", []), config)
    return {"schema_version": "medvlm.adapter-initialization.v1", "format": "native_training_checkpoint",
            "adapter_path": str(root), "checkpoint_sha256": file_hash(root / "checkpoint.json"),
            "parent_identity": identity, "source_config_sha256": file_hash(source_path),
            "source_manifest_sha256": file_hash(receipt_path), "source_task": "medical_vqa",
            "source_train_sha256": train_hash, "base_files_sha256": base_hash,
            "template_sha256": identity["template_sha256"], "target_modules": sorted(saved["target_modules"]),
            "adapter_files_sha256": canonical_hash(files), "model_id": identity["model_id"],
            "revision": identity["revision"], "quantization": identity["quantization"],
            "optimizer_scheduler_rng_restored": False}


def _v6_prepared_base(model):
    """Verify the preparation chain without importing a campaign or GPU worker."""
    receipt_path = Path(model["preparation_receipt"])
    if file_hash(receipt_path) != model.get("preparation_receipt_sha256"):
        raise ValueError("V6 model preparation receipt changed")
    receipt = _read(receipt_path)
    if (receipt.get("status") != "SUCCEEDED" or receipt.get("scope") != "MODEL_PREPARATION_ONLY"
            or receipt.get("model_id") != model["model_id"] or receipt.get("model_revision") != model["revision"]):
        raise ValueError("V6 base preparation identity mismatch")
    artifacts = {}
    for artifact in receipt.get("artifacts", []):
        path = Path(artifact["path"])
        if path.name in artifacts or file_hash(path) != artifact["sha256"]:
            raise ValueError("V6 base preparation artifact changed")
        artifacts[path.name] = path
    if not {"pin.json", "native_assets.json", "provenance.json"} <= artifacts.keys():
        raise ValueError("V6 base preparation artifacts are incomplete")
    values = {name: _read(path) for name, path in artifacts.items()}
    for name in ("pin.json", "native_assets.json", "provenance.json"):
        if any(values[name].get(key) != model[key] for key in ("model_id", "revision")):
            raise ValueError("V6 prepared native model identity mismatch")
    if values["native_assets.json"].get("snapshot_path") != model["snapshot_path"]:
        raise ValueError("V6 prepared snapshot path mismatch")
    files = values["provenance.json"]["files"]
    if canonical_hash(files) != model["model_files_hash"] or receipt.get("model_files_hash") != model["model_files_hash"]:
        raise ValueError("V6 prepared base files digest mismatch")
    snapshot = Path(model["snapshot_path"])
    if not any(name.endswith(".safetensors") for name in files):
        raise ValueError("V6 prepared base weights are missing")
    for name, metadata in files.items():
        path = _bound_file(snapshot, name, metadata["sha256"])
        if path.stat().st_size != metadata["bytes"]:
            raise ValueError("V6 prepared base file size changed: " + name)
    from .models import model_file_digests
    native = model_file_digests({"model_id": str(snapshot), "revision": model["revision"]})
    if any(name not in files or any(item[key] != files[name].get(key) for key in ("sha256", "bytes"))
           for name, item in native["files"].items()):
        raise ValueError("V6 loader would use a native file missing from the preparation receipt")
    return native["sha256"]


def _v6_parent(root, config):
    source_path = root / "config.json"
    source = _read(source_path)
    checkpoint = _read(root / "checkpoint.json")
    receipt_path = Path(config.get("initialization_manifest") or root / "completion.json")
    receipt = _read(receipt_path)
    if (source.get("schema_version") != V6_SCHEMA or source.get("scope") != V6_SCOPE
            or source.get("task") != "vqa" or source.get("cell_id") not in {"V1", "V2", "V3", "V4"}):
        raise ValueError("initialization supports completed V6 V1-V4 VQA finals, not text/smoke campaign outputs")
    if source.get("config_hash") != canonical_hash({k: v for k, v in source.items() if k != "config_hash"}):
        raise ValueError("V6 original config hash mismatch")
    if any(checkpoint.get(key) != source.get(key) or receipt.get(key) != source.get(key) for key in V6_IDENTITY):
        raise ValueError("V6 final checkpoint/completion identity mismatch")
    if (checkpoint.get("complete") is not True or checkpoint.get("final_checkpoint") is not True
            or checkpoint.get("selection_rule") != "ONE_PASS_FINAL_NOT_MINIMUM_LOSS"
            or checkpoint.get("resume_supported") is not False or checkpoint.get("optimizer_rng_saved") is not False
            or checkpoint.get("optimizer_steps") != source["recipe"]["optimizer_steps"]
            or checkpoint.get("samples_processed") != source["recipe"]["examples"]
            or checkpoint.get("sample_order_sha256") != source["data"]["sample_order_sha256"]):
        raise ValueError("V6 final checkpoint is incomplete or inconsistent")
    if (receipt.get("schema_version") != "medvlm.v6.main-train-receipt.v1"
            or receipt.get("status") != "SUCCEEDED" or receipt.get("scope") != V6_SCOPE
            or any(receipt.get(key) is not True for key in
                   ("full_one_pass_verified", "final_checkpoint_verified", "fresh_process_reload_verified", "v6_main_cell_training_completed"))
            or receipt.get("excluded_count") != 0 or receipt.get("dev_test_read") is not False
            or receipt.get("optimizer_steps") != checkpoint["optimizer_steps"]
            or receipt.get("samples_processed") != checkpoint["samples_processed"]):
        raise ValueError("V6 completed final receipt with verified reload is required")
    expected = {}
    original_output = Path(source["output_dir"])
    for artifact in receipt.get("artifacts", []):
        path = Path(artifact["path"])
        if not path.is_relative_to(original_output):
            raise ValueError("V6 final artifact escapes original output directory")
        name = path.relative_to(original_output).as_posix()
        if name in expected:
            raise ValueError("duplicate V6 final artifact")
        _bound_file(root, name, artifact["sha256"])
        expected[name] = artifact["sha256"]
    actual = {name: sha for name, sha in _tree(root).items() if name != "completion.json"}
    if actual != expected or not {"config.json", "checkpoint.json", "metrics.json", "reload.json", "final_selection.json"} <= actual.keys():
        raise ValueError("V6 final artifacts are missing, changed or unbound")
    for folder, key in (("adapter", "adapter_files"), ("processor", "processor_files")):
        if not checkpoint.get(key) or _tree(root / folder) != checkpoint[key]:
            raise ValueError("V6 saved " + folder + " files differ from the final checkpoint")
    model = source["model"]
    if (source.get("model_hash") != canonical_hash(model)
            or config["model_id"] != model["model_id"] or config["revision"] != model["revision"]
            or config.get("native_snapshot_path") != model["snapshot_path"]
            or config.get("quantization") != "nf4" or config.get("dtype") != "bfloat16"
            or config.get("v6_native") != {"cell_id": source["cell_id"], "fixed_template_date": source["fixed_template_date"]}):
        raise ValueError("V6 initialization base/revision/native recipe differs from training template")
    contract = _read(root / "metrics.json").get("native_contract", {})
    if (contract.get("revision") != model["revision"] or contract.get("snapshot_path") != model["snapshot_path"]
            or contract.get("files", {}).get("model_files_hash") != model["model_files_hash"]
            or contract.get("template", {}).get("fixed_date") != source["fixed_template_date"]):
        raise ValueError("V6 native model/template contract differs from completed output")
    templates = contract["template"]["templates"]
    active = templates.get("processor", {})
    if active.get("delegates_to_tokenizer"):
        active = templates.get("tokenizer", {})
    template_hash = active.get("frozen_sha256")
    if not template_hash:
        raise ValueError("V6 final native template digest is missing")
    targets = contract.get("architecture", {}).get("exact_target_modules", [])
    adapter = root / "adapter"
    saved = _adapter_config(adapter, checkpoint["adapter_files"], model["snapshot_path"], model["revision"], targets, config)
    base_hash = _v6_prepared_base(model)
    return {"schema_version": "medvlm.adapter-initialization.v1", "format": "v6_completed_vqa_final",
            "adapter_path": str(adapter), "checkpoint_sha256": file_hash(root / "checkpoint.json"),
            "parent_identity": {key: source[key] for key in V6_IDENTITY},
            "source_config_sha256": file_hash(source_path), "source_manifest_sha256": file_hash(receipt_path),
            "source_task": "medical_vqa", "base_files_sha256": base_hash, "template_sha256": template_hash,
            "target_modules": sorted(saved["target_modules"]), "adapter_files_sha256": canonical_hash(checkpoint["adapter_files"]),
            "model_id": model["model_id"], "revision": model["revision"], "quantization": "nf4",
            "source_model_files_sha256": model["model_files_hash"], "optimizer_scheduler_rng_restored": False}


def verify_initialization(config, model_metadata=None):
    """Revalidate a parent adapter and bind fresh-run/resume identities to it."""
    if not config.get("initialization_adapter"):
        if config.get("initialization_provenance") or config.get("initialization_manifest"):
            raise ValueError("initialization provenance requires an initialization_adapter")
        return None
    adapter = Path(config["initialization_adapter"]).expanduser().resolve()
    root = adapter.parent if adapter.name == "adapter" and (adapter.parent / "checkpoint.json").is_file() else adapter
    if not (root / "checkpoint.json").is_file():
        raise ValueError("initialization needs a repository checkpoint receipt; naked adapter weights are not accepted")
    info = _read(root / "checkpoint.json")
    provenance = _standard_parent(root, config) if "identity" in info else _v6_parent(root, config)
    if config.get("initialization_provenance") is not None and config["initialization_provenance"] != provenance:
        raise ValueError("initialization provenance changed; create a new run for a different parent")
    if model_metadata is not None:
        assert_initialization_model(provenance, model_metadata)
    return provenance


def assert_initialization_model(provenance, model_metadata):
    """Compare loaded metadata after the parent's file-only preflight."""
    for field, actual in (("base_files_sha256", model_metadata.get("native_files", {}).get("sha256")),
                          ("template_sha256", model_metadata.get("chat_template_sha256")),
                          ("target_modules", sorted(model_metadata.get("exact_target_modules", [])))):
        if provenance[field] != actual:
            raise ValueError("initialization loaded native model differs from parent: " + field)
