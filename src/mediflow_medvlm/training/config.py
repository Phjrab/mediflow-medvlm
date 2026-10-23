"""Immutable experiment identities and strict frozen train/validation selection."""
from __future__ import annotations

import json
from pathlib import Path
import re

from ..runtime import canonical_hash, file_hash


def load_config(path):
    config = json.loads(Path(path).read_text())
    required = {"run_id", "pair_id", "experiment_id", "model_id", "revision", "train_file",
                "validation_file", "freeze_manifest", "output_dir", "data_root"}
    missing = required - set(config)
    if missing:
        raise ValueError("missing config fields: " + ",".join(sorted(missing)))
    if not config.get("synthetic_fixture") and not re.fullmatch(r"[0-9a-f]{40}", config["revision"]):
        raise ValueError("revision must be an immutable Hub commit")
    if config.get("microbatch_size", 1) != 1:
        raise ValueError("v1 native multimodal worker supports microbatch_size=1")
    if not 1 <= int(config.get("max_optimizer_steps", 1000)) <= 1000:
        raise ValueError("v1 optimizer updates must be between 1 and 1000")
    if float(config.get("max_epochs", 1)) != 1:
        raise ValueError("v1 uses exactly one pass, additionally capped by optimizer updates")
    if config.get("push_to_hub", False) or config.get("external_tracking", False):
        raise ValueError("external upload/tracking disabled for this research")
    if int(config.get("gradient_accumulation_steps", 16)) < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    if config.get("mode", "v1") == "smoke" and config.get("max_optimizer_steps", 20) < 10 and not config.get("synthetic_fixture"):
        raise ValueError("medical smoke requires at least 10 real optimizer updates")
    return config


def load_frozen_data(config):
    freeze_path = Path(config["freeze_manifest"])
    freeze = json.loads(freeze_path.read_text())
    result, hashes = {}, {}
    for kind in ("train", "validation"):
        path = Path(config[kind + "_file"])
        digest = file_hash(path)
        matching = [entry for key, entry in freeze["splits"].items()
                    if key.endswith("/" + kind) and entry.get("sha256") == digest]
        if not matching:
            raise ValueError(f"{kind} file is not represented by a matching frozen split digest")
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not records:
            raise ValueError(kind + " split is empty")
        if any(row.get("split") != kind for row in records):
            raise ValueError("split mismatch: test data cannot enter trainer or checkpoint selection")
        if any(row.get("source_id") == "kormedmcqa" for row in records if kind == "train"):
            raise ValueError("KorMedMCQA is evaluation-only in v1")
        if any(not row.get("source_revision") for row in records):
            raise ValueError("source revision must be resolved")
        ids = [row["sample_id"] for row in records]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate sample_id in frozen split")
        result[kind], hashes[kind] = records, {"sha256": digest, "count": len(records), "ordered_ids_sha256": canonical_hash(ids)}
    if {row["sample_id"] for row in result["train"]} & {row["sample_id"] for row in result["validation"]}:
        raise ValueError("train/validation sample overlap")
    # Preserve test hashes only. Never open test files during training.
    hashes["test_references"] = {key: {k: value.get(k) for k in ("sha256", "count")}
                                for key, value in freeze["splits"].items() if key.endswith("/test")}
    hashes["freeze_manifest_sha256"] = file_hash(freeze_path)
    return result, hashes


def verified_checkpoint(checkpoint):
    root = Path(checkpoint).resolve()
    found = json.loads((root / "checkpoint.json").read_text())
    if not found.get("complete"):
        raise ValueError("incomplete checkpoint cannot resume")
    for relative, digest in found.get("artifacts", {}).get("files", {}).items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or file_hash(path) != digest:
            raise ValueError("checkpoint artifact hash mismatch: " + relative)
    return found


def assert_resume_identity(expected, checkpoint, *, legacy_source_snapshot=None, legacy_manifest=None):
    found = verified_checkpoint(checkpoint)
    actual = found["identity"]
    if actual.get("identity_schema_version") == "medvlm.training-identity.v2" and not actual.get("base_files_sha256"):
        raise ValueError("v2 identity requires base_files_sha256; cannot fall back to legacy")
    if actual == expected:
        return found
    legacy_keys = {"config_sha256", "data", "training_code_sha256", "model_id", "revision",
                   "template_sha256", "renderer_version", "target_modules", "quantization"}
    if not legacy_source_snapshot or set(actual) != legacy_keys or not legacy_manifest:
        raise ValueError("resume identity mismatch: config, data, model or template changed")
    if expected.get("identity_schema_version") != "medvlm.training-identity.v2":
        raise ValueError("legacy migration requires a v2 destination identity")
    if any(actual[key] != expected[key] for key in legacy_keys - {"training_code_sha256"}):
        raise ValueError("legacy migration cannot change config, data, model, template or recipe")
    if legacy_manifest.get("identity") != actual:
        raise ValueError("legacy identity must match the preserved original run manifest")
    if legacy_manifest.get("model", {}).get("native_files", {}).get("sha256") != expected["base_files_sha256"]:
        raise ValueError("legacy migration base file digest mismatch")
    snapshot = Path(legacy_source_snapshot).resolve()
    receipt_path = snapshot / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    captured = receipt["captured_python_files"]
    expected_files = {name: digest for name, digest in receipt["expected_code"]["source_files"].items()
                      if "/training/" in name or name.endswith("/runtime.py")}
    if captured != expected_files or canonical_hash(captured) != actual["training_code_sha256"]:
        raise ValueError("legacy source receipt does not match checkpoint training code")
    if receipt["expected_code"]["training_code_sha256"] != actual["training_code_sha256"]:
        raise ValueError("legacy source receipt code digest mismatch")
    for relative, digest in captured.items():
        path = (snapshot / relative).resolve()
        if not path.is_relative_to(snapshot) or file_hash(path) != digest:
            raise ValueError("legacy source snapshot file digest mismatch")
    found["identity_migration"] = {"policy": "explicit-verified-source-snapshot-v1",
                                   "from_schema": "legacy-v1", "to_schema": "medvlm.training-identity.v2",
                                   "receipt_sha256": file_hash(receipt_path),
                                   "previous_training_code_sha256": actual["training_code_sha256"],
                                   "current_training_code_sha256": expected["training_code_sha256"]}
    return found
