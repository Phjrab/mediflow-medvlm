"""Immutable experiment identities and strict frozen train/validation selection."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re

from ..runtime import canonical_hash, file_hash, resolve_resource_policy


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
    version = config.get("training_schema_version", "medvlm.training.v1")
    if version not in {"medvlm.training.v1", "medvlm.training.v2"}:
        raise ValueError("unsupported training schema")
    maximum = config.get("max_optimizer_steps", 1000)
    if type(maximum) is not int or maximum < 1:
        raise ValueError("max_optimizer_steps must be a positive finite integer")
    if version == "medvlm.training.v1" and maximum > 1000:
        raise ValueError("v1 optimizer updates must be between 1 and 1000")
    if float(config.get("max_epochs", 1)) != 1:
        raise ValueError("v1 uses exactly one pass, additionally capped by optimizer updates")
    if config.get("push_to_hub", False) or config.get("external_tracking", False):
        raise ValueError("external upload/tracking disabled for this research")
    if int(config.get("gradient_accumulation_steps", 16)) < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    if version == "medvlm.training.v2":
        if config.get("training_recipe") != "one_pass":
            raise ValueError("v2 requires an explicit finite one_pass training_recipe")
        cap = config.get("train_samples_cap")
        if type(cap) is not int or cap < 1:
            raise ValueError("v2 requires an explicit positive train_samples_cap")
        expected = math.ceil(cap / int(config.get("gradient_accumulation_steps", 16)))
        if config.get("mode") != "smoke" and maximum != expected:
            raise ValueError("v2 one_pass max_optimizer_steps must equal ceil(train_samples_cap / accumulation)")
    if config.get("mode", "v1") == "smoke" and config.get("max_optimizer_steps", 20) < 10 and not config.get("synthetic_fixture"):
        raise ValueError("medical smoke requires at least 10 real optimizer updates")
    resolve_resource_policy(config)
    return config


def training_limits(config, row_count, has_images=False):
    """Keep the original v1 caps; v2 declares a finite corpus and one pass."""
    version = config.get("training_schema_version", "medvlm.training.v1")
    if version == "medvlm.training.v2":
        cap = config["train_samples_cap"]
        if cap > row_count:
            raise ValueError("v2 declared corpus cap exceeds frozen training rows")
    else:
        original_cap = 5000 if has_images else 20000
        cap = min(config.get("train_samples_cap", original_cap), original_cap, row_count)
    return {"training_schema_version": version, "max_epochs": 1, "train_samples_cap": cap,
            "max_optimizer_steps": config.get("max_optimizer_steps", 1000),
            "corpus_rows": row_count, "overflow_policy": "exclude_intact_record_before_updates"}


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


RESOURCE_CONFIG_FIELDS = {"resource_policy", "resource_policy_path", "budget_mode", "total_gpu_hours",
                          "per_model_gpu_hours", "wall_seconds", "run_wall_seconds", "timeout_seconds",
                          "job_timeout_seconds", "campaign_wall_seconds", "checkpoint_reserve_seconds"}


def _verified_source_snapshot(snapshot, training_code_sha256):
    root = Path(snapshot).resolve()
    receipt = json.loads((root / "receipt.json").read_text())
    captured = receipt["captured_python_files"]
    expected = {name: digest for name, digest in receipt["expected_code"]["source_files"].items()
                if "/training/" in name or name.endswith("/runtime.py")}
    if captured != expected or canonical_hash(captured) != training_code_sha256:
        raise ValueError("resource migration source receipt identity mismatch")
    if receipt["expected_code"]["training_code_sha256"] != training_code_sha256:
        raise ValueError("resource migration source code digest mismatch")
    for name, digest in captured.items():
        source = (root / name).resolve()
        if not source.is_relative_to(root) or file_hash(source) != digest:
            raise ValueError("resource migration source bytes changed")
    return captured, file_hash(root / "receipt.json")


def validate_resource_policy_migration(previous_config, current_config, checkpoint_path, expected_identity, *,
                                       previous_manifest, previous_source_snapshot, current_source_snapshot,
                                       expected_sample_order_sha256, expected_training_state_sha256):
    """Read-only eligibility proof; never relax assert_resume_identity or resume.

    There are no unfinished v1 runs requiring migration. A future migration must
    additionally review source semantics and restore/compare optimizer/RNG state
    before claiming exact resume; this helper intentionally grants no such bypass.
    """
    if previous_manifest.get("status") != "SKIPPED_BUDGET" or previous_manifest.get("reason_code") not in {
            "GPU_BUDGET_EXHAUSTED", "BUDGET_OR_WALL_LIMIT", "JOB_TIMEOUT"}:
        raise ValueError("resource migration requires an unfinished run stopped by a time limit")
    previous_recipe = {k: v for k, v in previous_config.items() if k not in RESOURCE_CONFIG_FIELDS}
    current_recipe = {k: v for k, v in current_config.items() if k not in RESOURCE_CONFIG_FIELDS}
    if previous_recipe != current_recipe:
        raise ValueError("resource migration cannot change recipe, optimizer, seed, precision, model or data")
    found = verified_checkpoint(checkpoint_path)
    actual = found["identity"]
    if actual != previous_manifest.get("identity"):
        raise ValueError("resource migration original manifest identity mismatch")
    if actual.get("config_sha256") != canonical_hash(previous_config) or expected_identity.get("config_sha256") != canonical_hash(current_config):
        raise ValueError("resource migration config hash mismatch")
    keys = set(actual) | set(expected_identity)
    if any(actual.get(k) != expected_identity.get(k) for k in keys - {"config_sha256", "training_code_sha256"}):
        raise ValueError("resource migration cannot change checkpoint identity fields")
    if not expected_sample_order_sha256 or found.get("sample_order_sha256") != expected_sample_order_sha256:
        raise ValueError("resource migration sampler ordering mismatch")
    state_hash = found.get("artifacts", {}).get("files", {}).get("training_state.pt")
    if not state_hash or state_hash != expected_training_state_sha256:
        raise ValueError("resource migration optimizer/RNG state digest mismatch")
    old_sources, old_receipt = _verified_source_snapshot(previous_source_snapshot, actual["training_code_sha256"])
    new_sources, new_receipt = _verified_source_snapshot(current_source_snapshot, expected_identity["training_code_sha256"])
    changed_sources = sorted(k for k in set(old_sources) | set(new_sources) if old_sources.get(k) != new_sources.get(k))
    allowed_sources = {"src/mediflow_medvlm/runtime.py", "src/mediflow_medvlm/training/config.py",
                       "src/mediflow_medvlm/training/runner.py"}
    if not set(changed_sources) <= allowed_sources:
        raise ValueError("resource migration includes non-resource source changes")
    old_policy = previous_config.get("resource_policy") or {"budget_mode": previous_config.get("budget_mode", "finite"),
                    "total_gpu_hours": previous_config.get("total_gpu_hours", 24),
                    "per_model_gpu_hours": previous_config.get("per_model_gpu_hours", 6),
                    "run_wall_seconds": previous_config.get("run_wall_seconds", previous_config.get("wall_seconds")),
                    "job_timeout_seconds": previous_config.get("job_timeout_seconds"),
                    "campaign_wall_seconds": previous_config.get("campaign_wall_seconds")}
    return {"schema_version": "medvlm.resource-migration-review.v1", "status": "RESOURCE_ONLY_INPUTS_VERIFIED",
            "previous_policy": resolve_resource_policy(policy=old_policy).as_dict(),
            "current_policy": resolve_resource_policy(current_config).as_dict(),
            "previous_identity": actual, "proposed_identity": expected_identity,
            "previous_source_receipt_sha256": old_receipt, "current_source_receipt_sha256": new_receipt,
            "changed_source_files": changed_sources, "training_state_sha256": state_hash,
            "sample_order_sha256": expected_sample_order_sha256, "recipe_sha256": canonical_hash(previous_recipe),
            "automatic_resume_authorized": False, "exact_resume_verified": False,
            "remaining_gate": "review source semantics and verify restored optimizer/RNG before explicit migration"}
