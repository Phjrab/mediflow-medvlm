"""Prepare reviewable terminal commands for local-photo adaptation and evaluation.

Preparation never starts a subprocess, initializes CUDA, or downloads metadata.
The normal native worker owns the existing shared GPU lease and usage ledger.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import re
import sys

from ..runtime import atomic_json, canonical_hash, file_hash
from ..training.config import load_frozen_data, validate_config, validate_kiosk_supervision
from ..training.initialization import V6_SCHEMA, V6_SCOPE, verify_initialization
from .common import absolute_path, read_jsonl, workspace


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _run_path(config, run_name):
    if not isinstance(run_name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_name):
        raise ValueError("run_name must be 1-64 ASCII letters/digits/underscores/hyphens, starting with a letter or digit")
    return workspace(config) / "runs" / run_name


def _normalize_template(template, path):
    result = deepcopy(template)
    for key in ("train_file", "validation_file", "freeze_manifest", "output_dir", "data_root", "image_root",
                "cache_dir", "native_snapshot_path", "runtime_root", "resource_policy_path", "initialization_adapter", "initialization_manifest"):
        if result.get(key):
            result[key] = str(absolute_path(result[key], path.parent))
    model = result.get("model_id")
    if isinstance(model, str):
        expanded = os.path.expandvars(os.path.expanduser(model))
        if Path(expanded).is_absolute() or model.startswith((".", "~", "$")) or (path.parent / expanded).is_dir():
            result["model_id"] = str(absolute_path(expanded, path.parent))
    return result


def _v6_template(source, path, data_root):
    """Translate only the repository's actual completed V6 VQA recipe fields."""
    if (source.get("schema_version") != V6_SCHEMA or source.get("scope") != V6_SCOPE
            or source.get("task") != "vqa" or source.get("cell_id") not in {"V1", "V2", "V3", "V4"}):
        raise ValueError("Use a generic native training JSON or the V6 V1-V4 main-train config.json; text, smoke and campaign-plan JSONs are not training templates")
    if source.get("config_hash") != canonical_hash({k: v for k, v in source.items() if k != "config_hash"}):
        raise ValueError("V6 training template config_hash is invalid; use the unchanged original config.json")
    model, recipe = source["model"], source["recipe"]
    required_recipe = {"passes": 1, "microbatch": 1, "effective_batch_size": 16,
                       "quantization": "nf4_double_quant", "compute_dtype": "bfloat16",
                       "optimizer": "AdamW", "betas": [.9, .999], "eps": 1e-8,
                       "scheduler": "linear_decay", "warmup_steps": 0,
                       "lora_rank": 8, "lora_alpha": 16, "lora_dropout": .05,
                       "gradient_checkpointing": True, "partial_update_divisor": "actual_example_count"}
    if any(recipe.get(key) != value for key, value in required_recipe.items()):
        raise ValueError("V6 template recipe is not the supported native VQA one-pass recipe; use an explicit generic training template")
    if (source.get("device") != "cuda" or source.get("local_files_only") is not True
            or source.get("fixed_template_date") != "2026-11-14"):
        raise ValueError("V6 native device, local_files_only and fixed_template_date must remain unchanged")
    ledger = absolute_path(source["gpu_ledger_path"], path.parent)
    if ledger != Path(data_root) / "runtime" / "gpu_budget.json":
        raise ValueError("data_root must be the original MEDVLM_DATA_ROOT from the V6 GPU ledger; do not create an independent lock/ledger")
    result = {"model_id": model["model_id"], "revision": model["revision"],
              "native_snapshot_path": str(absolute_path(model["snapshot_path"], path.parent)),
              "cache_dir": str(absolute_path(model["cache_dir"], path.parent)),
              "data_root": str(data_root), "device": source["device"], "local_files_only": True,
              "dtype": recipe["compute_dtype"], "quantization": "nf4", "attn_implementation": "sdpa",
              "max_length": recipe["max_length"], "microbatch_size": recipe["microbatch"],
              "gradient_accumulation_steps": recipe["effective_batch_size"],
              "learning_rate": recipe["learning_rate"], "weight_decay": recipe["weight_decay"],
              "warmup_steps": recipe["warmup_steps"], "seed": recipe["seed"],
              "lora_rank": recipe["lora_rank"], "lora_alpha": recipe["lora_alpha"],
              "lora_dropout": recipe["lora_dropout"], "gradient_checkpointing": recipe["gradient_checkpointing"],
              "max_grad_norm": recipe["gradient_clip_norm"], "resource_policy": deepcopy(source["resource_policy"]),
              "v6_native": {"cell_id": source["cell_id"], "fixed_template_date": source["fixed_template_date"]},
              "push_to_hub": False, "external_tracking": False}
    return result


def prepare_training(config, run_name, max_steps=None):
    """Freeze a NEW native run and return its foreground train command."""
    from .dataset import validate_current_export
    run_dir = _run_path(config, run_name)
    start_from_base = config.get("start_from_base", False)
    if type(start_from_base) is not bool:
        raise ValueError("start_from_base must be an explicit boolean")
    if start_from_base and config.get("initialization_adapter"):
        raise ValueError("Choose initialization_adapter or start_from_base=true, not both")
    if not start_from_base and not config.get("initialization_adapter"):
        raise ValueError("Set initialization_adapter to your existing VQA checkpoint; a fresh base LoRA requires explicit start_from_base=true")
    if max_steps is not None and (type(max_steps) is not int or max_steps < 1):
        raise ValueError("max_steps must be a positive integer")
    if not config.get("training_template"):
        raise ValueError("Set training_template to the original native/V6 VQA training config.json before prepare")
    template_path = Path(config["training_template"])
    source = _read(template_path)
    data_root = Path(config["data_root"])
    if source.get("schema_version", "").startswith("medvlm.v6") or "model" in source:
        training = _v6_template(source, template_path, data_root)
    else:
        training = _normalize_template(source, template_path)
        if Path(training.get("data_root", "")) != data_root:
            raise ValueError("data_root must equal the training template's original MEDVLM_DATA_ROOT so the shared GPU lock/ledger is reused")
    if training.get("runtime_root") and Path(training["runtime_root"]) != data_root / "runtime":
        raise ValueError("runtime_root must use the original data_root/runtime shared GPU lock/ledger")
    if max_steps is not None and max_steps < 10 and not training.get("synthetic_fixture"):
        raise ValueError("Medical smoke requires at least 10 real optimizer updates; use --max-steps 10 or omit --max-steps for a finite one-pass run on a small dataset")
    if source.get("initialization_adapter") and not config.get("initialization_adapter") and not start_from_base:
        raise ValueError("The template has an earlier initialization adapter; explicitly set kiosk initialization_adapter to the checkpoint you intend to continue")
    for key in ("initialization_adapter", "initialization_manifest", "initialization_provenance"):
        training.pop(key, None)
    exported = validate_current_export(config)
    output_dir = run_dir / "output"
    training.update({"run_id": "kiosk-" + run_name, "pair_id": "kiosk-" + run_name + "-" + exported["dataset_id"][:12],
                     "experiment_id": "kiosk-local-photos-" + run_name,
                     "train_file": exported["train_file"], "validation_file": exported["validation_file"],
                     "freeze_manifest": exported["freeze_manifest"], "image_root": exported["image_root"],
                     "output_dir": str(output_dir), "data_root": str(data_root),
                     "training_schema_version": "medvlm.training.v2", "training_recipe": "one_pass", "max_epochs": 1,
                     "allow_pseudo_validation": config["pseudo_labels"]["allow_validation"]})
    rows, data_hashes = load_frozen_data(training)
    if any(not str(row.get("source_id", "")).startswith("kiosk_") for split in rows.values() for row in split):
        raise ValueError("kiosk preparation only accepts its frozen local-image export")
    accumulation = training.get("gradient_accumulation_steps", 16)
    if type(accumulation) is not int or accumulation < 1:
        raise ValueError("training template gradient_accumulation_steps must be a positive integer")
    available = len(rows["train"])
    if max_steps is not None:
        cap = min(available, max_steps * accumulation)
        if math.ceil(cap / accumulation) < max_steps:
            raise ValueError(f"Only {available} eligible training rows: {max_steps} updates require at least {(max_steps - 1) * accumulation + 1} rows at accumulation={accumulation}")
        training.update(mode="smoke", train_samples_cap=cap, max_optimizer_steps=max_steps)
    else:
        training.update(mode="one_pass", train_samples_cap=available, max_optimizer_steps=math.ceil(available / accumulation))
    if config.get("initialization_adapter"):
        training["initialization_adapter"] = config["initialization_adapter"]
        if config.get("initialization_manifest"):
            training["initialization_manifest"] = config["initialization_manifest"]
        provenance = verify_initialization(training)
        training["initialization_adapter"] = provenance["adapter_path"]
        training["initialization_provenance"] = provenance
    elif config.get("initialization_manifest"):
        raise ValueError("initialization_manifest requires initialization_adapter")
    validate_config(training)
    training_path = run_dir / "training.json"
    prepared = {"schema_version": "medvlm.kiosk-prepared.v1", "status": "PREPARED_NOT_RUN",
                "run_name": run_name, "run_dir": str(run_dir), "output_dir": str(output_dir),
                "training_config": str(training_path), "training_config_sha256": canonical_hash(training),
                "training_template": str(template_path), "training_template_sha256": file_hash(template_path),
                "dataset_id": exported["dataset_id"], "dataset": exported, "data_hashes": data_hashes,
                "validation_truth": exported["validation_truth"],
                "train_samples_cap": training["train_samples_cap"], "max_optimizer_steps": training["max_optimizer_steps"],
                "initialization": training.get("initialization_provenance"),
                "initialization_mode": "fresh_base" if start_from_base else "existing_vqa_adapter",
                "training_engine": "medvlm.training.v2",
                "template_conversion": ({"source_schema": V6_SCHEMA, "native_model_contract_preserved": True,
                                         "new_checkpoint_policy": "validation_selected_strict_resumable",
                                         "new_optimizer_scheduler": True} if source.get("schema_version") == V6_SCHEMA else None),
                "train_argv": [sys.executable, "-m", "mediflow_medvlm.training", "train", "--config", str(training_path)],
                "reload_argv_prefix": [sys.executable, "-m", "mediflow_medvlm.training", "reload", "--config", str(training_path)],
                "evaluation_root": str(run_dir / "evaluation"), "gpu_started": False}
    receipt_path = run_dir / "prepared.json"
    if run_dir.exists():
        if not training_path.is_file() or not receipt_path.is_file():
            raise ValueError("Run directory already exists without a complete preparation receipt; use a new run_name")
        if _read(training_path) != training or _read(receipt_path) != prepared:
            raise ValueError("Prepared run is immutable; changed data, template, parent or budget requires a new run_name")
        return _read(receipt_path)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir()
    atomic_json(training_path, training)
    atomic_json(receipt_path, prepared)
    return prepared


def prepare_evaluation(config, run_name, kind, split="test"):
    """Freeze one reviewed holdout protocol for base/tuned evaluation commands."""
    from .dataset import validate_current_export
    from ..evaluation.runner import freeze_protocol
    from ..training.config import verified_checkpoint
    if kind not in {"base", "tuned"}:
        raise ValueError("evaluation kind must be base or tuned")
    if split != "test":
        raise ValueError("kiosk terminal evaluation uses the held-out reviewed test split")
    run_dir = _run_path(config, run_name)
    prepared = _read(run_dir / "prepared.json")
    training_path = Path(prepared["training_config"])
    training = _read(training_path)
    if canonical_hash(training) != prepared["training_config_sha256"]:
        raise ValueError("Prepared training config changed")
    exported = validate_current_export(config, prepared["dataset"])
    freeze = _read(exported["freeze_manifest"])
    rows = read_jsonl(exported["test_file"])
    if not rows:
        raise ValueError("Reviewed test set is empty; label held-out test images and export a new dataset/run before evaluation")
    for row in rows:
        validate_kiosk_supervision(row, "test", training, freeze)
        if row.get("supervision", {}).get("supervision_kind") not in {"human_reviewed", "source_supported"}:
            raise ValueError("Test evaluation requires reviewed references; teacher pseudo-labels are not test truth")
    protocol_path = run_dir / "evaluation" / "test_protocol.json"
    freeze_protocol([exported["test_file"]], protocol_path, split="test", cap=0, seed=training.get("seed", 20261023))
    output = run_dir / "evaluation" / kind
    argv = [sys.executable, "-m", "mediflow_medvlm.evaluation", "evaluate", "--config", str(training_path),
            "--protocol", str(protocol_path), "--output-dir", str(output), "--pair-id", training["pair_id"]]
    checkpoint_path = None
    if kind == "tuned":
        selection_path = Path(prepared["output_dir"]) / "selection.json"
        if not selection_path.is_file():
            raise ValueError("Tuned evaluation requires a trained validation-selected checkpoint; run train and reload first")
        selection = _read(selection_path)
        if selection.get("reload_verified") is not True:
            raise ValueError("Run reload for the selected checkpoint before tuned test evaluation")
        checkpoint_path = Path(selection["checkpoint"])
        checkpoint = verified_checkpoint(checkpoint_path)
        if checkpoint.get("identity", {}).get("config_sha256") != canonical_hash(training):
            raise ValueError("Selected adapter belongs to a different training run")
        argv.extend(["--adapter", str(checkpoint_path)])
    return {"status": "PREPARED_NOT_RUN", "kind": kind, "split": "test", "protocol": str(protocol_path),
            "output_dir": str(output), "argv": argv, "adapter": str(checkpoint_path) if checkpoint_path else None,
            "test_samples": len(rows), "test_truth": "reviewed_only", "gpu_started": False}
