"""One local-only, CPU/meta V6 native TRAIN eligibility job.

CLI: python -m mediflow_medvlm.campaign_preflight --config ABS --receipt ABS
This worker neither freezes a common cohort nor owns campaign/recovery state.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

from .campaign_supervisor import atomic_json, canonical_hash, file_hash
from .campaign_v6 import cell_spec
from .runtime import utcnow

SCHEMA = "medvlm.v6.native-preflight-config.v1"
SOURCE_FILES = (
    "__init__.py", "campaign_preflight.py", "campaign_supervisor.py", "campaign_v6.py",
    "runtime.py", "domain.py", "validation.py", "training/__init__.py", "training/models.py", "training/rendering.py",
    "training/v6_native.py",
)
IDENTITY_KEYS = ("job_id", "run_id", "source_hash", "config_hash", "data_hash", "model_hash")
OUTPUT_FILES = {"config.json", "architecture.json", "fixtures.json", "eligibility.jsonl", "summary.json"}


def runtime_versions():
    result = {"python": sys.version.split()[0]}
    for name in ("torch", "transformers", "tokenizers", "Pillow", "torchvision", "num2words"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


class PreflightError(ValueError):
    """Controlled error code, safe to emit without private example contents."""


class IneligibleRow(ValueError):
    """A measured, intact native encoding that exceeds the eligibility context."""
    def __init__(self, reason, native):
        super().__init__(reason)
        self.native = native


def _absolute(path):
    path = Path(path)
    if not path.is_absolute():
        raise PreflightError("ABSOLUTE_PATH_REQUIRED")
    return path


def source_identity(repo=None):
    package = Path(__file__).resolve().parent if repo is None else Path(repo) / "src/mediflow_medvlm"
    files = {"src/mediflow_medvlm/" + name: file_hash(package / name) for name in SOURCE_FILES}
    return files, canonical_hash(files)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _prepared(receipt_path):
    receipt_path = _absolute(receipt_path)
    receipt = _read(receipt_path)
    if receipt.get("status") != "SUCCEEDED" or receipt.get("scope") != "MODEL_PREPARATION_ONLY":
        raise PreflightError("PREPARATION_NOT_SUCCEEDED")
    artifacts = receipt.get("artifacts", [])
    if not artifacts:
        raise PreflightError("PREPARATION_ARTIFACTS_MISSING")
    by_name = {}
    for artifact in artifacts:
        path = _absolute(artifact["path"])
        if file_hash(path) != artifact["sha256"]:
            raise PreflightError("PREPARATION_ARTIFACT_CHANGED")
        if path.name in by_name:
            raise PreflightError("PREPARATION_ARTIFACT_DUPLICATE")
        by_name[path.name] = path
    if not {"pin.json", "native_assets.json", "provenance.json"} <= by_name.keys():
        raise PreflightError("PREPARATION_ARTIFACTS_MISSING")
    pin, native, provenance = (_read(by_name[name]) for name in
                               ("pin.json", "native_assets.json", "provenance.json"))
    revision, model_id = receipt.get("model_revision"), receipt.get("model_id")
    if not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
        raise PreflightError("IMMUTABLE_REVISION_REQUIRED")
    if any(value.get("revision") != revision or value.get("model_id") != model_id
           for value in (pin, native, provenance)):
        raise PreflightError("PREPARATION_MODEL_IDENTITY_MISMATCH")
    if canonical_hash(provenance["files"]) != receipt["model_files_hash"]:
        # Preparation uses UTF-8 canonical JSON; model filenames and digest metadata
        # are ASCII, so its hash equals the supervisor convention used here.
        raise PreflightError("PREPARATION_MODEL_FILES_HASH_MISMATCH")
    snapshot = _absolute(native["snapshot_path"])
    cache = _absolute(native["cache_dir"])
    expected_snapshot = cache / ("models--" + model_id.replace("/", "--")) / "snapshots" / revision
    if snapshot != expected_snapshot or not snapshot.is_dir():
        raise PreflightError("LOCAL_PINNED_SNAPSHOT_REQUIRED")
    # Revalidate every small native asset, never rehash large weight shards here.
    native_files = {}
    for name, metadata in provenance["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise PreflightError("UNSAFE_MODEL_FILENAME")
        path = snapshot / relative
        if not path.is_file() or path.stat().st_size != metadata["bytes"]:
            raise PreflightError("PREPARED_MODEL_FILE_MISSING_OR_RESIZED")
        if name.endswith(".safetensors"):
            continue
        if file_hash(path) != metadata["sha256"]:
            raise PreflightError("NATIVE_ASSET_BYTES_CHANGED")
        native_files[name] = metadata["sha256"]
    return {"model_id": model_id, "revision": revision, "snapshot_path": str(snapshot),
            "cache_dir": str(cache), "preparation_receipt": str(receipt_path),
            "preparation_receipt_sha256": file_hash(receipt_path),
            "model_files_hash": receipt["model_files_hash"], "native_files": native_files}


def _train_rows(path, expected_count, task):
    rows, seen = [], set()
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
                raise PreflightError("MISSING_OR_DUPLICATE_SAMPLE_ID")
            if row.get("split") != "train":
                raise PreflightError("TRAIN_PARTITION_REQUIRED")
            is_vqa = row.get("task_type") == "medical_vqa"
            if is_vqa != (task == "vqa") or (task == "text" and row.get("input", {}).get("images")):
                raise PreflightError("TRAIN_AXIS_MISMATCH")
            if str(row.get("source_id", "")).lower() in {"aihub_71762", "71762", "medmcqa", "vqa_rad"}:
                raise PreflightError("FORBIDDEN_V6_TRAIN_SOURCE")
            rows.append(row)
            seen.add(sample_id)
    if len(rows) != expected_count:
        raise PreflightError("TRAIN_COUNT_MISMATCH")
    return rows


def build_config(cell_id, train_file, preparation_receipt, output_dir, *, repo=None,
                 image_root=None, expected_count=None, run_id=None):
    """Build reviewable config from existing files, without writing or encoding rows.

    expected_count defaults to the fixed R2 candidate count (8000 text/1979 VQA).
    repo must be the final immutable source snapshot before registering the job.
    """
    cell = cell_spec(cell_id)
    model = _prepared(preparation_receipt)
    if model["model_id"] != cell["model_id"]:
        raise PreflightError("CELL_MODEL_MISMATCH")
    train_file, output_dir = _absolute(train_file), _absolute(output_dir)
    count = expected_count if expected_count is not None else (8000 if cell["task"] == "text" else 1979)
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise PreflightError("INVALID_EXPECTED_COUNT")
    rows = _train_rows(train_file, count, cell["task"])
    if cell["task"] == "vqa" and image_root is None:
        raise PreflightError("VQA_IMAGE_ROOT_REQUIRED")
    files, source_hash = source_identity(repo)
    data = {"train_file": str(train_file), "train_sha256": file_hash(train_file),
            "expected_count": count, "partition": "train", "order_rule": "frozen_file_order",
            "sample_order_sha256": canonical_hash([r["sample_id"] for r in rows]),
            "image_root": str(_absolute(image_root)) if image_root is not None else None}
    run_id = run_id or "v6-native-preflight-" + cell_id
    config = {"schema_version": SCHEMA, **cell, "job_id": run_id, "run_id": run_id,
              "max_length": 2048, "fixed_template_date": "2026-11-14",
              "local_files_only": True, "trust_remote_code": False, "device": "cpu",
              "model": model, "data": data, "output_dir": str(output_dir),
              "runtime_versions": runtime_versions(),
              "source_files": files, "source_hash": source_hash,
              "model_hash": canonical_hash(model), "data_hash": canonical_hash(data)}
    config["config_hash"] = canonical_hash(config)
    return config


def _validate(config):
    if config.get("schema_version") != SCHEMA or config.get("max_length") != 2048:
        raise PreflightError("PREFLIGHT_SCHEMA_OR_CONTEXT_MISMATCH")
    if config.get("device") != "cpu" or config.get("local_files_only") is not True or config.get("trust_remote_code") is not False:
        raise PreflightError("CPU_LOCAL_NATIVE_ONLY_REQUIRED")
    if config["config_hash"] != canonical_hash({k: v for k, v in config.items() if k != "config_hash"}):
        raise PreflightError("CONFIG_HASH_MISMATCH")
    if config["runtime_versions"] != runtime_versions():
        raise PreflightError("NATIVE_RUNTIME_VERSIONS_CHANGED")
    for key, value in cell_spec(config["cell_id"]).items():
        if config.get(key) != value:
            raise PreflightError("CELL_IDENTITY_MISMATCH")
    actual_files, actual_hash = source_identity()
    if config["source_files"] != actual_files or config["source_hash"] != actual_hash:
        raise PreflightError("SOURCE_IDENTITY_MISMATCH")
    if _prepared(config["model"]["preparation_receipt"]) != config["model"] or canonical_hash(config["model"]) != config["model_hash"]:
        raise PreflightError("MODEL_IDENTITY_MISMATCH")
    if config["model"]["model_id"] != config["model_id"]:
        raise PreflightError("CELL_MODEL_MISMATCH")
    if canonical_hash(config["data"]) != config["data_hash"] or file_hash(config["data"]["train_file"]) != config["data"]["train_sha256"]:
        raise PreflightError("TRAIN_BYTES_CHANGED")
    for key in IDENTITY_KEYS:
        expected = os.environ.get("MEDVLM_SUPERVISOR_" + key.upper())
        if expected is not None and config[key] != expected:
            raise PreflightError("SUPERVISOR_IDENTITY_MISMATCH_" + key.upper())
    _absolute(config["output_dir"])


@contextmanager
def _cpu_only():
    # Must happen before importing torch/transformers. is_initialized never starts CUDA.
    if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized():
        raise PreflightError("CUDA_ALREADY_INITIALIZED")
    values = {"CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        import torch
        if torch.cuda.is_initialized():
            raise PreflightError("CUDA_ALREADY_INITIALIZED")
        yield torch
        if torch.cuda.is_initialized():
            raise PreflightError("CUDA_INITIALIZED_DURING_PREFLIGHT")
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _load_native(config):
    from transformers import AutoConfig
    from .training.models import load_processor
    from .training.v6_native import audit_native_architecture, freeze_native_template, verify_native_no_thinking
    local = {"model_id": config["model"]["snapshot_path"], "revision": config["model"]["revision"],
             "cache_dir": config["model"]["cache_dir"], "local_files_only": True}
    architecture = AutoConfig.from_pretrained(local["model_id"], local_files_only=True, trust_remote_code=False)
    processor = load_processor(local, architecture)
    audit = audit_native_architecture(architecture, config["cell_id"])
    template = freeze_native_template(processor, fixed_date=config["fixed_template_date"])
    thinking = verify_native_no_thinking(processor, architecture.model_type)
    return processor, architecture, {"architecture": audit, "template": template, "thinking": thinking}


def _checked_encode(row, processor, architecture, task, image_root, **kwargs):
    from .training.rendering import encode_sample
    # encode_sample already encodes the intact sequence before checking its bound.
    # Retain its safe measurements for overflow rows too; never truncate inputs.
    encoded, metadata = encode_sample(row, processor, architecture, max_length=sys.maxsize,
                                      image_root=image_root, **kwargs)
    ids, mask, labels = (encoded[k] for k in ("input_ids", "attention_mask", "labels"))
    if ids.shape != labels.shape or ids.shape != mask.shape or ids.ndim != 2 or ids.shape[0] != 1:
        raise PreflightError("INVALID_NATIVE_TOKEN_SHAPES")
    for tensor in encoded.values():
        if hasattr(tensor, "device") and tensor.device.type != "cpu":
            raise PreflightError("NON_CPU_PROCESSOR_TENSOR")
    start = metadata["assistant_start"]
    if not bool((labels[:, :start] == -100).all()) or not bool((labels[mask == 0] == -100).all()):
        raise PreflightError("PROMPT_OR_PAD_LOSS_LEAK")
    supervised = labels != -100
    if not bool((labels[supervised] == ids[supervised]).all()) or not bool(supervised[:, 1:].any()):
        raise PreflightError("TARGET_TOKEN_MASK_MISMATCH")
    for token_id in metadata["image_token_ids"]:
        if not bool((labels[ids == token_id] == -100).all()):
            raise PreflightError("IMAGE_TOKEN_LOSS_LEAK")
    if bool((labels[:, start:] == -100).any()) or bool((mask[:, start:] == 0).any()):
        raise PreflightError("TARGET_CONTAINS_MASKED_SPECIAL_OR_PAD_TOKEN")
    visual = [key for key in encoded if any(marker in key for marker in ("pixel", "image_grid", "image_sizes", "image_embeds"))]
    if task == "vqa":
        if not metadata["images"] or not visual or metadata["image_tokens"] <= 0:
            raise PreflightError("NATIVE_IMAGE_INPUT_OR_TOKENS_MISSING")
    elif metadata["images"] or visual or metadata["image_tokens"]:
        raise PreflightError("TEXT_PATH_HAS_IMAGE_INPUT")
    metadata.update(tensor_shapes={key: list(value.shape) for key, value in encoded.items() if hasattr(value, "shape")},
                    tensor_dtypes={key: str(value.dtype) for key, value in encoded.items() if hasattr(value, "dtype")},
                    all_tensors_cpu=True, prompt_image_pad_mask_verified=True,
                    full_target_preserved=True, visual_tensor_keys=visual)
    if ids.shape[-1] > 2048:
        raise IneligibleRow("CONTEXT_OVERFLOW", metadata)
    return encoded, metadata


def _fixtures(processor, architecture, task, directory):
    from .training.rendering import pad_text_batch
    tokenizer = getattr(processor, "tokenizer", processor)
    row = {"sample_id": "engineering-native-fixture", "task_type": "medical_qa",
           "input": {"question": "Return the word ready.", "images": []},
           "supervision": {"answer": "ready"}}
    if task == "vqa":
        from PIL import Image
        path = directory / "engineering-image.png"
        Image.new("RGB", (64, 64), (31, 79, 127)).save(path)
        row.update(task_type="medical_vqa")
        row["input"]["images"] = [str(path)]
    first, single = _checked_encode(row, processor, architecture, task, None)
    second, multi = _checked_encode(row, processor, architecture, task, None,
                                   history=[{"role": "user", "content": "Say start."},
                                            {"role": "assistant", "content": "start"}])
    projected = [{key: encoded[key] for key in ("input_ids", "attention_mask", "labels")}
                 for encoded in (first, second)]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    padded = pad_text_batch(projected, pad_id)
    if not bool((padded["labels"][padded["attention_mask"] == 0] == -100).all()):
        raise PreflightError("FIXTURE_PAD_MASK_FAILED")
    for original, padded_labels in zip(projected, padded["labels"]):
        if not bool(original["labels"][0].equal(padded_labels[:original["labels"].shape[-1]])):
            raise PreflightError("FIXTURE_EOS_PAD_SUPERVISION_CHANGED")
    errors = {}
    for name, answer, required in (("empty_target", "", "supervised answer cannot be empty"),
                                   ("overflow", "ready " * 8192, "CONTEXT_OVERFLOW")):
        fixture = {**row, "supervision": {"answer": answer}}
        try:
            _checked_encode(fixture, processor, architecture, task, None)
        except ValueError as exc:
            if str(exc) != required:
                raise PreflightError("FIXTURE_" + name.upper() + "_WRONG_REJECTION") from None
            errors[name] = "EXPECTED_REJECTION_VERIFIED"
        else:
            raise PreflightError("FIXTURE_" + name.upper() + "_NOT_REJECTED")
    return {"single_turn": single, "multiturn_last_assistant_only": multi,
            "padding": {"status": "VERIFIED", "scope": "token_tensor_projection_only" if task == "vqa" else "text_batch",
                        "eos_equals_pad": tokenizer.eos_token_id == pad_id,
                        "original_labels_preserved": True}, **errors}


def _reason(exc):
    value = str(exc)
    known = {"CONTEXT_OVERFLOW": "CONTEXT_OVERFLOW", "supervised answer cannot be empty": "EMPTY_TARGET",
             "ZERO_SUPERVISED_TOKENS": "ZERO_SUPERVISED_TOKENS", "question is required": "EMPTY_QUESTION",
             "image hash changed": "IMAGE_HASH_CHANGED", "VQA requires an actual verified image": "MISSING_IMAGE"}
    if isinstance(exc, PreflightError):
        return value
    if value.startswith("ASSISTANT_BOUNDARY_MISMATCH"):
        return "ASSISTANT_BOUNDARY_MISMATCH"
    if value.startswith("unsupported model input fields:"):
        return "UNSUPPORTED_INPUT_FIELDS"
    return known.get(value)


def _stratum(row):
    result = {key: row.get(key) for key in ("source_id", "language", "task_type")}
    metadata = row.get("metadata") or {}
    result.update({key: metadata.get(key) for key in ("q_type", "answer_type")})
    result["classification_label"] = None
    if row.get("input", {}).get("options"):
        label = str(row.get("supervision", {}).get("option_id", ""))
        result["classification_label"] = label if re.fullmatch(r"[A-Za-z0-9]{1,8}", label) else "UNRECOGNIZED_OPTION_ID"
    elif row.get("task_type") == "context_qa":
        label = str(row.get("supervision", {}).get("answer", "")).strip().lower()
        result["classification_label"] = label if label in {"yes", "no", "maybe"} else "UNRECOGNIZED_CLASS_LABEL"
    return result


def _completed_output(config):
    """Verify the immutable directory marker, including a rename/receipt crash gap."""
    output = Path(config["output_dir"])
    marker = output / "completion.json"
    if not marker.is_file():
        raise PreflightError("INCOMPLETE_OUTPUT_REQUIRES_NEW_OUTPUT_IDENTITY")
    receipt = _read(marker)
    if receipt.get("status") != "SUCCEEDED" or any(receipt.get(k) != config[k] for k in IDENTITY_KEYS):
        raise PreflightError("EXISTING_RECEIPT_IDENTITY_MISMATCH")
    artifacts = receipt.get("artifacts", [])
    if len(artifacts) != len(OUTPUT_FILES) or {Path(a["path"]).name for a in artifacts} != OUTPUT_FILES:
        raise PreflightError("EXISTING_RECEIPT_ARTIFACT_SET_MISMATCH")
    if {p.name for p in output.iterdir()} != OUTPUT_FILES | {"completion.json"}:
        raise PreflightError("EXISTING_OUTPUT_ARTIFACT_SET_MISMATCH")
    for artifact in artifacts:
        path = Path(artifact["path"])
        if path.parent != output or file_hash(path) != artifact["sha256"]:
            raise PreflightError("EXISTING_RECEIPT_ARTIFACT_CHANGED")
    if _read(output / "config.json") != config:
        raise PreflightError("EXISTING_CONFIG_BYTES_MISMATCH")
    return {**receipt, "artifacts": artifacts + [{"path": str(marker), "sha256": file_hash(marker)}]}


def _reuse(receipt_path, config):
    if receipt_path.exists():
        receipt = _read(receipt_path)
        if receipt.get("status") != "SUCCEEDED" or any(receipt.get(k) != config[k] for k in IDENTITY_KEYS):
            raise PreflightError("EXISTING_RECEIPT_IDENTITY_MISMATCH")
        completed = _completed_output(config)
        if receipt != completed:
            raise PreflightError("EXISTING_RECEIPT_MARKER_MISMATCH")
        return completed
    if Path(config["output_dir"]).exists():
        completed = _completed_output(config)
        atomic_json(receipt_path, completed)
        return completed
    return None


def run_preflight(config_path, receipt_path):
    """Run exactly one cell. A success means a completed eligibility scan only."""
    config_path, receipt_path = _absolute(config_path), _absolute(receipt_path)
    config = _read(config_path)
    output = _absolute(config["output_dir"])
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_path = output.with_name(output.name + ".worker.lock")
    with lock_path.open("a") as lock:
        os.chmod(lock_path, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _validate(config)
        prior = _reuse(receipt_path, config)
        if prior is not None:
            return prior
        staging = Path(tempfile.mkdtemp(prefix=output.name + ".candidate-", dir=output.parent))
        try:
            with _cpu_only() as torch:
                processor, architecture, audit = _load_native(config)
                fixture_directory = staging / "engineering"
                fixture_directory.mkdir(mode=0o700)
                fixtures = _fixtures(processor, architecture, config["task"], fixture_directory)
                shutil.rmtree(fixture_directory)
                rows = _train_rows(config["data"]["train_file"], config["data"]["expected_count"], config["task"])
                if canonical_hash([row["sample_id"] for row in rows]) != config["data"]["sample_order_sha256"]:
                    raise PreflightError("TRAIN_SAMPLE_ORDER_CHANGED")
                eligible, excluded, reasons, strata = [], [], Counter(), {}
                sequence_tokens = target_tokens = max_tokens = 0
                progress_path = os.environ.get("MEDVLM_SUPERVISOR_PROGRESS_PATH")
                with (staging / "eligibility.jsonl").open("x", encoding="utf-8") as stream:
                    os.chmod(staging / "eligibility.jsonl", 0o600)
                    for position, row in enumerate(rows):
                        stratum = _stratum(row)
                        key = json.dumps(stratum, sort_keys=True)
                        group = strata.setdefault(key, {**stratum, "total": 0, "eligible": 0, "excluded": 0, "reasons": {}})
                        group["total"] += 1
                        record = {"sample_id": row["sample_id"], "position": position, **stratum,
                                  "input_sha256": canonical_hash(row["input"]),
                                  "target_sha256": canonical_hash(row["supervision"])}
                        try:
                            encoded, metadata = _checked_encode(row, processor, architecture, config["task"], config["data"]["image_root"])
                            del encoded
                            record.update(eligible=True, native=metadata)
                            eligible.append(row["sample_id"])
                            group["eligible"] += 1
                            sequence_tokens += metadata["sequence_tokens"]
                            target_tokens += metadata["supervised_tokens_after_shift"]
                            max_tokens = max(max_tokens, metadata["sequence_tokens"])
                        except (ValueError, OSError) as exc:
                            if isinstance(exc, PreflightError):
                                raise
                            if str(exc) == "image hash changed":
                                raise PreflightError("TRAIN_IMAGE_HASH_CHANGED") from None
                            reason = _reason(exc)
                            if reason not in {"CONTEXT_OVERFLOW", "EMPTY_TARGET", "EMPTY_QUESTION",
                                              "ZERO_SUPERVISED_TOKENS", "ASSISTANT_BOUNDARY_MISMATCH",
                                              "UNSUPPORTED_INPUT_FIELDS"}:
                                raise PreflightError("TRAIN_IMAGE_ASSET_UNAVAILABLE" if isinstance(exc, OSError)
                                                     else "UNEXPECTED_NATIVE_ENCODING_FAILURE") from None
                            record.update(eligible=False, reason=reason, exception_type=type(exc).__name__)
                            if isinstance(exc, IneligibleRow):
                                record["native"] = exc.native
                            excluded.append(row["sample_id"])
                            reasons[reason] += 1
                            group["excluded"] += 1
                            group["reasons"][reason] = group["reasons"].get(reason, 0) + 1
                        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                        if progress_path and (position == 0 or (position + 1) % 25 == 0):
                            atomic_json(progress_path, {"status": "RUNNING", "stage": "native_train_eligibility",
                                "cell_id": config["cell_id"], "processed_rows": position + 1,
                                "total_rows": len(rows), "last_progress_at": utcnow(), "optimizer_step": 0,
                                "total_updates": 0, "weights_loaded": False, "uses_gpu": False})
                    stream.flush()
                    os.fsync(stream.fileno())
                # Bind completion to unchanged inputs after the full scan too.
                _validate(config)
                summary = {"schema_version": "medvlm.v6.native-eligibility.v1", "cell_id": config["cell_id"],
                    "scope": "TRAIN_PREFLIGHT_ELIGIBILITY_ONLY", "evidence_level": "CPU_META_ONLY",
                    "original_count": len(rows), "eligible_count": len(eligible), "excluded_count": len(excluded),
                    "eligible_ids": eligible, "excluded_ids": excluded, "eligible_ids_sha256": canonical_hash(eligible),
                    "excluded_ids_sha256": canonical_hash(excluded), "sample_order_sha256": config["data"]["sample_order_sha256"],
                    "excluded_reasons": dict(reasons), "strata": list(strata.values()),
                    "max_sequence_tokens": max_tokens, "eligible_sequence_tokens": sequence_tokens,
                    "eligible_supervised_tokens_after_shift": target_tokens, "max_length": 2048,
                    "full_targets_preserved": True, "common_train_freeze_decided": False,
                    "dev_test_read": False, "weights_loaded": False, "gpu_smoke_verified": False,
                    "training_performed": False, "runtime": {"cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                    "cuda_initialized": torch.cuda.is_initialized(), "local_files_only": True,
                    "trust_remote_code": False, "versions": config["runtime_versions"],
                    "weight_bytes_rehashed": False, "weight_load_performed": False}, "finished_at": utcnow()}
                for name, value in (("config.json", config), ("architecture.json", audit),
                                    ("fixtures.json", fixtures), ("summary.json", summary)):
                    atomic_json(staging / name, value)
            artifacts = [{"path": str(output / name), "sha256": file_hash(staging / name)} for name in sorted(OUTPUT_FILES)]
            result = {"schema_version": "medvlm.v6.native-preflight-receipt.v1", "status": "SUCCEEDED",
                **{key: config[key] for key in IDENTITY_KEYS}, "cell_id": config["cell_id"],
                "scope": "TRAIN_PREFLIGHT_ELIGIBILITY_ONLY", "evidence_level": "CPU_META_ONLY",
                "model_revision": config["model"]["revision"], "artifacts": artifacts,
                "eligible_count": len(eligible), "excluded_count": len(excluded), "finished_at": utcnow(),
                "weights_loaded": False, "gpu_smoke_verified": False, "training_performed": False,
                "common_train_freeze_decided": False, "campaign_terminal": False}
            atomic_json(staging / "completion.json", result)
            os.rename(staging, output)
            parent_fd = os.open(output.parent, os.O_DIRECTORY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
            result = _completed_output(config)
            atomic_json(receipt_path, result)
            return result
        finally:
            if staging.exists():
                shutil.rmtree(staging)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args(argv)
    try:
        result = run_preflight(args.config, args.receipt)
    except Exception as exc:
        # Native template/processor errors can contain private input. Never echo them.
        failure = {"status": "FAILED", "scope": "TRAIN_PREFLIGHT_ELIGIBILITY_ONLY",
                   "exception_type": type(exc).__name__, "finished_at": utcnow()}
        if isinstance(exc, PreflightError):
            failure["reason_code"] = str(exc)
        print(json.dumps(failure), file=sys.stderr)
        return 1
    print(json.dumps({key: result[key] for key in ("status", "scope", "cell_id", "eligible_count", "excluded_count")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
