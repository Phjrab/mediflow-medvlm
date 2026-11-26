"""One local photo through a saved VLM, with no reference-label requirement."""
from __future__ import annotations

import argparse
from contextlib import ExitStack, nullcontext
import json
import os
from pathlib import Path
import sys
import time
import uuid

from .common import absolute_path
from .dataset import _image_info, classification_question
from .training import _run_path
from ..runtime import (BudgetExceeded, BudgetLedger, GPULease, ResourceBusy, atomic_json,
                       canonical_hash, directory_digest, file_hash, resolve_resource_policy, utcnow)
from ..training.config import assert_resume_identity, load_config as load_training_config

REQUEST_SCHEMA = "medvlm.kiosk-inference-request.v1"
RESULT_SCHEMA = "medvlm.kiosk-inference-result.v1"


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _selected_run(run_dir, data_root):
    """Verify saved artifacts without reading train/validation examples."""
    run_dir = Path(run_dir).resolve()
    prepared_path = run_dir / "prepared.json"
    prepared = _read(prepared_path)
    config_path = Path(prepared["training_config"]).resolve()
    output = Path(prepared["output_dir"]).resolve()
    if config_path != run_dir / "training.json" or output != run_dir / "output":
        raise ValueError("Prepared run paths do not belong to this kiosk run.")
    config = load_training_config(config_path)
    config_hash = canonical_hash(config)
    if config_hash != prepared.get("training_config_sha256"):
        raise ValueError("Prepared training config changed.")
    if Path(config["output_dir"]).resolve() != output or Path(config["data_root"]).resolve() != Path(data_root).resolve():
        raise ValueError("Training output/data_root differs from the prepared kiosk run.")
    if config.get("device", "cuda") != "cuda":
        raise ValueError("Saved VLM prediction requires the configured CUDA runtime.")
    manifest_path, selection_path = output / "manifest.json", output / "selection.json"
    manifest, selection = _read(manifest_path), _read(selection_path)
    if manifest.get("status") != "SUCCEEDED" or selection.get("reload_verified") is not True:
        raise ValueError("Run train and reload for the selected checkpoint before predicting a new photo.")
    if manifest.get("config_hash") != config_hash:
        raise ValueError("Training manifest config identity differs from the prepared recipe.")
    checkpoint_path = Path(selection["checkpoint"]).resolve()
    if not checkpoint_path.is_relative_to(output / "checkpoints"):
        raise ValueError("Selected checkpoint is outside this training run.")
    checkpoint = assert_resume_identity(manifest["identity"], checkpoint_path,
                                        legacy_source_snapshot=manifest.get("legacy_source_snapshot"),
                                        legacy_manifest=manifest.get("legacy_identity_provenance"))
    if checkpoint["identity"].get("config_sha256") != config_hash:
        raise ValueError("Selected checkpoint belongs to another training config.")
    if selection.get("identity") != manifest["identity"]:
        raise ValueError("Selected checkpoint and training manifest identities differ.")
    checkpoint_digest = directory_digest(checkpoint_path)["sha256"]
    if selection.get("adapter", {}).get("digest") != checkpoint_digest:
        raise ValueError("Selected checkpoint directory digest changed.")
    binding = {"prepared_sha256": file_hash(prepared_path), "training_config_sha256": config_hash,
               "training_manifest_sha256": file_hash(manifest_path), "selection_sha256": file_hash(selection_path),
               "checkpoint_sha256": checkpoint_digest}
    return config, str(checkpoint_path), binding


def prepare_prediction(config, run_name, image_path, source_id, question=None):
    """Write a concrete subprocess request; never load a VLM or create labels."""
    run_dir = _run_path(config, run_name).resolve()
    source = next((row for row in config["sources"] if row["source_id"] == source_id), None)
    if source is None:
        raise ValueError("Choose a configured source_id for the photo's camera/modality.")
    question = classification_question(source["modality"]) if question is None else question
    if not isinstance(question, str) or not question.strip() or len(question) > 8000:
        raise ValueError("Prediction question must contain 1-8000 characters.")
    training, checkpoint, binding = _selected_run(run_dir, config["data_root"])
    image_path = absolute_path(image_path, Path.cwd())
    info = _image_info(image_path)
    output = run_dir / "predictions" / uuid.uuid4().hex
    request = {"schema_version": REQUEST_SCHEMA, "created_at": utcnow(), "run_name": run_name,
               "run_dir": str(run_dir), "data_root": str(Path(config["data_root"]).resolve()),
               "source_id": source_id, "modality": source["modality"], "question": question.strip(),
               "image": {"path": str(image_path), **info}, "checkpoint": checkpoint, "binding": binding,
               "model_id": training["model_id"], "model_revision": training["revision"],
               "output_dir": str(output), "generation": {"do_sample": False, "max_new_tokens": 64, "max_time": 120}}
    request["request_sha256"] = canonical_hash(request)
    output.mkdir(parents=True)
    if os.name == "posix":
        output.chmod(0o700)
    path = output / "request.json"
    atomic_json(path, request)
    return {"status": "PREPARED_NOT_RUN", "argv": [sys.executable, "-m", "mediflow_medvlm.kiosk.inference", "--request", str(path)],
            "output_dir": str(output), "result_file": str(output / "result.json"), "request_file": str(path),
            "request_sha256": request["request_sha256"], "gpu_started": False}


def _request(path):
    path = Path(path).resolve()
    request = _read(path)
    check = dict(request)
    expected = check.pop("request_sha256", None)
    if request.get("schema_version") != REQUEST_SCHEMA or canonical_hash(check) != expected:
        raise ValueError("Prediction request identity changed.")
    output = Path(request["output_dir"]).resolve()
    run_dir = Path(request["run_dir"]).resolve()
    if path != output / "request.json" or output.parent != run_dir / "predictions":
        raise ValueError("Prediction request/output location differs from its run.")
    generation = request.get("generation", {})
    if (generation.get("do_sample") is not False or type(generation.get("max_new_tokens")) is not int
            or not 1 <= generation["max_new_tokens"] <= 256 or generation.get("max_time") != 120):
        raise ValueError("Unsupported bounded greedy generation settings.")
    return request


def _verify_request_inputs(request):
    config, checkpoint, binding = _selected_run(request["run_dir"], request["data_root"])
    if binding != request["binding"] or checkpoint != request["checkpoint"]:
        raise ValueError("Saved model or training artifacts changed after prediction preparation.")
    if file_hash(request["image"]["path"]) != request["image"]["sha256"]:
        raise ValueError("Prediction image bytes changed after preparation.")
    if config["model_id"] != request["model_id"] or config["revision"] != request["model_revision"]:
        raise ValueError("Prediction model identity changed.")
    return config


def _inference_sample(request):
    # No supervision, teacher label, confidence, reviewer, or answer enters this object.
    return {"sample_id": "predict_" + request["request_sha256"][:32],
            "source_id": "kiosk_" + request["source_id"], "task_type": "medical_vqa",
            "input": {"question": request["question"], "images": [{"local_path": request["image"]["path"],
                                                                         "sha256": request["image"]["sha256"]}]}}


def _generate(request, config, ledger, policy, started):
    import torch
    from ..evaluation.runner import verify_adapter_parent
    from ..training.models import load_model, load_processor
    from ..training.rendering import encode_sample

    # A completed training run has already cached the base; prediction never downloads it.
    local_config = {**config, "local_files_only": True}
    model, _, metadata = load_model(local_config, training=False, adapter_path=request["checkpoint"])
    parent = verify_adapter_parent(request["checkpoint"], config, metadata)
    saved_config = {**local_config, "model_id": request["checkpoint"], "native_snapshot_path": None}
    processor = load_processor(saved_config, model.config)
    tokenizer = getattr(processor, "tokenizer", processor)
    template = canonical_hash(processor.chat_template or tokenizer.chat_template)
    if template != parent["checkpoint_identity"]["template_sha256"]:
        raise ValueError("Saved processor template differs from the trained checkpoint.")
    encoded, image_meta = encode_sample(_inference_sample(request), processor, model.config,
                                        include_labels=False, max_length=config.get("max_length", 2048))
    pixel_keys = [name for name in encoded if "pixel" in name]
    if not image_meta.get("images") or not pixel_keys:
        raise ValueError("The saved model did not receive image pixel tensors.")
    encoded = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in encoded.items()}
    if not ledger.can_continue(config["model_id"], 1) or not policy.can_continue(time.monotonic() - started):
        raise BudgetExceeded("Prediction budget exhausted after model loading.")
    available = [120.0]
    remaining = ledger.remaining(config["model_id"])
    if remaining is not None:
        available.append(remaining)
    for scope in ("run", "job"):
        deadline = policy.deadline(started, scope=scope)
        if deadline is not None:
            available.append(deadline - time.monotonic())
    maximum_time = min(available)
    if maximum_time <= 0:
        raise BudgetExceeded("Prediction wall-time limit reached.")
    generation = {**request["generation"], "max_time": maximum_time,
                  "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id}
    prefix = encoded["input_ids"].shape[-1]
    before = time.monotonic()
    generation_context = torch.autocast("cuda", dtype=torch.bfloat16) if config.get("v6_native") else nullcontext()
    with torch.inference_mode(), generation_context:
        generated = model.generate(**encoded, **generation)
    torch.cuda.synchronize()
    tokens = generated[0, prefix:]
    answer = tokenizer.decode(tokens, skip_special_tokens=True).strip()
    if not len(tokens) or not answer:
        raise ValueError("The saved VLM generated no answer tokens.")
    elapsed = time.monotonic() - before
    if elapsed >= maximum_time:
        raise TimeoutError("Prediction generation reached its time limit.")
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else (eos or [])
    finish = "stop" if int(tokens[-1]) in eos else (
        "length" if len(tokens) >= generation["max_new_tokens"] else "stop")
    return {"answer": answer, "generated_tokens": len(tokens), "finish_reason": finish,
            "vision_ingested": True, "pixel_tensor_keys": pixel_keys, "generation_seconds": elapsed,
            "model": metadata, "verified_parent": parent, "effective_generation": generation}


def predict_request(path):
    """Worker entrypoint, intended to run in a fresh foreground subprocess."""
    request = _request(path)
    result_path = Path(request["output_dir"]) / "result.json"
    if result_path.exists():
        raise ValueError("Prediction result already exists; prepare a new prediction request.")
    result = {"schema_version": RESULT_SCHEMA, "status": "RUNNING", "started_at": utcnow(), "pid": os.getpid(),
              "request_sha256": request["request_sha256"], "run_name": request["run_name"],
              "source_id": request["source_id"], "modality": request["modality"], "image": request["image"],
              "question": request["question"], "checkpoint": request["checkpoint"],
              "checkpoint_sha256": request["binding"]["checkpoint_sha256"], "answer": None,
              "clinical_truth_created": False, "reference_label_used": False,
              "review_status": "unreviewed_model_output", "evidence_level": "IMPLEMENTED_ONLY"}
    atomic_json(result_path, result)
    started = time.monotonic()
    try:
        config = _verify_request_inputs(request)
        policy = resolve_resource_policy(config)
        result["resource_policy"] = policy.as_dict()
        if os.name != "posix":
            raise RuntimeError("GPU prediction uses the Linux/WSL worker.")
        import torch
        if not torch.cuda.is_available():
            result.update(status="BLOCKED_HARDWARE", reason_code="GPU_NOT_AVAILABLE")
        else:
            runtime = Path(request["data_root"]) / "runtime"
            with ExitStack() as stack:
                stack.enter_context(GPULease(runtime / "medvlm_gpu.lock", "kiosk-predict-" + request["request_sha256"][:12]))
                ledger = BudgetLedger(runtime / "gpu_budget.json", policy=policy).start(config["model_id"], config["run_id"], "prediction")
                stack.callback(ledger.finish)
                generated = _generate(request, config, ledger, policy, started)
                _verify_request_inputs(request)
                result.update(generated, status="SUCCEEDED", evidence_level="REAL_MODEL_INFERENCE")
    except ImportError as exc:
        result.update(status="BLOCKED_HARDWARE", reason_code="DEPENDENCY_MISSING", error=str(exc)[:500])
    except BudgetExceeded as exc:
        result.update(status="SKIPPED_BUDGET", reason_code="PREDICTION_BUDGET", error=str(exc)[:500])
    except ResourceBusy as exc:
        result.update(status="RESOURCE_BUSY", reason_code="GPU_LEASE_BUSY", error=str(exc)[:500])
    except KeyboardInterrupt:
        result.update(status="INTERRUPTED", reason_code="USER_INTERRUPTED")
    except Exception as exc:
        result.update(status="FAILED_REPRODUCIBLE", reason_code=type(exc).__name__, error=str(exc)[:500])
    finally:
        result.update(finished_at=utcnow(), wall_seconds=time.monotonic() - started)
        atomic_json(result_path, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run one local photo through a saved kiosk VLM.")
    parser.add_argument("--request", required=True)
    args = parser.parse_args(argv)
    try:
        result = predict_request(args.request)
    except (ValueError, OSError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps({key: result.get(key) for key in ("status", "reason_code", "generated_tokens")}, ensure_ascii=False))
    return 0 if result["status"] == "SUCCEEDED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
