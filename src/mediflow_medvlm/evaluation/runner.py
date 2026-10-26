"""Immutable evaluation selection, resumable inference and paired summaries."""
from __future__ import annotations
import json
import os
from pathlib import Path
import platform
import time

from ..runtime import (atomic_json, canonical_hash, file_hash, utcnow,
                       GPULease, BudgetLedger, append_event, resolve_resource_policy)
from .metrics import parse_answer, score_rows, aggregate, paired_compare, korean_summary, vqa_subgroups, subject_name

PROTOCOL_VERSION = "medvlm.eval.v1"
GENERATION_BUDGET = {"medical_qa": 128, "context_qa": 16, "medical_mcq": 16, "medical_vqa": 32}


def verify_adapter_parent(adapter_path, config, model_meta):
    """Bind evaluation to the saved training base, including old checkpoint schemas."""
    checkpoint = json.loads((Path(adapter_path) / "checkpoint.json").read_text())
    parent = checkpoint["identity"]
    for key in ("model_id", "revision", "quantization"):
        if parent[key] != config[key]:
            raise ValueError("adapter parent mismatch: " + key)
    if parent["template_sha256"] != model_meta["chat_template_sha256"]:
        raise ValueError("adapter native template changed")
    run_manifest = Path(adapter_path).parent.parent / "manifest.json"
    previous = json.loads(run_manifest.read_text())
    if previous["model"]["native_files"]["sha256"] != model_meta["native_files"]["sha256"]:
        raise ValueError("adapter native base files changed")
    for name, digest in checkpoint["artifacts"]["files"].items():
        if file_hash(Path(adapter_path) / name) != digest:
            raise ValueError("adapter checkpoint artifact changed: " + name)
    return {"checkpoint_identity": parent, "training_manifest_sha256": file_hash(run_manifest)}


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def freeze_protocol(dataset_paths, output, *, split="test", cap=64, seed=20261023, caps=None):
    """Freeze IDs, input/supervision digests; selection never looks at answers."""
    paths = [Path(p).resolve() for p in dataset_paths]
    rows = [row for p in paths for row in read_jsonl(p)]
    if any(row["split"] != split for row in rows):
        raise ValueError("protocol split differs from canonical rows")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate sample IDs")
    rows.sort(key=lambda row: canonical_hash([seed, row["sample_id"]]))
    if caps is None:
        selected = rows[:cap] if cap else rows
    else:
        from collections import Counter
        counts = Counter(); selected = []
        for row in rows:
            rule = caps.get(row["source_id"], 0)
            group = row["source_id"]
            if isinstance(rule, dict):
                group += ":" + str(subject_name(row))
                limit = rule["per_subject"]
            else:
                limit = rule
            if counts[group] < limit:
                selected.append(row); counts[group] += 1
    from ..training.rendering import image_paths
    image_files = {}
    for row in selected:
        assets, metadata = image_paths(row)
        image_files[row["sample_id"]] = [{"path": str(p.resolve()), "sha256": m["sha256"]} for p, m in zip(assets, metadata)]
    identity = {"protocol_version": PROTOCOL_VERSION, "split": split, "seed": seed,
                "selection_policy": "sha256(seed,sample_id)_ascending_before_results", "cap": cap,
                "source_caps": caps,
                "files": {str(p): file_hash(p) for p in paths},
                "sample_ids": [row["sample_id"] for row in selected],
                "input_digests": {row["sample_id"]: canonical_hash(row["input"]) for row in selected},
                "image_files": image_files,
                "reference_digest": canonical_hash([row["supervision"] for row in selected]),
                "allowed_input_fields": ["question", "context", "options", "images"],
                "max_new_tokens": GENERATION_BUDGET, "do_sample": False,
                "thinking": False, "parser_version": "strict-label.v1",
                "clinical_review": "not_performed", "test_used_for_selection": False}
    identity["protocol_hash"] = canonical_hash(identity)
    destination = Path(output)
    if destination.exists():
        if json.loads(destination.read_text()) != identity:
            raise ValueError("refusing to overwrite a frozen protocol")
    else:
        atomic_json(destination, identity)
    return identity


def load_protocol(path):
    protocol = json.loads(Path(path).read_text())
    check = dict(protocol); digest = check.pop("protocol_hash")
    if canonical_hash(check) != digest:
        raise ValueError("protocol digest mismatch")
    indexed = {}
    for file, expected in protocol["files"].items():
        if file_hash(file) != expected:
            raise ValueError("frozen dataset file changed")
        for row in read_jsonl(file):
            indexed[row["sample_id"]] = row
    rows = [indexed[sid] for sid in protocol["sample_ids"]]
    if canonical_hash([r["supervision"] for r in rows]) != protocol["reference_digest"]:
        raise ValueError("frozen reference changed")
    for row in rows:
        if canonical_hash(row["input"]) != protocol["input_digests"][row["sample_id"]]:
            raise ValueError("frozen input changed")
        for image in protocol["image_files"][row["sample_id"]]:
            if file_hash(image["path"]) != image["sha256"]:
                raise ValueError("frozen image bytes changed")
    return protocol, rows


def validated_predictions(path, protocol, identity=None):
    predictions = read_jsonl(path) if Path(path).exists() else []
    seen = set()
    for pred in predictions:
        sid = pred["sample_id"]
        if sid in seen or sid not in protocol["input_digests"]:
            raise ValueError("duplicate or unscheduled prediction")
        seen.add(sid)
        if pred.get("input_hash") != protocol["input_digests"][sid]:
            raise ValueError("prediction input identity changed")
        if identity:
            for field, expected in (("run_id", identity["run_id"]), ("pair_id", identity["pair_id"]), ("model_identity", identity["identity_hash"])):
                if pred.get(field) != expected:
                    raise ValueError("prediction belongs to another run: " + field)
    return predictions


def verify_completed_evaluation(manifest, protocol, root, native_files_digest):
    """Reuse a success only while all scheduled predictions and measured artifacts exist."""
    root = Path(root)
    expected = manifest.get("artifact_hashes", {})
    if set(expected) != {"predictions.jsonl", "metrics.json"}:
        raise ValueError("completed evaluation lacks artifact integrity receipts")
    for name, digest in expected.items():
        if file_hash(root / name) != digest:
            raise ValueError("completed evaluation artifact changed: " + name)
    predictions = validated_predictions(root / "predictions.jsonl", protocol, manifest)
    if {p["sample_id"] for p in predictions} != set(protocol["sample_ids"]):
        raise ValueError("completed evaluation has missing scheduled predictions")
    if manifest["model"]["native_files"]["sha256"] != native_files_digest:
        raise ValueError("completed evaluation native base cache changed")
    return manifest


def summarize(protocol_path, predictions_path, output):
    protocol, rows = load_protocol(protocol_path)
    manifest_path = Path(predictions_path).parent / "run_manifest.json"
    identity = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    predictions = validated_predictions(predictions_path, protocol, identity)
    scores = score_rows(rows, predictions)
    if protocol.get("normalization_policy") == "nfkc-casefold-whitespace-terminal-punctuation.v2":
        from .v2 import normalize_vqa
        scores = score_rows(rows, predictions, vqa_normalizer=normalize_vqa)
    result = {"protocol_hash": protocol["protocol_hash"], "scheduled": len(rows),
              "predictions": len(predictions), "cohorts": aggregate(scores),
              "human_review": "not_performed", "scores": scores}
    result["korean_original_eval"] = korean_summary(result["cohorts"])
    result["vqa_subgroups"] = vqa_subgroups(scores)
    atomic_json(output, result)
    return result


def evaluate(config_path, protocol_path, output_dir, *, adapter_path=None, pair_id=None):
    import torch
    from ..training.models import load_model, model_file_digests
    from ..training.rendering import encode_sample, RENDERER_VERSION
    from ..training.runner import _git_identity, balanced_validation_rows
    from ..training.config import load_frozen_data
    config = json.loads(Path(config_path).read_text())
    protocol, samples = load_protocol(protocol_path)
    no_image = protocol.get("image_condition") == "no_image"
    if no_image:
        from .image_diagnostics import encode_no_image_sample, DIAGNOSTIC_VERSION
        encoder = encode_no_image_sample
    else:
        encoder = encode_sample
    if protocol.get("protocol_version") == "medvlm.eval.v2":
        from functools import partial
        from .v2 import encode_evaluation_sample, V2_RENDERER
        encoder = partial(encode_evaluation_sample, prompt_policy=protocol["prompt_policy"],
                          image_condition="no_image" if no_image else "image")
        active_renderer = V2_RENDERER
    else:
        active_renderer = DIAGNOSTIC_VERSION if no_image else RENDERER_VERSION
    root = Path(output_dir); root.mkdir(parents=True, exist_ok=True)
    os.chmod(root, 0o700)
    data_root = Path(config.get("data_root", os.environ.get("MEDVLM_DATA_ROOT", root.parent.parent)))
    runtime_root = Path(config.get("runtime_root", str(data_root / "runtime")))
    run_id = root.name
    identity = {"config_hash": canonical_hash(config), "protocol_hash": protocol["protocol_hash"],
                "model_id": config["model_id"], "model_revision": config["revision"],
                "adapter_path": str(Path(adapter_path).resolve()) if adapter_path else None,
                "pair_id": pair_id or config.get("pair_id", run_id),
                "renderer_version": active_renderer,
                "image_condition": "no_image" if no_image else "image"}
    if adapter_path:
        from ..runtime import directory_digest
        identity["adapter_digest"] = directory_digest(adapter_path)["sha256"]
    identity["identity_hash"] = canonical_hash(identity)
    path = root / "run_manifest.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous["identity_hash"] != identity["identity_hash"]:
            raise ValueError("resume identity mismatch")
        if previous["status"] == "SUCCEEDED":
            return verify_completed_evaluation(previous, protocol, root, model_file_digests(config)["sha256"])
    manifest = {**identity, "run_id": run_id, "started_at": utcnow(), "status": "RUNNING",
                "evidence_level": "IMPLEMENTED_ONLY", "generation": {"do_sample": False, "budgets": protocol["max_new_tokens"]},
                "scheduled": len(samples), "precision": config.get("quantization", "none"),
                "test_used_for_selection": False, "ttft_seconds": None, "energy_joules": None,
                "code": _git_identity(), "host": {"os": platform.system(), "architecture": platform.machine()},
                "measurement": "direct native inference; E2E includes image preprocessing, tokenization, generation, and parsing; load and validation warmup excluded; generated tokens / E2E is not decode-only throughput"}
    predictions_path = root / "predictions.jsonl"
    completed = {r["sample_id"] for r in validated_predictions(predictions_path, protocol, manifest)}
    atomic_json(path, manifest)
    resource_policy = resolve_resource_policy(config)
    manifest["resource_policy"] = resource_policy.as_dict()
    ledger = BudgetLedger(runtime_root / "gpu_budget.json", policy=resource_policy)
    start = time.monotonic()
    try:
        if not torch.cuda.is_available():
            manifest.update(status="BLOCKED_HARDWARE", reason_code="GPU_NOT_AVAILABLE")
            return manifest
        with GPULease(runtime_root / "medvlm_gpu.lock", run_id):
            ledger.start(config["model_id"], run_id, "evaluation")
            try:
                model, processor, model_meta = load_model(config, training=False, adapter_path=adapter_path)
                model.eval(); tokenizer = getattr(processor, "tokenizer", processor)
                manifest["model"] = model_meta
                manifest["gpu"] = {"name": torch.cuda.get_device_name(), "total_bytes": torch.cuda.get_device_properties(0).total_memory,
                                   "torch_cuda_version": torch.version.cuda}
                if adapter_path:
                    manifest["verified_parent"] = verify_adapter_parent(adapter_path, config, model_meta)
                manifest["load_seconds"] = time.monotonic() - start
                # Warm up on fixed validation inputs, never scored test examples.
                validation, _ = load_frozen_data(config)
                warmup_start = time.monotonic(); warmups = []
                for original in balanced_validation_rows(validation["validation"], 2, config.get("seed", 20261023)):
                    sample = {k: original[k] for k in ("sample_id", "task_type", "input")}
                    try:
                        encoded, _ = encoder(sample, processor, model.config, include_labels=False,
                                                   max_length=config.get("max_length", 2048), image_root=config.get("image_root", str(data_root)))
                        encoded = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in encoded.items()}
                        with torch.inference_mode():
                            model.generate(**encoded, do_sample=False, max_new_tokens=4, max_time=30,
                                           pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
                        torch.cuda.synchronize(); warmups.append({"sample_id": original["sample_id"], "status": "SUCCEEDED"})
                    except (ValueError, FileNotFoundError) as error:
                        warmups.append({"sample_id": original["sample_id"], "status": "FAILED_REPRODUCIBLE", "reason_code": type(error).__name__})
                manifest["warmup"] = {"split": "validation", "items": warmups, "seconds": time.monotonic()-warmup_start}
                torch.cuda.reset_peak_memory_stats()
                for original in samples:
                    if original["sample_id"] in completed:
                        continue
                    if not ledger.can_continue(config["model_id"], 90):
                        manifest.update(status="SKIPPED_BUDGET", reason_code="EVALUATION_BUDGET")
                        break
                    # This restricted object is the only data sent to the predictor.
                    sample = {k: original[k] for k in ("sample_id", "task_type", "input")}
                    sample["input"] = {**sample["input"], "images": protocol["image_files"][sample["sample_id"]]}
                    pred = {"sample_id": original["sample_id"], "run_id": run_id, "pair_id": identity["pair_id"],
                            "input_hash": protocol["input_digests"][sample["sample_id"]], "model_identity": identity["identity_hash"],
                            "attempted": True, "parsed_answer": None, "status": "load_error",
                            "image_condition": identity["image_condition"], "vision_ingested": False, "pixel_tensor_keys": [],
                            "generation_settings": {"do_sample": False, "max_new_tokens": protocol["max_new_tokens"][sample["task_type"]]},
                            "timing": {"ttft_seconds": None}}
                    began = time.monotonic()
                    try:
                        encoded, meta = encoder(sample, processor, model.config, include_labels=False,
                                                       max_length=config.get("max_length", 2048), image_root=config.get("image_root", str(data_root)))
                        encoded = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in encoded.items()}
                        prefix = encoded["input_ids"].shape[-1]
                        pred["prompt_token_sha256"] = canonical_hash({key: encoded[key].tolist() for key in
                            ("input_ids", "attention_mask", "token_type_ids", "mm_token_type_ids") if key in encoded})
                        torch.cuda.synchronize()
                        with torch.inference_mode():
                            generated = model.generate(**encoded, **pred["generation_settings"], max_time=120,
                                                       pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
                        torch.cuda.synchronize()
                        tokens = generated[0, prefix:]
                        text = tokenizer.decode(tokens, skip_special_tokens=True)
                        answer, status = parse_answer(text, sample["task_type"], sample["input"].get("options"))
                        elapsed = time.monotonic() - began
                        finish = "length" if len(tokens) >= pred["generation_settings"]["max_new_tokens"] else "stop"
                        if protocol.get("protocol_version") == "medvlm.eval.v2":
                            from .v2 import generation_finish
                            eos_ids=model.generation_config.eos_token_id
                            last_token=int(tokens[-1]) if len(tokens) else None
                            finish=generation_finish(len(tokens),last_token,pred["generation_settings"]["max_new_tokens"],eos_ids)
                            pred["generation_termination"]={"last_token_id":last_token,"eos_token_ids":eos_ids,
                                                            "policy":"eos_before_length.v2"}
                        if elapsed >= 120 and finish != "length":
                            status, answer, finish = "timeout", None, "timeout"
                        pred.update(status=status, parsed_answer=answer, raw_output=text, finish_reason=finish,
                                    output_tokens=len(tokens), vision_ingested=bool(meta.get("images")) and any("pixel" in k for k in encoded),
                                    pixel_tensor_keys=[k for k in encoded if "pixel" in k])
                        pred["timing"].update(e2e_seconds=elapsed, generated_tokens_per_e2e_second=len(tokens)/elapsed)
                    except torch.OutOfMemoryError:
                        pred.update(status="oom", error_reason="CUDA_OOM", finish_reason="error")
                        torch.cuda.empty_cache()
                    except (ValueError, FileNotFoundError) as error:
                        pred.update(status="image_missing" if isinstance(error, FileNotFoundError) else "unsupported_input",
                                    error_reason=str(error)[:200], finish_reason="error")
                    except Exception as error:
                        pred.update(status="unsupported_input" if no_image else "load_error",
                                    error_reason=("NO_IMAGE_" if no_image else "")+type(error).__name__,
                                    finish_reason="unsupported_ablation" if no_image else "error")
                    pred["timing"].setdefault("e2e_seconds", time.monotonic()-began)
                    append_event(predictions_path, pred)
                if manifest["status"] == "RUNNING":
                    manifest.update(status="SUCCEEDED", evidence_level="REAL_MEDICAL_EVALUATED")
                manifest["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                manifest["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            finally:
                ledger.finish()
    except Exception as error:
        manifest.update(status="FAILED_REPRODUCIBLE", reason_code=type(error).__name__)
        # Exception text can contain user paths or source data; retain locally only.
        (root / "failure.txt").write_text(str(error), encoding="utf-8")
    finally:
        manifest["finished_at"] = utcnow(); manifest["wall_seconds"] = time.monotonic()-start
        summary = summarize(protocol_path, predictions_path, root / "metrics.json")
        manifest["predictions"] = summary["predictions"]
        manifest["artifact_hashes"] = {name: file_hash(root / name) for name in ("predictions.jsonl", "metrics.json") if (root / name).is_file()}
        if not any(row["status"] in {"answered", "abstained", "invalid_format"} for row in summary["scores"]) and manifest["status"] == "SUCCEEDED":
            manifest.update(status="BLOCKED_HARDWARE" if no_image else "FAILED_REPRODUCIBLE", evidence_level="IMPLEMENTED_ONLY",
                            reason_code="UNSUPPORTED_NO_IMAGE" if no_image else "NO_COMPLETED_INFERENCE")
        atomic_json(path, manifest)
    return manifest


def compare(base_dir, tuned_dir, output):
    base = json.loads((Path(base_dir)/"run_manifest.json").read_text())
    tuned = json.loads((Path(tuned_dir)/"run_manifest.json").read_text())
    for field in ("pair_id", "protocol_hash", "config_hash", "model_revision", "precision", "renderer_version"):
        if base[field] != tuned[field]:
            raise ValueError("unmatched pair policy: " + field)
    for field in ("native_files", "chat_template_sha256", "dtype"):
        if base.get("model", {}).get(field) != tuned.get("model", {}).get(field):
            raise ValueError("unmatched loaded native model: " + field)
    left = json.loads((Path(base_dir)/"metrics.json").read_text())
    right = json.loads((Path(tuned_dir)/"metrics.json").read_text())
    result = {"pair_id": base["pair_id"], "base_run": base["run_id"], "tuned_run": tuned["run_id"],
              "preliminary_single_seed": True, "comparison": paired_compare(left["scores"], right["scores"]),
              "base": left["cohorts"], "tuned": right["cohorts"]}
    atomic_json(output, result)
    return result
