"""Optional matched image/no-image diagnostics; never produce training samples."""
from __future__ import annotations

import json
from pathlib import Path

from ..runtime import atomic_json, canonical_hash, file_hash
from ..training.rendering import RENDERER_VERSION, user_text
from .metrics import aggregate, paired_compare, score_rows, vqa_subgroups
from .runner import load_protocol, validated_predictions

DIAGNOSTIC_VERSION = "medvlm.no-image-diagnostic.v1"
TEXT_TENSOR_KEYS = {"input_ids", "attention_mask", "token_type_ids", "position_ids", "cache_position", "rope_deltas"}


class UnsupportedImageDiagnostic(ValueError):
    """Native text-only ablation unsupported; no blank-image fallback is allowed."""


def derive_no_image_protocol(parent_protocol_path, output, cap=32, seed=20261023):
    if type(cap) is not int or not 1 <= cap <= 32:
        raise ValueError("optional no-image diagnostic cap must be between 1 and 32")
    parent, rows = load_protocol(parent_protocol_path)
    if parent.get("image_condition", "image") != "image":
        raise ValueError("parent must be an image-present protocol")
    eligible = [r for r in rows if r["task_type"] == "medical_vqa" and parent["image_files"].get(r["sample_id"])]
    selected = sorted(eligible, key=lambda r: canonical_hash([seed, r["sample_id"]]))[:cap]
    if not selected:
        raise ValueError("no verified image VQA samples in parent protocol")
    ids = [r["sample_id"] for r in selected]
    protocol = {k: v for k, v in parent.items() if k != "protocol_hash"}
    protocol.update({"diagnostic_version": DIAGNOSTIC_VERSION, "diagnostic_only": True,
                     "training_data_changed": False, "image_condition": "no_image",
                     "parent_protocol_hash": parent["protocol_hash"],
                     "parent_protocol_file_hash": file_hash(parent_protocol_path),
                     "parent_reference_digest": parent["reference_digest"],
                     "parent_sample_count": len(parent["sample_ids"]),
                     "seed": seed, "cap": cap, "source_caps": None, "sample_ids": ids,
                     "selection_policy": "medical_vqa_parent_ids_sha256(seed,sample_id)_ascending_before_results",
                     "input_digests": {sid: parent["input_digests"][sid] for sid in ids},
                     "image_files": {sid: parent["image_files"][sid] for sid in ids},
                     "reference_digest": canonical_hash([r["supervision"] for r in selected]),
                     "encoding_policy": "same_user_text_native_template_without_image_content_or_pixel_tensors",
                     "unsupported_policy": "record_unsupported_ablation_never_insert_blank_images"})
    protocol["protocol_hash"] = canonical_hash(protocol)
    destination = Path(output)
    if destination.exists():
        if json.loads(destination.read_text()) != protocol:
            raise ValueError("refusing to overwrite a frozen diagnostic protocol")
    else:
        atomic_json(destination, protocol)
    return protocol


def encode_no_image_sample(sample, processor, model_config=None, *, max_length=2048,
                           include_labels=False, image_root=None, history=None):
    """Keep the exact VQA user_text, but send only text to the native processor.

    Original input/image identity stays in the protocol. This function never
    opens an image, forwards an image keyword, or reads supervision/metadata.
    Generation can still fail in the model and must be recorded as unsupported.
    """
    if include_labels or history:
        raise ValueError("no-image diagnostic cannot train or include conversation history")
    if sample["task_type"] != "medical_vqa":
        raise ValueError("no-image diagnostic requires medical_vqa")
    restricted = {key: sample[key] for key in ("sample_id", "task_type", "input")}
    text = user_text(restricted)
    messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    try:
        rendered = processor.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=False)
        encoded = dict(processor(text=[rendered], return_tensors="pt", padding=False, add_special_tokens=False))
    except Exception as exc:
        raise UnsupportedImageDiagnostic("NATIVE_NO_IMAGE_PROCESSOR_UNSUPPORTED:" + type(exc).__name__) from exc
    if "input_ids" not in encoded:
        raise UnsupportedImageDiagnostic("NATIVE_NO_IMAGE_INPUT_IDS_MISSING")
    if "mm_token_type_ids" in encoded:
        # Qwen's native processor emits this text-aligned mask even without images.
        # Only an all-zero, exactly aligned mask is text-only evidence.
        mask = encoded["mm_token_type_ids"]
        try:
            same_shape = mask.shape == encoded["input_ids"].shape
            all_zero = (mask == 0).all()
            all_zero = bool(all_zero.item() if hasattr(all_zero, "item") else all_zero)
        except (AttributeError, TypeError, ValueError) as exc:
            raise UnsupportedImageDiagnostic("INVALID_NO_IMAGE_MODALITY_MASK") from exc
        if not same_shape or not all_zero:
            raise UnsupportedImageDiagnostic("NON_TEXT_NO_IMAGE_MODALITY_MASK")
    unexpected = set(encoded) - TEXT_TENSOR_KEYS - {"mm_token_type_ids"}
    if unexpected or any(any(word in key.casefold() for word in ("pixel", "image", "vision", "video")) for key in encoded):
        raise UnsupportedImageDiagnostic("UNEXPECTED_NO_IMAGE_TENSORS:" + ",".join(sorted(unexpected or set(encoded))))
    if encoded["input_ids"].shape[-1] > max_length:
        raise UnsupportedImageDiagnostic("CONTEXT_OVERFLOW")
    return encoded, {"sample_id": sample["sample_id"], "image_condition": "no_image", "images": [],
                     "vision_ingested": False, "pixel_tensor_keys": [], "diagnostic_only": True,
                     "renderer_version": DIAGNOSTIC_VERSION, "parent_renderer_version": RENDERER_VERSION,
                     "input_hash": canonical_hash(sample["input"]), "encoded_text_hash": canonical_hash(text),
                     "messages_hash": canonical_hash(messages), "native_generation_supported": "not_yet_tested"}


def unsupported_no_image_prediction(sample_id, protocol, reason_code="NATIVE_NO_IMAGE_GENERATION_UNSUPPORTED"):
    if protocol.get("image_condition") != "no_image" or sample_id not in protocol["input_digests"]:
        raise ValueError("sample is not scheduled in a no-image protocol")
    return {"sample_id": sample_id, "input_hash": protocol["input_digests"][sample_id],
            "attempted": True, "parsed_answer": None, "status": "unsupported_input",
            "finish_reason": "unsupported_ablation", "error_reason": reason_code,
            "image_condition": "no_image", "vision_ingested": False, "pixel_tensor_keys": [],
            "diagnostic_only": True, "timing": {"ttft_seconds": None}}


def _matching_runs(image_path, no_image_path, parent, diagnostic):
    left = json.loads((Path(image_path).parent / "run_manifest.json").read_text())
    right = json.loads((Path(no_image_path).parent / "run_manifest.json").read_text())
    if left.get("protocol_hash") != parent["protocol_hash"] or right.get("protocol_hash") != diagnostic["protocol_hash"]:
        raise ValueError("diagnostic run/protocol identity mismatch")
    for key in ("model_id", "model_revision", "config_hash", "precision", "generation"):
        if key not in left or key not in right or left[key] != right[key]:
            raise ValueError("unmatched diagnostic model/generation: " + key)
    for key in ("adapter_digest", "adapter_path"):
        if left.get(key) != right.get(key):
            raise ValueError("unmatched diagnostic adapter: " + key)
    return left, right


def compare_image_conditions(parent_protocol_path, no_image_protocol_path, image_predictions_path,
                             no_image_predictions_path, output, *, resamples=2000, seed=20261023):
    parent, _ = load_protocol(parent_protocol_path)
    diagnostic, samples = load_protocol(no_image_protocol_path)
    if diagnostic.get("image_condition") != "no_image" or diagnostic.get("parent_protocol_hash") != parent["protocol_hash"]:
        raise ValueError("no-image protocol does not derive from this image protocol")
    if diagnostic["parent_protocol_file_hash"] != file_hash(parent_protocol_path):
        raise ValueError("parent protocol file changed")
    ids = set(diagnostic["sample_ids"])
    if not ids <= set(parent["sample_ids"]):
        raise ValueError("diagnostic contains IDs outside parent schedule")
    for sid in ids:
        if diagnostic["input_digests"][sid] != parent["input_digests"][sid] or diagnostic["image_files"][sid] != parent["image_files"][sid]:
            raise ValueError("diagnostic changed original input/image identity")
    image_run, no_image_run = _matching_runs(image_predictions_path, no_image_predictions_path, parent, diagnostic)
    image_predictions = [p for p in validated_predictions(image_predictions_path, parent, image_run) if p["sample_id"] in ids]
    no_image_predictions = validated_predictions(no_image_predictions_path, diagnostic, no_image_run)
    for pred in image_predictions:
        if pred.get("image_condition", "image") != "image":
            raise ValueError("image run prediction has another image condition")
        if pred["status"] == "answered" and pred.get("vision_ingested") is not True:
            raise ValueError("image-present answered prediction lacks pixel-ingestion evidence")
    for pred in no_image_predictions:
        if pred.get("image_condition") != "no_image" or pred.get("vision_ingested") is not False or pred.get("pixel_tensor_keys") != []:
            raise ValueError("no-image prediction lacks explicit no-pixel evidence")
    left = score_rows(samples, image_predictions)
    right = score_rows(samples, no_image_predictions)
    result = {"diagnostic_version": DIAGNOSTIC_VERSION, "diagnostic_only": True,
              "training_data_changed": False, "parent_protocol_hash": parent["protocol_hash"],
              "no_image_protocol_hash": diagnostic["protocol_hash"], "scheduled": len(samples),
              "image_run": image_run["run_id"], "no_image_run": no_image_run["run_id"],
              "image": aggregate(left), "no_image": aggregate(right),
              "image_subgroups": vqa_subgroups(left), "no_image_subgroups": vqa_subgroups(right),
              "paired_image_minus_no_image": paired_compare(right, left, resamples=resamples, seed=seed),
              "unsupported_ablation_count": sum(p.get("status") == "unsupported_input" for p in no_image_predictions),
              "human_review": "not_performed", "clinical_accuracy": None,
              "interpretation": "Matched lexical VQA diagnostic. Equal answers do not prove pixels are ignored; missing/unsupported attempts remain in denominators."}
    atomic_json(output, result)
    return result
