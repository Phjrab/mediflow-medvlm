"""Canonical input allowlist and exact native-template assistant span masking."""
from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse, unquote

from ..runtime import canonical_hash, file_hash

RENDERER_VERSION = "medvlm.native-chat.v1"
ALLOWED_INPUT_FIELDS = {"question", "context", "options", "images"}


def user_text(sample):
    source = sample["input"]
    unknown = set(source) - ALLOWED_INPUT_FIELDS
    if unknown:
        raise ValueError("unsupported model input fields: " + ",".join(sorted(unknown)))
    question = str(source.get("question", "")).strip()
    if not question:
        raise ValueError("question is required")
    parts = []
    if source.get("context"):
        context = source["context"]
        if isinstance(context, list):
            context = "\n".join(str(x) for x in context)
        parts.append("Context:\n" + str(context))
    parts.append("Question:\n" + question)
    options = source.get("options")
    if options:
        items = options.items() if isinstance(options, dict) else enumerate(options)
        parts.append("Options:\n" + "\n".join(f"{key}. {value}" for key, value in items))
        parts.append("Return only the option identifier.")
    elif sample["task_type"] == "context_qa":
        parts.append("Answer using the provided context. Return only yes, no, or maybe.")
    elif sample["task_type"] == "medical_vqa":
        parts.append("Answer the question using the supplied image. Give a concise answer.")
    else:
        parts.append("Answer the educational medical question. Do not invent source evidence.")
    return "\n\n".join(parts)


def assistant_target(sample):
    supervision = sample["supervision"]
    value = supervision.get("option_id") if sample["input"].get("options") else supervision.get("answer")
    if value is None or not str(value).strip():
        raise ValueError("supervised answer cannot be empty")
    return str(value)


def image_paths(sample, image_root=None):
    paths, metadata = [], []
    for image in sample["input"].get("images", []):
        value = image if isinstance(image, str) else next((image.get(k) for k in ("local_path", "path", "uri") if image.get(k)), None)
        if not value:
            raise ValueError("image needs a validated local path")
        if value.startswith("file://"):
            value = unquote(urlparse(value).path)
        if "://" in value:
            raise ValueError("training images must be local verified assets")
        path = Path(value)
        if not path.is_absolute() and image_root:
            path = Path(image_root) / path
        digest = file_hash(path)
        expected = image.get("sha256") if isinstance(image, dict) else None
        if not expected and len(sample["input"].get("images", [])) == 1:
            provenance = sample.get("provenance", {})
            expected = provenance.get("image_sha256") or provenance.get("local_asset_sha256")
        if expected and digest != expected:
            raise ValueError("image hash changed")
        paths.append(path)
        metadata.append({"sha256": digest, "order": len(paths) - 1})
    if sample["task_type"] == "medical_vqa" and not paths:
        raise ValueError("VQA requires an actual verified image")
    return paths, metadata


def render_messages(sample, include_answer=False, image_root=None, history=None):
    """No provenance, source metadata, rationale, or gold enters the user message."""
    paths, _ = image_paths(sample, image_root)
    text = user_text(sample)
    content = ([{"type": "image", "image": str(path)} for path in paths] +
               [{"type": "text", "text": text}]) if paths else text
    messages = list(history or []) + [{"role": "user", "content": content}]
    if include_answer:
        messages.append({"role": "assistant", "content": assistant_target(sample)})
    return messages


def image_token_ids(tokenizer, model_config):
    ids = set()
    def walk(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if isinstance(key, str) and any(word in key for word in ("image", "vision", "boi", "eoi")) and key.endswith(("token_id", "token_index")) and isinstance(child, int):
                    ids.add(child)
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    walk(model_config.to_dict() if hasattr(model_config, "to_dict") else model_config)
    for token, index in tokenizer.get_added_vocab().items():
        if any(word in token.lower() for word in ("image", "vision", "<img", "</img", "fake_token_around")):
            ids.add(index)
    return ids


def native_encode(processor, messages, *, generation_prompt, images=None):
    """Retain every processor tensor (pixel values, grids, masks, etc.)."""
    rendered = processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=generation_prompt,
                                             enable_thinking=False)
    kwargs = {"text": [rendered], "return_tensors": "pt", "padding": False, "add_special_tokens": False}
    if images:
        kwargs["images"] = images
    return dict(processor(**kwargs))


def encode_sample(sample, processor, model_config, *, max_length=2048, image_root=None,
                  include_labels=True, history=None):
    """Mask by a verified token-identical prompt prefix, never answer-token length.

    The last assistant only is supervised, including its termination tokens.
    Earlier turns, prompt images and padding remain -100. Overlong examples are
    rejected intact; questions and gold answers are never silently truncated.
    """
    tokenizer = getattr(processor, "tokenizer", processor)
    paths, metadata = image_paths(sample, image_root)
    opened = []
    if paths:
        from PIL import Image
        for path, item in zip(paths, metadata):
            with Image.open(path) as original:
                item.update({"width": original.width, "height": original.height})
                opened.append(original.convert("RGB"))
    prompt_messages = render_messages(sample, False, image_root, history)
    prompt = native_encode(processor, prompt_messages, generation_prompt=True, images=opened)
    if not include_labels:
        if prompt["input_ids"].shape[-1] > max_length:
            raise ValueError("CONTEXT_OVERFLOW")
        return prompt, {"sample_id": sample["sample_id"], "images": metadata,
                        "input_hash": canonical_hash(sample["input"])}
    full_messages = prompt_messages + [{"role": "assistant", "content": assistant_target(sample)}]
    encoded = native_encode(processor, full_messages, generation_prompt=False, images=opened)
    ids = encoded["input_ids"]
    boundary = prompt["input_ids"].shape[-1]
    if ids.shape[-1] > max_length:
        raise ValueError("CONTEXT_OVERFLOW")
    if boundary >= ids.shape[-1] or not ids[0, :boundary].equal(prompt["input_ids"][0]):
        raise ValueError("ASSISTANT_BOUNDARY_MISMATCH: native prompt is not an exact input_ids prefix")
    labels = ids.clone()
    labels[:, :boundary] = -100
    labels[encoded["attention_mask"] == 0] = -100
    special_ids = image_token_ids(tokenizer, model_config)
    for index in special_ids:
        labels[ids == index] = -100
    count = int((labels[:, 1:] != -100).sum().item())
    if count == 0:
        raise ValueError("ZERO_SUPERVISED_TOKENS")
    encoded["labels"] = labels
    return encoded, {"sample_id": sample["sample_id"], "assistant_start": boundary,
                     "sequence_tokens": ids.shape[-1], "supervised_tokens_after_shift": count,
                     "image_token_ids": sorted(special_ids),
                     "image_tokens": sum(int((ids == x).sum()) for x in special_ids),
                     "images": metadata, "prefix_verified": True,
                     "renderer_version": RENDERER_VERSION}


def pad_text_batch(examples, pad_token_id):
    """Right padding by positions, preserving supervised EOS even if EOS == PAD."""
    import torch
    maximum = max(item["input_ids"].shape[-1] for item in examples)
    keys = set(examples[0])
    if not keys <= {"input_ids", "attention_mask", "labels", "token_type_ids", "mm_token_type_ids"}:
        raise ValueError("use the native processor for multimodal batching")
    output = {}
    for key in keys:
        fill = pad_token_id if key == "input_ids" else -100 if key == "labels" else 0
        output[key] = torch.stack([torch.nn.functional.pad(item[key][0], (0, maximum - item[key].shape[-1]), value=fill) for item in examples])
    return output
