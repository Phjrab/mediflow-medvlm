"""Official Hub revisions, native architecture dispatch, and language-only LoRA."""
from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import re
import urllib.error
import urllib.request

from ..runtime import atomic_json, canonical_hash, file_hash, utcnow

VISION_MARKERS = ("vision", "visual", "projector", "multi_modal", "multimodal", "merger", "patch_embed")


def versions():
    result = {}
    for name in ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "safetensors", "huggingface-hub", "Pillow"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def hub_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "mediflow-medvlm-research/0.1"})
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            from huggingface_hub import get_token
            token = get_token()
        except ImportError:
            pass
    if token:
        request.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def probe_model(model_id, output=None, *, cache_dir=None, tokenizer_check=False):
    """Metadata/tokenizer only: never claim forward/backward from a config probe."""
    result = {"model_id": model_id, "source_url": "https://huggingface.co/" + model_id,
              "checked_at": utcnow(), "status": "PREPARED_NOT_RUN", "evidence_level": "CPU_TESTED",
              "weights_loaded": False, "forward_tested": False, "backward_tested": False,
              "versions": versions()}
    try:
        info = hub_json("https://huggingface.co/api/models/" + model_id)
        sha = info["sha"]
        result.update({"revision": sha, "gated": info.get("gated", False),
                       "declared_license": (info.get("cardData") or {}).get("license"),
                       "parameter_count": (info.get("safetensors") or {}).get("total"),
                       "files": [x["rfilename"] for x in info.get("siblings", [])]})
        config = hub_json("https://huggingface.co/" + model_id + "/resolve/" + sha + "/config.json")
        result.update({"model_type": config.get("model_type"), "architectures": config.get("architectures"),
                       "config_sha256": canonical_hash(config), "access_verified_config": True})
        try:
            import transformers
            native = [name for name in config.get("architectures", []) if hasattr(transformers, name)]
            result["native_classes_available"] = native
            if not native:
                result.update(status="BLOCKED_HARDWARE", reason_code="UNSUPPORTED_RUNTIME")
            if tokenizer_check:
                processor = load_processor({"model_id": model_id, "revision": sha, "cache_dir": cache_dir}, config)
                result["processor_class"] = type(processor).__name__
                tokenizer = getattr(processor, "tokenizer", processor)
                result["chat_template_sha256"] = canonical_hash(processor.chat_template or tokenizer.chat_template)
                from .rendering import encode_sample, pad_text_batch
                fixture = {"schema_version": "medvlm.dataset.v1", "sample_id": "engineering-tokenizer-check",
                           "task_type": "medical_qa", "input": {"question": "Return the word ready.", "images": []},
                           "supervision": {"answer": "ready"}, "derivation": "engineering_fixture"}
                encoded, check = encode_sample(fixture, processor, config)
                multi, multichek = encode_sample(fixture, processor, config, history=[{"role": "user", "content": "Say start."}, {"role": "assistant", "content": "start"}])
                pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                padded = pad_text_batch([encoded, multi], pad_id)
                assert (padded["labels"][padded["attention_mask"] == 0] == -100).all()
                result["mask_checks"] = {"text": check, "multiturn": multichek, "padding": "passed", "vlm_image": "not_run"}
        except ImportError as exc:
            result.update(status="BLOCKED_HARDWARE", reason_code="UNSUPPORTED_RUNTIME",
                          native_probe_reason=type(exc).__name__, error=str(exc)[:500],
                          evidence_level="IMPLEMENTED_ONLY")
    except urllib.error.HTTPError as exc:
        result.update(status="BLOCKED_ACCESS" if exc.code in (401, 403) else "FAILED_REPRODUCIBLE",
                      reason_code="AUTH_REQUIRED" if exc.code in (401, 403) else f"HTTP_{exc.code}", http_status=exc.code)
    except (OSError, urllib.error.URLError) as exc:
        result.update(status="NETWORK_UNREACHABLE", reason_code=type(exc).__name__)
    except Exception as exc:
        result.update(status="FAILED_REPRODUCIBLE", reason_code=type(exc).__name__, error=str(exc)[:500])
    if output:
        atomic_json(output, result)
    return result


def _common(config):
    return {"revision": config["revision"], "cache_dir": config.get("cache_dir"),
            "trust_remote_code": False, "local_files_only": config.get("local_files_only", False)}


def load_processor(config, model_config=None):
    from transformers import AutoProcessor, AutoTokenizer
    if model_config is None:
        from transformers import AutoConfig
        model_config = AutoConfig.from_pretrained(config["model_id"], **_common(config)).to_dict()
    elif hasattr(model_config, "to_dict"):
        model_config = model_config.to_dict()
    is_vlm = bool(model_config.get("vision_config"))
    cls = AutoProcessor if is_vlm else AutoTokenizer
    processor = cls.from_pretrained(config["model_id"], **_common(config))
    tokenizer = getattr(processor, "tokenizer", processor)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return processor


def native_loader(model_config):
    import transformers
    # Dispatch via declared native architecture, not a universal CausalLM guess.
    for name in model_config.architectures or []:
        if re.fullmatch(r"[A-Za-z0-9_]+", name) and hasattr(transformers, name):
            cls = getattr(transformers, name)
            if hasattr(cls, "from_pretrained"):
                return cls
    raise ValueError("UNSUPPORTED_RUNTIME: no native declared architecture is installed")


def model_file_digests(config):
    """Hash actually cached native weights/configs without a second download."""
    from huggingface_hub import try_to_load_from_cache
    def resolve(name):
        if Path(config["model_id"]).is_dir():
            path = Path(config["model_id"]) / name
            return path if path.is_file() else None
        found = try_to_load_from_cache(config["model_id"], name, cache_dir=config.get("cache_dir"), revision=config["revision"])
        return Path(found) if isinstance(found, str) else None
    names = {"config.json", "generation_config.json", "model.safetensors", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "preprocessor_config.json", "processor_config.json"}
    index = resolve("model.safetensors.index.json")
    if index:
        names.update(json.loads(index.read_text())["weight_map"].values())
    files = {}
    for name in sorted(names):
        path = resolve(name)
        if path:
            files[name] = {"sha256": file_hash(path), "bytes": path.stat().st_size}
    return {"files": files, "sha256": canonical_hash(files)}


def load_model(config, training=False, adapter_path=None):
    """Return (native model or PEFT model, processor, reproducibility metadata)."""
    import torch
    from transformers import AutoConfig, BitsAndBytesConfig
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    device = config.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU_NOT_AVAILABLE")
    if device == "cpu" and not config.get("synthetic_fixture", False):
        raise RuntimeError("large-model CPU fallback is disabled; use a training GPU")
    model_config = AutoConfig.from_pretrained(config["model_id"], **_common(config))
    loader = native_loader(model_config)
    dtype_name = config.get("dtype", "bfloat16" if device == "cuda" and torch.cuda.is_bf16_supported() else "float16")
    if device == "cpu":
        dtype_name = "float32"
    dtype = getattr(torch, dtype_name)
    kwargs = {**_common(config), "dtype": dtype, "device_map": {"": 0 if device == "cuda" else "cpu"},
              "attn_implementation": config.get("attn_implementation", "sdpa")}
    quantization = config.get("quantization", "none")
    if quantization == "nf4":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_use_double_quant=True,
                                                           bnb_4bit_compute_dtype=dtype)
    elif quantization != "none":
        raise ValueError("quantization must be none or nf4")
    model = loader.from_pretrained(config["model_id"], **kwargs)
    processor = load_processor(config, model_config)
    targets = []
    if training:
        if quantization == "nf4":
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=config.get("gradient_checkpointing", True))
        if config.get("gradient_checkpointing", True):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
        model.config.use_cache = False
    if adapter_path:
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=training)
        targets = sorted(model.peft_config["default"].target_modules)
    elif training:
        suffixes = config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        targets = [name for name, module in model.named_modules()
                   if name.split(".")[-1] in suffixes and not any(marker in name.lower() for marker in VISION_MARKERS)
                   and hasattr(module, "weight") and len(module.weight.shape) == 2]
        if not targets:
            raise ValueError("no language LoRA target modules matched native model")
        lora = LoraConfig(r=config.get("lora_rank", 8), lora_alpha=config.get("lora_alpha", 16),
                          lora_dropout=config.get("lora_dropout", 0.05), target_modules=targets,
                          task_type="CAUSAL_LM", bias="none", modules_to_save=None)
        model = get_peft_model(model, lora)
        model.peft_config["default"].revision = config["revision"]
    if training:
        for name, parameter in model.named_parameters():
            if any(marker in name.lower() for marker in VISION_MARKERS) and parameter.requires_grad:
                raise ValueError("vision/projector freeze policy violated: " + name)
            if parameter.requires_grad and "lora_" not in name:
                raise ValueError("base parameter unexpectedly trainable: " + name)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    tokenizer = getattr(processor, "tokenizer", processor)
    metadata = {"model_id": config["model_id"], "revision": config["revision"],
                "loader": loader.__name__, "processor_class": type(processor).__name__,
                "model_type": model_config.model_type, "dtype": dtype_name, "quantization": quantization,
                "exact_target_modules": sorted(targets), "trainable_parameters": trainable,
                "frozen_parameters": total - trainable, "vision_encoder_frozen": True,
                "chat_template_sha256": canonical_hash(processor.chat_template or tokenizer.chat_template),
                "native_files": model_file_digests(config),
                "versions": versions()}
    if training:
        model.train()
    else:
        model.eval()
    return model, processor, metadata
