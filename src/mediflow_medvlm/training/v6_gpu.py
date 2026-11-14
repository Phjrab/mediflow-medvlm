"""Opt-in V6 native NF4 runtime; importing this module never initializes CUDA.

The caller owns the GPU lock, immutable execution identity, TRAIN-only samples,
optimizer, artifacts, and fresh-process reload. Historical loaders are unchanged.
CPU/meta tests of these helpers do not establish CUDA or training capability.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

from ..runtime import canonical_hash, file_hash
from .v6_native import audit_native_architecture, freeze_native_template, verify_native_no_thinking


SCHEMA = "medvlm.v6.native-gpu-contract.v1"
RANK, ALPHA, DROPOUT = 8, 16, 0.05


@dataclass
class NativeGpuBundle:
    model: object
    processor: object
    architecture: object
    contract: dict


def _pinned_files(config):
    """Bind actual local weight bytes to the completed preparation receipt."""
    from ..campaign_preflight import _prepared
    prepared = _prepared(config["preparation_receipt"])
    if prepared != config:
        raise ValueError("GPU_PREPARED_IDENTITY_MISMATCH")
    snapshot = Path(config["snapshot_path"])
    receipt = json.loads(Path(config["preparation_receipt"]).read_text())
    provenance_path = next(Path(a["path"]) for a in receipt["artifacts"]
                           if Path(a["path"]).name == "provenance.json")
    provenance = json.loads(provenance_path.read_text())
    files = provenance["files"]
    weights = [name for name in files if name.endswith(".safetensors")]
    if not weights:
        raise ValueError("GPU_PREPARED_SAFETENSORS_MISSING")
    for name, metadata in files.items():
        # _prepared already rejects traversal and verifies artifact provenance.
        path = snapshot / name
        if file_hash(path) != metadata["sha256"]:
            raise ValueError("GPU_PREPARED_MODEL_BYTES_CHANGED")
    return {"model_files_hash": canonical_hash(files), "all_prepared_files_rehashed": True,
            "weight_file_count": len(weights), "weight_file_bytes": sum(files[n]["bytes"] for n in weights)}


def _native_model(model):
    return model.get_base_model() if hasattr(model, "peft_config") else model


def _adapter_layers(model, audit):
    """Require the exact native paths, never suffix/substring approximations."""
    native = _native_model(model)
    modules = dict(native.named_modules())
    expected = set(audit["exact_target_modules"])
    actual = {name for name, module in modules.items() if hasattr(module, "lora_A") or hasattr(module, "lora_B")}
    if actual != expected:
        raise ValueError("GPU_LORA_PATH_COVERAGE_MISMATCH")
    return [(entry, modules[entry["path"]]) for entry in audit["target_modules"]]


def assert_frozen_base(model, *, training=True):
    """Check parameters, not input-activation gradients used by checkpointing."""
    trainable, adapters = [], []
    for name, parameter in model.named_parameters():
        is_adapter = bool(re.search(r"\.lora_[AB]\.default\.weight$", name))
        if is_adapter:
            adapters.append(name)
        if parameter.requires_grad:
            if not is_adapter:
                raise ValueError("GPU_BASE_PARAMETER_TRAINABLE: " + name)
            trainable.append(name)
        if not is_adapter and parameter.grad is not None:
            raise ValueError("GPU_FROZEN_BASE_HAS_GRADIENT: " + name)
    if not adapters or (training and set(trainable) != set(adapters)) or (not training and trainable):
        raise ValueError("GPU_ADAPTER_TRAINABILITY_MISMATCH")
    return {"base_frozen": True, "vision_projector_embeddings_head_frozen": True,
            "adapter_tensor_count": len(adapters), "trainable_tensor_count": len(trainable)}


def audit_lora_coverage(model, audit, *, training=True, require_quantized=True):
    """Validate PEFT shapes/rank and every real quantized projection post-load."""
    import torch
    if require_quantized:
        from bitsandbytes.nn import Linear4bit, Params4bit
    config = getattr(model, "peft_config", {}).get("default")
    if config is None or set(model.peft_config) != {"default"}:
        raise ValueError("GPU_SINGLE_DEFAULT_ADAPTER_REQUIRED")
    if (config.r, config.lora_alpha, config.lora_dropout, config.bias) != (RANK, ALPHA, DROPOUT, "none"):
        raise ValueError("GPU_LORA_RECIPE_MISMATCH")
    if config.modules_to_save or config.use_dora or config.use_rslora or config.rank_pattern or config.alpha_pattern:
        raise ValueError("GPU_UNEXPECTED_ADAPTER_VARIANT")
    if not isinstance(config.target_modules, (set, list, tuple)) or set(config.target_modules) != set(audit["exact_target_modules"]):
        raise ValueError("GPU_CONFIG_TARGET_COVERAGE_MISMATCH")
    count, rows = 0, []
    for entry, layer in _adapter_layers(model, audit):
        path, (out_features, in_features) = entry["path"], entry["weight_shape"]
        if set(layer.lora_A) != {"default"} or set(layer.lora_B) != {"default"}:
            raise ValueError("GPU_SINGLE_DEFAULT_ADAPTER_REQUIRED")
        a, b = layer.lora_A["default"].weight, layer.lora_B["default"].weight
        if tuple(a.shape) != (RANK, in_features) or tuple(b.shape) != (out_features, RANK):
            raise ValueError("GPU_LORA_SHAPE_MISMATCH: " + path)
        if a.requires_grad != training or b.requires_grad != training:
            raise ValueError("GPU_ADAPTER_TRAINABILITY_MISMATCH")
        base = layer.get_base_layer()
        if (base.in_features, base.out_features) != (in_features, out_features):
            raise ValueError("GPU_BASE_PROJECTION_SHAPE_MISMATCH: " + path)
        if require_quantized:
            state = getattr(base.weight, "quant_state", None)
            if (not isinstance(base, Linear4bit) or not isinstance(base.weight, Params4bit)
                    or state is None or state.quant_type != "nf4" or not state.nested
                    or base.compute_dtype != torch.bfloat16
                    or tuple(state.shape) != (out_features, in_features)):
                raise ValueError("GPU_NF4_PROJECTION_MISMATCH: " + path)
        count += a.numel() + b.numel()
        rows.append({"path": path, "group": entry["group"], "native_weight_shape": entry["weight_shape"],
                     "base_module_type": type(base).__name__, "adapter_parameters": a.numel() + b.numel()})
    if count != audit["expected_lora_parameters"]:
        raise ValueError("GPU_LORA_PARAMETER_COUNT_MISMATCH")
    frozen = assert_frozen_base(model, training=training)
    return {"exact_target_modules": audit["exact_target_modules"], "target_count": len(rows),
            "target_modules_sha256": audit["target_modules_sha256"], "projections": rows,
            "adapter_parameters": count, "quantized_targets_verified": require_quantized, **frozen}


def adapter_digest(model):
    """Stable digest including tensor name, shape, dtype and exact adapter bytes."""
    digest, count = hashlib.sha256(), 0
    for name, parameter in sorted(model.named_parameters()):
        if not re.search(r"\.lora_[AB]\.default\.weight$", name):
            continue
        import torch
        if parameter.device.type == "meta" or not bool(torch.isfinite(parameter.detach()).all()):
            raise ValueError("GPU_ADAPTER_NONFINITE_OR_META")
        value = parameter.detach().cpu().contiguous()
        digest.update(json.dumps([name, list(value.shape), str(value.dtype)], separators=(",", ":")).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
        count += 1
    if not count:
        raise ValueError("GPU_ADAPTER_TENSORS_MISSING")
    return digest.hexdigest()


def projection_adapter_digests(model, audit):
    """Exact per-projection A/B digests, for proving each adapter was updated."""
    import torch
    result = {}
    for entry, layer in _adapter_layers(model, audit):
        result[entry["path"]] = {}
        for label in ("A", "B"):
            value = getattr(layer, "lora_" + label)["default"].weight.detach()
            if value.device.type == "meta" or not bool(torch.isfinite(value).all()):
                raise ValueError("GPU_ADAPTER_NONFINITE_OR_META")
            value = value.cpu().contiguous()
            result[entry["path"]][label] = hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()
    return result


def projection_gradient_report(model, audit, *, require_nonzero=False):
    """Observe gradients before zero_grad; zero initial LoRA-A grads are valid.

    A worker must accumulate nonzero observations over its updates per projection
    and per A/B tensor. LoRA-B starts at zero, so the first A gradient can be zero.
    Missing or nonfinite gradients always fail, even when nonzero is optional.
    """
    import torch
    rows = []
    for entry, layer in _adapter_layers(model, audit):
        tensors = {}
        for label in ("A", "B"):
            parameter = getattr(layer, "lora_" + label)["default"].weight
            grad = parameter.grad
            if grad is None:
                raise ValueError("GPU_ADAPTER_GRADIENT_MISSING: " + entry["path"] + ":" + label)
            if not bool(torch.isfinite(grad).all()):
                raise ValueError("GPU_ADAPTER_GRADIENT_NONFINITE: " + entry["path"] + ":" + label)
            norm = float(torch.linalg.vector_norm(grad.detach().float()).item())
            nonzero = bool(torch.count_nonzero(grad).item())
            if require_nonzero and not nonzero:
                raise ValueError("GPU_ADAPTER_GRADIENT_ZERO: " + entry["path"] + ":" + label)
            tensors[label] = {"finite": True, "nonzero": nonzero, "l2_norm": norm, "elements": grad.numel()}
        rows.append({"path": entry["path"], "group": entry["group"], "tensors": tensors})
    assert_frozen_base(model)
    return {"all_finite": True, "projection_count": len(rows), "projections": rows}


def parameter_storage_report(model, audit):
    """Separate original dense parameter counts from packed runtime storage.

    Includes unique parameter/buffer storage and bnb quantization-state tensors;
    excludes activations, optimizer, CUDA allocator overhead and driver memory.
    Packed numel is explicitly never called the original model parameter count.
    """
    import torch
    seen, storage_bytes, packed_numel = set(), 0, 0
    def add(tensor):
        nonlocal storage_bytes
        if not isinstance(tensor, torch.Tensor) or tensor.device.type == "meta":
            return
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
        if key not in seen:
            seen.add(key)
            storage_bytes += storage.nbytes()
    def state_tensors(state):
        if state is None:
            return
        for name in ("absmax", "code", "offset"):
            add(getattr(state, name, None))
        state_tensors(getattr(state, "state2", None))
    for parameter in model.parameters():
        packed_numel += parameter.numel()
        add(parameter)
        state_tensors(getattr(parameter, "quant_state", None))
    for buffer in model.buffers():
        add(buffer)
    adapters = sum(p.numel() for n, p in model.named_parameters()
                   if re.search(r"\.lora_[AB]\.default\.weight$", n))
    base_dtypes = [{"path": name, "dtype": str(parameter.dtype), "packed_shape": list(parameter.shape)}
                   for name, parameter in _native_model(model).named_parameters()
                   if not re.search(r"\.lora_[AB]\.default\.weight$", name)]
    return {"original_dense_parameter_counts": audit["parameter_counts"], "adapter_parameters": adapters,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "runtime_packed_parameter_elements": packed_numel, "runtime_unique_tensor_storage_bytes": storage_bytes,
            "storage_includes_quantization_state": True, "storage_excludes_optimizer_activations_allocator": True,
            "base_dtype_signature": canonical_hash(base_dtypes), "base_dtypes": base_dtypes}


def _cuda_guard(torch):
    if not torch.cuda.is_available():
        raise RuntimeError("GPU_NOT_AVAILABLE")
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("GPU_NATIVE_BF16_REQUIRED")
    torch.cuda.set_device(0)
    # Exercise the actual CUDA context; NVML alone is not a capability test.
    probe = torch.ones(1, dtype=torch.bfloat16, device="cuda:0")
    if float((probe + probe).item()) != 2.0:
        raise RuntimeError("GPU_CUDA_PROBE_FAILED")
    del probe


def _cuda_residency(model):
    for _, value in list(model.named_parameters()) + list(model.named_buffers()):
        if value.device.type != "cuda" or value.device.index != 0:
            raise ValueError("GPU_CPU_META_OR_OTHER_DEVICE_OFFLOAD_FORBIDDEN")


def _saved_processor(local, architecture, template, processor_path):
    from .models import load_processor
    processor_path = Path(processor_path)
    if not processor_path.is_absolute() or not processor_path.is_dir():
        raise ValueError("GPU_ABSOLUTE_PROCESSOR_DIRECTORY_REQUIRED")
    processor = load_processor({**local, "model_id": str(processor_path)}, architecture)
    # Compare raw saved bytes before replacing any date expression. Some native
    # processors delegate to their tokenizer; compare both roles explicitly.
    tokenizer = getattr(processor, "tokenizer", processor)
    objects = {"processor": processor}
    if tokenizer is not processor:
        objects["tokenizer"] = tokenizer
    if set(objects) != set(template["templates"]):
        raise ValueError("GPU_SAVED_PROCESSOR_TEMPLATE_ROLES_MISMATCH")
    for role, obj in objects.items():
        if canonical_hash(getattr(obj, "chat_template", None)) != template["templates"][role]["frozen_sha256"]:
            raise ValueError("GPU_SAVED_PROCESSOR_TEMPLATE_MISMATCH")
    freeze_native_template(processor, fixed_date=template["fixed_date"])
    return processor


def load_v6_gpu_bundle(model_config, cell_id, *, fixed_template_date="2026-11-14",
                       adapter_path=None, processor_path=None, training=True):
    """Load one exact prepared native base with strict V6 LoRA, never fallback.

    ``model_config`` is the nested ``model`` returned by CPU build_config.
    ``adapter_path`` is a private completed adapter directory for fresh reload.
    Callers must seed before training load and use BF16 autocast during forward.
    """
    import torch
    import transformers
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from .models import load_processor, versions
    if type(training) is not bool or fixed_template_date != "2026-11-14":
        raise ValueError("GPU_V6_RECIPE_MISMATCH")
    if not training and adapter_path is None:
        raise ValueError("GPU_EVAL_REQUIRES_SMOKE_ADAPTER")
    if not re.fullmatch(r"[0-9a-f]{40}", model_config.get("revision", "")):
        raise ValueError("GPU_IMMUTABLE_REVISION_REQUIRED")
    files = _pinned_files(model_config)
    local = {"model_id": model_config["snapshot_path"], "revision": model_config["revision"],
             "cache_dir": model_config["cache_dir"], "local_files_only": True}
    architecture = transformers.AutoConfig.from_pretrained(local["model_id"], local_files_only=True,
                                                           trust_remote_code=False)
    audit = audit_native_architecture(architecture, cell_id)
    processor = load_processor(local, architecture)
    template = freeze_native_template(processor, fixed_date=fixed_template_date)
    thinking = verify_native_no_thinking(processor, architecture.model_type)
    if processor_path is not None:
        processor = _saved_processor(local, architecture, template, processor_path)
        thinking = verify_native_no_thinking(processor, architecture.model_type)
    _cuda_guard(torch)
    quantization = transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=["lm_head", *audit["excluded_roots"]])
    loader = getattr(transformers, audit["architecture"])
    model = loader.from_pretrained(local["model_id"], revision=local["revision"], cache_dir=local["cache_dir"],
        local_files_only=True, trust_remote_code=False, use_safetensors=True, dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="sdpa", quantization_config=quantization)
    if type(model).__name__ != audit["architecture"] or not getattr(model, "is_loaded_in_4bit", False):
        raise ValueError("GPU_NATIVE_QUANTIZED_MODEL_REQUIRED")
    model.requires_grad_(False)
    # Apply the identical FP32 nonquantized-base policy during fresh reload too.
    # PEFT upcasts frozen embeddings/head/vision/projector as well as norms; this
    # memory cost is recorded rather than silently changing reload numerics.
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    if training:
        if not model.is_gradient_checkpointing:
            raise ValueError("GPU_GRADIENT_CHECKPOINTING_NOT_ENABLED")
    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False
    if adapter_path is not None:
        adapter_path = Path(adapter_path)
        if not adapter_path.is_absolute() or not adapter_path.is_dir():
            raise ValueError("GPU_ABSOLUTE_ADAPTER_DIRECTORY_REQUIRED")
        saved = json.loads((adapter_path / "adapter_config.json").read_text())
        if saved.get("revision") != local["revision"] or saved.get("base_model_name_or_path") != local["model_id"]:
            raise ValueError("GPU_ADAPTER_BASE_IDENTITY_MISMATCH")
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=training,
                                         local_files_only=True)
    else:
        lora = LoraConfig(r=RANK, lora_alpha=ALPHA, lora_dropout=DROPOUT,
                          target_modules=audit["exact_target_modules"], task_type="CAUSAL_LM", bias="none")
        model = get_peft_model(model, lora)
        model.peft_config["default"].revision = local["revision"]
    coverage = audit_lora_coverage(model, audit, training=training)
    _cuda_residency(model)
    model.train(training)
    contract = {"schema_version": SCHEMA, "cell_id": cell_id, "architecture": audit,
        "template": template, "thinking": thinking, "files": files, "coverage": coverage,
        "parameter_storage": parameter_storage_report(model, audit), "versions": versions(),
        "loader": loader.__name__, "processor_class": type(processor).__name__,
        "revision": local["revision"], "snapshot_path": local["model_id"], "local_files_only": True,
        "trust_remote_code": False, "quantization": {"method": "nf4", "double_quant": True,
        "compute_dtype": "bfloat16", "excluded_native_roots": audit["excluded_roots"]},
        "gradient_checkpointing": bool(training), "use_reentrant": False, "attention_backend": "sdpa",
        "cuda_residency_verified": True, "weights_loaded": True,
        "gradient_verified": False, "optimizer_verified": False, "reload_verified": False}
    return NativeGpuBundle(model, processor, architecture, contract)
