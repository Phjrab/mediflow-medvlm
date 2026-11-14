"""Opt-in V6 native architecture and deterministic rendering contracts.

The historical loader/renderer does not import this module. Architecture probes
construct only native meta tensors, never download/load weights, run a forward,
or establish GPU, quantization, gradient, optimizer, or reload capability.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import date
import hashlib
import json
import re

from ..runtime import canonical_hash


CONTRACT_VERSION = "medvlm.v6-native-contract.v1"
_SPECS = {
    "T1": ("qwen3", "Qwen3ForCausalLM", "model", ()),
    "T2": ("qwen3", "Qwen3ForCausalLM", "model", ()),
    "T3": ("gemma3", "Gemma3ForConditionalGeneration", "model.language_model",
           ("model.vision_tower", "model.multi_modal_projector")),
    "T4": ("gemma3", "Gemma3ForConditionalGeneration", "model.language_model",
           ("model.vision_tower", "model.multi_modal_projector")),
    "T5": ("phi3", "Phi3ForCausalLM", "model", ()),
    "T6": ("smollm3", "SmolLM3ForCausalLM", "model", ()),
    "V1": ("qwen3_5", "Qwen3_5ForConditionalGeneration", "model.language_model", ("model.visual",)),
    "V2": ("gemma3", "Gemma3ForConditionalGeneration", "model.language_model",
           ("model.vision_tower", "model.multi_modal_projector")),
    "V3": ("gemma3", "Gemma3ForConditionalGeneration", "model.language_model",
           ("model.vision_tower", "model.multi_modal_projector")),
    "V4": ("smolvlm", "SmolVLMForConditionalGeneration", "model.text_model",
           ("model.vision_model", "model.connector")),
}
_TEXT_TYPES = {"qwen3": "qwen3", "phi3": "phi3", "smollm3": "smollm3",
               "gemma3": "gemma3_text", "qwen3_5": "qwen3_5_text", "smolvlm": "llama"}


def _inside(name, prefix):
    return name == prefix or name.startswith(prefix + ".")


def _positive(config, key):
    value = getattr(config, key, None)
    if type(value) is not int or value < 1:
        raise ValueError("NATIVE_CONFIG_INVALID: " + key)
    return value


def _projection_shapes(config, model_type, layer_type):
    hidden = _positive(config, "hidden_size")
    intermediate = _positive(config, "intermediate_size")
    heads = _positive(config, "num_attention_heads")
    kv_heads = _positive(config, "num_key_value_heads")
    dim = getattr(config, "head_dim", None) or hidden // heads
    if type(dim) is not int or dim < 1 or heads % kv_heads:
        raise ValueError("NATIVE_CONFIG_INVALID: attention head dimensions")
    q, kv = heads * dim, kv_heads * dim
    if model_type == "phi3":
        return {"self_attn.qkv_proj": (q + 2 * kv, hidden),
                "self_attn.o_proj": (hidden, q),
                "mlp.gate_up_proj": (2 * intermediate, hidden),
                "mlp.down_proj": (hidden, intermediate)}
    shapes = {"mlp.gate_proj": (intermediate, hidden), "mlp.up_proj": (intermediate, hidden),
              "mlp.down_proj": (hidden, intermediate)}
    if model_type == "qwen3_5" and layer_type == "linear_attention":
        key = _positive(config, "linear_num_key_heads") * _positive(config, "linear_key_head_dim")
        value_heads = _positive(config, "linear_num_value_heads")
        value = value_heads * _positive(config, "linear_value_head_dim")
        shapes.update({"linear_attn.in_proj_qkv": (2 * key + value, hidden),
                       "linear_attn.in_proj_z": (value, hidden),
                       "linear_attn.in_proj_b": (value_heads, hidden),
                       "linear_attn.in_proj_a": (value_heads, hidden),
                       "linear_attn.out_proj": (hidden, value)})
    else:
        shapes.update({"self_attn.q_proj": (q * (2 if model_type == "qwen3_5" else 1), hidden),
                       "self_attn.k_proj": (kv, hidden), "self_attn.v_proj": (kv, hidden),
                       "self_attn.o_proj": (hidden, q)})
    return shapes


def audit_native_architecture(model_config, cell_id, *, model=None, lora_rank=8):
    """Require complete, architecture-specific decoder Linear coverage.

    ``model_config`` is a native config or its JSON dict. Without ``model`` only a
    native meta skeleton is built. A supplied dense CPU/meta native model is also
    accepted for adversarial tests or a separate preflight. Quantized/PEFT models
    are deliberately rejected; these counts describe original dense shapes.
    The returned exact paths can be passed explicitly to PEFT target_modules.
    """
    import torch
    import transformers
    from transformers import AutoConfig

    if cell_id not in _SPECS:
        raise ValueError("NATIVE_CELL_UNKNOWN: " + str(cell_id))
    if type(lora_rank) is not int or lora_rank < 1:
        raise ValueError("NATIVE_CONFIG_INVALID: lora_rank")
    if isinstance(model_config, dict):
        config_dict = deepcopy(model_config)
        model_type = config_dict.pop("model_type", None)
        if not isinstance(model_type, str):
            raise ValueError("NATIVE_CONFIG_INVALID: model_type")
        model_config = AutoConfig.for_model(model_type, **config_dict)
    model_config = deepcopy(model_config)
    model_type, architecture, language_root, excluded_roots = _SPECS[cell_id]
    if model_config.model_type != model_type or list(model_config.architectures or []) != [architecture]:
        raise ValueError("NATIVE_ARCHITECTURE_MISMATCH: " + cell_id)
    text_config = getattr(model_config, "text_config", model_config)
    if text_config.model_type != _TEXT_TYPES[model_type]:
        raise ValueError("NATIVE_DECODER_TYPE_MISMATCH: " + str(text_config.model_type))
    count = _positive(text_config, "num_hidden_layers")
    layer_types = getattr(text_config, "layer_types", None)
    if model_type == "qwen3_5":
        if not isinstance(layer_types, (list, tuple)) or len(layer_types) != count or set(layer_types) != {"linear_attention", "full_attention"}:
            raise ValueError("NATIVE_LAYER_TYPES_INVALID: hybrid full/linear attention required")
    else:
        layer_types = list(layer_types) if layer_types is not None else ["full_attention"] * count
        if len(layer_types) != count or not set(layer_types) <= {"full_attention", "sliding_attention"}:
            raise ValueError("NATIVE_LAYER_TYPES_INVALID")
    constructed_meta = model is None
    if constructed_meta:
        # Calling __init__ cannot fetch Hub weights or execute remote code.
        with torch.device("meta"):
            model = getattr(transformers, architecture)(model_config)
    if type(model).__name__ != architecture:
        raise ValueError("NATIVE_MODEL_CLASS_MISMATCH")
    devices = {p.device.type for p in model.parameters()}
    if not devices <= {"cpu", "meta"} or (constructed_meta and devices != {"meta"}):
        raise ValueError("NATIVE_CPU_META_ONLY_REQUIRED")
    meta_only = devices == {"meta"}
    if canonical_hash(model.config.to_dict()) != canonical_hash(model_config.to_dict()):
        raise ValueError("NATIVE_MODEL_CONFIG_MISMATCH")
    modules = dict(model.named_modules())
    decoder_root = language_root + ".layers"
    if decoder_root not in modules or len(modules[decoder_root]) != count:
        raise ValueError("NATIVE_LAYER_COVERAGE_MISMATCH: " + decoder_root)
    for prefix in excluded_roots:
        if prefix not in modules:
            raise ValueError("NATIVE_EXCLUDED_SUBTREE_MISSING: " + prefix)
    expected, coverage = {}, []
    for index, layer_type in enumerate(layer_types):
        prefix = decoder_root + "." + str(index)
        shapes = _projection_shapes(text_config, model_type, layer_type)
        expected.update({prefix + "." + key: shape for key, shape in shapes.items()})
        coverage.append({"layer_index": index, "layer_type": layer_type,
                         "expected_projection_count": len(shapes),
                         "projection_groups": sorted(shapes)})
    actual = {name: module for name, module in modules.items()
              if _inside(name, decoder_root) and
              (isinstance(module, torch.nn.Linear) or
               (hasattr(module, "weight") and getattr(module.weight, "ndim", 0) == 2))}
    if set(actual) != set(expected):
        raise ValueError("NATIVE_PROJECTION_COVERAGE_MISMATCH: " + json.dumps({
            "missing": sorted(set(expected) - set(actual)), "unexpected": sorted(set(actual) - set(expected))}))
    targets, groups, lora_parameters = [], Counter(), 0
    target_weight_ids = set()
    for name in sorted(expected):
        module = actual[name]
        if type(module) is not torch.nn.Linear or tuple(module.weight.shape) != expected[name]:
            raise ValueError("NATIVE_PROJECTION_SHAPE_OR_TYPE_MISMATCH: " + name)
        if id(module.weight) in target_weight_ids:
            raise ValueError("NATIVE_SHARED_TARGET_WEIGHT: " + name)
        target_weight_ids.add(id(module.weight))
        group = name.split(".layers.", 1)[1].split(".", 1)[1]
        groups[group] += 1
        shape = list(module.weight.shape)
        lora_parameters += lora_rank * sum(shape)
        targets.append({"path": name, "module_type": type(module).__name__,
                        "weight_shape": shape, "weight_parameters": module.weight.numel(),
                        "has_bias": module.bias is not None, "group": group})
    excluded_linears = [{"path": name, "module_type": type(module).__name__,
                         "weight_shape": list(module.weight.shape),
                         "reason": "vision_or_projector" if any(_inside(name, r) for r in excluded_roots)
                         else "outside_decoder_blocks"}
                        for name, module in modules.items() if isinstance(module, torch.nn.Linear) and name not in expected]
    # Disjoint original dense parameter counts; shared tensors counted once in
    # native named_parameters order. lm_head/embeddings never enter target paths.
    counts = Counter(total=0, language=0, vision=0, projector=0, other=0)
    dense_bytes = 0
    for name, parameter in model.named_parameters():
        counts["total"] += parameter.numel()
        dense_bytes += parameter.numel() * parameter.element_size()
        if _inside(name, language_root) or _inside(name, "lm_head"):
            category = "language"
        elif any(_inside(name, r) for r in excluded_roots):
            category = "projector" if any(word in name for word in ("projector", "connector", "merger")) else "vision"
        else:
            category = "other"
        counts[category] += parameter.numel()
    if counts["total"] != sum(counts[k] for k in ("language", "vision", "projector", "other")):
        raise ValueError("NATIVE_PARAMETER_PARTITION_MISMATCH")
    exact = sorted(expected)
    return {"schema_version": CONTRACT_VERSION, "cell_id": cell_id,
            "status": "CPU_META_ARCHITECTURE_VERIFIED" if meta_only else "CPU_DENSE_ARCHITECTURE_VERIFIED",
            "evidence_level": "CPU_META_ONLY" if meta_only else "CPU_DENSE_STRUCTURE_ONLY",
            "architecture": architecture, "model_type": model_type,
            "config_sha256": canonical_hash(model_config.to_dict()),
            "transformers_version": transformers.__version__, "torch_version": torch.__version__,
            "language_root": language_root, "decoder_root": decoder_root,
            "layer_count": count, "layer_type_counts": dict(Counter(layer_types)),
            "exact_target_modules": exact, "target_modules_sha256": canonical_hash(exact),
            "target_count": len(exact), "target_modules": targets,
            "coverage_by_projection": dict(sorted(groups.items())), "coverage_by_layer": coverage,
            "excluded_roots": list(excluded_roots), "excluded_linear_modules": excluded_linears,
            "excluded_from_lora": ["embeddings", "lm_head", "vision_encoder", "multimodal_projector",
                                   "normalization", "convolution", "nonlinear_state_parameters"],
            "parameter_counts": dict(counts), "dense_parameter_bytes_at_meta_dtype": dense_bytes,
            "allocated_weight_storage_bytes": 0 if meta_only else None,
            "expected_lora_parameters": lora_parameters, "lora_rank": lora_rank,
            "gpu_verified": False, "weights_loaded": False if meta_only else None,
            "model_origin": "native_config_meta_constructor" if constructed_meta else "caller_supplied_unverified_weights",
            "quantized_runtime_verified": False, "gradient_verified": False,
            "optimizer_verified": False, "reload_verified": False}


_DATE_CALL = re.compile(r"\bstrftime_now\(\s*(['\"])%d %B %Y\1\s*\)")
_MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December")


def _freeze_value(value, fixed):
    if isinstance(value, str):
        day = f"{fixed.day:02d} {_MONTHS[fixed.month - 1]} {fixed.year}"
        frozen, count = _DATE_CALL.subn(lambda _: json.dumps(day), value)
        if "strftime_now" in frozen:
            raise ValueError("NATIVE_TEMPLATE_UNKNOWN_CLOCK_EXPRESSION")
        return frozen, count
    if isinstance(value, dict) and value and all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
        pairs = {key: _freeze_value(child, fixed) for key, child in value.items()}
        return {key: pair[0] for key, pair in pairs.items()}, sum(pair[1] for pair in pairs.values())
    raise ValueError("NATIVE_TEMPLATE_MISSING_OR_MALFORMED")


def freeze_native_template(tokenizer_or_processor, fixed_date="2026-11-14"):
    """Freeze only recognized native date calls on the in-memory object.

    Preserve every other byte, including native system/thinking precedence. No
    save_pretrained or cache writes occur. Repeat calls validate immutability;
    changing the date or externally editing a frozen template is an error.
    """
    fixed = date.fromisoformat(fixed_date)
    if fixed.isoformat() != fixed_date:
        raise ValueError("NATIVE_TEMPLATE_DATE_MUST_BE_ISO")
    objects = [("processor", tokenizer_or_processor)]
    tokenizer = getattr(tokenizer_or_processor, "tokenizer", tokenizer_or_processor)
    if tokenizer is not tokenizer_or_processor:
        objects.append(("tokenizer", tokenizer))
    existing = getattr(tokenizer_or_processor, "_medvlm_v6_template_receipt", None)
    if existing is not None:
        if existing["fixed_date"] != fixed_date:
            raise ValueError("NATIVE_TEMPLATE_ALREADY_FROZEN_DIFFERENT_DATE")
        for role, obj in objects:
            if canonical_hash(getattr(obj, "chat_template", None)) != existing["templates"][role]["frozen_sha256"]:
                raise ValueError("NATIVE_TEMPLATE_FROZEN_BYTES_CHANGED")
        return deepcopy(existing)
    staged, receipts = [], {}
    for role, obj in objects:
        original = deepcopy(getattr(obj, "chat_template", None))
        # Some processors intentionally delegate to the tokenizer's template.
        if original is None and role == "processor" and tokenizer is not obj:
            receipts[role] = {"original_sha256": canonical_hash(None), "frozen_sha256": canonical_hash(None),
                              "clock_calls_replaced": 0, "delegates_to_tokenizer": True}
            continue
        frozen, replaced = _freeze_value(original, fixed)
        staged.append((obj, frozen))
        receipts[role] = {"original_sha256": canonical_hash(original), "frozen_sha256": canonical_hash(frozen),
                          "clock_calls_replaced": replaced, "delegates_to_tokenizer": False}
    # Validate all templates before assigning any of them.
    for obj, frozen in staged:
        obj.chat_template = frozen
    metadata = {"schema_version": CONTRACT_VERSION, "policy": "native-clock-literal-copy-v1",
                "fixed_date": fixed_date, "templates": receipts, "in_memory_only": True,
                "native_semantics_preserved_except_clock": True}
    tokenizer_or_processor._medvlm_v6_template_receipt = deepcopy(metadata)
    return deepcopy(metadata)


def verify_native_no_thinking(processor, model_type, messages=None):
    """Prove effective template control and assistant prefix, never generation.

    SmolLM3 system /think overrides are deliberately preserved by the freezer and
    rejected here. Families with no thinking switch are explicitly distinguished
    from families whose switch was exercised. Real generated output requires the
    separate output check and remains unverified by this CPU rendering probe.
    """
    if model_type not in _TEXT_TYPES:
        raise ValueError("NATIVE_THINKING_MODEL_UNKNOWN")
    tokenizer = getattr(processor, "tokenizer", processor)
    active_template = getattr(processor, "chat_template", None) or getattr(tokenizer, "chat_template", None)
    template_values = active_template.values() if isinstance(active_template, dict) else [active_template]
    if any(isinstance(value, str) and "strftime_now" in value for value in template_values):
        raise ValueError("NATIVE_TEMPLATE_CLOCK_NOT_FROZEN")
    messages = deepcopy(messages if messages is not None else [{"role": "user", "content": "Return only ready."}])
    if not messages or messages[-1].get("role") != "user":
        raise ValueError("NATIVE_THINKING_PROBE_REQUIRES_FINAL_USER")
    def render(thinking, generation=True, turns=None):
        return processor.apply_chat_template(turns or messages, tokenize=False,
                                             add_generation_prompt=generation, enable_thinking=thinking)
    disabled, enabled = render(False), render(True)
    if not isinstance(disabled, str) or disabled != render(False):
        raise ValueError("NATIVE_TEMPLATE_NONDETERMINISTIC")
    switch_family = model_type in {"qwen3", "qwen3_5", "smollm3"}
    if switch_family:
        ending = "<|im_start|>assistant\n<think>\n\n</think>\n" + ("" if model_type == "smollm3" else "\n")
        if not disabled.endswith(ending):
            raise ValueError("NATIVE_THINKING_DISABLE_INEFFECTIVE")
        if model_type == "smollm3" and "Reasoning Mode: /think\n" in disabled:
            raise ValueError("NATIVE_THINKING_SYSTEM_OVERRIDE")
        # Explicit /no_think overrides both kwargs in the native Smol template.
        system_no_think = model_type == "smollm3" and any(
            turn.get("role") == "system" and isinstance(turn.get("content"), str) and "/no_think" in turn["content"]
            for turn in messages)
        if disabled == enabled and not system_no_think:
            raise ValueError("NATIVE_THINKING_SWITCH_NOT_EXERCISED")
        status = "NATIVE_SYSTEM_NO_THINK_VERIFIED" if system_no_think else "NATIVE_DISABLE_SWITCH_VERIFIED"
    else:
        endings = {"gemma3": "<start_of_turn>model\n", "phi3": "<|assistant|>", "smolvlm": "Assistant:"}
        ending = endings[model_type]
        if disabled != enabled or not disabled.endswith(ending):
            raise ValueError("NATIVE_NO_SWITCH_TEMPLATE_MISMATCH")
        status = "NATIVE_TEMPLATE_HAS_NO_THINKING_SWITCH"
    full = render(False, generation=False, turns=messages + [{"role": "assistant", "content": "ready"}])
    if not full.startswith(disabled) or not full[len(disabled):].lstrip().startswith("ready"):
        raise ValueError("NATIVE_ASSISTANT_RENDER_PREFIX_MISMATCH")
    return {"schema_version": CONTRACT_VERSION, "status": status, "model_type": model_type,
            "enable_thinking": False, "native_control_verified": switch_family,
            "rendered_prompt_sha256": hashlib.sha256(disabled.encode()).hexdigest(),
            "enabled_prompt_sha256": hashlib.sha256(enabled.encode()).hexdigest(),
            "rendered_full_sha256": hashlib.sha256(full.encode()).hexdigest(),
            "assistant_prefix": ending, "assistant_prefix_verified": True,
            "deterministic_render_verified": True, "generated_output_verified": False,
            "hidden_reasoning_cost_verified": False}


def verify_native_no_thinking_output(output):
    """Check externally supplied decoded generation for explicit think spans.

    This lexical diagnostic cannot measure hidden reasoning or establish output
    correctness. Call on generated continuation only, excluding prompt messages.
    """
    if not isinstance(output, str):
        raise ValueError("NATIVE_OUTPUT_MUST_BE_TEXT")
    remaining = re.sub(r"<think>\s*</think>", "", output)
    if "<think>" in remaining or "</think>" in remaining:
        raise ValueError("NATIVE_OUTPUT_REASONING_SPAN_PRESENT")
    return {"explicit_nonempty_thinking_detected": False,
            "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
            "hidden_reasoning_cost_verified": False}
