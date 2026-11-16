"""Native V6 evaluation bundles. No training loader or archived source is changed.

Base loads the exact prepared native checkpoint, with no PEFT adapter created.
Tuned reuses the proven fresh-reload loader and saved final processor. Both use
identical pinned NF4/BF16, FP32 nonquantized base policy and native templates.
Importing this module performs no CUDA operation or model load.
"""
from __future__ import annotations

import re
from pathlib import Path

from ..runtime import canonical_hash
from ..training import v6_gpu as native
from ..training.models import load_processor, versions
from ..training.v6_native import (audit_native_architecture, freeze_native_template,
                                  verify_native_no_thinking)


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def normalized_base_dtype_signature(storage):
    # PEFT wraps the same frozen Linear4bit tensors under .base_layer names.
    records = [{**item, "path": item["path"].replace(".base_layer.", ".")}
               for item in storage["base_dtypes"]]
    return canonical_hash(records)


def _verify_contract(bundle, expected, role):
    actual = bundle.contract
    for key in ("architecture", "template", "thinking", "loader", "revision", "snapshot_path",
                "processor_class", "quantization", "native_runtime_compatibility"):
        _require(actual.get(key) == expected.get(key), "EVAL_NATIVE_TRAIN_CONTRACT_MISMATCH:" + key)
    _require(normalized_base_dtype_signature(actual["parameter_storage"]) ==
             normalized_base_dtype_signature(expected["parameter_storage"]), "EVAL_BASE_DTYPE_POLICY_CHANGED")
    _require(actual["parameter_storage"]["trainable_parameters"] == 0, "EVAL_TRAINABLE_PARAMETERS_FORBIDDEN")
    if role == "base":
        _require(not hasattr(bundle.model, "peft_config") and
            actual["parameter_storage"]["adapter_parameters"] == 0,
            "EVAL_BASE_MUST_HAVE_NO_ADAPTER")
    return bundle


def load_evaluation_bundle(model_config, cell_id, *, role, expected_contract,
                           final_directory=None, expected_adapter_sha256=None,
                           fixed_template_date="2026-11-14"):
    """Load one Base or audited final Tuned; never select a checkpoint here."""
    _require(role in {"base", "tuned"} and fixed_template_date == "2026-11-14",
             "EVAL_ROLE_OR_TEMPLATE_DATE_MISMATCH")
    if role == "tuned":
        directory = Path(final_directory)
        _require(directory.is_absolute() and directory.is_dir(), "EVAL_FINAL_DIRECTORY_REQUIRED")
        bundle = native.load_v6_gpu_bundle(model_config, cell_id, training=False,
            adapter_path=directory / "adapter", processor_path=directory / "processor",
            fixed_template_date=fixed_template_date)
        _require(native.adapter_digest(bundle.model) == expected_adapter_sha256,
                 "EVAL_FINAL_ADAPTER_WEIGHTS_CHANGED")
        bundle.contract["evaluation_role"] = role
        bundle.contract["evaluation_adapter_sha256"] = expected_adapter_sha256
        return _verify_contract(bundle, expected_contract, role)
    import torch
    import transformers
    from peft import prepare_model_for_kbit_training
    _require(bool(re.fullmatch(r"[0-9a-f]{40}", model_config.get("revision", ""))),
             "EVAL_PINNED_MODEL_REVISION_REQUIRED")
    files = native._pinned_files(model_config)
    local = {"model_id": model_config["snapshot_path"], "revision": model_config["revision"],
             "cache_dir": model_config["cache_dir"], "local_files_only": True}
    architecture = transformers.AutoConfig.from_pretrained(local["model_id"],
        local_files_only=True, trust_remote_code=False)
    audit = audit_native_architecture(architecture, cell_id)
    processor = load_processor(local, architecture)
    template = freeze_native_template(processor, fixed_date=fixed_template_date)
    thinking = verify_native_no_thinking(processor, architecture.model_type)
    native._cuda_guard(torch)
    quantization = transformers.BitsAndBytesConfig(load_in_4bit=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=["lm_head", *audit["excluded_roots"]])
    loader = getattr(transformers, audit["architecture"])
    model = loader.from_pretrained(local["model_id"], revision=local["revision"],
        cache_dir=local["cache_dir"], local_files_only=True, trust_remote_code=False,
        use_safetensors=True, dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="sdpa", quantization_config=quantization)
    _require(type(model).__name__ == audit["architecture"] and getattr(model, "is_loaded_in_4bit", False),
             "EVAL_NATIVE_QUANTIZED_MODEL_REQUIRED")
    compatibility = native._install_smolvlm_inputs_merger(model)
    # The established reload policy deliberately upcasts nonquantized base
    # tensors to FP32, including frozen embeddings/head/vision/projector.
    model.requires_grad_(False)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model.requires_grad_(False)
    modules = dict(model.named_modules())
    import bitsandbytes as bnb
    _require(all(isinstance(modules.get(path), bnb.nn.Linear4bit)
        for path in audit["exact_target_modules"]), "EVAL_BASE_NATIVE_QUANTIZED_TARGET_COVERAGE_MISMATCH")
    _require(not any("lora_" in name for name, _ in model.named_parameters()),
             "EVAL_BASE_RANDOM_ADAPTER_FORBIDDEN")
    native._cuda_residency(model)
    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False
    model.eval()
    contract = {"schema_version": "medvlm.v6.native-evaluation-contract.v1",
        "evaluation_role": role, "evaluation_adapter_sha256": None, "cell_id": cell_id, "architecture": audit,
        "template": template, "thinking": thinking, "files": files,
        "parameter_storage": native.parameter_storage_report(model, audit),
        "versions": versions(), "loader": loader.__name__,
        "processor_class": type(processor).__name__, "revision": local["revision"],
        "snapshot_path": local["model_id"], "local_files_only": True,
        "trust_remote_code": False, "quantization": {"method": "nf4", "double_quant": True,
        "compute_dtype": "bfloat16", "excluded_native_roots": audit["excluded_roots"]},
        "gradient_checkpointing": False, "attention_backend": "sdpa",
        "cuda_residency_verified": True, "weights_loaded": True}
    if compatibility is not None:
        contract["native_runtime_compatibility"] = compatibility
    return _verify_contract(native.NativeGpuBundle(model, processor, architecture, contract), expected_contract, role)
