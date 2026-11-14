"""V6 GPU loader guards and real native CPU/PEFT contracts; no CUDA loads."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mediflow_medvlm.runtime import canonical_hash, file_hash
from mediflow_medvlm.training.v6_gpu import (
    ALPHA, DROPOUT, RANK, _cuda_guard, _pinned_files, _saved_processor, adapter_digest,
    assert_frozen_base, audit_lora_coverage, load_v6_gpu_bundle,
    parameter_storage_report, projection_adapter_digests, projection_gradient_report,
)
from mediflow_medvlm.training.v6_native import audit_native_architecture, freeze_native_template


class RuntimeGuardTests(unittest.TestCase):
    def test_missing_cuda_and_native_bf16_fail_without_fallback(self):
        for available, bf16, error in ((False, True, "GPU_NOT_AVAILABLE"),
                                        (True, False, "NATIVE_BF16_REQUIRED")):
            cuda = SimpleNamespace(is_available=lambda: available,
                is_bf16_supported=lambda **kwargs: bf16,
                set_device=lambda _: self.fail("no CUDA operation before capability gates"))
            with self.subTest(error=error), self.assertRaisesRegex(RuntimeError, error):
                _cuda_guard(SimpleNamespace(cuda=cuda))

    def test_prepared_weights_are_actually_rehashed_and_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weight = root / "model.safetensors-00001-of-00001.safetensors"
            weight.write_bytes(b"fixture-weights")
            files = {weight.name: {"bytes": weight.stat().st_size, "sha256": file_hash(weight)}}
            provenance = root / "provenance.json"
            provenance.write_text(json.dumps({"files": files}))
            receipt = root / "receipt.json"
            receipt.write_text(json.dumps({"artifacts": [{"path": str(provenance)}]}))
            config = {"preparation_receipt": str(receipt), "snapshot_path": str(root)}
            with patch("mediflow_medvlm.campaign_preflight._prepared", return_value=config):
                result = _pinned_files(config)
                self.assertEqual(result["model_files_hash"], canonical_hash(files))
                self.assertTrue(result["all_prepared_files_rehashed"])
                self.assertEqual(result["weight_file_count"], 1)
                weight.write_bytes(b"mutated-weights")
                with self.assertRaisesRegex(ValueError, "BYTES_CHANGED"):
                    _pinned_files(config)
            with patch("mediflow_medvlm.campaign_preflight._prepared", return_value={}):
                with self.assertRaisesRegex(ValueError, "IDENTITY_MISMATCH"):
                    _pinned_files(config)


_HAS_NATIVE = all(importlib.util.find_spec(name) for name in ("torch", "transformers", "peft"))


@unittest.skipUnless(_HAS_NATIVE, "optional native CPU/PEFT profile")
class NativePeftTests(unittest.TestCase):
    @staticmethod
    def config(kind="qwen3"):
        from transformers import Qwen3Config, Phi3Config, SmolLM3Config
        cls = {"qwen3": Qwen3Config, "phi3": Phi3Config, "smollm3": SmolLM3Config}[kind]
        config = cls(vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=4, max_position_embeddings=128, original_max_position_embeddings=128,
            pad_token_id=0, bos_token_id=1, eos_token_id=2)
        config.architectures = [{"qwen3": "Qwen3ForCausalLM", "phi3": "Phi3ForCausalLM",
                                 "smollm3": "SmolLM3ForCausalLM"}[kind]]
        return config

    @staticmethod
    def peft(config, cell, *, meta=False):
        import torch
        import transformers
        from peft import LoraConfig, get_peft_model
        audit = audit_native_architecture(config, cell)
        with torch.device("meta" if meta else "cpu"):
            model = getattr(transformers, config.architectures[0])(deepcopy(config))
            model = get_peft_model(model, LoraConfig(r=RANK, lora_alpha=ALPHA,
                lora_dropout=DROPOUT, target_modules=audit["exact_target_modules"],
                task_type="CAUSAL_LM", bias="none"))
        return model, audit

    def test_real_native_qwen_phi_smol_meta_peft_exact_shapes_and_counts(self):
        for kind, cell in (("qwen3", "T1"), ("phi3", "T5"), ("smollm3", "T6")):
            with self.subTest(kind=kind):
                model, audit = self.peft(self.config(kind), cell, meta=True)
                coverage = audit_lora_coverage(model, audit, require_quantized=False)
                self.assertEqual(coverage["adapter_parameters"], audit["expected_lora_parameters"])
                self.assertEqual(coverage["target_count"], audit["target_count"])
                self.assertTrue(coverage["vision_projector_embeddings_head_frozen"])
                self.assertFalse(coverage["quantized_targets_verified"])
                self.assertEqual(parameter_storage_report(model, audit)["runtime_unique_tensor_storage_bytes"], 0)
                with self.assertRaisesRegex(ValueError, "NONFINITE_OR_META"):
                    adapter_digest(model)

    def test_actual_multimodal_classes_keep_hybrid_fused_and_excluded_subtrees(self):
        from transformers import Qwen3_5Config, Gemma3Config, SmolVLMConfig
        text = dict(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                    num_attention_heads=4, num_key_value_heads=2, head_dim=4,
                    max_position_embeddings=128, pad_token_id=0, bos_token_id=1, eos_token_id=2)
        vision = dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                      num_attention_heads=4, image_size=8, patch_size=2)
        hybrid = Qwen3_5Config(text_config=dict(text, linear_num_key_heads=2, linear_key_head_dim=4,
            linear_num_value_heads=2, linear_value_head_dim=4, layer_types=["linear_attention", "full_attention"]),
            vision_config=dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=4,
                out_hidden_size=16, patch_size=2, spatial_merge_size=2, num_position_embeddings=16))
        gemma = Gemma3Config(text_config=text, vision_config=vision)
        smol = SmolVLMConfig(text_config=dict(text, model_type="llama"), vision_config=vision, scale_factor=2)
        for config, cell, architecture in ((hybrid, "V1", "Qwen3_5ForConditionalGeneration"),
                (gemma, "T3", "Gemma3ForConditionalGeneration"), (smol, "V4", "SmolVLMForConditionalGeneration")):
            with self.subTest(cell=cell):
                config.architectures = [architecture]
                model, audit = self.peft(config, cell, meta=True)
                coverage = audit_lora_coverage(model, audit, require_quantized=False)
                native = model.get_base_model()
                self.assertEqual(type(native).__name__, architecture)
                for prefix in audit["excluded_roots"]:
                    self.assertFalse(any(p.requires_grad for p in native.get_submodule(prefix).parameters()))
                self.assertEqual(coverage["target_count"], audit["target_count"])
                if cell == "V1":
                    self.assertTrue(any("linear_attn.in_proj_qkv" in p for p in coverage["exact_target_modules"]))
                    self.assertTrue(any("self_attn.q_proj" in p for p in coverage["exact_target_modules"]))

    def test_cpu_forward_backward_per_projection_gradient_and_exact_updates(self):
        import torch
        torch.manual_seed(19)
        model, audit = self.peft(self.config(), "T1")
        before = projection_adapter_digests(model, audit)
        digest = adapter_digest(model)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
        ids = torch.tensor([[1, 5, 7, 11, 2]])
        first = model(input_ids=ids, labels=ids).loss
        self.assertTrue(torch.isfinite(first))
        first.backward()
        report = projection_gradient_report(model, audit)
        self.assertEqual(report["projection_count"], audit["target_count"])
        self.assertTrue(all(not row["tensors"]["A"]["nonzero"] for row in report["projections"]))
        self.assertTrue(all(row["tensors"]["B"]["nonzero"] for row in report["projections"]))
        with self.assertRaisesRegex(ValueError, "GRADIENT_ZERO"):
            projection_gradient_report(model, audit, require_nonzero=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        model(input_ids=ids, labels=ids).loss.backward()
        projection_gradient_report(model, audit, require_nonzero=True)
        optimizer.step()
        self.assertNotEqual(adapter_digest(model), digest)
        after = projection_adapter_digests(model, audit)
        self.assertTrue(all(before[p][ab] != after[p][ab] for p in before for ab in ("A", "B")))
        storage = parameter_storage_report(model, audit)
        self.assertEqual(storage["original_dense_parameter_counts"], audit["parameter_counts"])
        self.assertGreater(storage["runtime_unique_tensor_storage_bytes"], 0)

    def test_missing_nonfinite_gradients_and_trainable_base_fail_closed(self):
        import torch
        model, audit = self.peft(self.config(), "T1")
        with self.assertRaisesRegex(ValueError, "GRADIENT_MISSING"):
            projection_gradient_report(model, audit)
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameter.grad = torch.ones_like(parameter)
        parameter = next(p for p in model.parameters() if p.requires_grad)
        parameter.grad.flatten()[0] = float("nan")
        with self.assertRaisesRegex(ValueError, "GRADIENT_NONFINITE"):
            projection_gradient_report(model, audit)
        parameter.grad = None
        base = model.get_base_model().get_input_embeddings().weight
        base.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "BASE_PARAMETER_TRAINABLE"):
            assert_frozen_base(model)
        base.requires_grad_(False)
        base.grad = torch.zeros_like(base)
        with self.assertRaisesRegex(ValueError, "FROZEN_BASE_HAS_GRADIENT"):
            assert_frozen_base(model)

    def test_wrong_rank_paths_shapes_or_eval_trainability_rejected(self):
        import torch
        model, audit = self.peft(self.config(), "T1")
        config = model.peft_config["default"]
        config.r = 4
        with self.assertRaisesRegex(ValueError, "RECIPE_MISMATCH"):
            audit_lora_coverage(model, audit, require_quantized=False)
        config.r = RANK
        config.target_modules = {"q_proj"}
        with self.assertRaisesRegex(ValueError, "CONFIG_TARGET_COVERAGE"):
            audit_lora_coverage(model, audit, require_quantized=False)
        config.target_modules = set(audit["exact_target_modules"])
        layer = model.get_base_model().get_submodule(audit["exact_target_modules"][0])
        original = layer.lora_A["default"]
        layer.lora_A["default"] = torch.nn.Linear(16, 4, bias=False)
        with self.assertRaisesRegex(ValueError, "SHAPE_MISMATCH"):
            audit_lora_coverage(model, audit, require_quantized=False)
        layer.lora_A["default"] = original
        with self.assertRaisesRegex(ValueError, "TRAINABILITY_MISMATCH"):
            audit_lora_coverage(model, audit, training=False, require_quantized=False)
        model.requires_grad_(False)
        self.assertEqual(audit_lora_coverage(model, audit, training=False,
            require_quantized=False)["trainable_tensor_count"], 0)

    def test_loader_refuses_floating_revision_before_model_or_gpu_load(self):
        for config in ({}, {"revision": "main"}, {"revision": "a" * 39}):
            with self.subTest(config=config), patch("mediflow_medvlm.training.v6_gpu._pinned_files") as pin:
                with self.assertRaisesRegex(ValueError, "IMMUTABLE_REVISION"):
                    load_v6_gpu_bundle(config, "T1")
                pin.assert_not_called()

    @unittest.skipUnless(os.environ.get("MEDVLM_V6_NATIVE_ASSETS_ROOT"), "optional pinned local processor roundtrips")
    def test_real_cached_frozen_processors_roundtrip_all_input_tensors(self):
        from PIL import Image
        from transformers import AutoConfig
        from mediflow_medvlm.training.models import load_processor
        from mediflow_medvlm.campaign_preflight import _checked_encode
        assets = Path(os.environ["MEDVLM_V6_NATIVE_ASSETS_ROOT"])
        for name, task in (("HuggingFaceTB--SmolLM3-3B", "text"),
                ("google--gemma-3-4b-it", "vqa"), ("HuggingFaceTB--SmolVLM2-2.2B-Instruct", "vqa"),
                ("Qwen--Qwen3.5-2B", "vqa")):
            with self.subTest(model=name), tempfile.TemporaryDirectory() as tmp:
                prepared = json.loads((assets / name / "native_assets.json").read_text())
                local = {"model_id": prepared["snapshot_path"], "revision": prepared["revision"],
                         "cache_dir": prepared["cache_dir"], "local_files_only": True}
                architecture = AutoConfig.from_pretrained(local["model_id"], local_files_only=True,
                                                          trust_remote_code=False)
                processor = load_processor(local, architecture)
                template = freeze_native_template(processor)
                directory = Path(tmp)
                image = directory / "image.png"
                Image.new("RGB", (64, 64), (31, 79, 127)).save(image)
                row = {"sample_id": "roundtrip", "task_type": "medical_vqa" if task == "vqa" else "medical_qa",
                       "input": {"question": "Return ready.", "images": [str(image)] if task == "vqa" else []},
                       "supervision": {"answer": "ready"}}
                before, metadata = _checked_encode(row, processor, architecture, task, None)
                saved = directory / "processor"
                processor.save_pretrained(saved)
                reloaded = _saved_processor(local, architecture, template, saved)
                after, repeated = _checked_encode(row, reloaded, architecture, task, None)
                self.assertEqual(set(before), set(after))
                for key in before:
                    self.assertTrue(before[key].equal(after[key]), key)
                self.assertEqual(metadata["assistant_start"], repeated["assistant_start"])
                self.assertEqual(metadata["supervised_tokens_after_shift"], repeated["supervised_tokens_after_shift"])


if __name__ == "__main__":
    unittest.main()
