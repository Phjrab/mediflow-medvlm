"""V6 native contracts: dependency-free guards plus optional real meta modules."""
from copy import deepcopy
import importlib.util
from types import SimpleNamespace
import unittest

from mediflow_medvlm.training.v6_native import (
    audit_native_architecture, freeze_native_template,
    verify_native_no_thinking, verify_native_no_thinking_output,
)


class FrozenTemplateTests(unittest.TestCase):
    def test_native_date_copy_is_deterministic_and_preserves_other_bytes(self):
        source = '{%- set today = strftime_now("%d %B %Y") -%}\n/think /no_think {{ today }}'
        tokenizer = SimpleNamespace(chat_template=source)
        receipt = freeze_native_template(tokenizer)
        self.assertEqual(tokenizer.chat_template,
                         '{%- set today = "14 November 2026" -%}\n/think /no_think {{ today }}')
        self.assertEqual(receipt["templates"]["processor"]["clock_calls_replaced"], 1)
        self.assertEqual(freeze_native_template(tokenizer), receipt)
        receipt["fixed_date"] = "bad"
        self.assertEqual(freeze_native_template(tokenizer)["fixed_date"], "2026-11-14")
        with self.assertRaisesRegex(ValueError, "DIFFERENT_DATE"):
            freeze_native_template(tokenizer, "2026-11-15")
        tokenizer.chat_template += "edited"
        with self.assertRaisesRegex(ValueError, "BYTES_CHANGED"):
            freeze_native_template(tokenizer)

    def test_named_templates_and_processor_delegate_are_not_aliased(self):
        source = {"default": "{{ strftime_now('%d %B %Y') }}", "tool_use": "native tools"}
        original = deepcopy(source)
        tokenizer = SimpleNamespace(chat_template=source)
        processor = SimpleNamespace(chat_template=None, tokenizer=tokenizer)
        receipt = freeze_native_template(processor, "2024-02-29")
        self.assertEqual(source, original)
        self.assertIsNone(processor.chat_template)
        self.assertTrue(receipt["templates"]["processor"]["delegates_to_tokenizer"])
        self.assertEqual(tokenizer.chat_template["default"], '{{ "29 February 2024" }}')
        self.assertEqual(freeze_native_template(processor, "2024-02-29"), receipt)

    def test_unknown_dynamic_clock_fails_before_any_assignment(self):
        processor = SimpleNamespace(chat_template="{{ strftime_now('%d %B %Y') }}",
                                    tokenizer=SimpleNamespace(chat_template="{{ strftime_now('%Y') }}"))
        original = processor.chat_template
        with self.assertRaisesRegex(ValueError, "UNKNOWN_CLOCK"):
            freeze_native_template(processor)
        self.assertEqual(processor.chat_template, original)
        self.assertFalse(hasattr(processor, "_medvlm_v6_template_receipt"))

    def test_malformed_and_missing_templates_rejected(self):
        for template in (None, {}, ["template"], {"default": None}):
            with self.subTest(template=template), self.assertRaisesRegex(ValueError, "MALFORMED"):
                freeze_native_template(SimpleNamespace(chat_template=template))
        with self.assertRaises(ValueError):
            freeze_native_template(SimpleNamespace(chat_template="plain"), "2026-02-30")

    def test_output_check_never_claims_hidden_reasoning(self):
        for output in ("ready", "<think>\n \n</think>ready"):
            self.assertFalse(verify_native_no_thinking_output(output)["hidden_reasoning_cost_verified"])
        for output in ("<think>reasoning</think>ready", "<think>", "ready</think>"):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "REASONING_SPAN"):
                verify_native_no_thinking_output(output)


class ProbeTemplate:
    def __init__(self, mode="smollm3", forced_think=False, ignore_switch=False):
        self.mode, self.forced_think, self.ignore_switch = mode, forced_think, ignore_switch

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        thinking = self.forced_think or (enable_thinking and not self.ignore_switch)
        prefix = "<|im_start|>assistant\n"
        if not thinking:
            prefix += "<think>\n\n</think>\n" + ("" if self.mode == "smollm3" else "\n")
        text = "Reasoning Mode: " + ("/think" if thinking else "/no_think") + "\nuser\n"
        return text + prefix + ("" if add_generation_prompt else "ready<|im_end|>")


class ThinkingTests(unittest.TestCase):
    def test_switch_is_exercised_and_output_remains_unverified(self):
        result = verify_native_no_thinking(ProbeTemplate(), "smollm3")
        self.assertEqual(result["status"], "NATIVE_DISABLE_SWITCH_VERIFIED")
        self.assertTrue(result["assistant_prefix_verified"])
        self.assertFalse(result["generated_output_verified"])

    def test_ineffective_and_ignored_switches_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "DISABLE_INEFFECTIVE"):
            verify_native_no_thinking(ProbeTemplate(forced_think=True), "smollm3")
        with self.assertRaisesRegex(ValueError, "SWITCH_NOT_EXERCISED"):
            verify_native_no_thinking(ProbeTemplate(ignore_switch=True), "smollm3")

    def test_unknown_family_and_bad_turns_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "MODEL_UNKNOWN"):
            verify_native_no_thinking(ProbeTemplate(), "llama")
        with self.assertRaisesRegex(ValueError, "FINAL_USER"):
            verify_native_no_thinking(ProbeTemplate(), "smollm3", [{"role": "assistant", "content": "x"}])
        template = ProbeTemplate()
        template.chat_template = "{{ strftime_now('%d %B %Y') }}"
        with self.assertRaisesRegex(ValueError, "CLOCK_NOT_FROZEN"):
            verify_native_no_thinking(template, "smollm3")


_HAS_NATIVE = bool(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"))


@unittest.skipUnless(_HAS_NATIVE, "native CPU/meta profile requires installed torch/transformers")
class NativeArchitectureTests(unittest.TestCase):
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

    def test_native_qwen_phi_smol_complete_projection_and_dense_counts(self):
        for kind, cell, expected in (("qwen3", "T1", 14), ("phi3", "T5", 8), ("smollm3", "T6", 14)):
            with self.subTest(kind=kind):
                config = self.config(kind)
                original = deepcopy(config.to_dict())
                result = audit_native_architecture(config, cell)
                self.assertEqual(result["target_count"], expected)
                self.assertEqual(result["allocated_weight_storage_bytes"], 0)
                self.assertFalse(result["gpu_verified"])
                self.assertFalse(result["weights_loaded"])
                self.assertEqual(config.to_dict(), original)
                self.assertTrue(all(p.startswith("model.layers.") for p in result["exact_target_modules"]))
                self.assertFalse(any("lm_head" in p for p in result["exact_target_modules"]))
                counts = result["parameter_counts"]
                self.assertEqual(counts["total"], sum(counts[k] for k in ("language", "vision", "projector", "other")))
                self.assertEqual(audit_native_architecture(config.to_dict(), cell)["exact_target_modules"], result["exact_target_modules"])
                if kind == "phi3":
                    self.assertEqual(result["coverage_by_projection"]["self_attn.qkv_proj"], 2)
                    self.assertEqual(result["coverage_by_projection"]["mlp.gate_up_proj"], 2)

    def test_missing_unexpected_and_wrong_shape_projection_rejected(self):
        import torch
        from transformers import Qwen3ForCausalLM
        config = self.config()
        for mutation, error in (("missing", "COVERAGE_MISMATCH"), ("extra", "COVERAGE_MISMATCH"),
                                ("shape", "SHAPE_OR_TYPE"), ("alias", "COVERAGE_MISMATCH")):
            with self.subTest(mutation=mutation):
                with torch.device("meta"):
                    model = Qwen3ForCausalLM(deepcopy(config))
                    if mutation == "missing":
                        del model.model.layers[0].self_attn.q_proj
                    elif mutation == "extra":
                        model.model.layers[1].mlp.unknown_projection = torch.nn.Linear(16, 16)
                    elif mutation == "shape":
                        model.model.layers[0].self_attn.q_proj = torch.nn.Linear(16, 17)
                    else:
                        model.model.layers[0].self_attn.q_proj = model.model.layers[0].self_attn.o_proj
                with self.assertRaisesRegex(ValueError, error):
                    audit_native_architecture(config, "T1", model=model)

    def test_wrong_native_architecture_and_cell_are_rejected(self):
        config = self.config()
        with self.assertRaisesRegex(ValueError, "ARCHITECTURE_MISMATCH"):
            audit_native_architecture(config, "T5")
        with self.assertRaisesRegex(ValueError, "CELL_UNKNOWN"):
            audit_native_architecture(config, "T99")

    def test_hybrid_attention_and_multimodal_subtrees_use_native_meta_modules(self):
        from transformers import Qwen3_5Config, Gemma3Config, SmolVLMConfig
        text = dict(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                    num_attention_heads=4, num_key_value_heads=2, head_dim=4,
                    max_position_embeddings=128, pad_token_id=0, bos_token_id=1, eos_token_id=2)
        vision = dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                      num_attention_heads=4, image_size=8, patch_size=2)
        hybrid = Qwen3_5Config(text_config=dict(text, linear_num_key_heads=2, linear_key_head_dim=4,
                                               linear_num_value_heads=2, linear_value_head_dim=4,
                                               layer_types=["linear_attention", "full_attention"]),
                                vision_config=dict(depth=1, hidden_size=16, intermediate_size=32,
                                                   num_heads=4, out_hidden_size=16, patch_size=2,
                                                   spatial_merge_size=2, num_position_embeddings=16))
        gemma = Gemma3Config(text_config=text, vision_config=vision)
        smol = SmolVLMConfig(text_config=dict(text, model_type="llama"), vision_config=vision,
                            scale_factor=2)
        for config, cell, architecture, count, excluded in (
                (hybrid, "V1", "Qwen3_5ForConditionalGeneration", 15, "model.visual"),
                (gemma, "T3", "Gemma3ForConditionalGeneration", 14, "model.multi_modal_projector"),
                (smol, "V4", "SmolVLMForConditionalGeneration", 14, "model.connector")):
            config.architectures = [architecture]
            with self.subTest(cell=cell):
                result = audit_native_architecture(config, cell)
                self.assertEqual(result["target_count"], count)
                self.assertIn(excluded, result["excluded_roots"])
                self.assertFalse(any(path.startswith(excluded + ".") for path in result["exact_target_modules"]))
                self.assertGreater(result["parameter_counts"]["vision"], 0)
                self.assertGreater(result["parameter_counts"]["projector"], 0)
                if cell == "V1":
                    self.assertEqual(result["coverage_by_projection"]["linear_attn.in_proj_qkv"], 1)
                    self.assertEqual(result["coverage_by_projection"]["self_attn.q_proj"], 1)
                    config.text_config.layer_types[0] = "unknown_attention"
                    with self.assertRaisesRegex(ValueError, "LAYER_TYPES_INVALID"):
                        audit_native_architecture(config, cell)


if __name__ == "__main__":
    unittest.main()
