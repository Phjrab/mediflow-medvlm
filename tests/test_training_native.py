"""Opt-in real torch/PEFT fixture: no Hub download and no medical claim."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from mediflow_medvlm.runtime import atomic_json, file_hash


@unittest.skipUnless(os.environ.get("MEDVLM_RUN_TRAINING_TESTS") == "1", "optional native training profile")
class NativeTrainingTests(unittest.TestCase):
    def test_cpu_lora_checkpoint_reload_and_padding(self):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import WhitespaceSplit
        from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
        from mediflow_medvlm.training.rendering import encode_sample, pad_text_batch
        from mediflow_medvlm.training.runner import train

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            vocab = {word: i for i, word in enumerate(["[UNK]", "<|user|>", "<|assistant|>", "<|end|>", "ready", "Question:", "Return", "the", "word", "ready.", "start"])}
            raw = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
            raw.pre_tokenizer = WhitespaceSplit()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]", eos_token="<|end|>", pad_token="<|end|>")
            tokenizer.chat_template = "{% for message in messages %}{{ '<|' + message['role'] + '|> ' + message['content'] + ' <|end|> ' }}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|> ' }}{% endif %}"
            tokenizer.save_pretrained(base)
            architecture = Qwen3Config(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=512, eos_token_id=3, pad_token_id=3, tie_word_embeddings=True)
            Qwen3ForCausalLM(architecture).save_pretrained(base)
            splits = {}
            for kind, count in (("train", 8), ("validation", 2)):
                path = root / (kind + ".jsonl")
                rows = [{"schema_version": "medvlm.dataset.v1", "sample_id": kind + str(i), "source_id": "engineering_fixture", "source_revision": "fixture-v1", "task_type": "medical_qa", "split": kind, "input": {"question": "Return the word ready.", "images": []}, "supervision": {"answer": "ready"}, "derivation": "engineering_fixture"} for i in range(count)]
                path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
                splits["fixture/" + kind] = {"path": str(path), "sha256": file_hash(path), "count": count}
            freeze = root / "freeze.json"
            atomic_json(freeze, {"splits": splits})
            sample = rows[0]
            first, first_metadata = encode_sample(sample, tokenizer, architecture)
            second, _ = encode_sample(sample, tokenizer, architecture, history=[{"role": "user", "content": "start"}, {"role": "assistant", "content": "start"}])
            padded = pad_text_batch([first, second], tokenizer.pad_token_id)
            self.assertTrue(bool((padded["labels"][padded["attention_mask"] == 0] == -100).all()))
            self.assertTrue(bool((first["labels"] == tokenizer.eos_token_id).any()))
            self.assertGreater(first_metadata["supervised_tokens_after_shift"], 0)
            config = {"run_id": "fixture", "pair_id": "fixture-pair", "experiment_id": "fixture", "model_id": str(base), "revision": "fixture-v1", "train_file": str(root / "train.jsonl"), "validation_file": str(root / "validation.jsonl"), "freeze_manifest": str(freeze), "data_root": str(root), "output_dir": str(root / "run"), "device": "cpu", "synthetic_fixture": True, "max_optimizer_steps": 2, "gradient_accumulation_steps": 2, "max_length": 256, "validation_samples": 2, "checkpoint_every_steps": 1, "lora_rank": 2, "lora_alpha": 4, "gradient_checkpointing": False, "reload_max_new_tokens": 4}
            config_path = root / "config.json"
            atomic_json(config_path, config)
            result = train(config_path)
            self.assertEqual(result["status"], "SUCCEEDED", result.get("error"))
            self.assertEqual(result["optimizer_steps"], 2)
            self.assertEqual(result["microsteps"], 4)
            self.assertTrue(result["adapter_changed"])
            self.assertEqual(result["evidence_level"], "SYNTHETIC_PIPELINE_ONLY")
            checkpoint = result["best_checkpoint"]
            fresh = subprocess.run([sys.executable, "-m", "mediflow_medvlm.training", "reload", "--config", str(config_path), "--checkpoint", checkpoint], text=True, capture_output=True, timeout=90)
            self.assertEqual(fresh.returncode, 0, fresh.stdout + fresh.stderr)
            self.assertTrue(json.loads((root / "run" / "selection.json").read_text())["reload_verified"])
            resumed = train(config_path, resume=result["latest_checkpoint"])
            self.assertEqual(resumed["status"], "SUCCEEDED", resumed.get("error"))
            self.assertTrue(resumed["resume_optimizer_restored"])
            self.assertEqual(resumed["optimizer_steps"], 2)
            self.assertTrue(Path(resumed["best_checkpoint"]).is_dir())
