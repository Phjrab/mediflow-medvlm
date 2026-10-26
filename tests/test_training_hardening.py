"""Regression checks for failed resumes, native probe masks and base identity."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mediflow_medvlm.runtime import atomic_json, canonical_hash, file_hash
from mediflow_medvlm.training.runner import train
from mediflow_medvlm.runtime import BudgetLedger


def fixture_config(root):
    return {"run_id": "fixture", "pair_id": "pair", "experiment_id": "fixture",
            "model_id": str(root / "base"), "revision": "fixture", "synthetic_fixture": True,
            "train_file": str(root / "train.jsonl"), "validation_file": str(root / "validation.jsonl"),
            "freeze_manifest": str(root / "freeze.json"), "data_root": str(root),
            "output_dir": str(root / "run"), "device": "cpu", "max_optimizer_steps": 1,
            "gradient_accumulation_steps": 1, "max_length": 256, "validation_samples": 1,
            "checkpoint_every_steps": 1, "lora_rank": 2, "lora_alpha": 4,
            "gradient_checkpointing": False, "reload_max_new_tokens": 2}


class AttemptHistoryTests(unittest.TestCase):
    def test_budget_restore_uses_maximum_and_keeps_new_session_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "budget.json"
            ledger = BudgetLedger(path)
            with patch("mediflow_medvlm.runtime.time.time", return_value=100.0):
                ledger.start("fixture/model", "run", "train")
            with patch("mediflow_medvlm.runtime.time.time", return_value=130.0):
                saved = ledger.snapshot()
            with patch("mediflow_medvlm.runtime.time.time", return_value=140.0):
                ledger.finish()
            restored = BudgetLedger(path)
            self.assertEqual(restored.recover(saved)["fixture/model"], 0.0)
            self.assertEqual(restored.snapshot()["models"]["fixture/model"], 40.0)
            with patch("mediflow_medvlm.runtime.time.time", return_value=150.0):
                restored.start("fixture/model", "run", "train")
            with patch("mediflow_medvlm.runtime.time.time", return_value=160.0):
                restored.finish()
            self.assertEqual(restored.snapshot()["models"]["fixture/model"], 50.0)
            lost = BudgetLedger(Path(tmp) / "recovered-budget.json")
            self.assertEqual(lost.recover(saved)["fixture/model"], 30.0)
            self.assertEqual(lost.recover(saved)["fixture/model"], 0.0)
            with patch("mediflow_medvlm.runtime.time.time", return_value=200.0):
                lost.start("fixture/model", "run", "train")
            with patch("mediflow_medvlm.runtime.time.time", return_value=210.0):
                lost.finish()
            self.assertEqual(lost.snapshot()["models"]["fixture/model"], 40.0)
            self.assertEqual(lost.snapshot()["run_phases"]["run"]["train"], 40.0)

    def test_cumulative_attempt_time_deduplicates_archived_manifest(self):
        from mediflow_medvlm.training.runner import prior_attempt_wall_seconds
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = {"attempt_id": "first", "finished_at": "recorded", "wall_seconds": 10.0, "attempt_wall_seconds": 10.0}
            atomic_json(root / "attempts" / "first.json", old)
            atomic_json(root / "attempts" / "previous-first.json", old)
            atomic_json(root / "attempts" / "failed.json", {"attempt_id": "failed", "finished_at": "recorded", "attempt_wall_seconds": 4.0, "wall_seconds": 14.0})
            self.assertEqual(prior_attempt_wall_seconds(root, "current", old), 14.0)

    def test_explicit_legacy_migration_requires_source_receipt_and_preserved_base(self):
        from mediflow_medvlm.training.config import assert_resume_identity
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "source_snapshot"
            source_path = snapshot / "src/mediflow_medvlm/training/runner.py"
            source_path.parent.mkdir(parents=True)
            source_path.write_text("# original engineering fixture trainer\n")
            captured = {"src/mediflow_medvlm/training/runner.py": file_hash(source_path)}
            old = {"config_sha256": "config", "data": {"train": "digest"}, "training_code_sha256": canonical_hash(captured),
                   "model_id": "fixture/model", "revision": "immutable", "template_sha256": "template",
                   "renderer_version": "v1", "target_modules": ["q_proj"], "quantization": "none"}
            new = {**old, "training_code_sha256": "new-code", "identity_schema_version": "medvlm.training-identity.v2", "base_files_sha256": "base"}
            manifest = {"identity": old, "model": {"native_files": {"sha256": "base"}}}
            atomic_json(snapshot / "receipt.json", {"captured_python_files": captured,
                        "expected_code": {"source_files": captured, "training_code_sha256": canonical_hash(captured)}})
            checkpoint = root / "checkpoint"
            atomic_json(checkpoint / "checkpoint.json", {"complete": True, "identity": old, "artifacts": {"files": {}}})
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                assert_resume_identity(new, checkpoint)
            migrated = assert_resume_identity(new, checkpoint, legacy_source_snapshot=snapshot, legacy_manifest=manifest)
            self.assertEqual(migrated["identity_migration"]["policy"], "explicit-verified-source-snapshot-v1")
            self.assertEqual(assert_resume_identity(old, checkpoint)["identity"], old)
            source_path.write_text("changed")
            with self.assertRaisesRegex(ValueError, "snapshot file digest"):
                assert_resume_identity(new, checkpoint, legacy_source_snapshot=snapshot, legacy_manifest=manifest)
            broken_v2 = {**new}
            del broken_v2["base_files_sha256"]
            atomic_json(checkpoint / "checkpoint.json", {"complete": True, "identity": broken_v2})
            with self.assertRaisesRegex(ValueError, "v2 identity requires"):
                assert_resume_identity(broken_v2, checkpoint)

    def test_invalid_resume_preserves_prior_manifest_and_records_failed_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = fixture_config(root)
            config_path = root / "config.json"
            atomic_json(config_path, config)
            old = {"status": "SUCCEEDED", "optimizer_steps": 37, "best_checkpoint": "preserve-me"}
            manifest_path = root / "run" / "manifest.json"
            atomic_json(manifest_path, old)
            with patch("mediflow_medvlm.training.runner.load_frozen_data", side_effect=ValueError("deliberate freeze mismatch")):
                result = train(config_path, resume=root / "invalid-checkpoint")
            self.assertEqual(result["status"], "FAILED_REPRODUCIBLE")
            self.assertEqual(json.loads(manifest_path.read_text()), old)
            attempts = [json.loads(path.read_text()) for path in (root / "run" / "attempts").glob("*.json")]
            self.assertTrue(any(item.get("error") == "deliberate freeze mismatch" for item in attempts))
            self.assertIn(old, attempts)


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "optional torch profile")
class NativeProbeRegressionTests(unittest.TestCase):
    def test_probe_preserves_multimodal_token_type_ids_and_pad_zero(self):
        import torch
        from mediflow_medvlm.training.models import probe_model
        from mediflow_medvlm.training.rendering import pad_text_batch
        one = {"input_ids": torch.tensor([[1, 9]]), "attention_mask": torch.tensor([[1, 1]]),
               "labels": torch.tensor([[-100, 9]]), "mm_token_type_ids": torch.tensor([[0, 0]])}
        two = {"input_ids": torch.tensor([[1, 2, 9]]), "attention_mask": torch.tensor([[1, 1, 1]]),
               "labels": torch.tensor([[-100, -100, 9]]), "mm_token_type_ids": torch.tensor([[0, 0, 0]])}
        padded = pad_text_batch([one, two], 0)
        self.assertEqual(padded["mm_token_type_ids"].shape, (2, 3))
        self.assertEqual(padded["input_ids"][0, 2], 0)
        self.assertEqual(padded["labels"][0, 1], 9)
        self.assertEqual(padded["labels"][0, 2], -100)
        processor = SimpleNamespace(chat_template="native-template", pad_token_id=0, eos_token_id=9)
        hub = [{"sha": "a" * 40, "cardData": {"license": "apache-2.0"}, "siblings": []},
               {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}]
        with patch("mediflow_medvlm.training.models.hub_json", side_effect=hub), \
             patch("mediflow_medvlm.training.models.load_processor", return_value=processor), \
             patch("mediflow_medvlm.training.rendering.encode_sample", side_effect=[(one, {"prefix_verified": True}), (two, {"prefix_verified": True})]):
            result = probe_model("fixture/native-model", tokenizer_check=True)
        self.assertEqual(result["status"], "PREPARED_NOT_RUN", result)
        self.assertEqual(result["mask_checks"]["padding"], "passed")
        self.assertFalse(result["forward_tested"])

    def test_reload_rejects_mutated_config_before_model_load(self):
        from mediflow_medvlm.training.runner import reload_checkpoint
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = fixture_config(root)
            changed = {**original, "quantization": "nf4"}
            path = root / "config.json"
            atomic_json(path, changed)
            atomic_json(root / "run" / "manifest.json", {"config_hash": canonical_hash(original)})
            with patch("mediflow_medvlm.training.runner.load_model") as load:
                with self.assertRaisesRegex(ValueError, "reload config hash"):
                    reload_checkpoint(path, root / "checkpoint")
                load.assert_not_called()


@unittest.skipUnless(os.environ.get("MEDVLM_RUN_TRAINING_TESTS") == "1", "optional native training profile")
class BaseIdentityTests(unittest.TestCase):
    def test_v2_one_pass_1001_updates_survives_old_wall_limits_on_cpu(self):
        import itertools
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path, config = make_native_fixture(root, updates=1001, accumulation=1)
            config.update(training_schema_version="medvlm.training.v2", training_recipe="one_pass",
                          train_samples_cap=1001, checkpoint_every_steps=1001,
                          total_gpu_hours=24, per_model_gpu_hours=6, wall_seconds=21600)
            atomic_json(config_path, config)
            ticks = itertools.count(0, 37 * 3600)
            with patch("mediflow_medvlm.training.runner.time.monotonic", side_effect=lambda: next(ticks)):
                result = train(config_path)
            self.assertEqual(result["status"], "SUCCEEDED", result.get("error"))
            self.assertEqual(result["optimizer_steps"], 1001)
            self.assertEqual(result["samples_processed"], 1001)
            self.assertEqual(result["actual_epochs"], 1)
            self.assertTrue(result["adapter_changed"])
            self.assertEqual(result["resource_policy"]["budget_mode"], "unlimited")
            self.assertIsNone(result["resource_policy"]["run_wall_seconds"])

    def test_explicit_legacy_native_resume_and_fresh_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path, _ = make_native_fixture(root, updates=1, accumulation=1)
            before = train(config_path)
            self.assertEqual(before["status"], "SUCCEEDED", before.get("error"))
            checkpoint_path = Path(before["best_checkpoint"])
            checkpoint = json.loads((checkpoint_path / "checkpoint.json").read_text())
            snapshot = root / "run" / "source_snapshot"
            source_path = snapshot / "src/mediflow_medvlm/training/runner.py"
            source_path.parent.mkdir(parents=True)
            source_path.write_text("# preserved original engineering fixture trainer\n")
            captured = {"src/mediflow_medvlm/training/runner.py": file_hash(source_path)}
            old_identity = {key: value for key, value in before["identity"].items()
                            if key not in {"identity_schema_version", "base_files_sha256"}}
            old_identity["training_code_sha256"] = canonical_hash(captured)
            before["identity"] = old_identity
            checkpoint["identity"] = old_identity
            atomic_json(root / "run" / "manifest.json", before)
            atomic_json(checkpoint_path / "checkpoint.json", checkpoint)
            atomic_json(snapshot / "receipt.json", {"captured_python_files": captured,
                        "expected_code": {"source_files": captured,
                                          "training_code_sha256": canonical_hash(captured)}})
            rejected = train(config_path, resume=checkpoint_path)
            self.assertEqual(rejected["status"], "FAILED_REPRODUCIBLE")
            self.assertEqual(json.loads((root / "run" / "manifest.json").read_text())["identity"], old_identity)
            migrated = train(config_path, resume=checkpoint_path, legacy_source_snapshot=snapshot)
            self.assertEqual(migrated["status"], "SUCCEEDED", migrated.get("error"))
            self.assertTrue(migrated["resume_identity_verified"])
            self.assertEqual(migrated["optimizer_steps"], 1)
            self.assertEqual(migrated["legacy_identity_provenance"]["identity"], old_identity)
            self.assertEqual(migrated["resume_identity_migration"]["policy"], "explicit-verified-source-snapshot-v1")
            fresh = subprocess.run([sys.executable, "-m", "mediflow_medvlm.training", "reload", "--config", str(config_path), "--checkpoint", str(checkpoint_path)],
                                   text=True, capture_output=True, timeout=90)
            self.assertEqual(fresh.returncode, 0, fresh.stdout + fresh.stderr)
            reloaded = json.loads((root / "run" / "reload.json").read_text())
            self.assertEqual(reloaded["status"], "SUCCEEDED", reloaded)
            again = train(config_path, resume=checkpoint_path, legacy_source_snapshot=snapshot)
            self.assertEqual(again["status"], "SUCCEEDED", again.get("error"))
            self.assertEqual(again["legacy_identity_provenance"]["identity"], old_identity)

    def test_interrupted_microbatch_discards_group_and_preserves_rng_sampler(self):
        import signal
        import torch
        from mediflow_medvlm.training import runner
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path, config = make_native_fixture(root, updates=2, accumulation=2)
            captured = []
            original_capture = runner.capture_rng
            original_backward = torch.Tensor.backward
            def capture():
                value = original_capture()
                captured.append(value)
                return value
            calls = []
            def backward(tensor, *args, **kwargs):
                value = original_backward(tensor, *args, **kwargs)
                calls.append(True)
                if len(calls) == 1:
                    signal.raise_signal(signal.SIGTERM)
                return value
            with patch.object(runner, "capture_rng", side_effect=capture), patch.object(torch.Tensor, "backward", backward):
                stopped = train(config_path)
            self.assertEqual(stopped["status"], "SKIPPED_BUDGET", stopped.get("error"))
            self.assertEqual(stopped["optimizer_steps"], 0)
            self.assertFalse(stopped["adapter_changed"])
            saved = torch.load(Path(stopped["latest_checkpoint"]) / "training_state.pt", weights_only=False)
            self.assertEqual(saved["state"]["next_sample_position"], 0)
            self.assertEqual(saved["state"]["microsteps"], 0)
            self.assertTrue(torch.equal(saved["torch_rng"], captured[0]["torch"]))
            resumed = train(config_path, resume=stopped["latest_checkpoint"])
            self.assertEqual(resumed["status"], "SUCCEEDED", resumed.get("error"))
            self.assertEqual(resumed["optimizer_steps"], 2)
            self.assertGreater(resumed["wall_seconds"], resumed["attempt_wall_seconds"])
            full_config = {**config, "run_id": "uninterrupted", "output_dir": str(root / "uninterrupted")}
            full_path = root / "uninterrupted.json"
            atomic_json(full_path, full_config)
            uninterrupted = train(full_path)
            self.assertEqual(uninterrupted["status"], "SUCCEEDED", uninterrupted.get("error"))
            self.assertEqual(resumed["adapter_parameters_after_sha256"], uninterrupted["adapter_parameters_after_sha256"])

    def test_modified_native_base_is_rejected_on_resume_and_fresh_reload(self):
        from safetensors.torch import load_file, save_file
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import WhitespaceSplit
        from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = fixture_config(root)
            base = root / "base"
            vocab = {word: index for index, word in enumerate(["[UNK]", "<|user|>", "<|assistant|>", "<|end|>", "ready"])}
            raw = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
            raw.pre_tokenizer = WhitespaceSplit()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]", eos_token="<|end|>", pad_token="<|end|>")
            tokenizer.chat_template = "{% for message in messages %}{{ '<|' + message['role'] + '|> ' + message['content'] + ' <|end|> ' }}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|> ' }}{% endif %}"
            tokenizer.save_pretrained(base)
            architecture = Qwen3Config(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=512, eos_token_id=3, pad_token_id=3, tie_word_embeddings=True)
            Qwen3ForCausalLM(architecture).save_pretrained(base)
            splits = {}
            for kind in ("train", "validation"):
                row = {"schema_version": "medvlm.dataset.v1", "sample_id": kind, "source_id": "engineering_fixture", "source_revision": "fixture", "task_type": "medical_qa", "split": kind, "input": {"question": "Return ready.", "images": []}, "supervision": {"answer": "ready"}, "derivation": "engineering_fixture"}
                path = root / (kind + ".jsonl")
                path.write_text(json.dumps(row) + "\n")
                splits["fixture/" + kind] = {"sha256": file_hash(path), "count": 1, "path": str(path)}
            atomic_json(root / "freeze.json", {"splits": splits})
            config_path = root / "config.json"
            atomic_json(config_path, config)
            before = train(config_path)
            self.assertEqual(before["status"], "SUCCEEDED", before.get("error"))
            checkpoint = before["best_checkpoint"]
            manifest_before = (root / "run" / "manifest.json").read_text()
            weights_path = base / "model.safetensors"
            weights = load_file(weights_path)
            name = next(iter(weights))
            weights[name] = weights[name].clone()
            weights[name].view(-1)[0] += 0.125
            save_file(weights, weights_path, metadata={"format": "pt"})
            resumed = train(config_path, resume=checkpoint)
            self.assertEqual(resumed["status"], "FAILED_REPRODUCIBLE")
            self.assertIn("resume identity mismatch", resumed["error"])
            self.assertEqual((root / "run" / "manifest.json").read_text(), manifest_before)
            fresh = subprocess.run([sys.executable, "-m", "mediflow_medvlm.training", "reload", "--config", str(config_path), "--checkpoint", checkpoint], text=True, capture_output=True, timeout=90)
            self.assertEqual(fresh.returncode, 2, fresh.stdout + fresh.stderr)
            reloaded = json.loads((root / "run" / "reload.json").read_text())
            self.assertEqual(reloaded["status"], "FAILED_REPRODUCIBLE")
            self.assertIn("BASE_FILES_DIGEST_MISMATCH", reloaded["error"])


def make_native_fixture(root, updates=2, accumulation=2):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
    config = {**fixture_config(root), "max_optimizer_steps": updates, "gradient_accumulation_steps": accumulation}
    base = root / "base"
    vocab = {word: index for index, word in enumerate(["[UNK]", "<|user|>", "<|assistant|>", "<|end|>", "ready"])}
    raw = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    raw.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]", eos_token="<|end|>", pad_token="<|end|>")
    tokenizer.chat_template = "{% for message in messages %}{{ '<|' + message['role'] + '|> ' + message['content'] + ' <|end|> ' }}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|> ' }}{% endif %}"
    tokenizer.save_pretrained(base)
    architecture = Qwen3Config(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=512, eos_token_id=3, pad_token_id=3, tie_word_embeddings=True)
    Qwen3ForCausalLM(architecture).save_pretrained(base)
    splits = {}
    for kind, count in (("train", updates * accumulation), ("validation", 1)):
        rows = [{"schema_version": "medvlm.dataset.v1", "sample_id": kind + str(index), "source_id": "engineering_fixture", "source_revision": "fixture", "task_type": "medical_qa", "split": kind, "input": {"question": "Return ready.", "images": []}, "supervision": {"answer": "ready"}, "derivation": "engineering_fixture"} for index in range(count)]
        path = root / (kind + ".jsonl")
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        splits["fixture/" + kind] = {"sha256": file_hash(path), "count": count, "path": str(path)}
    atomic_json(root / "freeze.json", {"splits": splits})
    config_path = root / "config.json"
    atomic_json(config_path, config)
    return config_path, config
