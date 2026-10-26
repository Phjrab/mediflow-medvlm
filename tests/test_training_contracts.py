import json
from pathlib import Path
import tempfile
import unittest

from mediflow_medvlm.runtime import BudgetLedger, GPULease, ResourceBusy, atomic_json, canonical_hash
from mediflow_medvlm.training.config import assert_resume_identity
from mediflow_medvlm.training.rendering import assistant_target, user_text, image_paths


class TrainingContractTests(unittest.TestCase):
    def test_canonical_string_image_validates_provenance_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "image.png"
            path.write_bytes(b"changed-image-content")
            sample = {"task_type": "medical_vqa", "input": {"images": [str(path)]}, "provenance": {"image_sha256": "wrong-digest"}}
            with self.assertRaisesRegex(ValueError, "image hash changed"):
                image_paths(sample)

    def test_validation_selection_balances_sources(self):
        from mediflow_medvlm.training.runner import balanced_validation_rows
        rows = [{"sample_id": f"{source}-{i}", "source_id": source} for source in ("medquad", "pubmedqa_labeled") for i in range(100)]
        selected = balanced_validation_rows(rows)
        self.assertEqual(sum(row["source_id"] == "medquad" for row in selected), 16)
        self.assertEqual(selected, balanced_validation_rows(list(reversed(rows))))

    def test_gold_is_excluded_from_user_prompt(self):
        sample = {"task_type": "medical_mcq", "input": {"question": "Choose.", "options": {"A": "first", "B": "second"}},
                  "supervision": {"answer": "SECRET_GOLD", "option_id": "B", "rationale": "SECRET_RATIONALE"}}
        self.assertNotIn("SECRET", user_text(sample))
        self.assertEqual(assistant_target(sample), "B")
        sample["input"]["answer"] = "leak"
        with self.assertRaisesRegex(ValueError, "unsupported model input"):
            user_text(sample)

    def test_resume_rejects_changed_identity_and_incomplete_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.json"
            atomic_json(path, {"complete": True, "identity": {"config": "a", "model": "b"}})
            assert_resume_identity({"config": "a", "model": "b"}, tmp)
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                assert_resume_identity({"config": "a", "model": "different"}, tmp)
            atomic_json(path, {"complete": False, "identity": {}})
            with self.assertRaisesRegex(ValueError, "incomplete"):
                assert_resume_identity({}, tmp)

    def test_gpu_lease_excludes_second_owner(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gpu.lock"
            with GPULease(path, "first"):
                with self.assertRaises(ResourceBusy):
                    with GPULease(path, "second"):
                        pass
            with GPULease(path, "third"):
                self.assertEqual(json.loads(path.read_text())["run_id"], "third")

    def test_budget_is_model_scoped_not_revision_or_run_scoped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.json"
            first = BudgetLedger(path, 24, 6).start("Qwen/Qwen3-1.7B", "r1", "probe")
            first.finish()
            old = json.loads(path.read_text())
            old["models"]["Qwen/Qwen3-1.7B"] = 120
            atomic_json(path, old)
            second = BudgetLedger(path, 24, 6).start("Qwen/Qwen3-1.7B", "new-revision-r2", "evaluation")
            self.assertLess(second.remaining("Qwen/Qwen3-1.7B"), 6 * 3600 - 119)
            second.finish()
            self.assertEqual(len(json.loads(path.read_text())["sessions"]), 2)
