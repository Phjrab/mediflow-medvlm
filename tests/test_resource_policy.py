"""Current execution policy, accounting and conservative migration CPU contracts."""
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from mediflow_medvlm.runtime import (BudgetExceeded, BudgetLedger, ResourcePolicy, atomic_json,
                                    canonical_hash, directory_digest, file_hash, resolve_resource_policy)
from mediflow_medvlm.training.config import load_config, training_limits, validate_resource_policy_migration


class ResourcePolicyTests(unittest.TestCase):
    def test_unlimited_crosses_old_boundaries_with_null_serialization(self):
        policy = ResourcePolicy()
        for hours in (6, 24, 36, 100):
            self.assertTrue(policy.can_continue(hours * 3600))
            self.assertTrue(policy.can_continue(hours * 3600, scope="job"))
            self.assertTrue(policy.can_continue(hours * 3600, scope="campaign"))
            self.assertIsNone(policy.deadline(0, hours * 3600))
        serialized = json.dumps(policy.as_dict(), allow_nan=False)
        self.assertIsNone(json.loads(serialized)["total_gpu_hours"])
        self.assertNotIn("Infinity", serialized)

    def test_finite_is_positive_and_unclamped(self):
        policy = ResourcePolicy("finite", 100, 48, 200000, 200001, 300000)
        with tempfile.TemporaryDirectory() as tmp:
            ledger = BudgetLedger(Path(tmp) / "ledger.json", policy=policy)
            self.assertEqual(ledger.remaining("test/model"), 48 * 3600)
        self.assertTrue(policy.can_continue(36 * 3600))
        self.assertFalse(policy.can_continue(200000))
        for value in (0, -1, float("inf"), float("nan"), True, "24"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ResourcePolicy("finite", value, 6)
        with self.assertRaises(ValueError):
            ResourcePolicy("unlimited", 24, 6)
        with self.assertRaises(ValueError):
            ResourcePolicy("finite", None, None)

    def test_current_unlimited_overrides_legacy_recipe_and_explicit_finite_remains_available(self):
        original = {"total_gpu_hours": 24, "per_model_gpu_hours": 6, "wall_seconds": 21600}
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_resource_policy(original).budget_mode, "unlimited")
            finite = resolve_resource_policy({**original, "budget_mode": "finite"})
            self.assertEqual(finite.run_wall_seconds, 21600)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "policy.json"
                atomic_json(path, ResourcePolicy().as_dict())
                with patch.dict(os.environ, {"MEDVLM_RESOURCE_POLICY": str(path)}):
                    self.assertEqual(resolve_resource_policy({**original, "budget_mode": "finite"}).budget_mode, "unlimited")
        self.assertEqual(original, {"total_gpu_hours": 24, "per_model_gpu_hours": 6, "wall_seconds": 21600})

    def test_usage_survives_37_hours_and_old_checkpoint_limits_never_restore_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.json"
            ledger = BudgetLedger(path, policy=ResourcePolicy())
            with patch("mediflow_medvlm.runtime.time.time", return_value=100):
                ledger.start("test/model", "run", "train")
            with patch("mediflow_medvlm.runtime.time.time", return_value=100 + 37 * 3600):
                self.assertIsNone(ledger.remaining("test/model"))
                self.assertTrue(ledger.can_continue("test/model", 900))
                saved = ledger.snapshot()
                ledger.finish()
            self.assertEqual(ledger.session_seconds, 37 * 3600)
            saved["total_limit_seconds"], saved["per_model_limit_seconds"] = 86400, 21600
            saved["resource_policy"] = ResourcePolicy("finite", 24, 6).as_dict()
            restored = BudgetLedger(path, policy=ResourcePolicy())
            self.assertEqual(restored.recover(saved)["test/model"], 0)
            self.assertIsNone(restored.remaining("test/model"))
            self.assertIsNone(restored.snapshot()["total_limit_seconds"])
            empty = BudgetLedger(Path(tmp) / "lost-ledger.json", policy=ResourcePolicy())
            self.assertEqual(empty.recover(saved)["test/model"], 37 * 3600)
            self.assertEqual(empty.recover(saved)["test/model"], 0)
            self.assertEqual(empty.snapshot()["run_phases"]["run"]["train"], 37 * 3600)
            with patch("mediflow_medvlm.runtime.time.time", return_value=200000):
                empty.start("test/model", "run", "evaluation")
            with patch("mediflow_medvlm.runtime.time.time", return_value=200010):
                empty.finish()
            snapshot = empty.snapshot()
            self.assertEqual(snapshot["models"]["test/model"], 37 * 3600 + 10)
            self.assertEqual(snapshot["run_phases"]["run"]["evaluation"], 10)
            self.assertEqual(snapshot["resource_policy"]["budget_mode"], "unlimited")
            finite = BudgetLedger(path, policy=ResourcePolicy("finite", 24, 6))
            self.assertFalse(finite.can_continue("test/model"))
            with self.assertRaises(BudgetExceeded):
                finite.start("test/model", "new-run", "train")

    def test_cli_jobs_use_resolved_null_or_finite_timeout_without_36_hour_clamp(self):
        from mediflow_medvlm.cli import run_plan
        for policy in (ResourcePolicy(), ResourcePolicy("finite", 100, 48, job_timeout_seconds=200000)):
            with self.subTest(mode=policy.budget_mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                plan = {"resource_policy": policy.as_dict(), "jobs": [{"id": "one", "command": ["fixture"], "timeout_seconds": 21600}]}
                atomic_json(root / "plan.json", plan)
                process = Mock(pid=123)
                process.wait.return_value = 0
                with patch("mediflow_medvlm.cli.subprocess.Popen", return_value=process) as launch, patch.dict(os.environ, {}, clear=True):
                    result = run_plan(root / "plan.json", root / "state.json")
                process.wait.assert_called_once_with(timeout=policy.job_timeout_seconds)
                process.terminate.assert_not_called()
                self.assertEqual(result["jobs"]["one"]["status"], "SUCCEEDED")
                child_policy = launch.call_args.kwargs["env"]["MEDVLM_RESOURCE_POLICY"]
                self.assertEqual(json.loads(Path(child_policy).read_text()), policy.as_dict())

    def test_campaign_invocation_uses_current_policy(self):
        spec = importlib.util.spec_from_file_location("policy_campaign", Path(__file__).resolve().parents[1] / "scripts/complete_v1.py")
        campaign = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(campaign)
        process = Mock(pid=123)
        process.wait.return_value = 0
        with tempfile.TemporaryDirectory() as tmp, patch.object(campaign.subprocess, "Popen", return_value=process), patch.dict(os.environ, {}, clear=True):
            self.assertEqual(campaign.invoke(["-m", "fixture"], Path(tmp) / "log"), 0)
        process.wait.assert_called_once_with(timeout=None)

    def test_explicit_finite_campaign_stops_at_its_limit_and_preserves_prior_jobs(self):
        from mediflow_medvlm.cli import run_plan
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy = ResourcePolicy("finite", 100, 48, campaign_wall_seconds=40 * 3600)
            atomic_json(root / "plan.json", {"resource_policy": policy.as_dict(), "jobs": [{"id": "pending", "command": ["fixture"]}]})
            atomic_json(root / "state.json", {"jobs": {"completed": {"status": "SUCCEEDED"}}, "campaign_elapsed_seconds": 40 * 3600})
            with patch("mediflow_medvlm.cli.subprocess.Popen") as launch, patch.dict(os.environ, {}, clear=True):
                state = run_plan(root / "plan.json", root / "state.json")
            launch.assert_not_called()
            self.assertEqual(state["reason_code"], "CAMPAIGN_WALL_LIMIT")
            self.assertEqual(state["jobs"]["completed"]["status"], "SUCCEEDED")


class V2RecipeTests(unittest.TestCase):
    def test_explicit_one_pass_exceeds_1000_without_changing_v1_caps(self):
        config = {"run_id": "v2", "pair_id": "v2", "experiment_id": "v2", "model_id": "test/model",
                  "revision": "a" * 40, "train_file": "train", "validation_file": "dev", "freeze_manifest": "freeze",
                  "output_dir": "out", "data_root": "data", "max_optimizer_steps": 2000,
                  "gradient_accumulation_steps": 16, "train_samples_cap": 32000}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            atomic_json(path, config)
            with self.assertRaisesRegex(ValueError, "v1 optimizer"):
                load_config(path)
            config.update(training_schema_version="medvlm.training.v2", training_recipe="one_pass")
            atomic_json(path, config)
            checked = load_config(path)
            self.assertEqual(training_limits(checked, 32000)["train_samples_cap"], 32000)
            self.assertEqual(training_limits({}, 32000)["train_samples_cap"], 20000)
            self.assertEqual(training_limits({}, 32000, True)["train_samples_cap"], 5000)
            with self.assertRaisesRegex(ValueError, "exceeds frozen"):
                training_limits(checked, 31999)
            config["max_optimizer_steps"] = 1500
            atomic_json(path, config)
            with self.assertRaisesRegex(ValueError, "ceil"):
                load_config(path)


class ResourceMigrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.old_config = {"model_id": "test/model", "revision": "a" * 40, "seed": 20261023,
                           "dtype": "bfloat16", "learning_rate": .0001, "optimizer": "adamw", "max_optimizer_steps": 1000,
                           "total_gpu_hours": 24, "per_model_gpu_hours": 6, "wall_seconds": 21600}
        self.new_config = {**self.old_config, "resource_policy": ResourcePolicy().as_dict()}
        self.old_source = self.source("old", "# old resource handling\n")
        self.new_source = self.source("new", "# new resource handling\n")
        self.identity = {"config_sha256": canonical_hash(self.old_config), "training_code_sha256": self.old_source[1],
                         "data": {"train": "fixed"}, "model_id": "test/model", "revision": "a" * 40,
                         "template_sha256": "fixed", "quantization": "nf4", "target_modules": ["q_proj"]}
        self.expected = {**self.identity, "config_sha256": canonical_hash(self.new_config), "training_code_sha256": self.new_source[1]}
        self.checkpoint = self.root / "checkpoint"
        self.checkpoint.mkdir()
        (self.checkpoint / "training_state.pt").write_bytes(b"engineering optimizer RNG fixture")
        self.state_hash = file_hash(self.checkpoint / "training_state.pt")
        atomic_json(self.checkpoint / "checkpoint.json", {"complete": True, "identity": self.identity,
                    "sample_order_sha256": "fixed-order", "artifacts": directory_digest(self.checkpoint)})
        self.manifest = {"status": "SKIPPED_BUDGET", "reason_code": "GPU_BUDGET_EXHAUSTED", "identity": self.identity}

    def source(self, name, content):
        root = self.root / name
        file = root / "src/mediflow_medvlm/runtime.py"
        file.parent.mkdir(parents=True)
        file.write_text(content)
        files = {"src/mediflow_medvlm/runtime.py": file_hash(file)}
        digest = canonical_hash(files)
        atomic_json(root / "receipt.json", {"captured_python_files": files,
                    "expected_code": {"source_files": files, "training_code_sha256": digest}})
        return root, digest

    def validate(self, **overrides):
        values = {"previous_config": self.old_config, "current_config": self.new_config,
                  "checkpoint_path": self.checkpoint, "expected_identity": self.expected, "previous_manifest": self.manifest,
                  "previous_source_snapshot": self.old_source[0], "current_source_snapshot": self.new_source[0],
                  "expected_sample_order_sha256": "fixed-order", "expected_training_state_sha256": self.state_hash}
        return validate_resource_policy_migration(**{**values, **overrides})

    def test_resource_only_proof_preserves_identity_gates_and_never_grants_automatic_resume(self):
        before = directory_digest(self.root)
        result = self.validate()
        self.assertEqual(result["current_policy"]["budget_mode"], "unlimited")
        self.assertEqual(result["previous_policy"]["total_gpu_hours"], 24)
        self.assertFalse(result["automatic_resume_authorized"])
        self.assertFalse(result["exact_resume_verified"])
        self.assertEqual(before, directory_digest(self.root))

    def test_rejects_recipe_seed_optimizer_precision_data_and_sampler_changes(self):
        for key, value in {"learning_rate": .001, "seed": 2, "optimizer": "sgd", "dtype": "float16", "train_file": "other", "max_optimizer_steps": 2000}.items():
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "cannot change recipe"):
                self.validate(current_config={**self.new_config, key: value})
        with self.assertRaisesRegex(ValueError, "sampler"):
            self.validate(expected_sample_order_sha256="changed")
        with self.assertRaisesRegex(ValueError, "optimizer/RNG"):
            self.validate(expected_training_state_sha256="changed")
        with self.assertRaisesRegex(ValueError, "checkpoint identity"):
            self.validate(expected_identity={**self.expected, "target_modules": ["v_proj"]})

    def test_completed_interrupted_and_non_budget_runs_are_not_migrated(self):
        for status, reason in (("SUCCEEDED", None), ("SKIPPED_BUDGET", "INTERRUPTED"), ("BLOCKED_ACCESS", "AUTH_REQUIRED")):
            with self.subTest(status=status, reason=reason), self.assertRaisesRegex(ValueError, "time limit"):
                self.validate(previous_manifest={**self.manifest, "status": status, "reason_code": reason})
        (self.old_source[0] / "src/mediflow_medvlm/runtime.py").write_text("tampered")
        with self.assertRaisesRegex(ValueError, "source bytes changed"):
            self.validate()
