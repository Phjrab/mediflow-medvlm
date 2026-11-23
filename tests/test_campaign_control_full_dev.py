"""CPU-only temporary fixtures for the full-dev sole-supervisor refinement."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "scripts"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


operator = load("full_dev_under_test", HERE / "campaign_control_full_dev.py")
caps = load("frozen_caps_fixture", HERE / "campaign_control_caps.py")
engine = load("original_full_dev_supervisor_fixture", ROOT / "src/mediflow_medvlm/campaign_supervisor.py")
base = load("original_full_dev_base_fixture", HERE / "campaign_control.py")


def write(path, value):
    engine.atomic_json(path, value)
    return Path(path)


def ref(path):
    return {"path": str(path), "sha256": engine.file_hash(path)}


class HarnessCaps(base.CampaignControl):
    """Original native owner guards with synthetic immutable inherited proofs.

    Only old owner proof payloads and verify_engine use temporary fixtures;
    loading all exact prior facade bytes is separately checked.
    """
    def __init__(self, *, inherited, roots, **kwargs):
        self.inherited, self.roots, self.cap_sha256 = inherited, roots, "c" * 64
        super().__init__(**kwargs)
        self.supervisor.__class__ = caps.make_supervisor_class(self.engine, self, type(self.supervisor))

    def _caps(self):
        return self.inherited, None, self.inherited["manifest"]

    def _validation(self):
        return {"engine": {"root": self.roots["initial"]}}, None, None

    def _stage(self):
        return {"engine": {"root": self.roots["aggregate"]}}, None, None

    def _recovery(self):
        return {"replacement": {"engine_root": self.roots["t5"]}}, None, None

    def _manifest(self, value):
        projected = copy.deepcopy(value)
        for job in projected["jobs"]:
            if job.get("argv"):
                job["cwd"] = str(self.engine_root)
        base.CampaignControl._manifest(self, projected)
        return value


class FixtureControl(operator.FullDevMixin, HarnessCaps):
    base = base


class FullDevOperatorTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        f = self.f = SimpleNamespace()
        f.data = Path(temp.name) / "private"
        f.old = f.data / "v6_v7"
        f.root = f.old / "campaigns/cpu-full-dev"
        f.state_path = f.root / "state.json"
        f.root.mkdir(parents=True)
        for path in (f.old / "supervisor.lock", f.old / "worker.lock", f.data / "runtime/medvlm_gpu.lock"):
            path.parent.mkdir(parents=True, exist_ok=True); path.touch()
        old_manifest = write(f.old / "manifests/old.json", {"campaign_id": "old", "jobs": []})
        old_state = write(f.old / "state.json", {"jobs": {}})
        f.hold = write(f.data / "hold.json", {"writer": "TRAINING_OWNER", "dispatch_control": {"status": "MIGRATION_HOLD", "tick_prohibited": True},
            "active_workers": [], "old_campaign": {"campaign_root": str(f.old), "manifest_path": str(old_manifest),
                "manifest_sha256": engine.file_hash(old_manifest), "state_path": str(old_state), "state_sha256": engine.file_hash(old_state)}})
        f.roots = {}
        for index, name in enumerate(("base", "t5", "aggregate", "initial", "caps", "full_dev"), 1):
            root = f.root / "snapshots" / (str(index) * 40)
            package = root / "src/mediflow_medvlm"; package.mkdir(parents=True)
            for module in ("campaign_supervisor", "campaign_main_train", "campaign_recorded_first_seed", "campaign_validation_caps", "campaign_validation_cap_refinement", "campaign_full_dev", "evaluation/metrics", "evaluation/v2", "telemetry_import"):
                path = package / (module + ".py"); path.parent.mkdir(parents=True, exist_ok=True); path.write_text("# immutable CPU source fixture\n")
            (root / "README.md").write_text("CPU full archive fixture\n")
            f.roots[name] = str(root)
        f.base_engine = Path(f.roots["base"]); self.new_engine = Path(f.roots["full_dev"])
        engine_patch = patch.object(base, "verify_engine", return_value=engine); engine_patch.start(); self.addCleanup(engine_patch.stop)
        self.before = {"campaign_id": f.root.name, "campaign_root": str(f.root), "declared_seed": 20261023,
            "scientific_metadata": {"protected": True}, "jobs": []}
        self.states, self.proofs, self.old_configs = {}, {}, {}
        for job_id in operator.PREREQUISITES:
            if job_id.startswith("v6-first-seed-"):
                cell, role, phase = job_id.removeprefix("v6-first-seed-"), None, "native_main_train"
                source = Path(f.roots["t5"] if cell == "T5" else f.roots["base"])
                module = "mediflow_medvlm.campaign_main_train"
            elif job_id == operator.FIRST_SEED_ID:
                cell, role, phase, source, module = "ALL_V6", None, "first_seed_final", Path(f.roots["aggregate"]), "mediflow_medvlm.campaign_recorded_first_seed"
            elif job_id in operator.INITIAL_IDS:
                cell, role = job_id.removeprefix("v6-caps-initial-").split("-")
                phase, source, module = "validation_caps_calibration", Path(f.roots["initial"]), "mediflow_medvlm.campaign_validation_caps"
            elif job_id in operator.BUMPED_IDS:
                cell, role = job_id.removeprefix("v6-caps-bumped-").split("-")
                phase, source, module = "validation_caps_bumped", Path(f.roots["caps"]), "mediflow_medvlm.campaign_validation_cap_refinement"
            else:
                cell, role, phase, source, module = "ALL_V6", None, "validation_caps" if job_id == operator.CAPS_ID else "validation_cap_decision", Path(f.roots["caps"]), "mediflow_medvlm.campaign_validation_cap_refinement"
            path = f.root / "configs/old" / (job_id + ".json")
            job = self.executable(job_id, source, module, path)
            job.update(phase=phase, cell_id=cell, uses_gpu=cell != "ALL_V6")
            if cell != "ALL_V6": job.update(model_id="original/" + cell, task="text" if cell.startswith("T") else "vqa", family="family/" + cell, seed=20261023)
            if role: job.update(role=role)
            config = {**{k: job[k] for k in ("job_id", "run_id", "cell_id", "phase", *engine.HASH_FIELDS, "source_files", "success_receipt")},
                "output_dir": str(f.root / "old-results" / job_id), "model": {"base": cell}, "decode": {"do_sample": False, "max_time": None},
                "data": {"image_root": str(f.data / "images")}}
            config.update(**{k: job[k] for k in ("model_id", "task", "family", "role", "seed") if k in job})
            config["model_hash"] = engine.canonical_hash({"model": config["model"], "role": role})
            write(path, config); job["model_hash"] = config["model_hash"]
            self.old_configs[job_id] = config
            self.before["jobs"].append(job); self.success(job, failure_count=2 if job_id == "v6-first-seed-T5" else 0)
        for index in range(10):
            job_id = operator.GATE_ID if index == 0 else f"fixture-future-{index}"
            row = {"job_id": job_id, "run_id": job_id, "phase": "full_dev" if index == 0 else "future_protected", "cell_id": "ALL_V6",
                "implementation_pending": True, "dependencies": [operator.CAPS_ID] if index == 0 else [operator.GATE_ID],
                "dependency_policy": "terminal", "status": "PENDING_IMPLEMENTATION", "reason": "Keep original declaration"}
            self.before["jobs"].append(row)
            self.states[job_id] = {"status": "IMPLEMENTATION_PENDING", "attempts": 0, "failure_count": 0, "recovery_attempts": 0, "history": []}
        initial = {"schema_version": "medvlm.v6.validation-cap-protocol.v1", "campaign_id": f.root.name, "campaign_root": str(f.root),
            "initial_caps": {"medical_mcq": 64, "context_qa": 64, "medical_qa": 256, "medical_vqa": 256},
            "maximum_caps": {"medical_mcq": 128, "context_qa": 128, "medical_qa": 512, "medical_vqa": 512},
            "decode": {"do_sample": False, "max_time": None}, "fixed_template_date": "2026-11-14", "resource_policy": {"total_gpu_hours": None},
            "models": {cell: {"config": ref(next(j["argv"][4] for j in self.before["jobs"] if j["job_id"] == "v6-first-seed-" + cell)),
                "checkpoint": {"path": str(f.root / (cell + ".adapter")), "sha256": "a" * 64}} for cell in operator.CELL_IDS},
            "first_seed_receipt": ref(next(j["success_receipt"] for j in self.before["jobs"] if j["job_id"] == operator.FIRST_SEED_ID)),
            "validation_references": {}}
        for axis, count in (("text", 1516), ("vqa", 512)):
            path = f.root / "datasets" / (axis + ".jsonl"); path.parent.mkdir(parents=True, exist_ok=True)
            rows = [{"sample_id": f"{axis}-{i}", "task_type": "medical_mcq" if axis == "text" else "medical_vqa", "input": {"text": f"question{i}"}, "supervision": {"answer": "A"}} for i in range(count)]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows)); initial["validation_references"][axis] = ref(path)
        initial["protocol_hash"] = engine.canonical_hash(initial)
        initial_path = write(f.root / "protocols/initial.json", initial); self.initial = initial
        protocol = {"schema_version": "medvlm.v6.validation-caps-frozen.v1", "initial_protocol": ref(initial_path), "initial_protocol_hash": initial["protocol_hash"],
            "caps": initial["maximum_caps"], "eval_context": 4096, "train_context": 2048, "decode": initial["decode"], "fixed_template_date": "2026-11-14",
            "cap_protocol_frozen": True, "further_bump_allowed": False, "test_files_read": False, "test_unlocked": False,
            "full_dev_completed": False, "family_selection_completed": False, "clinical_quality_certified": False, "campaign_terminal": False}
        protocol["protocol_hash"] = engine.canonical_hash(protocol)
        cap_job = next(j for j in self.before["jobs"] if j["job_id"] == operator.CAPS_ID)
        self.protocol_path = write(Path(self.old_configs[operator.CAPS_ID]["output_dir"]) / "protocol.final.json", protocol)
        cap_receipt = base.read_json(cap_job["success_receipt"]); cap_receipt["artifacts"].append(ref(self.protocol_path)); write(cap_job["success_receipt"], cap_receipt)
        record = self.states[operator.CAPS_ID]; record["verification"] = engine.verify_success(cap_job)
        self.proofs[operator.CAPS_ID] = {**ref(cap_job["success_receipt"]), "job_hash": engine.canonical_hash(cap_job), "state_hash": engine.canonical_hash(record)}
        self.parent_path = write(f.root / "manifests/full-dev-parent.json", self.before)
        write(f.state_path, {"campaign_id": f.root.name, "manifest_hash": engine.canonical_hash(self.before), "manifest_path": str(self.parent_path),
            "manifest_revisions": [{"kind": "PRESERVED_OLD_AUDIT"}], "queue_owner": "supervisor", "jobs": self.states, "finalization": "NOT_DONE", "updated_at": "preserved"})
        self.inherited = {"replacement": {"manifest_hash": engine.canonical_hash(self.before)}, "manifest": self.before,
            "engine": {"root": f.roots["caps"]}, "protocol": ref(initial_path)}
        self.after, children, configs = copy.deepcopy(self.before), [], {}
        self.forge = {"output_root": str(Path(temp.name) / "outputs"), "data_root": str(f.data), "campaign_id": f.root.name}
        role_paths = [f.root / "configs/full-dev" / (job_id + ".json") for job_id in operator.ADDED_IDS]
        for job_id in operator.EXECUTABLE_IDS:
            gpu = job_id in operator.ADDED_IDS
            path = role_paths[operator.ADDED_IDS.index(job_id)] if gpu else f.root / "configs/full-dev/aggregate.json"
            job = self.executable(job_id, self.new_engine, operator.WORKER_MODULE, path)
            job.update(uses_gpu=gpu, requires_gpu=gpu, gpu_lock_mode="wrapper", kind="role" if gpu else "aggregate", seed=20261023)
            if gpu:
                cell, role = job_id.removeprefix("v6-full-dev-").split("-")
                original = next(j for j in self.before["jobs"] if j["job_id"] == f"v6-caps-initial-{cell}-{role}")
                job.update(**{k: original[k] for k in ("cell_id", "model_id", "task", "family", "role")}, phase="full_dev_evaluation", dependencies=[operator.CAPS_ID], dependency_policy="success")
            else: job.update(cell_id="ALL_V6", phase="full_dev", dependencies=[operator.CAPS_ID, *operator.ADDED_IDS], dependency_policy="terminal")
            data = {"protocol": ref(self.protocol_path), "initial_protocol": ref(initial_path), "cap_freeze": {"job": cap_job,
                "terminal_state": self.states[operator.CAPS_ID], "config": ref(cap_job["argv"][4]), "receipt": ref(cap_job["success_receipt"]), "final_protocol": ref(self.protocol_path)}}
            if gpu:
                validation = initial["validation_references"][job["task"]]
                ids = [json.loads(line)["sample_id"] for line in Path(validation["path"]).read_text().splitlines()]
                data.update(validation=validation, ordered_sample_ids=ids, ordered_sample_ids_sha256=engine.canonical_hash(ids), count=len(ids),
                    partition="validation", image_root=str(f.data / "images"), trained_cell=initial["models"][cell], first_seed_receipt=initial["first_seed_receipt"])
                rows = [json.loads(line) for line in Path(validation["path"]).read_text().splitlines()]
                source_files = job["source_files"]
                evaluation = {"protocol_version": "medvlm.v6.full-dev.v1", "split": "validation", "cap_protocol": ref(self.protocol_path),
                    "cap_protocol_hash": protocol["protocol_hash"], "caps": protocol["caps"], "eval_context": 4096, "decode": protocol["decode"],
                    "fixed_template_date": "2026-11-14", "sample_ids": ids, "input_digests": {r["sample_id"]: engine.canonical_hash(r["input"]) for r in rows},
                    "reference_digest": engine.canonical_hash([(r["sample_id"], r["supervision"]) for r in rows]), "files": {job["task"]: validation["sha256"]},
                    "parser": "existing anchored full-string parse_answer; no answer substring extraction",
                    "normalization": "NFKC casefold whitespace; VQA terminal punctuation only; preserve numbers/units/negation",
                    "scorer_source_files": {k: v for k, v in source_files.items() if k.endswith(("evaluation/metrics.py", "evaluation/v2.py", "campaign_full_dev.py"))},
                    "planned_failures_in_denominator": True, "further_bump_allowed": False, "test_files_read": False}
                evaluation["protocol_hash"] = engine.canonical_hash(evaluation); data["evaluation_protocol"] = evaluation
            else: data["roles"] = [ref(p) for p in role_paths]
            config = {"schema_version": operator.CONFIG_SCHEMA, "kind": job["kind"], "scope": "VALIDATION_ONLY_NATIVE_FULL_DEV" if gpu else "VALIDATION_ONLY_FULL_DEV_TERMINAL_AUDIT",
                **{k: job[k] for k in ("job_id", "run_id", "cell_id", "phase", "seed", "uses_gpu", "source_files", "source_hash", "success_receipt")},
                "state_path": str(f.state_path), "output_dir": str(f.root / "full-dev-results" / job_id), "resource_policy": initial["resource_policy"],
                "data": data, "data_hash": engine.canonical_hash(data), "model_hash": engine.canonical_hash(initial["models"])}
            if gpu:
                old = self.old_configs[f"v6-caps-initial-{cell}-{role}"]
                config.update(**{k: job[k] for k in ("model_id", "task", "family", "role")}, model=old["model"], model_hash=old["model_hash"], decode=old["decode"],
                    device="cuda", eval_context=4096, fixed_template_date="2026-11-14", local_files_only=True, trust_remote_code=False,
                    gpu_ledger_path=str(f.data / "runtime/gpu_budget.json"), supervisor_lock_paths=[str(f.old / "worker.lock"), str(f.data / "runtime/medvlm_gpu.lock")],
                    forge={**self.forge, "receipt_path": str(f.root / "receipts" / (job_id + ".forge-import.json"))},
                    split="validation", protocol_version="medvlm.v6.full-dev.v1", protocol_hash=evaluation["protocol_hash"], precision="nf4_double_quant_bfloat16")
            config["config_hash"] = engine.canonical_hash(config); write(path, config)
            job.update(**{k: config[k] for k in engine.HASH_FIELDS}); configs[job_id] = ref(path)
            if gpu: children.append(job)
            else: next(j for j in self.after["jobs"] if j["job_id"] == operator.GATE_ID).update({k: v for k, v in job.items() if k in operator.EXECUTION_FIELDS})
        position = next(i for i, j in enumerate(self.after["jobs"]) if j["job_id"] == operator.GATE_ID)
        self.after["jobs"][position:position] = children
        self.new_path = write(f.root / "manifests/full-dev-proposal.json", self.after)
        archive = {p.relative_to(self.new_engine).as_posix(): engine.file_hash(p) for p in sorted(self.new_engine.rglob("*")) if p.is_file()}
        sources = {k: v for k, v in archive.items() if k.startswith("src/") and k.endswith(".py")}
        self.binding = {"schema_version": operator.SCHEMA, "campaign_root": str(f.root), "data_root": str(f.data), "cap_operator_sha256": operator.CAP_OPERATOR_SHA256,
            "cap_binding_sha256": "c" * 64, "parent": {"manifest_path": str(self.parent_path), "manifest_sha256": engine.file_hash(self.parent_path), "manifest_hash": engine.canonical_hash(self.before)},
            "replacement": {"manifest_path": str(self.new_path), "manifest_sha256": engine.file_hash(self.new_path), "manifest_hash": engine.canonical_hash(self.after)},
            "parent_state_sha256": engine.file_hash(f.state_path), "engine": {"root": str(self.new_engine), "commit": self.new_engine.name, "archive_files": archive,
                "archive_tree_hash": base.digest(archive), "source_files": sources, "source_tree_hash": base.digest(sources)}, "configs": configs,
            "protocol": ref(self.protocol_path), "prerequisites": self.proofs, "forge": self.forge}
        self.binding_path = write(f.root / "operator/full-dev-binding.json", self.binding)
        self.control = FixtureControl(full_dev_binding=self.binding_path, full_dev_binding_sha256=engine.file_hash(self.binding_path),
            inherited=self.inherited, roots=f.roots, data_root=f.data, campaign_root=f.root, engine_root=f.base_engine, hold_receipt=f.hold)
        write(f.root / base.BINDING, self.control._binding())

    def executable(self, job_id, source, module, config):
        sources = {p.relative_to(source).as_posix(): engine.file_hash(p) for p in source.glob("src/mediflow_medvlm/**/*.py")}
        receipt = str(self.f.root / "receipts" / (job_id + ".json"))
        return {"job_id": job_id, "run_id": job_id, "implementation_pending": False, "dependencies": [], "uses_gpu": True,
            "gpu_lock_mode": "wrapper", "cwd": str(source), "source_files": sources, "source_hash": engine.canonical_hash(sources),
            "config_hash": "c" * 64, "data_hash": "d" * 64, "model_hash": "e" * 64, "success_receipt": receipt,
            "argv": [sys.executable, "-m", module, "--config", str(config), "--receipt", receipt]}

    def success(self, job, failure_count=0):
        proof = write(self.f.root / "proofs" / (job["job_id"] + ".json"), {"CPU_FIXTURE_ONLY": job["job_id"]})
        write(job["success_receipt"], {"status": "SUCCEEDED", "job_id": job["job_id"], "run_id": job["run_id"], **{k: job[k] for k in engine.HASH_FIELDS}, "artifacts": [ref(proof)]})
        record = {"status": "SUCCEEDED", "attempts": failure_count + 1, "failure_count": failure_count, "recovery_attempts": failure_count,
            "history": [{"status": "FAILED", "attempt": 1}] if failure_count else [], "exit_code": 0, "verification": engine.verify_success(job)}
        self.states[job["job_id"]] = record
        self.proofs[job["job_id"]] = {**ref(job["success_receipt"]), "job_hash": engine.canonical_hash(job), "state_hash": engine.canonical_hash(record)}
    def rebind(self, state=None):
        if state is not None:
            write(self.f.state_path, state)
            self.binding["parent_state_sha256"] = engine.file_hash(self.f.state_path)
        write(self.binding_path, self.binding)
        self.control.full_dev_sha256 = engine.file_hash(self.binding_path)


    def reject(self, operation=None, match=".*"):
        before = self.f.state_path.read_bytes()
        with patch.object(engine, "atomic_json", side_effect=AssertionError("rejected operation wrote")):
            with self.assertRaisesRegex((ValueError, OSError), match):
                (operation or (lambda: self.control.call("refine-full-dev", manifest=self.after)))()
        self.assertEqual(self.f.state_path.read_bytes(), before)


    def test_atomic_registration_preserves_all_old_records_and_activates_only_gate(self):
        f = self.f
        before = base.read_json(f.state_path)
        self.assertEqual(len(before["jobs"]), 63)
        writes, atomic = [], engine.atomic_json
        def observe(path, value):
            if Path(path) == f.state_path:
                self.assertEqual(base.read_json(value["manifest_path"]), self.after)
                self.assertEqual(base.read_json(path), before)
            writes.append(Path(path))
            atomic(path, value)
        with patch.object(engine, "atomic_json", side_effect=observe), patch.object(engine.subprocess, "Popen", side_effect=AssertionError("spawned")):
            result = self.control.call("refine-full-dev", manifest=self.after)
        self.assertEqual(writes, [Path(result["manifest_path"]), f.state_path])
        for job_id, record in before["jobs"].items():
            expected = copy.deepcopy(record)
            if job_id == operator.GATE_ID: expected["status"] = "PENDING"
            self.assertEqual(result["jobs"][job_id], expected)
        for job_id in operator.ADDED_IDS:
            self.assertEqual(result["jobs"][job_id], {"status": "PENDING", "attempts": 0, "failure_count": 0, "recovery_attempts": 0, "history": []})
        self.assertEqual(result["manifest_revisions"][:-1], before["manifest_revisions"])
        self.assertEqual(result["manifest_revisions"][-1]["added_jobs"], operator.ADDED_IDS)
        self.assertFalse((f.root / "supervisor.lock").exists())
        self.assertFalse((f.root / "worker.lock").exists())


    def test_external_byte_pins_are_exact(self):
        for path in (self.binding_path, self.new_path, self.parent_path, Path(self.binding["protocol"]["path"]),
            Path(self.binding["configs"][operator.ADDED_IDS[0]]["path"]), Path(self.binding["prerequisites"][operator.INITIAL_IDS[0]]["path"]), self.new_engine / "README.md"):
            raw = path.read_bytes(); path.write_bytes(raw + b" ")
            with self.subTest(path=path): self.reject()
            path.write_bytes(raw)


    def test_failed_prerequisite_changed_state_and_orphan_keys_rejected(self):
        original = base.read_json(self.f.state_path)
        for kind in ("orphan", "missing", "failure", "counter", "gate-attempt", "gate-history", "gate-bool", "owner"):
            state = copy.deepcopy(original)
            if kind == "orphan": state["jobs"][operator.ADDED_IDS[0]] = {"attempts": 7}
            elif kind == "missing": del state["jobs"][operator.PREREQUISITES[0]]
            elif kind == "failure": state["jobs"][operator.INITIAL_IDS[0]]["status"] = "FAILED"
            elif kind == "counter": state["jobs"][operator.PREREQUISITES[0]]["attempts"] += 1
            elif kind == "gate-attempt": state["jobs"][operator.GATE_ID]["attempts"] = 1
            elif kind == "gate-history": state["jobs"][operator.GATE_ID]["history"] = [{"prior": True}]
            elif kind == "gate-bool": state["jobs"][operator.GATE_ID]["failure_count"] = False
            else: state["queue_owner"] = "worker"
            self.rebind(state)
            with self.subTest(kind=kind): self.reject()
        self.rebind(original)
        state = copy.deepcopy(original); state["updated_at"] = "unpinned"
        write(self.f.state_path, state)
        self.reject(match="parent state bytes")


    def test_active_worker_orphan_identity_and_all_canonical_leases(self):
        original = base.read_json(self.f.state_path)
        for kind in ("status", "identity", "heartbeat"):
            state = copy.deepcopy(original); record = state["jobs"][operator.INITIAL_IDS[0]]
            if kind == "status": record["status"] = "STARTING"
            elif kind == "identity": record["worker_identity"] = engine.process_identity()
            else:
                record["attempt_dir"] = str(self.f.root / "attempts/orphan")
                write(Path(record["attempt_dir"]) / "heartbeat.json", {"child_identity": engine.process_identity()})
            self.rebind(state)
            with self.subTest(kind=kind): self.reject(match="worker active")
        self.rebind(original)
        for name in ("worker", "gpu"):
            with self.control._physical_lock(self.control.physical[name]): self.reject(match="lease busy")
        before = self.f.state_path.read_bytes()
        with self.control._physical_lock(self.control.physical["supervisor"]):
            self.assertEqual(self.control.call("refine-full-dev", manifest=self.after), {"status": "SUPERVISOR_BUSY"})
        self.assertEqual(self.f.state_path.read_bytes(), before)


    def test_idempotent_after_progress_never_rewrites(self):
        self.control.call("refine-full-dev", manifest=self.after)
        state = base.read_json(self.f.state_path)
        state["jobs"][operator.ADDED_IDS[0]].update(status="FAILED", attempts=3, failure_count=3, recovery_attempts=2)
        write(self.f.state_path, state)
        before = self.f.state_path.read_bytes()
        with patch.object(engine, "atomic_json", side_effect=AssertionError("rewrote")):
            self.control.call("refine-full-dev", manifest=self.after)
        self.assertEqual(self.f.state_path.read_bytes(), before)


    def test_atomic_pointer_interruption_keeps_parent_and_retries(self):
        before = self.f.state_path.read_bytes(); atomic = engine.atomic_json
        def interrupt(path, value):
            if Path(path) == self.f.state_path: raise OSError("CPU pointer interruption")
            atomic(path, value)
        with patch.object(engine, "atomic_json", side_effect=interrupt), self.assertRaises(OSError):
            self.control.call("refine-full-dev", manifest=self.after)
        self.assertEqual(self.f.state_path.read_bytes(), before)
        self.control.call("refine-full-dev", manifest=self.after)


    def test_inherited_hold_and_lock_inodes_and_namespace_aliases(self):
        f = self.f
        hold = base.read_json(f.hold); hold["dispatch_control"]["tick_prohibited"] = False
        write(f.hold, hold); self.reject(match="MIGRATION_HOLD")
        hold["dispatch_control"]["tick_prohibited"] = True; write(f.hold, hold)
        for name in ("worker.lock", "supervisor.lock"):
            path = f.root / name; path.touch(); self.reject(match="namespace lock files"); path.unlink()
        path = self.control.physical["gpu"]; new = path.with_name("new.lock"); new.touch(); os.replace(new, path)
        self.reject(match="physical lock inode")


    def test_only_added_registration_and_native_runtime_identity(self):
        cls = type(self.control.supervisor)
        self.assertEqual(engine.file_hash(engine.__file__), "66f71b17a4b82f4897201e2a71d8356677381543b3e75dd06b008711b1bdbd22")
        self.assertEqual({k for k, v in cls.__dict__.items() if callable(v)}, {"refine_full_dev"})
        for name in ("tick", "status", "load", "authorize_recovery", "_failure"):
            self.assertIs(getattr(cls, name), getattr(engine.Supervisor, name))
        self.assertTrue(hasattr(cls, "refine_caps"))

    def test_exact_matrix_old_metadata_and_gate_declarations(self):
        for kind in ("metadata", "old-train", "old-initial", "old-bumped", "cap", "gate-reason", "gate-phase", "gate-status", "missing", "extra", "order", "seed", "bool-seed", "role", "kind", "round", "deps", "unknown", "gpu", "argv"):
            value = copy.deepcopy(self.after); jobs = {j["job_id"]: j for j in value["jobs"]}
            child, gate = jobs[operator.ADDED_IDS[0]], jobs[operator.GATE_ID]
            if kind == "metadata": value["declared_seed"] += 1
            elif kind == "old-train": jobs["v6-first-seed-T5"]["config_hash"] = "f" * 64
            elif kind == "old-initial": jobs[operator.INITIAL_IDS[0]]["role"] = "tuned"
            elif kind == "old-bumped": jobs[operator.BUMPED_IDS[0]]["source_hash"] = "a" * 64
            elif kind == "cap": jobs[operator.CAPS_ID]["uses_gpu"] = True
            elif kind == "gate-reason": gate["reason"] = "fake"
            elif kind == "gate-phase": gate["phase"] = "family_selection"
            elif kind == "gate-status": gate["status"] = "SUCCEEDED"
            elif kind == "missing": value["jobs"].remove(child)
            elif kind == "extra": value["jobs"].append({**child, "job_id": "unreviewed", "run_id": "unreviewed"})
            elif kind == "order":
                i = value["jobs"].index(child); value["jobs"][i], value["jobs"][i + 1] = value["jobs"][i + 1], value["jobs"][i]
            elif kind == "seed": child["seed"] += 1
            elif kind == "bool-seed": child["seed"] = True
            elif kind == "role": child["role"] = "tuned"
            elif kind == "kind": child["kind"] = "aggregate"
            elif kind == "round": child["calibration_round"] = "bumped"
            elif kind == "deps": child["dependencies"] = [operator.ADDED_IDS[1]]
            elif kind == "unknown": child["hidden_queue"] = True
            elif kind == "gpu": child["uses_gpu"] = False
            else: child["argv"][2] = "another.module"
            with self.subTest(kind=kind), self.assertRaises(ValueError): operator.validate_pair(engine, self.before, value)

    def test_failed_full_dev_role_allows_independent_sibling(self):
        self.control.call("refine-full-dev", manifest=self.after)
        state = base.read_json(self.f.state_path)
        state["jobs"][operator.ADDED_IDS[0]].update(status="FAILED", attempts=3, failure_count=3, recovery_attempts=2)
        write(self.f.state_path, state)
        with patch.object(engine.subprocess, "Popen") as popen:
            popen.return_value.pid = 999999999; result = self.control.call("tick")
        self.assertEqual(popen.call_count, 1)
        self.assertEqual(result["jobs"][operator.ADDED_IDS[1]]["attempts"], 1)
        self.assertEqual(result["jobs"][operator.GATE_ID]["attempts"], 0)
        for job in self.before["jobs"]:
            if job["job_id"] != operator.GATE_ID: self.assertEqual(result["jobs"][job["job_id"]], state["jobs"][job["job_id"]])

    def test_terminal_failed_roles_allow_cpu_aggregate_only(self):
        self.control.call("refine-full-dev", manifest=self.after)
        state = base.read_json(self.f.state_path)
        for job_id in operator.ADDED_IDS: state["jobs"][job_id].update(status="FAILED", attempts=3, failure_count=3, recovery_attempts=2)
        write(self.f.state_path, state)
        with patch.object(engine.subprocess, "Popen") as popen:
            popen.return_value.pid = 999999999; result = self.control.call("tick")
        self.assertEqual(popen.call_count, 1)
        self.assertEqual(result["jobs"][operator.GATE_ID]["attempts"], 1)
        self.assertTrue(all(result["jobs"][f"fixture-future-{i}"]["attempts"] == 0 for i in range(1, 10)))

    def test_native_same_source_recovery_and_scope_limits(self):
        self.control.call("refine-full-dev", manifest=self.after)
        state = base.read_json(self.f.state_path)
        job_id = operator.ADDED_IDS[0]; job = next(j for j in self.after["jobs"] if j["job_id"] == job_id)
        record = state["jobs"][job_id]
        record.update(status="ANALYSIS_REQUIRED", attempts=2, failure_count=2, recovery_attempts=1,
            error_signature="CPU_FIXTURE", history=[{"status": "FAILED", "attempt": 1}, {"status": "FAILED", "attempt": 2}])
        write(self.f.state_path, state)
        evidence = write(self.f.root / "operator/CPU-evidence.json", {"CPU_FIXTURE_ONLY": True, "actual_model_execution": False})
        receipt = write(self.f.root / "operator/CPU-analysis.json", {"job_id": job_id, "run_id": job_id,
            "job_hash": engine.canonical_hash(job), "failure_count": 2, "error_signature": "CPU_FIXTURE", "actual_model": "gpt-6-astra",
            "reasoning_effort": "high", "original_model": "CPU_FIXTURE", "original_reasoning_effort": "CPU_FIXTURE",
            "escalation_reason": "Synthetic native contract fixture", "retry_authorized": True,
            "analysis_summary": "CPU fixture only; no actual model execution.", "model_execution_receipt": ref(evidence)})
        result = self.control.call("authorize-recovery", job_id=job_id, receipt=receipt)
        changed = result["jobs"][job_id]; self.assertEqual(changed["status"], "PENDING")
        for key in ("attempts", "failure_count", "recovery_attempts"): self.assertEqual(changed[key], record[key])
        self.assertEqual(changed["history"][:2], record["history"])
        for key in operator.PREREQUISITES: self.assertEqual(result["jobs"][key], state["jobs"][key])
        self.assertEqual(result["jobs"]["v6-first-seed-T5"]["attempts"], 3)
        for bad, manifest in ((operator.CAPS_ID, None), (operator.INITIAL_IDS[0], None), (operator.BUMPED_IDS[0], None), ("v6-first-seed-T5", None), ("fixture-future-1", None), (job_id, self.after)):
            with self.subTest(job=bad), self.assertRaisesRegex(ValueError, "same-source recovery"):
                self.control.call("authorize-recovery", job_id=bad, manifest=manifest, receipt=receipt)
        self.reject(lambda: self.control.call("authorize-recovery", job_id=operator.ADDED_IDS[1], receipt=receipt), "second-failure analysis hold")

    def test_facade_denies_other_operations_and_unpinned_manifest(self):
        for command in ("refine-caps", "refine-validation", "activate", "refine-first-seed", "prepare", "init", "finalize"):
            with self.subTest(command=command), self.assertRaises(ValueError): self.control.call(command)
        value = copy.deepcopy(self.after); value["scientific_metadata"] = {"changed": True}
        self.reject(lambda: self.control.call("refine-full-dev", manifest=value))

    def test_self_hashed_scientific_overrides_and_protected_paths_are_rejected(self):
        paths = [self.binding_path, self.new_path, *[Path(r["path"]) for r in self.binding["configs"].values()]]
        originals = {p: p.read_bytes() for p in paths}
        original_binding = copy.deepcopy(self.binding)
        kinds = ("unknown", "scope", "seed", "state-path", "policy", "cap-state", "cap-job", "cap-protocol", "split", "eval-protocol", "ids", "count", "model", "decode", "precision", "scorer", "aggregate-matrix", "protected-output", "old-output", "new-output", "old-receipt", "receipt-in-output", "forge-path", "forge-root")
        for kind in kinds:
            self.binding = copy.deepcopy(original_binding); candidate = copy.deepcopy(self.after)
            job_id = operator.GATE_ID if kind == "aggregate-matrix" else operator.ADDED_IDS[0]
            path = Path(self.binding["configs"][job_id]["path"]); config = base.read_json(path)
            if kind == "unknown": config["max_time"] = 30
            elif kind == "scope": config["scope"] = "TEST_EVALUATION"
            elif kind == "seed": config["seed"] = True
            elif kind == "state-path": config["state_path"] = str(self.f.old / "state.json")
            elif kind == "policy": config["resource_policy"] = {"total_gpu_hours": 1}
            elif kind == "cap-state": config["data"]["cap_freeze"]["terminal_state"]["attempts"] += 1
            elif kind == "cap-job": config["data"]["cap_freeze"]["job"]["uses_gpu"] = True
            elif kind == "cap-protocol": config["data"]["cap_freeze"]["final_protocol"]["sha256"] = "a" * 64
            elif kind == "split": config["split"] = "test"; config["data"]["partition"] = "test"
            elif kind == "eval-protocol": config["data"]["evaluation_protocol"]["planned_failures_in_denominator"] = False
            elif kind == "ids": config["data"]["ordered_sample_ids"].reverse()
            elif kind == "count": config["data"]["count"] -= 1
            elif kind == "model": config["model"] = {"unreviewed": True}
            elif kind == "decode": config["decode"]["max_time"] = 1
            elif kind == "precision": config["precision"] = "float32"
            elif kind == "scorer":
                del config["source_files"]["src/mediflow_medvlm/evaluation/metrics.py"]
                config["source_hash"] = engine.canonical_hash(config["source_files"])
            elif kind == "aggregate-matrix": config["data"]["roles"] = config["data"]["roles"][:-1]
            elif kind == "protected-output": config["output_dir"] = str(self.f.root / "configs")
            elif kind == "old-output": config["output_dir"] = self.old_configs[operator.CAPS_ID]["output_dir"]
            elif kind == "new-output": config["output_dir"] = base.read_json(self.binding["configs"][operator.ADDED_IDS[1]]["path"])["output_dir"]
            elif kind == "old-receipt": config["success_receipt"] = self.before["jobs"][0]["success_receipt"]
            elif kind == "receipt-in-output": config["success_receipt"] = str(Path(config["output_dir"]) / "claimed-success.json")
            elif kind == "forge-path": config["forge"]["receipt_path"] = config["success_receipt"]
            else: self.binding["forge"]["output_root"] = str(self.f.data.parent)
            config["data_hash"] = engine.canonical_hash(config["data"])
            config["config_hash"] = engine.canonical_hash({k: v for k, v in config.items() if k != "config_hash"})
            write(path, config); job = next(j for j in candidate["jobs"] if j["job_id"] == job_id)
            job.update(**{k: config[k] for k in engine.HASH_FIELDS}); job["source_files"] = config["source_files"]
            job["success_receipt"] = config["success_receipt"]; job["argv"][6] = config["success_receipt"]
            self.binding["configs"][job_id] = ref(path)
            if job_id != operator.GATE_ID:
                aggregate_path = Path(self.binding["configs"][operator.GATE_ID]["path"])
                aggregate = base.read_json(aggregate_path)
                aggregate["data"]["roles"][0] = ref(path)
                aggregate["data_hash"] = engine.canonical_hash(aggregate["data"])
                aggregate["config_hash"] = engine.canonical_hash({k: v for k, v in aggregate.items() if k != "config_hash"})
                write(aggregate_path, aggregate)
                next(j for j in candidate["jobs"] if j["job_id"] == operator.GATE_ID).update(**{k: aggregate[k] for k in engine.HASH_FIELDS})
                self.binding["configs"][operator.GATE_ID] = ref(aggregate_path)
            write(self.new_path, candidate)
            self.binding["replacement"].update(manifest_sha256=engine.file_hash(self.new_path), manifest_hash=engine.canonical_hash(candidate))
            self.rebind()
            expected_error = {"receipt-in-output": "must not overlap any new result output", "new-output": "distinct and nonoverlapping",
                "old-output": "overlap immutable old results", "forge-root": "disjoint external Forge"}.get(kind, ".*")
            with self.subTest(kind=kind): self.reject(lambda: self.control.call("refine-full-dev", manifest=candidate), expected_error)
            for old, raw in originals.items(): old.write_bytes(raw)
            self.binding = copy.deepcopy(original_binding); self.control.full_dev_sha256 = engine.file_hash(self.binding_path)

    def test_cli_forwards_recovery_and_refinement_arguments(self):
        flags = []
        for name in ("cap-operator", "validation-operator", "stage-operator", "recovery-operator", "base-operator", "data-root", "campaign-root", "engine-root", "hold-receipt", "recovery-binding", "stage-binding", "validation-binding", "cap-binding", "full-dev-binding"):
            flags.extend(["--" + name, str(self.f.root / name)])
        for name in ("recovery-binding-sha256", "stage-binding-sha256", "validation-binding-sha256", "cap-binding-sha256", "full-dev-binding-sha256"):
            flags.extend(["--" + name, "d" * 64])
        receipt = self.f.root / "analysis.json"
        with patch.object(operator, "make_control") as make, redirect_stdout(StringIO()):
            make.return_value.call.return_value = {}
            operator.main(flags + ["authorize-recovery", "--job-id", operator.ADDED_IDS[0], "--receipt", str(receipt)])
            self.assertNotIn("job_id", make.call_args.kwargs); self.assertNotIn("receipt", make.call_args.kwargs)
            make.return_value.call.assert_called_once_with("authorize-recovery", manifest=None, job_id=operator.ADDED_IDS[0], receipt=receipt)
        with patch.object(operator, "make_control") as make, redirect_stdout(StringIO()):
            make.return_value.base.read_json.return_value = self.after; make.return_value.call.return_value = {}
            operator.main(flags + ["refine-full-dev", "--manifest", str(self.new_path)])
            make.return_value.call.assert_called_once_with("refine-full-dev", manifest=self.after, job_id=None, receipt=None)

    def test_prior_facade_bytes_and_cold_cpu_import(self):
        loaded = operator.load_caps(HERE / "campaign_control_caps.py")
        validation = loaded.load_validation(HERE / "campaign_control_validation.py")
        stage = validation.load_stage(HERE / "campaign_control_stages.py")
        self.assertTrue(hasattr(stage, "StageMixin"))
        bad = self.f.root / "changed-caps.py"; bad.write_bytes((HERE / "campaign_control_caps.py").read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "operator changed"): operator.load_caps(bad)
        script = ("import importlib.util,sys; "
            f"s=importlib.util.spec_from_file_location('cold_full_dev', {str(HERE / 'campaign_control_full_dev.py')!r}); "
            "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);assert 'torch' not in sys.modules;assert 'transformers' not in sys.modules")
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"})
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__": unittest.main()
