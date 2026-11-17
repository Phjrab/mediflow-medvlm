#!/usr/bin/env python3
"""Pinned cap decision, bumped calibration and final cap-gate registration.

The existing sole supervisor and physical leases remain authoritative. This
facade adds one finite registration method; execution and recovery stay native.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
VALIDATION_OPERATOR_SHA256 = "18343920be4a601106bcf7deebb27e78c8aa5bc2b1c35ef1332e3a571494de53"
SCHEMA = "mediflow.recorded-cap-refinement.v1"
CONFIG_SCHEMA = "medvlm.v6.validation-cap-refinement-config.v1"
WORKER_MODULE = "mediflow_medvlm.campaign_validation_cap_refinement"
CAPS_ID = "v6-validation_caps"
FIRST_SEED_ID = "v6-first_seed_final"
DECISION_ID = "v6-caps-decision"
CELL_IDS = ["T1", "T2", "T3", "T4", "T5", "T6", "V1", "V2", "V3", "V4"]
ROLES = ("base", "tuned")
INITIAL_IDS = [f"v6-caps-initial-{cell}-{role}" for cell in CELL_IDS for role in ROLES]
BUMPED_IDS = [f"v6-caps-bumped-{cell}-{role}" for cell in CELL_IDS for role in ROLES]
ADDED_IDS = [DECISION_ID, *BUMPED_IDS]
EXECUTABLE_IDS = [*ADDED_IDS, CAPS_ID]
PREREQUISITES = ["v6-first-seed-" + cell for cell in CELL_IDS] + [FIRST_SEED_ID, *INITIAL_IDS]
EVENT = "RECORDED_VALIDATION_CAP_REFINEMENT"
EXECUTION_FIELDS = {"dependencies", "dependency_policy", "implementation_pending", "uses_gpu", "requires_gpu",
    "gpu_lock_mode", "argv", "cwd", "source_files", "source_hash", "config_hash", "data_hash", "model_hash", "success_receipt"}
JOB_FIELDS = {"job_id", "run_id", "phase", "cell_id", "model_id", "task", "family", "role", "calibration_round",
    "seed", "kind", *EXECUTION_FIELDS}
DECISION_JOB_FIELDS = {"job_id", "run_id", "phase", "cell_id", "kind", "seed", *EXECUTION_FIELDS}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_validation(path):
    path = Path(path)
    require(path.is_absolute() and ".." not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)),
        "absolute nonsymlink validation operator required")
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == VALIDATION_OPERATOR_SHA256, "original validation operator changed")
    spec = importlib.util.spec_from_loader("_pinned_validation_for_caps", loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def validate_pair(engine, parent, replacement):
    engine.validate_manifest(parent)
    engine.validate_manifest(replacement)
    same = lambda a, b: engine.canonical_hash(a) == engine.canonical_hash(b)
    require(same({k: v for k, v in parent.items() if k != "jobs"},
        {k: v for k, v in replacement.items() if k != "jobs"}), "campaign metadata must remain immutable")
    before = {j["job_id"]: j for j in parent["jobs"]}
    after = {j["job_id"]: j for j in replacement["jobs"]}
    require(set(PREREQUISITES + [CAPS_ID]).issubset(before) and not set(ADDED_IDS).intersection(before),
        "exact initial calibration parent required")
    order = [j["job_id"] for j in parent["jobs"]]
    position = order.index(CAPS_ID)
    order[position:position] = ADDED_IDS
    require([j["job_id"] for j in replacement["jobs"]] == order, "exact decision/twenty bumped order and old order required")
    gate = before[CAPS_ID]
    require(gate.get("phase") == "validation_caps" and gate.get("run_id") == CAPS_ID
        and gate.get("cell_id") == "ALL_V6" and gate.get("implementation_pending") is True
        and not gate.get("argv") and gate.get("dependencies") == [FIRST_SEED_ID, *INITIAL_IDS]
        and gate.get("dependency_policy") == "terminal", "untouched original cap gate required")
    for job_id, job in before.items():
        if job_id == CAPS_ID:
            require(same({k: v for k, v in job.items() if k not in EXECUTION_FIELDS},
                {k: v for k, v in after[job_id].items() if k not in EXECUTION_FIELDS}),
                "cap gate original scientific declaration must remain unchanged")
        else:
            require(same(after[job_id], job), "every existing started or unrelated job must remain identical")
    for job_id in EXECUTABLE_IDS:
        job = after[job_id]
        argv = job.get("argv", [])
        gpu = job_id in BUMPED_IDS
        require(job.get("run_id") == job_id and job.get("implementation_pending") is False
            and job.get("uses_gpu") is gpu and job.get("requires_gpu") is gpu
            and job.get("gpu_lock_mode") == "wrapper"
            and len(argv) == 7 and argv[1:4] == ["-m", WORKER_MODULE, "--config"]
            and argv[5:] == ["--receipt", job.get("success_receipt")], "invalid cap refinement executable wrapper")
        if job_id == CAPS_ID:
            require(job.get("dependencies") == [DECISION_ID, *BUMPED_IDS]
                and job.get("dependency_policy") == "terminal", "final gate must audit decision and all bumped terminal results")
        elif job_id == DECISION_ID:
            require(set(job) == DECISION_JOB_FIELDS and job.get("kind") == "decision"
                and type(job.get("seed")) is int and job["seed"] == 20261023 and job.get("phase") == "validation_cap_decision"
                and job.get("cell_id") == "ALL_V6" and job.get("dependencies") == INITIAL_IDS
                and job.get("dependency_policy") == "terminal", "exact CPU decision job required")
        else:
            cell, role = job_id.removeprefix("v6-caps-bumped-").split("-")
            original = before[f"v6-caps-initial-{cell}-{role}"]
            require(set(job) == JOB_FIELDS and job.get("kind") == "bumped" and job.get("phase") == "validation_caps_bumped"
                and all(same(job.get(k), original.get(k)) for k in ("cell_id", "model_id", "task", "family", "role", "seed"))
                and type(job.get("seed")) is int and job["seed"] == 20261023
                and job.get("calibration_round") == "bumped" and job.get("dependencies") == [DECISION_ID]
                and job.get("dependency_policy") == "success", "invalid independent bumped role declaration")
    return [after[job_id] for job_id in EXECUTABLE_IDS]


def make_supervisor_class(engine, owner, inherited):
    class CapRefinementSupervisor(inherited):
        def refine_caps(self, revised):
            with engine.try_lock(self.root / "supervisor.lock") as lock:
                if lock is None:
                    return {"status": "SUPERVISOR_BUSY"}
                binding, parent, replacement = owner._caps()
                require(engine.canonical_hash(revised) == engine.canonical_hash(replacement), "exact pinned cap replacement required")
                current, state = self.load()
                require(state.get("queue_owner") == "supervisor", "sole Supervisor ownership required")
                digest = engine.canonical_hash(replacement)
                if state["manifest_hash"] == digest:
                    events = [e for e in state.get("manifest_revisions", []) if e.get("kind") == EVENT
                        and e.get("manifest_hash") == digest and e.get("cap_binding_sha256") == owner.cap_sha256
                        and e.get("added_jobs") == ADDED_IDS and e.get("activated_jobs") == [CAPS_ID]]
                    require(len(events) == 1 and engine.canonical_hash(current) == digest, "exact immutable cap refinement audit required")
                    return self.status()
                require(engine.canonical_hash(current) == binding["parent"]["manifest_hash"]
                    and engine.canonical_hash(current) == engine.canonical_hash(parent), "cap refinement parent is not active")
                require(set(state["jobs"]) == {j["job_id"] for j in parent["jobs"]}, "exact original state/job keys required")
                require(owner.base.file_hash(self.state_path) == binding["parent_state_sha256"], "parent state bytes changed")
                cap = state["jobs"][CAPS_ID]
                require(cap.get("status") == "IMPLEMENTATION_PENDING"
                    and all(type(cap.get(k)) is int and cap[k] == 0 for k in ("attempts", "failure_count", "recovery_attempts"))
                    and not cap.get("history") and all(cap.get(k) is None for k in ("attempt_dir", "launch_token",
                        "worker_identity", "verification", "recovery_authorization_hash", "recovery_authorization")),
                    "original cap gate must be untouched and unstarted")
                for record in state["jobs"].values():
                    health = {}
                    if record.get("attempt_dir"):
                        path = owner.base.absolute(record["attempt_dir"]) / "heartbeat.json"
                        if path.exists():
                            health = owner.base.read_json(path)
                    require(record.get("status") not in {"STARTING", "RUNNING"}
                        and not any(engine.same_process(i) for i in (record.get("worker_identity"),
                            health.get("worker_identity"), health.get("child_identity"))), "worker active; cap refinement is registration only")
                with ExitStack() as leases:
                    for path in (self.root / "worker.lock", self.data_root / "runtime/medvlm_gpu.lock"):
                        require(leases.enter_context(engine.try_lock(path)) is not None, "worker/GPU lease busy")
                    jobs = {j["job_id"]: j for j in parent["jobs"]}
                    proofs = {}
                    for job_id in PREREQUISITES:
                        record, pins = state["jobs"][job_id], binding["prerequisites"][job_id]
                        require(record.get("status") == "SUCCEEDED" and record.get("exit_code") == 0
                            and type(record.get("attempts")) is int and record["attempts"] > 0
                            and engine.canonical_hash(record) == pins["state_hash"], "all thirty-one prerequisites must be verified successes")
                        proof = engine.verify_success(jobs[job_id])
                        verification = record.get("verification", {})
                        require(verification.get("receipt") == pins["path"] and verification.get("receipt_sha256") == pins["sha256"]
                            and verification.get("artifact_count") == proof["artifact_count"], "original success verification changed")
                        proofs[job_id] = proof
                    owner._verify()
                    require(owner.base.file_hash(self.state_path) == binding["parent_state_sha256"], "parent state changed during receipt audit")
                    version = self.root / "manifests" / (digest + ".json")
                    if version.exists():
                        require(engine.canonical_hash(owner.base.read_json(version)) == digest, "immutable manifest version changed")
                    else:
                        engine.atomic_json(version, replacement)
                    state.setdefault("manifest_revisions", []).append({"kind": EVENT,
                        "previous_hash": state["manifest_hash"], "previous_manifest_path": state["manifest_path"],
                        "manifest_hash": digest, "manifest_path": str(version), "added_jobs": ADDED_IDS,
                        "activated_jobs": [CAPS_ID], "cap_binding": str(owner.cap_path), "cap_binding_sha256": owner.cap_sha256,
                        "protocol": binding["protocol"], "prerequisite_proofs": proofs, "at": engine.now()})
                    for job_id in ADDED_IDS:
                        state["jobs"][job_id] = {"status": "PENDING", "attempts": 0, "failure_count": 0, "recovery_attempts": 0, "history": []}
                    cap["status"] = "PENDING"
                    state.update(manifest_hash=digest, manifest_path=str(version), updated_at=engine.now())
                    engine.atomic_json(self.state_path, state)
                return self.status()
    return CapRefinementSupervisor


class CapRefinementMixin:
    def __init__(self, *, cap_binding, cap_binding_sha256, **kwargs):
        require(isinstance(cap_binding_sha256, str) and re.fullmatch(r"[0-9a-f]{64}", cap_binding_sha256), "external cap binding SHA256 required")
        self.cap_path, self.cap_sha256 = Path(cap_binding), cap_binding_sha256
        super().__init__(**kwargs)
        self.cap_path = self.base.absolute(self.cap_path)
        self.supervisor.__class__ = make_supervisor_class(self.engine, self, type(self.supervisor))

    def _caps(self):
        raw = self.base.read_bytes(self.cap_path)
        require(hashlib.sha256(raw).hexdigest() == self.cap_sha256, "reviewed cap binding changed")
        binding = json.loads(raw)
        require(binding.get("schema_version") == SCHEMA and binding.get("campaign_root") == str(self.root)
            and binding.get("data_root") == str(self.data_root)
            and binding.get("validation_operator_sha256") == VALIDATION_OPERATOR_SHA256
            and binding.get("validation_binding_sha256") == self.validation_sha256, "cap namespace or inherited owner bindings changed")
        inherited, _, validation_parent = self._validation()
        require(binding["parent"]["manifest_hash"] == inherited["replacement"]["manifest_hash"], "caps must extend exact initial calibration manifest")
        manifests = []
        for pins in (binding["parent"], binding["replacement"]):
            path = self.base.absolute(pins["manifest_path"])
            require(path.is_relative_to(self.root / "manifests") and self.base.file_hash(path) == pins["manifest_sha256"], "cap manifest path or bytes changed")
            value = self.base.read_json(path)
            require(self.engine.canonical_hash(value) == pins["manifest_hash"], "cap manifest identity changed")
            manifests.append(value)
        require(self.engine.canonical_hash(manifests[0]) == self.engine.canonical_hash(validation_parent), "cap parent differs from inherited manifest")
        jobs = validate_pair(self.engine, *manifests)
        pins = binding["engine"]
        source = self.base.absolute(pins["root"])
        require(source.parent == self.root / "snapshots" and source.name == pins["commit"]
            and re.fullmatch(r"[0-9a-f]{40}", pins["commit"]), "new immutable source commit archive required")
        archive = {}
        for path in sorted(source.rglob("*")):
            require(not path.is_symlink(), "source archive symlinks forbidden")
            if not path.is_dir():
                archive[path.relative_to(source).as_posix()] = self.base.file_hash(path)
        require(bool(archive) and archive == pins["archive_files"] and self.base.digest(archive) == pins["archive_tree_hash"], "complete source archive changed")
        sources = {k: v for k, v in archive.items() if k.startswith("src/") and k.endswith(".py")}
        require(sources == pins["source_files"] and self.base.digest(sources) == pins["source_tree_hash"]
            and "src/mediflow_medvlm/campaign_validation_cap_refinement.py" in sources, "complete cap refinement source changed")
        require(set(binding["configs"]) == set(EXECUTABLE_IDS) and set(binding["prerequisites"]) == set(PREREQUISITES),
            "exact twenty-two configs and thirty-one prerequisites required")
        require(binding["protocol"] == inherited["protocol"], "original immutable CPU protocol reference required")
        protocol = self.base.read_json(self.base.absolute(binding["protocol"]["path"]))
        parents = {j["job_id"]: j for j in manifests[0]["jobs"]}
        for job_id, ref in binding["prerequisites"].items():
            path = self.base.absolute(ref["path"])
            require(path.is_relative_to(self.root) and str(path) == parents[job_id]["success_receipt"]
                and self.base.file_hash(path) == ref["sha256"] and self.engine.canonical_hash(parents[job_id]) == ref["job_hash"],
                "original prerequisite receipt or job pins changed")
        receipts = [job["success_receipt"] for job in jobs]
        require(len(set(receipts)) == len(receipts)
            and not set(receipts).intersection(j.get("success_receipt") for j in parents.values()),
            "new success receipts must be distinct from every original receipt")
        old_outputs = []
        for old in parents.values():
            if old.get("argv") and not old.get("implementation_pending"):
                old_config = self.base.read_json(self.base.absolute(old["argv"][4]))
                if old_config.get("output_dir"):
                    old_outputs.append(self.base.absolute(old_config["output_dir"]))
        outputs = []
        for job in jobs:
            require(job["cwd"] == str(source), "new cap executable cwd differs from pinned archive")
            self.engine.verify_source(job)
            reference = binding["configs"][job["job_id"]]
            path = self.base.absolute(job["argv"][4])
            require(path.is_relative_to(self.root / "configs") and str(path) == reference["path"]
                and self.base.file_hash(path) == reference["sha256"], "reviewed cap config bytes or path changed")
            config = self.base.read_json(path)
            require(config.get("schema_version") == CONFIG_SCHEMA
                and config.get("config_hash") == self.base.digest({k: v for k, v in config.items() if k != "config_hash"})
                and all(self.base.digest(config.get(k)) == self.base.digest(job.get(k)) for k in
                    ("job_id", "run_id", "cell_id", "source_files", "source_hash", "config_hash", "data_hash", "model_hash", "success_receipt"))
                and all(k in sources and sources[k] == sha for k, sha in config["source_files"].items()),
                "cap config identity/source differs from pinned job")
            outputs.append(self.base.absolute(config["output_dir"]))
            self._config_policy(config, job, binding, protocol)
        require(len(set(outputs)) == len(outputs) and all(not a.is_relative_to(b)
            for i, a in enumerate(outputs) for j, b in enumerate(outputs) if i != j),
            "new outputs must be distinct and nonoverlapping")
        require(all(not a.is_relative_to(b) and not b.is_relative_to(a) for a in outputs for b in old_outputs)
            and all(not self.base.absolute(r).is_relative_to(b) for r in receipts for b in old_outputs),
            "new outputs or receipts overlap original immutable results")
        return binding, manifests[0], manifests[1]

    def _config_policy(self, config, job, binding, protocol):
        """Verify frozen worker policy without importing a candidate engine package."""
        digest = self.base.digest
        same = lambda a, b: digest(a) == digest(b)
        kind = "decision" if job["job_id"] == DECISION_ID else "freeze" if job["job_id"] == CAPS_ID else "bumped"
        scopes = {"decision": "VALIDATION_CAP_DECISION_ONLY", "bumped": "VALIDATION_ONLY_NATIVE_ONCE_BUMPED_CAP_RERUN",
            "freeze": "VALIDATION_CAP_PROTOCOL_FREEZE_ONLY"}
        common = {"schema_version", "scope", "kind", "phase", "job_id", "run_id", "cell_id", "seed", "uses_gpu", "state_path",
            "output_dir", "success_receipt", "data", "resource_policy", "source_files", "source_hash", "data_hash", "model_hash", "config_hash"}
        gpu_fields = {"model_id", "task", "family", "role", "calibration_round", "device", "eval_context", "fixed_template_date",
            "model", "decode", "local_files_only", "trust_remote_code", "gpu_ledger_path", "supervisor_lock_paths"}
        require(set(config) == common | (gpu_fields if kind == "bumped" else set())
            and config.get("kind") == kind and config.get("scope") == scopes[kind]
            and config.get("phase") == job["phase"] and type(config.get("seed")) is int and config["seed"] == 20261023
            and config.get("uses_gpu") is (kind == "bumped") and config.get("state_path") == str(self.supervisor.state_path)
            and same(config.get("resource_policy"), protocol.get("resource_policy")), "cap config keyset, scope, seed or runtime policy changed")
        output, receipt = self.base.absolute(config["output_dir"]), self.base.absolute(config["success_receipt"])
        protected = [self.root / name for name in ("operator", "snapshots", "configs", "manifests", "receipts", "attempts")]
        require(output != self.root and output.is_relative_to(self.root) and receipt.is_relative_to(self.root)
            and not any(output.is_relative_to(p) or p.is_relative_to(output) for p in protected),
            "cap output escaped namespace or overlaps protected owner files")
        data = config["data"]
        require(data.get("protocol") == binding["protocol"] and data.get("protocol_hash") == protocol["protocol_hash"]
            and config["data_hash"] == digest(data), "original protocol or cap data identity changed")
        decision_ref = binding["configs"][DECISION_ID]
        decision = self.base.read_json(self.base.absolute(decision_ref["path"]))
        expected = decision["data"]["expected_decision"]
        if kind == "decision":
            require(set(data) == {"protocol", "protocol_hash", "protocol_source_repo", "initial_proofs", "expected_decision"}
                and data["protocol_source_repo"] == self._validation()[0]["engine"]["root"]
                and list(data["initial_proofs"]) == INITIAL_IDS, "exact original initial proof matrix required")
            parent = self.base.read_json(self.base.absolute(binding["parent"]["manifest_path"]))
            parents = {j["job_id"]: j for j in parent["jobs"]}
            bumped, incomplete, all_counts, receipts = set(), [], {}, {}
            for key in INITIAL_IDS:
                proof, pins = data["initial_proofs"][key], binding["prerequisites"][key]
                require(set(proof) == {"job", "terminal_state", "config", "receipt", "task_counts", "technical_complete"}
                    and same(proof["job"], parents[key]) and digest(proof["terminal_state"]) == pins["state_hash"]
                    and proof["terminal_state"].get("status") == "SUCCEEDED"
                    and proof["receipt"] == {k: pins[k] for k in ("path", "sha256")}
                    and proof["config"]["path"] == parents[key]["argv"][4]
                    and self.base.file_hash(self.base.absolute(proof["config"]["path"])) == proof["config"]["sha256"],
                    "initial immutable job/state/config/receipt proof changed")
                original = self.base.read_json(self.base.absolute(proof["config"]["path"]))
                metrics = self.base.read_json(self.base.absolute(original["output_dir"]) / "metrics.json")
                original_receipt = self.base.read_json(self.base.absolute(pins["path"]))
                counts = proof["task_counts"]
                require(same(counts, metrics.get("task_counts")) and type(proof["technical_complete"]) is bool
                    and proof["technical_complete"] is original_receipt.get("technical_complete"), "captured initial task counts or technical completeness changed")
                axis = parents[key]["task"]
                scheduled = {}
                for group in protocol["calibration_strata"].values():
                    if group["axis"] == axis:
                        task = group["task_type"]
                        scheduled[task] = scheduled.get(task, 0) + len(group["ordered_sample_ids"])
                require(set(counts) == set(scheduled), "native task grouping changed")
                for task, count in counts.items():
                    require(set(count) == {"scheduled", "successful", "technical_failures", "length_terminated"}
                        and all(type(v) is int and v >= 0 for v in count.values())
                        and count["scheduled"] == scheduled[task]
                        and count["successful"] + count["technical_failures"] == scheduled[task]
                        and count["length_terminated"] <= scheduled[task], "initial calibration denominator changed")
                    if count["length_terminated"] * 100 > count["scheduled"]:
                        bumped.add(task)
                if not proof["technical_complete"]:
                    incomplete.append(key)
                all_counts[key], receipts[key] = counts, proof["receipt"]
            calculated = {"initial_protocol_hash": protocol["protocol_hash"], "caps": {task: protocol["maximum_caps"][task]
                if task in bumped else cap for task, cap in protocol["initial_caps"].items()},
                "bumped_task_types": sorted(bumped), "bump_rounds": 1 if bumped else 0,
                "universal_across_models_and_roles": True, "failed_initial_conditions": [],
                "technical_incomplete_initial_conditions": incomplete, "initial_task_counts": all_counts,
                "initial_receipts": receipts, "rule": "length_terminated*100 > scheduled; once only; EOS at boundary is stop",
                "test_files_read": False, "full_dev_completed": False, "family_selection_completed": False,
                "clinical_quality_certified": False, "campaign_terminal": False}
            require(same(expected, calculated) and config["model_hash"] == digest(protocol["models"]), "global once-only cap decision changed")
        else:
            require(data.get("decision_config") == decision_ref and data.get("decision_receipt") == decision["success_receipt"],
                "exact immutable decision config and receipt route required")
            if kind == "freeze":
                require(set(data) == {"protocol", "protocol_hash", "decision_config", "decision_receipt", "bumped_configs"}
                    and data["bumped_configs"] == [binding["configs"][key] for key in BUMPED_IDS]
                    and config["model_hash"] == digest(protocol["models"]), "exact twenty bumped configs required for final cap gate")
            else:
                key = f"v6-caps-initial-{job['cell_id']}-{job['role']}"
                original = self.base.read_json(self.base.absolute(decision["data"]["initial_proofs"][key]["config"]["path"]))
                ids = [sample_id for name, group in sorted(protocol["calibration_strata"].items())
                    if group["axis"] == job["task"] and group["task_type"] in expected["bumped_task_types"]
                    for sample_id in group["ordered_sample_ids"]]
                require(set(data) == {"protocol", "protocol_hash", "decision_config", "decision_receipt", "expected_decision", "partition",
                    "image_root", "ordered_sample_ids", "ordered_sample_ids_sha256"}
                    and same(data["expected_decision"], expected) and data["partition"] == "validation"
                    and data["image_root"] == original["data"]["image_root"]
                    and data["ordered_sample_ids"] == ids and data["ordered_sample_ids_sha256"] == digest(ids)
                    and all(same(config.get(k), original.get(k)) for k in ("model_id", "task", "family", "role", "seed", "model", "decode", "model_hash"))
                    and config["calibration_round"] == "bumped" and config["device"] == "cuda"
                    and config["eval_context"] == 4096 and config["fixed_template_date"] == "2026-11-14"
                    and config["local_files_only"] is True and config["trust_remote_code"] is False
                    and config["gpu_ledger_path"] == str(self.data_root / "runtime/gpu_budget.json")
                    and config["supervisor_lock_paths"] == [str(self.physical["worker"]), str(self.physical["gpu"])],
                    "bumped role scientific identity, selected IDs or canonical GPU lease policy changed")

    def _verify(self, *, prepared=True):
        result = super()._verify(prepared=prepared)
        self._caps()
        return result

    def _manifest(self, value):
        binding, _, replacement = self._caps()
        actual = self.engine.canonical_hash(value)
        if actual == binding["parent"]["manifest_hash"]:
            return super()._manifest(value)
        require(actual == binding["replacement"]["manifest_hash"] and self.engine.canonical_hash(value) == self.engine.canonical_hash(replacement),
            "manifest outside exact reviewed cap pair")
        projected = copy.deepcopy(value)
        recovery, stage, validation = self._recovery()[0], self._stage()[0], self._validation()[0]
        roots = {"v6-first-seed-T5": recovery["replacement"]["engine_root"], FIRST_SEED_ID: stage["engine"]["root"],
            **{job_id: validation["engine"]["root"] for job_id in INITIAL_IDS},
            **{job_id: binding["engine"]["root"] for job_id in EXECUTABLE_IDS}}
        for job in projected["jobs"]:
            if job["job_id"] in roots:
                require(job["cwd"] == roots[job["job_id"]], "pinned executable cwd exception changed")
                job["cwd"] = str(self.engine_root)
        self.base.CampaignControl._manifest(self, projected)
        return value

    def call(self, command, *, manifest=None, job_id=None, receipt=None):
        require(command in {"status", "tick", "refine-caps", "authorize-recovery"}, "cap facade supports only status/tick/exact refinement/same-source recovery")
        if command == "authorize-recovery":
            require(job_id in EXECUTABLE_IDS and manifest is None and receipt is not None,
                "same-source recovery requires an exact new cap job, no revised manifest, and analysis receipt")
            return self.base.CampaignControl.call(self, command, manifest=None, job_id=job_id, receipt=receipt)
        if command != "refine-caps":
            return self.base.CampaignControl.call(self, command, manifest=manifest, job_id=job_id, receipt=receipt)
        self._verify()
        self._registered()
        self._manifest(manifest)
        with self.routed_locks():
            return self.supervisor.refine_caps(manifest)


def make_control(*, validation_operator, stage_operator, recovery_operator, base_operator, **kwargs):
    validation = load_validation(validation_operator)
    stage = validation.load_stage(stage_operator)
    recovery = stage.load_recovery(recovery_operator)
    original = recovery.load_base(base_operator)
    class CapRefinementControl(CapRefinementMixin, validation.ValidationMixin, stage.StageMixin, recovery.RecoveryMixin, original.CampaignControl):
        base = original
    return CapRefinementControl(**kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("validation-operator", "stage-operator", "recovery-operator", "base-operator", "data-root", "campaign-root", "engine-root",
                 "hold-receipt", "recovery-binding", "stage-binding", "validation-binding", "cap-binding"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("recovery-binding-sha256", "stage-binding-sha256", "validation-binding-sha256", "cap-binding-sha256"):
        parser.add_argument("--" + name, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    sub.add_parser("tick")
    refine = sub.add_parser("refine-caps")
    refine.add_argument("--manifest", type=Path, required=True)
    recovery = sub.add_parser("authorize-recovery")
    recovery.add_argument("--job-id", required=True)
    recovery.add_argument("--receipt", type=Path, required=True)
    args = vars(parser.parse_args(argv))
    command, manifest = args.pop("command"), args.pop("manifest", None)
    job_id, receipt = args.pop("job_id", None), args.pop("receipt", None)
    control = make_control(**args)
    value = control.base.read_json(manifest) if manifest is not None else None
    print(json.dumps(control.call(command, manifest=value, job_id=job_id, receipt=receipt), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
