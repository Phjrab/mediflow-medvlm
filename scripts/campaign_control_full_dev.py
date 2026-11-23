#!/usr/bin/env python3
"""Pinned full-dev registration through the existing sole native supervisor.

This adds twenty independent model/role jobs and activates only the untouched
CPU full-dev gate. Existing runtime methods, counters and leases stay native.
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
CAP_OPERATOR_SHA256 = "dcc22a54a892bcdfe32e998853510f14fb012725f0bfe3b89cfea2d0be59c050"
SCHEMA = "mediflow.recorded-full-dev-refinement.v1"
CONFIG_SCHEMA = "medvlm.v6.full-dev-config.v1"
WORKER_MODULE = "mediflow_medvlm.campaign_full_dev"
CAPS_ID = "v6-validation_caps"
GATE_ID = "v6-full_dev"
FIRST_SEED_ID = "v6-first_seed_final"
DECISION_ID = "v6-caps-decision"
CELL_IDS = ["T1", "T2", "T3", "T4", "T5", "T6", "V1", "V2", "V3", "V4"]
ROLES = ("base", "tuned")
INITIAL_IDS = [f"v6-caps-initial-{cell}-{role}" for cell in CELL_IDS for role in ROLES]
BUMPED_IDS = [f"v6-caps-bumped-{cell}-{role}" for cell in CELL_IDS for role in ROLES]
ADDED_IDS = [f"v6-full-dev-{cell}-{role}" for cell in CELL_IDS for role in ROLES]
EXECUTABLE_IDS = [*ADDED_IDS, GATE_ID]
PREREQUISITES = ["v6-first-seed-" + cell for cell in CELL_IDS] + [FIRST_SEED_ID, *INITIAL_IDS, DECISION_ID, *BUMPED_IDS, CAPS_ID]
EVENT = "RECORDED_FULL_DEV_REFINEMENT"
EXECUTION_FIELDS = {"dependencies", "dependency_policy", "implementation_pending", "uses_gpu", "requires_gpu",
    "gpu_lock_mode", "argv", "cwd", "source_files", "source_hash", "config_hash", "data_hash", "model_hash", "success_receipt"}
JOB_FIELDS = {"job_id", "run_id", "phase", "cell_id", "model_id", "task", "family", "role", "seed", "kind", *EXECUTION_FIELDS}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_caps(path):
    path = Path(path)
    require(path.is_absolute() and ".." not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)),
        "absolute nonsymlink cap operator required")
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == CAP_OPERATOR_SHA256, "original cap operator changed")
    spec = importlib.util.spec_from_loader("_pinned_caps_for_full_dev", loader=None)
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
    require(set(PREREQUISITES + [GATE_ID]).issubset(before) and not set(ADDED_IDS).intersection(before),
        "exact completed cap-refinement parent required")
    order = [j["job_id"] for j in parent["jobs"]]
    position = order.index(GATE_ID)
    order[position:position] = ADDED_IDS
    require([j["job_id"] for j in replacement["jobs"]] == order, "exact twenty full-dev roles and unchanged old order required")
    gate = before[GATE_ID]
    require(gate.get("phase") == "full_dev" and gate.get("run_id") == GATE_ID and gate.get("cell_id") == "ALL_V6"
        and gate.get("implementation_pending") is True and not gate.get("argv")
        and gate.get("dependencies") == [CAPS_ID] and gate.get("dependency_policy") == "terminal", "untouched original full-dev gate required")
    for job_id, job in before.items():
        if job_id == GATE_ID:
            require(same({k: v for k, v in job.items() if k not in EXECUTION_FIELDS},
                {k: v for k, v in after[job_id].items() if k not in EXECUTION_FIELDS}),
                "full-dev gate original scientific declaration must remain unchanged")
        else:
            require(same(after[job_id], job), "every existing started or unrelated job must remain identical")
    for job_id in EXECUTABLE_IDS:
        job, gpu = after[job_id], job_id in ADDED_IDS
        argv = job.get("argv", [])
        require(job.get("run_id") == job_id and job.get("implementation_pending") is False
            and job.get("uses_gpu") is gpu and job.get("requires_gpu") is gpu and job.get("gpu_lock_mode") == "wrapper"
            and len(argv) == 7 and argv[1:4] == ["-m", WORKER_MODULE, "--config"]
            and argv[5:] == ["--receipt", job.get("success_receipt")], "invalid full-dev executable wrapper")
        if job_id == GATE_ID:
            require(job.get("dependencies") == [CAPS_ID, *ADDED_IDS] and job.get("dependency_policy") == "terminal",
                "original full-dev gate must audit cap and twenty terminal roles")
        else:
            cell, role = job_id.removeprefix("v6-full-dev-").split("-")
            original = before[f"v6-caps-initial-{cell}-{role}"]
            require(set(job) == JOB_FIELDS and job.get("kind") == "role" and job.get("phase") == "full_dev_evaluation"
                and all(same(job.get(k), original.get(k)) for k in ("cell_id", "model_id", "task", "family", "role", "seed"))
                and type(job.get("seed")) is int and job["seed"] == 20261023
                and job.get("dependencies") == [CAPS_ID] and job.get("dependency_policy") == "success",
                "invalid independent full-dev role declaration")
    return [after[job_id] for job_id in EXECUTABLE_IDS]


def make_supervisor_class(engine, owner, inherited):
    class FullDevSupervisor(inherited):
        def refine_full_dev(self, revised):
            with engine.try_lock(self.root / "supervisor.lock") as lock:
                if lock is None:
                    return {"status": "SUPERVISOR_BUSY"}
                binding, parent, replacement = owner._full_dev()
                require(engine.canonical_hash(revised) == engine.canonical_hash(replacement), "exact pinned full-dev replacement required")
                current, state = self.load()
                require(state.get("queue_owner") == "supervisor", "sole Supervisor ownership required")
                digest = engine.canonical_hash(replacement)
                if state["manifest_hash"] == digest:
                    events = [e for e in state.get("manifest_revisions", []) if e.get("kind") == EVENT
                        and e.get("manifest_hash") == digest and e.get("full_dev_binding_sha256") == owner.full_dev_sha256
                        and e.get("added_jobs") == ADDED_IDS and e.get("activated_jobs") == [GATE_ID]]
                    require(len(events) == 1 and engine.canonical_hash(current) == digest, "exact immutable full-dev refinement audit required")
                    return self.status()
                require(engine.canonical_hash(current) == binding["parent"]["manifest_hash"]
                    and engine.canonical_hash(current) == engine.canonical_hash(parent), "full-dev refinement parent is not active")
                require(set(state["jobs"]) == {j["job_id"] for j in parent["jobs"]}, "exact original state/job keys required")
                require(owner.base.file_hash(self.state_path) == binding["parent_state_sha256"], "parent state bytes changed")
                cap = state["jobs"][GATE_ID]
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
                            health.get("worker_identity"), health.get("child_identity"))), "worker active; full-dev refinement is registration only")
                with ExitStack() as leases:
                    for path in (self.root / "worker.lock", self.data_root / "runtime/medvlm_gpu.lock"):
                        require(leases.enter_context(engine.try_lock(path)) is not None, "worker/GPU lease busy")
                    jobs = {j["job_id"]: j for j in parent["jobs"]}
                    proofs = {}
                    for job_id in PREREQUISITES:
                        record, pins = state["jobs"][job_id], binding["prerequisites"][job_id]
                        require(record.get("status") == "SUCCEEDED" and record.get("exit_code") == 0
                            and type(record.get("attempts")) is int and record["attempts"] > 0
                            and engine.canonical_hash(record) == pins["state_hash"], "all fifty-three prerequisites must be verified successes")
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
                        "activated_jobs": [GATE_ID], "full_dev_binding": str(owner.full_dev_path), "full_dev_binding_sha256": owner.full_dev_sha256,
                        "protocol": binding["protocol"], "prerequisite_proofs": proofs, "at": engine.now()})
                    for job_id in ADDED_IDS:
                        state["jobs"][job_id] = {"status": "PENDING", "attempts": 0, "failure_count": 0, "recovery_attempts": 0, "history": []}
                    cap["status"] = "PENDING"
                    state.update(manifest_hash=digest, manifest_path=str(version), updated_at=engine.now())
                    engine.atomic_json(self.state_path, state)
                return self.status()
    return FullDevSupervisor


class FullDevMixin:
    def __init__(self, *, full_dev_binding, full_dev_binding_sha256, **kwargs):
        require(isinstance(full_dev_binding_sha256, str) and re.fullmatch(r"[0-9a-f]{64}", full_dev_binding_sha256),
            "external full-dev binding SHA256 required")
        self.full_dev_path, self.full_dev_sha256 = Path(full_dev_binding), full_dev_binding_sha256
        super().__init__(**kwargs)
        self.full_dev_path = self.base.absolute(self.full_dev_path)
        self.supervisor.__class__ = make_supervisor_class(self.engine, self, type(self.supervisor))

    def _full_dev(self):
        raw = self.base.read_bytes(self.full_dev_path)
        require(hashlib.sha256(raw).hexdigest() == self.full_dev_sha256, "reviewed full-dev binding changed")
        binding = json.loads(raw)
        require(binding.get("schema_version") == SCHEMA and binding.get("campaign_root") == str(self.root)
            and binding.get("data_root") == str(self.data_root) and binding.get("cap_operator_sha256") == CAP_OPERATOR_SHA256
            and binding.get("cap_binding_sha256") == self.cap_sha256, "full-dev namespace or inherited owner binding changed")
        inherited, _, caps_parent = self._caps()
        require(binding["parent"]["manifest_hash"] == inherited["replacement"]["manifest_hash"], "full-dev must extend exact cap-refinement manifest")
        manifests = []
        for pins in (binding["parent"], binding["replacement"]):
            path = self.base.absolute(pins["manifest_path"])
            require(path.is_relative_to(self.root / "manifests") and self.base.file_hash(path) == pins["manifest_sha256"], "full-dev manifest path or bytes changed")
            value = self.base.read_json(path)
            require(self.engine.canonical_hash(value) == pins["manifest_hash"], "full-dev manifest identity changed")
            manifests.append(value)
        require(self.engine.canonical_hash(manifests[0]) == self.engine.canonical_hash(caps_parent), "full-dev parent differs from inherited manifest")
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
            and "src/mediflow_medvlm/campaign_full_dev.py" in sources, "complete full-dev source changed")
        require(set(binding["configs"]) == set(EXECUTABLE_IDS) and set(binding["prerequisites"]) == set(PREREQUISITES),
            "exact twenty-one configs and fifty-three prerequisites required")
        parents = {j["job_id"]: j for j in manifests[0]["jobs"]}
        for job_id, ref in binding["prerequisites"].items():
            path = self.base.absolute(ref["path"])
            require(path.is_relative_to(self.root) and str(path) == parents[job_id]["success_receipt"]
                and self.base.file_hash(path) == ref["sha256"] and self.engine.canonical_hash(parents[job_id]) == ref["job_hash"],
                "original prerequisite receipt or job pins changed")
        cap_job = parents[CAPS_ID]
        cap_config = self.base.read_json(self.base.absolute(cap_job["argv"][4]))
        protocol_path = self.base.absolute(binding["protocol"]["path"])
        require(protocol_path == self.base.absolute(cap_config["output_dir"]) / "protocol.final.json"
            and protocol_path.is_relative_to(self.root) and self.base.file_hash(protocol_path) == binding["protocol"]["sha256"],
            "exact frozen cap protocol path and bytes required")
        cap_receipt = self.base.read_json(self.base.absolute(binding["prerequisites"][CAPS_ID]["path"]))
        require(binding["protocol"] in cap_receipt["artifacts"], "frozen cap protocol must be in actual success artifacts")
        protocol = self.base.read_json(protocol_path)
        initial = self.base.read_json(self.base.absolute(inherited["protocol"]["path"]))
        require(protocol.get("schema_version") == "medvlm.v6.validation-caps-frozen.v1"
            and protocol.get("protocol_hash") == self.base.digest({k: v for k, v in protocol.items() if k != "protocol_hash"})
            and protocol.get("initial_protocol") == inherited["protocol"] and protocol.get("initial_protocol_hash") == initial["protocol_hash"]
            and protocol.get("cap_protocol_frozen") is True and protocol.get("further_bump_allowed") is False
            and protocol.get("eval_context") == 4096 and protocol.get("train_context") == 2048
            and protocol.get("decode") == initial["decode"] and protocol.get("fixed_template_date") == initial["fixed_template_date"]
            and protocol.get("test_files_read") is False and protocol.get("test_unlocked") is False
            and protocol.get("full_dev_completed") is False and protocol.get("family_selection_completed") is False
            and protocol.get("clinical_quality_certified") is False and protocol.get("campaign_terminal") is False
            and set(protocol.get("caps", {})) == set(initial["initial_caps"])
            and all(type(cap) is int and cap in (initial["initial_caps"][task], initial["maximum_caps"][task])
                for task, cap in protocol["caps"].items()), "frozen cap policy changed or claims later scientific completion")
        forge = binding["forge"]
        require(set(forge) == {"output_root", "data_root", "campaign_id"}
            and forge["data_root"] == str(self.data_root) and forge["campaign_id"] == self.root.name
            and self.base.absolute(forge["output_root"]) != self.root
            and not self.base.absolute(forge["output_root"]).is_relative_to(self.data_root)
            and not self.data_root.is_relative_to(self.base.absolute(forge["output_root"])), "exact disjoint external Forge telemetry namespace required")
        receipts = [job["success_receipt"] for job in jobs]
        require(len(set(receipts)) == len(receipts) and not set(receipts).intersection(j.get("success_receipt") for j in parents.values()),
            "new receipts must be distinct from every original receipt")
        old_outputs = []
        for old in parents.values():
            if old.get("argv") and not old.get("implementation_pending"):
                old_config = self.base.read_json(self.base.absolute(old["argv"][4]))
                if old_config.get("output_dir"):
                    old_outputs.append(self.base.absolute(old_config["output_dir"]))
        outputs = []
        for job in jobs:
            require(job["cwd"] == str(source), "new full-dev cwd differs from pinned archive")
            self.engine.verify_source(job)
            reference = binding["configs"][job["job_id"]]
            path = self.base.absolute(job["argv"][4])
            require(path.is_relative_to(self.root / "configs") and str(path) == reference["path"]
                and self.base.file_hash(path) == reference["sha256"], "reviewed full-dev config path or bytes changed")
            config = self.base.read_json(path)
            require(config.get("schema_version") == CONFIG_SCHEMA
                and config.get("config_hash") == self.base.digest({k: v for k, v in config.items() if k != "config_hash"})
                and all(self.base.digest(config.get(k)) == self.base.digest(job.get(k)) for k in
                    ("job_id", "run_id", "cell_id", "source_files", "source_hash", "config_hash", "data_hash", "model_hash", "success_receipt"))
                and all(k in sources and sources[k] == sha for k, sha in config["source_files"].items()),
                "full-dev config identity/source differs from pinned job")
            outputs.append(self.base.absolute(config["output_dir"]))
            self._config_policy_full_dev(config, job, binding, protocol, initial, parents)
        require(all(not self.base.absolute(r).is_relative_to(output) for r in receipts for output in outputs),
            "new success receipts must not overlap any new result output")
        require(len(set(outputs)) == len(outputs) and all(not a.is_relative_to(b)
            for i, a in enumerate(outputs) for j, b in enumerate(outputs) if i != j), "new outputs must be distinct and nonoverlapping")
        require(all(not a.is_relative_to(b) and not b.is_relative_to(a) for a in outputs for b in old_outputs)
            and all(not self.base.absolute(r).is_relative_to(b) for r in receipts for b in old_outputs), "new outputs or receipts overlap immutable old results")
        return binding, manifests[0], manifests[1]

    def _config_policy_full_dev(self, config, job, binding, protocol, initial, parents):
        digest = self.base.digest
        same = lambda a, b: digest(a) == digest(b)
        role = job["job_id"] in ADDED_IDS
        common = {"schema_version", "scope", "kind", "phase", "job_id", "run_id", "cell_id", "seed", "uses_gpu", "state_path",
            "output_dir", "success_receipt", "data", "model_hash", "resource_policy", "source_files", "source_hash", "data_hash", "config_hash"}
        gpu_fields = {"model_id", "task", "family", "role", "device", "model", "eval_context", "split", "protocol_version", "protocol_hash",
            "fixed_template_date", "decode", "local_files_only", "trust_remote_code", "precision", "gpu_ledger_path", "supervisor_lock_paths", "forge"}
        require(set(config) == common | (gpu_fields if role else set())
            and config["kind"] == ("role" if role else "aggregate")
            and config["scope"] == ("VALIDATION_ONLY_NATIVE_FULL_DEV" if role else "VALIDATION_ONLY_FULL_DEV_TERMINAL_AUDIT")
            and config["phase"] == job["phase"] and type(config["seed"]) is int and config["seed"] == 20261023
            and config["uses_gpu"] is role and config["state_path"] == str(self.supervisor.state_path)
            and same(config["resource_policy"], initial["resource_policy"]), "full-dev keyset, scope, seed or runtime policy changed")
        output, receipt = self.base.absolute(config["output_dir"]), self.base.absolute(config["success_receipt"])
        protected = [self.root / name for name in ("operator", "snapshots", "configs", "manifests", "receipts", "attempts")]
        require(output != self.root and output.is_relative_to(self.root) and receipt.is_relative_to(self.root)
            and not any(output.is_relative_to(p) or p.is_relative_to(output) for p in protected), "full-dev output overlaps protected owner files")
        data, pins = config["data"], binding["prerequisites"][CAPS_ID]
        proof = data["cap_freeze"]
        require(set(proof) == {"job", "terminal_state", "config", "receipt", "final_protocol"}
            and same(proof["job"], parents[CAPS_ID]) and digest(proof["terminal_state"]) == pins["state_hash"]
            and proof["terminal_state"].get("status") == "SUCCEEDED"
            and proof["receipt"] == {k: pins[k] for k in ("path", "sha256")}
            and proof["config"]["path"] == parents[CAPS_ID]["argv"][4]
            and self.base.file_hash(self.base.absolute(proof["config"]["path"])) == proof["config"]["sha256"]
            and proof["final_protocol"] == binding["protocol"] and data["protocol"] == binding["protocol"]
            and data["initial_protocol"] == protocol["initial_protocol"] and config["data_hash"] == digest(data),
            "actual frozen cap proof or initial/final protocol identity changed")
        if not role:
            require(set(data) == {"cap_freeze", "protocol", "initial_protocol", "roles"}
                and data["roles"] == [binding["configs"][key] for key in ADDED_IDS]
                and config["model_hash"] == digest(initial["models"]), "aggregate must audit exact ordered twenty role configs")
            return
        cell = job["cell_id"]
        model = initial["models"][cell]
        train = self.base.read_json(self.base.absolute(model["config"]["path"]))
        original = self.base.read_json(self.base.absolute(parents[f"v6-caps-initial-{cell}-{job['role']}"]["argv"][4]))
        reference = {k: initial["validation_references"][job["task"]][k] for k in ("path", "sha256")}
        path = self.base.absolute(reference["path"])
        require(self.base.file_hash(path) == reference["sha256"], "frozen validation dataset bytes changed")
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        ids = [row["sample_id"] for row in rows]
        require(len(ids) == (1516 if job["task"] == "text" else 512) and len(set(ids)) == len(ids), "exact full validation denominator required")
        scorer_files = {k: v for k, v in config["source_files"].items()
            if k.endswith(("evaluation/metrics.py", "evaluation/v2.py", "campaign_full_dev.py"))}
        require(set(scorer_files) == {"src/mediflow_medvlm/" + name for name in ("evaluation/metrics.py", "evaluation/v2.py", "campaign_full_dev.py")},
            "exact full-dev scorer sources required")
        evaluation = {"protocol_version": "medvlm.v6.full-dev.v1", "split": "validation", "cap_protocol": binding["protocol"],
            "cap_protocol_hash": protocol["protocol_hash"], "caps": protocol["caps"], "eval_context": 4096,
            "decode": protocol["decode"], "fixed_template_date": "2026-11-14", "sample_ids": ids,
            "input_digests": {row["sample_id"]: digest(row["input"]) for row in rows},
            "reference_digest": digest([(row["sample_id"], row["supervision"]) for row in rows]),
            "files": {job["task"]: reference["sha256"]}, "parser": "existing anchored full-string parse_answer; no answer substring extraction",
            "normalization": "NFKC casefold whitespace; VQA terminal punctuation only; preserve numbers/units/negation",
            "scorer_source_files": scorer_files, "planned_failures_in_denominator": True, "further_bump_allowed": False, "test_files_read": False}
        evaluation["protocol_hash"] = digest(evaluation)
        forge = {**binding["forge"], "receipt_path": str(self.root / "receipts" / (job["job_id"] + ".forge-import.json"))}
        require(set(data) == {"cap_freeze", "protocol", "initial_protocol", "validation", "ordered_sample_ids", "ordered_sample_ids_sha256", "count",
                "partition", "image_root", "trained_cell", "first_seed_receipt", "evaluation_protocol"}
            and data["validation"] == reference and data["ordered_sample_ids"] == ids and data["ordered_sample_ids_sha256"] == digest(ids)
            and type(data["count"]) is int and data["count"] == len(ids) and data["partition"] == "validation"
            and data["image_root"] == train["data"]["image_root"] and same(data["trained_cell"], model)
            and data["first_seed_receipt"] == initial["first_seed_receipt"] and same(data["evaluation_protocol"], evaluation)
            and all(same(config.get(k), original.get(k)) for k in ("model_id", "task", "family", "role", "seed", "model", "model_hash"))
            and same(config["model"], train["model"]) and config["device"] == "cuda" and config["eval_context"] == 4096
            and config["split"] == "validation" and config["protocol_version"] == "medvlm.v6.full-dev.v1"
            and config["protocol_hash"] == evaluation["protocol_hash"] and config["fixed_template_date"] == "2026-11-14"
            and config["decode"] == protocol["decode"] and config["local_files_only"] is True and config["trust_remote_code"] is False
            and config["precision"] == "nf4_double_quant_bfloat16" and config["gpu_ledger_path"] == str(self.data_root / "runtime/gpu_budget.json")
            and config["supervisor_lock_paths"] == [str(self.physical["worker"]), str(self.physical["gpu"])]
            and config["forge"] == forge, "full-dev scientific identity, scorer/IDs or Forge/native lease policy changed")

    def _verify(self, *, prepared=True):
        result = super()._verify(prepared=prepared)
        self._full_dev()
        return result

    def _manifest(self, value):
        binding, _, replacement = self._full_dev()
        actual = self.engine.canonical_hash(value)
        if actual == binding["parent"]["manifest_hash"]:
            return super()._manifest(value)
        require(actual == binding["replacement"]["manifest_hash"] and self.engine.canonical_hash(value) == self.engine.canonical_hash(replacement),
            "manifest outside exact reviewed full-dev pair")
        projected = copy.deepcopy(value)
        recovery, stage, validation, caps = self._recovery()[0], self._stage()[0], self._validation()[0], self._caps()[0]
        roots = {"v6-first-seed-T5": recovery["replacement"]["engine_root"], FIRST_SEED_ID: stage["engine"]["root"],
            **{key: validation["engine"]["root"] for key in INITIAL_IDS},
            **{key: caps["engine"]["root"] for key in [DECISION_ID, *BUMPED_IDS, CAPS_ID]},
            **{key: binding["engine"]["root"] for key in EXECUTABLE_IDS}}
        for job in projected["jobs"]:
            if job["job_id"] in roots:
                require(job["cwd"] == roots[job["job_id"]], "pinned executable cwd exception changed")
                job["cwd"] = str(self.engine_root)
        self.base.CampaignControl._manifest(self, projected)
        return value

    def call(self, command, *, manifest=None, job_id=None, receipt=None):
        require(command in {"status", "tick", "refine-full-dev", "authorize-recovery"},
            "full-dev facade supports only status/tick/exact refinement/same-source recovery")
        if command == "authorize-recovery":
            require(job_id in EXECUTABLE_IDS and manifest is None and receipt is not None,
                "same-source recovery requires an exact full-dev job, no revised manifest, and analysis receipt")
            return self.base.CampaignControl.call(self, command, manifest=None, job_id=job_id, receipt=receipt)
        if command != "refine-full-dev":
            return self.base.CampaignControl.call(self, command, manifest=manifest, job_id=job_id, receipt=receipt)
        self._verify()
        self._registered()
        self._manifest(manifest)
        with self.routed_locks():
            return self.supervisor.refine_full_dev(manifest)


def make_control(*, cap_operator, validation_operator, stage_operator, recovery_operator, base_operator, **kwargs):
    caps = load_caps(cap_operator)
    validation = caps.load_validation(validation_operator)
    stage = validation.load_stage(stage_operator)
    recovery = stage.load_recovery(recovery_operator)
    original = recovery.load_base(base_operator)
    class FullDevControl(FullDevMixin, caps.CapRefinementMixin, validation.ValidationMixin, stage.StageMixin, recovery.RecoveryMixin, original.CampaignControl):
        base = original
    return FullDevControl(**kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cap-operator", "validation-operator", "stage-operator", "recovery-operator", "base-operator", "data-root", "campaign-root", "engine-root",
                 "hold-receipt", "recovery-binding", "stage-binding", "validation-binding", "cap-binding", "full-dev-binding"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("recovery-binding-sha256", "stage-binding-sha256", "validation-binding-sha256", "cap-binding-sha256", "full-dev-binding-sha256"):
        parser.add_argument("--" + name, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    sub.add_parser("tick")
    refine = sub.add_parser("refine-full-dev")
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
