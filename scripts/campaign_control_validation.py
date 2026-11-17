#!/usr/bin/env python3
"""Externally pinned registration of twenty initial calibration roles.

The sole bf5 Supervisor gains one registration method. Its status, tick, worker,
recovery and cumulative-counter implementations remain inherited verbatim. The
existing facade still owns namespace fencing and the same physical lock files.
No candidate engine package is imported into this owner process.
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
STAGE_OPERATOR_SHA256 = "a4d1b996ecfd2f3c433ea42846e4e13520f9e44d57b163a81061ed1ee44936d7"
SCHEMA = "mediflow.recorded-validation-refinement.v1"
CAPS_ID = "v6-validation_caps"
FIRST_SEED_ID = "v6-first_seed_final"
CELL_IDS = ["T1", "T2", "T3", "T4", "T5", "T6", "V1", "V2", "V3", "V4"]
ROLES = ("base", "tuned")
INITIAL_IDS = [f"v6-caps-initial-{cell}-{role}" for cell in CELL_IDS for role in ROLES]
PREREQUISITES = ["v6-first-seed-" + cell for cell in CELL_IDS] + [FIRST_SEED_ID]
WORKER_MODULE = "mediflow_medvlm.campaign_validation_caps"
EVENT = "RECORDED_VALIDATION_INITIAL_REFINEMENT"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_stage(path):
    path = Path(path)
    require(path.is_absolute() and ".." not in path.parts
        and not any(p.is_symlink() for p in (path, *path.parents)), "absolute nonsymlink stage operator required")
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == STAGE_OPERATOR_SHA256, "original stage operator changed")
    spec = importlib.util.spec_from_loader("_pinned_stage_for_validation", loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def validate_pair(engine, parent, replacement):
    """Pure, exact matrix refinement; all old job metadata stays immutable."""
    engine.validate_manifest(parent)
    engine.validate_manifest(replacement)
    equal = lambda a, b: engine.canonical_hash(a) == engine.canonical_hash(b)
    require(equal({k: v for k, v in parent.items() if k != "jobs"},
        {k: v for k, v in replacement.items() if k != "jobs"}), "validation refinement cannot change campaign metadata")
    before = {j["job_id"]: j for j in parent["jobs"]}
    after = {j["job_id"]: j for j in replacement["jobs"]}
    require(CAPS_ID in before and not set(INITIAL_IDS).intersection(before), "unrefined cap gate required")
    order = [j["job_id"] for j in parent["jobs"]]
    position = order.index(CAPS_ID)
    order[position:position] = INITIAL_IDS
    require([j["job_id"] for j in replacement["jobs"]] == order,
        "exact twenty initial roles and unchanged original job order required")
    gate = before[CAPS_ID]
    require(gate.get("phase") == "validation_caps" and gate.get("run_id") == CAPS_ID
        and gate.get("cell_id") == "ALL_V6" and gate.get("implementation_pending") is True
        and not gate.get("argv") and gate.get("dependencies") == [FIRST_SEED_ID]
        and gate.get("dependency_policy") == "terminal", "original unimplemented cap gate required")
    for job_id, job in before.items():
        expected = copy.deepcopy(job)
        if job_id == CAPS_ID:
            expected["dependencies"] = [FIRST_SEED_ID, *INITIAL_IDS]
        require(equal(after[job_id], expected), "only original cap dependencies may change")
    allowed = {"job_id", "run_id", "phase", "cell_id", "model_id", "task", "family", "role", "calibration_round",
        "seed", "dependencies", "dependency_policy", "implementation_pending", "uses_gpu", "requires_gpu",
        "gpu_lock_mode", "argv", "cwd", "source_files", "source_hash", "config_hash", "data_hash", "model_hash",
        "success_receipt"}
    for cell in CELL_IDS:
        main = before.get("v6-first-seed-" + cell, {})
        require(main.get("cell_id") == cell and main.get("phase") == "native_main_train", "original ten training jobs required")
        for role in ROLES:
            job_id = f"v6-caps-initial-{cell}-{role}"
            job = after[job_id]
            argv = job.get("argv", [])
            require(not set(job) - allowed and job.get("run_id") == job_id
                and job.get("phase") == "validation_caps_calibration"
                and all(equal(job.get(k), main.get(k)) for k in ("cell_id", "model_id", "task", "family"))
                and job.get("role") == role and job.get("calibration_round") == "initial"
                and type(job.get("seed")) is int and job["seed"] == 20261023
                and job.get("dependencies") == [FIRST_SEED_ID] and job.get("dependency_policy") == "success"
                and job.get("uses_gpu") is True and job.get("requires_gpu") is True
                and job.get("implementation_pending") is False and job.get("gpu_lock_mode") == "wrapper"
                and len(argv) == 7 and argv[1:4] == ["-m", WORKER_MODULE, "--config"]
                and argv[5:] == ["--receipt", job.get("success_receipt")], "invalid initial calibration role identity")
    return [after[job_id] for job_id in INITIAL_IDS]


def make_supervisor_class(engine, owner):
    """The only added Supervisor behavior is a finite registration operation."""
    class ValidationSupervisor(engine.Supervisor):
        def refine_validation(self, revised):
            with engine.try_lock(self.root / "supervisor.lock") as lock:
                if lock is None:
                    return {"status": "SUPERVISOR_BUSY"}
                binding, parent, replacement = owner._validation()
                require(engine.canonical_hash(revised) == engine.canonical_hash(replacement),
                    "registration requires the exact externally pinned replacement")
                current, state = self.load()
                require(state.get("queue_owner") == "supervisor", "sole Supervisor ownership required")
                digest = engine.canonical_hash(replacement)
                if state["manifest_hash"] == digest:
                    matches = [event for event in state.get("manifest_revisions", [])
                        if event.get("kind") == EVENT and event.get("manifest_hash") == digest
                        and event.get("validation_binding_sha256") == owner.validation_sha256
                        and event.get("added_jobs") == INITIAL_IDS]
                    require(len(matches) == 1 and engine.canonical_hash(current) == digest,
                        "matching immutable validation refinement audit required")
                    return self.status()
                require(engine.canonical_hash(current) == binding["parent"]["manifest_hash"]
                    and engine.canonical_hash(current) == engine.canonical_hash(parent), "validation parent is not active")
                require(set(state["jobs"]) == {j["job_id"] for j in parent["jobs"]},
                    "exact original state/job keys required; orphan records may not be overwritten")
                require(owner.base.file_hash(self.state_path) == binding["parent_state_sha256"], "parent state bytes changed")
                cap_record = state["jobs"][CAPS_ID]
                require(cap_record.get("status") == "IMPLEMENTATION_PENDING"
                    and all(type(cap_record.get(k)) is int and cap_record[k] == 0
                        for k in ("attempts", "failure_count", "recovery_attempts"))
                    and not cap_record.get("history") and all(cap_record.get(k) is None for k in
                        ("attempt_dir", "launch_token", "worker_identity", "verification",
                         "recovery_authorization_hash", "recovery_authorization")), "cap gate must be untouched and unstarted")
                for record in state["jobs"].values():
                    health = {}
                    if record.get("attempt_dir"):
                        path = owner.base.absolute(record["attempt_dir"]) / "heartbeat.json"
                        if path.exists():
                            health = owner.base.read_json(path)
                    require(record.get("status") not in {"STARTING", "RUNNING"}
                        and not any(engine.same_process(identity) for identity in
                            (record.get("worker_identity"), health.get("worker_identity"), health.get("child_identity"))),
                        "worker active; refinement is registration only")
                with ExitStack() as leases:
                    for path in (self.root / "worker.lock", self.data_root / "runtime/medvlm_gpu.lock"):
                        require(leases.enter_context(engine.try_lock(path)) is not None,
                            "worker/GPU lease busy; refuse validation registration")
                    proofs = {}
                    jobs = {j["job_id"]: j for j in parent["jobs"]}
                    for job_id in PREREQUISITES:
                        record, pins = state["jobs"][job_id], binding["prerequisites"][job_id]
                        require(record.get("status") == "SUCCEEDED" and record.get("exit_code") == 0
                            and type(record.get("attempts")) is int and record["attempts"] > 0
                            and engine.canonical_hash(record) == pins["state_hash"], "all eleven original results must be verified successes")
                        verification = record.get("verification", {})
                        proof = engine.verify_success(jobs[job_id])
                        require(verification.get("receipt") == pins["path"]
                            and verification.get("receipt_sha256") == pins["sha256"]
                            and verification.get("artifact_count") == proof["artifact_count"], "original success verification changed")
                        proofs[job_id] = proof
                    # Recheck every external byte binding after the receipt audit.
                    owner._verify()
                    require(owner.base.file_hash(self.state_path) == binding["parent_state_sha256"],
                        "parent state changed during receipt audit")
                    version = self.root / "manifests" / (digest + ".json")
                    if version.exists():
                        require(engine.canonical_hash(owner.base.read_json(version)) == digest,
                            "immutable validation manifest version changed")
                    else:
                        engine.atomic_json(version, replacement)
                    state.setdefault("manifest_revisions", []).append({"kind": EVENT,
                        "previous_hash": state["manifest_hash"], "previous_manifest_path": state["manifest_path"],
                        "manifest_hash": digest, "manifest_path": str(version), "added_jobs": INITIAL_IDS,
                        "validation_binding": str(owner.validation_path),
                        "validation_binding_sha256": owner.validation_sha256,
                        "protocol": binding["protocol"], "prerequisite_proofs": proofs, "at": engine.now()})
                    for job_id in INITIAL_IDS:
                        state["jobs"][job_id] = {"status": "PENDING", "attempts": 0,
                            "failure_count": 0, "recovery_attempts": 0, "history": []}
                    state.update(manifest_hash=digest, manifest_path=str(version), updated_at=engine.now())
                    engine.atomic_json(self.state_path, state)
                return self.status()
    return ValidationSupervisor


class ValidationMixin:
    def __init__(self, *, validation_binding, validation_binding_sha256, **kwargs):
        require(isinstance(validation_binding_sha256, str) and re.fullmatch(r"[0-9a-f]{64}", validation_binding_sha256),
            "external validation binding SHA256 required")
        self.validation_path = Path(validation_binding)
        self.validation_sha256 = validation_binding_sha256
        super().__init__(**kwargs)
        self.validation_path = self.base.absolute(self.validation_path)
        # Keep the same instance, namespace and state; add exactly one method.
        self.supervisor.__class__ = make_supervisor_class(self.engine, self)

    def _validation(self):
        raw = self.base.read_bytes(self.validation_path)
        require(hashlib.sha256(raw).hexdigest() == self.validation_sha256, "reviewed validation binding changed")
        binding = json.loads(raw)
        require(binding.get("schema_version") == SCHEMA and binding.get("campaign_root") == str(self.root)
            and binding.get("data_root") == str(self.data_root)
            and binding.get("stage_operator_sha256") == STAGE_OPERATOR_SHA256
            and binding.get("stage_binding_sha256") == self.stage_sha256
            and binding.get("recovery_binding_sha256") == self.recovery_sha256,
            "validation namespace or inherited owner bindings changed")
        stage_binding, _, stage_parent = self._stage()
        require(binding["parent"]["manifest_hash"] == stage_binding["replacement"]["manifest_hash"],
            "validation must extend exactly the reviewed first-seed aggregate manifest")
        manifests = []
        for pins in (binding["parent"], binding["replacement"]):
            path = self.base.absolute(pins["manifest_path"])
            require(path.is_relative_to(self.root / "manifests")
                and self.base.file_hash(path) == pins["manifest_sha256"], "validation manifest path or bytes changed")
            value = self.base.read_json(path)
            require(self.engine.canonical_hash(value) == pins["manifest_hash"], "validation manifest identity changed")
            manifests.append(value)
        require(self.engine.canonical_hash(manifests[0]) == self.engine.canonical_hash(stage_parent),
            "validation parent differs from inherited exact stage manifest")
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
        require(bool(archive) and archive == pins["archive_files"]
            and self.base.digest(archive) == pins["archive_tree_hash"], "complete source archive changed")
        sources = {k: v for k, v in archive.items() if k.startswith("src/") and k.endswith(".py")}
        require(sources == pins["source_files"] and self.base.digest(sources) == pins["source_tree_hash"]
            and "src/mediflow_medvlm/campaign_validation_caps.py" in sources, "complete calibration source tree changed")
        require(set(binding["configs"]) == set(INITIAL_IDS) and set(binding["prerequisites"]) == set(PREREQUISITES),
            "exact twenty config pins and eleven prerequisite receipt pins required")
        protocol = self.base.absolute(binding["protocol"]["path"])
        require(protocol.is_relative_to(self.root) and self.base.file_hash(protocol) == binding["protocol"]["sha256"],
            "immutable CPU calibration protocol changed")
        protocol_value = self.base.read_json(protocol)
        require(protocol_value.get("schema_version") == "medvlm.v6.validation-cap-protocol.v1"
            and protocol_value.get("protocol_hash") == self.base.digest({k: v for k, v in protocol_value.items() if k != "protocol_hash"})
            and protocol_value.get("campaign_root") == str(self.root)
            and protocol_value.get("campaign_id") == self.root.name
            and protocol_value.get("first_seed_receipt") == binding["first_seed_receipt"]
            and self.base.digest(protocol_value.get("initial_caps")) == self.base.digest(
                {"medical_mcq": 64, "context_qa": 64, "medical_qa": 256, "medical_vqa": 256})
            and self.base.digest(protocol_value.get("maximum_caps")) == self.base.digest(
                {"medical_mcq": 128, "context_qa": 128, "medical_qa": 512, "medical_vqa": 512})
            and type(protocol_value.get("eval_context")) is int and protocol_value["eval_context"] == 4096
            and protocol_value.get("test_files_read") is False
            and protocol_value.get("planned_failures_stay_in_denominator") is True
            and protocol_value.get("decode", {}).get("do_sample") is False
            and protocol_value["decode"].get("max_time") is None
            and set(protocol_value.get("models", {})) == set(CELL_IDS), "frozen validation-only protocol policy changed")
        parents = {j["job_id"]: j for j in manifests[0]["jobs"]}
        for job_id, reference in binding["prerequisites"].items():
            path = self.base.absolute(reference["path"])
            require(path.is_relative_to(self.root) and str(path) == parents[job_id]["success_receipt"]
                and self.base.file_hash(path) == reference["sha256"]
                and self.engine.canonical_hash(parents[job_id]) == reference["job_hash"], "original result receipt or job pins changed")
        require(binding["first_seed_receipt"] == {k: binding["prerequisites"][FIRST_SEED_ID][k] for k in ("path", "sha256")},
            "aggregate first-seed receipt pin mismatch")
        for cell in CELL_IDS:
            model = protocol_value["models"][cell]
            trained = parents["v6-first-seed-" + cell]
            require(model.get("training_status") == "SUCCEEDED" and model.get("eligible_for_calibration") is True
                and model.get("receipt") == {k: binding["prerequisites"][trained["job_id"]][k] for k in ("path", "sha256")}
                and model.get("config", {}).get("path") == trained["argv"][4]
                and self.base.file_hash(self.base.absolute(model["config"]["path"])) == model["config"]["sha256"],
                "protocol must bind the ten exact original trained configurations and receipts")
        for job in jobs:
            require(job["cwd"] == str(source), "calibration cwd differs from reviewed new archive")
            self.engine.verify_source(job)
            reference = binding["configs"][job["job_id"]]
            path = self.base.absolute(job["argv"][4])
            require(path.is_relative_to(self.root / "configs") and str(path) == reference["path"]
                and self.base.file_hash(path) == reference["sha256"], "reviewed role config bytes or path changed")
            config = self.base.read_json(path)
            require(config.get("schema_version") == "medvlm.v6.validation-cap-config.v1"
                and config.get("scope") == "VALIDATION_ONLY_NATIVE_CAP_CALIBRATION"
                and config.get("config_hash") == self.base.digest({k: v for k, v in config.items() if k != "config_hash"})
                and all(self.base.digest(config.get(k)) == self.base.digest(job.get(k)) for k in
                    ("job_id", "run_id", "cell_id", "model_id", "task", "family", "role", "calibration_round",
                     "seed", "phase", "source_files", "source_hash", "config_hash", "data_hash", "model_hash", "success_receipt"))
                and all(k in sources and sources[k] == sha for k, sha in config["source_files"].items()),
                "role config identity/source does not match pinned job")
            data, cell = config.get("data", {}), job["cell_id"]
            require(config.get("uses_gpu") is True and config.get("device") == "cuda"
                and config.get("eval_context") == 4096 and config.get("local_files_only") is True
                and config.get("trust_remote_code") is False and config.get("fixed_template_date") == "2026-11-14"
                and config.get("decode") == protocol_value["decode"]
                and config.get("resource_policy") == protocol_value.get("resource_policy")
                and config.get("gpu_ledger_path") == str(self.data_root / "runtime/gpu_budget.json")
                and config.get("supervisor_lock_paths") == [str(self.physical["worker"]), str(self.physical["gpu"])]
                and data.get("protocol") == binding["protocol"] and data.get("protocol_hash") == protocol_value["protocol_hash"]
                and data.get("first_seed_receipt") == binding["first_seed_receipt"]
                and data.get("partition") == "validation" and data.get("decision") is None
                and data.get("trained_cell") == protocol_value["models"][cell]
                and config["data_hash"] == self.base.digest(data)
                and config["model_hash"] == self.base.digest({"model": config["model"], "role": job["role"],
                    "final_checkpoint": protocol_value["models"][cell]["checkpoint"] if job["role"] == "tuned" else None})
                and self.base.absolute(config["output_dir"]).is_relative_to(self.root),
                "role protocol, scientific identity or canonical lease/ledger binding changed")
        return binding, manifests[0], manifests[1]

    def _verify(self, *, prepared=True):
        result = super()._verify(prepared=prepared)
        self._validation()
        return result

    def _manifest(self, value):
        binding, _, replacement = self._validation()
        actual = self.engine.canonical_hash(value)
        if actual == binding["parent"]["manifest_hash"]:
            return super()._manifest(value)
        require(actual == binding["replacement"]["manifest_hash"]
            and self.engine.canonical_hash(value) == self.engine.canonical_hash(replacement),
            "manifest outside exact reviewed validation pair")
        projected = copy.deepcopy(value)
        recovery = self._recovery()[0]
        stage = self._stage()[0]
        expected_roots = {"v6-first-seed-T5": recovery["replacement"]["engine_root"],
            FIRST_SEED_ID: stage["engine"]["root"],
            **{job_id: binding["engine"]["root"] for job_id in INITIAL_IDS}}
        for job in projected["jobs"]:
            if job["job_id"] in expected_roots:
                require(job["cwd"] == expected_roots[job["job_id"]], "pinned executable cwd exception changed")
                job["cwd"] = str(self.engine_root)
        self.base.CampaignControl._manifest(self, projected)
        return value

    def call(self, command, *, manifest=None, job_id=None, receipt=None):
        require(command in {"status", "tick", "refine-validation", "authorize-recovery"},
            "validation facade supports only status, tick, exact refinement and same-source role recovery")
        if command == "authorize-recovery":
            require(job_id in INITIAL_IDS and manifest is None and receipt is not None,
                "same-source recovery requires an exact initial role, no revised manifest, and an analysis receipt")
            return self.base.CampaignControl.call(self, command, manifest=None, job_id=job_id, receipt=receipt)
        if command != "refine-validation":
            return self.base.CampaignControl.call(self, command, manifest=manifest, job_id=job_id, receipt=receipt)
        self._verify()
        self._registered()
        self._manifest(manifest)
        with self.routed_locks():
            return self.supervisor.refine_validation(manifest)


def make_control(*, stage_operator, recovery_operator, base_operator, **kwargs):
    stage = load_stage(stage_operator)
    recovery = stage.load_recovery(recovery_operator)
    original = recovery.load_base(base_operator)
    class ValidationControl(ValidationMixin, stage.StageMixin, recovery.RecoveryMixin, original.CampaignControl):
        base = original
    return ValidationControl(**kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stage-operator", "recovery-operator", "base-operator", "data-root", "campaign-root", "engine-root",
                 "hold-receipt", "recovery-binding", "stage-binding", "validation-binding"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("recovery-binding-sha256", "stage-binding-sha256", "validation-binding-sha256"):
        parser.add_argument("--" + name, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    sub.add_parser("tick")
    refine = sub.add_parser("refine-validation")
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
