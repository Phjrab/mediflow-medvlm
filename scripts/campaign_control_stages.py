#!/usr/bin/env python3
"""Pinned next-stage activation using the unchanged released Supervisor.

Only the never-started first-seed CPU aggregate may be activated. The ten
completed training jobs, T5 recovery proof, canonical locks and old hold remain
unchanged. No queue, timer, retry loop or replacement Supervisor exists here.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
RECOVERY_SHA256 = "44eb762ba5ebdbaed21226a856ee1f9b054255e9ca81f9ac4cec376065178673"
STAGE_ID = "v6-first_seed_final"
STAGE_MODULE = "mediflow_medvlm.campaign_recorded_first_seed"
ACTIVATION_FIELDS = {"argv", "cwd", "source_files", "source_hash", "config_hash",
    "data_hash", "model_hash", "success_receipt", "implementation_pending", "uses_gpu", "gpu_lock_mode"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_recovery(path):
    path = Path(path)
    require(path.is_absolute() and ".." not in path.parts
        and not any(p.is_symlink() for p in (path, *path.parents)), "absolute nonsymlink recovery operator required")
    import hashlib
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == RECOVERY_SHA256, "original recovery operator changed")
    spec = importlib.util.spec_from_loader("_pinned_recovery_for_stages", loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def validate_pair(engine, before, after):
    """Pure activation contract, also checked again by native activate()."""
    engine.validate_manifest(before)
    engine.validate_manifest(after)
    require({k:v for k,v in before.items() if k != "jobs"}
        == {k:v for k,v in after.items() if k != "jobs"}, "stage cannot change campaign metadata")
    require([j["job_id"] for j in before["jobs"]] == [j["job_id"] for j in after["jobs"]],
        "stage cannot change job membership or order")
    changed = []
    for previous, replacement in zip(before["jobs"], after["jobs"]):
        if previous == replacement:
            continue
        require(previous["job_id"] == STAGE_ID, "only unstarted CPU first-seed aggregate may change")
        require(previous.get("implementation_pending") is True and not previous.get("argv"),
            "stage parent must be an unimplemented placeholder")
        require(not ({k for k in set(previous) | set(replacement)
            if previous.get(k) != replacement.get(k)} - ACTIVATION_FIELDS),
            "stage cannot change scientific identity or dependency policy")
        require(replacement.get("implementation_pending") is False and replacement.get("uses_gpu") is False
            and not replacement.get("requires_gpu", False)
            and replacement.get("gpu_lock_mode", "wrapper") == "wrapper", "stage must be CPU only")
        argv = replacement.get("argv", [])
        require(len(argv) == 7 and argv[1:4] == ["-m", STAGE_MODULE, "--config"]
            and argv[5:] == ["--receipt", replacement.get("success_receipt")],
            "stage must use the exact reviewed aggregate command")
        changed.append(replacement)
    require(len(changed) == 1, "exactly one pending aggregate must activate")
    return changed[0]


class StageMixin:
    def __init__(self, *, stage_binding, stage_binding_sha256, **kwargs):
        require(isinstance(stage_binding_sha256, str) and re.fullmatch(r"[0-9a-f]{64}", stage_binding_sha256),
            "external stage binding SHA256 required")
        self.stage_path = Path(stage_binding)
        self.stage_sha256 = stage_binding_sha256
        super().__init__(**kwargs)
        self.stage_path = self.base.absolute(self.stage_path)

    def _stage(self):
        raw = self.base.read_bytes(self.stage_path)
        import hashlib
        require(hashlib.sha256(raw).hexdigest() == self.stage_sha256, "reviewed stage binding bytes changed")
        binding = json.loads(raw)
        require(binding.get("schema_version") == "mediflow.recorded-stage-activation.v1"
            and binding.get("job_id") == STAGE_ID
            and binding.get("campaign_root") == str(self.root)
            and binding.get("data_root") == str(self.data_root), "stage binding namespace mismatch")
        recovery, _, parent = self._recovery()
        require(binding.get("recovery_binding_sha256") == self.recovery_sha256
            and binding["parent"]["manifest_hash"] == recovery["replacement"]["manifest_hash"],
            "stage must extend exactly the reviewed T5 recovery manifest")
        manifests = []
        for pins in (binding["parent"], binding["replacement"]):
            path = self.base.absolute(pins["manifest_path"])
            require(path.is_relative_to(self.root / "manifests"), "stage manifest escaped namespace")
            require(self.base.file_hash(path) == pins["manifest_sha256"], "stage manifest bytes changed")
            value = self.base.read_json(path)
            require(self.engine.canonical_hash(value) == pins["manifest_hash"], "stage manifest identity changed")
            manifests.append(value)
        require(manifests[0] == parent, "stage parent differs from reviewed recovery manifest")
        job = validate_pair(self.engine, *manifests)
        pins = binding["engine"]
        source_root = self.base.absolute(pins["root"])
        require(source_root.parent == self.root / "snapshots" and source_root.name == pins["commit"]
            and re.fullmatch(r"[0-9a-f]{40}", pins["commit"]), "immutable stage source commit archive required")
        tree = {p.relative_to(source_root).as_posix(): self.base.file_hash(p)
            for p in sorted((source_root / "src").rglob("*.py"))}
        require(tree == pins["source_files"] and self.base.digest(tree) == pins["source_tree_hash"],
            "reviewed complete stage source tree changed")
        require(job["cwd"] == str(source_root), "stage cwd changed")
        self.engine.verify_source(job)
        path = self.base.absolute(job["argv"][4])
        require(path.is_relative_to(self.root / "configs") and self.base.file_hash(path) == binding["config"]["sha256"]
            and str(path) == binding["config"]["path"], "stage config path or bytes changed")
        config = self.base.read_json(path)
        require(config.get("config_hash") == self.base.digest({k:v for k,v in config.items() if k != "config_hash"})
            and all(config.get(k) == job.get(k) for k in ("job_id", "run_id", "source_hash", "config_hash", "data_hash", "model_hash", "source_files")),
            "stage config identity does not match job")
        return binding, manifests[0], manifests[1]

    def _verify(self, *, prepared=True):
        value = super()._verify(prepared=prepared)
        self._stage()
        return value

    def _manifest(self, value):
        binding, parent, replacement = self._stage()
        actual = self.engine.canonical_hash(value)
        if actual == binding["parent"]["manifest_hash"]:
            return super()._manifest(value)
        require(actual == binding["replacement"]["manifest_hash"] and value == replacement,
            "manifest outside explicitly reviewed stage pair")
        # Original namespace validator still checks every executable receipt and
        # path. Only two exact, independently pinned source roots are projected.
        projected = {**value, "jobs": [dict(job) for job in value["jobs"]]}
        recovery = self._recovery()[0]
        for job in projected["jobs"]:
            if job["job_id"] in (STAGE_ID, "v6-first-seed-T5"):
                expected = binding["engine"]["root"] if job["job_id"] == STAGE_ID else recovery["replacement"]["engine_root"]
                require(job["cwd"] == expected, "pinned job cwd changed")
                job["cwd"] = str(self.engine_root)
        self.base.CampaignControl._manifest(self, projected)
        return value

    def call(self, command, *, manifest=None, job_id=None, receipt=None):
        require(command in {"status", "tick", "activate"}, "stage facade supports status, tick and one reviewed activation only")
        if command == "activate":
            require(manifest == self._stage()[2], "activation must use the exact reviewed stage manifest")
        # Bypass only RecoveryMixin's narrower command allowlist. The original
        # facade retains locking, namespace checks and native Supervisor API.
        return self.base.CampaignControl.call(self, command, manifest=manifest, job_id=job_id, receipt=receipt)


def make_control(*, recovery_operator, base_operator, **kwargs):
    recovery = load_recovery(recovery_operator)
    original = recovery.load_base(base_operator)
    class StageControl(StageMixin, recovery.RecoveryMixin, original.CampaignControl):
        base = original
    return StageControl(**kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("recovery-operator", "base-operator", "data-root", "campaign-root", "engine-root", "hold-receipt", "recovery-binding", "stage-binding"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--recovery-binding-sha256", required=True)
    parser.add_argument("--stage-binding-sha256", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status"); sub.add_parser("tick")
    activate = sub.add_parser("activate")
    activate.add_argument("--manifest", type=Path, required=True)
    args = vars(parser.parse_args(argv))
    command, manifest = args.pop("command"), args.pop("manifest", None)
    control = make_control(**args)
    value = control.base.read_json(manifest) if manifest is not None else None
    print(json.dumps(control.call(command, manifest=value), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
