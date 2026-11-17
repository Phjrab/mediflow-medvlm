#!/usr/bin/env python3
"""Single-owner T5 source-only recovery facade; no scheduler or worker loop.

The ORIGINAL pinned operator and released bf5 Supervisor stay authoritative.
An optional externally SHA256-pinned binding permits only the reviewed T5 job
to use one new source snapshot. Existing namespace binding/locks/hold/ledger
are never reinitialized. No candidate engine package is imported here.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
BASE_OPERATOR_SHA256 = "d1fdd822286a459961845f7250136bdf617686771edbc5e0ffaa2f007e5293ca"
JOB_ID = "v6-first-seed-T5"
COMMANDS = {"status", "tick", "authorize-recovery"}


def load_base(path):
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("absolute nonsymlink base operator path required")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != BASE_OPERATOR_SHA256:
        raise ValueError("original frozen operator bytes changed")
    # Execute the SAME verified bytes, avoiding a second path read/import race.
    spec = importlib.util.spec_from_loader("_pinned_campaign_operator_for_recovery", loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


class RecoveryMixin:
    def __init__(self, *, recovery_binding=None, recovery_binding_sha256=None, **kwargs):
        if (recovery_binding is None) != (recovery_binding_sha256 is None):
            raise ValueError("recovery binding and externally pinned SHA256 must be supplied together")
        if recovery_binding_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", recovery_binding_sha256):
            raise ValueError("invalid recovery binding SHA256")
        self.recovery_path = self.base.absolute(recovery_binding) if recovery_binding is not None else None
        self.recovery_sha256 = recovery_binding_sha256
        super().__init__(**kwargs)

    def _source_tree(self, root):
        root = self.base.absolute(root)
        sources = {p.relative_to(root).as_posix(): self.base.file_hash(p)
                   for p in sorted((root / "src").rglob("*.py"))}
        if len(sources) != 55:
            raise ValueError("recovery snapshot must retain the complete 55-file source tree")
        return self.base.digest(sources)

    def _recovery(self):
        if self.recovery_path is None:
            return None
        raw = self.base.read_bytes(self.recovery_path)
        if hashlib.sha256(raw).hexdigest() != self.recovery_sha256:
            raise ValueError("reviewed recovery binding bytes changed")
        binding = json.loads(raw)
        if (binding.get("schema_version") != "mediflow.t5-source-recovery.v1" or binding.get("job_id") != JOB_ID
                or binding.get("data_root") != str(self.data_root) or binding.get("campaign_root") != str(self.root)
                or binding.get("hold_receipt") != str(self.hold_path)
                or binding.get("hold_sha256") != self.base.file_hash(self.hold_path)):
            raise ValueError("recovery binding namespace, old hold or T5 identity mismatch")
        old, new = binding["original"], binding["replacement"]
        if (old.get("engine_sha") != self.base.ENGINE_SHA or old.get("engine_root") != str(self.engine_root)
                or old.get("engine_source_hash") != self.base.ENGINE_SOURCE_HASH):
            raise ValueError("recovery cannot replace the original released Supervisor engine")
        new_root = self.base.absolute(new["engine_root"])
        if (new_root == self.engine_root or self._source_tree(new_root) != new.get("engine_source_hash")
                or new.get("engine_source_hash") == self.base.ENGINE_SOURCE_HASH):
            raise ValueError("reviewed replacement source tree changed or was not revised")
        manifests, jobs = [], []
        for pins in (old, new):
            path = self.base.absolute(pins["manifest_path"])
            if not path.is_relative_to(self.root):
                raise ValueError("recovery manifest must remain in this campaign namespace")
            data = self.base.read_bytes(path)
            manifest = json.loads(data)
            if (hashlib.sha256(data).hexdigest() != pins.get("manifest_sha256")
                    or self.engine.canonical_hash(manifest) != pins.get("manifest_hash")):
                raise ValueError("reviewed recovery manifest changed")
            self.engine.validate_manifest(manifest)
            matches = [j for j in manifest["jobs"] if j["job_id"] == JOB_ID]
            if len(matches) != 1 or self.engine.canonical_hash(matches[0]) != pins.get("job_hash"):
                raise ValueError("reviewed T5 job hash mismatch")
            job = matches[0]
            if (job.get("cell_id") != "T5" or job.get("run_id") != JOB_ID or job.get("phase") != "native_main_train"
                    or job.get("cwd") != pins["engine_root"]):
                raise ValueError("only native T5 may use the recovery snapshot")
            argv = job["argv"]
            if argv.count("--config") != 1 or argv[argv.index("--config") + 1] != pins.get("config_path"):
                raise ValueError("reviewed T5 config path mismatch")
            config_data = self.base.read_bytes(pins["config_path"])
            config = json.loads(config_data)
            if (hashlib.sha256(config_data).hexdigest() != pins.get("config_sha256")
                    or config.get("config_hash") != pins.get("config_hash")
                    or job.get("config_hash") != pins.get("config_hash")):
                raise ValueError("reviewed T5 config bytes/hash changed")
            manifests.append(manifest); jobs.append(job)
        before, after = manifests
        if (self.engine.canonical_hash({k: v for k, v in before.items() if k != "jobs"})
                != self.engine.canonical_hash({k: v for k, v in after.items() if k != "jobs"})
                or [j["job_id"] for j in before["jobs"]] != [j["job_id"] for j in after["jobs"]]):
            raise ValueError("recovery cannot change campaign metadata, job order or membership")
        for original, replacement in zip(before["jobs"], after["jobs"]):
            if original["job_id"] != JOB_ID and self.engine.canonical_hash(original) != self.engine.canonical_hash(replacement):
                raise ValueError("recovery cannot change any other job")
        if jobs[1].get("source_files") != new.get("source_files") or jobs[1].get("source_hash") != new.get("source_hash"):
            raise ValueError("reviewed T5 source files/hash mismatch")
        # This is the ORIGINAL bf5 helper. Its relative imports resolve to bf5
        # modules; repo=new_root is used only to inspect candidate files. No
        # candidate module is imported into this owner process. Full input checks
        # remain mandatory inside the native authorize_recovery operation.
        self.engine.verify_gpu_source_rebind(jobs[0], jobs[1], validate_inputs=False)
        return binding, before, after

    def _verify(self, *, prepared=True):
        result = super()._verify(prepared=prepared)
        self._recovery()
        return result  # Existing original-engine namespace binding is unchanged.

    def _manifest(self, value):
        reviewed = self._recovery()
        if reviewed is None:
            return super()._manifest(value)
        binding, original, replacement = reviewed
        actual = self.engine.canonical_hash(value)
        if actual not in {binding["original"]["manifest_hash"], binding["replacement"]["manifest_hash"]}:
            raise ValueError("manifest is outside the explicitly reviewed recovery pair")
        if actual == binding["original"]["manifest_hash"]:
            return super()._manifest(value)
        # Reuse all original generic namespace/receipt/path checks; only the
        # exact pinned T5 cwd exception is projected back to the original root.
        projected = {**value, "jobs": [dict(job) for job in value["jobs"]]}
        for job in projected["jobs"]:
            if job["job_id"] == JOB_ID:
                if job["cwd"] != binding["replacement"]["engine_root"]:
                    raise ValueError("T5 recovery cwd differs from the reviewed snapshot")
                job["cwd"] = str(self.engine_root)
        super()._manifest(projected)
        return value

    def call(self, command, *, manifest=None, job_id=None, receipt=None):
        if command not in COMMANDS:
            raise ValueError("recovery facade supports only status, tick and explicit authorize-recovery")
        if command == "authorize-recovery" and self.recovery_path is not None:
            if job_id != JOB_ID or manifest is None:
                raise ValueError("reviewed source recovery requires explicit T5 and replacement manifest")
            reviewed = self._recovery()
            if self.engine.canonical_hash(manifest) != reviewed[0]["replacement"]["manifest_hash"]:
                raise ValueError("authorization requires the exact reviewed replacement manifest")
        return super().call(command, manifest=manifest, job_id=job_id, receipt=receipt)


def make_control(*, base_operator, **kwargs):
    original = load_base(base_operator)

    class RecoveryCampaignControl(RecoveryMixin, original.CampaignControl):
        base = original

    return RecoveryCampaignControl(**kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base-operator", "data-root", "campaign-root", "engine-root", "hold-receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--recovery-binding", type=Path)
    parser.add_argument("--recovery-binding-sha256")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status"); sub.add_parser("tick")
    authorize = sub.add_parser("authorize-recovery")
    authorize.add_argument("--manifest", type=Path)
    authorize.add_argument("--job-id", required=True)
    authorize.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    facade = make_control(base_operator=args.base_operator, data_root=args.data_root,
                          campaign_root=args.campaign_root, engine_root=args.engine_root,
                          hold_receipt=args.hold_receipt, recovery_binding=args.recovery_binding,
                          recovery_binding_sha256=args.recovery_binding_sha256)
    manifest = facade.base.read_json(args.manifest) if getattr(args, "manifest", None) else None
    result = facade.call(args.command, manifest=manifest, job_id=getattr(args, "job_id", None), receipt=getattr(args, "receipt", None))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
