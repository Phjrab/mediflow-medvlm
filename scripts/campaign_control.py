#!/usr/bin/env python3
"""Operator facade for ONE frozen Supervisor in a new campaign namespace.

No daemon, loop, scheduler, CUDA import, hardlinks or source rewrites. The old
queue is never ticked. The existing automation/service must be routed exclusively
here before activation: shared locks prevent simultaneous owners, but cannot
fence a later invocation of the old controller after this command returns.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import stat
import sys

sys.dont_write_bytecode = True
ENGINE_SHA = "bf5c8d7d718dafa511d6371d1862622801044bf7"
# All 55 Python sources under src/, verified against the released engine tree.
ENGINE_SOURCE_HASH = "3af3e42add7debaab82572d88de4583c3c3c86927eb26300ffe12b4374387f23"
BINDING = "operator_namespace_binding.json"
METHODS = {"init": "initialize", "status": "status", "tick": "tick",
           "activate": "activate", "refine-preflight": "refine_preflight",
           "refine-first-seed": "refine_first_seed",
           "authorize-recovery": "authorize_recovery", "finalize": "finalize"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def absolute(path):
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("absolute normalized paths required")
    for parent in reversed((path, *path.parents)):
        if parent.is_symlink():
            raise ValueError("symlink paths are forbidden")
    return path


def read_bytes(path):
    path = absolute(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("regular file required")
        return stream.read()


def read_json(path):
    return json.loads(read_bytes(path))


def file_hash(path):
    return hashlib.sha256(read_bytes(path)).hexdigest()


def lock_identity(path):
    path = absolute(path)
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("existing regular physical lock required")
    return {"device": info.st_dev, "inode": info.st_ino}


def verify_engine(root):
    root = absolute(root)
    sources = {p.relative_to(root).as_posix(): file_hash(p)
               for p in sorted((root / "src").rglob("*.py"))}
    if digest(sources) != ENGINE_SOURCE_HASH:
        raise ValueError("frozen released engine sources changed")
    for name, module in tuple(sys.modules.items()):
        if (name == "mediflow_medvlm" or name.startswith("mediflow_medvlm.")) and getattr(module, "__file__", None):
            if not Path(module.__file__).is_relative_to(root / "src"):
                raise ValueError("another engine package is already imported")
    sys.path.insert(0, str(root / "src"))
    try:
        return importlib.import_module("mediflow_medvlm.campaign_supervisor")
    finally:
        sys.path.pop(0)


class CampaignControl:
    def __init__(self, *, data_root, campaign_root, engine_root, hold_receipt):
        self.data_root = absolute(data_root)
        self.root = absolute(campaign_root)
        self.engine_root = absolute(engine_root)
        self.hold_path = absolute(hold_receipt)
        self.old_root = self.data_root / "v6_v7"
        if (self.root.parent != self.old_root / "campaigns"
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", self.root.name)):
            raise ValueError("campaign root must be data_root/v6_v7/campaigns/<campaign_id>")
        self.physical = {"supervisor": self.old_root / "supervisor.lock",
                         "worker": self.old_root / "worker.lock",
                         "gpu": self.data_root / "runtime" / "medvlm_gpu.lock"}
        self.engine = verify_engine(self.engine_root)
        # Exactly the frozen Supervisor, with only its namespace attributes set.
        self.supervisor = self.engine.Supervisor(self.data_root)
        self.supervisor.root = self.root
        self.supervisor.manifest_path = self.root / "manifest.json"
        self.supervisor.state_path = self.root / "state.json"

    def _hold(self):
        hold = read_json(self.hold_path)
        control, old = hold.get("dispatch_control", {}), hold.get("old_campaign", {})
        if (hold.get("writer") != "TRAINING_OWNER" or control.get("status") != "MIGRATION_HOLD"
                or control.get("tick_prohibited") is not True or hold.get("active_workers") != []):
            raise ValueError("explicit old campaign MIGRATION_HOLD without active workers required")
        if (absolute(old.get("campaign_root", "")) != self.old_root
                or absolute(old.get("state_path", "")) != self.old_root / "state.json"):
            raise ValueError("hold receipt belongs to another old campaign")
        manifest_path = absolute(old.get("manifest_path", ""))
        if not manifest_path.is_relative_to(self.old_root):
            raise ValueError("old manifest must remain in old namespace")
        if (file_hash(self.old_root / "state.json") != old.get("state_sha256")
                or file_hash(manifest_path) != old.get("manifest_sha256")):
            raise ValueError("old campaign state/manifest changed since MIGRATION_HOLD")
        state = read_json(self.old_root / "state.json")
        for record in state["jobs"].values():
            identities = [record.get("worker_identity")]
            if record.get("attempt_dir"):
                attempt = absolute(record["attempt_dir"])
                if not attempt.is_relative_to(self.old_root):
                    raise ValueError("old attempt escaped old namespace")
                heartbeat = attempt / "heartbeat.json"
                if heartbeat.exists():
                    health = read_json(heartbeat)
                    identities += [health.get("worker_identity"), health.get("child_identity")]
            if any(self.engine.same_process(identity) for identity in identities):
                raise ValueError("old campaign worker still alive; no new dispatch")
        return {"hold_sha256": file_hash(self.hold_path), "old_state_sha256": old["state_sha256"],
                "old_manifest_sha256": old["manifest_sha256"]}

    def _binding(self):
        return {"schema_version": "mediflow.operator-namespace.v1", "engine_sha": ENGINE_SHA,
                "engine_source_hash": ENGINE_SOURCE_HASH, "engine_root": str(self.engine_root),
                "data_root": str(self.data_root), "campaign_root": str(self.root),
                "hold_receipt": str(self.hold_path), **self._hold(),
                "physical_locks": {k: {"path": str(v), **lock_identity(v)} for k, v in self.physical.items()},
                "gpu_ledger_path": str(self.data_root / "runtime" / "gpu_budget.json")}

    def _verify(self, *, prepared=True):
        absolute(self.root)
        verify_engine(self.engine_root)
        binding = self._binding()
        if prepared and read_json(self.root / BINDING) != binding:
            raise ValueError("namespace binding or physical lock inode changed")
        # A namespace alias must not silently become a second physical lock.
        if any((self.root / (name + ".lock")).exists() or (self.root / (name + ".lock")).is_symlink()
               for name in ("worker", "supervisor")):
            raise ValueError("new namespace lock files are forbidden; use canonical routing only")
        return binding

    @contextmanager
    def _physical_lock(self, path):
        """Open an EXISTING canonical file; never create, link, replace or truncate."""
        info = lock_identity(path)
        fd = os.open(absolute(path), os.O_RDWR | os.O_NOFOLLOW)
        with os.fdopen(fd, "r+") as stream:
            actual = os.fstat(stream.fileno())
            if info != {"device": actual.st_dev, "inode": actual.st_ino}:
                raise ValueError("physical lock changed while opening")
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                yield None
            else:
                try:
                    yield stream
                finally:
                    # Frozen tick closes its parent copy after passing the open
                    # description to the detached worker. NEVER unlock that copy.
                    if not stream.closed:
                        fcntl.flock(stream, fcntl.LOCK_UN)

    @contextmanager
    def routed_locks(self):
        original = self.engine.try_lock
        mapping = {self.root / "worker.lock": self.physical["worker"],
                   self.root / "supervisor.lock": self.physical["supervisor"],
                   **{p: p for p in self.physical.values()}}

        @contextmanager
        def route(path):
            path = Path(path)
            if path not in mapping:
                with original(path) as stream:
                    yield stream
                return
            with self._physical_lock(mapping[path]) as stream:
                if stream is not None and mapping[path] == self.physical["supervisor"]:
                    # Recheck the hold and inode binding INSIDE the same physical
                    # supervisor lock, before the frozen method mutates anything.
                    self._verify()
                yield stream

        self.engine.try_lock = route
        try:
            yield
        finally:
            self.engine.try_lock = original

    def prepare(self):
        binding = self._verify(prepared=False)
        with self._physical_lock(self.physical["supervisor"]) as lock:
            if lock is None:
                return {"status": "SUPERVISOR_BUSY"}
            binding = self._verify(prepared=False)
            with ExitStack() as stack:
                for key in ("worker", "gpu"):
                    if stack.enter_context(self._physical_lock(self.physical[key])) is None:
                        raise ValueError("existing worker/GPU lease busy; refuse namespace preparation")
                path = self.root / BINDING
                if path.exists():
                    if read_json(path) != binding:
                        raise ValueError("existing namespace binding differs")
                else:
                    self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
                    self.engine.atomic_json(path, binding)
        return {"status": "NAMESPACE_PREPARED", **binding}

    def _manifest(self, value):
        if not isinstance(value, dict) or value.get("campaign_id") != self.root.name:
            raise ValueError("manifest campaign_id must equal namespace directory")
        if "campaign_root" in value and value["campaign_root"] != str(self.root):
            raise ValueError("manifest campaign_root mismatch")
        self.engine.validate_manifest(value)
        for job in value.get("jobs", []):
            if job.get("argv") and not job.get("implementation_pending"):
                if absolute(job["cwd"]) != self.engine_root:
                    raise ValueError("executable jobs must use the pinned released engine root")
                if not absolute(job["success_receipt"]).is_relative_to(self.root):
                    raise ValueError("job success receipt escaped new campaign namespace")
        return value

    def _registered(self):
        state = read_json(self.supervisor.state_path)
        version = absolute(state.get("manifest_path", ""))
        if not version.is_relative_to(self.root / "manifests"):
            raise ValueError("registered manifest pointer escaped new namespace")
        self._manifest(read_json(version))
        if state.get("campaign_id") != self.root.name or state.get("queue_owner") != "supervisor":
            raise ValueError("new namespace must have the single Supervisor owner")
        for record in state.get("jobs", {}).values():
            if record.get("attempt_dir") and not absolute(record["attempt_dir"]).is_relative_to(self.root / "attempts"):
                raise ValueError("attempt path escaped new namespace")

    def call(self, command, *, manifest=None, job_id=None, receipt=None):
        if command not in METHODS:
            raise ValueError("unsupported forwarding operation")
        self._verify()
        if manifest is not None:
            self._manifest(manifest)
        if command != "init":
            self._registered()
        elif self.supervisor.state_path.exists():
            self._registered()
        method = getattr(self.supervisor, METHODS[command])
        with self.routed_locks():
            if command in {"init", "activate", "refine-preflight", "refine-first-seed"}:
                return method(manifest)
            if command == "authorize-recovery":
                return method(job_id, absolute(receipt), manifest)
            if command == "finalize":
                return method(absolute(receipt))
            return method()  # Exactly one status/tick invocation, never a loop.


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "campaign-root", "engine-root", "hold-receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    for name in METHODS:
        child = sub.add_parser(name)
        if name in {"init", "activate", "refine-preflight", "refine-first-seed"}:
            child.add_argument("--manifest", type=Path, required=True)
        if name == "authorize-recovery":
            child.add_argument("--manifest", type=Path)
            child.add_argument("--job-id", required=True)
            child.add_argument("--receipt", type=Path, required=True)
        if name == "finalize":
            child.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    facade = CampaignControl(data_root=args.data_root, campaign_root=args.campaign_root,
                             engine_root=args.engine_root, hold_receipt=args.hold_receipt)
    result = facade.prepare() if args.command == "prepare" else facade.call(
        args.command, manifest=read_json(args.manifest) if getattr(args, "manifest", None) else None,
        job_id=getattr(args, "job_id", None), receipt=getattr(args, "receipt", None))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
