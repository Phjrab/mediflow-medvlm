"""Small dependency-free atomic state, process lease and persistent GPU budget."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import time
import uuid

STATUSES = {"PENDING", "RUNNING", "SUCCEEDED", "BLOCKED_ACCESS", "BLOCKED_LICENSE",
            "BLOCKED_HARDWARE", "NETWORK_UNREACHABLE", "FAILED_REPRODUCIBLE",
            "SKIPPED_BUDGET", "PREPARED_NOT_RUN"}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_digest(path) -> dict:
    root = Path(path)
    files = {str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob("*")) if p.is_file()}
    return {"sha256": canonical_hash(files), "files": files}


def atomic_json(path, value) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        if os.name == "posix":
            fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)


class ResourceBusy(RuntimeError):
    pass


class BudgetExceeded(RuntimeError):
    pass


class GPULease:
    """Advisory OS lock. Never kills processes; a crashed process releases its fd."""
    def __init__(self, path, run_id="unknown"):
        self.path, self.run_id, self.stream = Path(path), run_id, None

    def __enter__(self):
        import fcntl
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.stream.close()
            raise ResourceBusy("RESOURCE_BUSY: another worker holds the GPU lease") from exc
        self.stream.seek(0)
        self.stream.truncate()
        json.dump({"pid": os.getpid(), "host": socket.gethostname(), "run_id": self.run_id,
                   "started_at": utcnow()}, self.stream)
        self.stream.flush()
        return self

    def __exit__(self, *args):
        import fcntl
        fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        self.stream.close()
        # Do not unlink the inode: waiters may already have opened it.


class BudgetLedger:
    """GPU occupancy seconds aggregated by canonical model ID across all runs.

    Caller holds the GPU lease for the whole session. An interrupted active
    reservation is charged until the next holder recovers it (conservative).
    """
    def __init__(self, path, total_hours=24, per_model_hours=6):
        self.path = Path(path)
        self.total_seconds = min(float(total_hours), 24.0) * 3600
        self.model_seconds = min(float(per_model_hours), 6.0) * 3600
        self.session_id = None
        self.session_seconds = 0.0

    @contextmanager
    def _locked(self):
        with GPULease(str(self.path) + ".lock", "budget-ledger"):
            data = json.loads(self.path.read_text()) if self.path.exists() else {"schema_version": "medvlm.budget.v1", "models": {}, "sessions": [], "active": None}
            if "run_phases" not in data:
                data["run_phases"] = {}
                for session in data["sessions"]:
                    phases = data["run_phases"].setdefault(session["run_id"], {})
                    phases[session["phase"]] = phases.get(session["phase"], 0.0) + session["seconds"]
            yield data
            atomic_json(self.path, data)

    @staticmethod
    def _close_active(data, now):
        active = data.get("active")
        if active:
            seconds = max(0.0, now - active["started_epoch"])
            data["models"][active["model_id"]] = data["models"].get(active["model_id"], 0.0) + seconds
            phases = data["run_phases"].setdefault(active["run_id"], {})
            phases[active["phase"]] = phases.get(active["phase"], 0.0) + seconds
            data["sessions"].append({**active, "finished_epoch": now, "seconds": seconds})
            data["active"] = None

    def start(self, model_id, run_id, phase="train"):
        if "/" not in model_id:
            raise ValueError("budget key must be the canonical full model repository ID")
        with self._locked() as data:
            self._close_active(data, time.time())
            if self._remaining(data, model_id) <= 0:
                raise BudgetExceeded("GPU budget exhausted")
            self.session_id = uuid.uuid4().hex
            data["active"] = {"id": self.session_id, "model_id": model_id, "run_id": run_id,
                              "phase": phase, "started_epoch": time.time()}
        return self

    def _remaining(self, data, model_id):
        used_model = data["models"].get(model_id, 0)
        used_total = sum(data["models"].values())
        active = data.get("active")
        if active:
            elapsed = max(0, time.time() - active["started_epoch"])
            used_total += elapsed
            if active["model_id"] == model_id:
                used_model += elapsed
        return min(self.model_seconds - used_model, self.total_seconds - used_total)

    def remaining(self, model_id):
        with self._locked() as data:
            return max(0.0, self._remaining(data, model_id))

    def snapshot(self):
        """A monotonic lower bound for recovery, including current occupancy."""
        with self._locked() as data:
            models = dict(data["models"])
            run_phases = {run: dict(phases) for run, phases in data["run_phases"].items()}
            active = data.get("active")
            if active:
                elapsed = max(0.0, time.time() - active["started_epoch"])
                models[active["model_id"]] = models.get(active["model_id"], 0.0) + elapsed
                phases = run_phases.setdefault(active["run_id"], {})
                phases[active["phase"]] = phases.get(active["phase"], 0.0) + elapsed
            return {"schema_version": "medvlm.budget-snapshot.v1", "captured_at": utcnow(),
                    "models": models, "run_phases": run_phases,
                    "total_gpu_seconds": sum(models.values()),
                    "per_model_limit_seconds": self.model_seconds, "total_limit_seconds": self.total_seconds}

    def recover(self, snapshot):
        """Restore missing usage by maximum, never add an already charged snapshot.

        Called only while owning the GPU lease and before start(). A surviving
        ledger remains authoritative when it records more usage than a checkpoint.
        """
        import math
        if self.session_id is not None:
            raise RuntimeError("recover budget before starting a new session")
        if snapshot.get("schema_version") != "medvlm.budget-snapshot.v1":
            raise ValueError("unsupported budget snapshot schema")
        for model, value in snapshot["models"].items():
            if "/" not in model or not math.isfinite(value) or value < 0:
                raise ValueError("invalid model budget snapshot")
        for phases in snapshot.get("run_phases", {}).values():
            if any(not math.isfinite(value) or value < 0 for value in phases.values()):
                raise ValueError("invalid run budget snapshot")
        with self._locked() as data:
            self._close_active(data, time.time())
            restored = {}
            for model, floor in snapshot["models"].items():
                old = data["models"].get(model, 0.0)
                data["models"][model] = max(old, floor)
                restored[model] = max(0.0, floor - old)
            for run, phases in snapshot.get("run_phases", {}).items():
                current = data["run_phases"].setdefault(run, {})
                for phase, floor in phases.items():
                    current[phase] = max(current.get(phase, 0.0), floor)
            data.setdefault("recoveries", []).append({"snapshot_sha256": canonical_hash(snapshot),
                                                      "restored_seconds": restored, "at": utcnow()})
        return restored

    def finish(self):
        with self._locked() as data:
            if data.get("active", {}) and data["active"]["id"] != self.session_id:
                raise RuntimeError("GPU budget session ownership changed")
            self._close_active(data, time.time())
            if self.session_id:
                self.session_seconds = next((row["seconds"] for row in reversed(data["sessions"])
                                             if row["id"] == self.session_id), 0.0)


def append_event(path, event):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"at": utcnow(), **event}, ensure_ascii=False) + "\n")
        stream.flush()
