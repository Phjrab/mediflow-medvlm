"""Small dependency-free atomic state, process lease and persistent GPU budget."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
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


@dataclass(frozen=True)
class ResourcePolicy:
    """Current execution limits, separate from immutable recipes and usage."""
    budget_mode: str = "unlimited"
    total_gpu_hours: float | None = None
    per_model_gpu_hours: float | None = None
    run_wall_seconds: float | None = None
    job_timeout_seconds: float | None = None
    campaign_wall_seconds: float | None = None

    def __post_init__(self):
        if self.budget_mode not in {"finite", "unlimited"}:
            raise ValueError("budget_mode must be finite or unlimited")
        limits = {name: value for name, value in asdict(self).items() if name != "budget_mode"}
        for name, value in limits.items():
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not math.isfinite(value) or value <= 0):
                raise ValueError(name + " must be a positive finite number or null")
        if self.budget_mode == "unlimited" and any(value is not None for value in limits.values()):
            raise ValueError("unlimited resource limits must be explicit null values")
        if self.budget_mode == "finite" and (self.total_gpu_hours is None or self.per_model_gpu_hours is None):
            raise ValueError("finite policy requires positive total_gpu_hours and per_model_gpu_hours")

    def as_dict(self):
        return {"schema_version": "medvlm.resource-policy.v1", **asdict(self)}

    def deadline(self, started, prior_seconds=0.0, *, scope="run"):
        limit = {"run": self.run_wall_seconds, "job": self.job_timeout_seconds,
                 "campaign": self.campaign_wall_seconds}[scope]
        return None if limit is None else started + max(0.0, limit - prior_seconds)

    def can_continue(self, elapsed_seconds, *, scope="run"):
        limit = {"run": self.run_wall_seconds, "job": self.job_timeout_seconds,
                 "campaign": self.campaign_wall_seconds}[scope]
        return limit is None or elapsed_seconds < limit


def resolve_resource_policy(config=None, *, policy=None, policy_path=None):
    """Resolve one policy for trainer, reload, evaluator, diagnostics and jobs.

    Explicit policy/path, environment override, a config's explicit policy, then
    the repository current policy take precedence in that order. Legacy recipe
    fields without budget_mode never override current execution policy. Reading
    an old config or checkpoint does not mutate it or revive its old limits.
    """
    if isinstance(policy, ResourcePolicy):
        return policy
    config = config or {}
    selected_path = policy_path or os.environ.get("MEDVLM_RESOURCE_POLICY") or config.get("resource_policy_path")
    if policy is not None:
        selected = policy
    elif selected_path:
        selected = json.loads(Path(selected_path).read_text())
    elif "resource_policy" in config:
        selected = config["resource_policy"]
    elif "budget_mode" in config:
        selected = config
    else:
        current = Path(__file__).resolve().parents[2] / "configs" / "resource_policy.json"
        selected = json.loads(current.read_text()) if current.exists() else ResourcePolicy().as_dict()
    if not isinstance(selected, dict):
        raise ValueError("resource policy must be an object")
    if selected.get("schema_version", "medvlm.resource-policy.v1") != "medvlm.resource-policy.v1":
        # A run config has its own schema; only nested/standalone policy schemas
        # must identify the policy format.
        if selected is not config:
            raise ValueError("unsupported resource policy schema")
    if "budget_mode" not in selected:
        raise ValueError("explicit resource policy requires budget_mode")
    fields = {name: selected.get(name) for name in ResourcePolicy.__dataclass_fields__}
    if selected is config and fields["budget_mode"] == "finite":
        fields["run_wall_seconds"] = selected.get("run_wall_seconds", selected.get("wall_seconds"))
        fields["job_timeout_seconds"] = selected.get("job_timeout_seconds", selected.get("timeout_seconds"))
    return ResourcePolicy(**fields)


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
    def __init__(self, path, total_hours=None, per_model_hours=None, *, policy=None):
        self.path = Path(path)
        if total_hours is not None or per_model_hours is not None:
            if policy is not None:
                raise ValueError("choose explicit finite limits or a resource policy")
            policy = ResourcePolicy("finite", total_hours, per_model_hours)
        self.policy = resolve_resource_policy(policy=policy)
        self.total_seconds = None if self.policy.total_gpu_hours is None else self.policy.total_gpu_hours * 3600
        self.model_seconds = None if self.policy.per_model_gpu_hours is None else self.policy.per_model_gpu_hours * 3600
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
            remaining = self._remaining(data, model_id)
            if remaining is not None and remaining <= 0:
                raise BudgetExceeded("GPU budget exhausted")
            current = self.policy.as_dict()
            if data.get("current_policy") != current:
                data.setdefault("policy_events", []).append({"at": utcnow(), "previous": data.get("current_policy"),
                                                            "current": current, "usage_preserved": True})
                data["current_policy"] = current
            self.session_id = uuid.uuid4().hex
            data["active"] = {"id": self.session_id, "model_id": model_id, "run_id": run_id,
                              "phase": phase, "started_epoch": time.time(), "resource_policy": current}
        return self

    def _remaining(self, data, model_id):
        if self.policy.budget_mode == "unlimited":
            return None
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
            remaining = self._remaining(data, model_id)
            return None if remaining is None else max(0.0, remaining)

    def can_continue(self, model_id, reserve_seconds=0.0):
        if not math.isfinite(reserve_seconds) or reserve_seconds < 0:
            raise ValueError("reserve_seconds must be finite and nonnegative")
        remaining = self.remaining(model_id)
        return remaining is None or remaining > reserve_seconds

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
                    "resource_policy": self.policy.as_dict(),
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
