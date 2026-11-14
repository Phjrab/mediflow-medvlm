"""Durable, single-owner V6/V7 queue; CPU-only supervisor and detached workers.

Only ``tick`` changes queue/recovery state. Workers own their attempt heartbeat,
stdout/stderr and exit receipt. Neither elapsed time nor stale heartbeat kills a
process. The immutable manifest binds commands to a verified source snapshot.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

HASH_FIELDS = ("source_hash", "config_hash", "data_hash", "model_hash")
TERMINAL = {"SUCCEEDED", "FAILED", "BLOCKED_PREREQUISITE"}
MAX_FAILURES = 3  # Initial failure plus at most two technical recovery attempts.


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def terminal_state_hash(state):
    """Bind final receipts to outcomes, excluding heartbeat/update timestamps."""
    fields = ("status", "attempts", "failure_count", "recovery_attempts", "exit_code",
              "error_signature", "verification", "launch_token", "recovery_authorization_hash")
    return canonical_hash({job_id: {key: record.get(key) for key in fields}
                           for job_id, record in state["jobs"].items()})


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            os.chmod(temporary, 0o600)
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def process_identity(pid=None):
    """Linux PID identity, including boot and proc start ticks (not wall time)."""
    pid = os.getpid() if pid is None else int(pid)
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        # comm may contain spaces and parentheses; fields after its last ')' start at 3.
        fields = text[text.rfind(")") + 2:].split()
        if fields[0] == "Z":
            return None
        return {"worker_pid": pid, "process_start_time": fields[19],
                "boot_identity": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def same_process(identity):
    if not identity or not identity.get("worker_pid"):
        return False
    actual = process_identity(identity["worker_pid"])
    return bool(actual and all(str(actual.get(key)) == str(identity.get(key)) for key in actual))


@contextlib.contextmanager
def try_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    stream = path.open("a+")
    try:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield None
        else:
            yield stream
            if not stream.closed:
                fcntl.flock(stream, fcntl.LOCK_UN)
    finally:
        stream.close()


def validate_manifest(manifest):
    if not isinstance(manifest.get("campaign_id"), str) or not manifest["campaign_id"]:
        raise ValueError("campaign_id is required")
    jobs = manifest.get("jobs", [])
    ids = [job.get("job_id") for job in jobs]
    if len(set(ids)) != len(ids) or any(not re.fullmatch(r"[A-Za-z0-9_.-]+", str(i)) or i in {".", ".."} for i in ids):
        raise ValueError("unique path-safe job_id values are required")
    for job in jobs:
        for key in ("phase", "cell_id", "run_id"):
            if not job.get(key):
                raise ValueError(f"job {job['job_id']} needs {key}")
        if set(job.get("dependencies", [])) - set(ids) or job["job_id"] in job.get("dependencies", []):
            raise ValueError("invalid dependencies")
        if job.get("implementation_pending") or not job.get("argv"):
            continue
        if not isinstance(job["argv"], list) or any(not isinstance(x, str) for x in job["argv"]):
            raise ValueError("argv must be a string list, never a shell command")
        if not Path(job["argv"][0]).is_absolute() or not Path(job.get("cwd", "")).is_absolute():
            raise ValueError("absolute executable and snapshot cwd required")
        if any(not isinstance(job.get(key), str) or not job[key] for key in HASH_FIELDS):
            raise ValueError("all four immutable identity hashes required")
        if not Path(job.get("success_receipt", "")).is_absolute():
            raise ValueError("absolute success_receipt required")
        sources = job.get("source_files")
        if not sources or canonical_hash(sources) != job["source_hash"]:
            raise ValueError("source_hash must bind source_files")
        root = Path(job["cwd"]).resolve()
        for name in sources:
            if not (root / name).resolve().is_relative_to(root):
                raise ValueError("source file escapes snapshot")
        wrapper = "src/mediflow_medvlm/campaign_supervisor.py"
        if wrapper not in sources:
            raise ValueError("source_files must capture the worker wrapper")
        argv = job["argv"]
        if len(argv) >= 3 and argv[1] == "-m":
            module = "src/" + argv[2].replace(".", "/")
            captured = module + ".py" in sources or module + "/__main__.py" in sources
        elif len(argv) >= 2:
            script = (root / argv[1]).resolve()
            captured = script.is_relative_to(root) and str(script.relative_to(root)) in sources
        else:
            captured = False
        if not captured:
            raise ValueError("command must execute a captured snapshot module or script")
        if job.get("gpu_lock_mode", "wrapper") not in {"wrapper", "child"}:
            raise ValueError("gpu_lock_mode must be wrapper or child")
    # A cycle must be reported at registration rather than waiting forever.
    remaining = {j["job_id"]: set(j.get("dependencies", [])) for j in jobs}
    while remaining:
        ready = {key for key, deps in remaining.items() if not deps}
        if not ready:
            raise ValueError("dependency cycle")
        remaining = {key: deps - ready for key, deps in remaining.items() if key not in ready}


def verify_source(job):
    for name, expected in job["source_files"].items():
        if file_hash(Path(job["cwd"]) / name) != expected:
            raise ValueError(f"source snapshot changed: {name}")


def verify_success(job):
    """A zero exit status is insufficient: verify identity and actual artifact bytes."""
    receipt = read_json(job["success_receipt"])
    if not receipt or receipt.get("status") != "SUCCEEDED":
        raise ValueError("missing SUCCEEDED artifact receipt")
    identity = {**receipt.get("identity", {}), **receipt}
    for key in ("run_id", *HASH_FIELDS):
        if identity.get(key) != job[key]:
            raise ValueError(f"success receipt identity mismatch: {key}")
    if identity.get("job_id", job["job_id"]) != job["job_id"]:
        raise ValueError("success receipt job mismatch")
    artifacts = receipt.get("artifacts", [])
    if not artifacts:
        raise ValueError("success receipt has no artifacts")
    for artifact in artifacts:
        path = Path(artifact["path"])
        if not path.is_absolute() or file_hash(path) != artifact["sha256"]:
            raise ValueError("artifact hash mismatch")
    return {"receipt": job["success_receipt"], "receipt_sha256": file_hash(job["success_receipt"]),
            "artifact_count": len(artifacts), "verified_at": now()}


class Supervisor:
    def __init__(self, data_root):
        self.data_root = Path(data_root).resolve()
        self.root = self.data_root / "v6_v7"
        self.manifest_path = self.root / "manifest.json"
        self.state_path = self.root / "state.json"

    def initialize(self, manifest, legacy_observer=None):
        validate_manifest(manifest)
        with try_lock(self.root / "supervisor.lock") as lock:
            if lock is None:
                return {"status": "SUPERVISOR_BUSY"}
            old = read_json(self.manifest_path)
            if old is not None and canonical_hash(old) != canonical_hash(manifest):
                raise ValueError("registered manifest is immutable; refuse changed commands/identity")
            if old is not None and self.state_path.exists():
                return self.status()
            atomic_json(self.manifest_path, manifest)
            version = self.root / "manifests" / (canonical_hash(manifest) + ".json")
            atomic_json(version, manifest)
            state = {"schema_version": "medvlm.supervisor.v1", "campaign_id": manifest["campaign_id"],
                     "manifest_hash": canonical_hash(manifest), "manifest_path": str(version), "manifest_revisions": [], "scheduler_identity": manifest.get("scheduler_identity"),
                     "queue_owner": "legacy" if legacy_observer else "supervisor",
                     "legacy_observer": legacy_observer, "ownership_handoffs": 0,
                     "jobs": {j["job_id"]: {"status": "IMPLEMENTATION_PENDING" if j.get("implementation_pending") or not j.get("argv") else "PENDING",
                                "attempts": 0, "failure_count": 0, "recovery_attempts": 0, "history": []} for j in manifest["jobs"]},
                     "finalization": "NOT_DONE", "updated_at": now()}
            atomic_json(self.state_path, state)
            return self.status()

    def load(self):
        state = read_json(self.state_path)
        manifest = read_json(state.get("manifest_path", self.manifest_path)) if state else None
        if not manifest or not state:
            raise ValueError("initialize supervisor first")
        if state["manifest_hash"] != canonical_hash(manifest):
            raise ValueError("immutable manifest hash mismatch")
        return manifest, state

    def activate(self, revised):
        """Bind newly implemented work, preserving every started job and scientific ID.

        Version files are written before the single atomic state-pointer update, so
        interruption cannot leave the queue bound to a half-installed manifest.
        manifest.json remains the original registration/audit record; load() follows
        state.manifest_path to the active immutable version.
        """
        validate_manifest(revised)
        with try_lock(self.root / "supervisor.lock") as lock:
            if lock is None:
                return {"status": "SUPERVISOR_BUSY"}
            current, state = self.load()
            if {k: v for k, v in revised.items() if k != "jobs"} != {k: v for k, v in current.items() if k != "jobs"}:
                raise ValueError("activation cannot change campaign metadata")
            if [j["job_id"] for j in revised["jobs"]] != [j["job_id"] for j in current["jobs"]]:
                raise ValueError("activation cannot add, remove, or reorder jobs")
            allowed = {"argv", "cwd", "source_files", *HASH_FIELDS, "success_receipt",
                       "implementation_pending", "uses_gpu", "gpu_lock_mode"}
            activated = []
            for before, after in zip(current["jobs"], revised["jobs"]):
                if before == after:
                    continue
                record = state["jobs"][before["job_id"]]
                if record["attempts"] or record["status"] != "IMPLEMENTATION_PENDING":
                    raise ValueError("activation only changes never-started implementation-pending jobs")
                if not (before.get("implementation_pending") or not before.get("argv")):
                    raise ValueError("activation cannot replace an already registered executable job")
                changed = {key for key in set(before) | set(after) if before.get(key) != after.get(key)}
                if changed - allowed:
                    raise ValueError("activation cannot change scientific IDs, dependencies, or protocol")
                if after.get("implementation_pending") or not after.get("argv"):
                    raise ValueError("activation must install an executable job")
                verify_source(after)
                activated.append(before["job_id"])
            if not activated:
                return self.status()
            digest = canonical_hash(revised)
            version = self.root / "manifests" / (digest + ".json")
            atomic_json(version, revised)
            state["manifest_revisions"].append({"previous_hash": state["manifest_hash"],
                "manifest_hash": digest, "activated_jobs": activated, "at": now()})
            state.update(manifest_hash=digest, manifest_path=str(version), updated_at=now())
            for job_id in activated:
                state["jobs"][job_id]["status"] = "PENDING"
            atomic_json(self.state_path, state)
            return self.status()

    def status(self):
        manifest, state = self.load()
        # Status reads worker-owned health without rewriting it.
        output = {**state, "pending_cells": [], "terminal_cells": [], "workers": []}
        for job in manifest["jobs"]:
            record = state["jobs"][job["job_id"]]
            (output["terminal_cells"] if record["status"] in TERMINAL else output["pending_cells"]).append(job["cell_id"])
            if record.get("attempt_dir"):
                attempt = Path(record["attempt_dir"])
                output["workers"].append({"job_id": job["job_id"], "phase": job["phase"],
                    "cell_id": job["cell_id"], "run_id": job["run_id"],
                    **{key: job.get(key) for key in HASH_FIELDS},
                    **(read_json(attempt / "heartbeat.json", {}) or {}),
                    "exit": read_json(attempt / "exit.json")})
        output["all_terminal"] = all(r["status"] in TERMINAL for r in state["jobs"].values())
        output["terminal_state_hash"] = terminal_state_hash(state)
        output["ready_to_stop_scheduler"] = output["all_terminal"] and state["finalization"] == "VERIFIED"
        return output

    def handoff(self):
        with try_lock(self.root / "supervisor.lock") as lock:
            if lock is None:
                return {"status": "SUPERVISOR_BUSY"}
            _, state = self.load()
            if state["queue_owner"] == "supervisor":
                return self.status()
            legacy = state.get("legacy_observer") or {}
            if same_process(legacy) or any(same_process(x) for x in legacy.get("worker_identities", [])):
                raise ValueError("legacy queue owner/worker still alive; observation only")
            with try_lock(self.data_root / "runtime" / "medvlm_gpu.lock") as gpu:
                if gpu is None:
                    raise ValueError("GPU lease still held; cannot hand off")
                state.update(queue_owner="supervisor", ownership_handoffs=state["ownership_handoffs"] + 1,
                             handoff_at=now(), updated_at=now())
                atomic_json(self.state_path, state)
            return self.status()

    def _failure(self, record, reason, exit_code=None):
        record["failure_count"] += 1
        record["status"] = ("FAILED" if record["failure_count"] >= MAX_FAILURES else
                            "ANALYSIS_REQUIRED" if record["failure_count"] == 2 else "PENDING")
        record["error_signature"] = canonical_hash({"reason": reason, "exit_code": exit_code})
        record["error"] = reason
        record["exit_code"] = exit_code
        record["history"].append({"attempt": record["attempts"], "status": "FAILED", "reason": reason,
                                  "exit_code": exit_code, "at": now()})

    def authorize_recovery(self, job_id, receipt_path):
        """After two failures, require recorded stronger-model analysis before retry 3."""
        with try_lock(self.root / "supervisor.lock") as lock:
            if lock is None:
                return {"status": "SUPERVISOR_BUSY"}
            manifest, state = self.load()
            job = next((j for j in manifest["jobs"] if j["job_id"] == job_id), None)
            if job is None:
                raise ValueError("unknown recovery job")
            record = state["jobs"][job_id]
            receipt = read_json(receipt_path, {})
            digest = file_hash(receipt_path)
            if record.get("recovery_authorization_hash") == digest:
                return self.status()
            if record["status"] != "ANALYSIS_REQUIRED" or record["failure_count"] != 2:
                raise ValueError("recovery authorization requires the second-failure analysis hold")
            expected = {"job_id": job_id, "run_id": job["run_id"], "job_hash": canonical_hash(job),
                        "failure_count": 2, "error_signature": record["error_signature"]}
            if any(receipt.get(key) != value for key, value in expected.items()):
                raise ValueError("analysis receipt does not bind the failed job identity")
            if receipt.get("actual_model") != "gpt-6-astra" or receipt.get("reasoning_effort") not in {"low", "medium", "high", "xhigh", "max", "ultra"}:
                raise ValueError("actual stronger-model analysis requires GPT-6 Astra and a supported reasoning effort")
            if any(not str(receipt.get(key, "")).strip() for key in
                   ("original_model", "original_reasoning_effort", "escalation_reason")):
                raise ValueError("record original model/effort and escalation reason")
            if receipt.get("retry_authorized") is not True or not str(receipt.get("analysis_summary", "")).strip():
                raise ValueError("analysis must explain and explicitly authorize the remaining recovery")
            evidence = receipt.get("model_execution_receipt", {})
            if not Path(evidence.get("path", "")).is_absolute() or file_hash(evidence["path"]) != evidence.get("sha256"):
                raise ValueError("actual model execution evidence is missing or changed")
            record.update(status="PENDING", recovery_authorization_hash=digest,
                          recovery_authorization_receipt=str(Path(receipt_path).resolve()))
            record["history"].append({"status": "RECOVERY_AUTHORIZED", "at": now(),
                "analysis_receipt_hash": digest, "actual_model": receipt["actual_model"],
                "reasoning_effort": receipt["reasoning_effort"], "original_model": receipt["original_model"],
                "original_reasoning_effort": receipt["original_reasoning_effort"],
                "escalation_reason": receipt["escalation_reason"], "failure_count": record["failure_count"]})
            state["updated_at"] = now()
            atomic_json(self.state_path, state)
            return self.status()

    def tick(self):
        with try_lock(self.root / "supervisor.lock") as lock:
            if lock is None:
                return {"status": "SUPERVISOR_BUSY"}
            manifest, state = self.load()
            if state["queue_owner"] == "legacy":
                state.update(legacy_alive=same_process(state.get("legacy_observer")), updated_at=now())
                atomic_json(self.state_path, state)
                return self.status()
            # Check every worker before queue transitions. A stale heartbeat is never a stop signal.
            for job in manifest["jobs"]:
                record = state["jobs"][job["job_id"]]
                if record["status"] not in {"STARTING", "RUNNING"}:
                    continue
                attempt = Path(record["attempt_dir"])
                health = read_json(attempt / "heartbeat.json", {})
                exited = read_json(attempt / "exit.json")
                identity = health.get("worker_identity") or record.get("worker_identity")
                if same_process(identity) or same_process(health.get("child_identity")):
                    record["status"] = "RUNNING"
                    record["worker_identity"] = identity
                    state["updated_at"] = now()
                    atomic_json(self.state_path, state)
                    return self.status()
                # Retained fd also covers launcher/receipt races and orphaned children.
                with try_lock(self.root / "worker.lock") as worker_lock:
                    if worker_lock is None:
                        state["updated_at"] = now()
                        atomic_json(self.state_path, state)
                        return self.status()
                if exited and exited.get("launch_token") != record["launch_token"]:
                    self._failure(record, "exit receipt launch identity mismatch")
                elif exited and exited.get("exit_code") == 0:
                    try:
                        record["verification"] = verify_success(job)
                        record.update(status="SUCCEEDED", exit_code=0)
                        record["history"].append({"attempt": record["attempts"], "status": "SUCCEEDED", "at": now()})
                    except (ValueError, OSError, KeyError, TypeError) as exc:
                        self._failure(record, str(exc), 0)
                else:
                    self._failure(record, (exited or {}).get("error", "worker exited without success receipt"), (exited or {}).get("exit_code"))
            for job in manifest["jobs"]:
                record = state["jobs"][job["job_id"]]
                if record["status"] != "PENDING":
                    continue
                deps = [state["jobs"][key]["status"] for key in job.get("dependencies", [])]
                if job.get("dependency_policy") == "terminal":
                    ready = all(value in TERMINAL for value in deps)
                else:
                    if any(value in {"FAILED", "BLOCKED_PREREQUISITE"} for value in deps):
                        record.update(status="BLOCKED_PREREQUISITE", error="required dependency failed")
                        continue
                    ready = all(value == "SUCCEEDED" for value in deps)
                if not ready:
                    continue
                with try_lock(self.root / "worker.lock") as worker_lock:
                    if worker_lock is None:
                        break
                    with contextlib.ExitStack() as stack:
                        gpu = None
                        if job.get("uses_gpu", True):
                            gpu = stack.enter_context(try_lock(self.data_root / "runtime" / "medvlm_gpu.lock"))
                            if gpu is None:
                                state["waiting_reason"] = "GPU_LEASE_BUSY"
                                break
                        try:
                            verify_source(job)
                        except (ValueError, OSError) as exc:
                            # Configuration corruption is a block, not a synthetic training attempt.
                            record.update(status="IMPLEMENTATION_PENDING", error=str(exc))
                            continue
                        record["attempts"] += 1
                        record["recovery_attempts"] = record["failure_count"]
                        attempt = self.root / "attempts" / job["job_id"] / f"{record['attempts']:04d}"
                        token = uuid.uuid4().hex
                        record.update(status="STARTING", attempt_dir=str(attempt), launch_token=token)
                        launch = {"job": job, "job_hash": canonical_hash(job), "launch_token": token,
                                  "attempt": record["attempts"], "campaign_id": manifest["campaign_id"],
                                  "data_root": str(self.data_root), "created_at": now()}
                        atomic_json(attempt / "launch.json", launch)
                        state["updated_at"] = now()
                        atomic_json(self.state_path, state)  # Intent precedes detached spawn.
                        env = dict(os.environ)
                        env["PYTHONPATH"] = str(Path(job["cwd"]) / "src")
                        inherited = [worker_lock.fileno()]
                        worker_fd = str(worker_lock.fileno())
                        gpu_fd = ""
                        if gpu and job.get("gpu_lock_mode", "wrapper") == "wrapper":
                            inherited.append(gpu.fileno())
                            gpu_fd = str(gpu.fileno())
                        elif gpu:
                            # Existing training runner atomically obtains its own GPULease.
                            stack.close()
                        try:
                            with (attempt / "worker.log").open("ab") as log:
                                process = subprocess.Popen([sys.executable, "-m", "mediflow_medvlm.campaign_supervisor", "--data-root", str(self.data_root),
                                    "_worker", "--launch", str(attempt / "launch.json"), "--worker-fd", worker_fd,
                                    "--gpu-fd", gpu_fd], cwd=job["cwd"], env=env, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=log, close_fds=True, pass_fds=tuple(inherited), start_new_session=True)
                            record["worker_identity"] = process_identity(process.pid)
                            record["status"] = "RUNNING"
                        except OSError as exc:
                            self._failure(record, f"worker launch failed: {exc}")
                        # Do not LOCK_UN an inherited open-file description: closing parent
                        # copies leaves the worker's lock intact. ExitStack must not unlock it.
                        if gpu and gpu_fd:
                            gpu.close()
                        worker_lock.close()
                        state["updated_at"] = now()
                        atomic_json(self.state_path, state)
                        return self.status()
            state["updated_at"] = now()
            atomic_json(self.state_path, state)
            return self.status()

    def finalize(self, receipt_path):
        with try_lock(self.root / "supervisor.lock") as lock:
            if lock is None:
                return {"status": "SUPERVISOR_BUSY"}
            manifest, state = self.load()
            if not all(value["status"] in TERMINAL for value in state["jobs"].values()):
                raise ValueError("campaign jobs are not all terminal")
            receipt = read_json(receipt_path, {})
            if receipt.get("campaign_id") != state["campaign_id"] or receipt.get("status") != "SUCCEEDED":
                raise ValueError("final report/CI receipt identity mismatch")
            if receipt.get("reports_verified") is not True or receipt.get("ci_verified") is not True:
                raise ValueError("final report and CI verification required")
            if receipt.get("manifest_hash") != state["manifest_hash"]:
                raise ValueError("finalization manifest hash is stale or missing")
            if receipt.get("terminal_state_hash") != terminal_state_hash(state):
                raise ValueError("finalization terminal state hash is stale or missing")
            failed_reports = [job["job_id"] for job in manifest["jobs"]
                              if (job.get("report_job") or "REPORT" in job["phase"].upper())
                              and state["jobs"][job["job_id"]]["status"] != "SUCCEEDED"]
            evidence_keys = ["report_artifacts", "ci_artifacts"]
            if failed_reports:
                if not receipt.get("alternate_report_artifacts"):
                    raise ValueError("report jobs did not succeed; alternate report artifact evidence required")
                evidence_keys.append("alternate_report_artifacts")
            for key in evidence_keys:
                if not receipt.get(key):
                    raise ValueError(f"missing {key}")
                for artifact in receipt[key]:
                    if file_hash(artifact["path"]) != artifact["sha256"]:
                        raise ValueError("finalization artifact changed")
            state.update(finalization="VERIFIED", finalization_receipt=str(receipt_path),
                         finalization_receipt_hash=file_hash(receipt_path), updated_at=now())
            atomic_json(self.state_path, state)
            return self.status()


def run_worker(launch_path, worker_fd, gpu_fd=None):
    """One allocated job; never advances the queue or decides recovery."""
    launch = read_json(launch_path)
    job, attempt = launch["job"], Path(launch_path).parent
    identity = process_identity()
    env = dict(os.environ)
    values = {"JOB_ID": job["job_id"], "RUN_ID": job["run_id"], "ATTEMPT": str(launch["attempt"]),
              "JOB_HASH": launch["job_hash"], **{key.upper(): job[key] for key in HASH_FIELDS}}
    env.update({"MEDVLM_SUPERVISOR_" + key: value for key, value in values.items()})
    env["MEDVLM_SUPERVISOR_PROGRESS_PATH"] = str(attempt / "progress.json")
    health = {"worker_identity": identity, **identity, "launch_token": launch["launch_token"],
              "heartbeat_at": now(), "last_progress_at": None, "optimizer_step": None,
              "total_updates": None, "last_complete_checkpoint": None}
    error, code, child = None, None, None
    try:
        atomic_json(attempt / "heartbeat.json", health)
        verify_source(job)
        fds = tuple(int(fd) for fd in (worker_fd, gpu_fd) if fd not in (None, ""))
        with (attempt / "stdout.log").open("ab") as stdout, (attempt / "stderr.log").open("ab") as stderr:
            child = subprocess.Popen(job["argv"], cwd=job["cwd"], env=env, stdin=subprocess.DEVNULL,
                                     stdout=stdout, stderr=stderr, pass_fds=fds)
            health["child_identity"] = process_identity(child.pid)
            while True:
                # Optional progress file is owned by the job, never fabricated from heartbeat.
                try:
                    progress = read_json(attempt / "progress.json", {})
                    if not isinstance(progress, dict):
                        progress = {}
                except (ValueError, OSError):
                    progress = {}  # A partial optional progress record cannot stop a live job.
                for key in ("last_progress_at", "optimizer_step", "total_updates", "last_complete_checkpoint"):
                    if key in progress:
                        health[key] = progress[key]
                health["heartbeat_at"] = now()
                atomic_json(attempt / "heartbeat.json", health)
                try:
                    code = child.wait(timeout=5)
                    break
                except subprocess.TimeoutExpired:
                    pass
    except Exception as exc:
        error, code = f"{type(exc).__name__}: {exc}", 1
    atomic_json(attempt / "exit.json", {"status": "EXITED",
                "launch_token": launch["launch_token"], "worker_identity": identity,
                "child_identity": health.get("child_identity"), "exit_code": code,
                "error": error, "exited_at": now()})
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--manifest", type=Path, required=True)
    init.add_argument("--legacy-observer", type=Path)
    activate = sub.add_parser("activate")
    activate.add_argument("--manifest", type=Path, required=True)
    sub.add_parser("status")
    sub.add_parser("tick")
    sub.add_parser("handoff")
    authorize = sub.add_parser("authorize-recovery")
    authorize.add_argument("--job-id", required=True)
    authorize.add_argument("--analysis-receipt", type=Path, required=True)
    final = sub.add_parser("finalize")
    final.add_argument("--receipt", type=Path, required=True)
    worker = sub.add_parser("_worker")
    worker.add_argument("--launch", type=Path, required=True)
    worker.add_argument("--worker-fd", required=True)
    worker.add_argument("--gpu-fd")
    args = parser.parse_args(argv)
    if args.command == "_worker":
        return run_worker(args.launch, args.worker_fd, args.gpu_fd)
    supervisor = Supervisor(args.data_root)
    if args.command == "init":
        result = supervisor.initialize(read_json(args.manifest), read_json(args.legacy_observer) if args.legacy_observer else None)
    elif args.command == "authorize-recovery":
        result = supervisor.authorize_recovery(args.job_id, args.analysis_receipt)
    elif args.command == "activate":
        result = supervisor.activate(read_json(args.manifest))
    elif args.command == "finalize":
        result = supervisor.finalize(args.receipt)
    else:
        result = getattr(supervisor, args.command)()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
