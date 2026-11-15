"""Durable scalar telemetry. No torch import, CUDA initialization, network or RNG.

The coordinator alone owns events and snapshots; its training child alone owns
steps/resources. A stable persisted attempt UUID joins them without shared writers.
Failures are isolated and recorded in a private health sidecar when storage permits.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone

SCHEMA = "mediflow.telemetry.v1"
PRODUCER = "mediflow.campaign"
SAFE_ID = re.compile(r"[A-Za-z0-9_-]{1,100}\Z")
STEP_FIELDS = tuple("optimizer_step total_updates batch_examples samples_processed samples_total train_loss loss_objective lr_used lr_next gradient_norm_before_clip supervised_tokens supervised_tokens_cumulative sequence_tokens sequence_tokens_cumulative step_wall_seconds step_time_scope samples_per_second supervised_tokens_per_second sequence_tokens_per_second cuda_allocated_bytes cuda_reserved_bytes cuda_peak_allocated_bytes cuda_peak_reserved_bytes finite availability".split())
GPU_FIELDS = tuple("gpu_uuid gpu_index gpu_utilization_percent gpu_memory_used_bytes gpu_memory_total_bytes gpu_temperature_celsius gpu_power_watts gpu_sm_clock_mhz gpu_memory_clock_mhz".split())
EVENT_PHASE = {"model_loading": "loading", "training_started": "training", "optimizer_update": "training", "checkpoint_saved": "saving", "reload_started": "reloading", "reload_succeeded": "reloading", "evaluation_started": "evaluating", "evaluation_completed": "evaluating", "succeeded": "succeeded", "failed": "failed", "recovery": "recovering", "hold": "held", "telemetry_gap": "unknown", "telemetry_error": "unknown"}


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp-" + str(os.getpid()) + "-" + str(threading.get_ident()))
    with temporary.open("w", encoding="utf-8") as stream:
        os.chmod(temporary, 0o600)
        stream.write(canonical(value) + "\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def missing(reason="not_measured"):
    return {"available": False, "reason": reason, "provenance": "unmeasured"}


def normalize(value):
    if isinstance(value, float) and not math.isfinite(value): return None
    if isinstance(value, dict): return {k: normalize(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [normalize(v) for v in value]
    return value


def safe_component(value):
    value = str(value)
    return value if SAFE_ID.fullmatch(value) else "id-" + hashlib.sha256(value.encode()).hexdigest()[:32]


def manifest_for(config, attempt_id, campaign_id):
    model = config.get("model_id") or config.get("model", {}).get("model_id")
    data, recipe = config.get("data", {}), config.get("recipe", {})
    code_sha = os.environ.get("MEDVLM_TELEMETRY_CODE_SHA")
    if code_sha and re.fullmatch(r"[a-f0-9]{40}", code_sha) is None: code_sha = None
    return {"schema_version": SCHEMA, "telemetry_version": SCHEMA, "campaign_id": campaign_id,
        "job_id": config.get("job_id", config.get("run_id", "unknown")), "run_id": config.get("run_id", "unknown"),
        "attempt_id": attempt_id, "source": "mediflow_live", "producer": PRODUCER,
        "model": model, "task": config.get("task"), "seed": recipe.get("seed"), "data_hash": config.get("data_hash"),
        "scientific_recipe_hash": digest(recipe), "code_sha": code_sha,
        "eval_protocol": None, "precision": recipe.get("compute_dtype", "bfloat16"), "parent_run_id": None,
        "freeze_id": data.get("freeze_id"), "dataset_ids": (["v6-common-" + config["task"]] if data.get("freeze_id") else []),
        "total_updates": recipe.get("optimizer_steps"), "samples_total": recipe.get("examples"),
        "created_at": utcnow(), "historical": False, "artifacts": [], "source_artifacts": [], "cohort_id": data.get("freeze_id")}


class TelemetryWriter:
    """One owner for each stream, synchronous flush on every line, bounded metadata.

    Public hook methods never propagate logging errors into training. No background
    queue/drop policy is needed: sampler runs independently; updates use local append.
    """
    def __init__(self, directory=None, manifest=None, *, role="lifecycle", fsync_seconds=5):
        self.directory = Path(directory) if directory else None
        self.manifest = dict(manifest or {})
        self.role, self.fsync_seconds = role, max(0., float(fsync_seconds))
        self.seqs = {s: 0 for s in ("steps", "resources", "events")}
        self.streams, self.last_sync = {}, {}
        self.error, self.dropped, self.gaps = None, 0, []
        self.write_seconds, self.records_written = 0., 0
        self.stop_event, self.thread = threading.Event(), None
        self.observer_stop, self.observer = threading.Event(), None
        self.status = {"schema_version": SCHEMA, "status": "running", "phase": "loading", "optimizer_step": 0,
            "total_updates": self.manifest.get("total_updates"), "samples_processed": 0,
            "samples_total": self.manifest.get("samples_total"), "last_emitted_at": None,
            "training_succeeded": None, "reload_succeeded": None, "evaluation_completed": None,
            "telemetry_complete": False, "telemetry_error": None, "dropped_records": 0, "gaps": [], "writer_health": "ok"}
        if self.directory:
            try:
                self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
                if shutil.disk_usage(self.directory).free < 64 * 1024 * 1024:
                    raise OSError("LOW_DISK_SPACE")
                if role == "lifecycle":
                    if (self.directory / "manifest.json").exists(): raise ValueError("ATTEMPT_ALREADY_EXISTS")
                    atomic_json(self.directory / "manifest.json", self.manifest)
                    self.snapshot()
                else:
                    original = json.loads((self.directory / "manifest.json").read_text())
                    for key in ("campaign_id", "run_id", "attempt_id", "producer"):
                        if original[key] != self.manifest[key]: raise ValueError("ATTEMPT_IDENTITY_CHANGED")
                    self.manifest = original
            except Exception as exc: self._failure("low_disk_space" if isinstance(exc, OSError) and str(exc) == "LOW_DISK_SPACE" else "initialization_failed")

    @property
    def enabled(self): return self.directory is not None and self.error is None

    def _failure(self, reason, stream=None, seq=None):
        self.error = reason; self.dropped += 1
        if stream: self.gaps.append({"stream": stream, "from_seq": seq, "to_seq": seq, "reason": reason})
        if self.directory:
            try: atomic_json(self.directory / (self.role + ".health.json"), self.health())
            except Exception: pass

    def health(self):
        return {"writer_health": "degraded" if self.error else "ok", "telemetry_error": self.error,
            "dropped_records": self.dropped, "gaps": self.gaps[-100:], "write_seconds": self.write_seconds,
            "records_written": self.records_written, "telemetry_complete": self.error is None}

    def envelope(self, stream, phase, *, emitted_at=None, source_artifact=None):
        now = utcnow()
        historical = self.manifest.get("historical", False)
        return {**{k: self.manifest.get(k) for k in ("schema_version", "campaign_id", "job_id", "run_id", "attempt_id", "source", "producer", "model", "task")},
            "seq": self.seqs[stream], "emitted_at": emitted_at if historical else now,
            "monotonic_ns": None if historical else time.monotonic_ns(), "observed_at": now, "phase": phase,
            "provenance": "historical_reconstructed" if historical else "live_measured", "source_artifact": source_artifact}

    def append(self, stream, payload, phase, *, emitted_at=None, source_artifact=None):
        if self.directory is None: return
        if self.role == "training" and stream not in {"steps", "resources"}: return
        if self.role == "lifecycle" and stream != "events": return
        self.seqs[stream] += 1
        if self.error:
            self.dropped += 1; return
        started = time.monotonic()
        try:
            record = normalize({**self.envelope(stream, phase, emitted_at=emitted_at, source_artifact=source_artifact), **payload})
            if stream not in self.streams:
                path = self.directory / (stream + ".jsonl")
                # Exclusive creation prevents accidental second-owner seq reuse.
                self.streams[stream] = path.open("x", encoding="utf-8")
                os.chmod(path, 0o600)
            handle = self.streams[stream]
            handle.write(canonical(record) + "\n"); handle.flush()
            now = time.monotonic()
            if now - self.last_sync.get(stream, 0) >= self.fsync_seconds:
                os.fsync(handle.fileno()); self.last_sync[stream] = now
            self.records_written += 1
        except Exception: self._failure("append_or_flush_failed", stream, self.seqs[stream])
        finally: self.write_seconds += time.monotonic() - started

    def step(self, payload):
        self.status["phase"] = "training"
        values = {k: payload.get(k) for k in STEP_FIELDS}
        availability = dict(values.get("availability") or {})
        for key, value in values.items():
            if key != "availability" and (value is None or isinstance(value, float) and not math.isfinite(value)):
                availability.setdefault(key, missing("nonfinite" if value is not None else "not_measured"))
        values["availability"] = availability
        self.append("steps", values, "training")

    def event(self, event, *, step=None, details=None):
        if event not in EVENT_PHASE: return
        allowed = {"optimizer_steps", "samples_processed", "receipt_verified", "training_succeeded", "reload_succeeded"}
        clean = {k: v for k, v in (details or {}).items() if k in allowed and type(v) in (int, float, bool, type(None))}
        self.append("events", {"event": event, "message": None, "optimizer_step": step, "details": clean}, EVENT_PHASE[event])
        self.status.update(phase=EVENT_PHASE[event], last_emitted_at=utcnow())
        if event == "succeeded": self.status["status"] = "succeeded"
        elif event == "failed": self.status["status"] = "failed"
        elif event == "hold": self.status["status"] = "held"
        self.snapshot()

    def snapshot(self, **fields):
        if self.directory is None or self.role != "lifecycle": return
        self.status.update(fields)
        try:
            health = self.health()
            child = self.directory / "training.health.json"
            if child.exists():
                child_health = json.loads(child.read_text())
                if child_health.get("writer_health") != "ok":
                    health.update(writer_health="degraded", telemetry_complete=False,
                        telemetry_error=child_health.get("telemetry_error"),
                        dropped_records=health["dropped_records"] + child_health.get("dropped_records", 0),
                        gaps=health["gaps"] + child_health.get("gaps", []))
            self.status.update({k: health[k] for k in ("writer_health", "telemetry_error", "dropped_records", "gaps")})
            if health["writer_health"] != "ok": self.status["telemetry_complete"] = False
            atomic_json(self.directory / "status.json", self.status)
        except Exception: self._failure("status_write_failed")

    def record_artifacts(self, directory, *, reload_verified=False):
        if not self.enabled or self.role != "lifecycle": return
        try:
            artifacts = []
            for name, kind in (("checkpoint.json", "checkpoint"), ("final_selection.json", "selection"), ("reload.json", "reload")):
                path = Path(directory) / name
                if path.is_file():
                    artifacts.append({"artifact_id": name, "kind": kind, "sha256": file_hash(path),
                        "exists": True, "verified": True, "reload_verified": reload_verified,
                        "selection": "final" if name == "final_selection.json" else None})
            self.manifest["artifacts"] = artifacts
            atomic_json(self.directory / "manifest.json", self.manifest)
        except Exception: self._failure("artifact_manifest_write_failed"); self.snapshot()

    def finish(self, metrics, *, succeeded=True, reload_succeeded=None):
        if self.directory is None: return
        try:
            records = []
            step_path = self.directory / "steps.jsonl"
            if step_path.exists():
                with step_path.open() as stream: records = [json.loads(line) for line in stream]
            checks = reconcile(records, metrics)
            resource = metrics.get("resource", {})
            complete = checks["status"] == "passed" and self.error is None
            child = self.directory / "training.health.json"
            if not child.exists() or json.loads(child.read_text()).get("writer_health") != "ok": complete = False
            summary = {"train_loss_mean": metrics.get("loss_mean"), "optimizer_steps": metrics.get("optimizer_steps"),
                "samples_processed": metrics.get("samples_processed"), "supervised_tokens": metrics.get("supervised_tokens"),
                "sequence_tokens": metrics.get("sequence_tokens"), "wall_seconds": resource.get("wall_seconds"),
                "wall_time_scope": "model_load_audit_training_verify_save_and_final_sync",
                "peak_allocated_bytes": resource.get("peak_allocated_bytes"), "peak_reserved_bytes": resource.get("peak_reserved_bytes"),
                "reconciliation": checks, "telemetry_complete": complete}
            atomic_json(self.directory / "summary.json", normalize(summary))
            self.snapshot(training_succeeded=succeeded, reload_succeeded=reload_succeeded,
                optimizer_step=metrics.get("optimizer_steps"), samples_processed=metrics.get("samples_processed"), telemetry_complete=complete)
            self.event("succeeded" if succeeded and reload_succeeded is True else "failed")
        except Exception: self._failure("reconciliation_or_summary_failed"); self.snapshot(telemetry_complete=False)

    def observe_steps(self):
        """Coordinator observes child-flushed records; never touches the trainer.

        Event timestamps denote coordinator observation, not reconstructed GPU time.
        This is the sole events/status writer throughout load, train, save, reload.
        """
        if not self.enabled or self.role != "lifecycle": return
        def run():
            offset, started = 0, False
            while True:
                try:
                    path = self.directory / "steps.jsonl"
                    if path.exists():
                        with path.open("rb") as stream:
                            stream.seek(offset)
                            while True:
                                line = stream.readline()
                                if not line or not line.endswith(b"\n"): break
                                record = json.loads(line)
                                offset = stream.tell()
                                if not started:
                                    self.event("training_started"); started = True
                                self.event("optimizer_update", step=record["optimizer_step"])
                                self.snapshot(phase="training", optimizer_step=record["optimizer_step"],
                                    samples_processed=record["samples_processed"])
                except Exception: self._failure("step_observer_failed"); return
                if self.observer_stop.wait(.25):
                    # One final complete-line read after the child has exited.
                    try:
                        if path.exists() and path.stat().st_size > offset:
                            with path.open("rb") as stream:
                                stream.seek(offset)
                                for line in stream:
                                    if not line.endswith(b"\n"): break
                                    record = json.loads(line)
                                    if not started: self.event("training_started"); started = True
                                    self.event("optimizer_update", step=record["optimizer_step"])
                                    self.snapshot(phase="training", optimizer_step=record["optimizer_step"],
                                        samples_processed=record["samples_processed"])
                    except Exception: self._failure("step_observer_failed")
                    return
        self.observer = threading.Thread(target=run, name="mediflow-step-observer", daemon=True)
        try: self.observer.start()
        except Exception: self.observer = None; self._failure("step_observer_start_failed")

    def stop_observer(self):
        self.observer_stop.set()
        if self.observer:
            self.observer.join(timeout=5)
            if self.observer.is_alive(): self._failure("step_observer_stop_timeout")

    def start_sampler(self, interval=5):
        if not self.enabled or self.role != "training": return
        interval = max(.1, float(interval))
        def run():
            last, failures, last_cpu = None, 0, None
            while not self.stop_event.is_set():
                started = time.monotonic()
                try:
                    record = sample_resources()
                    cpu = time.process_time()
                    if last_cpu is not None and last is not None and started > last:
                        record["cpu_percent"] = 100 * (cpu - last_cpu) / (started - last)
                        record["availability"].pop("cpu_percent", None)
                    last_cpu = cpu
                    failures = failures + 1 if record["tool_status"] != "ok" else 0
                    record.update(sample_interval_seconds=interval, sample_gap_seconds=None if last is None else started - last)
                    self.append("resources", record, self.status["phase"])
                except Exception: self._failure("resource_sampler_failed", "resources", self.seqs["resources"] + 1)
                last = started
                if self.stop_event.wait(min(60., interval * 2 ** min(failures, 3))): break
        self.thread = threading.Thread(target=run, name="mediflow-resource-sampler", daemon=True)
        try: self.thread.start()
        except Exception: self.thread = None; self._failure("resource_sampler_start_failed")

    def close(self):
        self.stop_observer()
        self.stop_event.set()
        if self.thread: self.thread.join(timeout=4)
        for stream in self.streams.values():
            if stream.closed: continue
            try: stream.flush(); os.fsync(stream.fileno()); stream.close()
            except Exception: self._failure("final_flush_failed")
        if self.directory:
            try: atomic_json(self.directory / (self.role + ".health.json"), self.health())
            except Exception: self._failure("health_write_failed")


def attempt_writer(config, staging, *, role="lifecycle"):
    """Opt in by private root env; config science identity and locks are untouched."""
    root = os.environ.get("MEDVLM_TELEMETRY_ROOT")
    if not root: return TelemetryWriter()
    try:
        root = Path(root)
        if not root.is_absolute(): raise ValueError("ABSOLUTE_ROOT_REQUIRED")
        attempt = json.loads((Path(staging) / "attempt.json").read_text())["attempt_id"]
        campaign = os.environ.get("MEDVLM_TELEMETRY_CAMPAIGN", "mediflow-v6")
        if not SAFE_ID.fullmatch(campaign): raise ValueError("SAFE_CAMPAIGN_ID_REQUIRED")
        manifest = manifest_for(config, attempt, campaign)
        return TelemetryWriter(root / safe_component(campaign) / safe_component(manifest["run_id"]) / safe_component(attempt),
            manifest, role=role, fsync_seconds=float(os.environ.get("MEDVLM_TELEMETRY_FSYNC_SECONDS", "5")))
    except Exception:
        result = TelemetryWriter(); result.error = "initialization_failed"
        # Staging is private and exists before any hook; no raw exception disclosure.
        try: atomic_json(Path(staging) / "telemetry.health.json", result.health())
        except Exception: pass
        return result


def allocator_metrics(torch):
    fields = dict.fromkeys(("cuda_allocated_bytes", "cuda_reserved_bytes", "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"))
    # is_initialized does not create a context; never use is_available from sampler.
    try:
        if torch.cuda.is_initialized():
            for name, method in zip(fields, ("memory_allocated", "memory_reserved", "max_memory_allocated", "max_memory_reserved")):
                fields[name] = int(getattr(torch.cuda, method)())
    except Exception: pass
    return fields


def update_payload(*, step, total, losses, tokens, sequences, samples, samples_total, lr_used, lr_next, norm, wall, allocator=None):
    count = len(losses)
    record = {"optimizer_step": step, "total_updates": total, "batch_examples": count, "samples_processed": samples,
        "samples_total": samples_total, "train_loss": sum(losses) / count, "loss_objective": "example_mean_of_target_token_mean",
        "lr_used": lr_used, "lr_next": lr_next, "gradient_norm_before_clip": norm,
        "supervised_tokens": sum(tokens), "sequence_tokens": sum(sequences),
        "step_wall_seconds": wall, "step_time_scope": "update_host_wall_no_extra_cuda_sync",
        "finite": all(math.isfinite(x) for x in (*losses, norm, lr_used, lr_next)), "availability": {}}
    for name, amount in (("samples", count), ("supervised_tokens", sum(tokens)), ("sequence_tokens", sum(sequences))):
        key = name + "_per_second"
        record[key] = amount / wall if wall > 0 else None
        record["availability"][key] = {"available": wall > 0, "reason": None if wall > 0 else "invalid_wall_interval", "provenance": "derived"}
    record.update(allocator or {})
    return record


def reconcile(steps, metrics):
    def close(a, b): return a is not None and b is not None and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
    count = sum(x["batch_examples"] for x in steps)
    checks = {"optimizer_steps": len(steps) == metrics.get("optimizer_steps"),
        "step_order": [x["optimizer_step"] for x in steps] == list(range(1, len(steps) + 1)),
        "samples_processed": count == metrics.get("samples_processed"),
        "loss": bool(count) and close(sum(x["train_loss"] * x["batch_examples"] for x in steps) / count, metrics.get("loss_mean")),
        "supervised_tokens": sum(x.get("supervised_tokens") or 0 for x in steps) == metrics.get("supervised_tokens"),
        "sequence_tokens": all(x.get("sequence_tokens") is not None for x in steps) and sum(x["sequence_tokens"] for x in steps) == metrics.get("sequence_tokens"),
        "learning_rates": [x.get("lr_used") for x in steps] == metrics.get("learning_rates"),
        "gradient_norms": [x.get("gradient_norm_before_clip") for x in steps] == metrics.get("gradient_norms_before_clip"),
        "final_learning_rate": bool(steps) and close(steps[-1].get("lr_next"), metrics.get("final_learning_rate"))}
    return {"status": "passed" if all(checks.values()) else "failed", "checks": checks}


def sample_resources():
    """Read device-shared nvidia-smi and /proc; never import torch/NVML/CUDA."""
    record = {**dict.fromkeys(GPU_FIELDS), "process_id": os.getpid(), "cpu_percent": None,
        "host_ram_used_bytes": None, "host_ram_total_bytes": None, "process_rss_bytes": None,
        "attribution": "unavailable", "tool_status": "unavailable", "availability": {}}
    reason = "nvidia_smi_unavailable"
    try:
        query = "uuid,index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm,clocks.mem"
        result = subprocess.run(["nvidia-smi", "--query-gpu=" + query, "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2, check=False)
        if result.returncode == 0:
            rows = list(csv.reader(result.stdout.splitlines()))
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
            row = next((r for r in rows if len(r) == 9 and (r[1].strip() == visible or r[0].strip() == visible)), None)
            if row:
                record["gpu_uuid"] = row[0].strip()
                for key, value in zip(GPU_FIELDS[1:], row[1:]):
                    try: record[key] = float(value.strip())
                    except ValueError: record[key] = None
                for key in ("gpu_memory_used_bytes", "gpu_memory_total_bytes"):
                    if record[key] is not None: record[key] = int(record[key] * 1024 * 1024)
                for key in ("gpu_index", "gpu_sm_clock_mhz", "gpu_memory_clock_mhz"):
                    if record[key] is not None: record[key] = int(record[key])
                record.update(attribution="device_shared", tool_status="ok")
                reason = "sensor_unsupported"
            else: record["tool_status"] = "error"; reason = "visible_device_unresolved"
        else: record["tool_status"] = "error"; reason = "nvidia_smi_failed"
    except (OSError, subprocess.SubprocessError): pass
    try:
        values = {line.split(":")[0]: int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith(("MemTotal:", "MemAvailable:"))}
        record["host_ram_total_bytes"] = values["MemTotal"]
        record["host_ram_used_bytes"] = values["MemTotal"] - values["MemAvailable"]
        record["process_rss_bytes"] = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, KeyError): pass
    for key, value in record.items():
        if value is None: record["availability"][key] = missing("cpu_interval_not_measured" if key == "cpu_percent" else reason if key in GPU_FIELDS else "proc_unavailable")
    return record
