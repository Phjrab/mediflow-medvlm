"""CPU-only, immutable historical and frozen metadata export for Forge.

CLI never imports a training runtime, reads predictions, rewrites scientific
artifacts, infers missing GPU history, or modifies a queue. Output is a separate
integration-owned catalog. Only known scalar/aggregate/metadata fields are copied.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile

from . import telemetry as t

IMPORT_VERSION = "mediflow.history-mapping.v2"

FILES = ("config.json", "train.metrics.json", "metrics.json", "events.jsonl", "manifest.json", "run_manifest.json", "reload.json", "checkpoint.json", "final_selection.json", "completion.json")
AGGREGATE_FIELDS = set("scheduled attempted completed valid_answers correct exact_matches coverage answered_accuracy end_to_end_correct_fraction answered_lexical_exact_match end_to_end_lexical_exact_match lexical_token_f1 macro_f1 invalid_format_rate not_attempted_count technical_failure_rate truncated_count clinical_accuracy".split())
EVAL_LABELS = set("source_id task_type language subject".split())
METADATA_FIELDS = tuple("sample_id source_id source_revision group_ids split task_type language image_sha256 license_ref restrictions derivation review_status freeze_id".split())


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def logical(value):
    """IDs/controlled labels only; raw path-like or control text is never exported."""
    if not isinstance(value, str) or len(value) > 200 or value.startswith(("/", "~")) or "\\" in value or ".." in value or any(ord(c) < 32 for c in value): return None
    return value


def scalar(value):
    return value if value is None or type(value) in (int, bool) or type(value) is float and math.isfinite(value) else None


def availability(record):
    record.setdefault("availability", {})
    for key in t.STEP_FIELDS:
        if key != "availability" and record.get(key) is None: record["availability"].setdefault(key, t.missing("not_in_source_artifact"))
    return record


def reconstruct_v6(metrics):
    losses = metrics.get("example_losses")
    counts = metrics.get("update_example_counts")
    # Existing native smoke has an authoritative fixed 20 x 16 recipe. Main
    # training always requires recorded boundaries; never assume full batches.
    if counts is None and metrics.get("optimizer_steps") == 20 and metrics.get("samples_processed") == 320:
        counts = [16] * 20
    if not isinstance(losses, list) or not isinstance(counts, list) or any(type(n) is not int or n <= 0 for n in counts):
        raise ValueError("V6_UPDATE_BOUNDARIES_REQUIRED")
    if sum(counts) != len(losses) or len(counts) != metrics.get("optimizer_steps") or len(losses) != metrics.get("samples_processed"):
        raise ValueError("V6_LOSS_BOUNDARY_MISMATCH")
    if not all(type(x) in (int, float) and math.isfinite(x) for x in losses): raise ValueError("V6_NONFINITE_LOSS")
    arrays = {k: metrics.get(k) for k in ("learning_rates", "gradient_norms_before_clip", "example_supervised_tokens")}
    for key, values in arrays.items():
        expected = len(losses) if key == "example_supervised_tokens" else len(counts)
        if values is not None and (not isinstance(values, list) or len(values) != expected): raise ValueError("V6_SCALAR_LENGTH_MISMATCH")
        if values is not None:
            if key == "example_supervised_tokens":
                if any(type(x) is not int or x <= 0 for x in values): raise ValueError("V6_INVALID_TARGET_TOKEN_COUNT")
            elif any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in values):
                raise ValueError("V6_NONFINITE_OR_NEGATIVE_SCALAR")
    final_rate = metrics.get("final_learning_rate")
    if final_rate is not None and (type(final_rate) not in (int, float) or not math.isfinite(final_rate) or final_rate < 0):
        raise ValueError("V6_NONFINITE_OR_NEGATIVE_SCALAR")
    result, start, token_total = [], 0, 0
    for i, count in enumerate(counts):
        end = start + count
        token_values = arrays["example_supervised_tokens"]
        tokens = sum(token_values[start:end]) if token_values is not None else None
        if tokens is not None: token_total += tokens
        rates = arrays["learning_rates"]
        record = {k: None for k in t.STEP_FIELDS}
        record.update(optimizer_step=i + 1, total_updates=len(counts), batch_examples=count,
            samples_processed=end, samples_total=len(losses), train_loss=sum(losses[start:end]) / count,
            loss_objective="example_mean_of_target_token_mean", lr_used=rates[i] if rates is not None else None,
            lr_next=(rates[i + 1] if i + 1 < len(rates) else metrics.get("final_learning_rate")) if rates is not None else None,
            gradient_norm_before_clip=arrays["gradient_norms_before_clip"][i] if arrays["gradient_norms_before_clip"] is not None else None,
            supervised_tokens=tokens, supervised_tokens_cumulative=token_total if tokens is not None else None,
            finite=True, availability={})
        result.append(availability(record)); start = end
    if arrays["example_supervised_tokens"] is not None and token_total != metrics.get("supervised_tokens"):
        raise ValueError("V6_TARGET_TOKEN_TOTAL_MISMATCH")
    if not math.isclose(sum(losses) / len(losses), metrics.get("loss_mean", math.nan), rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("V6_OBJECTIVE_MISMATCH")
    return result


def legacy_steps(events, total_updates=None, samples_total=None):
    result, samples, tokens = [], 0, 0
    for event in events:
        if event.get("event") != "optimizer_update": continue
        count = event.get("samples")
        if type(count) is not int or count <= 0 or type(event.get("step")) is not int: raise ValueError("LEGACY_UPDATE_INVALID")
        samples += count
        target = event.get("target_tokens")
        if target is not None: tokens += target
        wall = event.get("seconds")
        record = dict.fromkeys(t.STEP_FIELDS)
        record.update(optimizer_step=event["step"], total_updates=total_updates, batch_examples=count,
            samples_processed=samples, samples_total=samples_total, train_loss=scalar(event.get("loss")),
            loss_objective="example_mean_of_target_token_mean", lr_used=None, lr_next=scalar(event.get("lr")),
            gradient_norm_before_clip=scalar(event.get("gradient_norm")), supervised_tokens=target,
            supervised_tokens_cumulative=tokens if target is not None else None, step_wall_seconds=wall,
            step_time_scope="update_host_wall_no_extra_cuda_sync" if wall is not None else None,
            finite=all(type(event.get(k)) in (int, float) and math.isfinite(event[k]) for k in ("loss", "gradient_norm")), availability={})
        record["availability"]["lr_used"] = t.missing("legacy_lr_recorded_after_scheduler_step")
        if type(wall) in (int, float) and wall > 0:
            record["samples_per_second"] = count / wall
            record["supervised_tokens_per_second"] = target / wall if target is not None else None
            for key in ("samples_per_second", "supervised_tokens_per_second"):
                if record[key] is not None: record["availability"][key] = {"available": True, "reason": None, "provenance": "derived"}
        result.append((availability(record), event.get("at")))
    return result


def aggregate_metrics(cohort):
    result = {key: scalar(value) for key, value in cohort.items() if key in AGGREGATE_FIELDS}
    result.update({key: logical(cohort.get(key)) for key in EVAL_LABELS if key in cohort})
    for key in ("statuses", "label_support"):
        values = cohort.get(key)
        if isinstance(values, dict): result[key] = {logical(k): scalar(v) for k, v in values.items() if logical(k) and type(v) in (int, float)}
    latency = cohort.get("e2e_latency_seconds")
    if isinstance(latency, dict):
        result["e2e_latency_seconds"] = {k: scalar(latency.get(k)) for k in ("n", "median", "min", "max")}
        result["e2e_latency_seconds"]["scope"] = logical(latency.get("scope"))
    if cohort.get("human_review") in {"not_performed", "performed"}: result["human_review"] = cohort["human_review"]
    return result


def evaluations_from(metrics, config, manifest, source_artifact, receipt_verified=None):
    output = []
    for i, cohort in enumerate(metrics.get("cohorts", [])):
        if not isinstance(cohort, dict): continue
        statuses = cohort.get("statuses", {})
        scheduled, attempted, valid = cohort.get("scheduled"), cohort.get("attempted"), cohort.get("valid_answers")
        invalid = statuses.get("invalid_format")
        # Preserve original status counts without pretending omissions are zero.
        failures = sum(v for k, v in statuses.items() if k not in {"answered", "abstained", "invalid_format", "not_attempted"}) if statuses else None
        variant = "tuned" if config.get("adapter_path") or config.get("adapter") or "tuned" in manifest["run_id"].lower() else "base"
        protocol_hash = metrics.get("protocol_hash") or config.get("protocol_hash")
        row = {"evaluation_id": "aggregate-" + str(i), "run_id": manifest["run_id"], "variant": variant,
            "dataset_id": logical(cohort.get("source_id")), "source_id": logical(cohort.get("source_id")),
            "language": logical(cohort.get("language")), "task": logical(cohort.get("task_type")), "split": config.get("split"),
            "protocol": protocol_hash, "protocol_version": config.get("protocol_version"), "data_hash": manifest.get("data_hash"),
            "model": manifest.get("model"), "precision": manifest.get("precision"),
            "cohort_id": t.digest([manifest.get("model"), cohort.get("task_type"), cohort.get("source_id"), cohort.get("language"), cohort.get("subject"), protocol_hash, config.get("image_condition"), config.get("renderer_version")]) if protocol_hash else None,
            "image_condition": logical(config.get("image_condition")), "renderer_version": logical(config.get("renderer_version")),
            "status": "final" if config.get("status") == "SUCCEEDED" else "partial", "scheduled": scheduled,
            "attempted": attempted, "valid": valid, "invalid": invalid, "failed": failures,
            "unattempted": cohort.get("not_attempted_count"), "metrics": aggregate_metrics(cohort), "subgroups": [],
            "provenance": "historical_reconstructed", "source_artifact": source_artifact, "receipt_verified": receipt_verified}
        for sub in metrics.get("vqa_subgroups", []):
            if isinstance(sub, dict) and sub.get("source_id") == cohort.get("source_id"):
                row["subgroups"].append({"source_id": logical(sub.get("source_id")), "dimension": logical(sub.get("dimension")),
                    "value": logical(sub.get("value")), "metrics": aggregate_metrics(sub.get("metrics", {}))})
        output.append(row)
    return output


def source_bundle(directory, expected_hashes=None):
    directory = Path(directory).resolve(strict=True)
    files = {}
    for name in FILES:
        path = directory / name
        if not path.exists(): continue
        if path.is_symlink() or path.resolve().parent != directory: raise ValueError("SOURCE_SYMLINK_FORBIDDEN")
        files[name] = t.file_hash(path)
    for name, expected in (expected_hashes or {}).items():
        if files.get(name) != expected: raise ValueError("SOURCE_RECEIPT_HASH_MISMATCH")
    if not files: raise ValueError("NO_SUPPORTED_ARTIFACTS")
    # Native completion is authoritative only for hashes it actually lists.
    completion = read(directory / "completion.json") if "completion.json" in files else {}
    verified = set()
    for artifact in completion.get("artifacts", []):
        name = Path(artifact.get("path", "")).name
        if name in files:
            if files[name] != artifact.get("sha256"): raise ValueError("COMPLETION_ARTIFACT_HASH_MISMATCH")
            verified.add(name)
    if "run_manifest.json" in files:
        native = read(directory / "run_manifest.json")
        for name, expected in native.get("artifact_hashes", {}).items():
            if name in files:
                if files[name] != expected: raise ValueError("EVALUATION_ARTIFACT_HASH_MISMATCH")
                verified.add(name)
    verified.update(expected_hashes or {})
    return directory, files, verified


def import_directory(directory, output_root, campaign_id, *, expected_hashes=None, protocol_catalog=None):
    directory, files, verified = source_bundle(directory, expected_hashes)
    config = read(directory / "config.json") if "config.json" in files else {}
    native = read(directory / "run_manifest.json") if "run_manifest.json" in files else read(directory / "manifest.json") if "manifest.json" in files else {}
    config = {**config, **{k: v for k, v in native.items() if k not in config}}
    run_id = logical(config.get("run_id") or config.get("job_id") or directory.name)
    if not run_id: raise ValueError("LOGICAL_RUN_ID_REQUIRED")
    evaluation_metrics = read(directory / "metrics.json") if "metrics.json" in files else {}
    protocol_hash = evaluation_metrics.get("protocol_hash") or config.get("protocol_hash")
    protocol = (protocol_catalog or {}).get(protocol_hash)
    if protocol:
        config.update(split=protocol["split"], protocol_version=protocol["protocol_version"], data_hash=protocol["data_hash"])
        files["evaluation_protocol.json"] = protocol["sha256"]
    attempt_id = "hist-" + t.digest({"source_artifacts": files, "mapping_version": IMPORT_VERSION})[:24]
    destination = Path(output_root) / t.safe_component(campaign_id) / t.safe_component(run_id) / attempt_id
    sources = [{"artifact_id": name, "sha256": value} for name, value in sorted(files.items())]
    if destination.exists():
        existing = read(destination / "manifest.json")
        if existing.get("source_artifacts") != sources: raise ValueError("IMMUTABLE_IMPORT_CONFLICT")
        return {"campaign_id": campaign_id, "run_id": run_id, "attempt_id": attempt_id, "imported": False,
            "steps": sum(1 for _ in (destination / "steps.jsonl").open()) if (destination / "steps.jsonl").exists() else 0,
            "evaluations": len(read(destination / "evaluations.json")["evaluations"])}
    model = config.get("model_id") or (config.get("model", {}).get("model_id") if isinstance(config.get("model"), dict) else config.get("model"))
    recipe = config.get("recipe", {})
    metrics_name = "train.metrics.json" if "train.metrics.json" in files else "metrics.json" if "metrics.json" in files else None
    metrics = read(directory / metrics_name) if metrics_name else {}
    final_metrics = read(directory / "metrics.json") if "metrics.json" in files else metrics
    total_steps = metrics.get("optimizer_steps", config.get("optimizer_steps"))
    total_samples = metrics.get("samples_processed", config.get("samples_processed"))
    manifest = {"schema_version": t.SCHEMA, "telemetry_version": t.SCHEMA, "campaign_id": campaign_id,
        "job_id": logical(config.get("job_id")) or run_id, "run_id": run_id, "attempt_id": attempt_id,
        "source": "mediflow_historical", "producer": "mediflow.telemetry_import", "model": logical(model),
        "task": config.get("task"), "seed": recipe.get("seed", config.get("seed")),
        "data_hash": config.get("data_hash"), "scientific_recipe_hash": t.digest(recipe) if recipe else None,
        "code_sha": config.get("code", {}).get("commit") if isinstance(config.get("code"), dict) else None,
        "eval_protocol": final_metrics.get("protocol_hash", config.get("protocol_hash")),
        "precision": config.get("precision", config.get("dtype", recipe.get("compute_dtype"))), "parent_run_id": None,
        "freeze_id": config.get("data", {}).get("freeze_id"), "dataset_ids": [], "total_updates": total_steps,
        "samples_total": total_samples, "created_at": config.get("started_at"), "historical": True,
        "source_artifacts": sources, "cohort_id": None, "artifacts": []}
    for name in ("checkpoint.json", "reload.json", "final_selection.json"):
        if name in files:
            manifest["artifacts"].append({"artifact_id": name, "kind": "reload" if name == "reload.json" else "checkpoint",
                "sha256": files[name], "exists": True, "verified": True if name in verified else None,
                "reload_verified": None, "selection": "final" if name == "final_selection.json" else "historical_selected"})
    events = []
    if "events.jsonl" in files:
        with (directory / "events.jsonl").open() as stream:
            for line in stream:
                if not line.endswith("\n"): raise ValueError("SOURCE_PARTIAL_EVENT_LINE")
                events.append(json.loads(line))
    is_v6 = "example_losses" in metrics
    steps = [(r, None) for r in reconstruct_v6(metrics)] if is_v6 else legacy_steps(events, total_steps, total_samples)
    evaluation_source = {"artifact_id": "metrics.json", "sha256": files["metrics.json"]} if "metrics.json" in files else None
    evaluations = evaluations_from(final_metrics, config, manifest, evaluation_source, True if "metrics.json" in verified else None)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = Path(tempfile.mkdtemp(prefix=".import-", dir=destination.parent))
    writer = t.TelemetryWriter(temporary, manifest)
    if writer.error: raise ValueError("IMPORT_OUTPUT_INITIALIZATION_FAILED")
    writer.role = "historical"
    source_name = metrics_name if is_v6 else "events.jsonl"
    source_artifact = {"artifact_id": source_name, "sha256": files[source_name]} if source_name in files else None
    try:
        for step, at in steps:
            writer.append("steps", step, "training", emitted_at=at, source_artifact=source_artifact)
        mapping = {"optimizer_update": "optimizer_update", "checkpoint_saved": "checkpoint_saved", "training_started": "training_started",
            "reload_verified": "reload_succeeded", "run_succeeded": "succeeded", "run_failed": "failed", "run_held": "hold"}
        for event in events:
            name = mapping.get(event.get("event"))
            if name:
                writer.append("events", {"event": name, "message": None, "optimizer_step": event.get("step"), "details": {}},
                    t.EVENT_PHASE[name], emitted_at=event.get("at"), source_artifact={"artifact_id": "events.jsonl", "sha256": files["events.jsonl"]})
        if "completion.json" in files:
            completion = read(directory / "completion.json")
            writer.append("events", {"event": "succeeded" if completion.get("status") == "SUCCEEDED" else "failed", "message": None,
                "optimizer_step": total_steps, "details": {}}, "succeeded" if completion.get("status") == "SUCCEEDED" else "failed",
                emitted_at=completion.get("finished_at"), source_artifact={"artifact_id": "completion.json", "sha256": files["completion.json"]})
        status_raw = config.get("status") or (read(directory / "completion.json").get("status") if "completion.json" in files else None)
        status_name = {"SUCCEEDED": "succeeded", "FAILED": "failed", "RUNNING": "unknown", "HELD": "held"}.get(status_raw, "unknown")
        reload = read(directory / "reload.json") if "reload.json" in files else {}
        reload_ok = reload.get("status") == "SUCCEEDED" if "status" in reload else native.get("reload_verified")
        training_ok = status_name == "succeeded" if steps else None
        last_at = config.get("finished_at") or (events[-1].get("at") if events else None)
        checks = {"optimizer_steps": len(steps) == total_steps if total_steps is not None else None,
            "samples_processed": sum(r[0]["batch_examples"] for r in steps) == total_samples if total_samples is not None else None}
        if is_v6:
            # Step sequence token counts and timings were not recorded in V6.
            checks.update(loss=math.isclose(sum(r[0]["train_loss"] * r[0]["batch_examples"] for r in steps) / total_samples,
                metrics["loss_mean"], rel_tol=1e-9, abs_tol=1e-9),
                learning_rates=[r[0]["lr_used"] for r in steps] == metrics.get("learning_rates"),
                gradient_norms=[r[0]["gradient_norm_before_clip"] for r in steps] == metrics.get("gradient_norms_before_clip"),
                supervised_tokens=sum(r[0]["supervised_tokens"] or 0 for r in steps) == metrics.get("supervised_tokens"))
        reconciliation = {"status": "passed" if checks and all(v is True for v in checks.values()) else "unavailable" if not steps else "failed", "checks": checks}
        complete = bool(steps) and reconciliation["status"] == "passed" and writer.error is None
        status = {**writer.status, "status": status_name, "phase": "succeeded" if status_name == "succeeded" else "failed" if status_name == "failed" else "unknown",
            "optimizer_step": steps[-1][0]["optimizer_step"] if steps else None, "total_updates": total_steps,
            "samples_processed": sum(r[0]["batch_examples"] for r in steps) if steps else total_samples,
            "samples_total": total_samples, "last_emitted_at": last_at, "training_succeeded": training_ok,
            "reload_succeeded": reload_ok, "evaluation_completed": bool(evaluations) and all(x["status"] == "final" for x in evaluations) if evaluations else None,
            "telemetry_complete": complete, "writer_health": "ok" if writer.error is None else "degraded"}
        resource = metrics.get("resource", {})
        weighted_loss = sum(r[0]["train_loss"] * r[0]["batch_examples"] for r in steps) / sum(r[0]["batch_examples"] for r in steps) if steps and all(r[0]["train_loss"] is not None for r in steps) else None
        summary = {"train_loss_mean": metrics.get("loss_mean", weighted_loss), "optimizer_steps": total_steps,
            "samples_processed": total_samples, "supervised_tokens": metrics.get("supervised_tokens", native.get("supervised_tokens")),
            "sequence_tokens": metrics.get("sequence_tokens"), "wall_seconds": resource.get("wall_seconds", native.get("attempt_wall_seconds", native.get("wall_seconds"))),
            "wall_time_scope": "model_load_audit_training_verify_save_and_final_sync" if is_v6 else "legacy_attempt_wall",
            "peak_allocated_bytes": resource.get("peak_allocated_bytes", native.get("peak_cuda_allocated_bytes")),
            "peak_reserved_bytes": resource.get("peak_reserved_bytes", native.get("peak_cuda_reserved_bytes")),
            "reconciliation": reconciliation, "telemetry_complete": complete}
        t.atomic_json(temporary / "status.json", t.normalize(status))
        t.atomic_json(temporary / "summary.json", t.normalize(summary))
        t.atomic_json(temporary / "evaluations.json", {"evaluations": t.normalize(evaluations)})
        writer.close()
        if writer.error: raise ValueError("IMPORT_WRITE_FAILED")
        os.rename(temporary, destination)
    finally:
        writer.close()
    return {"campaign_id": campaign_id, "run_id": run_id, "attempt_id": attempt_id, "imported": True,
        "steps": len(steps), "evaluations": len(evaluations), "reconciliation": reconciliation["status"]}


def load_protocol_catalog(data_root):
    """Hash-match frozen protocol metadata without loading any raw QA/predictions."""
    root = Path(data_root)
    catalog = {}
    locations = ("manifests/evaluation", "v2/text", "v2/vqa/protocols", "v4")
    for location in locations:
        for path in sorted((root / location).glob("*.json")):
            if path.is_symlink() or path.stat().st_size > 16 * 1024 * 1024: continue
            protocol = read(path)
            if not isinstance(protocol, dict) or not all(k in protocol for k in ("protocol_hash", "protocol_version", "split", "sample_ids", "input_digests", "reference_digest", "files")): continue
            expected = protocol["protocol_hash"]
            if expected != t.digest({k: v for k, v in protocol.items() if k != "protocol_hash"}): raise ValueError("EVALUATION_PROTOCOL_HASH_MISMATCH")
            # Dataset identity is derived solely from native scheduled identities,
            # input/reference hashes and source bytes. Paths never leave this CLI.
            data_hash = t.digest({"sample_ids": protocol["sample_ids"], "input_digests": protocol["input_digests"],
                "reference_digest": protocol["reference_digest"], "source_file_hashes": sorted(protocol["files"].values())})
            catalog[expected] = {"split": protocol["split"], "protocol_version": protocol["protocol_version"],
                "data_hash": data_hash, "sha256": t.file_hash(path)}
    return catalog


def import_comparisons(data_root, output_root):
    """Preserve already computed paired bootstrap CIs; never rescore predictions."""
    root = Path(data_root).resolve()
    results = []
    for campaign, location in (("medvlm-v1", "runs"), ("medvlm-v2", "v2/text"), ("medvlm-v2", "v2/vqa"), ("medvlm-v4", "v4")):
        for path in sorted((root / location).glob("*.json")):
            if path.is_symlink() or path.stat().st_size > 4 * 1024 * 1024: continue
            original = read(path)
            if not isinstance(original, dict) or not all(k in original for k in ("comparison", "base_run", "tuned_run")): continue
            if not isinstance(original["comparison"], list): continue
            sha = t.file_hash(path)
            run_id = "comparison-" + t.safe_component(path.stem)
            attempt = "hist-" + t.digest({"source_sha256": sha, "mapping_version": IMPORT_VERSION})[:24]
            destination = Path(output_root) / campaign / run_id / attempt
            if destination.exists():
                results.append({"campaign_id": campaign, "run_id": run_id, "attempt_id": attempt, "imported": False, "steps": 0,
                    "evaluations": len(read(destination / "evaluations.json")["evaluations"])}); continue
            config = {}
            tuned_path = Path(str(original["tuned_run"]))
            if tuned_path.is_absolute() and tuned_path.resolve().is_relative_to(root):
                receipt = tuned_path / "run_manifest.json"
                if receipt.is_file() and not receipt.is_symlink(): config = read(receipt)
            manifest = t.manifest_for({"run_id": run_id, "job_id": run_id, "model_id": config.get("model_id"),
                "task": None, "recipe": {}}, attempt, campaign)
            manifest.update(source="mediflow_historical", producer="mediflow.telemetry_import", historical=True,
                created_at=None, precision=config.get("precision"), parent_run_id=logical(config.get("run_id")),
                source_artifacts=[{"artifact_id": "paired-comparison.json", "sha256": sha}])
            evaluations = []
            for i, item in enumerate(original["comparison"]):
                if not isinstance(item, dict): continue
                metrics = {k: scalar(item.get(k)) for k in ("delta", "group_count", "bootstrap_resamples", "training_seed_uncertainty_measured")}
                metrics["bootstrap_unit"] = logical(item.get("bootstrap_unit")); metrics["metric"] = logical(item.get("metric"))
                ci = item.get("ci95")
                metrics["ci95"] = [scalar(x) for x in ci] if isinstance(ci, list) and len(ci) == 2 else None
                evaluations.append({"evaluation_id": "paired-" + str(i), "run_id": run_id, "variant": "tuned",
                    "dataset_id": logical(item.get("source_id")), "source_id": logical(item.get("source_id")),
                    "language": logical(item.get("language")), "task": logical(item.get("task_type")),
                    "split": None, "protocol": config.get("protocol_hash"), "protocol_version": None,
                    "data_hash": None, "model": manifest["model"], "precision": manifest["precision"], "cohort_id": None,
                    "image_condition": logical(config.get("image_condition")), "renderer_version": logical(config.get("renderer_version")),
                    "status": "final", "scheduled": item.get("n"), "attempted": None, "valid": None, "invalid": None,
                    "failed": None, "unattempted": None, "metrics": metrics, "subgroups": [],
                    "provenance": "historical_reconstructed", "source_artifact": {"artifact_id": "paired-comparison.json", "sha256": sha}, "receipt_verified": None})
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            temporary = Path(tempfile.mkdtemp(prefix=".comparison-", dir=destination.parent))
            writer = t.TelemetryWriter(temporary, manifest)
            status = {**writer.status, "status": "succeeded", "phase": "succeeded", "optimizer_step": None,
                "samples_processed": None, "training_succeeded": None, "evaluation_completed": True, "telemetry_complete": False}
            t.atomic_json(temporary / "status.json", status)
            t.atomic_json(temporary / "evaluations.json", {"evaluations": evaluations})
            writer.close()
            if writer.error: raise ValueError("COMPARISON_IMPORT_WRITE_FAILED")
            os.rename(temporary, destination)
            results.append({"campaign_id": campaign, "run_id": run_id, "attempt_id": attempt, "imported": True, "steps": 0, "evaluations": len(evaluations)})
    return results


def discover(data_root):
    """Finite historical run/evaluation trees; no raw, prepared, cache or predictions."""
    root = Path(data_root)
    layouts = {"medvlm-v1": ["runs"], "medvlm-v2": ["v2/text/runs", "v2/text/evaluations", "v2/vqa/runs", "v2/vqa/evaluations"],
        "medvlm-v4": ["v4/runs", "v4/evaluations"], "medvlm-v5": ["v5/runs"],
        "medvlm-v6": ["v6_v7/gpu_smoke", "v6_v7/main_train/first_seed", "v6_v7/runs"]}
    for campaign, paths in layouts.items():
        for relative in paths:
            base = root / relative
            if not base.exists(): continue
            for directory in sorted(base.iterdir()):
                if not directory.is_dir() or directory.is_symlink() or ".attempt-" in directory.name: continue
                if any((directory / name).is_file() for name in ("train.metrics.json", "metrics.json", "events.jsonl", "run_manifest.json", "manifest.json")):
                    yield campaign, directory


def metadata_row(row, freeze_id):
    # Copy the native provenance structure without input/supervision/image paths.
    record = {key: row.get(key) for key in METADATA_FIELDS if key in row}
    record["freeze_id"] = freeze_id
    provenance = row.get("provenance", {})
    record["license_ref"] = provenance.get("license_ref")
    record["image_sha256"] = None
    images = row.get("input", {}).get("images", [])
    if isinstance(images, list) and images:
        image_hash = provenance.get("image_sha256") or row.get("metadata", {}).get("image_sha256")
        if isinstance(image_hash, str) and len(image_hash) == 64:
            record["image_sha256"] = image_hash
        else:
            sha = [x.get("sha256") for x in images if isinstance(x, dict) and isinstance(x.get("sha256"), str) and len(x["sha256"]) == 64]
            if sha: record["image_sha256"] = sha[0] if len(sha) == 1 else sha
    return record


def export_freeze(freeze_directory, output_root):
    directory = Path(freeze_directory)
    freeze = read(directory / "freeze.json")
    freeze_id = freeze["freeze_id"]
    if freeze_id != t.digest({k: v for k, v in freeze.items() if k != "freeze_id"}): raise ValueError("FREEZE_ID_MISMATCH")
    result = []
    for task in ("text", "vqa"):
        source = directory / (task + ".train.jsonl")
        axis = freeze["axes"][task]
        sha = t.file_hash(source)
        if sha != axis["train_sha256"]: raise ValueError("FROZEN_DATA_BYTES_CHANGED")
        dataset_id = "v6-common-" + task
        destination = Path(output_root) / "datasets" / dataset_id
        if destination.exists():
            existing = read(destination / "manifest.json")
            if existing.get("data_hash") != sha or existing.get("freeze_id") != freeze_id: raise ValueError("IMMUTABLE_DATASET_CONFLICT")
            result.append({"dataset_id": dataset_id, "record_count": existing["record_count"], "imported": False}); continue
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = Path(tempfile.mkdtemp(prefix=".dataset-", dir=destination.parent))
        sources, count = {}, 0
        with source.open() as inputs, (temporary / "records.metadata.jsonl").open("w", encoding="utf-8") as output:
            for line in inputs:
                row = json.loads(line)
                record = metadata_row(row, freeze_id)
                output.write(t.canonical(record) + "\n"); count += 1
                key = row["source_id"]
                sources[key] = {"source_id": key, "source_revision": row.get("source_revision"),
                    "license_ref": record.get("license_ref"), "restrictions": row.get("restrictions")}
            output.flush(); os.fsync(output.fileno())
        if count != axis["common_count"]: raise ValueError("FROZEN_DATA_COUNT_MISMATCH")
        manifest = {"schema_version": "mediflow.dataset-reference.v1", "dataset_id": dataset_id, "name": "V6 common frozen " + task + " train",
            "freeze_id": freeze_id, "data_hash": sha, "record_count": count, "split_counts": {"train": count, "validation": 0, "test": 0},
            "sources": list(sources.values()), "source_artifacts": [{"artifact_id": task + ".train.jsonl", "sha256": sha},
                {"artifact_id": "freeze.json", "sha256": t.file_hash(directory / "freeze.json")}],
            "frozen": True, "status": "verified", "metadata_index": "records.metadata.jsonl",
            "metadata_sha256": t.file_hash(temporary / "records.metadata.jsonl"), "provenance_preserved": True}
        t.atomic_json(temporary / "manifest.json", manifest)
        os.rename(temporary, destination)
        result.append({"dataset_id": dataset_id, "record_count": count, "imported": True})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    single = commands.add_parser("import-run"); single.add_argument("--directory", type=Path, required=True); single.add_argument("--campaign-id", required=True)
    single.add_argument("--expected-hashes", type=Path)
    history = commands.add_parser("import-history"); history.add_argument("--data-root", type=Path, required=True)
    freeze = commands.add_parser("export-freeze"); freeze.add_argument("--freeze-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "import-run":
        result = import_directory(args.directory, args.output_root, args.campaign_id, expected_hashes=read(args.expected_hashes) if args.expected_hashes else None)
    elif args.command == "export-freeze": result = export_freeze(args.freeze_directory, args.output_root)
    else:
        result, failures = [], []
        protocols = load_protocol_catalog(args.data_root)
        for campaign, directory in discover(args.data_root):
            try: result.append(import_directory(directory, args.output_root, campaign, protocol_catalog=protocols))
            except Exception as exc: failures.append({"campaign_id": campaign, "run_id": t.safe_component(directory.name), "reason": str(exc) if isinstance(exc, ValueError) and str(exc).isupper() else type(exc).__name__})
        result.extend(import_comparisons(args.data_root, args.output_root))
        result = {"runs": result, "failures": failures, "availability": {"medvlm-v3": "no_training_or_aggregate_metrics_found", "medvlm-v5": "run_manifests_only_no_aggregate_metrics_found"}}
    print(t.canonical(result))
    return 0


if __name__ == "__main__": raise SystemExit(main())
