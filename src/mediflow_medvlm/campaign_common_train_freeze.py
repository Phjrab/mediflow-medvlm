"""Freeze V6 common train cohorts and a reproducible, unexecuted main run plan.

CPU-only planner: build_config(manifest_path, state_path, private_output,
protection={name: {path, sha256}}, repo=immutable_source_snapshot). Save its result
once; activation_manifest returns a Supervisor.activate-compatible proposal.
The worker never changes campaign state and never loads a model or tokenizer.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

from . import campaign_preflight_gate as gate
from .data.io import digest
from .data.schema import input_digest
from .campaign_supervisor import HASH_FIELDS, atomic_json, canonical_hash, file_hash, validate_manifest, verify_source
from .runtime import utcnow

SCHEMA = "medvlm.v6.common-train-freeze-config.v1"
SCOPE = "COMMON_TRAIN_FREEZE_AND_MAIN_RUN_PLAN_ONLY"
JOB_ID = "v6-common_train_freeze"
SEEDS = [20261023, 20261114, 20261115]
SOURCE_FILES = (*gate.SOURCE_FILES, "campaign_common_train_freeze.py", "data/__init__.py", "data/io.py", "data/schema.py")
IDENTITY_KEYS = gate.IDENTITY_KEYS
PROTECTION_KEYS = {"text_manifest", "text_freeze", "text_ancestor_freeze", "protected_freeze", "global_audit",
    "text_verification", "slake_build", "slake_image_audit", "source_audit", "selection", "semantic_audit",
    "internal_audit", "semantic_policy", "semantic_encoding"}
OUTPUT_FILES = {"config.json", "freeze.json", "summary.json", "text.train.jsonl", "vqa.train.jsonl"}
_require, _read, _absolute, _equal = gate._require, gate._read, gate._absolute, gate._equal


def source_identity(repo=None):
    package = Path(__file__).resolve().parent if repo is None else Path(repo) / "src/mediflow_medvlm"
    files = {"src/mediflow_medvlm/" + name: file_hash(package / name) for name in SOURCE_FILES}
    return files, canonical_hash(files)


def _snapshot(manifest, state):
    value = gate._snapshot(manifest, state)
    jobs = {j["job_id"]: j for j in manifest["jobs"]}
    job = jobs.get(JOB_ID, {})
    _require(job.get("phase") == "common_train_freeze" and job.get("cell_id") == "ALL_V6"
        and job.get("run_id") == JOB_ID and job.get("dependencies") == [gate.JOB_ID]
        and job.get("dependency_policy") == "terminal", "COMMON_GATE_DECLARATION_MISMATCH")
    _require(manifest.get("seeds") == SEEDS, "EXACT_DECLARED_SEEDS_REQUIRED")
    _require(state["jobs"][gate.JOB_ID].get("status") == "SUCCEEDED", "AGGREGATE_SUCCESS_REQUIRED")
    value["jobs"][gate.JOB_ID] = jobs[gate.JOB_ID]
    value["states"][gate.JOB_ID] = state["jobs"][gate.JOB_ID]
    return copy.deepcopy(value)


def _native(snapshot):
    return {**snapshot, "jobs": {k: v for k, v in snapshot["jobs"].items() if k != gate.JOB_ID},
            "states": {k: v for k, v in snapshot["states"].items() if k != gate.JOB_ID}}


def _aggregate(snapshot):
    job = snapshot["jobs"][gate.JOB_ID]
    verify_source(job)
    receipt = gate._success(job, snapshot["states"][gate.JOB_ID])
    argv = job["argv"]
    _require(argv[1:3] == ["-m", "mediflow_medvlm.campaign_preflight_gate"] and argv.count("--config") == 1,
             "AGGREGATE_COMMAND_MISMATCH")
    config = _read(_absolute(argv[argv.index("--config") + 1]))
    _require(config.get("schema_version") == gate.SCHEMA and config.get("scope") == gate.SCOPE
        and config.get("config_hash") == canonical_hash({k: v for k, v in config.items() if k != "config_hash"})
        and all(_equal(config.get(k), job.get(k)) for k in (*IDENTITY_KEYS, "source_files"))
        and _equal(config["prerequisites"], _native(snapshot)), "AGGREGATE_CONFIG_BINDING_MISMATCH")
    _require(_equal(gate._completed(config), receipt), "AGGREGATE_COMPLETION_MISMATCH")
    audit = gate._audit(_native(snapshot))
    summary = _read(Path(config["output_dir"]) / "summary.json")
    supported = [r["cell_id"] for r in audit["cells"] if r["status"] == "SUPPORTED"]
    _require(_equal(config["evidence"], audit["evidence"])
        and _equal(summary.get("cells"), audit["cells"]) and summary.get("supported_cells") == supported
        and receipt.get("supported_cells") == supported and summary.get("aggregate_audit_succeeded") is True
        and summary.get("common_train_freeze_decided") is False and summary.get("comparison_cohort_frozen") is False,
        "AGGREGATE_RESULT_MISMATCH")
    _require(all(any(r["status"] == "SUPPORTED" and r["task"] == axis for r in audit["cells"])
                 for axis in ("text", "vqa")), "BOTH_AXES_REQUIRE_SUPPORTED_CELLS")
    return audit


def _ref(reference):
    path = _absolute(reference["path"])
    _require(not path.is_symlink() and file_hash(path) == reference["sha256"], "PROTECTION_REFERENCE_CHANGED")
    return _read(path)


def _rows(entry):
    path = _absolute(entry.get("snapshot_path", entry.get("path")))
    _require(not path.is_symlink() and file_hash(path) == entry["sha256"], "FROZEN_SPLIT_CHANGED")
    lines = path.read_bytes().splitlines(keepends=True)
    _require(all(line.endswith(b"\n") and line.strip() for line in lines), "INVALID_PARENT_JSONL_BYTES")
    rows = [json.loads(line) for line in lines]
    ids = [row["sample_id"] for row in rows]
    _require(len(rows) == entry["count"] and len(set(ids)) == len(ids)
        and ("ordered_sample_ids" not in entry or entry["ordered_sample_ids"] == ids), "FROZEN_SPLIT_IDENTITY_MISMATCH")
    return rows, lines


def _freeze_identity(value, v2=False):
    omitted = {"freeze_id"} if v2 else {"freeze_id", "status", "evidence_level"}
    _require(value.get("status") == "SUCCEEDED" and value.get("freeze_id") == digest(
        {k: v for k, v in value.items() if k not in omitted}), "HISTORICAL_FREEZE_ID_MISMATCH")


def _groups(row):
    source = row["source_id"]
    result = set()
    for key, value in row.get("group_ids", {}).items():
        for member in (value if isinstance(value, list) else [value]):
            if member:
                result.add((source, key, str(member)))
    meta = row.get("metadata", {})
    if source == "pubmedqa_labeled" and meta.get("pmid"):
        result.add((source, "pmid", str(meta["pmid"])))
    if source == "medquad" and row.get("provenance", {}).get("source_url"):
        result.add((source, "url", row["provenance"]["source_url"]))
    if row.get("input", {}).get("images") or source in {"slake", "vqa_rad"}:
        result.update(("GLOBAL_IMAGE", k, str(meta[k])) for k in ("image_sha256", "decoded_pixel_hash") if meta.get(k))
    return result


def _projection(row):
    """Explicitly discard heldout supervision; no target lengths or scores enter selection."""
    return {"sample_id": row["sample_id"], "source_id": row["source_id"], "split": row["split"],
            "group_ids": row.get("group_ids", {}), "input_digest": input_digest(row),
            "groups": sorted(_groups(row))}


def _protection(references, parents):
    _require(set(references) == PROTECTION_KEYS, "EXACT_PROTECTION_REFERENCES_REQUIRED")
    docs = {k: _ref(v) for k, v in references.items()}
    mix, old, protected = (docs[k] for k in ("text_freeze", "text_ancestor_freeze", "protected_freeze"))
    for value, v2 in ((mix, True), (old, False), (protected, False)):
        _freeze_identity(value, v2)
    tm, verification = docs["text_manifest"], docs["text_verification"]
    _require(tm.get("status") == verification.get("status") == "SUCCEEDED"
        and tm["corpora"]["mix"]["freeze_manifest"] == references["text_freeze"]["path"]
        and tm["corpora"]["mix"]["freeze_manifest_sha256"] == references["text_freeze"]["sha256"]
        and verification.get("manifest_sha256") == references["text_manifest"]["sha256"]
        and verification.get("checks") and all(v is True for v in verification["checks"].values()), "TEXT_LINEAGE_MISMATCH")
    for name, field in (("source_audit", "source_audit_path"), ("selection", "selection_manifest_path"),
            ("semantic_audit", "semantic_audit_path"), ("internal_audit", "internal_audit_path")):
        _require(tm.get(field) == references[name]["path"], "TEXT_CHILD_REFERENCE_MISMATCH")
    for name, key in (("source_audit", "source_audit_sha256"), ("text_ancestor_freeze", "v1_freeze_sha256"),
            ("selection", "selection_manifest_sha256"), ("semantic_audit", "semantic_audit_sha256"),
            ("internal_audit", "internal_split_audit_sha256")):
        _require(mix["provenance"].get(key) == references[name]["sha256"], "TEXT_PROVENANCE_MISMATCH")
    _require(mix["provenance"].get("test_used_for_selection") is False
        and docs["selection"].get("corpus_selection_used_test_performance") is False
        and protected.get("audit_sha256") == references["global_audit"]["sha256"]
        and docs["global_audit"].get("status") == "SUCCEEDED"
        and docs["global_audit"].get("critical_error_count") == 0, "PROTECTED_AUDIT_MISMATCH")
    bridge, heldout = {}, []
    for prefix in ("kormedmcqa", "medquad", "pubmedqa_labeled", "slake", "text", "vqa_rad"):
        for split in ("train", "validation", "test"):
            key = prefix + "/" + split
            b = protected["splits"][key]
            if prefix != "vqa_rad":
                a = old["splits"][key]
                _require(all(a[k] == b[k] for k in ("count", "sha256", "ordered_sample_ids")), "PROTECTED_LINEAGE_BRIDGE_MISMATCH")
            rows, _ = _rows(b)
            bridge[key] = {k: b[k] for k in ("count", "sha256")}
            if split != "train":
                heldout.extend(_projection(row) for row in rows)
    for axis, entry in (("text", mix["splits"]["mix_v2/train"]), ("vqa", protected["splits"]["slake/train"])):
        _require(entry["sha256"] == parents[axis]["sha256"] and entry["count"] == parents[axis]["count"], "PARENT_CORPUS_LINEAGE_MISMATCH")
    for split in ("validation", "test"):
        rows, _ = _rows(mix["splits"]["mix_v2/" + split])
        heldout.extend(_projection(row) for row in rows)
    holdout_path = _absolute(tm["aihub_holdout_freeze"])
    holdout = _read(holdout_path)
    _freeze_identity(holdout, True)
    aihub = []
    for split in ("validation", "test"):
        entry = next(v for k, v in holdout["splits"].items() if k.endswith("/" + split))
        rows, _ = _rows(entry)
        aihub.extend(rows)
        heldout.extend(_projection(row) for row in rows)
    _require(len(aihub) == docs["internal_audit"]["protected_count"]
        and docs["internal_audit"].get("protected_rows_modified") is False, "AIHUB_INTERNAL_PROTECTION_MISMATCH")
    source = docs["source_audit"]
    _require(source.get("protected_freeze_sha256") == references["text_ancestor_freeze"]["sha256"], "SOURCE_PROTECTED_FREEZE_MISMATCH")
    protected_ids = set()
    for split in ("validation", "test"):
        a, b = source["protected_files"][split], old["splits"]["kormedmcqa/" + split]
        _require(all(a[k] == b[k] for k in ("count", "sha256")), "KORMED_PROTECTION_MISMATCH")
        protected_ids.update(b["ordered_sample_ids"])
    sem, policy, encoding = (docs[k] for k in ("semantic_audit", "semantic_policy", "semantic_encoding"))
    _require(len(protected_ids) == sem.get("protected_count") == policy.get("protected_count") == 4068
        and set(policy["sample_ids"][:4068]) == protected_ids and sem.get("status") == "SUCCEEDED"
        and sem.get("all_input_tokens_covered") is True and encoding.get("all_input_tokens_covered") is True
        and encoding.get("truncated_rows") == 0 and encoding.get("rows") == len(policy["sample_ids"])
        and encoding.get("policy_sha256") == references["semantic_policy"]["sha256"], "SEMANTIC_PROTECTED_COVERAGE_MISMATCH")
    for name in ("policy", "encoding"):
        _require(sem[name + "_path"] == references["semantic_" + name]["path"]
            and sem[name + "_sha256"] == references["semantic_" + name]["sha256"], "SEMANTIC_REFERENCE_MISMATCH")
    _require(sem.get("semantic_equivalence_verified") is False and sem.get("translation_equivalence_verified") is False,
             "SEMANTIC_LIMITATION_MISMATCH")
    candidates = {r["sample_id"] for r in sem["candidate_pairs"]}
    internal_candidates = {r["sample_id"] for name in ("confirmed_overlaps", "near_candidates")
                           for r in docs["internal_audit"].get(name, [])}
    for split in ("train", "validation"):
        _require(not (candidates | internal_candidates).intersection(mix["splits"]["mix_v2/" + split]["ordered_sample_ids"]), "SEMANTIC_QUARANTINE_VIOLATION")
    build, image = docs["slake_build"], docs["slake_image_audit"]
    _require(build.get("status") == image.get("status") == "SUCCEEDED"
        and _equal(build.get("image_inventory"), image.get("image_inventory"))
        and _equal(build.get("near_duplicate_policy"), image.get("near_duplicate_policy"))
        and build.get("patient_group_verified") is False, "SLAKE_IMAGE_LINEAGE_MISMATCH")
    for split in ("train", "validation", "test"):
        _require(all(build["canonical_files"][split][k] == protected["splits"]["slake/" + split][k]
            for k in ("count", "sha256")), "SLAKE_SPLIT_LINEAGE_MISMATCH")
    unique = {}
    for projection in heldout:
        sid = projection["sample_id"]
        _require(sid not in unique or _equal(unique[sid], projection), "CONFLICTING_PROTECTED_SAMPLE_METADATA")
        unique[sid] = projection
    return {"lineage_bridge": bridge, "heldout_metadata_sha256": canonical_hash(sorted(unique.values(), key=lambda r: r["sample_id"])),
        "heldout_unique_count": len(unique), "aihub_holdout_freeze": {"path": str(holdout_path), "sha256": file_hash(holdout_path)},
        "protected_kormed_count": len(protected_ids), "semantic_test_candidates_preserved": len(candidates.intersection(mix["splits"]["mix_v2/test"]["ordered_sample_ids"])),
        "original_ancestor_audit_reproduced": False, "semantic_equivalence_verified": False,
        "translation_equivalence_verified": False, "patient_study_independence_verified": False}, list(unique.values())


def _strata(rows):
    groups = {}
    for row in rows:
        value = gate.cpu._stratum(row)
        key = canonical_hash(value)
        groups.setdefault(key, {**value, "count": 0})["count"] += 1
    return [groups[k] for k in sorted(groups)]


def _orders(ids):
    result = {}
    for seed in SEEDS:
        indices = sorted(range(len(ids)), key=lambda i: (canonical_hash([seed, ids[i]]), ids[i]))
        order = [ids[i] for i in indices]
        batches = [len(indices[i:i + 16]) for i in range(0, len(indices), 16)]
        result[str(seed)] = {"indices": indices, "indices_sha256": canonical_hash(indices),
            "sample_order_sha256": canonical_hash(order), "optimizer_steps": len(batches),
            "update_example_counts": batches, "final_update_examples": batches[-1],
            "micro_loss_divisor_per_update": batches, "training_executed": False,
            "requires_family_selection_before_execution": seed != SEEDS[0]}
    return result


def _plan(snapshot, references):
    audit = _aggregate(snapshot)
    supported = [r for r in audit["cells"] if r["status"] == "SUPPORTED"]
    axes, payloads, parents, native = {}, {}, {}, {}
    for axis in ("text", "vqa"):
        cells = [r["cell_id"] for r in supported if r["task"] == axis]
        configs, eligible = {}, {}
        for cell in cells:
            evidence = audit["evidence"]["v6-native-preflight-" + cell]
            config = _ref(evidence["config"])
            _require(file_hash(evidence["eligibility"]["path"]) == evidence["eligibility"]["sha256"], "ELIGIBILITY_CHANGED")
            records = [json.loads(line) for line in Path(evidence["eligibility"]["path"]).read_text().splitlines()]
            configs[cell], eligible[cell] = config, records
        first = configs[cells[0]]["data"]
        _require(all(_equal(c["data"], first) for c in configs.values()), "AXIS_PARENT_DATA_MISMATCH")
        parent = {"path": first["train_file"], "sha256": first["train_sha256"], "count": first["expected_count"]}
        rows, lines = _rows(parent)
        ids = [r["sample_id"] for r in rows]
        _require(first["sample_order_sha256"] == canonical_hash(ids), "PARENT_SOURCE_ORDER_MISMATCH")
        allow = {"aihub_71875", "medquad", "pubmedqa_labeled"} if axis == "text" else {"slake"}
        _require(all(r["source_id"] in allow and r["split"] == "train" for r in rows), "UNSUPPORTED_TRAIN_SOURCE_OR_SPLIT")
        for cell, records in eligible.items():
            _require([r["sample_id"] for r in records] == ids and all(r["position"] == i
                and r["input_sha256"] == canonical_hash(rows[i]["input"])
                and r["target_sha256"] == canonical_hash(rows[i]["supervision"])
                and all(r.get(k) == v for k, v in gate.cpu._stratum(rows[i]).items()) for i, r in enumerate(records)),
                "ELIGIBILITY_PARENT_CONTENT_MISMATCH")
        keep = [i for i in range(len(rows)) if all(eligible[c][i].get("eligible") is True for c in cells)]
        _require(bool(keep), "EMPTY_COMMON_TRAIN_INTERSECTION")
        retained = [rows[i] for i in keep]
        payload = b"".join(lines[i] for i in keep)
        per_model = {}
        for cell in cells:
            records = eligible[cell]
            per_model[cell] = {"eligibility": audit["evidence"]["v6-native-preflight-" + cell]["eligibility"],
                "original_count": len(rows), "eligible_count": sum(r["eligible"] is True for r in records),
                "common_count": len(keep),
                "eligible_sequence_tokens": sum(r["native"]["sequence_tokens"] for r in records if r["eligible"]),
                "eligible_target_tokens": sum(r["native"]["supervised_tokens_after_shift"] for r in records if r["eligible"]),
                "eligible_strata": _strata([rows[i] for i, r in enumerate(records) if r["eligible"]]),
                "excluded": [{"sample_id": r["sample_id"], "position": i, "reason": r["reason"], **gate.cpu._stratum(rows[i])}
                             for i, r in enumerate(records) if r["eligible"] is False],
                "common_sequence_tokens": sum(records[i]["native"]["sequence_tokens"] for i in keep),
                "common_target_tokens": sum(records[i]["native"]["supervised_tokens_after_shift"] for i in keep)}
            native[cell] = {**gate.cell_spec(cell), "model": configs[cell]["model"], "task": axis,
                "native_preflight_config": audit["evidence"]["v6-native-preflight-" + cell]["config"],
                "gpu_smoke_config": audit["evidence"]["v6-native-smoke-" + cell]["config"],
                "max_length": configs[cell]["max_length"], "fixed_template_date": configs[cell]["fixed_template_date"],
                "adapter_initialization": None, "initialization_source": "original_pinned_base",
                "smoke_adapter_reuse": False, "historical_training_reused": False,
                "prior_training_reuse_certified": False, "new_training_required": True,
                "new_training_required_reason": "New V6 cohort/order/native recipe identity; historical adapter equivalence not certified"}
        axes[axis] = {"cells": cells, "parent": parent, "original_count": len(rows), "common_count": len(keep),
            "excluded_count": len(rows) - len(keep), "source_positions": keep,
            "excluded": [{"sample_id": r["sample_id"], "position": i,
                "blocking_cells": {c: eligible[c][i]["reason"] for c in cells if eligible[c][i]["eligible"] is False},
                **gate.cpu._stratum(r)} for i, r in enumerate(rows) if i not in set(keep)],
            "sample_ids": [r["sample_id"] for r in retained], "sample_order_sha256": canonical_hash([r["sample_id"] for r in retained]),
            "train_sha256": hashlib.sha256(payload).hexdigest(), "original_strata": _strata(rows),
            "common_strata": _strata(retained), "excluded_strata": _strata([r for i, r in enumerate(rows) if i not in set(keep)]),
            "per_model": per_model, "training_orders": _orders([r["sample_id"] for r in retained])}
        parents[axis], payloads[axis] = parent, payload
    protection, heldout = _protection(references, parents)
    held_ids = {r["sample_id"] for r in heldout}
    held_inputs = {r["input_digest"] for r in heldout}
    held_groups = {tuple(g) for r in heldout for g in r["groups"]}
    for axis in axes:
        rows, _ = _rows(parents[axis])
        retained = [rows[i] for i in axes[axis]["source_positions"]]
        _require(all(r["sample_id"] not in held_ids and input_digest(r) not in held_inputs
            and not _groups(r).intersection(held_groups) for r in retained), "COMMON_TRAIN_PROTECTED_OVERLAP")
    recipe = {"passes": 1, "max_length": 2048, "full_input_and_answer_preserved": True,
        "loss": "mean_examples_of_mean_full_target_token_cross_entropy", "microbatch": 1,
        "effective_batch_size": 16, "partial_update_divisor": "actual_example_count",
        "repeat_or_drop_examples": False, "quantization": "nf4_double_quant", "compute_dtype": "bfloat16",
        "lora_rank": 8, "lora_alpha": 16, "lora_dropout": 0.05,
        "projection_targets": "native_smoke_verified_exact_architecture_projection_inventory",
        "base_vision_projector_embeddings_head_frozen": True, "base_load_dtype": "float32_before_nf4",
        "assistant_target_only": True, "prompt_image_pad_labels": -100, "gradient_checkpointing": True,
        "optimizer": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0,
        "text_learning_rate": 2e-4, "vqa_learning_rate": 1e-4, "scheduler": "linear_decay",
        "warmup_steps": 0, "gradient_clip_norm": 1.0, "checkpoint": "one_pass_final_only",
        "fresh_process_final_reload_required": True, "seed_order_rule": "canonical_hash([seed,sample_id]) ascending; sample_id tie",
        "first_seed": SEEDS[0], "additional_seeds": SEEDS[1:], "additional_seed_execution_requires_family_selection": True,
        "main_worker_implemented_or_validated": False}
    plan = {"supported_cells": [r["cell_id"] for r in supported], "unsupported_cells": [r for r in audit["cells"] if r["status"] != "SUPPORTED"],
        "axes": axes, "per_cell_run_plan": native, "recipe": recipe, "protection": protection,
        "protected_eval_metadata_read": True, "eval_answers_used_for_selection": False, "eval_scores_read": False,
        "raw_parent_bytes_and_order_preserved": True, "token_budgets_equalized": False,
        "comparison_cohort_frozen": True, "later_failure_may_widen_cohort": False,
        "v6_main_training_completed": False, "gpu_work_performed": False, "tokenizer_executed": False}
    plan["freeze_id"] = canonical_hash(plan)
    return plan, payloads


def build_config(manifest_path, state_path, output_dir, *, protection, repo=None, success_receipt=None):
    manifest_path, state_path, output_dir = map(_absolute, (manifest_path, state_path, output_dir))
    manifest, state = gate._active(state_path)
    _require(_absolute(state["manifest_path"]).resolve() == manifest_path.resolve(), "PLANNING_MANIFEST_NOT_ACTIVE")
    record = state["jobs"][JOB_ID]
    _require(record.get("status") == "IMPLEMENTATION_PENDING" and all(type(record.get(k)) is int and record[k] == 0
        for k in ("attempts", "failure_count", "recovery_attempts")) and not record.get("history"), "ONLY_UNSTARTED_COMMON_GATE_MAY_BE_PLANNED")
    snapshot = _snapshot(manifest, state)
    plan, _ = _plan(snapshot, protection)
    _require(_equal(_snapshot(*gate._active(state_path)), snapshot), "PREREQUISITES_CHANGED_DURING_FREEZE")
    root = _absolute(manifest["campaign_root"])
    receipt = _absolute(success_receipt or root / "receipts" / (JOB_ID + ".json"))
    for path in (output_dir, receipt):
        gate._private(path, root)
    files, source = source_identity(repo)
    config = {"schema_version": SCHEMA, "scope": SCOPE, "job_id": JOB_ID, "run_id": JOB_ID,
        "uses_gpu": False, "campaign_id": manifest["campaign_id"], "state_path": str(state_path),
        "output_dir": str(output_dir), "success_receipt": str(receipt), "source_files": files, "source_hash": source,
        "registration_manifest": {"path": str(manifest_path), "sha256": file_hash(manifest_path)},
        "prerequisites": snapshot, "protection_references": protection, "freeze_id": plan["freeze_id"],
        "data_hash": canonical_hash({"snapshot": snapshot, "protection": protection, "freeze_id": plan["freeze_id"]}),
        "model_hash": canonical_hash(plan["per_cell_run_plan"])}
    config["config_hash"] = canonical_hash(config)
    return config


def activation_manifest(config_path, manifest, *, python=None, repo=None):
    config_path = _absolute(config_path)
    config = _read(config_path)
    _require(config["config_hash"] == canonical_hash({k: v for k, v in config.items() if k != "config_hash"}), "CONFIG_HASH_MISMATCH")
    _require(_equal(_snapshot(manifest, _read(config["state_path"])), config["prerequisites"]), "ACTIVATION_PREREQUISITE_MISMATCH")
    files, source = source_identity(repo)
    _require(files == config["source_files"] and source == config["source_hash"], "ACTIVATION_SOURCE_MISMATCH")
    revised = copy.deepcopy(manifest)
    job = next(j for j in revised["jobs"] if j["job_id"] == JOB_ID)
    job.update(implementation_pending=False, uses_gpu=False, gpu_lock_mode="wrapper",
        argv=[str(python or sys.executable), "-m", "mediflow_medvlm.campaign_common_train_freeze", "--config", str(config_path), "--receipt", config["success_receipt"]],
        cwd=str(Path(repo).resolve() if repo else Path(__file__).resolve().parents[2]),
        source_files=files, **{k: config[k] for k in HASH_FIELDS}, success_receipt=config["success_receipt"])
    validate_manifest(revised)
    return revised


def _validate(config):
    _require(config.get("schema_version") == SCHEMA and config.get("scope") == SCOPE
        and config.get("job_id") == config.get("run_id") == JOB_ID and config.get("uses_gpu") is False
        and config.get("config_hash") == canonical_hash({k: v for k, v in config.items() if k != "config_hash"}), "CONFIG_IDENTITY_MISMATCH")
    _require(_equal(source_identity(), (config["source_files"], config["source_hash"])), "SOURCE_CHANGED")
    _ref(config["registration_manifest"])
    manifest, state = gate._active(config["state_path"])
    snapshot = _snapshot(manifest, state)
    _require(_equal(snapshot, config["prerequisites"]), "TERMINAL_PREREQUISITES_CHANGED")
    plan, payloads = _plan(snapshot, config["protection_references"])
    _require(plan["freeze_id"] == config["freeze_id"] and config["model_hash"] == canonical_hash(plan["per_cell_run_plan"])
        and config["data_hash"] == canonical_hash({"snapshot": snapshot, "protection": config["protection_references"], "freeze_id": plan["freeze_id"]}), "FROZEN_PLAN_IDENTITY_MISMATCH")
    _require(_equal(_snapshot(*gate._active(config["state_path"])), snapshot), "PREREQUISITES_CHANGED_DURING_FREEZE")
    for k in IDENTITY_KEYS:
        supplied = os.environ.get("MEDVLM_SUPERVISOR_" + k.upper())
        _require(supplied is None or supplied == config[k], "SUPERVISOR_IDENTITY_MISMATCH")
    for k in ("output_dir", "success_receipt"):
        gate._private(_absolute(config[k]), _absolute(manifest["campaign_root"]))
    return plan, payloads


def _completed(config):
    output = _absolute(config["output_dir"])
    marker = _read(output / "completion.json")
    _require(marker.get("status") == "SUCCEEDED" and marker.get("scope") == SCOPE
        and marker.get("freeze_id") == config["freeze_id"] and all(marker.get(k) == config[k] for k in IDENTITY_KEYS), "EXISTING_OUTPUT_IDENTITY_MISMATCH")
    _require({p.name for p in output.iterdir()} == OUTPUT_FILES | {"completion.json"}
        and not any(p.is_symlink() for p in output.iterdir()), "EXISTING_OUTPUT_ARTIFACT_SET_MISMATCH")
    _require(marker["artifacts"] == [{"path": str(output / n), "sha256": file_hash(output / n)} for n in sorted(OUTPUT_FILES)]
        and _equal(_read(output / "config.json"), config), "EXISTING_OUTPUT_CHANGED")
    return {**marker, "artifacts": marker["artifacts"] + [{"path": str(output / "completion.json"), "sha256": file_hash(output / "completion.json")}]}


def _summary(plan):
    value = {k: v for k, v in plan.items() if k not in {"axes", "per_cell_run_plan"}}
    value["axes"] = {axis: {k: v for k, v in axis_plan.items() if k not in
        {"sample_ids", "source_positions", "per_model", "training_orders", "excluded"}}
        for axis, axis_plan in plan["axes"].items()}
    return value


def run_gate(config_path, receipt_path):
    config_path, receipt_path = _absolute(config_path), _absolute(receipt_path)
    config, initial = _read(config_path), file_hash(config_path)
    _require(str(receipt_path) == config["success_receipt"], "RECEIPT_PATH_MISMATCH")
    plan, payloads = _validate(config)
    output = _absolute(config["output_dir"])
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with output.with_name(output.name + ".worker.lock").open("a") as lock:
        os.chmod(lock.name, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if output.exists():
            result = _completed(config)
            _require(_equal(_read(output / "freeze.json"), plan)
                and _equal({k: v for k, v in _read(output / "summary.json").items() if k != "finished_at"}, _summary(plan))
                and all(file_hash(output / (axis + ".train.jsonl")) == hashlib.sha256(payload).hexdigest()
                        for axis, payload in payloads.items())
                and result.get("supported_cells") == plan["supported_cells"]
                and result.get("common_train_freeze_decided") is True
                and result.get("v6_main_training_completed") is False and result.get("campaign_terminal") is False,
                "EXISTING_OUTPUT_RECOMPUTATION_MISMATCH")
            _require(file_hash(config_path) == initial, "CONFIG_CHANGED_DURING_FREEZE")
            _require(not receipt_path.exists() or _equal(_read(receipt_path), result), "EXISTING_RECEIPT_MISMATCH")
            if not receipt_path.exists():
                atomic_json(receipt_path, result)
            return result
        _require(not receipt_path.exists(), "RECEIPT_WITHOUT_OUTPUT")
        staging = Path(tempfile.mkdtemp(prefix=output.name + ".candidate-", dir=output.parent))
        atomic_json(staging / "config.json", config)
        atomic_json(staging / "freeze.json", plan)
        for axis, payload in payloads.items():
            path = staging / (axis + ".train.jsonl")
            with path.open("xb") as stream:
                os.chmod(path, 0o600)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        summary = _summary(plan)
        summary["finished_at"] = utcnow()
        atomic_json(staging / "summary.json", summary)
        checked, checked_payloads = _validate(config)
        _require(_equal(checked, plan) and checked_payloads == payloads and file_hash(config_path) == initial, "INPUT_CHANGED_DURING_FREEZE")
        result = {"schema_version": "medvlm.v6.common-train-freeze-receipt.v1", "status": "SUCCEEDED", "scope": SCOPE,
            **{k: config[k] for k in IDENTITY_KEYS}, "freeze_id": plan["freeze_id"], "supported_cells": plan["supported_cells"],
            "common_train_freeze_decided": True, "v6_main_training_completed": False, "campaign_terminal": False,
            "artifacts": [{"path": str(output / n), "sha256": file_hash(staging / n)} for n in sorted(OUTPUT_FILES)], "finished_at": utcnow()}
        atomic_json(staging / "completion.json", result)
        os.rename(staging, output)
        fd = os.open(output.parent, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        result = _completed(config)
        atomic_json(receipt_path, result)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = run_gate(args.config, args.receipt)
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "exception_type": type(exc).__name__,
            **({"reason_code": str(exc)} if isinstance(exc, gate.GateError) else {})}), file=sys.stderr)
        return 1
    print(json.dumps({"status": result["status"], "freeze_id": result["freeze_id"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
