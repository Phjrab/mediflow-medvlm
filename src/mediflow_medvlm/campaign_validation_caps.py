"""Finite validation-only cap calibration: one model, one role, one GPU job.

The Supervisor owns the model queue and inherited GPU/worker leases. CPU planning
and receipt verification never load GPU libraries, read test rows, mutate the
queue, or record GPU usage. A GPU CLI launch accounts its own single session.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import fcntl
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback

from . import campaign_recorded_first_seed as first
from . import campaign_preflight_gate as gate
from .campaign_supervisor import HASH_FIELDS, TERMINAL, atomic_json, canonical_hash, file_hash, validate_manifest
from .campaign_v6 import CELLS, cell_spec
from .runtime import BudgetLedger, ResourcePolicy, utcnow

SCHEMA = "medvlm.v6.validation-cap-config.v1"
SCOPE = "VALIDATION_ONLY_NATIVE_CAP_CALIBRATION"
PROTOCOL_SCHEMA = "medvlm.v6.validation-cap-protocol.v1"
IDENTITY_KEYS = gate.IDENTITY_KEYS
ROLES = ("base", "tuned")
INITIAL_CAPS = {"medical_mcq": 64, "context_qa": 64, "medical_qa": 256, "medical_vqa": 256}
MAXIMUM_CAPS = {"medical_mcq": 128, "context_qa": 128, "medical_qa": 512, "medical_vqa": 512}
INITIAL_IDS = [f"v6-caps-initial-{c[0]}-{role}" for c in CELLS for role in ROLES]
SOURCE_FILES = tuple(dict.fromkeys((*first.SOURCE_FILES, "campaign_validation_caps.py",
    "evaluation/__init__.py", "evaluation/v2.py", "evaluation/v6_native.py")))
_require, _read, _absolute, _equal = gate._require, gate._read, gate._absolute, gate._equal
SELECTION_NAMESPACE = "medvlm-v6-recorded-cap-calibration-v1/20261023/"


class CalibrationFatal(RuntimeError):
    """The current model cannot continue. Its supervisor siblings remain runnable."""


def source_identity(repo=None):
    package = Path(__file__).resolve().parent if repo is None else Path(repo) / "src/mediflow_medvlm"
    files = {"src/mediflow_medvlm/" + name: file_hash(package / name) for name in SOURCE_FILES}
    return files, canonical_hash(files)


def _reference(path):
    path = _absolute(path)
    _require(not path.is_symlink() and path.is_file(), "IMMUTABLE_REGULAR_FILE_REQUIRED")
    return {"path": str(path), "sha256": file_hash(path)}


def _ref(reference):
    _require(_reference(reference["path"]) == reference, "REFERENCED_BYTES_CHANGED")
    return _read(reference["path"])


def _rehash_receipt(reference):
    receipt = _ref(reference)
    _require(receipt.get("status") == "SUCCEEDED" and bool(receipt.get("artifacts")), "SUCCESS_RECEIPT_REQUIRED")
    artifacts = receipt["artifacts"]
    _require(len({a["path"] for a in artifacts}) == len(artifacts), "DUPLICATE_RECEIPT_ARTIFACT")
    for artifact in artifacts:
        _require(_reference(artifact["path"]) == artifact, "RECEIPT_ARTIFACT_CHANGED")
    return receipt


def _stratum(row):
    if row["source_id"] == "aihub_71875":
        return "71875_" + {"1": "mcq", "2": "short", "3": "long"}[str(row["metadata"]["q_type"])]
    if row["source_id"] == "slake":
        return "slake_" + row["metadata"]["answer_type"].lower()
    return row["source_id"]


def _validation_rows(reference, expected_count):
    path = _absolute(reference["path"])
    _require(path.name == "validation.jsonl" and not path.is_symlink()
        and file_hash(path) == reference["sha256"], "ONLY_EXACT_FROZEN_VALIDATION_FILE_ALLOWED")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    ids = [row["sample_id"] for row in rows]
    _require(len(rows) == expected_count and len(set(ids)) == len(ids)
        and all(row.get("split") == "validation" for row in rows), "VALIDATION_COUNT_SPLIT_OR_IDS_MISMATCH")
    return rows


def _proposal(path):
    reference = _reference(path)
    proposal = _read(path)
    _require(proposal.get("schema_version") == "medvlm.recorded-validation-calibration-ids.proposal.v1"
        and proposal.get("proposal_hash") == canonical_hash({k: v for k, v in proposal.items() if k != "proposal_hash"})
        and proposal.get("validation_only") is True and proposal.get("test_files_read") is False
        and proposal.get("selection_uses_question_or_answer_content") is False
        and proposal.get("seed") == 20261023 and proposal.get("initial_caps") == INITIAL_CAPS
        and proposal.get("maximum_caps") == MAXIMUM_CAPS, "CALIBRATION_PROPOSAL_POLICY_MISMATCH")
    groups = {}
    for axis, count in (("text", 1516), ("vqa", 512)):
        ref = proposal["references"][axis]
        rows = _validation_rows({k: ref[k] for k in ("path", "sha256")}, count)
        grouped = defaultdict(list)
        for row in rows: grouped[_stratum(row)].append(row)
        for name, group in grouped.items():
            chosen = sorted(group, key=lambda row: (
                __import__("hashlib").sha256((SELECTION_NAMESPACE + row["sample_id"]).encode("utf-8")).hexdigest(),
                row["sample_id"]))[:min(64, len(group))]
            ids = [row["sample_id"] for row in chosen]
            groups[name] = {"axis": axis, "task_type": chosen[0]["task_type"],
                "full_validation_count": len(group), "calibration_count": len(ids),
                "ordered_sample_ids": ids, "ordered_ids_sha256": canonical_hash(ids)}
    _require(_equal(groups, proposal["strata"]), "PREDECLARED_CALIBRATION_IDS_CHANGED")
    return proposal, reference


def build_protocol(ids_proposal_path, first_seed_receipt_path, *, repo=None):
    """Return a frozen initial scientific protocol before any cap prediction."""
    proposal, ids_reference = _proposal(ids_proposal_path)
    receipt_reference = _reference(first_seed_receipt_path)
    aggregate = _rehash_receipt(receipt_reference)
    _require(aggregate.get("scope") == first.SCOPE and aggregate.get("first_seed_stage_terminal") is True
        and aggregate.get("campaign_terminal") is False, "RECORDED_FIRST_SEED_TERMINAL_AUDIT_REQUIRED")
    config_artifact = next(a for a in aggregate["artifacts"] if Path(a["path"]).name == "config.json")
    aggregate_config = _ref(config_artifact)
    # This audit binds the original freeze and exact terminal training configs;
    # semantic full-pass/final/reload proof is in the aggregate's hashed summary.
    _require(aggregate_config.get("config_hash") == canonical_hash({k: v for k, v in aggregate_config.items() if k != "config_hash"}),
             "FIRST_SEED_CONFIG_HASH_MISMATCH")
    active_manifest, active_state = gate._active(aggregate_config["state_path"])
    actual_inputs, _ = first._inputs(active_manifest, active_state)
    _require(_equal(actual_inputs, aggregate_config["prerequisites"]), "FIRST_SEED_TERMINAL_PROOF_CHANGED")
    summary_artifact = next(a for a in aggregate["artifacts"] if Path(a["path"]).name == "summary.json")
    summary = _ref(summary_artifact)
    _require(summary.get("aggregate_audit_succeeded") is True and summary.get("first_seed_stage_terminal") is True,
             "RECORDED_TERMINAL_SUMMARY_REQUIRED")
    models = {}
    for row in summary["cells"]:
        cell = row["cell_id"]
        child = aggregate_config["prerequisites"]["children"]["v6-first-seed-" + cell]
        config = _ref(child["config"])
        if row["status"] != "SUCCEEDED":
            models[cell] = {"training_status": row["status"], "config": child["config"],
                            "eligible_for_calibration": False}
            continue
        _require(all(row.get(k) is True for k in ("final_checkpoint_verified", "full_pass_verified",
            "fresh_process_reload_verified")), "AUDITED_FINAL_TRAINING_REQUIRED")
        _rehash_receipt(child["receipt"])
        output = _absolute(config["output_dir"])
        checkpoint_ref = _reference(output / "checkpoint.json")
        checkpoint = _ref(checkpoint_ref)
        _require(checkpoint.get("final_checkpoint") is True and checkpoint.get("complete") is True,
                 "ONE_PASS_FINAL_CHECKPOINT_REQUIRED")
        models[cell] = {"training_status": "SUCCEEDED", "eligible_for_calibration": True,
            "config": child["config"], "receipt": child["receipt"], "checkpoint": checkpoint_ref,
            "metrics": _reference(output / "metrics.json"), "reload": _reference(output / "reload.json"),
            "final_selection": _reference(output / "final_selection.json"),
            "model_hash": config["model_hash"], "revision": config["model"]["revision"],
            "adapter_sha256": checkpoint["adapter_sha256"], "adapter_files": checkpoint["adapter_files"],
            "processor_files": checkpoint["processor_files"]}
    _require(list(models) == [c[0] for c in CELLS], "EXACT_TEN_TRAINED_CELL_REFERENCES_REQUIRED")
    files, source = source_identity(repo)
    protocol = {"schema_version": PROTOCOL_SCHEMA, "scope": SCOPE,
        "campaign_id": aggregate_config["prerequisites"]["campaign_id"],
        "campaign_root": aggregate_config["prerequisites"]["campaign_root"],
        "first_seed_receipt": receipt_reference, "first_seed_config": config_artifact,
        "calibration_ids_proposal": ids_reference, "calibration_ids_proposal_hash": proposal["proposal_hash"],
        "validation_references": proposal["references"], "calibration_strata": proposal["strata"],
        "selection_recipe": {"namespace": SELECTION_NAMESPACE, "rank_hash": "sha256 UTF-8(namespace + sample_id)",
            "tie_break": "sample_id Unicode lexicographic", "count_per_stratum": "min(64, frozen stratum count)",
            "ordered_ids_hash": "sha256 UTF-8 JSON ensure_ascii=False, sort_keys=True, separators=(',', ':')"},
        "task_grouping": "native task_type, common medical_qa cap across short/long/MedQuAD and common VQA cap across OPEN/CLOSED",
        "initial_caps": INITIAL_CAPS.copy(), "maximum_caps": MAXIMUM_CAPS.copy(),
        "bump_rule": "any model and role: length_terminated_count*100 > scheduled_task_count; exactly one universal bump per native task_type",
        "eval_context": 4096, "train_context": 2048, "full_input_required": True,
        "reserve_generation_budget": True, "decode": {"do_sample": False, "num_beams": 1,
            "use_cache": True, "max_time": None, "warmup_requests": 0},
        "prompt_policy": "medical-vqa-short-answer.v2", "fixed_template_date": "2026-11-14",
        "thinking": "native disabled control and raw continuation nonempty-think-span rejection",
        "planned_failures_stay_in_denominator": True, "generation_score_validity_check": True,
        "logits_check_policy": "processed current-step scores: reject NaN/+inf/no finite candidate per batch row, permit -inf masks; raw logits finiteness unverified; remove_invalid_values=False",
        "test_files_read": False,
        "selection_scores_computed": False, "models": models,
        "source_files": files, "source_hash": source, "resource_policy": ResourcePolicy().as_dict()}
    protocol["protocol_hash"] = canonical_hash(protocol)
    return protocol


def validate_protocol(protocol, *, repo=None):
    _require(protocol.get("schema_version") == PROTOCOL_SCHEMA and protocol.get("scope") == SCOPE
        and protocol.get("protocol_hash") == canonical_hash({k: v for k, v in protocol.items() if k != "protocol_hash"})
        and (protocol.get("source_files"), protocol.get("source_hash")) == source_identity(repo),
        "VALIDATION_PROTOCOL_SOURCE_OR_HASH_MISMATCH")
    expected = build_protocol(protocol["calibration_ids_proposal"]["path"], protocol["first_seed_receipt"]["path"], repo=repo)
    _require(_equal(expected, protocol), "FROZEN_VALIDATION_PROTOCOL_CHANGED")
    return protocol


def _selected(protocol, axis, task_types=None):
    rows = _validation_rows({k: protocol["validation_references"][axis][k] for k in ("path", "sha256")},
                            1516 if axis == "text" else 512)
    by_id = {row["sample_id"]: row for row in rows}
    selected = []
    for name in sorted(protocol["calibration_strata"]):
        group = protocol["calibration_strata"][name]
        if group["axis"] != axis or (task_types is not None and group["task_type"] not in task_types): continue
        selected.extend((by_id[sample_id], name) for sample_id in group["ordered_sample_ids"])
    return selected


def build_config(protocol_path, cell_id, role, output_dir, *, repo=None, success_receipt=None,
                 calibration_round="initial", decision_reference=None):
    protocol_ref = _reference(protocol_path)
    protocol = validate_protocol(_ref(protocol_ref), repo=repo)
    _require(cell_id in [c[0] for c in CELLS] and role in ROLES
        and calibration_round == "initial", "CALIBRATION_INITIAL_ROUND_ONLY")
    spec = cell_spec(cell_id)
    model_ref = protocol["models"][cell_id]
    _require(model_ref["eligible_for_calibration"] is True, "CALIBRATION_REQUIRES_AUDITED_TRAINING_SUCCESS")
    train = _ref(model_ref["config"])
    selected = _selected(protocol, spec["task"])
    job_id = f"v6-caps-{calibration_round}-{cell_id}-{role}"
    root = _absolute(protocol["campaign_root"])
    output = _absolute(output_dir)
    receipt = _absolute(success_receipt or root / "receipts" / (job_id + ".json"))
    for path in (output, receipt): gate._private(path, root)
    _require(decision_reference is None, "CAP_INITIAL_DECISION_REFERENCE_FORBIDDEN")
    data = {"protocol": protocol_ref, "protocol_hash": protocol["protocol_hash"],
        "validation": {k: protocol["validation_references"][spec["task"]][k] for k in ("path", "sha256")},
        "ordered_sample_ids": [row["sample_id"] for row, _ in selected],
        "ordered_sample_ids_sha256": canonical_hash([row["sample_id"] for row, _ in selected]),
        "partition": "validation", "image_root": train["data"]["image_root"],
        "first_seed_receipt": protocol["first_seed_receipt"], "trained_cell": model_ref,
        "decision": decision_reference}
    files, source = source_identity(repo)
    data_root = Path(train["gpu_ledger_path"]).parent.parent
    value = {"schema_version": SCHEMA, "scope": SCOPE, "phase": "validation_caps_calibration", **spec,
        "job_id": job_id, "run_id": job_id, "role": role, "calibration_round": calibration_round,
        "seed": 20261023, "uses_gpu": True, "device": "cuda", "eval_context": 4096,
        "output_dir": str(output), "success_receipt": str(receipt), "data": data,
        "model": train["model"], "decode": protocol["decode"], "fixed_template_date": "2026-11-14",
        "local_files_only": True, "trust_remote_code": False, "resource_policy": ResourcePolicy().as_dict(),
        "gpu_ledger_path": str(data_root / "runtime/gpu_budget.json"),
        "supervisor_lock_paths": [str(data_root / "v6_v7/worker.lock"), str(data_root / "runtime/medvlm_gpu.lock")],
        "source_files": files, "source_hash": source, "data_hash": canonical_hash(data),
        "model_hash": canonical_hash({"model": train["model"], "role": role,
            "final_checkpoint": model_ref["checkpoint"] if role == "tuned" else None})}
    value["config_hash"] = canonical_hash(value)
    return value


def validate_config(config, *, repo=None):
    _require(config.get("schema_version") == SCHEMA and config.get("config_hash") == canonical_hash(
        {k: v for k, v in config.items() if k != "config_hash"}), "CAP_CONFIG_SCHEMA_OR_HASH_MISMATCH")
    expected = build_config(config["data"]["protocol"]["path"], config["cell_id"], config["role"], config["output_dir"],
        repo=repo, success_receipt=config["success_receipt"], calibration_round=config["calibration_round"],
        decision_reference=config["data"].get("decision"))
    _require(_equal(config, expected), "CAP_CONFIG_FROZEN_IDENTITY_CHANGED")
    return config


def job_from_config(config_path, *, repo=None, python=None):
    path, config = _absolute(config_path), _read(config_path)
    validate_config(config, repo=repo)
    return {"job_id": config["job_id"], "run_id": config["run_id"], "phase": config["phase"],
        **cell_spec(config["cell_id"]), "role": config["role"], "calibration_round": config["calibration_round"],
        "seed": config["seed"], "dependencies": [first.JOB_ID], "dependency_policy": "success",
        "implementation_pending": False, "uses_gpu": True, "requires_gpu": True, "gpu_lock_mode": "wrapper",
        "argv": [str(python or sys.executable), "-m", "mediflow_medvlm.campaign_validation_caps", "--config",
                 str(path), "--receipt", config["success_receipt"]],
        "cwd": str(Path(repo).resolve() if repo is not None else Path(__file__).resolve().parents[2]),
        "success_receipt": config["success_receipt"], "source_files": config["source_files"],
        **{k: config[k] for k in HASH_FIELDS}}


def _task_counts(rows):
    counts = {}
    for task in sorted({r["task_type"] for r in rows}):
        group = [r for r in rows if r["task_type"] == task]
        counts[task] = {"scheduled": len(group), "successful": sum(r["status"] == "SUCCEEDED" for r in group),
            "technical_failures": sum(r["status"] != "SUCCEEDED" for r in group),
            "length_terminated": sum(r.get("finish_reason") == "length" for r in group)}
    return counts


def cap_decision_from_counts(protocol, conditions):
    """Pure once-bump rule; callers must first audit all twenty terminal jobs.

    This helper cannot certify receipts, freeze a final protocol, register a
    rerun, or enable test. The unstarted CPU cap gate must do those audits.
    """
    _require(set(conditions) == set(INITIAL_IDS), "EXACT_TWENTY_CAP_CONDITIONS_REQUIRED")
    bumped, failed = set(), []
    for cell in CELLS:
        expected = Counter(row["task_type"] for row, _ in _selected(protocol, cell_spec(cell[0])["task"]))
        for role in ROLES:
            job_id = f"v6-caps-initial-{cell[0]}-{role}"
            condition = conditions[job_id]
            _require(condition.get("status") in TERMINAL,
                     "CAP_CONDITION_NOT_TERMINAL")
            if condition["status"] != "SUCCEEDED":
                failed.append(job_id)
                continue
            counts = condition["task_counts"]
            _require(set(counts) == set(expected), "CAP_TASK_GROUPING_CHANGED")
            for task, count in counts.items():
                _require(set(count) == {"scheduled", "successful", "technical_failures", "length_terminated"}
                    and all(type(v) is int and v >= 0 for v in count.values())
                    and count["scheduled"] == expected[task]
                    and count["successful"] + count["technical_failures"] == expected[task]
                    and count["length_terminated"] <= expected[task], "CAP_CALIBRATION_DENOMINATOR_CHANGED")
                if count["length_terminated"] * 100 > count["scheduled"]:
                    bumped.add(task)
    return {"protocol_hash": protocol["protocol_hash"], "caps": {
        task: MAXIMUM_CAPS[task] if task in bumped else INITIAL_CAPS[task] for task in INITIAL_CAPS},
        "bumped_task_types": sorted(bumped), "failed_conditions": failed,
        "universal_across_models_and_roles": True, "bump_rounds": 1 if bumped else 0,
        "requires_rerun_before_full_dev": bool(bumped), "cap_protocol_frozen": False,
        "scope": "UNEXECUTED_CAP_DECISION_HELPER_ONLY"}


def _caps(config, protocol):
    _require(config["calibration_round"] == "initial" and config["data"].get("decision") is None,
             "CAP_INITIAL_ROUND_ONLY_PENDING_AUDITED_RERUN_IMPLEMENTATION")
    return protocol["initial_caps"], None


def _failure(row, stratum, cap, reason, *, fatal=False):
    return {"sample_id": row["sample_id"], "task_type": row["task_type"], "stratum": stratum,
        "input_sha256": canonical_hash(row["input"]), "status": "TECHNICAL_FAILURE",
        "reason_code": reason, "model_fatal": fatal, "max_new_tokens": cap,
        "finish_reason": None, "generated_tokens": 0, "generated_ids": []}


class GenerationScoreGuard:
    """Validate current-step scores without retaining a vocabulary tensor.

    Negative infinity may intentionally mask a token. These processed scores do
    not establish that every raw forward logit was finite before processors.
    """
    def __init__(self): self.checked_steps = 0

    def __call__(self, input_ids, scores):
        import torch
        if bool((torch.isnan(scores) | torch.isposinf(scores)).any().item()) or not bool(
                torch.isfinite(scores).any(dim=-1).all().item()):
            raise ValueError("INVALID_CURRENT_STEP_GENERATION_SCORES")
        self.checked_steps += 1
        return scores


def _predict_one(bundle, row, stratum, cap, config):
    import torch
    from .evaluation.v2 import encode_evaluation_sample, generation_finish
    from .training.v6_native import verify_native_no_thinking_output
    base = _failure(row, stratum, cap, "UNINITIALIZED")
    started = time.monotonic()
    try:
        encoded, metadata = encode_evaluation_sample(row, bundle.processor, bundle.architecture,
            max_length=config["eval_context"] - cap, image_root=config["data"]["image_root"],
            prompt_policy="medical-vqa-short-answer.v2")
        length = int(encoded["input_ids"].shape[-1])
        _require(length + cap <= config["eval_context"], "CONTEXT_OVERFLOW")
        pixels = [k for k in encoded if any(word in k.lower() for word in ("pixel", "image", "vision", "video"))]
        actual_pixels = any("pixel_values" in key.lower() and hasattr(encoded[key], "numel")
                            and encoded[key].numel() > 0 for key in pixels)
        _require(actual_pixels if config["task"] == "vqa" else not pixels,
                 "VQA_PIXELS_REQUIRED" if config["task"] == "vqa" else "TEXT_IMAGE_TENSORS_FORBIDDEN")
        moved = {key: value.to(bundle.model.device) if hasattr(value, "to") else value for key, value in encoded.items()}
        tokenizer = getattr(bundle.processor, "tokenizer", bundle.processor)
        eos = getattr(bundle.model.generation_config, "eos_token_id", None)
        if eos is None: eos = tokenizer.eos_token_id
        pad = getattr(bundle.model.generation_config, "pad_token_id", None)
        if pad is None: pad = getattr(tokenizer, "pad_token_id", None)
    except Exception as exc:
        return {**base, "reason_code": "INPUT_" + type(exc).__name__ + ":" +
                (str(exc) if isinstance(exc, ValueError) else "ENCODING_FAILED"), "elapsed_seconds": time.monotonic() - started}
    try:
        torch.cuda.synchronize()
        begin_generate = time.monotonic()
        score_guard = GenerationScoreGuard()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            generated = bundle.model.generate(**moved, max_new_tokens=cap, do_sample=False,
                num_beams=1, use_cache=True, eos_token_id=eos, pad_token_id=pad,
                max_time=None, remove_invalid_values=False, logits_processor=[score_guard])
        torch.cuda.synchronize()
        continuation = generated[0, length:]
        ids = continuation.tolist()
        _require(bool(ids) and len(ids) <= cap and all(type(token) is int and token >= 0 for token in ids),
                 "EMPTY_OR_INVALID_GENERATED_TOKEN_SEQUENCE")
        _require(generated[0, :length].tolist() == encoded["input_ids"][0].tolist(),
                 "GENERATION_PROMPT_PREFIX_CHANGED")
        _require(score_guard.checked_steps == len(ids), "GENERATION_SCORE_TOKEN_COVERAGE_MISMATCH")
        generation_seconds = time.monotonic() - begin_generate
    except Exception as exc:
        reason = str(exc) if isinstance(exc, ValueError) and str(exc).startswith(("INVALID_CURRENT_STEP_", "GENERATION_SCORE_", "GENERATION_PROMPT_", "EMPTY_OR_INVALID_")) else type(exc).__name__
        return {**base, "reason_code": "GENERATION_" + reason, "generation_error_private": str(exc), "model_fatal": True,
            "elapsed_seconds": time.monotonic() - started}
    raw = tokenizer.decode(continuation, skip_special_tokens=False)
    text = tokenizer.decode(continuation, skip_special_tokens=True)
    finish = generation_finish(len(ids), ids[-1], cap, eos)
    try:
        thinking = verify_native_no_thinking_output(raw)
        status, reason = "SUCCEEDED", None
    except ValueError:
        thinking, status, reason = {"explicit_nonempty_thinking_detected": True}, "TECHNICAL_FAILURE", "NATIVE_THINKING_OUTPUT_PRESENT"
    return {**base, "status": status, "reason_code": reason, "model_fatal": False,
        "input_tokens": length, "generated_tokens": len(ids), "generated_ids": ids,
        "generated_ids_sha256": canonical_hash(ids), "generated_raw_text": raw, "generated_text": text,
        "finish_reason": finish, "eos_token_ids": eos, "input_metadata": metadata,
        "pixel_tensor_keys": pixels, "input_tensor_shapes": {k: list(v.shape) for k, v in encoded.items() if hasattr(v, "shape")},
        "input_tensor_dtypes": {k: str(v.dtype) for k, v in encoded.items() if hasattr(v, "dtype")},
        "no_thinking_output": thinking, "generation_scores_valid": True, "raw_forward_logits_finite_verified": False,
        "generation_scores_checked_steps": score_guard.checked_steps, "generation_seconds": generation_seconds,
        "elapsed_seconds": time.monotonic() - started}


def _execute(config, protocol, selected, caps, stream):
    from .evaluation.v6_native import load_evaluation_bundle
    import torch
    model_ref = protocol["models"][config["cell_id"]]
    train = _ref(model_ref["config"])
    metrics = _ref(model_ref["metrics"])
    rows, fatal = [], None
    try:
        bundle = load_evaluation_bundle(config["model"], config["cell_id"], role=config["role"],
            expected_contract=metrics["native_contract"], final_directory=train["output_dir"],
            expected_adapter_sha256=model_ref["adapter_sha256"], fixed_template_date=config["fixed_template_date"])
        torch.cuda.reset_peak_memory_stats()
    except Exception as exc:
        bundle, fatal = None, "MODEL_LOAD_" + type(exc).__name__
    for row, stratum in selected:
        cap = caps[row["task_type"]]
        try:
            result = _failure(row, stratum, cap, fatal, fatal=True) if fatal else _predict_one(bundle, row, stratum, cap, config)
        except Exception as exc:
            result = _failure(row, stratum, cap, "POSTPROCESS_" + type(exc).__name__, fatal=True)
        rows.append(result)
        stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        if result.get("model_fatal"): fatal = result["reason_code"]
        progress = os.environ.get("MEDVLM_SUPERVISOR_PROGRESS_PATH")
        if progress:
            atomic_json(progress, {**{k: config[k] for k in IDENTITY_KEYS}, "status": "RUNNING",
                "stage": "validation_cap_calibration", "processed": len(rows), "scheduled": len(selected),
                "technical_failures": sum(r["status"] != "SUCCEEDED" for r in rows), "updated_at": utcnow()})
    stream.flush()
    os.fsync(stream.fileno())
    if fatal: raise CalibrationFatal(fatal)
    return rows, {"native_contract": bundle.contract,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved()}


def _verify_rows(config, protocol, rows, selected, caps):
    _require([r.get("sample_id") for r in rows] == [r[0]["sample_id"] for r in selected],
             "CALIBRATION_ROWS_DROPPED_REORDERED_OR_DUPLICATED")
    from .evaluation.v2 import generation_finish
    for result, (row, stratum) in zip(rows, selected):
        cap = caps[row["task_type"]]
        _require(result.get("task_type") == row["task_type"] and result.get("stratum") == stratum
            and result.get("input_sha256") == canonical_hash(row["input"])
            and result.get("max_new_tokens") == cap and result.get("status") in {"SUCCEEDED", "TECHNICAL_FAILURE"}
            and result.get("model_fatal") is False, "CALIBRATION_ROW_POLICY_MISMATCH")
        ids = result.get("generated_ids", [])
        _require(len(ids) == result.get("generated_tokens") and len(ids) <= cap,
                 "CALIBRATION_TOKEN_COUNT_MISMATCH")
        if ids:
            _require(result.get("generated_ids_sha256") == canonical_hash(ids)
                and result.get("finish_reason") == generation_finish(len(ids), ids[-1], cap, result.get("eos_token_ids"))
                and result.get("input_tokens", 4097) + cap <= protocol["eval_context"], "CALIBRATION_FINISH_OR_CONTEXT_MISMATCH")
        if result["status"] == "SUCCEEDED":
            _require(bool(ids) and result.get("no_thinking_output", {}).get("explicit_nonempty_thinking_detected") is False
                and result.get("generation_scores_valid") is True
                and result.get("raw_forward_logits_finite_verified") is False
                and result.get("generation_scores_checked_steps") == len(ids),
                     "CALIBRATION_SUCCESS_WITHOUT_TOKEN_OR_THINKING_PROOF")
    return _task_counts(rows)


def verify_result(config, *, repo=None):
    validate_config(config, repo=repo)
    protocol = _ref(config["data"]["protocol"])
    caps, changed = _caps(config, protocol)
    selected = _selected(protocol, config["task"], changed)
    output = _absolute(config["output_dir"])
    result = _read(output / "completion.json")
    _require(result.get("status") == "SUCCEEDED" and result.get("scope") == SCOPE
        and all(result.get(k) == config[k] for k in IDENTITY_KEYS)
        and result.get("test_files_read") is False and result.get("campaign_terminal") is False
        and result.get("protocol_hash") == protocol["protocol_hash"], "CAP_COMPLETION_IDENTITY_MISMATCH")
    _require({p.name for p in output.iterdir()} == {"config.json", "predictions.private.jsonl", "metrics.json", "completion.json"}
        and not output.is_symlink() and not any(p.is_symlink() for p in output.iterdir())
        and _equal(_read(output / "config.json"), config), "CAP_COMPLETION_FILE_SET_CHANGED")
    artifacts = [_reference(output / name) for name in ("config.json", "metrics.json", "predictions.private.jsonl")]
    _require(result.get("artifacts") == artifacts, "CAP_COMPLETION_ARTIFACT_CHANGED")
    rows = [json.loads(line) for line in (output / "predictions.private.jsonl").read_text().splitlines()]
    counts = _verify_rows(config, protocol, rows, selected, caps)
    metrics = _read(output / "metrics.json")
    if selected:
        from .evaluation.v6_native import normalized_base_dtype_signature
        contract = metrics.get("native_contract", {})
        expected = _ref(protocol["models"][config["cell_id"]]["metrics"])["native_contract"]
        _require(all(_equal(contract.get(k), expected.get(k)) for k in
            ("architecture", "template", "thinking", "loader", "revision", "snapshot_path", "processor_class",
             "quantization", "native_runtime_compatibility"))
            and contract.get("weights_loaded") is True and contract.get("cuda_residency_verified") is True
            and contract.get("evaluation_role") == config["role"]
            and contract.get("parameter_storage", {}).get("trainable_parameters") == 0
            and normalized_base_dtype_signature(contract["parameter_storage"]) ==
                normalized_base_dtype_signature(expected["parameter_storage"]), "CAP_NATIVE_EVALUATION_CONTRACT_INVALID")
        _require(contract.get("evaluation_adapter_sha256") ==
            (protocol["models"][config["cell_id"]]["adapter_sha256"] if config["role"] == "tuned" else None)
            and (contract["parameter_storage"]["adapter_parameters"] > 0 if config["role"] == "tuned"
                 else contract["parameter_storage"]["adapter_parameters"] == 0), "CAP_ROLE_ADAPTER_PROOF_INVALID")
        account = metrics.get("gpu_accounting", {})
        _require(type(account.get("session_seconds")) in (int, float) and 0 <= account["session_seconds"] < float("inf")
            and account.get("ledger_path") == config["gpu_ledger_path"]
            and account.get("snapshot", {}).get("resource_policy") == ResourcePolicy().as_dict(),
            "CAP_GPU_ACCOUNTING_PROOF_INVALID")
    _require(metrics.get("task_counts") == counts and metrics.get("scheduled_count") == len(selected)
        and metrics.get("technical_failure_count") == sum(r["status"] != "SUCCEEDED" for r in rows)
        and result.get("scheduled_count") == len(selected)
        and result.get("skipped_no_task_cap_change") is (len(selected) == 0)
        and result.get("technical_complete") is all(r["status"] == "SUCCEEDED" for r in rows),
        "CAP_COMPLETION_DENOMINATOR_MISMATCH")
    return {**result, "artifacts": artifacts + [_reference(output / "completion.json")]}


def run_job(config_path, receipt_path):
    config_path, receipt_path = _absolute(config_path), _absolute(receipt_path)
    before, config = file_hash(config_path), _read(config_path)
    _require(str(receipt_path) == config["success_receipt"], "CAP_RECEIPT_PATH_MISMATCH")
    validate_config(config)
    # Mandatory inherited owner leases and exact execution hash environment.
    # This must precede CUDA import, model load and BudgetLedger.start.
    first.main_train.smoke._inherited_supervisor_fds(config)
    protocol = _ref(config["data"]["protocol"])
    caps, changed = _caps(config, protocol)
    selected = _selected(protocol, config["task"], changed)
    output = _absolute(config["output_dir"])
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with output.with_name(output.name + ".worker.lock").open("a") as lock:
        os.chmod(lock.name, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not output.exists():
            _require(not receipt_path.exists(), "CAP_RECEIPT_WITHOUT_OUTPUT")
            staging = Path(tempfile.mkdtemp(prefix=output.name + ".attempt-", dir=output.parent))
            old_umask = os.umask(0o077)
            ledger = None
            try:
                atomic_json(staging / "config.json", config)
                with (staging / "predictions.private.jsonl").open("x") as stream:
                    if selected:
                        ledger = BudgetLedger(config["gpu_ledger_path"], policy=ResourcePolicy())
                        ledger.start(config["model_id"], config["run_id"], "validation_cap_" + config["calibration_round"])
                        try: rows, native_metrics = _execute(config, protocol, selected, caps, stream)
                        finally: ledger.finish()
                    else:
                        rows, native_metrics = [], {"no_gpu_work_needed": True}
                        stream.flush()
                        os.fsync(stream.fileno())
                counts = _verify_rows(config, protocol, rows, selected, caps)
                metrics = {"task_counts": counts, "scheduled_count": len(selected),
                    "technical_failure_count": sum(r["status"] != "SUCCEEDED" for r in rows),
                    "gpu_accounting": {"session_seconds": ledger.session_seconds,
                        "ledger_path": config["gpu_ledger_path"], "snapshot": ledger.snapshot()} if ledger else None,
                    **native_metrics, "finished_at": utcnow()}
                atomic_json(staging / "metrics.json", metrics)
                validate_config(config)
                _require(file_hash(config_path) == before, "CAP_CONFIG_CHANGED_DURING_EXECUTION")
                result = {"schema_version": "medvlm.v6.validation-cap-receipt.v1", "status": "SUCCEEDED", "scope": SCOPE,
                    **{k: config[k] for k in IDENTITY_KEYS}, "cell_id": config["cell_id"], "role": config["role"],
                    "calibration_round": config["calibration_round"], "protocol_hash": protocol["protocol_hash"],
                    "scheduled_count": len(selected), "skipped_no_task_cap_change": not selected,
                    "technical_complete": all(r["status"] == "SUCCEEDED" for r in rows),
                    "test_files_read": False, "campaign_terminal": False,
                    "artifacts": [{"path": str(output / name), "sha256": file_hash(staging / name)}
                        for name in ("config.json", "metrics.json", "predictions.private.jsonl")]}
                atomic_json(staging / "completion.json", result)
                os.rename(staging, output)
                fd = os.open(output.parent, os.O_DIRECTORY)
                try: os.fsync(fd)
                finally: os.close(fd)
            except BaseException as exc:
                if staging.exists():
                    atomic_json(staging / "failure.private.json", {"status": "FAILED", "exception_type": type(exc).__name__,
                        "exception_message": str(exc), "traceback": traceback.format_exc(), "finished_at": utcnow()})
                raise
            finally:
                os.umask(old_umask)
        result = verify_result(config)
        _require(file_hash(config_path) == before and (not receipt_path.exists()
            or _equal(_read(receipt_path), result)), "CAP_RECEIPT_OR_CONFIG_CHANGED")
        if not receipt_path.exists(): atomic_json(receipt_path, result)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run_job(args.config, args.receipt)
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "scope": SCOPE, "exception_type": type(exc).__name__,
            **({"reason_code": str(exc)} if isinstance(exc, (gate.GateError, CalibrationFatal)) else {})}))
        return 1
    print(json.dumps({"status": result["status"], "technical_complete": result["technical_complete"],
                      "scheduled_count": result["scheduled_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
