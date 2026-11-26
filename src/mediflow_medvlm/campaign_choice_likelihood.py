"""Separate native, validation-only choice conditional log-likelihood auxiliary.

No generation, training, selection, test reader, or telemetry import is used.
EOS is excluded: the event scored is the exact label text immediately after the
validated native generation prefix, before any native termination/control suffix.
This fixed convention avoids making a label comparison depend on family-specific
termination strings. Unsupported BPE boundaries fail explicitly, without repair.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import traceback

from . import campaign_full_dev as dev
from .campaign_supervisor import HASH_FIELDS, atomic_json, canonical_hash, file_hash
from .runtime import BudgetLedger, ResourcePolicy, utcnow, canonical_hash as encoder_input_hash

SCHEMA = "medvlm.v6.choice-likelihood-config.v1"
SCOPES = {"role": "VALIDATION_ONLY_NATIVE_CHOICE_LIKELIHOOD_AUXILIARY",
          "aggregate": "VALIDATION_ONLY_CHOICE_LIKELIHOOD_TERMINAL_AUDIT"}
TEXT_CELLS = tuple(c[0] for c in dev.CELLS if c[2] == "text")
ROLE_IDS = [f"v6-choice-likelihood-{cell}-{role}" for cell in TEXT_CELLS for role in ("base", "tuned")]
AGGREGATE_ID = "v6-choice-likelihood"
SOURCE_FILES = (*dev.SOURCE_FILES, "campaign_choice_likelihood.py")
IDENTITY_KEYS = dev.IDENTITY_KEYS
_require, _read, _absolute, _equal = dev._require, dev._read, dev._absolute, dev._equal
_reference, _ref = dev._reference, dev._ref
FLAGS = {"test_files_read": False, "test_unlocked": False, "family_selection_completed": False,
         "clinical_quality_certified": False, "full_dev_completed": False,
         "r2_full_evaluation_completed": False, "campaign_terminal": False,
         "generation_validity_affected": False}
PROCEDURE = {
    "event": "exact label text appended to native generation assistant prefix",
    "score": "sum of all conditional label-token log probabilities; causal shift; no length normalization",
    "eos_included": False, "termination_tokens_included": False,
    "eos_reason": "score exact label text before family-specific EOS/control suffix; fixed before execution",
    "boundary": "candidate processor IDs have exact native prompt prefix; continuation decodes to exact label",
    "completed_assistant_whitespace": "record native discrepancy; never silently insert or substitute whitespace",
    "label_order": "medical_mcq native input options iteration order; context_qa yes,no,maybe",
    "tie_policy": "exact numeric maximum; preserve all tied labels; choose first in declared label order",
    "context": 4096, "input_truncation": False, "generation_used": False,
    "supported": {"aihub_71875": "medical_mcq", "pubmedqa_labeled": "context_qa"},
    "classification_scheduled": 637, "classification_source_counts": {"aihub_71875": 587, "pubmedqa_labeled": 50},
    "text_all_planned": 1516, "unsupported_text": 879,
    "unsupported_vqa": {"status": "UNSUPPORTED", "reason": "choice likelihood is undefined for freeform/image QA", "validation_rows_per_role": 512, "model_roles": 8},
    "denominator": "all 637 planned classification rows, including technical failures; unsupported text rows reported separately",
    "interpretation": "format-reduced auxiliary comparison; not pure clinical capability or clinical truth",
}
TEXT_KEYS = {"input_ids", "attention_mask", "token_type_ids", "mm_token_type_ids"}


def relevant_source_files():
    return SOURCE_FILES


def source_identity(repo=None):
    package = Path(__file__).resolve().parent if repo is None else Path(repo)/"src/mediflow_medvlm"
    files = {"src/mediflow_medvlm/"+name: file_hash(package/name) for name in SOURCE_FILES}
    return files, canonical_hash(files)


def _native_inputs(role_config_path):
    """Bind the original registered successful role, never a regenerated recipe."""
    reference = _reference(role_config_path); child = _ref(reference)
    _require(child.get("kind") == "role" and child.get("cell_id") in TEXT_CELLS
             and child.get("role") in {"base", "tuned"}, "CHOICE_NATIVE_TEXT_ROLE_REQUIRED")
    manifest, state = dev.caps._active(child["state_path"])
    jobs = {job["job_id"]: job for job in manifest["jobs"]}
    job = jobs[child["job_id"]]; record = state["jobs"][child["job_id"]]
    registered, registered_ref = dev.caps._child_config(job, "campaign_full_dev")
    _require(registered_ref["sha256"] == reference["sha256"] and _equal(registered, child)
             and record.get("status") == "SUCCEEDED", "CHOICE_ORIGINAL_SUCCESSFUL_ROLE_REQUIRED")
    native_repo = job["cwd"]
    dev.validate_config(child, repo=native_repo)
    receipt = dev.caps.initial.gate._success(job, record)
    _require(_equal(dev.verify_result(child, repo=native_repo), receipt), "CHOICE_NATIVE_RESULT_CHANGED")
    aggregate_job = jobs[dev.AGGREGATE_ID]; aggregate_record = state["jobs"][dev.AGGREGATE_ID]
    _require(aggregate_record.get("status") == "SUCCEEDED", "CHOICE_NATIVE_AGGREGATE_SUCCESS_REQUIRED")
    aggregate, aggregate_ref = dev.caps._child_config(aggregate_job, "campaign_full_dev")
    aggregate_receipt = dev.caps.initial.gate._success(aggregate_job, aggregate_record)
    _require(aggregate_receipt.get("all_roles_succeeded") is True
             and registered_ref in aggregate["data"]["roles"], "CHOICE_FULL_DEV_MATRIX_REQUIRED")
    original = _ref(child["data"]["initial_protocol"])
    selected = dev._rows(original, "text")
    training, trained = dev._training(original, child["cell_id"])
    _require(trained == child["data"]["trained_cell"], "CHOICE_ORIGINAL_CHECKPOINT_CHANGED")
    proof = {"config": registered_ref, "receipt": _reference(job["success_receipt"]),
             "artifacts": copy.deepcopy(receipt["artifacts"]), "native_source_repo": native_repo,
             "aggregate_config": aggregate_ref, "aggregate_receipt": _reference(aggregate_job["success_receipt"]),
             "generation_technical_complete": receipt["technical_complete"],
             "initial_protocol": child["data"]["initial_protocol"], "trained_cell": trained,
             "validation": child["data"]["validation"], "cap_freeze": child["data"]["cap_freeze"]}
    return child, proof, original, selected, training


def labels_for(row):
    """Only task/source/input determine candidates; gold is never consulted."""
    _require(row.get("split") == "validation", "CHOICE_VALIDATION_ONLY")
    task, source = row["task_type"], row["source_id"]
    if task == "medical_mcq" and source == "aihub_71875":
        options = row["input"].get("options")
        _require(isinstance(options, (dict, list)) and len(options) >= 2, "CHOICE_OPTIONS_REQUIRED")
        labels = [str(key) for key in options] if isinstance(options, dict) else [str(i) for i in range(len(options))]
    elif task == "context_qa" and source == "pubmedqa_labeled":
        _require(not row["input"].get("options"), "CHOICE_UNEXPECTED_CONTEXT_OPTIONS")
        labels = ["yes", "no", "maybe"]
    else:
        return None
    _require(len(set(labels)) == len(labels) and all(label and label.strip() == label for label in labels), "CHOICE_INVALID_LABELS")
    return labels


def _coverage(selected):
    _require(len({row["sample_id"] for row, _ in selected}) == len(selected), "CHOICE_UNIQUE_IDS_REQUIRED")
    supported = [row for row, _ in selected if labels_for(row) is not None]
    counts = dict(Counter(row["source_id"] for row in supported))
    _require(len(selected) == PROCEDURE["text_all_planned"] and len(supported) == PROCEDURE["classification_scheduled"]
             and counts == PROCEDURE["classification_source_counts"], "CHOICE_EXACT_VALIDATION_DENOMINATORS_REQUIRED")
    return {"text_all_planned": len(selected), "classification_scheduled": len(supported),
            "classification_source_counts": counts, "unsupported_text": len(selected)-len(supported),
            "unsupported_vqa": copy.deepcopy(PROCEDURE["unsupported_vqa"])}


def _destination(root, output_dir, receipt, job_id):
    output, receipt = _absolute(output_dir), _absolute(receipt or root/"receipts"/(job_id+".json"))
    auxiliary = root/"auxiliary/choice_likelihood"
    for path in (output, receipt):
        dev.caps.initial.gate._private(path, root)
        _require(not path.is_symlink() and not any(p.is_symlink() for p in path.parents), "CHOICE_SYMLINK_DESTINATION_FORBIDDEN")
    _require(output.resolve().is_relative_to(auxiliary.resolve()) and output != auxiliary
             and receipt.parent == root/"receipts" and receipt.name == job_id+".json", "CHOICE_DEDICATED_OUTPUT_AND_RECEIPT_REQUIRED")
    return output, receipt


def prepare_config(full_dev_role_config_path, output_dir, *, repo=None, success_receipt=None):
    child, proof, original, selected, _ = _native_inputs(full_dev_role_config_path)
    job_id = f"v6-choice-likelihood-{child['cell_id']}-{child['role']}"
    output, receipt = _destination(_absolute(original["campaign_root"]), output_dir, success_receipt, job_id)
    _require(child["resource_policy"] == ResourcePolicy().as_dict(), "CHOICE_UNLIMITED_EXISTING_RESOURCE_POLICY_REQUIRED")
    data = {"native": proof, "ordered_sample_ids": [row["sample_id"] for row, _ in selected],
            "ordered_sample_ids_sha256": canonical_hash([row["sample_id"] for row, _ in selected]), "coverage": _coverage(selected)}
    files, source = source_identity(repo)
    config = {"schema_version": SCHEMA, "scope": SCOPES["role"], "kind": "role", "phase": "choice_likelihood_evaluation",
              **dev.cell_spec(child["cell_id"]), "role": child["role"], "job_id": job_id, "run_id": job_id,
              "seed": 20261023, "uses_gpu": True, "device": "cuda", "split": "validation", "eval_context": 4096,
              "state_path": child["state_path"], "output_dir": str(output), "success_receipt": str(receipt),
              "data": data, "procedure": copy.deepcopy(PROCEDURE), "model": child["model"], "model_hash": child["model_hash"],
              "fixed_template_date": child["fixed_template_date"], "precision": child["precision"],
              "local_files_only": True, "trust_remote_code": False,
              "resource_policy": child["resource_policy"], "gpu_ledger_path": child["gpu_ledger_path"],
              "supervisor_lock_paths": child["supervisor_lock_paths"], "source_files": files, "source_hash": source,
              "data_hash": canonical_hash(data)}
    config["config_hash"] = canonical_hash(config)
    return config


def prepare_aggregate_config(role_config_paths_ordered12, output_dir, *, repo=None, success_receipt=None):
    refs = [_reference(path) for path in role_config_paths_ordered12]; configs = [_ref(ref) for ref in refs]
    _require([config.get("job_id") for config in configs] == ROLE_IDS, "CHOICE_EXACT_ORDERED_TWELVE_ROLES_REQUIRED")
    for config in configs: validate_config(config, repo=repo)
    _require(len({c["state_path"] for c in configs}) == 1
             and all(c["data"]["native"]["aggregate_config"] == configs[0]["data"]["native"]["aggregate_config"] for c in configs), "CHOICE_COMMON_NATIVE_MATRIX_REQUIRED")
    original = _ref(configs[0]["data"]["native"]["initial_protocol"])
    output, receipt = _destination(_absolute(original["campaign_root"]), output_dir, success_receipt, AGGREGATE_ID)
    data = {"roles": refs, "initial_protocol": configs[0]["data"]["native"]["initial_protocol"],
            "coverage": configs[0]["data"]["coverage"]}
    files, source = source_identity(repo)
    config = {"schema_version": SCHEMA, "scope": SCOPES["aggregate"], "kind": "aggregate", "phase": "choice_likelihood",
              "job_id": AGGREGATE_ID, "run_id": AGGREGATE_ID, "cell_id": "ALL_TEXT_V6", "seed": 20261023,
              "uses_gpu": False, "device": "cpu", "state_path": configs[0]["state_path"], "output_dir": str(output),
              "success_receipt": str(receipt), "data": data, "procedure": copy.deepcopy(PROCEDURE),
              "model_hash": canonical_hash([c["model_hash"] for c in configs]), "resource_policy": ResourcePolicy().as_dict(),
              "source_files": files, "source_hash": source, "data_hash": canonical_hash(data)}
    config["config_hash"] = canonical_hash(config)
    return config


def validate_config(config, *, repo=None):
    _require(config.get("schema_version") == SCHEMA
             and config.get("config_hash") == canonical_hash({k:v for k,v in config.items() if k != "config_hash"})
             and (config.get("source_files"), config.get("source_hash")) == source_identity(repo), "CHOICE_CONFIG_HASH_OR_SOURCE_CHANGED")
    if config.get("kind") == "role":
        expected = prepare_config(config["data"]["native"]["config"]["path"], config["output_dir"], repo=repo, success_receipt=config["success_receipt"])
    else:
        _require(config.get("kind") == "aggregate", "CHOICE_CONFIG_KIND_REQUIRED")
        expected = prepare_aggregate_config([ref["path"] for ref in config["data"]["roles"]], config["output_dir"], repo=repo, success_receipt=config["success_receipt"])
    _require(_equal(config, expected), "CHOICE_FIXED_PROCEDURE_OR_NATIVE_PROVENANCE_CHANGED")
    return config


def job_from_config(config_path, *, repo=None, python=None):
    path = _absolute(config_path); config = _read(path); validate_config(config, repo=repo)
    role = config["kind"] == "role"
    dependencies = ["v6-bootstrap", dev.AGGREGATE_ID, _ref(config["data"]["native"]["config"])["job_id"]] if role else ["v6-bootstrap", dev.AGGREGATE_ID, *ROLE_IDS]
    job = {"job_id": config["job_id"], "run_id": config["run_id"], "cell_id": config["cell_id"], "phase": config["phase"],
           "kind": config["kind"], "seed": 20261023, "dependencies": dependencies, "dependency_policy": "success" if role else "terminal",
           "implementation_pending": False, "uses_gpu": role, "requires_gpu": role, "gpu_lock_mode": "wrapper",
           "argv": [str(python or sys.executable), "-m", "mediflow_medvlm.campaign_choice_likelihood", "--config", str(path), "--receipt", config["success_receipt"]],
           "cwd": str(Path(repo).resolve() if repo else Path(__file__).resolve().parents[2]), "success_receipt": config["success_receipt"],
           "source_files": config["source_files"], **{k:config[k] for k in HASH_FIELDS}}
    if role: job.update(**dev.cell_spec(config["cell_id"]), role=config["role"])
    return job


def conditional_log_likelihood(logits, label_ids, start):
    """All-token causal likelihood; nested lists provide a CPU reference path.

    ``start`` is the prompt length; logits[start-1] predicts label_ids[0].
    NaN/+inf, all -inf distributions and impossible target tokens fail closed.
    The tensor path normalizes in FP32, then sums the stored values in Python.
    """
    _require(type(start) is int and start > 0 and bool(label_ids)
             and all(type(token) is int and token >= 0 for token in label_ids), "CHOICE_INVALID_SCORE_BOUNDARY_OR_LABEL_IDS")
    if hasattr(logits, "shape"):
        import torch
        _require(len(logits.shape) == 2 and start+len(label_ids)-1 <= logits.shape[0], "CHOICE_LOGIT_SHAPE")
        scores = logits[start-1:start+len(label_ids)-1].float()
        _require(max(label_ids) < scores.shape[-1] and scores.shape[-1] > 0, "CHOICE_LABEL_ID_OUTSIDE_VOCAB")
        _require(not bool((torch.isnan(scores)|torch.isposinf(scores)).any().item())
                 and bool(torch.isfinite(scores).any(dim=-1).all().item()), "CHOICE_INVALID_LOGIT_DISTRIBUTION")
        indices = torch.tensor(label_ids, device=scores.device).unsqueeze(-1)
        values = scores.log_softmax(dim=-1).gather(-1, indices).squeeze(-1).tolist()
    else:
        _require(start+len(label_ids)-1 <= len(logits), "CHOICE_LOGIT_SHAPE")
        values = []
        for offset, target in enumerate(label_ids):
            scores = logits[start-1+offset]
            _require(bool(scores) and target < len(scores), "CHOICE_LABEL_ID_OUTSIDE_VOCAB")
            _require(all(isinstance(value, (int, float)) and not math.isnan(value) and value != math.inf for value in scores)
                     and any(math.isfinite(value) for value in scores), "CHOICE_INVALID_LOGIT_DISTRIBUTION")
            maximum = max(scores)
            values.append(scores[target]-maximum-math.log(sum(math.exp(value-maximum) for value in scores)))
    _require(all(math.isfinite(value) and value <= 0 for value in values), "CHOICE_NONFINITE_TARGET_LOG_PROBABILITY")
    return {"token_log_probabilities": values, "conditional_log_likelihood": math.fsum(values)}


def _tensor_ids(encoded):
    _require(set(encoded) <= TEXT_KEYS and {"input_ids", "attention_mask"} <= set(encoded), "CHOICE_UNSUPPORTED_PROCESSOR_TENSOR_KEYS")
    ids = encoded["input_ids"]
    _require(len(ids.shape) == 2 and ids.shape[0] == 1 and ids.shape[-1] > 0, "CHOICE_INPUT_SHAPE")
    for key, value in encoded.items():
        _require(value.shape == ids.shape, "CHOICE_PROCESSOR_TENSOR_SHAPE")
        _require(all(type(item) is int for item in value[0].tolist()), "CHOICE_PROCESSOR_INTEGER_TENSORS_REQUIRED")
    _require(all(value >= 0 for value in ids[0].tolist()) and all(value == 1 for value in encoded["attention_mask"][0].tolist()), "CHOICE_UNPADDED_INPUT_REQUIRED")
    if "mm_token_type_ids" in encoded:
        _require(all(value == 0 for value in encoded["mm_token_type_ids"][0].tolist()), "CHOICE_NON_TEXT_MODALITY_FORBIDDEN")
    return ids[0].tolist()


def encode_candidates(row, bundle, config):
    """Use every native processor tensor, with no gold or metadata in messages."""
    from .evaluation.v2 import encode_evaluation_sample, evaluation_text
    from .training.rendering import native_encode
    from .training.v6_native import verify_native_no_thinking
    labels = labels_for(row); _require(labels is not None, "CHOICE_UNSUPPORTED_ROW")
    restricted = {key:row[key] for key in ("sample_id", "task_type", "input")}
    _require(not restricted["input"].get("images"), "CHOICE_TEXT_IMAGES_FORBIDDEN")
    messages = [{"role": "user", "content": evaluation_text(restricted, "medical-vqa-short-answer.v2")}]
    thinking = verify_native_no_thinking(bundle.processor, bundle.architecture.model_type, messages=messages)
    prompt, metadata = encode_evaluation_sample(restricted, bundle.processor, bundle.architecture,
                                               max_length=config["eval_context"])
    repeated = native_encode(bundle.processor, messages, generation_prompt=True)
    prompt_ids = _tensor_ids(prompt)
    _require(set(prompt) == set(repeated) and all(prompt[key].tolist() == repeated[key].tolist() for key in prompt), "CHOICE_NATIVE_GENERATION_PROMPT_CHANGED")
    prefix = bundle.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    tokenizer = getattr(bundle.processor, "tokenizer", bundle.processor)
    candidates = []
    for label in labels:
        full = bundle.processor.apply_chat_template(messages+[{"role":"assistant", "content":label}],
                                                    tokenize=False, add_generation_prompt=False, enable_thinking=False)
        _require(isinstance(full, str) and full.startswith(prefix), "CHOICE_COMPLETED_ASSISTANT_PREFIX_MISMATCH")
        suffix = full[len(prefix):]
        leading = suffix[:len(suffix)-len(suffix.lstrip())]
        _require(suffix[len(leading):].startswith(label), "CHOICE_NATIVE_ASSISTANT_LABEL_RENDER_MISMATCH")
        encoded = dict(bundle.processor(text=[prefix+label], return_tensors="pt", padding=False, add_special_tokens=False))
        ids = _tensor_ids(encoded); boundary = len(prompt_ids)
        _require(len(ids) <= config["eval_context"], "CHOICE_CONTEXT_OVERFLOW")
        _require(len(ids) > boundary and set(encoded) == set(prompt)
                 and all(encoded[key][0][:boundary].tolist() == prompt[key][0].tolist() for key in prompt), "CHOICE_BPE_PROMPT_BOUNDARY_MISMATCH")
        continuation = ids[boundary:]
        _require(not set(continuation).intersection(getattr(tokenizer, "all_special_ids", []))
                 and tokenizer.decode(continuation, skip_special_tokens=False, clean_up_tokenization_spaces=False) == label, "CHOICE_LABEL_CONTINUATION_DECODE_MISMATCH")
        candidates.append((encoded, {"label":label, "token_ids":continuation,
            "token_count":len(continuation), "native_completed_leading_whitespace":leading,
            "native_completed_render_sha256":canonical_hash(full), "candidate_ids_sha256":canonical_hash(ids),
            "prefix_verified":True, "label_decode_verified":True, "eos_included":False,
            "processor_tensor_shapes":{key:list(value.shape) for key,value in encoded.items()},
            "processor_tensor_dtypes":{key:str(value.dtype) for key,value in encoded.items()}}))
    return candidates, {"input_tokens":len(prompt_ids), "prompt_ids_sha256":canonical_hash(prompt_ids),
                        "input_metadata":metadata, "native_prefix":thinking, "prompt_text_sha256":canonical_hash(prefix)}


def _empty(row, stratum, reason=None, *, attempted=False):
    labels = labels_for(row)
    return {"sample_id":row["sample_id"], "source_id":row["source_id"], "task_type":row["task_type"], "stratum":stratum,
            "status":"UNSUPPORTED" if labels is None else "TECHNICAL_FAILURE", "reason_code":"CHOICE_FREEFORM_UNSUPPORTED" if labels is None else reason,
            "attempted":attempted, "input_hash":canonical_hash(row["input"]), "labels":labels or [], "label_scores":[],
            "selected_label":None, "tied_labels":[], "exact_match":0.0}


def select_label(scores):
    _require(bool(scores) and all(math.isfinite(s["conditional_log_likelihood"]) for s in scores), "CHOICE_FINITE_LABEL_SCORES_REQUIRED")
    maximum = max(s["conditional_log_likelihood"] for s in scores)
    tied = [s["label"] for s in scores if s["conditional_log_likelihood"] == maximum]
    return tied[0], tied


def _score_one(bundle, row, stratum, config):
    import torch
    result = _empty(row, stratum, "UNINITIALIZED", attempted=True)
    try:
        candidates, metadata = encode_candidates(row, bundle, config)
        result.update(metadata)
    except Exception as exc:
        return {**result, "reason_code":str(exc) if isinstance(exc, ValueError) else "CHOICE_ENCODING_"+type(exc).__name__,
                "error_private":str(exc), "model_fatal":False}
    try:
        scores = []
        for encoded, evidence in candidates:
            moved = {key:value.to(bundle.model.device) for key,value in encoded.items()}
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = bundle.model(**moved, use_cache=False, return_dict=True)
            scores.append({**evidence, **conditional_log_likelihood(output.logits[0], evidence["token_ids"], metadata["input_tokens"])})
            del output, moved
        selected, ties = select_label(scores)
        # Gold is first accessed only after every candidate has been scored.
        gold = row["supervision"]["option_id"] if row["task_type"] == "medical_mcq" else row["supervision"]["answer"]
        _require(gold in result["labels"], "CHOICE_GOLD_OUTSIDE_DECLARED_LABELS")
        return {**result, "status":"SUCCEEDED", "reason_code":None, "label_scores":scores,
                "selected_label":selected, "tied_labels":ties, "exact_match":float(selected == gold), "model_fatal":False}
    except Exception as exc:
        return {**result, "reason_code":str(exc) if isinstance(exc, ValueError) else "CHOICE_FORWARD_"+type(exc).__name__,
                "error_private":str(exc), "model_fatal":True}


def _metrics(selected, rows):
    _require([r["sample_id"] for r in rows] == [r["sample_id"] for r,_ in selected], "CHOICE_ORDERED_RESULT_IDS_CHANGED")
    strata = {}
    for sample, stratum in selected:
        strata.setdefault(stratum, [])
    for (sample, stratum), row in zip(selected, rows):
        labels = labels_for(sample)
        expected = _empty(sample, stratum)
        _require(all(row.get(key) == expected[key] for key in ("sample_id","source_id","task_type","stratum","input_hash","labels")), "CHOICE_ROW_IDENTITY_CHANGED")
        _require(type(row.get("attempted")) is bool and ("model_fatal" not in row or type(row["model_fatal"]) is bool), "CHOICE_ROW_ATTEMPT_FLAG_CHANGED")
        if labels is None:
            _require(row == expected, "CHOICE_UNSUPPORTED_ROW_CHANGED")
        elif row.get("status") == "SUCCEEDED":
            scores = row["label_scores"]
            _require([s["label"] for s in scores] == labels and row.get("attempted") is True and row.get("reason_code") is None, "CHOICE_LABEL_ORDER_CHANGED")
            _require(type(row.get("input_tokens")) is int and row["input_tokens"] > 0
                     # evaluation.v2 emits runtime's UTF-8 JSON input hash; the
                     # campaign hash escapes Unicode and binds other artifacts.
                     and row.get("input_metadata",{}).get("input_hash") == encoder_input_hash(sample["input"])
                     and row.get("native_prefix",{}).get("assistant_prefix_verified") is True, "CHOICE_NATIVE_INPUT_EVIDENCE_CHANGED")
            for score in scores:
                ids, values = score["token_ids"], score["token_log_probabilities"]
                _require(bool(ids) and all(type(token) is int and token >= 0 for token in ids)
                         and len(ids) == len(values) == score["token_count"] and row["input_tokens"]+len(ids) <= 4096
                         and all(isinstance(v,(int,float)) and math.isfinite(v) and v <= 0 for v in values)
                         and score["conditional_log_likelihood"] == math.fsum(values)
                         and score.get("prefix_verified") is True and score.get("label_decode_verified") is True and score.get("eos_included") is False,
                         "CHOICE_TOKEN_SUM_OR_BOUNDARY_EVIDENCE_CHANGED")
            choice, ties = select_label(scores)
            gold = sample["supervision"]["option_id"] if sample["task_type"] == "medical_mcq" else sample["supervision"]["answer"]
            _require(row["selected_label"] == choice and row["tied_labels"] == ties and row["exact_match"] == float(choice == gold), "CHOICE_SELECTION_OR_GOLD_SCORE_CHANGED")
        else:
            _require(row.get("status") == "TECHNICAL_FAILURE" and isinstance(row.get("reason_code"),str) and bool(row["reason_code"])
                     and row.get("label_scores") == [] and row.get("selected_label") is None and row.get("tied_labels") == []
                     and row.get("exact_match") == 0., "CHOICE_FAILURE_ROW_CHANGED")
        strata[stratum].append(row)
    results = {}
    for name, members in strata.items():
        supported = [r for r in members if r["status"] != "UNSUPPORTED"]
        successes = [r for r in supported if r["status"] == "SUCCEEDED"]
        n = len(supported); failures = n-len(successes)
        results[name] = {"text_all_planned":len(members), "classification_scheduled":n,
                         "unsupported":len(members)-n, "succeeded":len(successes), "technical_failures":failures,
                         "correct":sum(r["exact_match"] for r in successes),
                         "all_planned_accuracy":sum(r["exact_match"] for r in successes)/n if n else None,
                         "technical_failure_rate":failures/n if n else None,
                         "tie_count":sum(len(r["tied_labels"])>1 for r in successes),
                         "technical_failure_reasons":dict(sorted(Counter(r["reason_code"] for r in supported if r["status"] == "TECHNICAL_FAILURE").items()))}
    return {"coverage":_coverage(selected), "strata":results, "classification_scheduled":sum(v["classification_scheduled"] for v in results.values()),
            "succeeded":sum(v["succeeded"] for v in results.values()), "technical_failure_count":sum(v["technical_failures"] for v in results.values()),
            "unsupported_count":sum(v["unsupported"] for v in results.values()), **FLAGS}


def _execute(config, original, selected, stream):
    from .evaluation.v6_native import load_evaluation_bundle
    import torch
    ref = original["models"][config["cell_id"]]; train = _ref(ref["config"]); metrics = _ref(ref["metrics"])
    fatal = None; fatal_error = None; rows = []; bundle = None
    try:
        bundle = load_evaluation_bundle(config["model"], config["cell_id"], role=config["role"], expected_contract=metrics["native_contract"],
            final_directory=train["output_dir"], expected_adapter_sha256=ref["adapter_sha256"], fixed_template_date=config["fixed_template_date"])
        torch.cuda.reset_peak_memory_stats()
    except Exception as exc: fatal, fatal_error = "CHOICE_MODEL_LOAD_"+type(exc).__name__, str(exc)
    for row, stratum in selected:
        if labels_for(row) is None: result = _empty(row, stratum)
        elif fatal: result = _empty(row, stratum, fatal)
        else: result = _score_one(bundle, row, stratum, config)
        if result.get("model_fatal"): fatal, fatal_error = result["reason_code"], result.get("error_private")
        rows.append(result); stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True)+"\n"); stream.flush()
        progress = os.environ.get("MEDVLM_SUPERVISOR_PROGRESS_PATH")
        if progress: atomic_json(progress, {**{k:config[k] for k in IDENTITY_KEYS}, "status":"RUNNING", "stage":"choice_likelihood",
            "rows_completed":len(rows), "scheduled_text_rows":len(selected), "technical_failures":sum(r["status"] == "TECHNICAL_FAILURE" for r in rows), "updated_at":utcnow()})
    stream.flush(); os.fsync(stream.fileno())
    return rows, {"native_contract":bundle.contract if bundle else None, "model_load_succeeded":bundle is not None,
                  "fatal_reason":fatal, "fatal_error_private":fatal_error, "peak_allocated_bytes":torch.cuda.max_memory_allocated() if bundle else None,
                  "peak_reserved_bytes":torch.cuda.max_memory_reserved() if bundle else None}


def _native_proof(config, native):
    contract = native.get("native_contract")
    _require(type(native.get("model_load_succeeded")) is bool, "CHOICE_NATIVE_MODEL_LOAD_FLAG_CHANGED")
    if not native.get("model_load_succeeded"):
        _require(contract is None and isinstance(native.get("fatal_reason"),str) and native["fatal_reason"].startswith("CHOICE_MODEL_LOAD_"), "CHOICE_LOAD_FAILURE_EVIDENCE_CHANGED")
        return
    from .evaluation.v6_native import normalized_base_dtype_signature
    expected = _ref(config["data"]["native"]["trained_cell"]["metrics"])["native_contract"]
    keys = ("architecture", "template", "thinking", "loader", "revision", "snapshot_path", "processor_class", "quantization", "native_runtime_compatibility")
    _require(all(contract.get(key) == expected.get(key) for key in keys)
             and normalized_base_dtype_signature(contract["parameter_storage"]) == normalized_base_dtype_signature(expected["parameter_storage"])
             and contract["parameter_storage"]["trainable_parameters"] == 0 and contract.get("evaluation_role") == config["role"]
             and contract.get("weights_loaded") is True and contract.get("cuda_residency_verified") is True,
             "CHOICE_NATIVE_CONTRACT_CHANGED")
    adapter = config["data"]["native"]["trained_cell"]["adapter_sha256"] if config["role"] == "tuned" else None
    _require(contract.get("evaluation_adapter_sha256") == adapter
             and (config["role"] != "base" or contract["parameter_storage"]["adapter_parameters"] == 0), "CHOICE_ADAPTER_IDENTITY_CHANGED")


def _role_completed(config):
    _, proof, original, selected, _ = _native_inputs(config["data"]["native"]["config"]["path"])
    _require(_equal(proof, config["data"]["native"]), "CHOICE_NATIVE_PROOF_CHANGED")
    output = Path(config["output_dir"])
    rows = [json.loads(line) for line in (output/"scores.private.jsonl").read_text().splitlines()]
    expected = _metrics(selected, rows); metrics = _read(output/"metrics.json")
    _require(all(_equal(metrics.get(key), value) for key,value in expected.items()), "CHOICE_METRICS_CHANGED")
    _native_proof(config, metrics["native"])
    _require(metrics["native"]["model_load_succeeded"] is True and metrics["native"].get("fatal_reason") is None,
             "CHOICE_FATAL_ATTEMPT_CANNOT_SUCCEED")
    _require(metrics.get("procedure") == PROCEDURE and metrics.get("generation_technical_complete") is proof["generation_technical_complete"], "CHOICE_PROCEDURE_OR_GENERATION_EVIDENCE_CHANGED")
    accounting = metrics.get("gpu_accounting", {})
    _require(accounting.get("ledger_path") == config["gpu_ledger_path"] and isinstance(accounting.get("session_seconds"), (int,float))
             and math.isfinite(accounting["session_seconds"]) and accounting["session_seconds"] >= 0
             and accounting.get("snapshot",{}).get("resource_policy") == config["resource_policy"], "CHOICE_GPU_ACCOUNTING_CHANGED")
    return expected


def _aggregate_summary(config):
    manifest, state = dev.caps._active(config["state_path"]); jobs = {job["job_id"]:job for job in manifest["jobs"]}
    conditions = {}; rows_by_role = {}
    original = _ref(config["data"]["initial_protocol"]); selected = dev._rows(original, "text")
    for reference, key in zip(config["data"]["roles"], ROLE_IDS):
        child = _ref(reference); job, record = jobs[key], state["jobs"][key]
        dev.caps._terminal(record)
        actual, actual_ref = dev.caps._child_config(job, "campaign_choice_likelihood")
        _require(actual_ref == reference and _equal(actual, child), "CHOICE_REGISTERED_ROLE_CHANGED")
        if record["status"] == "SUCCEEDED":
            receipt = dev.caps.initial.gate._success(job, record)
            _require(_equal(verify_result(child, repo=job["cwd"]), receipt), "CHOICE_ROLE_RESULT_CHANGED")
            rows = [json.loads(line) for line in (Path(child["output_dir"])/"scores.private.jsonl").read_text().splitlines()]
            metrics = _metrics(selected, rows)
            conditions[key] = {"status":"SUCCEEDED", "receipt":_reference(job["success_receipt"]), "strata":metrics["strata"],
                               "technical_complete":metrics["technical_failure_count"] == 0}
        else:
            rows = [_empty(row, stratum, "CHOICE_ROLE_JOB_UNAVAILABLE") for row,stratum in selected]
            conditions[key] = {"status":record["status"], "receipt":None, "strata":_metrics(selected, rows)["strata"],
                               "technical_complete":False, "error":record.get("error"), "error_signature":record.get("error_signature")}
        rows_by_role[key] = rows
    transitions = {}
    for cell in TEXT_CELLS:
        counts = {}
        left, right = (rows_by_role[f"v6-choice-likelihood-{cell}-{role}"] for role in ("base", "tuned"))
        for a,b in zip(left,right):
            if not a["labels"]: continue
            c = counts.setdefault(a["stratum"], {"classification_scheduled":0,"correct_transitions":{},"scorable_transitions":{}})
            c["classification_scheduled"] += 1
            for name,x,y in (("correct_transitions", a["exact_match"] == 1., b["exact_match"] == 1.),
                             ("scorable_transitions",a["status"] == "SUCCEEDED",b["status"] == "SUCCEEDED")):
                label = f"{int(x)}->{int(y)}"; c[name][label] = c[name].get(label,0)+1
        transitions[cell] = counts
    return {"scope":SCOPES["aggregate"], "likelihood_auxiliary_terminal":True,
            "all_roles_succeeded":all(c["status"] == "SUCCEEDED" for c in conditions.values()),
            "all_roles_technical_complete":all(c["technical_complete"] for c in conditions.values()),
            "conditions":conditions, "paired_transitions":transitions, "coverage":config["data"]["coverage"],
            "procedure":copy.deepcopy(PROCEDURE), **FLAGS}


def verify_result(config, *, repo=None):
    validate_config(config, repo=repo)
    output = Path(config["output_dir"]); marker = _read(output/"completion.json")
    role = config["kind"] == "role"; names = ("config.json", "metrics.json", "scores.private.jsonl") if role else ("config.json", "summary.json")
    _require(not output.is_symlink() and not any(p.is_symlink() for p in output.iterdir())
             and {p.name for p in output.iterdir()} == set(names)|{"completion.json"}
             and _equal(_read(output/"config.json"), config)
             and marker.get("schema_version") == "medvlm.v6.choice-likelihood-receipt.v1"
             and marker.get("scope") == config["scope"] and marker.get("status") == "SUCCEEDED"
             and all(marker.get(key) == config[key] for key in IDENTITY_KEYS)
             and all(marker.get(key) is value for key,value in FLAGS.items())
             and marker.get("artifacts") == [_reference(output/name) for name in names], "CHOICE_RESULT_ARTIFACTS_CHANGED")
    if role:
        metrics = _role_completed(config)
        _require(marker.get("likelihood_auxiliary_completed") is True and marker.get("classification_scheduled") == metrics["classification_scheduled"]
                 and marker.get("technical_complete") is (metrics["technical_failure_count"] == 0), "CHOICE_ROLE_COMPLETION_CHANGED")
    else:
        expected = _aggregate_summary(config)
        _require(_equal({k:v for k,v in _read(output/"summary.json").items() if k != "finished_at"}, expected)
                 and marker.get("likelihood_auxiliary_terminal") is True and marker.get("all_roles_succeeded") is expected["all_roles_succeeded"]
                 and marker.get("all_roles_technical_complete") is expected["all_roles_technical_complete"], "CHOICE_AGGREGATE_RESULT_CHANGED")
    return {**marker, "artifacts":marker["artifacts"]+[_reference(output/"completion.json")]}


def run_job(config_path, receipt_path):
    path, receipt_path = _absolute(config_path), _absolute(receipt_path); config = _read(path); before = file_hash(path)
    _require("torch" not in sys.modules, "CHOICE_FRESH_PROCESS_REQUIRED")
    if config.get("kind") == "aggregate": _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CHOICE_FRESH_CPU_PROCESS_REQUIRED")
    validate_config(config); dev.caps._env(config)
    _require(str(receipt_path) == config["success_receipt"], "CHOICE_RECEIPT_ROUTE_CHANGED")
    role = config["kind"] == "role"
    if role: dev.caps.initial.first.main_train.smoke._inherited_supervisor_fds(config)
    output = Path(config["output_dir"]); output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with output.with_name(output.name+".worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not output.exists():
            _require(not receipt_path.exists(), "CHOICE_RECEIPT_WITHOUT_OUTPUT")
            staging = Path(tempfile.mkdtemp(prefix=output.name+".attempt-", dir=output.parent)); mask = os.umask(0o077)
            try:
                atomic_json(staging/"config.json", config)
                if role:
                    _, _, original, selected, _ = _native_inputs(config["data"]["native"]["config"]["path"])
                    ledger = BudgetLedger(config["gpu_ledger_path"], policy=ResourcePolicy())
                    ledger.start(config["model_id"], config["run_id"], "native_choice_likelihood")
                    try:
                        with (staging/"scores.private.jsonl").open("x") as stream: rows, native = _execute(config, original, selected, stream)
                    finally: ledger.finish()
                    metrics = _metrics(selected, rows); _native_proof(config, native)
                    metrics.update(native=native, procedure=copy.deepcopy(PROCEDURE),
                        generation_technical_complete=config["data"]["native"]["generation_technical_complete"],
                        gpu_accounting={"session_seconds":ledger.session_seconds, "ledger_path":config["gpu_ledger_path"], "snapshot":ledger.snapshot()})
                    atomic_json(staging/"metrics.json", metrics)
                    # Retain every planned row and final accounting in this
                    # private attempt before reporting a fatal job failure.
                    # This lets the Supervisor's existing recovery/Astra gates
                    # see model load/OOM/forward errors rather than a false
                    # successful auxiliary. Generation receipts are untouched.
                    if native.get("fatal_reason"):
                        raise dev.caps.initial.CalibrationFatal(native["fatal_reason"])
                    names = ("config.json", "metrics.json", "scores.private.jsonl")
                    extra = {"likelihood_auxiliary_completed":True, "classification_scheduled":metrics["classification_scheduled"], "technical_complete":metrics["technical_failure_count"] == 0}
                else:
                    summary = _aggregate_summary(config); atomic_json(staging/"summary.json", {**summary, "finished_at":utcnow()})
                    names = ("config.json", "summary.json")
                    extra = {key:summary[key] for key in ("likelihood_auxiliary_terminal", "all_roles_succeeded", "all_roles_technical_complete")}
                _require(file_hash(path) == before, "CHOICE_CONFIG_CHANGED_DURING_EXECUTION"); validate_config(config)
                marker = {"schema_version":"medvlm.v6.choice-likelihood-receipt.v1", "status":"SUCCEEDED", "scope":config["scope"],
                    **{key:config[key] for key in IDENTITY_KEYS}, **extra, **FLAGS,
                    "artifacts":[{"path":str(output/name), "sha256":file_hash(staging/name)} for name in names]}
                atomic_json(staging/"completion.json", marker); os.rename(staging, output)
                fd = os.open(output.parent, os.O_DIRECTORY)
                try: os.fsync(fd)
                finally: os.close(fd)
            except BaseException as exc:
                if staging.exists(): atomic_json(staging/"failure.private.json", {"status":"FAILED", "exception_type":type(exc).__name__, "exception_message":str(exc), "traceback":traceback.format_exc(), "finished_at":utcnow()})
                raise
            finally: os.umask(mask)
        result = verify_result(config)
        _require(file_hash(path) == before and (not receipt_path.exists() or _equal(_read(receipt_path), result)), "CHOICE_COMPLETED_REUSE_CHANGED")
        if not receipt_path.exists(): atomic_json(receipt_path, result)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if _read(args.config).get("kind") == "aggregate": os.environ["CUDA_VISIBLE_DEVICES"] = ""
        result = run_job(args.config, args.receipt)
    except Exception as exc:
        print(json.dumps({"status":"FAILED", "exception_type":type(exc).__name__,
            **({"reason_code":str(exc)} if isinstance(exc, (dev.caps.initial.gate.GateError, dev.caps.initial.CalibrationFatal)) else {})})); return 1
    print(json.dumps({"status":result["status"], "scope":result["scope"]})); return 0


if __name__ == "__main__": raise SystemExit(main())
