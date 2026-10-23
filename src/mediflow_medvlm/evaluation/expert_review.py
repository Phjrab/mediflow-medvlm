"""Prepare a bounded, blinded local review packet; never invent human ratings."""
from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path

from ..runtime import canonical_hash, file_hash
from .runner import load_protocol, validated_predictions

REVIEW_VERSION = "medvlm.expert-review-packet.v1"
RATING_FIELDS = [f"{dimension}_{side}" for dimension in
                 ("factuality", "reference_grounding", "safety", "completeness", "uncertainty", "critical_safety_issue")
                 for side in ("A", "B")]
REVIEW_FIELDS = RATING_FIELDS + ["preference", "reviewer_id", "reviewed_at", "evidence_notes"]


def _local_destination(path):
    destination = Path(path)
    if destination.is_symlink():
        raise ValueError("review destination cannot be a symlink")
    destination = destination.resolve()
    # The desktop sandbox has empty .git sentinel directories at writable roots.
    # A real repository has metadata; a worktree has a .git pointer file.
    markers = [parent / ".git" for parent in (destination, *destination.parents)]
    if any(marker.is_file() or (marker.is_dir() and any(marker.iterdir())) for marker in markers):
        raise ValueError("raw review text must stay outside Git worktrees")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("refusing to overwrite an existing review packet")
    return destination


def _private_write(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _csv_text(value):
    text = str(value)
    # Retain unaltered text in original local predictions; avoid spreadsheet formulas.
    return "'" + text if text.lstrip(" \t\r\n").startswith(("=", "+", "-", "@")) else text


def _review_runs(paths, protocol):
    runs, predictions = [], []
    for path in paths:
        manifest = json.loads((Path(path).parent / "run_manifest.json").read_text())
        if manifest.get("protocol_hash") != protocol["protocol_hash"]:
            raise ValueError("review prediction run has another protocol")
        runs.append(manifest)
        predictions.append({p["sample_id"]: p for p in validated_predictions(path, protocol, manifest)})
    if runs[0]["run_id"] == runs[1]["run_id"] or runs[0]["identity_hash"] == runs[1]["identity_hash"]:
        raise ValueError("expert review needs two distinct model runs")
    for key in ("model_id", "model_revision", "config_hash", "precision", "generation", "pair_id"):
        if key not in runs[0] or key not in runs[1] or runs[0][key] != runs[1][key]:
            raise ValueError("unmatched expert review pair: " + key)
    return runs, predictions


def prepare_expert_review(protocol_path, baseline_predictions_path, tuned_predictions_path,
                          output_dir, cap=16, seed=20261023):
    """Create CSV, private A/B key and manifest outside Git, without scoring.

    Selection is fixed by eligible scheduled IDs before prediction inspection.
    Failed or missing answers remain selected and are shown with explicit status.
    ID pseudonyms and A/B ordering are not patient-data de-identification.
    """
    if type(cap) is not int or not 1 <= cap <= 16:
        raise ValueError("optional expert review cap must be between 1 and 16")
    destination = _local_destination(output_dir)
    protocol, samples = load_protocol(protocol_path)
    if protocol["split"] not in {"validation", "test"}:
        raise ValueError("expert review requires a frozen validation or test protocol")
    eligible = [r for r in samples if r["source_id"] == "medquad" and r["task_type"] == "medical_qa"
                and r["split"] in {"validation", "test"}]
    selected = sorted(eligible, key=lambda r: canonical_hash([seed, r["sample_id"]]))[:cap]
    if not selected:
        raise ValueError("no eligible MedQuAD validation/test rows")
    paths = [Path(baseline_predictions_path), Path(tuned_predictions_path)]
    runs, predictions = _review_runs(paths, protocol)
    table, key = [], []
    for sample in selected:
        sid = sample["sample_id"]
        review_id = "review-" + canonical_hash([seed, protocol["protocol_hash"], sid, "review_id"])[:16]
        order = [0, 1] if int(canonical_hash([seed, sid, "blind_order"])[0], 16) % 2 == 0 else [1, 0]
        row = {"review_id": review_id, "question": sample["input"]["question"],
               "reference_answer": sample["supervision"]["answer"]}
        for side, index in zip(("A", "B"), order):
            pred = predictions[index].get(sid, {"status": "not_attempted"})
            row["answer_" + side] = pred.get("raw_output") or pred.get("parsed_answer") or "[NO MODEL ANSWER]"
            row["status_" + side] = pred["status"]
        row.update({field: "" for field in REVIEW_FIELDS})
        table.append(row)
        key.append({"review_id": review_id, "sample_id": sid, "input_hash": protocol["input_digests"][sid],
                    "A_run_id": runs[order[0]]["run_id"], "B_run_id": runs[order[1]]["run_id"],
                    "A_role": ("baseline", "tuned")[order[0]], "B_role": ("baseline", "tuned")[order[1]]})
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=["review_id", "question", "reference_answer", "answer_A", "answer_B",
                                               "status_A", "status_B", *REVIEW_FIELDS])
    writer.writeheader()
    writer.writerows({field: _csv_text(value) for field, value in row.items()} for row in table)
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(destination, 0o700)
    _private_write(destination / "review.csv", buffer.getvalue())
    private_key = {"review_version": REVIEW_VERSION, "protocol_hash": protocol["protocol_hash"],
                   "baseline_run_id": runs[0]["run_id"], "tuned_run_id": runs[1]["run_id"], "items": key}
    _private_write(destination / "private_key.json", json.dumps(private_key, ensure_ascii=False, indent=2) + "\n")
    manifest = {"review_version": REVIEW_VERSION, "status": "PREPARED_NOT_RUN", "diagnostic_only": True,
                "human_review": "not_performed", "human_review_count": 0, "clinical_accuracy": None,
                "clinical_scores": None, "training_data_changed": False, "protocol_hash": protocol["protocol_hash"],
                "split": protocol["split"], "seed": seed, "cap": cap, "selected_count": len(table),
                "eligible_count": len(eligible), "selection_policy": "eligible_scheduled_ids_sha256(seed,sample_id)_before_predictions",
                "blinding": "pseudonymous_IDs_and_deterministic_AB_run_order; not_patient_deidentification",
                "missing_or_failed_answers": "retained_with_explicit_status",
                "source_prediction_files_sha256": [file_hash(path) if path.exists() else None for path in paths],
                "files_sha256": {name: file_hash(destination / name) for name in ("review.csv", "private_key.json")},
                "rubric": "docs/research/EXPERT_REVIEW_RUBRIC_KO.md",
                "review_scores_generated": False, "test_review_must_not_select_training": True}
    _private_write(destination / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest
