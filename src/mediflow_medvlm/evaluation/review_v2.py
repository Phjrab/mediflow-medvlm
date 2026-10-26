"""Prepare a fixed, blinded AI-Hub short/long-answer packet without human scores."""
from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path

from ..runtime import canonical_hash, directory_digest, file_hash
from .expert_review import REVIEW_FIELDS, _csv_text, _local_destination, _private_write, _review_runs
from .runner import load_protocol, verify_completed_evaluation
from .v2 import V2_PROTOCOL, V2_RENDERER

REVIEW_VERSION = "medvlm.aihub-expert-review-packet.v2"
SOURCE = "aihub_71875"
QUOTAS = {"2": 8, "3": 8}


def _destination(protocol_path, output_dir):
    protocol_path = Path(protocol_path).resolve()
    v2_root = next((parent for parent in protocol_path.parents if parent.name == "v2"), None)
    if v2_root is None:
        raise ValueError("AI-Hub review requires a protocol under data_root/v2")
    destination = _local_destination(output_dir)
    allowed = v2_root / "review"
    if destination == allowed or not destination.is_relative_to(allowed):
        raise ValueError("raw review must stay under the same data_root/v2/review/packet")
    return destination


def _verified_runs(paths, protocol):
    runs, predictions = _review_runs(paths, protocol)
    for field in ("native_files", "chat_template_sha256", "dtype"):
        if (field not in runs[0].get("model", {}) or field not in runs[1].get("model", {})
                or runs[0]["model"][field] != runs[1]["model"][field]):
            raise ValueError("unmatched review native model: " + field)
    native = runs[0]["model"]["native_files"]
    if not native.get("files") or native.get("sha256") != canonical_hash(native["files"]):
        raise ValueError("invalid review native-file receipt")
    if runs[0].get("adapter_path") is not None or runs[0].get("adapter_digest") is not None:
        raise ValueError("baseline review run must use the original base")
    adapter = runs[1].get("adapter_path")
    if not adapter or not runs[1].get("adapter_digest"):
        raise ValueError("tuned review run must identify its adapter")
    if directory_digest(adapter)["sha256"] != runs[1]["adapter_digest"]:
        raise ValueError("review adapter files changed")
    receipts = []
    for path, run in zip(paths, runs):
        if run.get("renderer_version") != V2_RENDERER or run.get("image_condition") != "image":
            raise ValueError("review run must use the declared native v2 text renderer")
        if run.get("status") not in {"SUCCEEDED", "FAILED_REPRODUCIBLE", "BLOCKED_HARDWARE", "SKIPPED_BUDGET", "INTERRUPTED"}:
            raise ValueError("review needs a terminal evaluation run")
        if run.get("scheduled") != len(protocol["sample_ids"]):
            raise ValueError("review schedule differs from its protocol")
        expected = run.get("artifact_hashes", {})
        present = {name for name in ("predictions.jsonl", "metrics.json") if (path.parent / name).is_file()}
        if "metrics.json" not in present or set(expected) != present:
            raise ValueError("review evaluation lacks artifact integrity receipts")
        for name, digest in expected.items():
            if file_hash(path.parent / name) != digest:
                raise ValueError("review evaluation artifact changed: " + name)
        if path.name != "predictions.jsonl":
            raise ValueError("review must use the original predictions.jsonl artifact")
        if run["status"] == "SUCCEEDED":
            verify_completed_evaluation(run, protocol, path.parent, native["sha256"])
        receipts.append({"run_id": run["run_id"], "identity_hash": run["identity_hash"],
                         "manifest_path": str(path.parent / "run_manifest.json"),
                         "manifest_sha256": file_hash(path.parent / "run_manifest.json"),
                         "predictions_path": str(path), "artifact_hashes": expected,
                         "status": run["status"]})
    return runs, predictions, receipts


def prepare_aihub_expert_review(protocol, base_predictions, tuned_predictions, output_dir,
                                cap=16, seed=20261023):
    """Write 8 short and 8 long answers per pair under data_root/v2/review.

    Eligibility and order use only the frozen source/task/q_type/IDs, before
    inspecting predictions. Failures and missing answers remain selected. This
    prepares human review; it does not rate answers or perform de-identification.
    """
    if type(cap) is not int or cap != 16:
        raise ValueError("AI-Hub review cap is fixed at 16 (8 short + 8 long)")
    if type(seed) is not int:
        raise ValueError("review seed must be an integer")
    protocol_path = Path(protocol).resolve()
    destination = _destination(protocol_path, output_dir)
    frozen, samples = load_protocol(protocol_path)
    if frozen.get("protocol_version") != V2_PROTOCOL or frozen.get("split") not in {"validation", "test"}:
        raise ValueError("AI-Hub review requires a frozen v2 validation or test protocol")
    groups = {qtype: [sample for sample in samples if sample["source_id"] == SOURCE
                     and sample["task_type"] == "medical_qa" and sample["split"] == frozen["split"]
                     and str(sample.get("metadata", {}).get("q_type")) == qtype] for qtype in QUOTAS}
    if any(len(groups[qtype]) < count for qtype, count in QUOTAS.items()):
        raise ValueError("AI-Hub review requires at least 8 eligible rows in each q_type 2 and 3")
    selected = [sample for qtype, count in QUOTAS.items()
                for sample in sorted(groups[qtype], key=lambda row: canonical_hash([seed, row["sample_id"]]))[:count]]
    paths = [Path(base_predictions).resolve(), Path(tuned_predictions).resolve()]
    runs, predictions, receipts = _verified_runs(paths, frozen)
    protocol_file_sha = file_hash(protocol_path)
    table, keys = [], []
    for sample in selected:
        sid = sample["sample_id"]
        review_id = "review-" + canonical_hash([seed, frozen["protocol_hash"], sid, "review_id"])[0:16]
        order = [0, 1] if int(canonical_hash([seed, sid, "blind_order"])[0], 16) % 2 == 0 else [1, 0]
        row = {"review_id": review_id, "q_type": str(sample["metadata"]["q_type"]),
               "question": sample["input"]["question"], "reference_answer": sample["supervision"]["answer"]}
        for side, index in zip(("A", "B"), order):
            prediction = predictions[index].get(sid, {"status": "not_attempted"})
            row["answer_" + side] = prediction.get("raw_output") or prediction.get("parsed_answer") or "[NO MODEL ANSWER]"
            row["status_" + side] = prediction["status"]
        row.update({field: "" for field in REVIEW_FIELDS})
        table.append(row)
        keys.append({"review_id": review_id, "sample_id": sid, "q_type": row["q_type"],
                     "input_hash": frozen["input_digests"][sid],
                     "reference_sha256": canonical_hash(sample["supervision"]),
                     "A_run_id": runs[order[0]]["run_id"], "B_run_id": runs[order[1]]["run_id"],
                     "A_role": ("baseline", "tuned")[order[0]], "B_role": ("baseline", "tuned")[order[1]]})
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=["review_id", "q_type", "question", "reference_answer",
                                               "answer_A", "answer_B", "status_A", "status_B", *REVIEW_FIELDS])
    writer.writeheader()
    writer.writerows({field: _csv_text(value) for field, value in row.items()} for row in table)
    # Refuse a changing evaluation directory rather than publish a mixed receipt.
    if file_hash(protocol_path) != protocol_file_sha:
        raise ValueError("review protocol changed during preparation")
    for receipt, path in zip(receipts, paths):
        if file_hash(receipt["manifest_path"]) != receipt["manifest_sha256"]:
            raise ValueError("review manifest changed during preparation")
        for name, digest in receipt["artifact_hashes"].items():
            if file_hash(path.parent / name) != digest:
                raise ValueError("review artifact changed during preparation")
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(destination, 0o700)
    _private_write(destination / "review.csv", buffer.getvalue())
    private_key = {"review_version": REVIEW_VERSION, "protocol_hash": frozen["protocol_hash"],
                   "baseline_run_id": runs[0]["run_id"], "tuned_run_id": runs[1]["run_id"], "items": keys}
    _private_write(destination / "private_key.json", json.dumps(private_key, ensure_ascii=False, indent=2) + "\n")
    instructions = ("This packet is prepared for human review; all ratings are blank.\n"
                    "q_type 2 = short answer; q_type 3 = long answer. Eight fixed held-out rows per type.\n"
                    "AI-Hub 71875 includes LLM-generated source material; per-record expert authorship/review is not established.\n"
                    "Treat the reference as source evidence, not an independently verified clinical gold standard.\n"
                    "Use the rating dimensions and 0-3/NA criteria in docs/research/EXPERT_REVIEW_RUBRIC_KO.md; "
                    "its MedQuAD eligibility section does not apply to this AI-Hub packet.\n"
                    "Do not replace failed/missing slots, fill automatic ratings, infer clinical accuracy, or select models using this test review.\n"
                    "Keep private_key.json separate from reviewers. A/B labels hide run names, not identifying content in answers.\n")
    _private_write(destination / "instructions.txt", instructions)
    manifest = {"review_version": REVIEW_VERSION, "status": "PREPARED_NOT_RUN", "diagnostic_only": True,
                "human_review": "not_performed", "human_review_count": 0, "clinical_accuracy": None,
                "clinical_scores": None, "review_scores_generated": False, "training_data_changed": False,
                "source_id": SOURCE, "split": frozen["split"], "seed": seed, "cap": cap,
                "selected_count": len(table), "selected_by_q_type": QUOTAS,
                "eligible_by_q_type": {key: len(rows) for key, rows in groups.items()},
                "selection_policy": "8_per_q_type_2_3_sha256(seed,sample_id)_before_predictions",
                "blinding": "pseudonymous_IDs_and_deterministic_AB_run_order; not_patient_deidentification",
                "missing_or_failed_answers": "retained_with_explicit_status",
                "protocol_path": str(protocol_path), "protocol_hash": frozen["protocol_hash"],
                "protocol_file_sha256": protocol_file_sha, "source_runs": receipts,
                "matched_native_files_sha256": runs[0]["model"]["native_files"]["sha256"],
                "matched_chat_template_sha256": runs[0]["model"]["chat_template_sha256"],
                "matched_dtype": runs[0]["model"]["dtype"],
                "files_sha256": {name: file_hash(destination / name) for name in ("review.csv", "private_key.json", "instructions.txt")},
                "rubric": "docs/research/EXPERT_REVIEW_RUBRIC_KO.md#차원별-0–3점-척도",
                "source_includes_llm_generation": True, "per_record_human_review": "unknown",
                "test_review_must_not_select_training": True}
    _private_write(destination / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest
