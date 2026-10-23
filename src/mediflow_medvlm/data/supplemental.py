"""Strict readers for sources that require resolved rights/access before use.

These functions do not infer a data license from a repository code license, do not
pair questions by row position, and never score MedMCQA's unlabeled official test.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from .schema import normalized, sample


def _prerequisites(revision, license_ref):
    if not revision or not license_ref:
        raise ValueError("resolved source revision and applicable data-license evidence required")


def medqa_records(rows, *, revision, original_split, license_ref, fetched_at, variant="US/4_options"):
    """English 4-option only; normalized question identity joins option variants."""
    _prerequisites(revision, license_ref)
    if variant != "US/4_options":
        raise ValueError("v1 reader requires explicit US/4_options variant")
    records, excluded = [], Counter()
    for index, raw in enumerate(rows):
        options = raw.get("options", {})
        if set(options) != set("ABCD"):
            excluded["not_four_options"] += 1
            continue
        label = raw.get("answer_idx")
        if label not in options or normalized(options[label]) != normalized(raw.get("answer", "")):
            excluded["invalid_or_conflicting_label"] += 1
            continue
        question = raw.get("question", "")
        if not question.strip():
            excluded["empty_question"] += 1
            continue
        # Source lacks a stable shared cross-language ID: do not claim translations verified.
        group = "medqa:question:" + hashlib.sha256(normalized(question).encode()).hexdigest()
        records.append(sample("medqa", revision, f"{variant}/{original_split}/{index}", "medical_mcq", question,
                              options[label], {"dev": "validation"}.get(original_split, original_split),
                              original_split=original_split, options=options, option_id=label, document=group,
                              source_url="https://github.com/jind11/MedQA", license_ref=license_ref, fetched_at=fetched_at,
                              metadata={"variant": variant, "original_line": index, "cross_language_pairing": "unknown"}))
    return records, dict(excluded)


def medmcqa_records(rows, *, revision, original_split, license_ref, fetched_at, cop_base):
    """cop_base must be verified for the exact artifact (raw=1, HF features=0)."""
    _prerequisites(revision, license_ref)
    if cop_base not in (0, 1):
        raise ValueError("explicit verified cop_base=0 or 1 is required")
    if original_split == "test":
        return [], {"official_test_labels_unavailable": len(rows)}
    records, excluded = [], Counter()
    for raw in rows:
        if raw.get("choice_type") != "single":
            excluded["not_single_choice"] += 1
            continue
        options = {letter: raw.get(field) for letter, field in zip("ABCD", ("opa", "opb", "opc", "opd"))}
        if not all(isinstance(v, str) and v.strip() for v in options.values()):
            excluded["missing_option"] += 1
            continue
        cop = raw.get("cop")
        if isinstance(cop, bool) or not isinstance(cop, int) or not 0 <= cop - cop_base < 4:
            excluded["invalid_cop"] += 1
            continue
        label = "ABCD"[cop - cop_base]
        row = sample("medmcqa", revision, raw["id"], "medical_mcq", raw["question"], options[label],
                     {"dev": "validation"}.get(original_split, original_split), original_split=original_split,
                     source_url="https://github.com/medmcqa/medmcqa", options=options, option_id=label,
                     license_ref=license_ref, fetched_at=fetched_at,
                     metadata={"choice_type": raw["choice_type"], "source_cop": cop, "cop_base": cop_base,
                               "subject_name": raw.get("subject_name"), "topic_name": raw.get("topic_name"),
                               "evaluation_role": "official_validation_not_official_test" if original_split in ("dev", "validation") else "train"})
        row["supervision"]["rationale"] = raw.get("exp") or None
        records.append(row)
    return records, dict(excluded)


def aihub_71875_records(rows, *, revision, split, license_ref, fetched_at):
    _prerequisites(revision, license_ref)
    records = []
    for raw in rows:
        if not all(raw.get(key) for key in ("qa_id", "question", "answer", "q_type")):
            raise ValueError("71875 required qa_id/question/answer/q_type missing")
        row = sample("aihub_71875", revision, raw["qa_id"], "medical_qa", raw["question"], raw["answer"], split,
                     language="ko", source_url="https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71875",
                     original_split=split, license_ref=license_ref, fetched_at=fetched_at,
                     annotation="official_source_includes_LLM_generation; per-record expert authorship unknown",
                     metadata={"q_type": raw["q_type"], "domain": raw.get("domain"), "source_review": raw.get("review_status"),
                               "source_includes_llm_generation": True, "per_record_generation_method": "unknown"})
        row["review_status"] = "human_review_pending"
        row["group_ids"]["parent"] = raw.get("generation_parent")
        records.append(row)
    return records


def join_aihub_71762(questions, answers, *, question_key, answer_key, guide_evidence):
    """Join only verified explicit identifiers. No positional or disease/intent joins."""
    forbidden = {"disease", "disease_name", "intent", "q_intent", "index", "row", "file_index"}
    if not guide_evidence or not question_key or not answer_key or question_key in forbidden or answer_key in forbidden:
        raise ValueError("UNRESOLVED_PAIRING: explicit ID contract and source guide required")
    answer_map = {}
    for answer in answers:
        key = answer.get(answer_key)
        if key is None or key in answer_map:
            raise ValueError("UNRESOLVED_PAIRING: missing or ambiguous answer ID")
        answer_map[key] = answer
    joined = []
    for question in questions:
        key = question.get(question_key)
        if key is None or key not in answer_map:
            raise ValueError("UNRESOLVED_PAIRING: unmatched question ID")
        joined.append({"question_record": question, "answer_record": answer_map[key],
                       "pairing_evidence": guide_evidence, "parent_id": question.get("original_parent")})
    return joined
