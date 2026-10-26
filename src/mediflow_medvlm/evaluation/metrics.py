"""Conservative answer parsing and denominators that preserve every failure."""
from collections import Counter, defaultdict
import json
import random
import re
import unicodedata
import statistics


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split())


def subject_name(sample):
    value = sample.get("metadata", {}).get("subject")
    return "pharm" if sample["source_id"] == "kormedmcqa" and value == "pharmacy" else value


def parse_answer(output, task, options=None):
    text = output.strip()
    if not text:
        return None, "invalid_format"
    if normalize(text) in {"abstain", "i don't know", "i do not know", "cannot answer"}:
        return None, "abstained"
    if task not in {"medical_mcq", "context_qa"}:
        return text, "answered"
    try:
        value = json.loads(text)
        if isinstance(value, dict) and set(value) == {"final_answer"}:
            text = str(value["final_answer"]).strip()
    except (json.JSONDecodeError, TypeError):
        pass
    if task == "medical_mcq":
        match = re.fullmatch(r"(?:final_answer\s*[:=]\s*)?\(?([A-E])\)?[.]?", text, re.I)
        answer = match.group(1).upper() if match else None
        valid = answer in (options or {x: x for x in "ABCDE"})
    else:
        match = re.fullmatch(r"(?:final_answer\s*[:=]\s*)?(yes|no|maybe)[.]?", text, re.I)
        answer = match.group(1).lower() if match else None
        valid = answer in {"yes", "no", "maybe"}
    return (answer, "answered") if valid else (None, "invalid_format")


def lexical_f1(prediction, gold):
    # Keep numbers, units and negation; never claim this is medical correctness.
    pred, ref = Counter(normalize(prediction).split()), Counter(normalize(gold).split())
    overlap = sum((pred & ref).values())
    if not pred or not ref:
        return float(pred == ref)
    return 2 * overlap / (sum(pred.values()) + sum(ref.values()))


def score_rows(samples, predictions, *, vqa_normalizer=None):
    lookup = {}
    for pred in predictions:
        sid = pred["sample_id"]
        if sid in lookup:
            raise ValueError("duplicate final prediction; resolve attempts explicitly")
        lookup[sid] = pred
    sample_ids = {s["sample_id"] for s in samples}
    if len(sample_ids) != len(samples) or set(lookup) - sample_ids:
        raise ValueError("prediction/sample identity mismatch")
    rows = []
    for sample in samples:
        pred = lookup.get(sample["sample_id"], {"status": "not_attempted", "attempted": False})
        valid = pred["status"] == "answered" and pred.get("parsed_answer") is not None
        gold = sample["supervision"].get("option_id") if sample["task_type"] == "medical_mcq" else sample["supervision"]["answer"]
        normalizer = vqa_normalizer if sample["task_type"] == "medical_vqa" and vqa_normalizer else normalize
        score = float(normalizer(pred["parsed_answer"]) == normalizer(gold)) if valid else 0.0
        group = (sample.get("group_ids") or {}).get("duplicate_cluster") or (sample.get("group_ids") or {}).get("image") or (sample.get("group_ids") or {}).get("document") or sample["sample_id"]
        rows.append({"sample_id": sample["sample_id"], "source_id": sample["source_id"],
                     "task_type": sample["task_type"], "language": sample["language"],
                     "subject": subject_name(sample), "group_id": str(group),
                     "answer_type": sample.get("metadata", {}).get("answer_type"),
                     "modality": sample.get("metadata", {}).get("modality"),
                     "status": pred["status"], "attempted": pred.get("attempted", True),
                     "valid": valid, "exact_match": score,
                     "token_f1": lexical_f1(normalizer(pred["parsed_answer"]), normalizer(gold)) if valid else 0.0,
                     "reference_label": str(gold) if sample["task_type"] in {"medical_mcq", "context_qa"} else None,
                     "predicted_label": pred.get("parsed_answer") if sample["task_type"] in {"medical_mcq", "context_qa"} else None,
                     "finish_reason": pred.get("finish_reason"),
                     "latency_seconds": pred.get("timing", {}).get("e2e_seconds")})
    return rows


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (row["source_id"], row["task_type"], row["language"], row.get("subject"))
        groups[key].append(row)
    output = []
    for key, items in sorted(groups.items(), key=lambda x: str(x[0])):
        n = len(items); counts = Counter(x["status"] for x in items)
        valid = sum(x["valid"] for x in items); correct = sum(x["exact_match"] for x in items)
        classification = key[1] in {"medical_mcq", "context_qa"}
        labels = sorted({x["reference_label"] for x in items if x["reference_label"] is not None})
        class_f1 = []
        support = {}
        for label in labels:
            tp = sum(x["valid"] and x["predicted_label"] == label and x["reference_label"] == label for x in items)
            fp = sum(x["valid"] and x["predicted_label"] == label and x["reference_label"] != label for x in items)
            fn = sum(x["reference_label"] == label and (not x["valid"] or x["predicted_label"] != label) for x in items)
            class_f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
            support[label] = sum(x["reference_label"] == label for x in items)
        completed = sum(counts[x] for x in ("answered", "abstained", "invalid_format"))
        latencies = sorted(x["latency_seconds"] for x in items if x["latency_seconds"] is not None)
        output.append({"source_id": key[0], "task_type": key[1], "language": key[2], "subject": key[3],
                       "scheduled": n, "attempted": sum(x["attempted"] for x in items), "completed": completed,
                       "valid_answers": valid, "correct" if classification else "exact_matches": int(correct),
                       "statuses": dict(counts), "coverage": valid / n,
                       "answered_accuracy" if classification else "answered_lexical_exact_match": correct / valid if valid else None,
                       "end_to_end_correct_fraction" if classification else "end_to_end_lexical_exact_match": correct / n,
                       "lexical_token_f1": None if classification else sum(x["token_f1"] for x in items) / n,
                       "macro_f1": sum(class_f1) / len(class_f1) if class_f1 else None, "label_support": support,
                       "invalid_format_rate": counts["invalid_format"] / n,
                       "not_attempted_count": counts["not_attempted"],
                       "technical_failure_rate": (n - completed - counts["not_attempted"]) / n,
                       "truncated_count": sum(x["finish_reason"] == "length" for x in items),
                       "e2e_latency_seconds": {"n": len(latencies), "median": statistics.median(latencies) if latencies else None,
                                               "min": min(latencies) if latencies else None, "max": max(latencies) if latencies else None,
                                               "scope": "includes_input_preprocess_and_generation_including_failed_attempts_when_measured"},
                       "human_review": "not_performed", "clinical_accuracy": None})
    return output


def vqa_subgroups(rows):
    output = []
    for key in ("answer_type", "modality"):
        groups = defaultdict(list)
        for row in rows:
            if row["task_type"] == "medical_vqa":
                groups[(row["source_id"], row.get(key))].append(row)
        for (source, value), items in sorted(groups.items(), key=lambda x: str(x[0])):
            output.append({"source_id": source, "dimension": key, "value": value,
                           "metrics": aggregate(items)[0]})
    return output


def paired_compare(base, tuned, resamples=2000, seed=20261023):
    left = {x["sample_id"]: x for x in base}; right = {x["sample_id"]: x for x in tuned}
    if len(left) != len(base) or len(right) != len(tuned) or set(left) != set(right):
        raise ValueError("paired comparison requires identical unique scheduled IDs")
    grouped = defaultdict(list)
    for sid, a in left.items():
        b = right[sid]
        if any(a.get(k) != b.get(k) for k in ("source_id", "task_type", "language", "group_id")):
            raise ValueError("pair cohort/group differs")
        grouped[(a["source_id"], a["task_type"], a["language"])].append((a, b))
    results = []
    rng = random.Random(seed)
    for cohort, pairs in sorted(grouped.items()):
        group_values = defaultdict(list)
        for a, b in pairs:
            group_values[a["group_id"]].append(b["exact_match"] - a["exact_match"])
        groups = list(group_values.values()); draws = []
        for _ in range(resamples):
            values = [v for group in rng.choices(groups, k=len(groups)) for v in group]
            draws.append(sum(values) / len(values))
        draws.sort()
        results.append({"source_id": cohort[0], "task_type": cohort[1], "language": cohort[2], "n": len(pairs),
                        "group_count": len(groups), "delta": sum(b["exact_match"]-a["exact_match"] for a,b in pairs)/len(pairs),
                        "ci95": [draws[int(.025*len(draws))], draws[min(len(draws)-1,int(.975*len(draws)))]] if draws else None,
                        "bootstrap_unit": "observed_source_group", "bootstrap_resamples": resamples,
                        "training_seed_uncertainty_measured": False,
                        "metric": "end_to_end_correct_fraction" if cohort[1] in {"medical_mcq", "context_qa"} else "end_to_end_lexical_exact_match"})
    return results


def korean_summary(cohorts):
    groups = [x for x in cohorts if x["source_id"] == "kormedmcqa" and x["language"] == "ko"]
    if not groups:
        return None
    n = sum(x["scheduled"] for x in groups); valid = sum(x["valid_answers"] for x in groups)
    correct = sum(x["correct"] for x in groups)
    return {"scheduled": n, "valid_answers": valid, "correct": correct,
            "micro_end_to_end_correct_fraction": correct/n if n else None,
            "micro_answered_accuracy": correct/valid if valid else None,
            "coverage": valid/n if n else None,
            "macro_subject_end_to_end_correct_fraction": sum(x["end_to_end_correct_fraction"] for x in groups)/len(groups),
            "subject_count": len(groups), "subjects": [x["subject"] for x in groups]}
