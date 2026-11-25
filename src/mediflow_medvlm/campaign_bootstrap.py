"""CPU-only, validation-only paired cluster bootstrap auxiliary.

This measures within-training-seed item uncertainty. It does not measure
independent training-seed variance, select a family, or certify clinical quality.
The sampling recipe is fixed in the config before any intervals are computed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile

from . import campaign_full_dev as dev
from .campaign_supervisor import HASH_FIELDS, atomic_json, canonical_hash, file_hash
from .runtime import utcnow

SCHEMA = "medvlm.v6.bootstrap-config.v1"
SCOPE = "VALIDATION_ONLY_PAIRED_GROUP_BOOTSTRAP_AUXILIARY"
JOB_ID = "v6-bootstrap"
SOURCE_FILES = (*dev.SOURCE_FILES, "campaign_bootstrap.py")
IDENTITY_KEYS = dev.IDENTITY_KEYS
_require, _read, _absolute, _equal = dev._require, dev._read, dev._absolute, dev._equal
_reference, _ref = dev._reference, dev._ref
PROCEDURE = {
    "seed": 20261023, "resamples": 2000, "confidence": 0.95, "interval": "percentile",
    "quantile": "linear interpolation at (n-1)*p",
    "sampling": "connected components, stratified by source and stratum-membership profile; draw K components with replacement within each profile",
    "pairing": "same component multiplicities for Base and Tuned and every metric",
    "denominator": "sampled-cluster ratio: sampled weighted numerator / sampled row count; equivalent calibration to original stratum N; raw draw N may vary; retain technical failures",
    "empty_strata": "impossible under stratum-membership profiles; assert positive sampled count",
    "undefined_ratios": "report undefined counts; withhold interval if any replicate is undefined",
    "singleton_profiles": "one-component profiles have degenerate cluster uncertainty",
    "text_grouping": "source-scoped document/question/duplicate IDs, XML/subset-qualified MedQuAD local IDs, source QA IDs, PMID, and MedQuAD source URL",
    "vqa_grouping": "image/duplicate IDs, image SHA256, decoded pixel hash, and source-scoped image path",
    "missing_grouping": "explicit sample-ID singleton fallback; no independence claim",
    "uncertainty": "within-training-seed item uncertainty; not independent-seed variance",
}
FLAGS = {"test_files_read": False, "test_unlocked": False, "family_selection_completed": False,
         "clinical_quality_certified": False, "full_dev_completed": False,
         "r2_full_evaluation_completed": False, "campaign_terminal": False}
RATE_KEYS = ("normalized_em", "raw_em", "valid_format_rate", "invalid_format_rate", "abstention_rate",
             "technical_failure_rate", "truncated_rate", "lexical_f1", "strict_all_row_accuracy", "answered_accuracy")


def source_identity(repo=None):
    package = Path(__file__).resolve().parent if repo is None else Path(repo)/"src/mediflow_medvlm"
    files = {"src/mediflow_medvlm/"+name: file_hash(package/name) for name in SOURCE_FILES}
    return files, canonical_hash(files)


def _cpu():
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and "torch" not in sys.modules,
             "BOOTSTRAP_FRESH_CPU_PROCESS_REQUIRED")


def _native_inputs(aggregate_config_path):
    """Verify native execution, including scoring inputs, through its original verifier."""
    aggregate_ref = _reference(aggregate_config_path); aggregate = _ref(aggregate_ref)
    _require(aggregate.get("kind") == "aggregate" and aggregate.get("job_id") == dev.AGGREGATE_ID,
             "BOOTSTRAP_NATIVE_AGGREGATE_REQUIRED")
    manifest, state = dev.caps._active(aggregate["state_path"])
    jobs = {job["job_id"]: job for job in manifest["jobs"]}
    job = jobs[dev.AGGREGATE_ID]; record = state["jobs"][dev.AGGREGATE_ID]
    _require(record.get("status") == "SUCCEEDED", "BOOTSTRAP_AGGREGATE_SUCCESS_REQUIRED")
    registered, registered_ref = dev.caps._child_config(job, "campaign_full_dev")
    _require(registered_ref["sha256"] == aggregate_ref["sha256"] and _equal(registered, aggregate), "BOOTSTRAP_AGGREGATE_BINDING_CHANGED")
    # Accept the native output config copy, but bind the registered original.
    aggregate_ref = registered_ref
    native_repo = job["cwd"]
    receipt = dev.caps.initial.gate._success(job, record)
    _require(_equal(dev.verify_result(aggregate, repo=native_repo), receipt)
             and receipt.get("all_roles_succeeded") is True, "BOOTSTRAP_NATIVE_AGGREGATE_CHANGED")
    original = _ref(aggregate["data"]["initial_protocol"])
    selected = {axis: dev._rows(original, axis) for axis in ("text", "vqa")}
    roles = []
    scored = {}
    for reference, role_id in zip(aggregate["data"]["roles"], dev.ROLE_IDS):
        child = _ref(reference)
        dev.validate_config(child, repo=native_repo)
        child_receipt = dev.caps.initial.gate._success(jobs[role_id], state["jobs"][role_id])
        _require(state["jobs"][role_id].get("status") == "SUCCEEDED", "BOOTSTRAP_ALL_TWENTY_NATIVE_ROLES_REQUIRED")
        output = Path(child["output_dir"])
        rows = [json.loads(line) for line in (output/"scored.private.jsonl").read_text().splitlines()]
        _require([row["sample_id"] for row in rows] == [row["sample_id"] for row, _ in selected[child["task"]]],
                 "BOOTSTRAP_ORDERED_SCORE_IDS_CHANGED")
        roles.append({"job_id": role_id, "config": reference, "receipt": _reference(child["success_receipt"]),
                      "artifacts": copy.deepcopy(child_receipt["artifacts"])})
        scored[role_id] = rows
    proof = {"aggregate_config": aggregate_ref, "aggregate_receipt": _reference(job["success_receipt"]),
             "native_source_repo": native_repo, "roles": roles,
             "validation": {axis: {k: original["validation_references"][axis][k] for k in ("path", "sha256")} for axis in selected}}
    return aggregate, proof, selected, scored


def _values(value):
    return value if isinstance(value, list) else [value]


def grouping(selected, axis):
    """Build transitive components from observed source fields, never scored group_id."""
    _require(axis in {"text", "vqa"}, "BOOTSTRAP_AXIS_REQUIRED")
    ids = [row["sample_id"] for row, _ in selected]
    _require(len(ids) == len(set(ids)) and bool(ids), "BOOTSTRAP_UNIQUE_NONEMPTY_IDS_REQUIRED")
    parent = list(range(len(ids))); tokens_seen = {}; evidence = []; reason_counts = Counter()
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    def join(a, b):
        a, b = root(a), root(b)
        if a != b: parent[max(a, b)] = min(a, b)
    for i, (row, stratum) in enumerate(selected):
        _require(row.get("split") == "validation", "BOOTSTRAP_VALIDATION_ROWS_ONLY")
        source = row["source_id"]; meta = row.get("metadata", {}); groups = row.get("group_ids", {})
        tokens = set(); fields = set()
        def add(namespace, key, value):
            for item in _values(value):
                if item is not None and item != "" and item != []:
                    _require(isinstance(item, (str, int)), "BOOTSTRAP_GROUP_ID_MUST_BE_SCALAR")
                    tokens.add((namespace, key, str(item))); fields.add(key)
        for key in (("document", "question", "duplicate_cluster", "parent") if axis == "text" else ("image", "duplicate_cluster")):
            add(source, "group_ids."+key, groups.get(key))
        if axis == "text":
            for key in ("source_document_id", "source_question_id", "source_qa_id", "pmid"):
                namespace = source
                if source == "medquad" and key in {"source_document_id", "source_question_id"}:
                    context = meta.get("source_xml_file")
                    if not context:
                        reason_counts["MEDQUAD_LOCAL_ID_CONTEXT_UNAVAILABLE_IGNORED"] += bool(meta.get(key))
                        continue
                    namespace = canonical_hash([source, meta.get("subset"), context])
                add(namespace, "metadata."+key, meta.get(key))
            if source == "medquad": add(source, "provenance.source_url", row.get("provenance", {}).get("source_url"))
            if not any("document" in key or key == "metadata.pmid" for key in fields):
                reason_counts["SOURCE_DOCUMENT_GROUP_UNAVAILABLE"] += 1
            if not any("question" in key or key == "metadata.source_qa_id" for key in fields):
                reason_counts["SOURCE_QUESTION_GROUP_UNAVAILABLE"] += 1
        else:
            for mapping_name, mapping in (("metadata", meta), ("provenance", row.get("provenance", {}))):
                for key in ("image_sha256", "decoded_pixel_hash"):
                    # Same digest namespace across metadata/provenance and sources.
                    add("GLOBAL_IMAGE", key, mapping.get(key))
            for image in row.get("input", {}).get("images", []):
                add(source, "input.images", image)
            if not any("image" in key or "pixel_hash" in key for key in fields):
                reason_counts["IMAGE_GROUP_UNAVAILABLE"] += 1
        fallback = not tokens
        if fallback:
            reason_counts["NO_OBSERVED_GROUP_USING_SAMPLE_ID_SINGLETON"] += 1
            add(source, "sample_id_fallback", row["sample_id"])
        for token in tokens:
            if token in tokens_seen: join(i, tokens_seen[token])
            else: tokens_seen[token] = i
        evidence.append({"sample_id": row["sample_id"], "source_id": source, "stratum": stratum,
                         "source_row_sha256": canonical_hash(row), "tokens_sha256": canonical_hash(sorted(tokens)),
                         "observed_fields": sorted(fields), "singleton_fallback": fallback,
                         "grouping_quality": row.get("grouping_quality", {})})
    components = defaultdict(list)
    for i in range(len(ids)): components[root(i)].append(i)
    units = list(components.values())
    profiles = defaultdict(list)
    for unit_id, indices in enumerate(units):
        for i in indices: evidence[i]["component_index"] = unit_id
        sources = tuple(sorted({selected[i][0]["source_id"] for i in indices}))
        strata = tuple(sorted({selected[i][1] for i in indices}))
        profiles[(sources, strata)].append(unit_id)
    projection = [{"indices": indices, "sample_ids_sha256": canonical_hash([ids[i] for i in indices])} for indices in units]
    return {"axis": axis, "scheduled": len(ids), "group_count": len(units),
            "stratum_counts": dict(sorted(Counter(stratum for _, stratum in selected).items())),
            "group_size_counts": {str(size):count for size, count in sorted(Counter(len(unit) for unit in units).items())},
            "unavailable_reason_counts": dict(sorted(reason_counts.items())),
            "singleton_fallback_count": sum(e["singleton_fallback"] for e in evidence),
            "source_evidence_sha256": canonical_hash(evidence), "components_sha256": canonical_hash(projection),
            "profiles": [{"sources": list(sources), "strata": list(strata), "group_count": len(members)} for (sources, strata), members in profiles.items()],
            "patient_or_study_independence_certified": False}, units, list(profiles.values()), evidence


def prepare_config(aggregate_config_path, output_dir, *, repo=None, success_receipt=None):
    aggregate, proof, selected, _ = _native_inputs(aggregate_config_path)
    original = _ref(aggregate["data"]["initial_protocol"]); root = _absolute(original["campaign_root"])
    output = _absolute(output_dir); receipt = _absolute(success_receipt or root/"receipts"/(JOB_ID+".json"))
    # The auxiliary has a dedicated namespace and can never replace native outputs.
    auxiliary_root = root/"auxiliary"/"bootstrap"
    for path in (output, receipt):
        dev.caps.initial.gate._private(path, root)
        _require(not path.is_symlink() and not any(p.is_symlink() for p in path.parents), "BOOTSTRAP_SYMLINK_DESTINATION_FORBIDDEN")
    _require(output.resolve().is_relative_to(auxiliary_root.resolve()) and output != auxiliary_root
             and receipt.parent == root/"receipts" and receipt.name == JOB_ID+".json",
             "BOOTSTRAP_DEDICATED_OUTPUT_AND_RECEIPT_REQUIRED")
    proof["grouping"] = {axis: grouping(rows, axis)[0] for axis, rows in selected.items()}
    files, source = source_identity(repo)
    config = {"schema_version": SCHEMA, "scope": SCOPE, "kind": "bootstrap", "phase": "bootstrap",
              "job_id": JOB_ID, "run_id": JOB_ID, "cell_id": "ALL_V6", "seed": 20261023,
              "uses_gpu": False, "device": "cpu", "state_path": aggregate["state_path"],
              "output_dir": str(output), "success_receipt": str(receipt), "data": proof,
              "procedure": copy.deepcopy(PROCEDURE), "model_hash": aggregate["model_hash"],
              "resource_policy": aggregate["resource_policy"], "source_files": files, "source_hash": source,
              "data_hash": canonical_hash(proof)}
    config["config_hash"] = canonical_hash(config)
    return config


def validate_config(config, *, repo=None):
    _require(config.get("schema_version") == SCHEMA and config.get("config_hash") == canonical_hash({k:v for k,v in config.items() if k != "config_hash"})
             and (config.get("source_files"), config.get("source_hash")) == source_identity(repo), "BOOTSTRAP_CONFIG_HASH_OR_SOURCE_CHANGED")
    expected = prepare_config(config["data"]["aggregate_config"]["path"], config["output_dir"], repo=repo, success_receipt=config["success_receipt"])
    _require(_equal(config, expected), "BOOTSTRAP_FIXED_PROCEDURE_OR_NATIVE_PROVENANCE_CHANGED")
    return config


def job_from_config(config_path, *, repo=None, python=None):
    path = _absolute(config_path); config = _read(path); validate_config(config, repo=repo)
    return {"job_id": JOB_ID, "run_id": JOB_ID, "cell_id": "ALL_V6", "phase": "bootstrap", "kind": "bootstrap", "seed": 20261023,
            "dependencies": [dev.AGGREGATE_ID], "dependency_policy": "success", "implementation_pending": False,
            "uses_gpu": False, "requires_gpu": False, "gpu_lock_mode": "wrapper",
            "argv": [str(python or sys.executable), "-m", "mediflow_medvlm.campaign_bootstrap", "--config", str(path), "--receipt", config["success_receipt"]],
            "cwd": str(Path(repo).resolve() if repo else Path(__file__).resolve().parents[2]),
            "success_receipt": config["success_receipt"], "source_files": config["source_files"], **{k:config[k] for k in HASH_FIELDS}}


def _vector(row):
    # Every status occupies its scheduled row, including failed/unattempted rows.
    return (float(row["exact_match"]), float(row["raw_exact_match"]), float(row["valid"]),
            float(row["status"] == "invalid_format"), float(row["status"] == "abstained"),
            float(row["status"] == "technical_failure"), float(row["finish_reason"] == "length"), float(row["token_f1"]))


def _metric_vector(totals, denominator, classification):
    values = [value/denominator for value in totals]
    return dict(zip(RATE_KEYS, (*values[:7], None if classification else values[7],
                               values[0] if classification else None, totals[0]/totals[2] if totals[2] else None)))


def _quantile(values, probability):
    ordered = sorted(values); position = (len(ordered)-1)*probability
    left = math.floor(position); right = math.ceil(position)
    return ordered[left]+(ordered[right]-ordered[left])*(position-left)


def _interval(point, values, reason=None):
    complete = len(values) == PROCEDURE["resamples"]
    return {"estimate": point, "lower": _quantile(values, .025) if complete else None,
            "upper": _quantile(values, .975) if complete else None, "defined_resamples": len(values),
            "undefined_resamples": PROCEDURE["resamples"]-len(values),
            "unavailable_reason": None if complete else (reason if not values else "UNDEFINED_IN_SOME_RESAMPLES")}


def bootstrap_pair(selected, base, tuned, units, profiles):
    """One fixed paired computation; no mutation, GPU, or data file reads."""
    ids = [row["sample_id"] for row, _ in selected]
    _require([r["sample_id"] for r in base] == ids == [r["sample_id"] for r in tuned], "BOOTSTRAP_PAIR_IDS_CHANGED")
    strata = sorted({stratum for _, stratum in selected}); prepared = {}
    for stratum in strata:
        indices = [i for i, (_, s) in enumerate(selected) if s == stratum]
        classification = selected[indices[0]][0]["task_type"] in {"medical_mcq", "context_qa"}
        _require(len({selected[i][0]["task_type"] for i in indices}) == 1, "BOOTSTRAP_HOMOGENEOUS_STRATA_REQUIRED")
        members = set(indices); grouped = []
        for unit_id, unit in enumerate(units):
            within = [i for i in unit if i in members]
            if within:
                grouped.append((unit_id, len(within), tuple(sum(_vector(rows[i])[k] for i in within) for rows in (base, tuned) for k in range(8))))
        totals = [sum(group[2][k] for group in grouped) for k in range(16)]
        points = [_metric_vector(totals[offset:offset+8], len(indices), classification) for offset in (0, 8)]
        prepared[stratum] = (indices, classification, grouped, points, {key:[[], [], []] for key in RATE_KEYS})
    rng = random.Random(PROCEDURE["seed"])
    for _ in range(PROCEDURE["resamples"]):
        weights = [0]*len(units)
        for members in profiles:
            for _ in members: weights[members[rng.randrange(len(members))]] += 1
        for _, classification, grouped, _, distributions in prepared.values():
            denominator = sum(weights[unit_id]*count for unit_id, count, _ in grouped)
            _require(denominator > 0, "BOOTSTRAP_EMPTY_SAMPLED_STRATUM")
            totals = [sum(weights[unit_id]*vector[k] for unit_id, _, vector in grouped) for k in range(16)]
            pair = [_metric_vector(totals[offset:offset+8], denominator, classification) for offset in (0, 8)]
            for key, values in distributions.items():
                a, b = pair[0][key], pair[1][key]
                if a is not None: values[0].append(a)
                if b is not None: values[1].append(b)
                if a is not None and b is not None: values[2].append(b-a)
    result = {}
    for stratum, (indices, classification, _, points, distributions) in prepared.items():
        metrics = {}
        for key, values in distributions.items():
            a, b = points[0][key], points[1][key]
            reason = "NOT_APPLICABLE_TO_TASK" if (key == "lexical_f1" and classification or key == "strict_all_row_accuracy" and not classification) else "NO_VALID_ANSWERS"
            metrics[key] = {name:_interval(point, distribution, reason) for name, point, distribution in
                            zip(("base", "tuned", "tuned_minus_base"), (a, b, b-a if a is not None and b is not None else None), values)}
        result[stratum] = {"scheduled": len(indices), "denominator_policy": PROCEDURE["denominator"],
                           "observed_counts": {name:{"valid_answers":sum(bool(rows[i]["valid"]) for i in indices),
                               "technical_failures":sum(rows[i]["status"] == "technical_failure" for i in indices)} for name, rows in (("base", base), ("tuned", tuned))},
                           "metrics": metrics, "clinical_accuracy": None}
    return result


def _summary(config, progress=None):
    _, proof, selected, scored = _native_inputs(config["data"]["aggregate_config"]["path"])
    grouping_reports = {}; internals = {}; evidence = {}
    for axis, rows in selected.items():
        report, units, profiles, projection = grouping(rows, axis)
        _require(_equal(report, config["data"]["grouping"][axis]), "BOOTSTRAP_GROUPING_CHANGED")
        grouping_reports[axis] = report; internals[axis] = (units, profiles); evidence[axis] = projection
    _require(all(_equal(proof[key], config["data"][key]) for key in proof), "BOOTSTRAP_INPUT_PROOF_CHANGED")
    pairs = {}
    for cell in dev.CELLS:
        cell_id = cell[0]; axis = dev.cell_spec(cell_id)["task"]
        pairs[cell_id] = {"axis": axis, "strata": bootstrap_pair(selected[axis], scored[f"v6-full-dev-{cell_id}-base"],
                            scored[f"v6-full-dev-{cell_id}-tuned"], *internals[axis])}
        if progress is not None: progress(len(pairs))
    return {"scope": SCOPE, "bootstrap_auxiliary_completed": True, "procedure": config["procedure"],
            "grouping": grouping_reports, "pairs": pairs, "pair_count": len(pairs), "training_seed_count": 1,
            "independent_seed_variance_estimated": False, "statistical_superiority_claimed": False, "all_planned_rows_included": True,
            "native_input_proof_sha256": canonical_hash(config["data"]), **FLAGS}, evidence


def verify_result(config, *, repo=None):
    validate_config(config, repo=repo)
    output = Path(config["output_dir"]); names = ("config.json", "summary.json", "grouping.private.json")
    _require(not output.is_symlink() and {p.name for p in output.iterdir()} == set(names)|{"completion.json"}
             and not any(p.is_symlink() for p in output.iterdir()), "BOOTSTRAP_OUTPUT_SET_CHANGED")
    marker = _read(output/"completion.json"); summary, evidence = _summary(config)
    _require(_equal(_read(output/"config.json"), config) and _equal(_read(output/"summary.json"), summary)
             and _equal(_read(output/"grouping.private.json"), evidence)
             and marker.get("schema_version") == "medvlm.v6.bootstrap-receipt.v1" and marker.get("status") == "SUCCEEDED"
             and marker.get("scope") == SCOPE and marker.get("bootstrap_auxiliary_completed") is True
             and all(marker.get(key) == value for key, value in FLAGS.items())
             and all(marker.get(k) == config[k] for k in IDENTITY_KEYS)
             and marker.get("artifacts") == [_reference(output/name) for name in names], "BOOTSTRAP_RESULT_CHANGED")
    return {**marker, "artifacts": marker["artifacts"]+[_reference(output/"completion.json")]}


def run_job(config_path, receipt_path):
    _cpu(); path = _absolute(config_path); config = _read(path); before = file_hash(path)
    validate_config(config); dev.caps._env(config)
    receipt = _absolute(receipt_path); _require(str(receipt) == config["success_receipt"], "BOOTSTRAP_RECEIPT_ROUTE_CHANGED")
    output = Path(config["output_dir"]); output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with output.with_name(output.name+".worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not output.exists():
            _require(not receipt.exists(), "BOOTSTRAP_RECEIPT_WITHOUT_OUTPUT")
            staging = Path(tempfile.mkdtemp(prefix=output.name+".attempt-", dir=output.parent)); mask = os.umask(0o077)
            try:
                def progress(completed):
                    progress_path = os.environ.get("MEDVLM_SUPERVISOR_PROGRESS_PATH")
                    if progress_path:
                        atomic_json(progress_path, {**{k:config[k] for k in IDENTITY_KEYS}, "status":"RUNNING",
                            "stage":"bootstrap", "pairs_completed":completed, "total_pairs":len(dev.CELLS),
                            "resamples":PROCEDURE["resamples"], "bootstrap_auxiliary_completed":None, "updated_at":utcnow()})
                summary, evidence = _summary(config, progress=progress)
                for name, value in (("config.json", config), ("summary.json", summary), ("grouping.private.json", evidence)):
                    atomic_json(staging/name, value)
                validate_config(config); _cpu(); _require(file_hash(path) == before, "BOOTSTRAP_CONFIG_CHANGED_DURING_EXECUTION")
                marker = {"schema_version": "medvlm.v6.bootstrap-receipt.v1", "status": "SUCCEEDED", "scope": SCOPE,
                          **{k:config[k] for k in IDENTITY_KEYS}, "bootstrap_auxiliary_completed": True, **FLAGS,
                          "artifacts": [{"path":str(output/name), "sha256":file_hash(staging/name)} for name in ("config.json", "summary.json", "grouping.private.json")]}
                atomic_json(staging/"completion.json", marker); os.rename(staging, output)
                fd = os.open(output.parent, os.O_DIRECTORY)
                try: os.fsync(fd)
                finally: os.close(fd)
            finally: os.umask(mask)
        result = verify_result(config); _cpu()
        _require(file_hash(path) == before and (not receipt.exists() or _equal(_read(receipt), result)), "BOOTSTRAP_COMPLETED_REUSE_CHANGED")
        if not receipt.exists(): atomic_json(receipt, result)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    # Supervisor CPU jobs inherit the owner environment; isolate only this child.
    try:
        _require("torch" not in sys.modules, "BOOTSTRAP_FRESH_CPU_PROCESS_REQUIRED")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        result = run_job(args.config, args.receipt)
    except Exception as exc:
        print(json.dumps({"status":"FAILED", "exception_type":type(exc).__name__, "reason_code":str(exc) if isinstance(exc, dev.caps.initial.gate.GateError) else "BOOTSTRAP_FAILURE"})); return 1
    print(json.dumps({"status":result["status"], "scope":result["scope"]})); return 0


if __name__ == "__main__": raise SystemExit(main())
