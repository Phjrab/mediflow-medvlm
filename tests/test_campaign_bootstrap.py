"""Fixed CPU bootstrap math, grouping provenance, and immutable auxiliary lifecycle."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from mediflow_medvlm import campaign_bootstrap as b

REPO = Path(__file__).resolve().parents[1]


def row(sample_id, *, source="synthetic", task="medical_mcq", groups=None, metadata=None):
    return {"sample_id":sample_id, "source_id":source, "split":"validation", "task_type":task,
            "input":{"images":[]}, "group_ids":groups or {}, "metadata":metadata or {}, "provenance":{}}


def score(sample_id, *, correct=0, valid=True, status="valid", f1=0., finish="stop"):
    return {"sample_id":sample_id, "exact_match":correct, "raw_exact_match":correct, "valid":valid,
            "status":status, "token_f1":f1, "finish_reason":finish}


def paired(selected, base, tuned):
    _, units, profiles, _ = b.grouping(selected, "text")
    return b.bootstrap_pair(selected, base, tuned, units, profiles)


def test_fixed_protocol_cpu_import_and_source_identity():
    assert b.PROCEDURE["seed"] == 20261023 and b.PROCEDURE["resamples"] == 2000
    assert b.PROCEDURE["confidence"] == .95 and b.PROCEDURE["interval"] == "percentile"
    files, digest = b.source_identity(REPO)
    native, _ = b.dev.source_identity(REPO)
    assert set(files) == set(native)|{"src/mediflow_medvlm/campaign_bootstrap.py"}
    assert digest == b.canonical_hash(files)
    completed = subprocess.run([sys.executable, "-c", "import sys; from mediflow_medvlm import campaign_bootstrap; assert 'torch' not in sys.modules"],
                               cwd=REPO, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_grouping_transitive_edges_and_cross_stratum_pair_units():
    selected = [(row("a", groups={"document":"doc"}), "short"),
                (row("b", groups={"document":"doc", "duplicate_cluster":"dup"}), "long"),
                (row("c", groups={"duplicate_cluster":"dup"}), "long")]
    report, units, profiles, evidence = b.grouping(selected, "text")
    assert units == [[0, 1, 2]] and profiles == [[0]]
    assert report["profiles"][0]["strata"] == ["long", "short"]
    assert report["group_count"] == 1 and report["scheduled"] == 3
    assert report["source_evidence_sha256"] == b.canonical_hash(evidence)
    changed = copy.deepcopy(selected); changed[0][0]["input"]["question"] = "synthetic changed content"
    assert b.grouping(changed, "text")[0]["source_evidence_sha256"] != report["source_evidence_sha256"]


def test_medquad_local_ids_qualified_by_xml_and_canonical_duplicates_preserved():
    selected = [(row("a", source="medquad", metadata={"source_document_id":"1", "source_question_id":"1-1", "source_xml_file":"A/1.xml", "subset":"A"}), "medquad"),
                (row("b", source="medquad", metadata={"source_document_id":"1", "source_question_id":"1-1", "source_xml_file":"B/1.xml", "subset":"B"}), "medquad")]
    assert b.grouping(selected, "text")[1] == [[0], [1]]
    for r, _ in selected: r["group_ids"]["duplicate_cluster"] = "observed-duplicate"
    assert b.grouping(selected, "text")[1] == [[0, 1]]
    selected[0][0]["metadata"].pop("source_xml_file")
    assert b.grouping(selected, "text")[0]["unavailable_reason_counts"]["MEDQUAD_LOCAL_ID_CONTEXT_UNAVAILABLE_IGNORED"] == 2


def test_vqa_global_image_hash_joins_sources_and_duplicate_bridge():
    selected = [(row("a", source="one", metadata={"image_sha256":"image-A"}), "open"),
                (row("b", source="two", metadata={"image_sha256":"image-A"}, groups={"duplicate_cluster":"dup"}), "closed"),
                (row("c", source="two", groups={"duplicate_cluster":"dup"}), "open")]
    report, units, _, _ = b.grouping(selected, "vqa")
    assert units == [[0, 1, 2]] and report["group_count"] == 1
    assert report["unavailable_reason_counts"] == {"IMAGE_GROUP_UNAVAILABLE":1}
    assert report["patient_or_study_independence_certified"] is False


def test_explicit_singleton_fallback_and_no_use_of_scored_group_id():
    selected = [(row("a"), "s"), (row("b"), "s")]
    report, units, _, evidence = b.grouping(selected, "text")
    assert units == [[0], [1]] and report["singleton_fallback_count"] == 2
    assert report["unavailable_reason_counts"]["NO_OBSERVED_GROUP_USING_SAMPLE_ID_SINGLETON"] == 2
    assert all(e["singleton_fallback"] for e in evidence)
    selected[0][0]["split"] = "test"
    with pytest.raises(ValueError, match="VALIDATION_ROWS_ONLY"): b.grouping(selected, "text")


def test_paired_identical_predictions_have_exact_zero_delta_every_metric():
    selected = [(row(str(i), groups={"document":str(i//2)}), "s") for i in range(8)]
    scores = [score(str(i), correct=i%2, f1=i%2) for i in range(8)]
    result = paired(selected, scores, scores)["s"]["metrics"]
    for name, metric in result.items():
        delta = metric["tuned_minus_base"]
        if name == "lexical_f1": assert delta["lower"] is None
        else: assert delta["estimate"] == delta["lower"] == delta["upper"] == 0


def test_planned_technical_failures_remain_in_pair_and_denominator():
    selected = [(row(str(i)), "s") for i in range(10)]
    base = [score(str(i), correct=int(i>=6), valid=i>=6, status="technical_failure" if i<6 else "valid") for i in range(10)]
    tuned = [score(str(i), correct=1) for i in range(10)]
    result = paired(selected, base, tuned)["s"]
    assert result["scheduled"] == 10 and result["observed_counts"]["base"]["technical_failures"] == 6
    assert result["metrics"]["strict_all_row_accuracy"]["base"]["estimate"] == .4
    assert result["metrics"]["technical_failure_rate"]["base"]["estimate"] == .6
    assert result["metrics"]["normalized_em"]["tuned_minus_base"]["estimate"] == .6
    assert result["metrics"]["answered_accuracy"]["base"]["estimate"] == 1


def test_unequal_cluster_sizes_use_sampled_ratio_and_deterministic_percentiles():
    selected = [(row(str(i), groups={"document":"large" if i else "small"}), "s") for i in range(4)]
    base = [score(str(i)) for i in range(4)]
    tuned = [score(str(i), correct=int(i>0)) for i in range(4)]
    first = paired(selected, base, tuned)
    assert paired(selected, base, tuned) == first
    metric = first["s"]["metrics"]["normalized_em"]["tuned"]
    assert metric["estimate"] == .75 and metric["lower"] == 0 and metric["upper"] == 1
    assert metric["defined_resamples"] == 2000
    assert b._quantile([0., 1.], .025) == .025


def test_shared_stratum_profile_weights_and_degenerate_singleton_profile():
    selected = [(row("a", groups={"document":"D"}), "one"),
                (row("b", groups={"document":"D"}), "two")]
    base = [score("a"), score("b")]; tuned = [score("a", correct=1), score("b", correct=1)]
    results = paired(selected, base, tuned)
    for stratum in results.values():
        metric = stratum["metrics"]["normalized_em"]["tuned_minus_base"]
        assert metric["lower"] == metric["upper"] == 1 and metric["defined_resamples"] == 2000


def test_undefined_answered_accuracy_withholds_interval_and_reports_missing_draws():
    selected = [(row(str(i)), "s") for i in range(2)]
    base = [score("0", correct=1), score("1", valid=False, status="technical_failure")]
    tuned = [score("0", valid=False, status="invalid_format"), score("1", valid=False, status="invalid_format")]
    results = paired(selected, base, tuned)["s"]["metrics"]["answered_accuracy"]
    assert results["base"]["estimate"] == 1 and results["base"]["lower"] is None
    assert 0 < results["base"]["undefined_resamples"] < 2000
    assert results["base"]["unavailable_reason"] == "UNDEFINED_IN_SOME_RESAMPLES"
    assert results["tuned"]["estimate"] is None and results["tuned"]["undefined_resamples"] == 2000
    assert results["tuned_minus_base"]["estimate"] is None


def test_pair_order_tamper_rejected():
    selected = [(row("a"), "s"), (row("b"), "s")]
    with pytest.raises(ValueError, match="PAIR_IDS_CHANGED"):
        paired(selected, [score("a"), score("b")], [score("b"), score("a")])


@pytest.fixture
def auxiliary(tmp_path, monkeypatch):
    root = tmp_path/"campaign"; root.mkdir()
    original = root/"original.json"; b.atomic_json(original, {"campaign_root":str(root)})
    aggregate_path = root/"aggregate-config.json"; b.atomic_json(aggregate_path, {"synthetic":True})
    aggregate = {"data":{"initial_protocol":b._reference(original)}, "state_path":str(root/"state.json"),
                 "model_hash":"synthetic", "resource_policy":{}}
    selected = {axis:[(row(axis+str(i), task="visual_qa" if axis=="vqa" else "medical_mcq"), "s")] for axis in ("text", "vqa") for i in [0]}
    proof = {"aggregate_config":b._reference(aggregate_path), "aggregate_receipt":{"path":"synthetic", "sha256":"synthetic"},
             "native_source_repo":str(REPO), "roles":[], "validation":{}}
    scored = {f"v6-full-dev-{cell[0]}-{role}":[score(b.dev.cell_spec(cell[0])["task"]+"0", correct=role=="tuned")]
              for cell in b.dev.CELLS for role in ("base", "tuned")}
    calls = []
    def native(path):
        calls.append(str(path))
        return copy.deepcopy(aggregate), copy.deepcopy(proof), copy.deepcopy(selected), copy.deepcopy(scored)
    monkeypatch.setattr(b, "_native_inputs", native)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    config = b.prepare_config(aggregate_path, root/"auxiliary/bootstrap/first_seed", repo=REPO)
    config_path = root/"bootstrap-config.json"; b.atomic_json(config_path, config)
    return root, config, config_path, calls


def test_config_fixed_scientific_fields_cannot_be_self_rehashed(auxiliary):
    _, config, _, _ = auxiliary
    assert b.validate_config(config, repo=REPO) == config
    for key, value in (("procedure", {**b.PROCEDURE, "resamples":1}), ("uses_gpu", True), ("seed", 5)):
        changed = copy.deepcopy(config); changed[key] = value
        changed["config_hash"] = b.canonical_hash({k:v for k,v in changed.items() if k!="config_hash"})
        with pytest.raises(ValueError, match="FIXED_PROCEDURE_OR_NATIVE_PROVENANCE"): b.validate_config(changed, repo=REPO)


def test_dedicated_namespace_and_cpu_job(auxiliary):
    root, config, path, _ = auxiliary
    for output in (root/"full_dev/aggregate", root/"auxiliary/bootstrap"):
        with pytest.raises(ValueError, match="DEDICATED_OUTPUT"): b.prepare_config(config["data"]["aggregate_config"]["path"], output)
    job = b.job_from_config(path, repo=REPO)
    assert job["dependencies"] == ["v6-full_dev"] and job["dependency_policy"] == "success"
    assert job["requires_gpu"] is False and job["uses_gpu"] is False
    assert job["argv"][1:3] == ["-m", "mediflow_medvlm.campaign_bootstrap"]


def test_immutable_cpu_lifecycle_idempotence_and_progress_only_during_execution(auxiliary, monkeypatch):
    root, config, path, _ = auxiliary
    progress = root/"progress.json"; monkeypatch.setenv("MEDVLM_SUPERVISOR_PROGRESS_PATH", str(progress))
    result = b.run_job(path, config["success_receipt"])
    assert result["status"] == "SUCCEEDED" and result["bootstrap_auxiliary_completed"] is True
    assert all(result[key] is False for key in b.FLAGS)
    summary = b._read(Path(config["output_dir"])/"summary.json")
    assert summary["pair_count"] == 10 and summary["independent_seed_variance_estimated"] is False
    assert b._read(progress)["pairs_completed"] == 10
    before = progress.read_bytes()
    assert b.verify_result(config, repo=REPO) == result and progress.read_bytes() == before
    assert b.run_job(path, config["success_receipt"]) == result
    output = Path(config["output_dir"]); changed = b._read(output/"summary.json"); changed["clinical_quality_certified"] = True
    b.atomic_json(output/"summary.json", changed)
    # Even coordinated artifact hash updates cannot bypass semantic recomputation.
    marker = b._read(output/"completion.json"); marker["artifacts"] = [b._reference(output/name) for name in ("config.json", "summary.json", "grouping.private.json")]
    b.atomic_json(output/"completion.json", marker)
    with pytest.raises(ValueError, match="RESULT_CHANGED"): b.verify_result(config, repo=REPO)


def test_cpu_guard_precedes_native_validation(auxiliary, monkeypatch):
    _, config, path, calls = auxiliary
    count = len(calls); monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="FRESH_CPU_PROCESS"): b.run_job(path, config["success_receipt"])
    assert len(calls) == count
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ""); monkeypatch.setitem(sys.modules, "torch", object())
    with pytest.raises(ValueError, match="FRESH_CPU_PROCESS"): b.run_job(path, config["success_receipt"])


def test_native_proof_rejects_unregistered_or_failed_aggregate_before_scoring(tmp_path, monkeypatch):
    path = tmp_path/"aggregate.json"
    b.atomic_json(path, {"kind":"aggregate", "job_id":b.dev.AGGREGATE_ID, "state_path":"synthetic"})
    monkeypatch.setattr(b.dev.caps, "_active", lambda _:({"jobs":[{"job_id":b.dev.AGGREGATE_ID}]}, {"jobs":{b.dev.AGGREGATE_ID:{"status":"FAILED"}}}))
    with pytest.raises(ValueError, match="AGGREGATE_SUCCESS_REQUIRED"): b._native_inputs(path)


def test_cli_hides_cuda_in_its_own_child_environment(tmp_path):
    # Exercise the same entrypoint used by a Supervisor job, without a model or dataset.
    script = """import os
from pathlib import Path
from mediflow_medvlm import campaign_bootstrap as b
os.environ.pop('CUDA_VISIBLE_DEVICES', None)
def checked(config, receipt):
    b._cpu()
    return {'status':'SUCCEEDED', 'scope':b.SCOPE}
b.run_job=checked
assert b.main(['--config','synthetic','--receipt','synthetic']) == 0
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=REPO, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_multidigit_group_sizes_and_saved_config_have_stable_json_hashes(auxiliary, monkeypatch):
    root, original_config, _, _ = auxiliary
    source = b._native_inputs
    def shaped(path):
        aggregate, proof, selected, scored = source(path)
        for axis in ("text", "vqa"):
            selected[axis] = [(row(axis+str(i), groups={"document":"small" if i<7 else "large",
                "image":"small" if i<7 else "large"}), "s") for i in range(17)]
        return aggregate, proof, selected, scored
    monkeypatch.setattr(b, "_native_inputs", shaped)
    config = b.prepare_config(original_config["data"]["aggregate_config"]["path"], root/"auxiliary/bootstrap/multidigit", repo=REPO)
    for axis in ("text", "vqa"):
        report = config["data"]["grouping"][axis]
        assert report["group_size_counts"] == {"7":1, "10":1}
        assert b.canonical_hash(report) == b.canonical_hash(json.loads(json.dumps(report)))
    saved = root/"multidigit-config.json"; b.atomic_json(saved, config)
    loaded = b._read(saved)
    assert loaded["data_hash"] == b.canonical_hash(loaded["data"])
    assert loaded["config_hash"] == b.canonical_hash({k:v for k,v in loaded.items() if k!="config_hash"})
    assert b.validate_config(loaded, repo=REPO) == loaded
