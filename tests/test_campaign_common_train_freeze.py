"""CPU-only cohort, lineage, order and atomic-output contracts."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import subprocess

import pytest

from mediflow_medvlm import campaign_common_train_freeze as c


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True))
    return {"path": str(path), "sha256": c.file_hash(path)}


def row(index, source="aihub_71875", split="train"):
    return {"sample_id": f"{source}/{split}/{index}", "source_id": source, "split": split,
        "language": "en", "task_type": "medical_qa", "input": {"question": f"{source} {split} question {index}"},
        "supervision": {"answer": f"answer {index}"}, "group_ids": {"document": f"{source}/{split}/doc/{index}"},
        "metadata": {"q_type": "treatment"}, "provenance": {}}


def splitfile(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Deliberately use a noncanonical row encoding: subset must retain these bytes.
    path.write_bytes(b"".join((json.dumps(r, ensure_ascii=False, indent=None) + "\n").encode() for r in rows))
    return {"path": str(path), "snapshot_path": str(path), "sha256": c.file_hash(path), "count": len(rows),
            "ordered_sample_ids": [r["sample_id"] for r in rows]}


def freeze(path, splits, *, v2=False, **extra):
    value = {"status": "SUCCEEDED", "schema_version": "medvlm.freeze.v1", "evidence_level": "TEST",
             "splits": splits, **extra}
    value["freeze_id"] = c.digest(value if v2 else {k: v for k, v in value.items() if k not in {"status", "evidence_level"}})
    return write(path, value)


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    audit = {"cells": [], "evidence": {}}
    original, payload = {}, {}
    for axis, size in (("text", 35), ("vqa", 27)):
        rows = [row(i, "aihub_71875" if axis == "text" else "slake") for i in range(size)]
        parent = splitfile(tmp_path / (axis + ".jsonl"), rows)
        original[axis] = rows
        payload[axis] = Path(parent["path"]).read_bytes()
        cells = [cell for cell in c.gate.CELL_IDS if c.gate.cell_spec(cell)["task"] == axis]
        for n, cell in enumerate(cells):
            audit["cells"].append({**c.gate.cell_spec(cell), "status": "SUPPORTED"})
            config = {**c.gate.cell_spec(cell), "model": {"model_id": c.gate.cell_spec(cell)["model_id"], "revision": "pin"},
                "max_length": 2048, "fixed_template_date": "2026-11-14", "data": {"train_file": parent["path"],
                "train_sha256": parent["sha256"], "expected_count": size,
                "sample_order_sha256": c.canonical_hash(parent["ordered_sample_ids"])}}
            cfgref = write(tmp_path / cell / "config.json", config)
            records = []
            for i, r in enumerate(rows):
                eligible = not (axis == "text" and i == n)
                record = {"sample_id": r["sample_id"], "position": i, **c.gate.cpu._stratum(r),
                    "input_sha256": c.canonical_hash(r["input"]), "target_sha256": c.canonical_hash(r["supervision"]),
                    "eligible": eligible, "native": {"sequence_tokens": i + 10, "supervised_tokens_after_shift": i + 1}}
                if not eligible:
                    record["reason"] = "CONTEXT_OVERFLOW"
                records.append(record)
            ep = tmp_path / cell / "eligibility.jsonl"
            ep.write_text("".join(json.dumps(r) + "\n" for r in records))
            audit["evidence"]["v6-native-preflight-" + cell] = {"config": cfgref,
                "eligibility": {"path": str(ep), "sha256": c.file_hash(ep)}}
            audit["evidence"]["v6-native-smoke-" + cell] = {"config": {"path": "bound-smoke", "sha256": "bound"}}
    monkeypatch.setattr(c, "_aggregate", lambda _: copy.deepcopy(audit))
    monkeypatch.setattr(c, "_protection", lambda refs, parents: ({"bound": True}, []))
    return audit, original, payload


def test_intersection_bytes_order_strata_targets_and_recipe(cohort):
    audit, rows, raw = cohort
    plan, payloads = c._plan({}, {})
    text, vqa = plan["axes"]["text"], plan["axes"]["vqa"]
    assert (text["common_count"], vqa["common_count"]) == (29, 27)
    assert payloads["text"] == b"".join(raw["text"].splitlines(keepends=True)[6:])
    assert payloads["vqa"] == raw["vqa"]
    assert text["sample_ids"] == [r["sample_id"] for r in rows["text"]][6:]
    assert text["per_model"]["T1"]["common_target_tokens"] == sum(range(7, 36))
    assert sum(r["count"] for r in text["common_strata"]) == 29
    assert len(text["excluded"]) == 6
    assert text["excluded"][0]["blocking_cells"] == {"T1": "CONTEXT_OVERFLOW"}
    assert text["training_orders"][str(c.SEEDS[0])]["final_update_examples"] == 13
    assert vqa["training_orders"][str(c.SEEDS[0])]["final_update_examples"] == 11
    assert plan["recipe"]["betas"] == [0.9, 0.999]
    assert plan["recipe"]["partial_update_divisor"] == "actual_example_count"
    assert all(p["adapter_initialization"] is None and not p["smoke_adapter_reuse"]
               and not p["historical_training_reused"] for p in plan["per_cell_run_plan"].values())
    assert not plan["v6_main_training_completed"] and not plan["eval_answers_used_for_selection"]
    cold = subprocess.run([sys.executable, "-c",
        "import sys; from mediflow_medvlm import campaign_common_train_freeze as c; "
        "c._orders([\"one\"]); assert \"torch\" not in sys.modules; "
        "assert \"transformers\" not in sys.modules"], capture_output=True, text=True)
    assert cold.returncode == 0, cold.stderr


@pytest.mark.parametrize("size", [1, 15, 16, 17, 29, 27])
def test_three_seed_exact_permutations_and_actual_batch_divisors(size):
    ids = [f"id{i}" for i in range(size)]
    orders = c._orders(ids)
    assert set(orders) == set(map(str, c.SEEDS))
    for seed in c.SEEDS:
        value = orders[str(seed)]
        expected = sorted(range(size), key=lambda i: (c.canonical_hash([seed, ids[i]]), ids[i]))
        assert value["indices"] == expected
        assert sorted(value["indices"]) == list(range(size))
        assert value["indices_sha256"] == c.canonical_hash(expected)
        assert value["sample_order_sha256"] == c.canonical_hash([ids[i] for i in expected])
        assert sum(value["update_example_counts"]) == size
        assert value["micro_loss_divisor_per_update"] == value["update_example_counts"]
        assert value["requires_family_selection_before_execution"] == (seed != c.SEEDS[0])


def test_failed_cell_cannot_change_frozen_cohort_after_plan(cohort):
    audit, _, _ = cohort
    before, _ = c._plan({}, {})
    audit["cells"][0]["status"] = "FAILED"
    after, _ = c._plan({}, {})
    assert after["supported_cells"] != before["supported_cells"]
    assert after["freeze_id"] != before["freeze_id"]
    assert before["later_failure_may_widen_cohort"] is False
    assert after["unsupported_cells"][0]["status"] == "FAILED"


@pytest.mark.parametrize("kind", ["input", "target", "order", "source", "file"])
def test_reject_changed_parent_or_eligibility(cohort, kind):
    audit, _, _ = cohort
    evidence = audit["evidence"]["v6-native-preflight-T1"]
    path = Path(evidence["eligibility"]["path"])
    records = [json.loads(l) for l in path.read_text().splitlines()]
    if kind == "file":
        path.write_bytes(path.read_bytes().replace(b"question", b"QUESTION", 1) + b" ")
    elif kind in {"input", "target"}:
        records[0][kind + "_sha256"] = "bad"
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
        evidence["eligibility"]["sha256"] = c.file_hash(path)
    elif kind == "order":
        records.reverse()
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
        evidence["eligibility"]["sha256"] = c.file_hash(path)
    else:
        config = c._read(evidence["config"]["path"])
        config["data"]["train_sha256"] = "bad"
        evidence["config"] = write(Path(evidence["config"]["path"]), config)
    with pytest.raises(c.gate.GateError):
        c._plan({}, {})


@pytest.mark.parametrize("conflict", ["sample", "group", "input"])
def test_protected_conflict_fails_without_extra_exclusions(cohort, monkeypatch, conflict):
    _, rows, _ = cohort
    r = rows["text"][10]
    protected = c._projection(r)
    if conflict != "sample":
        protected["sample_id"] = "heldout"
    if conflict != "group":
        protected["groups"] = []
    if conflict != "input":
        protected["input_digest"] = "other"
    monkeypatch.setattr(c, "_protection", lambda *_: ({}, [protected]))
    with pytest.raises(c.gate.GateError, match="COMMON_TRAIN_PROTECTED_OVERLAP"):
        c._plan({}, {})


@pytest.fixture
def lineage(tmp_path):
    refs, splits = {}, {}
    for source in ("kormedmcqa", "medquad", "pubmedqa_labeled", "slake", "text", "vqa_rad"):
        for split in ("train", "validation", "test"):
            count = (1059 if split == "validation" else 3009) if source == "kormedmcqa" and split != "train" else 0
            if source != "kormedmcqa" and split != "train":
                count = 1
            splits[source + "/" + split] = splitfile(tmp_path / source / (split + ".jsonl"), [row(i, source, split) for i in range(count)])
    refs["global_audit"] = write(tmp_path / "audit.json", {"status": "SUCCEEDED", "critical_error_count": 0})
    refs["text_ancestor_freeze"] = freeze(tmp_path / "old.json", {k:v for k,v in splits.items() if not k.startswith("vqa_rad/")})
    refs["protected_freeze"] = freeze(tmp_path / "protected.json", splits, audit_sha256=refs["global_audit"]["sha256"])
    refs["source_audit"] = write(tmp_path / "source.json", {"protected_freeze_sha256": refs["text_ancestor_freeze"]["sha256"],
        "protected_files": {s: splits["kormedmcqa/" + s] for s in ("validation", "test")}})
    protected_ids = [sid for s in ("validation", "test") for sid in splits["kormedmcqa/" + s]["ordered_sample_ids"]]
    refs["semantic_policy"] = write(tmp_path / "policy.json", {"protected_count": 4068, "sample_ids": protected_ids})
    refs["semantic_encoding"] = write(tmp_path / "encoding.json", {"rows": 4068, "truncated_rows": 0,
        "all_input_tokens_covered": True, "policy_sha256": refs["semantic_policy"]["sha256"]})
    refs["semantic_audit"] = write(tmp_path / "semantic.json", {"status": "SUCCEEDED", "protected_count": 4068,
        "all_input_tokens_covered": True, "semantic_equivalence_verified": False, "translation_equivalence_verified": False,
        "policy_path": refs["semantic_policy"]["path"], "policy_sha256": refs["semantic_policy"]["sha256"],
        "encoding_path": refs["semantic_encoding"]["path"], "encoding_sha256": refs["semantic_encoding"]["sha256"],
        "candidate_pairs": [{"sample_id": "preserved-test-candidate"}]})
    refs["internal_audit"] = write(tmp_path / "internal.json", {"protected_count": 2,
        "protected_rows_modified": False, "confirmed_overlaps": [], "near_candidates": []})
    refs["selection"] = write(tmp_path / "selection.json", {"corpus_selection_used_test_performance": False})
    mix = {"mix_v2/" + s: splitfile(tmp_path / "mix" / (s + ".jsonl"),
        [row(0, "aihub_71875", s)] if s != "test" else [{**row(0, "aihub_71875", s), "sample_id": "preserved-test-candidate"}])
        for s in ("train", "validation", "test")}
    refs["text_freeze"] = freeze(tmp_path / "mix.json", mix, v2=True, provenance={
        "source_audit_sha256": refs["source_audit"]["sha256"], "v1_freeze_sha256": refs["text_ancestor_freeze"]["sha256"],
        "selection_manifest_sha256": refs["selection"]["sha256"], "semantic_audit_sha256": refs["semantic_audit"]["sha256"],
        "internal_split_audit_sha256": refs["internal_audit"]["sha256"], "test_used_for_selection": False})
    hold = {"aihub/" + s: splitfile(tmp_path / "aihub" / (s + ".jsonl"), [row(50, "aihub_71875", s)]) for s in ("validation", "test")}
    href = freeze(tmp_path / "holdout.json", hold, v2=True)
    refs["text_manifest"] = write(tmp_path / "text.json", {"status": "SUCCEEDED", "corpora": {"mix": {
        "freeze_manifest": refs["text_freeze"]["path"], "freeze_manifest_sha256": refs["text_freeze"]["sha256"]}},
        "aihub_holdout_freeze": href["path"], **{field: refs[key]["path"] for key,field in
        (("source_audit","source_audit_path"),("selection","selection_manifest_path"),("semantic_audit","semantic_audit_path"),("internal_audit","internal_audit_path"))}})
    refs["text_verification"] = write(tmp_path / "verification.json", {"status": "SUCCEEDED",
        "manifest_sha256": refs["text_manifest"]["sha256"], "checks": {"all": True}})
    image = {"status": "SUCCEEDED", "image_inventory": {}, "near_duplicate_policy": {"clinical_equivalence_verified": False}, "patient_group_verified": False}
    refs["slake_image_audit"] = write(tmp_path / "image.json", image)
    refs["slake_build"] = write(tmp_path / "build.json", {**image, "canonical_files": {s:splits["slake/" + s] for s in ("train", "validation", "test")}})
    return refs, {"text": mix["mix_v2/train"], "vqa": splits["slake/train"]}


def test_full_lineage_historical_hash_forms_and_metadata_only(lineage):
    refs, parents = lineage
    proof, heldout = c._protection(refs, parents)
    assert proof["protected_kormed_count"] == 4068
    assert proof["semantic_test_candidates_preserved"] == 1
    assert not proof["original_ancestor_audit_reproduced"]
    assert all("supervision" not in r for r in heldout)
    assert proof["heldout_unique_count"] == len(heldout)


@pytest.mark.parametrize("key", sorted(c.PROTECTION_KEYS))
def test_every_protection_reference_is_rehashed(lineage, key):
    refs, parents = lineage
    path = Path(refs[key]["path"])
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(c.gate.GateError, match="PROTECTION_REFERENCE_CHANGED"):
        c._protection(refs, parents)


@pytest.mark.parametrize("kind", ["bridge", "semantic_coverage", "audit", "selection", "image", "parent"])
def test_semantically_wrong_resealed_protection_rejected(lineage, kind):
    refs, parents = lineage
    key = {"bridge":"protected_freeze", "semantic_coverage":"semantic_policy", "audit":"global_audit",
           "selection":"selection", "image":"slake_image_audit", "parent":"text_freeze"}[kind]
    obj = c._read(refs[key]["path"])
    if kind == "bridge": obj["splits"]["medquad/test"]["count"] += 1
    if kind == "semantic_coverage": obj["sample_ids"][0] = "bad"
    if kind == "audit": obj["critical_error_count"] = 1
    if kind == "selection": obj["corpus_selection_used_test_performance"] = True
    if kind == "image": obj["image_inventory"] = {"bad":"bad"}
    if kind == "parent": obj["splits"]["mix_v2/train"]["count"] += 1
    refs[key] = write(Path(refs[key]["path"]), obj)
    with pytest.raises(c.gate.GateError): c._protection(refs, parents)


@pytest.fixture
def worker(tmp_path, monkeypatch, cohort):
    plan, payloads = c._plan({}, {})
    config = {"output_dir": str(tmp_path / "out"), "success_receipt": str(tmp_path / "receipt.json"),
        "freeze_id": plan["freeze_id"], **{k:"identity-" + k for k in c.IDENTITY_KEYS}}
    path = tmp_path / "config.json"
    write(path, config)
    monkeypatch.setattr(c, "_validate", lambda cfg: (copy.deepcopy(plan), payloads.copy()))
    return path, config, plan


def test_atomic_private_output_idempotent_and_receipt_gap(worker):
    path, config, plan = worker
    first = c.run_gate(path, Path(config["success_receipt"]))
    output = Path(config["output_dir"])
    hashes = {p.name:c.file_hash(p) for p in output.iterdir()}
    assert set(hashes) == c.OUTPUT_FILES | {"completion.json"}
    assert all(p.stat().st_mode & 0o077 == 0 for p in output.iterdir())
    assert len(first["artifacts"]) == 6
    assert c.run_gate(path, Path(config["success_receipt"])) == first
    Path(config["success_receipt"]).unlink()
    assert c.run_gate(path, Path(config["success_receipt"])) == first
    assert hashes == {p.name:c.file_hash(p) for p in output.iterdir()}


@pytest.mark.parametrize("kind", ["artifact", "partial", "receipt", "symlink"])
def test_partial_or_tampered_output_fails_closed(worker, kind):
    path, config, _ = worker
    receipt = Path(config["success_receipt"])
    c.run_gate(path, receipt)
    output = Path(config["output_dir"])
    if kind == "artifact":
        p = output / "text.train.jsonl"; p.write_bytes(p.read_bytes().replace(b"answer", b"ANSWER", 1))
    elif kind == "partial": (output / "completion.json").unlink()
    elif kind == "receipt": write(receipt, {"fake":True})
    else:
        (output / "text.train.jsonl").unlink(); (output / "text.train.jsonl").symlink_to(output / "vqa.train.jsonl")
    with pytest.raises((c.gate.GateError, FileNotFoundError)): c.run_gate(path, receipt)


def test_changed_config_during_audit_not_published(worker, monkeypatch):
    path, config, plan = worker
    old = c._validate
    calls = []
    def changed(cfg):
        result = old(cfg)
        calls.append(1)
        if len(calls) == 2: path.write_text(path.read_text() + " ")
        return result
    monkeypatch.setattr(c, "_validate", changed)
    with pytest.raises(c.gate.GateError, match="INPUT_CHANGED_DURING_FREEZE"): c.run_gate(path, Path(config["success_receipt"]))
    assert not Path(config["output_dir"]).exists() and not Path(config["success_receipt"]).exists()


def test_list_groups_and_cross_source_image_hashes():
    training = row(1, "slake")
    training["group_ids"] = {"parent": ["first", "second"]}
    training["metadata"]["image_sha256"] = "same-image"
    heldout = row(5, "vqa_rad", "test")
    heldout["metadata"]["image_sha256"] = "same-image"
    assert ("slake", "parent", "first") in c._groups(training)
    assert ("slake", "parent", "second") in c._groups(training)
    assert c._groups(training).intersection(c._groups(heldout)) == {("GLOBAL_IMAGE", "image_sha256", "same-image")}


def test_duplicate_protected_ids_with_conflicting_metadata_rejected(lineage, monkeypatch):
    refs, parents = lineage
    original = c._projection
    def conflict(r):
        value = original(r)
        value["sample_id"] = "forced-same-id"
        return value
    monkeypatch.setattr(c, "_projection", conflict)
    with pytest.raises(c.gate.GateError, match="CONFLICTING_PROTECTED_SAMPLE_METADATA"):
        c._protection(refs, parents)


@pytest.mark.parametrize("artifact", ["freeze.json", "summary.json", "text.train.jsonl", "vqa.train.jsonl"])
def test_resealed_wrong_output_rejected_against_recomputed_plan(worker, artifact):
    path, config, _ = worker
    receipt = Path(config["success_receipt"])
    c.run_gate(path, receipt)
    output = Path(config["output_dir"])
    target = output / artifact
    if artifact.endswith("jsonl"):
        target.write_bytes(target.read_bytes().replace(b"answer", b"ANSWER", 1))
    else:
        value = c._read(target); value["v6_main_training_completed"] = True; write(target, value)
    marker = c._read(output / "completion.json")
    for item in marker["artifacts"]:
        if item["path"] == str(target): item["sha256"] = c.file_hash(target)
    write(output / "completion.json", marker)
    receipt.unlink()
    with pytest.raises(c.gate.GateError, match="EXISTING_OUTPUT_RECOMPUTATION_MISMATCH"):
        c.run_gate(path, receipt)


@pytest.fixture
def declarations(monkeypatch):
    native = {"declaration": {"seeds": c.SEEDS}, "jobs": {"native": {"immutable": True}},
              "states": {"native": {"status": "SUCCEEDED", "attempts": 2}}}
    monkeypatch.setattr(c.gate, "_snapshot", lambda m, s: copy.deepcopy(native))
    common = {"job_id": c.JOB_ID, "phase": "common_train_freeze", "cell_id": "ALL_V6", "run_id": c.JOB_ID,
              "dependencies": [c.gate.JOB_ID], "dependency_policy": "terminal", "status": "PENDING_IMPLEMENTATION", "reason": "historical"}
    manifest = {"seeds": c.SEEDS.copy(), "jobs": [common, {"job_id": c.gate.JOB_ID}]}
    state = {"jobs": {c.gate.JOB_ID: {"status": "SUCCEEDED", "attempts": 1},
                      c.JOB_ID: {"status": "IMPLEMENTATION_PENDING", "attempts": 0, "failure_count": 0, "recovery_attempts": 0}}}
    return manifest, state


def test_snapshot_excludes_own_activation_and_attempt_but_includes_prerequisites(declarations):
    manifest, state = declarations
    before = c._snapshot(manifest, state)
    manifest["jobs"][0]["argv"] = ["new-worker"]
    state["jobs"][c.JOB_ID].update(status="RUNNING", attempts=1)
    assert c._snapshot(manifest, state) == before
    state["jobs"][c.gate.JOB_ID]["attempts"] += 1
    assert c._snapshot(manifest, state) != before


@pytest.mark.parametrize("change", ["seed", "aggregate_failure", "dependency", "phase"])
def test_snapshot_rejects_wrong_frozen_declaration(declarations, change):
    manifest, state = declarations
    if change == "seed": manifest["seeds"].reverse()
    elif change == "aggregate_failure": state["jobs"][c.gate.JOB_ID]["status"] = "FAILED"
    elif change == "dependency": manifest["jobs"][0]["dependencies"] = []
    else: manifest["jobs"][0]["phase"] = "first_seed_final"
    with pytest.raises(c.gate.GateError): c._snapshot(manifest, state)


def test_activation_changes_only_gate_and_allowed_fields(tmp_path, declarations, monkeypatch):
    manifest, state = declarations
    snapshot = c._snapshot(manifest, state)
    statepath = tmp_path / "state.json"; write(statepath, state)
    files, source = c.source_identity()
    config = {"state_path": str(statepath), "prerequisites": snapshot, "source_files": files,
              "source_hash": source, "success_receipt": str(tmp_path / "receipt.json"),
              **{k: "identity-" + k for k in c.HASH_FIELDS if k != "source_hash"}}
    config["config_hash"] = c.canonical_hash({k:v for k,v in config.items() if k != "config_hash"})
    path = tmp_path / "config.json"; write(path, config)
    monkeypatch.setattr(c, "validate_manifest", lambda _: None)
    revised = c.activation_manifest(path, manifest, python="/frozen/python")
    assert revised["jobs"][1] == manifest["jobs"][1]
    assert revised["jobs"][0]["status"] == "PENDING_IMPLEMENTATION"
    assert revised["jobs"][0]["reason"] == "historical"
    assert revised["jobs"][0]["dependencies"] == [c.gate.JOB_ID]
    assert "requires_gpu" not in revised["jobs"][0]
    assert revised["jobs"][0]["argv"][0] == "/frozen/python"


def test_build_config_is_pure_and_binds_exact_plan(tmp_path, declarations, cohort, monkeypatch):
    manifest, state = declarations
    manifest.update(campaign_id="test", campaign_root=str(tmp_path))
    statepath, manifestpath = tmp_path / "state.json", tmp_path / "manifest.json"
    state["manifest_path"] = str(manifestpath)
    write(manifestpath, manifest); write(statepath, state)
    monkeypatch.setattr(c.gate, "_active", lambda _: (copy.deepcopy(manifest), copy.deepcopy(state)))
    before = {p.name:c.file_hash(p) for p in tmp_path.iterdir() if p.is_file()}
    config = c.build_config(manifestpath, statepath, tmp_path / "private", protection={})
    assert config["config_hash"] == c.canonical_hash({k:v for k,v in config.items() if k != "config_hash"})
    assert before == {p.name:c.file_hash(p) for p in tmp_path.iterdir() if p.is_file()}
    assert not (tmp_path / "private").exists()
    state["jobs"][c.JOB_ID]["attempts"] = 1
    with pytest.raises(c.gate.GateError, match="ONLY_UNSTARTED"): c.build_config(manifestpath, statepath, tmp_path / "private", protection={})
