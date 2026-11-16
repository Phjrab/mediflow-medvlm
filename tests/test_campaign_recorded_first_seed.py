"""Recorded aggregate invariants; synthetic artifacts, no GPU or live state writes."""
import copy
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

from mediflow_medvlm import campaign_recorded_first_seed as r
from mediflow_medvlm import campaign_supervisor as s


def write(path, value):
    s.atomic_json(path, value)
    return Path(path)


@pytest.fixture
def planned(tmp_path, monkeypatch):
    source = Path(r.__file__).resolve().parents[2]
    sup = s.Supervisor(tmp_path / "private")
    old_state = {"campaign_id": "old", "jobs": {r.main_train.common.JOB_ID:
        {"status": "SUCCEEDED", "attempts": 1, "exit_code": 0}}}
    old_job = {"job_id": r.main_train.common.JOB_ID, "actual_original_proof": True}
    old_manifest = {"campaign_id": "old", "jobs": [old_job]}
    old_manifest_path = write(tmp_path / "old-manifest.json", old_manifest)
    old_state.update(manifest_path=str(old_manifest_path), manifest_hash=s.canonical_hash(old_manifest))
    old_state_path = write(tmp_path / "old-state.json", old_state)
    freeze_config_path = write(tmp_path / "freeze/config.json", {"fixture": True})
    freeze_path = write(tmp_path / "freeze/receipt.json", {"fixture": True})
    reference = {"path": str(freeze_path), "sha256": s.file_hash(freeze_path),
        "job_sha256": s.canonical_hash(old_job),
        "terminal_state_sha256": s.canonical_hash(old_state["jobs"][r.main_train.common.JOB_ID])}
    monkeypatch.setattr(r.main_train, "_freeze", lambda _: (
        {"state_path": str(old_state_path), "output_dir": str(freeze_config_path.parent)},
        {"freeze_id": "fixed-freeze"}, reference.copy()))

    def validate(config, *, repo=None):
        assert config["config_hash"] == s.canonical_hash({k: v for k, v in config.items() if k != "config_hash"})
        assert config["data_hash"] == s.canonical_hash(config["data"])
        assert config["source_hash"] == s.canonical_hash(config["source_files"])
        assert config["recipe"] == {"passes": 1, "seed": 20261023, "full_target": True}

    monkeypatch.setattr(r.main_train, "validate_config_identity", validate)
    calls = []

    def verify(config, *, repo=None):
        calls.append(config["cell_id"])
        receipt = r._read(config["success_receipt"])
        if not all(receipt.get(k) for k in ("full_pass", "fresh_reload", "native_targets")):
            raise ValueError("MISSING_SEMANTIC_PROOF")
        return receipt

    monkeypatch.setattr(r.main_train, "verify_result", verify)
    files, source_hash = r.main_train.source_identity(source)
    jobs, records = [], {}
    for cell in r.CELLS:
        cell_id = cell[0]
        job_id = "v6-first-seed-" + cell_id
        receipt_path = sup.root / "receipts" / (job_id + ".json")
        config_path = sup.root / "configs" / (cell_id + ".json")
        data = {"common_freeze": reference.copy(), "freeze_id": "fixed-freeze"}
        config = {**r.cell_spec(cell_id), "job_id": job_id, "run_id": job_id,
            "output_dir": str(sup.root / "runs" / cell_id), "success_receipt": str(receipt_path),
            "data": data, "recipe": {"passes": 1, "seed": 20261023, "full_target": True},
            "source_files": files, "source_hash": source_hash, "data_hash": s.canonical_hash(data),
            "model_hash": s.canonical_hash(cell[1])}
        config["config_hash"] = s.canonical_hash(config)
        write(config_path, config)
        job = {**r.cell_spec(cell_id), "job_id": job_id, "run_id": job_id, "phase": "native_main_train",
            "dependencies": [], "dependency_policy": "success", "seed": 20261023,
            "uses_gpu": True, "requires_gpu": True, "implementation_pending": False,
            "gpu_lock_mode": "wrapper", "external_prerequisite": {k: reference[k] for k in ("path", "sha256")},
            "argv": [sys.executable, "-m", "mediflow_medvlm.campaign_main_train", "--config", str(config_path),
                     "--receipt", str(receipt_path)], "cwd": str(source), "success_receipt": str(receipt_path),
            "source_files": files, **{k: config[k] for k in r.HASH_FIELDS}}
        artifact = write(sup.root / "runs" / cell_id / "proof.json", {"test_fixture": True})
        receipt = {"status": "SUCCEEDED", **{k: config[k] for k in r.IDENTITY_KEYS},
            "full_pass": True, "fresh_reload": True, "native_targets": True,
            "artifacts": [{"path": str(artifact), "sha256": s.file_hash(artifact)}]}
        write(receipt_path, receipt)
        jobs.append(job)
        records[job_id] = {"status": "SUCCEEDED", "attempts": 1, "failure_count": 0, "recovery_attempts": 0,
                          "exit_code": 0, "verification": s.verify_success(job)}
    jobs.append({"job_id": r.JOB_ID, "run_id": r.JOB_ID, "phase": "first_seed_final", "cell_id": "ALL_V6",
                 "dependencies": r.CHILD_IDS, "dependency_policy": "terminal", "implementation_pending": True,
                 "reason": "retain recorded audit placeholder"})
    jobs.append({"job_id": "v6-validation_caps", "run_id": "v6-validation_caps", "phase": "validation_caps",
                 "cell_id": "ALL_V6", "dependencies": [r.JOB_ID], "implementation_pending": True})
    manifest = {"campaign_id": "recorded-test", "campaign_root": str(sup.root), "seeds": list(r.SEEDS),
        "cells": [{**r.cell_spec(c[0]), "seed": 20261023} for c in r.CELLS],
        "external_common_freeze": {k: reference[k] for k in ("path", "sha256")}, "jobs": jobs}
    sup.initialize(manifest)
    state = r._read(sup.state_path)
    state["jobs"].update(records)
    write(sup.state_path, state)
    config = r.build_config(sup.state_path, sup.root / "aggregate", repo=source)
    path = write(sup.root / "aggregate-config.json", config)
    return SimpleNamespace(sup=sup, manifest=manifest, config=config, path=path, source=source,
        calls=calls, old_state=old_state, old_state_path=old_state_path, old_manifest=old_manifest,
        reference=reference)


def test_activates_only_unstarted_gate_and_preserves_ten_original_jobs(planned):
    p = planned
    before = r._read(p.sup.state_path)
    revised = r.activation_manifest(p.path, p.manifest, repo=p.source)
    assert revised["jobs"][:10] == p.manifest["jobs"][:10]
    assert revised["jobs"][-1] == p.manifest["jobs"][-1]
    assert revised["jobs"][10]["reason"] == p.manifest["jobs"][10]["reason"]
    activated = p.sup.activate(revised)
    assert activated["jobs"][r.JOB_ID]["status"] == "PENDING"
    assert {k: activated["jobs"][k] for k in r.CHILD_IDS} == {k: before["jobs"][k] for k in r.CHILD_IDS}
    assert r.main_train.common.JOB_ID not in activated["jobs"]


def test_success_receipt_requires_semantic_verification_and_replays_idempotently(planned):
    p = planned
    before = p.sup.state_path.read_bytes()
    out = r.run_gate(p.path, p.config["success_receipt"])
    assert out["all_cells_succeeded"] is True
    assert p.calls == [c[0] for c in r.CELLS]
    assert p.sup.state_path.read_bytes() == before
    first_bytes = Path(p.config["success_receipt"]).read_bytes()
    assert r.run_gate(p.path, p.config["success_receipt"]) == out
    assert Path(p.config["success_receipt"]).read_bytes() == first_bytes
    assert r._read(Path(p.config["output_dir"]) / "summary.json")["dev_test_read"] is False


def test_failure_is_terminal_without_fake_training_success(planned):
    p = planned
    state = r._read(p.sup.state_path)
    state["jobs"][r.CHILD_IDS[-1]].update(status="FAILED", exit_code=1, failure_count=1)
    write(p.sup.state_path, state)
    config = r.build_config(p.sup.state_path, p.config["output_dir"], repo=p.source)
    write(p.path, config)
    result = r.run_gate(p.path, config["success_receipt"])
    assert result["status"] == "SUCCEEDED"  # The terminal audit itself completed.
    assert result["all_cells_succeeded"] is False
    summary = r._read(Path(config["output_dir"]) / "summary.json")
    assert summary["v6_main_training_completed"] is False
    assert summary["cells"][-1]["fresh_process_reload_verified"] is False
    assert p.calls == [c[0] for c in r.CELLS[:-1]]


@pytest.mark.parametrize("kind", ["freeze_job", "freeze_state", "child_freeze_job", "child_receipt", "artifact", "recipe", "nonterminal", "config_bytes"])
def test_rejects_tampering_without_publishing_receipt(planned, kind):
    p = planned
    job = p.manifest["jobs"][0]
    if kind == "freeze_job":
        changed = copy.deepcopy(p.old_manifest)
        changed["jobs"][0]["actual_original_proof"] = False
        path = write(p.old_state_path.parent / "changed-old-manifest.json", changed)
        value = {**p.old_state, "manifest_path": str(path), "manifest_hash": s.canonical_hash(changed)}
        write(p.old_state_path, value)
    elif kind == "freeze_state":
        value = copy.deepcopy(p.old_state)
        value["jobs"][r.main_train.common.JOB_ID]["attempts"] = 2
        write(p.old_state_path, value)
    elif kind in {"child_freeze_job", "recipe", "config_bytes"}:
        path = Path(job["argv"][4])
        value = r._read(path)
        if kind == "child_freeze_job": value["data"]["common_freeze"]["job_sha256"] = "forged"
        elif kind == "recipe": value["recipe"]["full_target"] = False
        else: value["unused"] = "edited"
        write(path, value)
    elif kind == "child_receipt":
        receipt = r._read(job["success_receipt"])
        receipt["model_hash"] = "other"
        write(job["success_receipt"], receipt)
    elif kind == "artifact":
        Path(r._read(job["success_receipt"])["artifacts"][0]["path"]).write_text("edited")
    else:
        state = r._read(p.sup.state_path)
        state["jobs"][r.CHILD_IDS[0]]["status"] = "RUNNING"
        write(p.sup.state_path, state)
    with pytest.raises((ValueError, AssertionError)):
        r.run_gate(p.path, p.config["success_receipt"])
    assert not Path(p.config["success_receipt"]).exists()
    assert not Path(p.config["output_dir"]).exists()


def test_success_status_without_full_reload_proof_is_rejected(planned):
    p = planned
    job = p.manifest["jobs"][0]
    receipt = r._read(job["success_receipt"])
    receipt["fresh_reload"] = False
    write(job["success_receipt"], receipt)
    state = r._read(p.sup.state_path)
    state["jobs"][job["job_id"]]["verification"] = s.verify_success(job)
    write(p.sup.state_path, state)
    config = r.build_config(p.sup.state_path, p.config["output_dir"], repo=p.source)
    write(p.path, config)
    with pytest.raises(ValueError, match="MISSING_SEMANTIC_PROOF"):
        r.run_gate(p.path, config["success_receipt"])
