"""First-seed registration and aggregate contracts; all state is temporary."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from mediflow_medvlm import campaign_first_seed as first
from mediflow_medvlm import campaign_supervisor as supervisor_module
from mediflow_medvlm import campaign_v6


def write(path, value):
    supervisor_module.atomic_json(path, value)
    return Path(path)


def job(manifest, key):
    return next(j for j in manifest["jobs"] if j["job_id"] == key)


@pytest.fixture
def planned(tmp_path, monkeypatch):
    from mediflow_medvlm import campaign_main_train as main
    package = Path(first.__file__).parent
    old, new = tmp_path / "archived", tmp_path / "candidate-source"
    for root in (old, new):
        shutil.copytree(package, root / "src/mediflow_medvlm", ignore=shutil.ignore_patterns("__pycache__"))
    supervisor = supervisor_module.Supervisor(tmp_path / "private")
    manifest = campaign_v6.build_plan(supervisor.data_root, supervisor.root, repo=old, python=sys.executable)
    manifest["jobs"].insert(0, {"job_id": "historical-doctor", "run_id": "historical-doctor",
        "phase": "doctor", "cell_id": "ENGINEERING", "dependencies": [], "implementation_pending": True})
    position = next(i for i, j in enumerate(manifest["jobs"]) if j["job_id"] == "v6-preflight_smoke")
    placeholders = [{"job_id": f"v6-native-{kind}-{cell[0]}", "run_id": f"v6-native-{kind}-{cell[0]}",
        "phase": "native_" + kind, **campaign_v6.cell_spec(cell[0]), "dependencies": [],
        "implementation_pending": True} for kind in ("preflight", "smoke") for cell in campaign_v6.CELLS]
    manifest["jobs"][position:position] = placeholders
    common_job = job(manifest, first.freeze.JOB_ID)
    files, digest = first.freeze.source_identity(old)
    freeze_config = {"schema_version": first.freeze.SCHEMA, "scope": first.freeze.SCOPE,
        "job_id": first.freeze.JOB_ID, "run_id": first.freeze.JOB_ID,
        "output_dir": str(supervisor.root / "common_train/freeze"),
        "source_files": files, "source_hash": digest, "data_hash": "frozen-data", "model_hash": "frozen-models"}
    freeze_config["config_hash"] = first.canonical_hash(freeze_config)
    freeze_path = write(supervisor.root / "configs/common.json", freeze_config)
    freeze_receipt = supervisor.root / "receipts/common.json"
    common_job.update(implementation_pending=False, uses_gpu=False, gpu_lock_mode="wrapper",
        argv=[sys.executable, "-m", "mediflow_medvlm.campaign_common_train_freeze", "--config", str(freeze_path),
            "--receipt", str(freeze_receipt)], cwd=str(old), source_files=files,
        **{k: freeze_config[k] for k in first.HASH_FIELDS}, success_receipt=str(freeze_receipt))
    evidence = write(supervisor.root / "common_train/freeze/proof.json", {"engineering_fixture": True})
    receipt = {"status": "SUCCEEDED", **{k: freeze_config[k] for k in first.IDENTITY_KEYS},
        "artifacts": [{"path": str(evidence), "sha256": first.file_hash(evidence)}]}
    write(freeze_receipt, receipt)
    supervisor.initialize(manifest)
    state = supervisor_module.read_json(supervisor.state_path)
    gate_position = next(i for i, j in enumerate(manifest["jobs"]) if j["job_id"] == first.JOB_ID)
    for row in manifest["jobs"][:gate_position]:
        state["jobs"][row["job_id"]].update(status="SUCCEEDED", attempts=1, exit_code=0)
    state["jobs"][first.freeze.JOB_ID]["verification"] = supervisor_module.verify_success(common_job)
    state["jobs"]["prepare-T1"].update(attempts=3, failure_count=2, recovery_attempts=2,
        history=[{"status": "FAILED", "attempt": 1}, {"status": "SUCCEEDED", "attempt": 3}],
        recovery_authorization_hash="historical-analysis")
    write(supervisor.state_path, state)
    # Freeze/worker result internals have their own tests. This fixture isolates
    # supervisor registration, immutable config bindings and aggregate decisions.
    monkeypatch.setattr(first.freeze, "_completed", lambda _: supervisor_module.read_json(freeze_receipt))

    def fake_build(frozen, cell, output, *, repo=None, success_receipt=None, job_id=None):
        child_files, source = main.source_identity(repo or new)
        record = supervisor_module.read_json(supervisor.state_path)["jobs"][first.freeze.JOB_ID]
        data = {"common_freeze": {"path": str(frozen), "sha256": first.file_hash(frozen),
            "job_sha256": first.canonical_hash(common_job), "terminal_state_sha256": first.canonical_hash(record)}}
        value = {"schema_version": main.SCHEMA, "scope": main.SCOPE,
            **campaign_v6.cell_spec(cell), "job_id": job_id, "run_id": job_id,
            "output_dir": str(output), "success_receipt": str(success_receipt), "recipe": {"seed": 20261023, "passes": 1},
            "data": data, "model": {"model_id": campaign_v6.cell_spec(cell)["model_id"], "adapter_initialization": None},
            "source_files": child_files, "source_hash": source, "data_hash": first.canonical_hash(data),
            "model_hash": first.canonical_hash(campaign_v6.cell_spec(cell))}
        value["config_hash"] = first.canonical_hash(value)
        return value

    def validate(config, *, repo=None):
        expected = fake_build(config["data"]["common_freeze"]["path"], config["cell_id"], config["output_dir"],
            repo=repo, success_receipt=config["success_receipt"], job_id=config["job_id"])
        if not first._equal(expected, config):
            raise ValueError("FAKE_FROZEN_CONFIG_MISMATCH")
        return [], []

    def verify(config, *, repo=None):
        validate(config, repo=repo)
        result = supervisor_module.read_json(config["success_receipt"])
        if result.get("full_pass") is not True or result.get("fresh_reload") is not True:
            raise ValueError("MISSING_MAIN_PROOF")
        return result

    def validate_identity(config, *, repo=None):
        assert config["config_hash"] == first.canonical_hash({k: v for k, v in config.items() if k != "config_hash"})
        files, digest = main.source_identity(repo or new)
        if config["source_files"] != files or config["source_hash"] != digest:
            raise ValueError("STATIC_SOURCE_MISMATCH")
        return config

    fake = SimpleNamespace(SOURCE_FILES=main.SOURCE_FILES, build_config=fake_build,
        validate_config=validate, validate_config_identity=validate_identity, verify_result=verify)
    monkeypatch.setattr(first, "_main", lambda: fake)
    # The recovery helper imports the public validation API directly.
    monkeypatch.setattr(main, "validate_config", validate)
    monkeypatch.setattr(main, "validate_config_identity", validate_identity)
    real_source = first.source_identity
    monkeypatch.setattr(first, "source_identity", lambda repo=None: real_source(repo or new))
    result = first.build_refinement(state["manifest_path"], supervisor.state_path,
        supervisor.root / "first_seed/configs", supervisor.root / "first_seed/runs", repo=new)
    for path, config in result["configs"].items():
        write(path, config)
    gate_config = first.build_config(result["manifest"], supervisor.state_path,
        supervisor.root / "first_seed/aggregate", repo=new)
    config_path = write(supervisor.root / "first_seed/gate-config.json", gate_config)
    revised = first.activation_manifest(config_path, result["manifest"], repo=new)
    return SimpleNamespace(supervisor=supervisor, original=manifest, revised=revised,
        gate_config=gate_config, gate_path=config_path, new=new, old=old, fake=fake)


def reject_without_writes(p, proposal=None, match=None):
    supervisor = p.supervisor
    state = supervisor.state_path.read_bytes()
    versions = {f: f.read_bytes() for f in (supervisor.root / "manifests").iterdir()}
    with patch.object(supervisor_module, "atomic_json", side_effect=AssertionError("rejection wrote state")):
        with pytest.raises((ValueError, OSError, KeyError), match=match):
            supervisor.refine_first_seed(proposal or p.revised)
    assert supervisor.state_path.read_bytes() == state
    assert {f: f.read_bytes() for f in versions} == versions


def test_atomic_registration_preserves_42_jobs_states_history_and_gate_metadata(planned):
    p, writes = planned, []
    before = supervisor_module.read_json(p.supervisor.state_path)
    original_bytes = p.supervisor.manifest_path.read_bytes()
    atomic = supervisor_module.atomic_json

    def observe(path, value):
        if Path(path) == p.supervisor.state_path:
            assert supervisor_module.read_json(value["manifest_path"]) == p.revised
            assert supervisor_module.read_json(path) == before
        writes.append(Path(path))
        atomic(path, value)

    with patch.object(supervisor_module, "atomic_json", side_effect=observe), patch.object(
            supervisor_module.subprocess, "Popen", side_effect=AssertionError("registration spawned")):
        result = p.supervisor.refine_first_seed(p.revised)
    assert len(p.supervisor.load()[0]["jobs"]) == 53
    assert writes == [Path(result["manifest_path"]), p.supervisor.state_path]
    assert p.supervisor.manifest_path.read_bytes() == original_bytes
    for key, record in before["jobs"].items():
        assert result["jobs"][key] == ({**record, "status": "PENDING"} if key == first.JOB_ID else record)
        if key != first.JOB_ID:
            assert job(p.original, key) == job(p.revised, key)
    for key in first.CHILD_IDS:
        assert result["jobs"][key] == {"status": "PENDING", "attempts": 0,
            "failure_count": 0, "recovery_attempts": 0, "history": []}
    assert job(p.revised, first.JOB_ID)["dependencies"] == first.CHILD_IDS
    assert job(p.revised, first.JOB_ID)["dependency_policy"] == "terminal"
    assert result["manifest_revisions"][-1]["added_jobs"] == first.CHILD_IDS


@pytest.mark.parametrize("change", ["metadata", "old-job", "missing", "extra", "order", "location",
    "gate-dependencies", "gate-phase", "gate-reason", "gate-gpu", "pending-gate", "child-config"])
def test_refinement_rejects_plan_tamper(planned, change):
    p = planned
    candidate = copy.deepcopy(p.revised)
    if change == "metadata": candidate["recipe"]["passes"] = 2
    elif change == "old-job": job(candidate, "prepare-T1")["config_hash"] = "changed"
    elif change == "missing": candidate["jobs"].remove(job(candidate, first.CHILD_IDS[-1]))
    elif change == "extra": candidate["jobs"].append({**job(candidate, first.CHILD_IDS[0]), "job_id": "extra", "run_id": "extra"})
    elif change == "order": candidate["jobs"][0], candidate["jobs"][1] = candidate["jobs"][1], candidate["jobs"][0]
    elif change == "location": candidate["jobs"].append(candidate["jobs"].pop(30))
    elif change == "gate-dependencies": job(candidate, first.JOB_ID)["dependencies"] += [first.freeze.JOB_ID]
    elif change == "gate-phase": job(candidate, first.JOB_ID)["phase"] = "other"
    elif change == "gate-reason": job(candidate, first.JOB_ID)["reason"] = "rewritten"
    elif change == "gate-gpu": job(candidate, first.JOB_ID)["uses_gpu"] = True
    elif change == "pending-gate": job(candidate, first.JOB_ID)["implementation_pending"] = True
    else: job(candidate, first.CHILD_IDS[0])["config_hash"] = "other"
    reject_without_writes(p, candidate)


@pytest.mark.parametrize("key,value", [("cell_id", "T2"), ("model_id", "other"), ("family", "other"),
    ("task", "vqa"), ("seed", 20261114), ("seed", True), ("phase", "native_smoke"),
    ("run_id", "other"), ("dependencies", ["v6-first-seed-T2"]), ("dependency_policy", "terminal"),
    ("uses_gpu", False), ("requires_gpu", False), ("gpu_lock_mode", "child"),
    ("implementation_pending", True), ("extra_seed", 7)])
def test_refinement_rejects_child_identity(planned, key, value):
    candidate = copy.deepcopy(planned.revised)
    job(candidate, first.CHILD_IDS[0])[key] = value
    reject_without_writes(planned, candidate)


@pytest.mark.parametrize("marker,value", [("attempts", 1), ("failure_count", 1), ("recovery_attempts", 1),
    ("attempts", False), ("history", [{"attempt": 1}]), ("launch_token", "old"),
    ("verification", {}), ("status", "SUCCEEDED")])
@pytest.mark.parametrize("target", [first.JOB_ID, "v7-P0"])
def test_refinement_rejects_started_gate_descendants(planned, marker, value, target):
    state = supervisor_module.read_json(planned.supervisor.state_path)
    state["jobs"][target][marker] = value
    write(planned.supervisor.state_path, state)
    reject_without_writes(planned)


@pytest.mark.parametrize("active", ["status", "identity", "child-heartbeat", "worker-lease", "gpu-lease", "owner"])
def test_refinement_checks_all_idle_guards(planned, active):
    p, s = planned, planned.supervisor
    state = supervisor_module.read_json(s.state_path)
    record = state["jobs"]["prepare-T1"]
    if active == "status": record["status"] = "RUNNING"
    elif active == "identity": record["worker_identity"] = supervisor_module.process_identity()
    elif active == "child-heartbeat":
        record["attempt_dir"] = str(s.root / "fake-attempt")
        write(Path(record["attempt_dir"]) / "heartbeat.json", {"child_identity": supervisor_module.process_identity()})
    elif active == "owner": state["queue_owner"] = "legacy"
    write(s.state_path, state)
    if active.endswith("lease"):
        path = s.root / "worker.lock" if active == "worker-lease" else s.data_root / "runtime/medvlm_gpu.lock"
        with supervisor_module.try_lock(path):
            reject_without_writes(p)
    else:
        reject_without_writes(p)


def test_refinement_supervisor_lock_and_idempotent_after_progress(planned):
    p = planned
    with supervisor_module.try_lock(p.supervisor.root / "supervisor.lock"):
        assert p.supervisor.refine_first_seed(p.revised) == {"status": "SUPERVISOR_BUSY"}
    p.supervisor.refine_first_seed(p.revised)
    state = supervisor_module.read_json(p.supervisor.state_path)
    state["jobs"][first.CHILD_IDS[0]].update(status="FAILED", attempts=3, failure_count=3, recovery_attempts=2)
    write(p.supervisor.state_path, state)
    before = p.supervisor.state_path.read_bytes()
    p.supervisor.refine_first_seed(p.revised)
    assert p.supervisor.state_path.read_bytes() == before
    assert len(p.supervisor.status()["manifest_revisions"]) == 1
    cfg = Path(job(p.revised, first.CHILD_IDS[0])["argv"][4])
    cfg.write_text(cfg.read_text() + " ")
    reject_without_writes(p, match="bytes changed|CONFIG_CHANGED|REGISTERED_MAIN_CONFIG_CHANGED")


@pytest.mark.parametrize("orphan", [first.CHILD_IDS[0], "unknown-orphan", "missing-old"])
def test_refinement_never_overwrites_orphan_state_or_counters(planned, orphan):
    state = supervisor_module.read_json(planned.supervisor.state_path)
    if orphan == "missing-old":
        del state["jobs"]["prepare-T1"]
    else:
        state["jobs"][orphan] = {"status": "FAILED", "attempts": 3, "failure_count": 3,
            "recovery_attempts": 2, "history": [{"status": "FAILED"}]}
    write(planned.supervisor.state_path, state)
    reject_without_writes(planned, match="state/job keys")


def test_refinement_crash_before_pointer_is_retryable(planned):
    s = planned.supervisor
    original = s.state_path.read_bytes()
    atomic = supervisor_module.atomic_json
    def interrupt(path, value):
        if Path(path) == s.state_path:
            raise OSError("synthetic before pointer")
        atomic(path, value)
    with patch.object(supervisor_module, "atomic_json", side_effect=interrupt), pytest.raises(OSError):
        s.refine_first_seed(planned.revised)
    assert s.state_path.read_bytes() == original
    assert s.refine_first_seed(planned.revised)["manifest_hash"] == first.canonical_hash(planned.revised)


def terminalize(p, *, failed=None):
    p.supervisor.refine_first_seed(p.revised)
    state = supervisor_module.read_json(p.supervisor.state_path)
    for key in first.CHILD_IDS:
        row = job(p.revised, key)
        record = state["jobs"][key]
        if key == failed:
            record.update(status="FAILED", attempts=3, failure_count=3, recovery_attempts=2,
                error="synthetic terminal failure", error_signature="retained")
            continue
        proof = write(p.supervisor.root / "first_seed/proofs" / (key + ".json"), {"fixture": True})
        receipt = {"status": "SUCCEEDED", "full_pass": True, "fresh_reload": True,
            **{k: row[k] for k in first.IDENTITY_KEYS},
            "artifacts": [{"path": str(proof), "sha256": first.file_hash(proof)}]}
        write(row["success_receipt"], receipt)
        record.update(status="SUCCEEDED", attempts=1, exit_code=0, verification=supervisor_module.verify_success(row))
    write(p.supervisor.state_path, state)
    return state


@pytest.mark.parametrize("reader", ["current", "deployed-v1"])
def test_terminal_failure_does_not_block_next_cell_with_old_coordinator(planned, reader):
    p = planned
    p.supervisor.refine_first_seed(p.revised)
    state = supervisor_module.read_json(p.supervisor.state_path)
    state["jobs"][first.CHILD_IDS[0]].update(status="FAILED", attempts=3, failure_count=3, recovery_attempts=2)
    write(p.supervisor.state_path, state)
    # Original wrapper implementation remains capable of the new ordinary jobs.
    coordinator = supervisor_module
    if reader == "deployed-v1":
        source_path = os.environ.get("MEDVLM_COMPAT_SUPERVISOR_SOURCE")
        if source_path:
            source = Path(source_path).read_bytes()
        else:
            try:
                archived = subprocess.run(["git", "show", "1602f35:src/mediflow_medvlm/campaign_supervisor.py"],
                    capture_output=True)
            except FileNotFoundError:
                pytest.skip("historical coordinator requires Git object or MEDVLM_COMPAT_SUPERVISOR_SOURCE")
            if archived.returncode:
                pytest.skip("historical coordinator requires Git object or MEDVLM_COMPAT_SUPERVISOR_SOURCE")
            source = archived.stdout
        import hashlib
        assert hashlib.sha256(source).hexdigest() == "1e04250bf2ea48934031232813417d1c6e193d57ce2f7cb50fbd359d8adaaa6b"
        module_path = p.supervisor.root / "old-coordinator.py"
        module_path.write_bytes(source)
        spec = importlib.util.spec_from_file_location("original_first_seed_coordinator", module_path)
        coordinator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(coordinator)
        assert not hasattr(coordinator.Supervisor, "refine_first_seed")
    with patch.object(coordinator.subprocess, "Popen") as popen:
        popen.return_value.pid = 999999999
        result = coordinator.Supervisor(p.supervisor.data_root).tick()
    assert result["jobs"][first.CHILD_IDS[0]]["attempts"] == 3
    assert result["jobs"][first.CHILD_IDS[1]]["status"] == "RUNNING"
    assert result["jobs"][first.CHILD_IDS[1]]["attempts"] == 1
    assert result["jobs"][first.JOB_ID]["attempts"] == 0
    assert popen.call_count == 1


@pytest.mark.parametrize("failed", [None, first.CHILD_IDS[0]])
def test_aggregate_truthful_atomic_result_and_receipt_gap_reuse(planned, failed, monkeypatch):
    p = planned
    terminalize(p, failed=failed)
    for key in first.IDENTITY_KEYS:
        monkeypatch.setenv("MEDVLM_SUPERVISOR_" + key.upper(), p.gate_config[key])
    before = p.supervisor.state_path.read_bytes()
    result = first.run_gate(p.gate_path, p.gate_config["success_receipt"])
    assert result["status"] == "SUCCEEDED"
    assert result["all_cells_succeeded"] is (failed is None)
    summary = first._read(Path(p.gate_config["output_dir"]) / "summary.json")
    assert summary["first_seed_stage_terminal"] is True
    assert summary["v6_main_training_completed"] is (failed is None)
    assert len(summary["cells"]) == 10
    if failed:
        assert summary["cells"][0]["status"] == "FAILED"
        assert summary["cells"][0]["final_checkpoint_verified"] is False
    assert p.supervisor.state_path.read_bytes() == before
    receipt_path = Path(p.gate_config["success_receipt"])
    receipt_stat = receipt_path.stat()
    with patch.object(first, "atomic_json", side_effect=AssertionError("completed reuse must not rewrite")):
        assert first.run_gate(p.gate_path, receipt_path) == result
    assert receipt_path.stat().st_ino == receipt_stat.st_ino
    assert receipt_path.stat().st_mtime_ns == receipt_stat.st_mtime_ns
    Path(p.gate_config["success_receipt"]).unlink()
    assert first.run_gate(p.gate_path, p.gate_config["success_receipt"]) == result


def test_aggregate_rejects_unfinished_child_and_incomplete_success_proof(planned):
    p = planned
    p.supervisor.refine_first_seed(p.revised)
    with pytest.raises(ValueError, match="NOT_TERMINAL"):
        first.run_gate(p.gate_path, p.gate_config["success_receipt"])
    terminalize(p)
    row = job(p.revised, first.CHILD_IDS[0])
    receipt = first._read(row["success_receipt"])
    receipt["fresh_reload"] = False
    write(row["success_receipt"], receipt)
    state = first._read(p.supervisor.state_path)
    state["jobs"][first.CHILD_IDS[0]]["verification"] = supervisor_module.verify_success(row)
    write(p.supervisor.state_path, state)
    with pytest.raises(ValueError, match="MISSING_MAIN_PROOF"):
        first.run_gate(p.gate_path, p.gate_config["success_receipt"])


def test_failed_model_cache_unavailable_does_not_prevent_truthful_terminal_audit(planned, monkeypatch):
    p = planned
    terminalize(p, failed=first.CHILD_IDS[0])
    # Full planning/execution validation would fail on the failed model's cache.
    def unavailable(config, *, repo=None):
        if config["cell_id"] == "T1":
            raise OSError("model cache genuinely inaccessible")
        return config
    monkeypatch.setattr(p.fake, "validate_config", unavailable)
    result = first.run_gate(p.gate_path, p.gate_config["success_receipt"])
    assert result["all_cells_succeeded"] is False
    assert first._read(Path(p.gate_config["output_dir"]) / "summary.json")["cells"][0]["status"] == "FAILED"


def test_resealed_summary_cannot_claim_all_failed_cells_succeeded(planned):
    p = planned
    terminalize(p, failed=first.CHILD_IDS[0])
    first.run_gate(p.gate_path, p.gate_config["success_receipt"])
    output = Path(p.gate_config["output_dir"])
    summary = first._read(output / "summary.json")
    summary["all_cells_succeeded"] = summary["v6_main_training_completed"] = True
    write(output / "summary.json", summary)
    marker = first._read(output / "completion.json")
    marker["artifacts"] = [{"path": str(output / name), "sha256": first.file_hash(output / name)}
        for name in sorted(first.OUTPUT_FILES)]
    marker["all_cells_succeeded"] = True
    write(output / "completion.json", marker)
    with pytest.raises(ValueError, match="AUDIT_CHANGED"):
        first.run_gate(p.gate_path, p.gate_config["success_receipt"])


@pytest.mark.parametrize("kind", ["cell", "seed", "recipe", "data", "model", "order"])
def test_main_source_rebind_rejects_scientific_changes(planned, tmp_path, kind):
    p = planned
    before = job(p.revised, first.CHILD_IDS[0])
    after = copy.deepcopy(before)
    snapshot = tmp_path / "replacement"
    shutil.copytree(p.new, snapshot)
    file = snapshot / "src/mediflow_medvlm/campaign_main_train.py"
    file.write_text(file.read_text() + "\n# independently audited fixture fix\n")
    config = first._read(before["argv"][4])
    files = {name: first.file_hash(snapshot / name) for name in config["source_files"]}
    config.update(source_files=files, source_hash=first.canonical_hash(files))
    if kind == "cell": config["cell_id"] = "T2"
    elif kind == "seed": config["recipe"]["seed"] += 1
    elif kind == "recipe": config["recipe"]["passes"] = 2
    elif kind == "data": config["data"]["changed_data"] = True
    elif kind == "model": config["model"]["adapter_initialization"] = "smoke"
    else: config["data"]["order"] = "changed"
    config["config_hash"] = first.canonical_hash({k: v for k, v in config.items() if k != "config_hash"})
    path = write(p.supervisor.root / "replacement.json", config)
    after.update(cwd=str(snapshot), source_files=files, source_hash=config["source_hash"], config_hash=config["config_hash"])
    after["argv"][4] = str(path)
    with pytest.raises(ValueError):
        supervisor_module.verify_gpu_source_rebind(before, after)


def test_main_source_rebind_accepts_only_new_source_and_new_config(planned, tmp_path):
    p = planned
    before = job(p.revised, first.CHILD_IDS[0])
    after = copy.deepcopy(before)
    snapshot = tmp_path / "replacement"
    shutil.copytree(p.new, snapshot)
    file = snapshot / "src/mediflow_medvlm/campaign_main_train.py"
    file.write_text(file.read_text() + "\n# independently audited fixture fix\n")
    config = first._read(before["argv"][4])
    files = {name: first.file_hash(snapshot / name) for name in config["source_files"]}
    config.update(source_files=files, source_hash=first.canonical_hash(files))
    config["config_hash"] = first.canonical_hash({k: v for k, v in config.items() if k != "config_hash"})
    path = write(p.supervisor.root / "replacement.json", config)
    after.update(cwd=str(snapshot), source_files=files, source_hash=config["source_hash"], config_hash=config["config_hash"])
    after["argv"][4] = str(path)
    proof = supervisor_module.verify_gpu_source_rebind(before, after)
    assert proof["original_config_path"] != proof["replacement_config_path"]
    assert proof["replacement_config_sha256"] == first.file_hash(path)


def test_refine_cli_never_ticks_or_spawns(planned, capsys):
    path = write(planned.supervisor.root / "candidate.json", planned.revised)
    with patch.object(supervisor_module.subprocess, "Popen", side_effect=AssertionError("spawned")):
        assert supervisor_module.main(["--data-root", str(planned.supervisor.data_root),
            "refine-first-seed", "--manifest", str(path)]) == 0
    assert "V6_FIRST_SEED_REFINEMENT" in capsys.readouterr().out


def test_installed_layout_source_identity_and_cold_import_are_cpu_only(tmp_path):
    package = Path(first.__file__).parent
    installed = tmp_path / "site-packages"
    shutil.copytree(package, installed / "mediflow_medvlm", ignore=shutil.ignore_patterns("__pycache__"))
    script = ("import sys,json; from mediflow_medvlm import campaign_first_seed as f; "
        "a=f.source_identity(); assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules; "
        "assert 'campaign_first_seed.py' in [k.rsplit('/',1)[-1] for k in a[0]]; print(json.dumps(a))")
    run = subprocess.run([sys.executable, "-c", script], cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(installed), "CUDA_VISIBLE_DEVICES": ""}, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    files, digest = json.loads(run.stdout)
    assert all(k.startswith("src/mediflow_medvlm/") for k in files)
    assert first.canonical_hash(files) == digest
    assert all(first.file_hash(installed / k.removeprefix("src/")) == sha for k, sha in files.items())
