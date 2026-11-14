"""Bounded native-path registration; all durable mutations use temporary roots."""
import copy
import importlib.util
import os
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

import pytest

from mediflow_medvlm import campaign_supervisor as campaign
from mediflow_medvlm import campaign_v6


GATE = "v6-preflight_smoke"
CELLS = [cell[0] for cell in campaign_v6.CELLS]
CHILDREN = [f"v6-native-{kind}-{cell}" for kind in ("preflight", "smoke") for cell in CELLS]


@pytest.fixture
def refinement(tmp_path):
    snapshot = tmp_path / "source"
    package = snapshot / "src/mediflow_medvlm"
    package.mkdir(parents=True)
    source_package = Path(campaign.__file__).parent
    for name in ("campaign_v6.py", "campaign_supervisor.py", "runtime.py", "__init__.py", "domain.py", "validation.py"):
        shutil.copyfile(source_package / name, package / name)
    (snapshot / "fixture.py").write_text("# Synthetic executable; tests never launch it.\n")
    supervisor = campaign.Supervisor(tmp_path / "private")
    original = campaign_v6.build_plan(supervisor.data_root, supervisor.root, repo=snapshot, python=sys.executable)
    supervisor.initialize(original)
    revised = copy.deepcopy(original)
    sources = {name: campaign.file_hash(snapshot / name) for name in original["jobs"][0]["source_files"]}
    sources["fixture.py"] = campaign.file_hash(snapshot / "fixture.py")
    children = []
    for kind in ("preflight", "smoke"):
        for cell in campaign_v6.CELLS:
            cell_id, model_id, task, family = cell
            job_id = f"v6-native-{kind}-{cell_id}"
            smoke = kind == "smoke"
            prepare_id = "prepare-" + {"V2": "T3", "V3": "T4"}.get(cell_id, cell_id)
            children.append({"job_id": job_id, "phase": "native_" + kind, "cell_id": cell_id,
                "run_id": job_id, "model_id": model_id, "task": task, "family": family,
                "dependencies": [f"v6-native-preflight-{cell_id}" if smoke else prepare_id],
                "uses_gpu": smoke, "requires_gpu": smoke, "implementation_pending": smoke,
                "argv": [] if smoke else [sys.executable, str(snapshot / "fixture.py"), "--cell", cell_id],
                "cwd": str(snapshot), "source_files": sources, "source_hash": campaign.canonical_hash(sources),
                "config_hash": campaign.canonical_hash({"cell": cell_id, "kind": kind}),
                "model_hash": campaign.canonical_hash({"model_id": model_id}),
                "data_hash": "synthetic-no-data", "success_receipt": str(supervisor.root / "receipts" / (job_id + ".json"))})
    position = next(n for n, job in enumerate(revised["jobs"]) if job["job_id"] == GATE)
    revised["jobs"][position]["dependencies"] += CHILDREN
    revised["jobs"][position:position] = children
    return supervisor, original, revised


def job(manifest, job_id):
    return next(row for row in manifest["jobs"] if row["job_id"] == job_id)


def assert_rejected_without_writes(supervisor, proposal, match=None):
    before = supervisor.state_path.read_bytes()
    versions = {path: path.read_bytes() for path in (supervisor.root / "manifests").iterdir()}
    with patch.object(campaign, "atomic_json", side_effect=AssertionError("rejection must not write")):
        with pytest.raises((ValueError, OSError), match=match):
            supervisor.refine_preflight(proposal)
    assert supervisor.state_path.read_bytes() == before
    assert {path: path.read_bytes() for path in (supervisor.root / "manifests").iterdir()} == versions


def test_refine_versions_before_pointer_and_preserves_all_old_state(refinement):
    supervisor, original, revised = refinement
    before = campaign.read_json(supervisor.state_path)
    before["jobs"]["prepare-T1"].update(status="SUCCEEDED", attempts=3, failure_count=2,
        recovery_attempts=2, history=[{"status": "FAILED", "attempt": 1}, {"status": "SUCCEEDED", "attempt": 3}],
        verification={"artifact_count": 1}, recovery_authorization_hash="preserved-analysis")
    campaign.atomic_json(supervisor.state_path, before)
    old_version = Path(before["manifest_path"])
    old_bytes = old_version.read_bytes()
    writes = []
    atomic = campaign.atomic_json

    def observe(path, value):
        if Path(path) == supervisor.state_path:
            assert campaign.read_json(value["manifest_path"]) == revised
            assert campaign.read_json(path) == before
        writes.append(Path(path))
        atomic(path, value)

    with patch.object(campaign, "atomic_json", side_effect=observe), patch.object(
            campaign.subprocess, "Popen", side_effect=AssertionError("registration must not spawn")):
        result = supervisor.refine_preflight(revised)
    assert supervisor.load()[0] == revised
    assert campaign.read_json(supervisor.manifest_path) == original
    assert old_version.read_bytes() == old_bytes
    assert writes == [Path(result["manifest_path"]), supervisor.state_path]
    for job_id, record in before["jobs"].items():
        assert result["jobs"][job_id] == record
    for job_id in CHILDREN:
        assert result["jobs"][job_id] == {"status": "IMPLEMENTATION_PENDING" if "-smoke-" in job_id else "PENDING",
            "attempts": 0, "failure_count": 0, "recovery_attempts": 0, "history": []}
    assert result["jobs"][GATE]["status"] == "IMPLEMENTATION_PENDING"
    assert not result["all_terminal"] and not result["ready_to_stop_scheduler"]
    audit = result["manifest_revisions"][-1]
    assert audit["kind"] == "V6_PREFLIGHT_REFINEMENT"
    assert audit["previous_hash"] == before["manifest_hash"]
    assert audit["previous_manifest_path"] == before["manifest_path"]
    assert audit["manifest_hash"] == campaign.canonical_hash(revised)
    assert audit["manifest_path"] == result["manifest_path"]
    assert audit["added_jobs"] == CHILDREN
    assert audit["original_gate_hash"] == campaign.canonical_hash(job(original, GATE))
    assert audit["revised_gate_hash"] == campaign.canonical_hash(job(revised, GATE))
    assert audit["child_job_hashes"] == {key: campaign.canonical_hash(job(revised, key)) for key in CHILDREN}
    for key in set(before) - {"jobs", "manifest_hash", "manifest_path", "manifest_revisions", "updated_at"}:
        assert result[key] == before[key]


@pytest.mark.parametrize("change", ["missing-cpu", "missing-smoke", "extra", "old-reorder", "child-reorder", "child-location", "absent-gate", "old-remove", "metadata", "config", "argv", "source", "gate-policy", "gate-identity", "gate-deps", "cycle"])
def test_refine_rejects_unbounded_or_changed_old_plan(refinement, change):
    supervisor, _, revised = refinement
    if change in {"missing-cpu", "missing-smoke", "absent-gate", "old-remove"}:
        remove = {"missing-cpu": CHILDREN[0], "missing-smoke": CHILDREN[10], "absent-gate": GATE, "old-remove": "v7-report"}[change]
        revised["jobs"] = [row for row in revised["jobs"] if row["job_id"] != remove]
    elif change == "extra":
        revised["jobs"].append({**job(revised, CHILDREN[-1]), "job_id": "v6-native-smoke-V5", "run_id": "extra"})
    elif change == "old-reorder":
        revised["jobs"][0], revised["jobs"][1] = revised["jobs"][1], revised["jobs"][0]
    elif change == "child-reorder":
        revised["jobs"][8], revised["jobs"][9] = revised["jobs"][9], revised["jobs"][8]
    elif change == "child-location":
        revised["jobs"].append(revised["jobs"].pop(8))
    elif change == "metadata":
        revised["recipe"]["passes"] = 2
    elif change in {"config", "argv", "source"}:
        key = {"config": "config_hash", "argv": "argv", "source": "cwd"}[change]
        job(revised, "prepare-T1")[key] = {"config": "changed", "argv": [sys.executable, "-m", "mediflow_medvlm.campaign_v6", "changed"], "source": "/tmp/changed"}[change]
    elif change == "gate-policy":
        job(revised, GATE)["dependency_policy"] = "success"
    elif change == "gate-identity":
        job(revised, GATE)["run_id"] = "changed"
    elif change == "gate-deps":
        job(revised, GATE)["dependencies"].reverse()
    else:
        job(revised, CHILDREN[0])["dependencies"] = [GATE]
    assert_rejected_without_writes(supervisor, revised)


@pytest.mark.parametrize("field,value", [
    ("model_id", "Qwen/undeclared"), ("task", "vqa"), ("family", "gemma"), ("cell_id", "ALL_V6"),
    ("run_id", "v6-native-preflight-T2"), ("phase", "first_seed_final"), ("uses_gpu", True),
    ("uses_gpu", 0), ("requires_gpu", True), ("implementation_pending", True),
    ("dependencies", ["prepare-T2"]), ("dependency_policy", "terminal"), ("extra_seed", 20261115),
    ("argv", []), ("source_hash", "changed"),
])
def test_refine_rejects_wrong_native_cpu_identity(refinement, field, value):
    supervisor, _, revised = refinement
    job(revised, CHILDREN[0])[field] = value
    assert_rejected_without_writes(supervisor, revised)


@pytest.mark.parametrize("field,value", [
    ("implementation_pending", False), ("uses_gpu", False), ("requires_gpu", False),
    ("dependencies", ["prepare-T1"]), ("task", "vqa"), ("model_id", "Qwen/other"),
    ("argv", [sys.executable, "-m", "mediflow_medvlm.campaign_v6"]),
])
def test_refine_keeps_gpu_smoke_unimplemented(refinement, field, value):
    supervisor, _, revised = refinement
    job(revised, "v6-native-smoke-T1")[field] = value
    assert_rejected_without_writes(supervisor, revised)


@pytest.mark.parametrize("cell,wrong", [("V2", "prepare-V1"), ("V3", "prepare-T3"), ("T5", "prepare-T6")])
def test_native_paths_require_their_own_prepared_base(refinement, cell, wrong):
    supervisor, _, revised = refinement
    job(revised, "v6-native-preflight-" + cell)["dependencies"] = [wrong]
    assert_rejected_without_writes(supervisor, revised)


@pytest.mark.parametrize("job_id", [GATE, "v6-common_train_freeze", "v7-P0", "v7-report"])
@pytest.mark.parametrize("marker,value", [("attempts", 1), ("failure_count", 1), ("recovery_attempts", 1),
    ("history", [{"attempt": 1}]), ("launch_token", "old-launch"), ("attempt_dir", "/tmp/old-attempt"),
    ("status", "RUNNING"), ("status", "SUCCEEDED")])
def test_refine_refuses_any_started_gate_or_descendant(refinement, job_id, marker, value):
    supervisor, _, revised = refinement
    state = campaign.read_json(supervisor.state_path)
    state["jobs"][job_id][marker] = value
    campaign.atomic_json(supervisor.state_path, state)
    assert_rejected_without_writes(supervisor, revised)


@pytest.mark.parametrize("active", ["status", "identity", "child-heartbeat", "lease"])
def test_refine_refuses_active_workers_without_spawning(refinement, active):
    supervisor, _, revised = refinement
    state = campaign.read_json(supervisor.state_path)
    record = state["jobs"]["prepare-T1"]
    if active == "status":
        record["status"] = "STARTING"
    elif active == "identity":
        record["worker_identity"] = campaign.process_identity()
    elif active == "child-heartbeat":
        record["attempt_dir"] = str(supervisor.root / "synthetic-attempt")
        campaign.atomic_json(Path(record["attempt_dir"]) / "heartbeat.json", {"child_identity": campaign.process_identity()})
    campaign.atomic_json(supervisor.state_path, state)
    if active == "lease":
        with campaign.try_lock(supervisor.root / "worker.lock"):
            assert_rejected_without_writes(supervisor, revised, "worker lease")
    else:
        assert_rejected_without_writes(supervisor, revised, "worker active")


def test_refine_serializes_with_existing_supervisor_lock(refinement):
    supervisor, _, revised = refinement
    before = supervisor.state_path.read_bytes()
    with campaign.try_lock(supervisor.root / "supervisor.lock"):
        assert supervisor.refine_preflight(revised) == {"status": "SUPERVISOR_BUSY"}
    assert supervisor.state_path.read_bytes() == before


def test_refine_rejects_legacy_owner(refinement):
    supervisor, _, revised = refinement
    state = campaign.read_json(supervisor.state_path)
    state["queue_owner"] = "legacy"
    campaign.atomic_json(supervisor.state_path, state)
    assert_rejected_without_writes(supervisor, revised, "supervisor ownership")


@pytest.mark.parametrize("source", ["fixture.py", "src/mediflow_medvlm/campaign_v6.py"])
def test_refine_rechecks_new_and_preserved_old_source_bytes(refinement, source):
    supervisor, _, revised = refinement
    (Path(job(revised, CHILDREN[0])["cwd"]) / source).write_text("# tampered after candidate construction\n")
    assert_rejected_without_writes(supervisor, revised, "source snapshot changed")


def test_refine_idempotent_even_after_progress_but_rechecks_source(refinement):
    supervisor, _, revised = refinement
    result = supervisor.refine_preflight(revised)
    state = campaign.read_json(supervisor.state_path)
    state["jobs"][CHILDREN[0]].update(status="SUCCEEDED", attempts=1)
    campaign.atomic_json(supervisor.state_path, state)
    before = supervisor.state_path.read_bytes()
    version = Path(result["manifest_path"])
    mtime = version.stat().st_mtime_ns
    assert supervisor.refine_preflight(revised)["jobs"][CHILDREN[0]]["attempts"] == 1
    assert supervisor.state_path.read_bytes() == before
    assert version.stat().st_mtime_ns == mtime
    assert len(supervisor.status()["manifest_revisions"]) == 1
    source = Path(job(revised, CHILDREN[0])["cwd"]) / "fixture.py"
    source.write_text("# tampered after refinement\n")
    assert_rejected_without_writes(supervisor, revised, "source snapshot changed")


def test_refine_crash_before_pointer_update_is_retryable(refinement):
    supervisor, _, revised = refinement
    before = supervisor.state_path.read_bytes()
    atomic = campaign.atomic_json

    def crash(path, value):
        if Path(path) == supervisor.state_path:
            assert campaign.read_json(value["manifest_path"]) == revised
            raise OSError("synthetic crash before pointer commit")
        atomic(path, value)

    with patch.object(campaign, "atomic_json", side_effect=crash):
        with pytest.raises(OSError, match="synthetic crash"):
            supervisor.refine_preflight(revised)
    assert supervisor.state_path.read_bytes() == before
    version = supervisor.root / "manifests" / (campaign.canonical_hash(revised) + ".json")
    assert campaign.read_json(version) == revised
    mtime = version.stat().st_mtime_ns
    result = campaign.Supervisor(supervisor.data_root).refine_preflight(revised)
    assert len(result["manifest_revisions"]) == 1
    assert version.stat().st_mtime_ns == mtime


def test_refine_rejects_corrupted_existing_version(refinement):
    supervisor, _, revised = refinement
    version = supervisor.root / "manifests" / (campaign.canonical_hash(revised) + ".json")
    campaign.atomic_json(version, {"corrupt": True})
    assert_rejected_without_writes(supervisor, revised, "immutable preflight manifest")


@pytest.mark.parametrize("reader", ["current", "deployed-v1"])
def test_failed_preparation_does_not_block_independent_cpu_path(refinement, reader):
    supervisor, _, revised = refinement
    supervisor.refine_preflight(revised)
    coordinator = campaign
    if reader == "deployed-v1":
        source = os.environ.get("MEDVLM_COMPAT_SUPERVISOR_SOURCE")
        if not source:
            pytest.skip("set MEDVLM_COMPAT_SUPERVISOR_SOURCE to verify an actual deployed v1 coordinator")
        spec = importlib.util.spec_from_file_location("deployed_v1_supervisor", source)
        coordinator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(coordinator)
        assert not hasattr(coordinator.Supervisor, "refine_preflight")
    coordinator.validate_manifest(revised)
    assert coordinator.Supervisor(supervisor.data_root).load()[0] == revised
    state = campaign.read_json(supervisor.state_path)
    for row in revised["jobs"]:
        if row["phase"] == "model_preparation":
            state["jobs"][row["job_id"]].update(status="FAILED" if row["job_id"] == "prepare-T1" else "SUCCEEDED")
    campaign.atomic_json(supervisor.state_path, state)
    # No actual process, GPU lease, model or data access occurs in this test.
    with patch.object(coordinator.subprocess, "Popen") as popen:
        popen.return_value.pid = 999999999
        result = coordinator.Supervisor(supervisor.data_root).tick()
    assert result["jobs"][CHILDREN[0]]["status"] == "BLOCKED_PREREQUISITE"
    assert result["jobs"][CHILDREN[1]]["attempts"] == 1
    assert popen.call_count == 1
    launch_path = Path(result["jobs"][CHILDREN[1]]["attempt_dir"]) / "launch.json"
    assert campaign.read_json(launch_path)["job"]["job_id"] == CHILDREN[1]
    assert result["jobs"][GATE]["status"] == "IMPLEMENTATION_PENDING"
    assert result["jobs"]["v6-native-smoke-T1"]["status"] == "IMPLEMENTATION_PENDING"
    assert job(supervisor.load()[0], GATE)["dependency_policy"] == "terminal"


def test_refine_cli_registers_only_and_keeps_original_manifest(refinement, capsys, tmp_path):
    supervisor, original, revised = refinement
    candidate = tmp_path / "revised.json"
    campaign.atomic_json(candidate, revised)
    with patch.object(campaign.subprocess, "Popen", side_effect=AssertionError("CLI must not spawn")):
        assert campaign.main(["--data-root", str(supervisor.data_root), "refine-preflight", "--manifest", str(candidate)]) == 0
    assert '"V6_PREFLIGHT_REFINEMENT"' in capsys.readouterr().out
    assert campaign.read_json(supervisor.manifest_path) == original
    assert supervisor.status()["jobs"][GATE]["attempts"] == 0


def test_original_gate_still_uses_terminal_dependency_policy_when_implemented(refinement):
    supervisor, _, revised = refinement
    supervisor.refine_preflight(revised)
    implemented = copy.deepcopy(revised)
    cpu = job(revised, CHILDREN[0])
    execution_fields = {"argv", "cwd", "source_files", *campaign.HASH_FIELDS, "success_receipt", "uses_gpu"}
    job(implemented, GATE).update({key: cpu[key] for key in execution_fields}, implementation_pending=False)
    # Activate only the barrier; this is the existing bounded activation API.
    supervisor.activate(implemented)
    state = campaign.read_json(supervisor.state_path)
    for row in implemented["jobs"]:
        if row["job_id"] in CHILDREN or row["phase"] == "model_preparation":
            state["jobs"][row["job_id"]].update(status="FAILED", attempts=3, failure_count=3, recovery_attempts=2)
    campaign.atomic_json(supervisor.state_path, state)
    with patch.object(campaign.subprocess, "Popen") as popen:
        popen.return_value.pid = 999999999
        result = supervisor.tick()
    assert result["jobs"][GATE]["attempts"] == 1
    assert result["jobs"][GATE]["status"] == "RUNNING"
    assert popen.call_count == 1
    assert result["jobs"]["v6-common_train_freeze"]["attempts"] == 0
