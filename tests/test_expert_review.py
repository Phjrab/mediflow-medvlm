"""Review packets remain local, fixed, blinded and unscored until human review."""
import csv
import json
import stat

import pytest

from mediflow_medvlm.evaluation.expert_review import REVIEW_FIELDS, prepare_expert_review
from mediflow_medvlm.evaluation.runner import freeze_protocol
from mediflow_medvlm.runtime import canonical_hash


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def fixture(tmp_path, count=20, split="test"):
    rows = [{"sample_id": "medquad-" + str(i), "source_id": "medquad", "task_type": "medical_qa",
             "language": "en", "split": split, "input": {"question": "=fixture question " + str(i), "images": []},
             "supervision": {"answer": "reference fixture"}, "group_ids": {"document": str(i)}} for i in range(count)]
    data = tmp_path / "dataset.jsonl"; write_rows(data, rows)
    protocol_path = tmp_path / "protocol.json"; protocol = freeze_protocol([data], protocol_path, split=split, cap=0)
    paths = []
    for role in ("baseline", "tuned"):
        folder = tmp_path / role; folder.mkdir()
        run = {"run_id": role, "pair_id": "fixed-pair", "identity_hash": role + "-identity",
               "protocol_hash": protocol["protocol_hash"], "model_id": "fixture/model", "model_revision": "fixed",
               "config_hash": "config", "precision": "none", "generation": {"do_sample": False, "budgets": protocol["max_new_tokens"]}}
        (folder / "run_manifest.json").write_text(json.dumps(run))
        predictions = [{"sample_id": row["sample_id"], "input_hash": canonical_hash(row["input"]),
                        "run_id": role, "pair_id": "fixed-pair", "model_identity": run["identity_hash"],
                        "status": "answered", "attempted": True, "parsed_answer": role + " fixture answer"} for row in rows]
        path = folder / "predictions.jsonl"; write_rows(path, predictions)
        paths.append(path)
    return protocol_path, paths, rows


def test_packet_fixed_blinded_local_and_unscored(tmp_path):
    protocol, paths, rows = fixture(tmp_path)
    result = prepare_expert_review(protocol, *paths, tmp_path / "packet")
    assert result["status"] == "PREPARED_NOT_RUN" and result["selected_count"] == 16
    assert result["human_review_count"] == 0 and result["clinical_scores"] is None
    assert result["review_scores_generated"] is False
    with (tmp_path / "packet/review.csv").open() as stream:
        table = list(csv.DictReader(stream))
    key = json.loads((tmp_path / "packet/private_key.json").read_text())["items"]
    expected = sorted(rows, key=lambda r: canonical_hash([20261023, r["sample_id"]]))[:16]
    assert [item["sample_id"] for item in key] == [row["sample_id"] for row in expected]
    assert all(row[field] == "" for row in table for field in REVIEW_FIELDS)
    assert "sample_id" not in table[0] and "run_id" not in table[0]
    assert all(row["question"].startswith("'=fixture") for row in table)
    assert {row["A_role"] for row in key} == {"baseline", "tuned"}
    prepare_expert_review(protocol, *paths, tmp_path / "packet-copy")
    assert (tmp_path / "packet/review.csv").read_bytes() == (tmp_path / "packet-copy/review.csv").read_bytes()
    assert stat.S_IMODE((tmp_path / "packet").stat().st_mode) == 0o700
    assert all(stat.S_IMODE(file.stat().st_mode) == 0o600 for file in (tmp_path / "packet").iterdir())
    with pytest.raises(ValueError, match="overwrite"):
        prepare_expert_review(protocol, *paths, tmp_path / "packet")


def test_missing_and_failed_answers_do_not_change_selection(tmp_path):
    protocol, paths, rows = fixture(tmp_path, count=3)
    failures = [json.loads(line) for line in paths[1].read_text().splitlines()]
    failures[0].update(status="oom", parsed_answer=None)
    write_rows(paths[1], failures[:1])
    prepare_expert_review(protocol, *paths, tmp_path / "packet", cap=3)
    with (tmp_path / "packet/review.csv").open() as stream:
        table = list(csv.DictReader(stream))
    assert len(table) == 3
    statuses = [row[field] for row in table for field in ("status_A", "status_B")]
    assert statuses.count("oom") == 1 and statuses.count("not_attempted") == 2
    assert sum(row[field] == "[NO MODEL ANSWER]" for row in table for field in ("answer_A", "answer_B")) == 3


@pytest.mark.parametrize("git_marker", ["directory", "file"])
def test_raw_review_cannot_be_written_inside_git_worktree(tmp_path, git_marker):
    protocol, paths, _ = fixture(tmp_path, count=1)
    repo = tmp_path / "repo"; repo.mkdir()
    if git_marker == "directory":
        (repo / ".git").mkdir(); (repo / ".git/HEAD").write_text("ref: refs/heads/main")
    else: (repo / ".git").write_text("gitdir: elsewhere")
    with pytest.raises(ValueError, match="outside Git"):
        prepare_expert_review(protocol, *paths, repo / "ignored-data/packet")
    assert not (repo / "ignored-data").exists()


def test_training_rows_and_mismatched_pair_rejected(tmp_path):
    protocol, paths, _ = fixture(tmp_path, count=1, split="train")
    with pytest.raises(ValueError, match="validation or test"):
        prepare_expert_review(protocol, *paths, tmp_path / "packet")
    other = tmp_path / "other"; other.mkdir()
    protocol, paths, _ = fixture(other, count=1)
    run_path = paths[1].parent / "run_manifest.json"
    run = json.loads(run_path.read_text()); run["pair_id"] = "other-pair"; run_path.write_text(json.dumps(run))
    with pytest.raises(ValueError):
        prepare_expert_review(protocol, *paths, tmp_path / "packet")


@pytest.mark.parametrize("cap", [0, 17, True])
def test_expert_review_cap_is_finite(tmp_path, cap):
    with pytest.raises(ValueError, match="cap"):
        prepare_expert_review("absent", "absent", "absent", tmp_path / "packet", cap=cap)
