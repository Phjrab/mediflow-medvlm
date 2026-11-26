"""Integrity checks using generated, non-medical image fixtures only."""
import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from mediflow_medvlm.kiosk.common import load_config, read_jsonl, write_jsonl
from mediflow_medvlm.kiosk.dataset import (REVIEW_COLUMNS, classification_question, export_dataset, load_inventory,
                                        scan, status_data, validate_current_export, write_review_template)
from mediflow_medvlm.runtime import atomic_json, canonical_hash, file_hash
from mediflow_medvlm.training.config import load_frozen_data
from mediflow_medvlm.training.rendering import image_paths


def _write_groups(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["relative_path", "subject_id", "session_id", "parent_id", "split"])
        writer.writeheader()
        writer.writerows(rows)


def fixture_config(tmp_path, sources=("eye",), explicit=True):
    entries = []
    for source_index, source in enumerate(sources):
        directory = tmp_path / "photos" / source
        directory.mkdir(parents=True)
        groups = []
        for number, split in enumerate(("train", "train", "validation", "test")):
            name = f"synthetic_{number}.png"
            Image.new("RGB", (17 + number, 19), (number * 40 + source_index * 10, 20, 100)).save(directory / name)
            groups.append({"relative_path": name, "subject_id": f"{source}-{number}", "split": split if explicit else ""})
        groups_path = tmp_path / (source + "_groups.csv")
        _write_groups(groups_path, groups)
        entries.append({"source_id": source, "image_dir": str(directory),
                        "modality": "external_eye_rgb" if source == "eye" else "scalp_microscope",
                        "groups_csv": str(groups_path)})
    path = tmp_path / "config.json"
    atomic_json(path, {"schema_version": "medvlm.kiosk-terminal.v1", "work_dir": str(tmp_path / "work"),
                      "data_root": str(tmp_path / "data"), "sources": entries,
                      "teachers": {item["source_id"]: {"backend": "predictions", "modality": item["modality"],
                                  "class_names": ["fixture_a", "fixture_b"]} for item in entries}})
    return load_config(path)


def predictions(config, label="fixture_a", confidence=0.99):
    inventory = load_inventory(config)
    signatures = {item["source_id"]: canonical_hash([item["source_id"], "synthetic_teacher"])
                  for item in inventory}
    rows = []
    for row in inventory:
        scores = {"fixture_a": 1 - confidence, "fixture_b": 1 - confidence}
        scores[label] = confidence
        rows.append({"image_id": row["image_id"], "image_sha256": row["sha256"],
                     "source_id": row["source_id"], "predicted_label": label, "scores": scores,
                     "confidence": confidence, "margin": confidence - (1 - confidence), "status": "predicted",
                     "teacher": {"backend": "predictions", "signature": signatures[row["source_id"]],
                                 "class_names": list(scores), "modality": row["modality"]}})
    root = Path(config["work_dir"])
    write_jsonl(root / "predictions.jsonl", rows)
    atomic_json(root / "predictions_meta.json", {"inventory_sha256": file_hash(root / "inventory.jsonl"),
                "predictions_sha256": file_hash(root / "predictions.jsonl"),
                "teacher_config_sha256": canonical_hash(config["teachers"]), "teacher_signatures": signatures})
    return rows


def set_reviews(config, changes):
    write_review_template(config)
    path = Path(config["work_dir"]) / "reviews.csv"
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        if row["image_id"] in changes:
            row.update(changes[row["image_id"]])
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def approve(label="fixture_b"):
    return {"decision": "approved", "label": label, "label_source": "human_reviewed", "reviewer": "fixture_reviewer"}


def test_scan_decode_accounting_and_no_subject_inference(tmp_path, monkeypatch):
    config = fixture_config(tmp_path)
    source = config["sources"][0]
    source["groups_csv"] = None
    image_dir = Path(source["image_dir"])
    (image_dir / "broken.jpg").write_bytes(b"not an image")
    Image.new("RGB", (50, 50), "white").save(image_dir / "oversized_fixture.png")
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1000)
    (image_dir / "notes.txt").write_text("synthetic fixture")
    first = scan(config)
    assert first["decoded_count"] == 4
    assert first["decode_failed_count"] == 2
    assert first["unsupported_file_count"] == 1
    assert first["subject_ids_provided_count"] == 0
    assert first["patient_independence"] == "unverified_missing_subject_ids"
    assert scan(config)["reused"] is True


def test_exact_pixel_duplicates_cannot_cross_splits(tmp_path):
    config = fixture_config(tmp_path)
    source = config["sources"][0]
    directory = Path(source["image_dir"])
    with Image.open(directory / "synthetic_0.png") as image:
        image.save(directory / "duplicate.bmp")
    groups = list(csv.DictReader(Path(source["groups_csv"]).open()))
    groups.append({"relative_path": "duplicate.bmp", "subject_id": "different-id", "split": "validation"})
    _write_groups(source["groups_csv"], groups)
    with pytest.raises(ValueError, match="Conflicting explicit splits"):
        scan(config)
    groups[-1]["split"] = ""
    _write_groups(source["groups_csv"], groups)
    summary = scan(config)
    rows = load_inventory(config)
    pair = [row for row in rows if row["relative_path"] in {"synthetic_0.png", "duplicate.bmp"}]
    assert len({row["group_id"] for row in pair}) == 1
    assert {row["split"] for row in pair} == {"train"}
    assert summary["exact_pixel_duplicate_count"] == 1


def test_subject_ids_group_across_domains(tmp_path):
    config = fixture_config(tmp_path, sources=("eye", "scalp"))
    for source in config["sources"]:
        path = Path(source["groups_csv"])
        groups = list(csv.DictReader(path.open()))
        groups[0]["subject_id"] = "same-person-across-domains"
        _write_groups(path, groups)
    scan(config)
    rows = [row for row in load_inventory(config) if row["relative_path"] == "synthetic_0.png"]
    assert rows[0]["group_id"] == rows[1]["group_id"]
    assert rows[0]["subject_id"] == rows[1]["subject_id"]
    assert "same-person" not in rows[0]["subject_id"]


def test_groups_typo_or_path_escape_rejected(tmp_path):
    config = fixture_config(tmp_path)
    path = Path(config["sources"][0]["groups_csv"])
    path.write_text("relative_path,patient\nsynthetic_0.png,a\n")
    with pytest.raises(ValueError, match="Unknown"):
        scan(config)
    path.write_text("relative_path,subject_id\n../outside.png,a\n")
    with pytest.raises(ValueError, match="safe paths"):
        scan(config)


def test_source_changes_and_new_images_require_new_snapshot(tmp_path):
    config = fixture_config(tmp_path)
    scan(config)
    directory = Path(config["sources"][0]["image_dir"])
    Image.new("RGB", (17, 19), "white").save(directory / "synthetic_0.png")
    with pytest.raises(ValueError, match="changed since scan"):
        load_inventory(config, verify_images=True)
    with pytest.raises(ValueError, match="Source images"):
        scan(config)


def test_symlink_outside_source_not_read(tmp_path):
    config = fixture_config(tmp_path)
    outside = tmp_path / "outside.png"
    Image.new("RGB", (4, 4), "white").save(outside)
    try:
        (Path(config["sources"][0]["image_dir"]) / "linked.png").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable")
    result = scan(config)
    assert result["skipped_path_count"] == 1
    assert result["decoded_count"] == 4


def test_review_file_preserved_and_gold_overrides_teacher(tmp_path):
    config = fixture_config(tmp_path)
    scan(config)
    predictions(config)
    inventory = load_inventory(config)
    updates = {row["image_id"]: approve() for row in inventory if row["split"] in {"validation", "test"}}
    train = next(row for row in inventory if row["split"] == "train")
    updates[train["image_id"]] = approve()
    set_reviews(config, updates)
    before = file_hash(Path(config["work_dir"]) / "reviews.csv")
    assert write_review_template(config)["status"] == "EXISTS_NOT_OVERWRITTEN"
    assert file_hash(Path(config["work_dir"]) / "reviews.csv") == before
    result = export_dataset(config)
    assert result["counts"] == {"train": 2, "validation": 1, "test": 1}
    assert result["validation_truth"] == "verified"
    rows, _ = load_frozen_data(result)
    reviewed = next(row for row in rows["train"] if row["original_id"] == train["image_id"])
    assert reviewed["supervision"]["answer"] == "fixture_b"
    assert reviewed["supervision"]["supervision_kind"] == "human_reviewed"
    assert "fixture_b" not in reviewed["input"]["question"]
    assert reviewed["input"]["question"] == classification_question("external_eye_rgb")
    assert reviewed["restrictions"]["sft_allowed"] is True
    pseudo = next(row for row in rows["train"] if row["supervision"]["supervision_kind"] == "teacher_pseudo")
    assert pseudo["metadata"]["kiosk"]["teacher_margin"] == pytest.approx(0.98)
    assert image_paths(reviewed)[0][0].is_file()
    test = read_jsonl(result["test_file"])[0]
    assert test["restrictions"]["sft_allowed"] is False
    assert export_dataset(config)["dataset_id"] == result["dataset_id"]


def test_unreviewed_holdout_excluded_and_explicit_pseudo_validation(tmp_path):
    config = fixture_config(tmp_path)
    scan(config)
    predictions(config)
    with pytest.raises(ValueError, match="nonempty train and validation"):
        export_dataset(config)
    config["pseudo_labels"]["allow_validation"] = True
    result = export_dataset(config)
    assert result["counts"] == {"train": 2, "validation": 1, "test": 0}
    assert result["validation_truth"] == "teacher_pseudo"
    assert result["test_truth"] == "missing"
    assert not read_jsonl(result["test_file"])
    assert "export_diagnostics" not in status_data(config)


def test_rejected_label_never_falls_back_to_pseudo(tmp_path):
    config = fixture_config(tmp_path)
    scan(config)
    predictions(config)
    inventory = load_inventory(config)
    train = next(row for row in inventory if row["split"] == "train")
    updates = {row["image_id"]: approve() for row in inventory if row["split"] == "validation"}
    updates[train["image_id"]] = {"decision": "rejected"}
    set_reviews(config, updates)
    result = export_dataset(config)
    assert result["counts"]["train"] == 1
    assert result["excluded"]["review_rejected"] == 1
    assert train["image_id"] not in {row["original_id"] for row in read_jsonl(result["train_file"])}


def test_review_only_flow_needs_reviewer_and_source_evidence(tmp_path):
    config = fixture_config(tmp_path)
    scan(config)
    inventory = load_inventory(config)
    updates = {row["image_id"]: approve() for row in inventory}
    updates[inventory[0]["image_id"]]["reviewer"] = ""
    set_reviews(config, updates)
    with pytest.raises(ValueError, match="label and reviewer"):
        export_dataset(config)
    updates[inventory[0]["image_id"]] = {**approve(), "label_source": "source_supported"}
    set_reviews(config, updates)
    with pytest.raises(ValueError, match="evidence"):
        export_dataset(config)
    updates[inventory[0]["image_id"]]["evidence"] = "fixture-only reference"
    set_reviews(config, updates)
    result = export_dataset(config)
    assert result["supervision_counts"] == {"human_reviewed": 3, "source_supported": 1}


def test_prediction_hash_change_or_unknown_id_rejected(tmp_path):
    config = fixture_config(tmp_path)
    scan(config)
    rows = predictions(config)
    root = Path(config["work_dir"])
    rows[0]["image_sha256"] = "0" * 64
    write_jsonl(root / "predictions.jsonl", rows)
    with pytest.raises(ValueError, match="prediction file changed"):
        export_dataset(config)
    meta = json.loads((root / "predictions_meta.json").read_text())
    meta["predictions_sha256"] = file_hash(root / "predictions.jsonl")
    atomic_json(root / "predictions_meta.json", meta)
    with pytest.raises(ValueError, match="identity mismatch"):
        export_dataset(config)


def test_frozen_bytes_cannot_be_overwritten(tmp_path):
    config = fixture_config(tmp_path)
    scan(config)
    inventory = load_inventory(config)
    set_reviews(config, {row["image_id"]: approve() for row in inventory})
    result = export_dataset(config)
    Path(result["train_file"]).write_text("{}\n")
    with pytest.raises(ValueError, match="Frozen dataset file changed"):
        export_dataset(config)


def test_unknown_never_becomes_normal_and_incomplete_review_blocked(tmp_path):
    config = fixture_config(tmp_path)
    scan(config)
    rows = load_inventory(config)
    updates = {row["image_id"]: approve() for row in rows}
    train = next(row for row in rows if row["split"] == "train")
    updates[train["image_id"]] = approve("unknown")
    set_reviews(config, updates)
    result = export_dataset(config)
    assert result["excluded"]["unknown_label"] == 1
    assert result["counts"]["train"] == 1
    updates[train["image_id"]] = {**approve(), "decision": ""}
    set_reviews(config, updates)
    with pytest.raises(ValueError, match="Incomplete review"):
        export_dataset(config)
    assert status_data(config)["dataset"]["dataset_id"] == result["dataset_id"]


def test_revoked_review_blocks_previous_export_after_failed_reexport(tmp_path):
    config = fixture_config(tmp_path)
    scan(config)
    rows = load_inventory(config)
    set_reviews(config, {row["image_id"]: approve() for row in rows})
    result = export_dataset(config)
    assert validate_current_export(config) == result
    validation = next(row for row in rows if row["split"] == "validation")
    set_reviews(config, {validation["image_id"]: {"decision": "rejected"}})
    with pytest.raises(ValueError, match="nonempty train and validation"):
        export_dataset(config)
    assert status_data(config)["dataset"]["dataset_id"] == result["dataset_id"]
    with pytest.raises(ValueError, match="reviews_sha256 differs"):
        validate_current_export(config)
    with pytest.raises(ValueError, match="reviews_sha256 differs"):
        validate_current_export(config, result)


@pytest.mark.parametrize("changed", ["policy", "predictions"])
def test_current_export_rejects_changed_policy_or_teacher_output(tmp_path, changed):
    config = fixture_config(tmp_path)
    scan(config)
    predictions(config)
    set_reviews(config, {row["image_id"]: approve() for row in load_inventory(config) if row["split"] == "validation"})
    result = export_dataset(config)
    if changed == "policy":
        config["pseudo_labels"]["min_confidence"] = 0.97
        message = "selection_policy differs"
    else:
        predictions(config, confidence=0.98)
        message = "predictions_sha256 differs"
    with pytest.raises(ValueError, match=message):
        validate_current_export(config, result)
    fresh = export_dataset(config)
    assert fresh["dataset_id"] != result["dataset_id"]
    assert validate_current_export(config) == fresh


def test_prepared_receipt_checked_without_requiring_latest_pointer(tmp_path):
    config = fixture_config(tmp_path)
    scan(config)
    set_reviews(config, {row["image_id"]: approve() for row in load_inventory(config)})
    result = export_dataset(config)
    Path(config["work_dir"], "latest_dataset.json").unlink()
    assert validate_current_export(config, result) == result
    Path(result["test_file"]).write_text("{}\n")
    with pytest.raises(ValueError, match="Frozen dataset file changed"):
        validate_current_export(config, result)


def test_teacher_checkpoint_hashed_once_per_validation(tmp_path, monkeypatch):
    from mediflow_medvlm.kiosk import dataset
    config = fixture_config(tmp_path)
    checkpoint = tmp_path / "synthetic-model.bin"
    checkpoint.write_bytes(b"fixture-only: not an executable model")
    config["teachers"]["eye"]["checkpoint"] = str(checkpoint)
    scan(config)
    rows = predictions(config)
    for row in rows:
        row["teacher"]["checkpoint_sha256"] = file_hash(checkpoint)
    root = Path(config["work_dir"])
    write_jsonl(root / "predictions.jsonl", rows)
    meta = json.loads((root / "predictions_meta.json").read_text())
    meta["predictions_sha256"] = file_hash(root / "predictions.jsonl")
    atomic_json(root / "predictions_meta.json", meta)
    reads = []
    def counted_hash(path):
        if Path(path) == checkpoint:
            reads.append(path)
        return file_hash(path)
    monkeypatch.setattr(dataset, "file_hash", counted_hash)
    parsed, _ = dataset._predictions(config, load_inventory(config))
    assert len(parsed) == 4
    assert len(reads) == 1
