"""Exercise the delivered terminal script across real files, without model execution."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

from PIL import Image

from mediflow_medvlm.kiosk.common import load_config, read_jsonl, write_jsonl
from mediflow_medvlm.kiosk.training import prepare_evaluation
from mediflow_medvlm.runtime import atomic_json

REPO = Path(__file__).resolve().parents[1]


def test_script_scan_to_training_and_test_preparation_without_gpu(tmp_path):
    images, shared, work = tmp_path / "photos", tmp_path / "shared", tmp_path / "workspace"
    images.mkdir()
    groups = tmp_path / "groups.csv"
    with groups.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["relative_path", "subject_id", "split"])
        writer.writeheader()
        for index in range(24):
            filename = f"fixture_{index:02}.png"
            Image.new("RGB", (16, 12), (index * 7, 40, 110)).save(images / filename)
            writer.writerow({"relative_path": filename, "subject_id": f"fixture-person-{index}",
                             "split": "train" if index < 20 else "validation" if index < 22 else "test"})
    template = tmp_path / "template.json"
    # This is a configuration-only fixture; no such model is loaded or downloaded.
    atomic_json(template, {"run_id": "fixture", "pair_id": "fixture", "experiment_id": "fixture",
                          "model_id": "fixture/no-model-is-loaded", "revision": "1" * 40,
                          "data_root": str(shared), "train_file": "unused", "validation_file": "unused",
                          "freeze_manifest": "unused", "output_dir": "unused", "device": "cuda",
                          "quantization": "none", "gradient_accumulation_steps": 2,
                          "gradient_checkpointing": False, "local_files_only": True})
    config_path = tmp_path / "kiosk.local.json"
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    def cli(*args, expected=0):
        result = subprocess.run([sys.executable, str(REPO / "scripts/kiosk_terminal.py"),
                                 "--config", str(config_path), "--json", *args], cwd=REPO, env=env,
                                text=True, capture_output=True, timeout=20)
        assert result.returncode == expected, result.stderr + result.stdout
        return json.loads(result.stdout) if result.stdout.strip() else None
    cli("init", "--image-dir", str(images), "--data-root", str(shared),
        "--work-dir", str(work), "--training-template", str(template))
    cfg = json.loads(config_path.read_text())
    cfg["sources"][0]["groups_csv"] = str(groups)
    cfg["start_from_base"] = True
    cfg["teachers"]["eye"] = {"backend": "predictions", "modality": "external_eye_rgb",
                               "predictions_file": str(tmp_path / "fixture-scores.jsonl"),
                               "class_names": ["fixture_a", "fixture_b"], "output_type": "probabilities"}
    atomic_json(config_path, cfg)
    cli("scan")
    inventory = read_jsonl(work / "inventory.jsonl")
    assert len(inventory) == 24
    write_jsonl(tmp_path / "fixture-scores.jsonl", [
        {"image_id": row["image_id"], "image_sha256": row["sha256"],
         "scores": {"fixture_a": 0.99, "fixture_b": 0.01}} for row in inventory
    ])
    assert cli("label")["status"] == "SUCCEEDED"
    cli("review")
    reviews = work / "reviews.csv"
    with reviews.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns, rows = reader.fieldnames, list(reader)
    for row in rows:
        if row["split"] != "train":
            row.update(decision="approved", label="fixture_a", label_source="human_reviewed",
                       reviewer="synthetic-test-fixture-only")
    with reviews.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)
    exported = cli("export")
    assert exported["validation_truth"] == "verified"
    assert exported["test_truth"] == "reviewed_only"
    assert len(read_jsonl(exported["train_file"])) == 20
    prepared = cli("prepare", "--name", "fixture-pilot", "--max-steps", "10")
    assert prepared["gpu_started"] is False
    assert prepared["max_optimizer_steps"] == 10
    assert prepared["train_samples_cap"] == 20
    evaluation = prepare_evaluation(load_config(config_path), "fixture-pilot", "base")
    assert evaluation["test_samples"] == 2 and evaluation["gpu_started"] is False
    assert cli("results")["runs"][0]["status"] == "PREPARED_NOT_RUN"
    assert not (Path(prepared["output_dir"]) / "manifest.json").exists()
    assert not (shared / "runtime").exists()
    # Editing a frozen recipe cannot start a child process under the old receipt.
    recipe = json.loads(Path(prepared["training_config"]).read_text())
    recipe["learning_rate"] = 0.5
    atomic_json(prepared["training_config"], recipe)
    cli("train", "--name", "fixture-pilot", expected=2)
    assert not (work / "runs/fixture-pilot/terminal_worker.json").exists()
