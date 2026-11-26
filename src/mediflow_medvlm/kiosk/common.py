"""Small shared contracts; importing the terminal never imports a model runtime."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import uuid

from ..runtime import atomic_json, canonical_hash, file_hash, utcnow

SCHEMA = "medvlm.kiosk-terminal.v1"
REPO = Path(__file__).resolve().parents[3]


def absolute_path(value, base):
    path = Path(os.path.expandvars(str(value))).expanduser()
    return path.resolve() if path.is_absolute() else (Path(base) / path).resolve()


def load_config(path):
    path = Path(path).expanduser().resolve()
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    return normalize_config(config, path)


def normalize_config(config, path):
    """Validate paths before creating a workspace or launching an operation."""
    path = Path(path).expanduser().resolve()
    if not isinstance(config, dict):
        raise ValueError("Kiosk config must be a JSON object.")
    if config.get("schema_version") != SCHEMA:
        raise ValueError("Unsupported kiosk config schema; use init to create a config.")
    if path.is_relative_to(REPO):
        raise ValueError("Keep the local kiosk config outside the git repository; copy the example to your data directory.")
    base = path.parent
    for key in ("work_dir", "data_root"):
        if not config.get(key):
            raise ValueError("Missing config field: " + key)
        config[key] = str(absolute_path(config[key], base))
    if Path(config["work_dir"]) == Path(config["data_root"]):
        raise ValueError("work_dir must be a dedicated subdirectory, different from data_root.")
    if Path(config["work_dir"]).is_relative_to(REPO):
        raise ValueError("work_dir must be outside the git repository (it contains images, labels, and local paths).")
    for key in ("training_template", "initialization_adapter", "initialization_manifest"):
        if config.get(key):
            config[key] = str(absolute_path(config[key], base))
    if type(config.setdefault("start_from_base", False)) is not bool:
        raise ValueError("start_from_base must be a boolean.")
    if config["start_from_base"] and config.get("initialization_adapter"):
        raise ValueError("Choose an initialization_adapter or explicitly start_from_base, not both.")
    sources = config.get("sources", [])
    if not isinstance(sources, list) or not sources:
        raise ValueError("At least one image source is required.")
    ids = set()
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("Each source must be a JSON object.")
        sid = source.get("source_id", "")
        if not isinstance(sid, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", sid) or sid in ids:
            raise ValueError("source_id must be a unique, short ASCII identifier.")
        ids.add(sid)
        if not source.get("image_dir"):
            raise ValueError("Missing image_dir for " + sid)
        source["image_dir"] = str(absolute_path(source["image_dir"], base))
        if source.get("groups_csv"):
            source["groups_csv"] = str(absolute_path(source["groups_csv"], base))
        if source.get("modality") not in {"external_eye_rgb", "scalp_microscope", "skin_rgb", "fundus", "other"}:
            raise ValueError("Set a supported modality explicitly for " + sid)
        image_root = Path(source["image_dir"])
        work_root = Path(config["work_dir"])
        if image_root.is_relative_to(REPO):
            raise ValueError("Keep source photos outside the git repository: " + sid)
        if image_root == work_root or image_root.is_relative_to(work_root) or work_root.is_relative_to(image_root):
            raise ValueError("image_dir and work_dir must not contain each other.")
    teachers = config.setdefault("teachers", {})
    if not isinstance(teachers, dict) or set(teachers) - ids:
        raise ValueError("teachers must be keyed by a configured source_id.")
    for teacher in teachers.values():
        if not isinstance(teacher, dict):
            raise ValueError("Each teacher config must be a JSON object.")
        for key in ("checkpoint", "class_names_file", "predictions_file"):
            if teacher.get(key):
                teacher[key] = str(absolute_path(teacher[key], base))
    split = config.setdefault("split", {"train": 0.8, "validation": 0.1, "test": 0.1, "seed": 20261126})
    if not isinstance(split, dict) or type(split.get("seed")) is not int:
        raise ValueError("split must include an integer seed.")
    weights = [split.get(k) for k in ("train", "validation", "test")]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in weights):
        raise ValueError("Split fractions must be finite and positive.")
    if abs(sum(weights) - 1.0) > 1e-8:
        raise ValueError("Split fractions must sum to one.")
    policy = config.setdefault("pseudo_labels", {})
    if not isinstance(policy, dict):
        raise ValueError("pseudo_labels must be a JSON object.")
    policy.setdefault("allow_train", True)
    policy.setdefault("allow_validation", False)
    policy.setdefault("min_confidence", 0.95)
    policy.setdefault("min_margin", 0.15)
    policy.setdefault("note", "Experimental thresholds; not calibrated clinical probabilities.")
    for key in ("min_confidence", "min_margin"):
        value = policy[key]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(key + " must be between zero and one.")
    for key in ("allow_train", "allow_validation"):
        if type(policy[key]) is not bool:
            raise ValueError(key + " must be a boolean.")
    config["_config_path"] = str(path)
    return config


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def workspace(config):
    return Path(config["work_dir"])


def source_signature(config):
    return canonical_hash({"sources": config["sources"], "split": config["split"]})
