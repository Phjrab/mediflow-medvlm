"""Local image inventories, explicit reviews, and immutable kiosk VQA exports.

No model is imported here. A teacher prediction is always distinguished from a
reviewed label; final test references never come from automatic pseudo-labels.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import uuid

from .common import read_jsonl, source_signature, workspace, write_jsonl
from ..data.schema import sample, validate_record
from ..runtime import atomic_json, canonical_hash, file_hash, utcnow

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
GROUP_COLUMNS = {"relative_path", "subject_id", "session_id", "parent_id", "split"}
REVIEW_COLUMNS = ["image_id", "image_sha256", "source_id", "split", "image_path",
                  "teacher_label", "decision", "label", "label_source", "reviewer", "evidence"]
UNKNOWN_LABELS = {"unknown", "unlabeled", "ungradable", "not_assessable", "미확인", "판단 불가", "판단불가"}
SPLITS = ("train", "validation", "test")


def _groups(source):
    if not source.get("groups_csv"):
        return {}, None
    path = Path(source["groups_csv"])
    result = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "relative_path" not in reader.fieldnames:
            raise ValueError("groups_csv needs a relative_path column: " + str(path))
        if set(reader.fieldnames) - GROUP_COLUMNS or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("Unknown or duplicate groups_csv columns: " + str(path))
        for number, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed groups_csv row {number}")
            if not any(value.strip() for value in row.values()):
                continue
            relative = row["relative_path"].strip().replace("\\", "/")
            parsed = PurePosixPath(relative)
            if not relative or parsed.is_absolute() or ".." in parsed.parts or ":" in relative:
                raise ValueError("groups_csv paths must be safe paths relative to image_dir")
            relative = parsed.as_posix()
            if relative in result:
                raise ValueError("Duplicate groups_csv relative_path: " + relative)
            split = row.get("split", "").strip()
            if split and split not in SPLITS:
                raise ValueError("groups_csv split must be train, validation or test")
            result[relative] = {key: value.strip() for key, value in row.items()}
    return result, file_hash(path)


def _identity_token(kind, value):
    return canonical_hash([kind, value]) if value else None


def _image_info(path):
    from PIL import Image, ImageOps
    before = file_hash(path)
    with Image.open(path) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
        image.load()
        digest = hashlib.sha256()
        digest.update(f"RGB:{image.width}:{image.height}:".encode())
        digest.update(image.tobytes())
        result = {"sha256": before, "pixel_sha256": digest.hexdigest(),
                  "width": image.width, "height": image.height,
                  "orientation_policy": "exif_transpose_rgb"}
    if before != file_hash(path):
        raise ValueError("image changed while being scanned")
    return result


def _collect(config):
    from PIL import Image
    rows, errors, skipped, fingerprints = [], [], [], []
    groups_hashes, unsupported = {}, 0
    for source in sorted(config["sources"], key=lambda item: item["source_id"]):
        root = Path(source["image_dir"]).resolve()
        if not root.is_dir():
            raise ValueError("Image directory not found: " + str(root))
        mapping, groups_hash = _groups(source)
        groups_hashes[source["source_id"]] = groups_hash
        candidates = set()
        for directory, dirs, files in os.walk(root, followlinks=False):
            for name in list(dirs):
                child = Path(directory) / name
                if child.is_symlink():
                    dirs.remove(name)
                    skipped.append({"source_id": source["source_id"],
                                    "relative_path": child.relative_to(root).as_posix(),
                                    "reason": "symlink_directory_skipped"})
            dirs.sort()
            for name in sorted(files):
                path = Path(directory) / name
                if path.suffix.lower() not in IMAGE_EXTENSIONS:
                    unsupported += 1
                    continue
                relative = path.relative_to(root).as_posix()
                candidates.add(relative)
                if path.is_symlink() or not path.resolve().is_relative_to(root):
                    item = {"source_id": source["source_id"], "relative_path": relative,
                            "reason": "symlink_image_skipped"}
                    skipped.append(item)
                    fingerprints.append(item)
                    continue
                fingerprint = {"source_id": source["source_id"], "relative_path": relative}
                try:
                    info = _image_info(path)
                    group = mapping.get(relative, {})
                    row = {"image_id": "img_" + canonical_hash([source["source_id"], relative])[:32],
                           "source_id": source["source_id"], "path": str(path.resolve()),
                           "relative_path": relative, "modality": source["modality"], **info,
                           "subject_id": _identity_token("subject", group.get("subject_id")),
                           "session_id": _identity_token("session", group.get("session_id")),
                           "parent_id": _identity_token("parent", group.get("parent_id")),
                           "_explicit_split": group.get("split", "")}
                    rows.append(row)
                    fingerprint.update(sha256=info["sha256"], status="decoded")
                except (OSError, ValueError, RuntimeError, SyntaxError, Image.DecompressionBombError) as exc:
                    try:
                        fingerprint["sha256"] = file_hash(path)
                    except OSError:
                        fingerprint["sha256"] = None
                    fingerprint["status"] = "decode_failed"
                    errors.append({**fingerprint, "error_type": type(exc).__name__})
                fingerprints.append(fingerprint)
        unknown = set(mapping) - candidates
        if unknown:
            raise ValueError("groups_csv references image paths not found under " + source["source_id"] +
                             ": " + ", ".join(sorted(unknown)[:5]))
    return rows, errors, skipped, fingerprints, groups_hashes, unsupported


def _assign_groups(rows, split_config):
    parents = list(range(len(rows)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    seen = {}
    for index, row in enumerate(rows):
        for field in ("subject_id", "session_id", "parent_id", "sha256", "pixel_sha256"):
            value = row.get(field)
            if not value:
                continue
            key = (field, value)
            if key in seen:
                parents[find(index)] = find(seen[key])
            else:
                seen[key] = index
    components = defaultdict(list)
    for index in range(len(rows)):
        components[find(index)].append(index)
    for indices in components.values():
        group_id = "grp_" + canonical_hash(sorted(rows[index]["image_id"] for index in indices))[:32]
        explicit = {rows[index]["_explicit_split"] for index in indices if rows[index]["_explicit_split"]}
        if len(explicit) > 1:
            raise ValueError("Conflicting explicit splits within a subject/parent/exact duplicate group")
        if explicit:
            selected = next(iter(explicit))
        else:
            bucket = int(canonical_hash([split_config.get("seed", 20261126), group_id])[:16], 16) / 2**64
            selected = "train" if bucket < split_config["train"] else (
                "validation" if bucket < split_config["train"] + split_config["validation"] else "test")
        for index in indices:
            rows[index].update(group_id=group_id, split=selected)
            rows[index].pop("_explicit_split")
    return rows


def scan(config):
    """Decode local images and freeze one inventory; changed sources need a new work_dir."""
    root = workspace(config)
    rows, errors, skipped, fingerprints, groups_hashes, unsupported = _collect(config)
    rows = sorted(_assign_groups(rows, config["split"]), key=lambda row: row["image_id"])
    signature = canonical_hash({"source_signature": source_signature(config),
                                "files": fingerprints, "groups_csv_sha256": groups_hashes})
    summary = {"schema_version": "medvlm.kiosk-inventory.v1", "status": "SCANNED",
               "source_signature": source_signature(config), "snapshot_signature": signature,
               "groups_csv_sha256": groups_hashes, "created_at": utcnow(),
               "decoded_count": len(rows), "decode_failed_count": len(errors),
               "skipped_path_count": len(skipped), "unsupported_file_count": unsupported,
               "split_counts": {key: sum(row["split"] == key for row in rows) for key in SPLITS},
               "source_counts": dict(Counter(row["source_id"] for row in rows)),
               "group_count": len({row["group_id"] for row in rows}),
               "exact_pixel_duplicate_count": len(rows) - len({row["pixel_sha256"] for row in rows}),
               "subject_ids_provided_count": sum(bool(row["subject_id"]) for row in rows),
               "patient_independence": "user_group_ids_not_independently_verified" if rows and all(
                   row["subject_id"] for row in rows) else "unverified_missing_subject_ids",
               "errors": errors, "skipped_paths": skipped,
               "inventory_file": str(root / "inventory.jsonl")}
    inventory, meta = root / "inventory.jsonl", root / "inventory_meta.json"
    if inventory.exists() or meta.exists():
        if not inventory.is_file() or not meta.is_file():
            raise ValueError("Incomplete inventory snapshot; use a new work_dir.")
        previous = json.loads(meta.read_text(encoding="utf-8"))
        if previous.get("inventory_sha256") != file_hash(inventory):
            raise ValueError("Inventory file changed; use a new work_dir.")
        if previous.get("snapshot_signature") != signature:
            raise ValueError("Source images, grouping or split changed; use a new work_dir and scan again.")
        return {**previous, "reused": True}
    if not rows:
        atomic_json(root / "scan_diagnostics.json", summary)
        raise ValueError("No decodable images found. See scan_diagnostics.json; no inventory was frozen.")
    write_jsonl(inventory, rows)
    summary["inventory_sha256"] = file_hash(inventory)
    atomic_json(meta, summary)
    return summary


def load_inventory(config, verify_images=False):
    root = workspace(config)
    inventory, meta_path = root / "inventory.jsonl", root / "inventory_meta.json"
    if not inventory.is_file() or not meta_path.is_file():
        raise ValueError("Run scan first: inventory.jsonl and inventory_meta.json are required.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("inventory_sha256") != file_hash(inventory):
        raise ValueError("Inventory digest mismatch")
    if meta.get("source_signature") != source_signature(config):
        raise ValueError("Inventory config changed; use a new work_dir and scan again.")
    for source in config["sources"]:
        actual = file_hash(source["groups_csv"]) if source.get("groups_csv") else None
        if actual != meta.get("groups_csv_sha256", {}).get(source["source_id"]):
            raise ValueError("Grouping CSV changed; use a new work_dir and scan again.")
    rows = read_jsonl(inventory)
    if len({row["image_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate image_id in inventory")
    if verify_images:
        sources = {row["source_id"]: Path(row["image_dir"]).resolve() for row in config["sources"]}
        for row in rows:
            path = Path(row["path"])
            if path.is_symlink() or not path.resolve().is_relative_to(sources[row["source_id"]]):
                raise ValueError("Inventory image escapes its source directory")
            if file_hash(path) != row["sha256"]:
                raise ValueError("Image bytes changed since scan: " + row["image_id"])
    return rows


def _predictions(config, inventory):
    root = workspace(config)
    path, meta_path = root / "predictions.jsonl", root / "predictions_meta.json"
    if not path.exists() and not meta_path.exists():
        return {}, None
    if not path.is_file() or not meta_path.is_file():
        raise ValueError("Incomplete teacher predictions; both predictions files are required.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("inventory_sha256") != file_hash(root / "inventory.jsonl"):
        raise ValueError("Teacher predictions belong to another inventory")
    prediction_hash = file_hash(path)
    if meta.get("predictions_sha256") != prediction_hash:
        raise ValueError("Teacher prediction file changed")
    if meta.get("teacher_config_sha256") != canonical_hash(config.get("teachers", {})):
        raise ValueError("Teacher configuration changed; regenerate in a new work_dir.")
    indexed, result = {row["image_id"]: row for row in inventory}, {}
    signatures = meta.get("teacher_signatures", {})
    teacher_file_hashes = {}
    for pred in read_jsonl(path):
        sid = pred.get("image_id")
        if sid not in indexed or sid in result:
            raise ValueError("Unknown or duplicate image_id in teacher predictions")
        image = indexed[sid]
        if pred.get("image_sha256") != image["sha256"] or pred.get("source_id") != image["source_id"]:
            raise ValueError("Teacher image/source identity mismatch")
        if pred.get("status") not in {"predicted", "failed"}:
            raise ValueError("Unknown teacher prediction status")
        if pred["status"] == "predicted":
            teacher = pred.get("teacher", {})
            signature = signatures.get(image["source_id"])
            if not signature or teacher.get("signature") != signature or teacher.get("modality") != image["modality"]:
                raise ValueError("Teacher provenance or modality mismatch")
            names, scores = teacher.get("class_names"), pred.get("scores")
            if not isinstance(names, list) or not names or len(set(names)) != len(names) or not isinstance(scores, dict) or set(names) != set(scores):
                raise ValueError("Teacher scores/class map mismatch")
            values = list(scores.values())
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
                raise ValueError("Teacher scores must be finite probabilities")
            if abs(sum(values) - 1) > 1e-5:
                raise ValueError("Teacher probability scores do not sum to one")
            ordered = sorted(values, reverse=True)
            margin = ordered[0] - (ordered[1] if len(ordered) > 1 else 0.0)
            if pred.get("predicted_label") not in scores or abs(scores[pred["predicted_label"]] - ordered[0]) > 1e-7:
                raise ValueError("Teacher predicted label does not match scores")
            for key, expected in (("confidence", ordered[0]), ("margin", margin)):
                value = pred.get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value - expected) > 1e-6:
                    raise ValueError("Teacher confidence/margin mismatch")
            configured = config.get("teachers", {}).get(image["source_id"], {})
            if not configured:
                raise ValueError("No registered teacher for a predicted image source")
            for field in ("checkpoint", "class_names_file", "predictions_file"):
                configured_path = configured.get(field)
                if not configured_path:
                    continue
                if configured_path not in teacher_file_hashes:
                    teacher_file_hashes[configured_path] = file_hash(configured_path)
                if teacher.get(field + "_sha256") != teacher_file_hashes[configured_path]:
                    raise ValueError("Teacher " + field + " changed since labeling")
        result[sid] = pred
    if file_hash(path) != prediction_hash:
        raise ValueError("Teacher prediction file changed while being read")
    return result, prediction_hash


def write_review_template(config):
    inventory = load_inventory(config, verify_images=True)
    predictions, _ = _predictions(config, inventory)
    path = workspace(config) / "reviews.csv"
    if path.exists():
        return {"status": "EXISTS_NOT_OVERWRITTEN", "review_file": str(path), "image_count": len(inventory)}
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=REVIEW_COLUMNS)
            writer.writeheader()
            for row in inventory:
                writer.writerow({"image_id": row["image_id"], "image_sha256": row["sha256"],
                                 "source_id": row["source_id"], "split": row["split"],
                                 "image_path": row["path"],
                                 "teacher_label": predictions.get(row["image_id"], {}).get("predicted_label", "")})
        # Never overwrite an edited review file, including a racing writer.
        with path.open("x", encoding="utf-8-sig", newline="") as stream:
            stream.write(temporary.read_text(encoding="utf-8-sig"))
    finally:
        temporary.unlink(missing_ok=True)
    return {"status": "REVIEW_TEMPLATE_CREATED", "review_file": str(path), "image_count": len(inventory)}


def _reviews(config, inventory):
    path = workspace(config) / "reviews.csv"
    if not path.exists():
        return {}, None
    review_hash = file_hash(path)
    indexed, result = {row["image_id"]: row for row in inventory}, {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != REVIEW_COLUMNS:
            raise ValueError("reviews.csv header changed; preserve the generated column names/order")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Malformed review CSV row")
            if not any(value.strip() for value in row.values()):
                continue
            sid = row["image_id"].strip()
            if sid not in indexed or sid in result:
                raise ValueError("Unknown or duplicate image_id in reviews.csv")
            image = indexed[sid]
            if row["image_sha256"] != image["sha256"] or row["source_id"] != image["source_id"] or row["split"] != image["split"]:
                raise ValueError("Review identity/hash/split changed")
            decision = row["decision"].strip().lower()
            if decision not in {"", "approved", "rejected"}:
                raise ValueError("Review decision must be blank, approved or rejected")
            row = {key: value.strip() for key, value in row.items()}
            row["decision"] = decision
            if not decision and any(row[key] for key in ("label", "label_source", "reviewer", "evidence")):
                raise ValueError("Incomplete review: set decision to approved or rejected before export")
            if decision == "approved":
                if not row["label"] or not row["reviewer"]:
                    raise ValueError("An approved review needs a nonempty label and reviewer")
                if row["label_source"] not in {"human_reviewed", "source_supported"}:
                    raise ValueError("Approved label_source must be human_reviewed or source_supported")
                if row["label_source"] == "source_supported" and not row["evidence"]:
                    raise ValueError("source_supported labels need an evidence reference")
            result[sid] = row
    if file_hash(path) != review_hash:
        raise ValueError("reviews.csv changed while being read; finish editing before export")
    return result, review_hash


def _select(config, inventory, predictions, reviews):
    selected, exclusions = [], Counter()
    policy = config["pseudo_labels"]
    for row in inventory:
        review, prediction = reviews.get(row["image_id"], {}), predictions.get(row["image_id"], {})
        if review.get("decision") == "rejected":
            exclusions["review_rejected"] += 1
            continue
        if review.get("decision") == "approved":
            label, kind = review["label"], review["label_source"]
        elif prediction.get("status") == "predicted":
            if row["split"] == "test" or not policy.get("allow_" + row["split"], False):
                exclusions["review_required_" + row["split"]] += 1
                continue
            if prediction["confidence"] < policy["min_confidence"] or prediction["margin"] < policy["min_margin"]:
                exclusions["pseudo_threshold"] += 1
                continue
            label, kind = prediction["predicted_label"], "teacher_pseudo"
        else:
            exclusions["no_eligible_label"] += 1
            continue
        if label.strip().lower() in UNKNOWN_LABELS:
            exclusions["unknown_label"] += 1
            continue
        selected.append((row, label, kind, review, prediction))
    # Exact duplicates cannot supply conflicting answers to the same task.
    labels, unique = {}, []
    for item in sorted(selected, key=lambda value: (value[2] == "teacher_pseudo", value[0]["image_id"])):
        row, label, kind, review, prediction = item
        key = (row["pixel_sha256"], row["modality"])
        if key in labels:
            if labels[key] != label:
                raise ValueError("Conflicting labels for an exact duplicate image; correct reviews before export")
            exclusions["duplicate_labeled_image"] += 1
            continue
        labels[key] = label
        unique.append(item)
    return unique, dict(exclusions)


def classification_question(modality):
    names = {"external_eye_rgb": "눈 외부 사진", "scalp_microscope": "두피 현미경 사진",
             "skin_rgb": "피부 사진", "fundus": "안저 사진", "other": "사진"}
    if modality not in names:
        raise ValueError("Unsupported image modality; use one of: " + ", ".join(names))
    return f"이 {names[modality]}의 분류 항목을 짧게 답하세요."


def _canonical(row, label, kind, review, prediction, revision, policy_hash):
    result = sample("kiosk_" + row["source_id"], revision, row["image_id"], "medical_vqa",
                    classification_question(row["modality"]), label, row["split"], language="ko",
                    raw_sha256=row["sha256"], annotation=kind,
                    metadata={"supervision_kind": kind, "modality": row["modality"],
                              "decoded_pixel_hash": row["pixel_sha256"], "image_sha256": row["sha256"],
                              "kiosk": {"image_id": row["image_id"], "source_id": row["source_id"],
                                        "selection_policy_sha256": policy_hash,
                                        "orientation_policy": row["orientation_policy"],
                                        "reviewer": review.get("reviewer") if kind != "teacher_pseudo" else None,
                                        "evidence": review.get("evidence") if kind != "teacher_pseudo" else None,
                                        "teacher": prediction.get("teacher") if kind == "teacher_pseudo" else None,
                                        "teacher_confidence": prediction.get("confidence") if kind == "teacher_pseudo" else None,
                                        "teacher_margin": prediction.get("margin") if kind == "teacher_pseudo" else None}})
    result["input"]["images"] = [{"local_path": row["path"], "sha256": row["sha256"]}]
    result["supervision"]["supervision_kind"] = kind
    result["group_ids"].update(patient=row["subject_id"], study=row["session_id"],
                               image=row["pixel_sha256"], parent=row["parent_id"], duplicate_cluster=row["group_id"])
    result["grouping_quality"] = {"status": "provided_groups" if row["subject_id"] else "proxy",
                                   "level": "subject" if row["subject_id"] else "image_or_explicit_group",
                                   "patient_group_verified": False}
    result["review_status"] = {"teacher_pseudo": "policy_accepted_pseudo", "human_reviewed": "reviewed",
                              "source_supported": "source_supported"}[kind]
    result["restrictions"].update(sft_allowed=row["split"] != "test")
    result["derivation"] = "teacher_pseudo" if kind == "teacher_pseudo" else "reviewed_image_label"
    errors = validate_record(result)
    if errors:
        raise ValueError("Invalid canonical record: " + ", ".join(errors))
    return result


def export_dataset(config):
    """Export reviewed references and explicitly allowed train/validation pseudo-labels."""
    root = workspace(config)
    inventory = load_inventory(config, verify_images=True)
    predictions, prediction_hash = _predictions(config, inventory)
    reviews, review_hash = _reviews(config, inventory)
    selected, exclusions = _select(config, inventory, predictions, reviews)
    counts = {key: sum(item[0]["split"] == key for item in selected) for key in SPLITS}
    if not counts["train"] or not counts["validation"]:
        diagnostic = {"status": "BLOCKED_LABELS", "counts": counts, "excluded": exclusions,
                      "reason": "Nonempty train and validation are required; review labels, especially validation."}
        atomic_json(root / "export_diagnostics.json", diagnostic)
        raise ValueError(f"Export needs nonempty train and validation; counts={counts}. "
                         "Fill reviews.csv, or explicitly allow exploratory pseudo validation. "
                         "Small group counts may require a new grouping/split workspace.")
    policy_hash = canonical_hash(config["pseudo_labels"])
    identity = {"schema_version": "medvlm.kiosk-dataset.v1",
                "inventory_sha256": file_hash(root / "inventory.jsonl"),
                "predictions_sha256": prediction_hash, "reviews_sha256": review_hash,
                "selection_policy": config["pseudo_labels"],
                "selected": [{"image_id": row["image_id"], "image_sha256": row["sha256"],
                              "split": row["split"], "label": label, "kind": kind}
                             for row, label, kind, _, _ in selected]}
    dataset_id = canonical_hash(identity)
    destination = root / "datasets" / dataset_id
    records = [_canonical(*item, dataset_id, policy_hash) for item in selected]
    by_split = {key: [row for row in records if row["split"] == key] for key in SPLITS}
    validation_kinds = {row["supervision"]["supervision_kind"] for row in by_split["validation"]}
    validation_truth = "teacher_pseudo" if validation_kinds == {"teacher_pseudo"} else (
        "mixed" if "teacher_pseudo" in validation_kinds else "verified")
    if destination.exists():
        manifest_path = destination / "freeze.json"
        if not manifest_path.is_file():
            raise ValueError("Incomplete frozen dataset directory; use a new work_dir")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("dataset_id") != dataset_id:
            raise ValueError("Frozen dataset identity mismatch")
        for key in SPLITS:
            if file_hash(destination / (key + ".jsonl")) != manifest["splits"]["kiosk/" + key]["sha256"]:
                raise ValueError("Frozen dataset file changed")
    else:
        temporary = destination.with_name(".pending-" + uuid.uuid4().hex)
        temporary.mkdir(parents=True)
        try:
            for key in SPLITS:
                write_jsonl(temporary / (key + ".jsonl"), by_split[key])
            manifest = {"schema_version": "medvlm.kiosk-freeze.v1", "dataset_id": dataset_id,
                        "identity": identity, "created_at": utcnow(), "validation_truth": validation_truth,
                        "splits": {"kiosk/" + key: {"path": str(destination / (key + ".jsonl")),
                                    "sha256": file_hash(temporary / (key + ".jsonl")), "count": counts[key]}
                                   for key in SPLITS}}
            atomic_json(temporary / "freeze.json", manifest)
            temporary.rename(destination)
        except Exception:
            # Leave a visibly incomplete private directory for diagnosis; never publish its pointer.
            raise
    result = {"status": "READY_FOR_TRAINING", "schema_version": "medvlm.kiosk-export.v1",
              "dataset_id": dataset_id, "dataset_dir": str(destination),
              "train_file": str(destination / "train.jsonl"),
              "validation_file": str(destination / "validation.jsonl"),
              "test_file": str(destination / "test.jsonl"),
              "freeze_manifest": str(destination / "freeze.json"),
              "freeze_sha256": file_hash(destination / "freeze.json"),
              "counts": counts, "image_root": str(root), "validation_truth": validation_truth,
              "supervision_counts": dict(Counter(row["supervision"]["supervision_kind"] for row in records)),
              "excluded": exclusions, "test_truth": "reviewed_only" if counts["test"] else "missing",
              "unresolved": {"test_labels_missing": counts["test"] == 0,
                             "patient_independence_verified": False,
                             "validation_uses_teacher_pseudo": validation_truth != "verified"},
              "clinical_quality_certified": False}
    (root / "export_diagnostics.json").unlink(missing_ok=True)
    atomic_json(root / "latest_dataset.json", result)
    return result


def validate_current_export(config, exported=None):
    """Require an immutable export that still matches current labels and policy.

    With no explicit receipt, use latest_dataset.json. A prepared run may pass
    its own complete export receipt; the newest pointer need not be identical,
    but edited/revoked reviews or a changed labeling policy invalidate that run.
    """
    root = workspace(config)
    if exported is None:
        pointer = root / "latest_dataset.json"
        if not pointer.is_file():
            raise ValueError("Run export before preparing training: latest_dataset.json is missing")
        exported = json.loads(pointer.read_text(encoding="utf-8"))
    if not isinstance(exported, dict) or exported.get("schema_version") != "medvlm.kiosk-export.v1":
        raise ValueError("Invalid kiosk export receipt; run export again")
    dataset_id = exported.get("dataset_id")
    if not isinstance(dataset_id, str) or len(dataset_id) != 64 or any(c not in "0123456789abcdef" for c in dataset_id):
        raise ValueError("Invalid frozen dataset id")
    destination = root / "datasets" / dataset_id
    paths = {"dataset_dir": destination, "freeze_manifest": destination / "freeze.json",
             **{key + "_file": destination / (key + ".jsonl") for key in SPLITS}}
    for key, expected in paths.items():
        if not exported.get(key) or Path(exported[key]).resolve() != expected.resolve():
            raise ValueError("Frozen export path mismatch: " + key)
    freeze_path = paths["freeze_manifest"]
    if not freeze_path.is_file() or file_hash(freeze_path) != exported.get("freeze_sha256"):
        raise ValueError("Frozen export manifest changed; run export and prepare a new run")
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    identity = frozen.get("identity", {})
    if frozen.get("schema_version") != "medvlm.kiosk-freeze.v1" or frozen.get("dataset_id") != dataset_id or canonical_hash(identity) != dataset_id:
        raise ValueError("Frozen export identity mismatch")
    inventory = load_inventory(config, verify_images=True)
    _, prediction_hash = _predictions(config, inventory)
    _, review_hash = _reviews(config, inventory)
    current = {"inventory_sha256": file_hash(root / "inventory.jsonl"),
               "predictions_sha256": prediction_hash, "reviews_sha256": review_hash,
               "selection_policy": config["pseudo_labels"]}
    for key, value in current.items():
        if identity.get(key) != value:
            raise ValueError("Current " + key + " differs from the frozen dataset. "
                             "Run export again and prepare a new run; existing runs keep their frozen data.")
    if frozen.get("validation_truth") != exported.get("validation_truth"):
        raise ValueError("Frozen validation label provenance mismatch")
    for key in SPLITS:
        entry = frozen.get("splits", {}).get("kiosk/" + key, {})
        path = paths[key + "_file"]
        if not entry.get("path") or Path(entry["path"]).resolve() != path.resolve():
            raise ValueError("Frozen split path mismatch: " + key)
        if not path.is_file() or file_hash(path) != entry.get("sha256"):
            raise ValueError("Frozen dataset file changed: " + key)
        if entry.get("count") != exported.get("counts", {}).get(key):
            raise ValueError("Frozen split count mismatch: " + key)
    return exported


def status_data(config):
    root = workspace(config)
    result = {"work_dir": str(root), "inventory": None, "predictions": None,
              "review_file": str(root / "reviews.csv") if (root / "reviews.csv").exists() else None,
              "dataset": None}
    for key, name in (("inventory", "inventory_meta.json"), ("predictions", "predictions_meta.json"),
                      ("dataset", "latest_dataset.json"), ("export_diagnostics", "export_diagnostics.json")):
        path = root / name
        if path.exists():
            try:
                result[key] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                result[key] = {"status": "UNREADABLE", "path": str(path)}
    return result
