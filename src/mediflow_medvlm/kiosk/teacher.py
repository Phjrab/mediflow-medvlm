"""Local classifier predictions, kept separate from reviewed medical labels.

TensorFlow and PyTorch are imported only by an explicitly requested label run.
These adapters accept one RGB image and a single-label class-score vector. They
do not guess architectures, class order, normalization, or diagnosis text.
"""
from __future__ import annotations

from contextlib import contextmanager
from collections import Counter
import hashlib
import inspect
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import time
import uuid

from .common import read_jsonl, write_jsonl, workspace
from ..runtime import atomic_json, canonical_hash, file_hash, utcnow

PREDICTION_SCHEMA = "medvlm.kiosk-predictions.v1"
PREPROCESSING = {"raw_0_255", "zero_one", "minus_one_one", "imagenet"}
RESAMPLING = {"nearest", "bilinear", "bicubic", "lanczos"}
EYE_CHECKPOINT_SHA256 = "68c52d0b1f4848484e9546109ef9eb836bb56004fa9cd8f05ed87c3126ad5c15"
EYE_CLASS_NAMES = ["결막염 (Conjunctivitis)", "다래끼 (Eyelid)", "백내장 (Cataract)",
                   "일반 (Normal)", "포도막염 (Uveitis)"]


def _class_names(spec):
    direct, filename = spec.get("class_names"), spec.get("class_names_file")
    if direct is not None and filename:
        raise ValueError("Set class_names OR class_names_file, not both.")
    names = json.loads(Path(filename).read_text(encoding="utf-8-sig")) if filename else direct
    if (not isinstance(names, list) or len(names) < 2 or
            any(not isinstance(name, str) or not name.strip() or name != name.strip() for name in names)):
        raise ValueError("class_names must be an ordered JSON list of at least two nonempty strings.")
    if len(set(names)) != len(names):
        raise ValueError("Duplicate class_names are not allowed.")
    return names


def _specification(source, configured):
    if not isinstance(configured, dict) or not configured:
        raise ValueError("No teacher is configured for this source.")
    backend = configured.get("backend")
    if backend not in {"keras", "torchscript", "predictions", "mediflow_eye"}:
        raise ValueError("backend must be keras, torchscript, predictions, or mediflow_eye.")
    if configured.get("modality") != source["modality"]:
        raise ValueError("Teacher modality must explicitly match the image source modality.")
    if configured.get("device", "cpu") != "cpu":
        raise ValueError("Teacher labeling currently supports device=cpu; use a separate inference export for other devices.")
    names = _class_names(configured)
    if configured.get("output_type") not in {"probabilities", "logits"}:
        raise ValueError("Set output_type explicitly to probabilities or logits.")
    spec = dict(configured)
    spec.update(class_names=names, device="cpu")
    file_fields = {"class_names_file": "class_names_file_sha256"}
    if backend == "predictions":
        file_fields["predictions_file"] = "predictions_file_sha256"
    else:
        file_fields["checkpoint"] = "checkpoint_sha256"
        size = spec.get("input_size")
        if (not isinstance(size, list) or len(size) != 2 or
                any(type(v) is not int or v < 1 or v > 4096 for v in size)):
            raise ValueError("input_size must be [height, width], each from 1 to 4096.")
        if spec.get("preprocessing") not in PREPROCESSING:
            raise ValueError("Set preprocessing explicitly: raw_0_255, zero_one, minus_one_one, or imagenet.")
        if spec.get("color_order") not in {"rgb", "bgr"}:
            raise ValueError("Set color_order explicitly to rgb or bgr.")
        spec.setdefault("resize_method", "bilinear")
        if spec["resize_method"] not in RESAMPLING:
            raise ValueError("Unsupported resize_method.")
        spec.setdefault("resize_mode", "stretch")
        if spec["resize_mode"] not in {"stretch", "center_crop"}:
            raise ValueError("resize_mode must be stretch or center_crop.")
        layout = "nhwc" if backend == "keras" else "nchw"
        if spec.get("input_layout", layout) != layout:
            raise ValueError(backend + " adapter requires input_layout=" + layout)
        spec["input_layout"] = layout
        suffix = Path(str(spec.get("checkpoint", ""))).suffix.lower()
        if backend == "keras" and suffix not in {".keras", ".h5", ".hdf5"}:
            raise ValueError("Keras teacher must be a complete local .keras/.h5 model, not weights-only.")
    processor = spec.get("image_processor", "pillow")
    if processor not in {"pillow", "tensorflow_legacy"}:
        raise ValueError("image_processor must be pillow or tensorflow_legacy.")
    if processor == "tensorflow_legacy" and (
            backend != "keras" or spec.get("preprocessing") != "raw_0_255"
            or spec.get("color_order") != "rgb" or spec.get("resize_method") != "bilinear"
            or spec.get("resize_mode") != "stretch"):
        raise ValueError("tensorflow_legacy requires Keras, raw_0_255, rgb, bilinear and stretch.")
    if backend == "mediflow_eye":
        scope = spec.get("input_scope", source.get("input_scope"))
        if (scope != "single_eye_crop" or source.get("input_scope") not in {None, "single_eye_crop"}):
            raise ValueError("mediflow_eye requires explicit input_scope=single_eye_crop; no automatic face/eye crop is performed.")
        if (source["modality"] != "external_eye_rgb" or names != EYE_CLASS_NAMES
                or spec["input_size"] != [224, 224] or spec["preprocessing"] != "imagenet"
                or spec["color_order"] != "rgb" or spec["output_type"] != "logits"
                or spec["resize_method"] != "bilinear" or spec["resize_mode"] != "stretch"
                or processor != "pillow"):
            raise ValueError("mediflow_eye requires the pinned five-class RGB/ImageNet 224px single-eye classifier contract.")
        if spec.get("expected_checkpoint_sha256", EYE_CHECKPOINT_SHA256) != EYE_CHECKPOINT_SHA256:
            raise ValueError("mediflow_eye only supports its pinned checkpoint SHA-256.")
        spec.update(input_scope=scope, image_processor="pillow", expected_checkpoint_sha256=EYE_CHECKPOINT_SHA256)
    expected = spec.get("expected_checkpoint_sha256")
    if expected is not None and (backend == "predictions" or not isinstance(expected, str)
                                 or not re.fullmatch(r"[0-9a-f]{64}", expected)):
        raise ValueError("expected_checkpoint_sha256 must be a lowercase SHA-256 for a local checkpoint backend.")
    hashes = {"checkpoint_sha256": None, "class_names_file_sha256": None,
              "predictions_file_sha256": None}
    for field, digest_name in file_fields.items():
        value = spec.get(field)
        if not value and field != "class_names_file":
            raise ValueError("Missing teacher " + field)
        if value:
            path = Path(value)
            if not path.is_file():
                raise ValueError("Teacher " + field + " is not an existing local file.")
            hashes[digest_name] = file_hash(path)
    if expected is not None and hashes["checkpoint_sha256"] != expected:
        raise ValueError("Teacher checkpoint SHA-256 does not match expected_checkpoint_sha256.")
    identity = {"backend": backend, "modality": source["modality"], "class_names": names,
                "device": "cpu", "output_type": spec["output_type"], **hashes}
    for key in ("input_size", "preprocessing", "color_order", "resize_method", "resize_mode", "input_layout"):
        identity[key] = spec.get(key)
    # Preserve old default signatures while binding every explicit new contract.
    for key in ("image_processor", "input_scope", "expected_checkpoint_sha256"):
        if key in spec:
            identity[key] = spec[key]
    if spec.get("upstream_model"):
        identity["upstream_model"] = str(spec["upstream_model"])
    identity["signature"] = canonical_hash(identity)
    return spec, identity


def inspect_teachers(config):
    """Inspect local file identities and declared contracts without model imports."""
    results = []
    for source in config["sources"]:
        sid = source["source_id"]
        try:
            _, identity = _specification(source, config.get("teachers", {}).get(sid))
            results.append({"source_id": sid, "status": "READY", "errors": [], "teacher": identity})
        except (ValueError, OSError, TypeError, json.JSONDecodeError) as exc:
            results.append({"source_id": sid, "status": "BLOCKED_CONFIGURATION", "errors": [str(exc)]})
    return {"status": "READY" if all(row["status"] == "READY" for row in results) else "BLOCKED_CONFIGURATION",
            "teachers": results, "model_runtime_loaded": False}


def _probabilities(values, classes, output_type):
    """Validate explicit softmax-style output; no sigmoid/multilabel guessing."""
    if isinstance(values, dict):
        if set(values) != set(classes):
            raise ValueError("Classifier score keys do not match the ordered class names.")
        values = [values[name] for name in classes]
    if hasattr(values, "tolist"):
        values = values.tolist()
    if not isinstance(values, (list, tuple)) or len(values) != len(classes):
        raise ValueError("Classifier output class count does not match class_names.")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
           for value in values):
        raise ValueError("Classifier output must contain finite numeric scores.")
    values = [float(value) for value in values]
    if output_type == "logits":
        peak = max(values)
        values = [math.exp(value - peak) for value in values]
    elif output_type == "probabilities":
        if any(value < 0 or value > 1 for value in values) or abs(sum(values) - 1.0) > 1e-3:
            raise ValueError("Expected normalized single-label probabilities, not independent sigmoid scores.")
    else:
        raise ValueError("Unknown output_type.")
    total = sum(values)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Classifier output has no probability mass.")
    return {name: value / total for name, value in zip(classes, values)}


def _prepare_image(path, spec, expected_sha256=None):
    import numpy as np
    from PIL import Image, ImageOps
    raw = Path(path).read_bytes()
    if expected_sha256 and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("IMAGE_BYTES_CHANGED")
    height, width = spec["input_size"]
    if spec.get("image_processor") == "tensorflow_legacy":
        import tensorflow as tf
        # Match mediflow-datasets' frozen classifier input: decoded pixels are
        # not EXIF-transposed, and TensorFlow bilinear differs from Pillow.
        image = tf.io.decode_image(raw, channels=3, expand_animations=False)
        resized = tf.image.resize(image, (height, width), method="bilinear", antialias=False)
        return np.ascontiguousarray(tf.cast(resized, tf.float32), dtype=np.float32)
    resampling = getattr(Image.Resampling, spec.get("resize_method", "bilinear").upper())
    with Image.open(io.BytesIO(raw)) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
        if spec.get("resize_mode", "stretch") == "center_crop":
            image = ImageOps.fit(image, (width, height), method=resampling)
        else:
            image = image.resize((width, height), resample=resampling)
        values = np.asarray(image, dtype=np.float32)
    if spec["color_order"] == "bgr":
        values = values[..., ::-1].copy()
    method = spec["preprocessing"]
    if method == "zero_one":
        values = values / 255.0
    elif method == "minus_one_one":
        values = values / 127.5 - 1.0
    elif method == "imagenet":
        # Statistics follow the configured channel order; no hidden RGB/BGR swap.
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        if spec["color_order"] == "bgr":
            mean, std = mean[::-1], std[::-1]
        values = (values / 255.0 - mean) / std
    elif method != "raw_0_255":
        raise ValueError("Unknown preprocessing.")
    return np.ascontiguousarray(values, dtype=np.float32)


def _validate_keras_shape(model, spec):
    shape = getattr(model, "input_shape", None)
    expected = (1, *spec["input_size"], 3)
    if shape is not None:
        if not isinstance(shape, (tuple, list)) or len(shape) != 4 or any(isinstance(v, (tuple, list)) for v in shape):
            raise ValueError("Keras adapter requires one NHWC image input.")
        if any(actual is not None and int(actual) != target for actual, target in zip(shape, expected)):
            raise ValueError("Keras model input shape differs from the configured image contract.")
    shape = getattr(model, "output_shape", None)
    if shape is not None and (not isinstance(shape, (tuple, list)) or len(shape) != 2 or
                              shape[-1] != len(spec["class_names"])):
        raise ValueError("Keras model must output one class vector with the configured class count.")


def _load_predictor(spec):
    """Return a per-image callable, loading only a trusted local model file."""
    if spec.get("expected_checkpoint_sha256") and file_hash(spec["checkpoint"]) != spec["expected_checkpoint_sha256"]:
        raise ValueError("Teacher checkpoint changed before model loading.")
    if spec["backend"] == "keras":
        previous = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        try:
            import tensorflow as tf
            tf.config.set_visible_devices([], "GPU")
        finally:
            if previous is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = previous
        # Older Keras loaders predate safe_mode. Never retry a failed modern
        # loader with safe_mode disabled, nor import arbitrary custom objects.
        loader = tf.keras.models.load_model
        kwargs = {"compile": False}
        try:
            parameters = inspect.signature(loader).parameters
        except (TypeError, ValueError):
            parameters = None
        if parameters is None or "safe_mode" in parameters or any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
            kwargs["safe_mode"] = True
        model = loader(spec["checkpoint"], **kwargs)
        _validate_keras_shape(model, spec)

        def predict(path, image_sha256):
            values = _prepare_image(path, spec, image_sha256)
            output = model(values[None, ...], training=False)
            if hasattr(output, "numpy"):
                output = output.numpy()
            if getattr(output, "shape", None) != (1, len(spec["class_names"])):
                raise ValueError("Classifier must return a [1, number_of_classes] tensor.")
            return _probabilities(output[0], spec["class_names"], spec["output_type"])
        return predict
    if spec["backend"] in {"torchscript", "mediflow_eye"}:
        if spec["backend"] == "mediflow_eye":
            # This adapter supports one verified repository model, not arbitrary
            # pickled architectures. Never fall back to weights_only=False.
            configured = dict(spec)
            if configured.get("class_names_file"):
                configured.pop("class_names", None)
            spec, _ = _specification({"modality": "external_eye_rgb"}, configured)
        import torch
        if spec["backend"] == "mediflow_eye":
            from torchvision.models import efficientnet_b0
            model = efficientnet_b0(weights=None)
            model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(EYE_CLASS_NAMES))
            state = torch.load(spec["checkpoint"], map_location="cpu", weights_only=True)
            if not isinstance(state, dict) or not state:
                raise ValueError("mediflow_eye checkpoint must contain a state dict.")
            wrappers = [key for key in ("state_dict", "model_state_dict") if key in state]
            if len(wrappers) > 1:
                raise ValueError("mediflow_eye checkpoint has ambiguous state dict wrappers.")
            if wrappers:
                state = state[wrappers[0]]
            if not isinstance(state, dict) or not state or any(not isinstance(key, str) for key in state):
                raise ValueError("mediflow_eye checkpoint must contain named parameter tensors.")
            prefixed = [key.startswith("module.") for key in state]
            if any(prefixed):
                if not all(prefixed):
                    raise ValueError("mediflow_eye checkpoint has mixed module. prefixes.")
                state = {key.removeprefix("module."): value for key, value in state.items()}
            model.load_state_dict(state, strict=True)
            model.to("cpu")
        else:
            model = torch.jit.load(spec["checkpoint"], map_location="cpu")
        model.eval()

        def predict(path, image_sha256):
            values = _prepare_image(path, spec, image_sha256)
            batch = torch.from_numpy(values.transpose(2, 0, 1).copy()).unsqueeze(0)
            with torch.inference_mode():
                output = model(batch)
            if not isinstance(output, torch.Tensor) or tuple(output.shape) != (1, len(spec["class_names"])):
                raise ValueError("Torch classifier must return a [1, number_of_classes] tensor.")
            return _probabilities(output.detach().float().cpu().numpy()[0], spec["class_names"], spec["output_type"])
        return predict
    raise ValueError("The predictions backend does not load a model.")


def _import_predictions(spec, images):
    expected = {row["image_id"]: row for row in images}
    values = {}
    for row in read_jsonl(spec["predictions_file"]):
        sid = row.get("image_id")
        if sid not in expected or sid in values:
            raise ValueError("Imported predictions contain an unknown or duplicate image_id.")
        image = expected[sid]
        if row.get("image_sha256") != image["sha256"] or row.get("source_id", image["source_id"]) != image["source_id"]:
            raise ValueError("Imported prediction image hash/source does not match the inventory.")
        if not isinstance(row.get("scores"), dict):
            raise ValueError("Imported predictions require class-score objects, not label-only records.")
        values[sid] = _probabilities(row["scores"], spec["class_names"], spec["output_type"])
    return values


@contextmanager
def _label_lock(path):
    """Process lifetime advisory lock, with no stale PID cleanup or process kills."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+b")
    acquired = False
    try:
        if os.name == "nt":
            import msvcrt
            if path.stat().st_size == 0:
                stream.write(b" ")
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("Another label process holds this workspace lock.") from exc
        else:
            import fcntl
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Another label process holds this workspace lock.") from exc
        acquired = True
        stream.seek(0)
        stream.truncate()
        stream.write(json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "started_at": utcnow()}).encode())
        stream.flush()
        yield
    finally:
        if acquired:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def label(config, *, progress=None):
    """Generate weak predictions only when explicitly called by the user."""
    root = workspace(config)
    with _label_lock(root / "locks" / "label.lock"):
        return _label_locked(config, progress=progress)


def _label_locked(config, *, progress=None):
    from .dataset import load_inventory
    root = workspace(config)
    images = load_inventory(config, verify_images=True)
    if not images:
        raise ValueError("No decodable images in the inventory; run scan first.")
    inventory_hash = file_hash(root / "inventory.jsonl")
    inspections = inspect_teachers(config)
    checks = {row["source_id"]: row for row in inspections["teachers"]}
    identities = {sid: row["teacher"] for sid, row in checks.items() if row["status"] == "READY"}
    signatures = {sid: identity["signature"] for sid, identity in identities.items()}
    config_hash = canonical_hash(config.get("teachers", {}))
    binding = {"inventory_sha256": inventory_hash, "teacher_config_sha256": config_hash,
               "teacher_signatures": signatures}
    predictions_path, meta_path = root / "predictions.jsonl", root / "predictions_meta.json"
    if predictions_path.exists() or meta_path.exists():
        if not predictions_path.exists() or not meta_path.exists():
            raise ValueError("Incomplete previous prediction files; preserve them and choose a new work_dir.")
        previous = json.loads(meta_path.read_text(encoding="utf-8"))
        if any(previous.get(key) != value for key, value in binding.items()):
            raise ValueError("Teacher/inventory identity changed; choose a new work_dir for new predictions.")
        if previous.get("predictions_sha256") != file_hash(predictions_path):
            raise ValueError("Existing predictions were modified; do not reuse their provenance.")
        if previous.get("status") == "SUCCEEDED":
            return {**previous, "reused": True, "predictions_file": str(predictions_path), "metadata_file": str(meta_path)}
        history = root / "label_attempts" / uuid.uuid4().hex
        history.mkdir(parents=True)
        shutil.copy2(predictions_path, history / predictions_path.name)
        shutil.copy2(meta_path, history / meta_path.name)
    for sid, configured in config.get("teachers", {}).items():
        if configured.get("backend") == "predictions" and configured.get("predictions_file"):
            if Path(configured["predictions_file"]).resolve() in {predictions_path.resolve(), meta_path.resolve()}:
                raise ValueError("Imported predictions must be a separate input file, not the generated output.")
    started = utcnow()
    rows = []
    total = len(images)
    counts = Counter(total=total, predicted=0, failed=0)
    by_source = {}
    last_progress = 0.0

    def report(status, source_id=None, force=False):
        nonlocal last_progress
        if not force and time.monotonic() - last_progress < 1.0:
            return
        update = {"status": status, "done": len(rows), "total": total,
                  "predicted": counts["predicted"], "failed": counts["failed"],
                  "source_id": source_id, "started_at": started, "updated_at": utcnow()}
        atomic_json(root / "label_progress.json", update)
        if progress:
            progress(update)
        last_progress = time.monotonic()

    report("RUNNING", force=True)
    try:
        for source in config["sources"]:
            sid = source["source_id"]
            selected = [image for image in images if image["source_id"] == sid]
            if not selected:
                continue
            check = checks[sid]
            identity = identities.get(sid, {"backend": "unconfigured", "modality": source["modality"]})
            predictor = imported = None
            setup_error = None
            if check["status"] != "READY":
                setup_error = "TEACHER_CONFIGURATION_INVALID"
            else:
                try:
                    spec, fresh_identity = _specification(source, config["teachers"][sid])
                    if fresh_identity != identity:
                        raise ValueError("Teacher files changed during preparation.")
                    if spec["backend"] == "predictions":
                        imported = _import_predictions(spec, selected)
                    else:
                        predictor = _load_predictor(spec)
                except ImportError as exc:
                    setup_error = "DEPENDENCY_MISSING"
                    check["errors"] = ["Install the configured teacher runtime in this environment: " + type(exc).__name__]
                except Exception as exc:
                    setup_error = "TEACHER_LOAD_OR_IMPORT_FAILED"
                    check["errors"] = [str(exc)[:400]]
            source_counts = Counter(total=len(selected), predicted=0, failed=0)
            for image in selected:
                item = {"image_id": image["image_id"], "image_sha256": image["sha256"],
                        "source_id": sid, "teacher": identity, "status": "failed",
                        "predicted_label": None, "scores": {}, "confidence": None, "margin": None}
                try:
                    if setup_error:
                        raise ValueError(setup_error)
                    if imported is not None:
                        if image["image_id"] not in imported:
                            raise ValueError("MISSING_IMPORTED_PREDICTION")
                        scores = imported[image["image_id"]]
                    else:
                        scores = predictor(image["path"], image["sha256"])
                    ordered = sorted(identity["class_names"], key=lambda name: -scores[name])
                    item.update(status="predicted", scores=scores, predicted_label=ordered[0],
                                confidence=scores[ordered[0]], margin=scores[ordered[0]] - scores[ordered[1]])
                except Exception as exc:
                    item["error_code"] = setup_error or (str(exc) if str(exc) in {"MISSING_IMPORTED_PREDICTION", "IMAGE_BYTES_CHANGED"}
                                                         else "CLASSIFIER_OUTPUT_INVALID")
                rows.append(item)
                state = "predicted" if item["status"] == "predicted" else "failed"
                counts[state] += 1
                source_counts[state] += 1
                report("RUNNING", sid)
            by_source[sid] = dict(source_counts)
            del predictor
        # Do not publish scores for a model or inventory that changed mid-run.
        load_inventory(config, verify_images=True)
        current = inspect_teachers(config)
        current_signatures = {row["source_id"]: row["teacher"]["signature"] for row in current["teachers"] if row["status"] == "READY"}
        if current_signatures != signatures or file_hash(root / "inventory.jsonl") != inventory_hash:
            raise ValueError("Teacher or inventory changed while labeling; generated scores were not published.")
        status = "SUCCEEDED" if counts["predicted"] == total else ("PARTIAL" if counts["predicted"] else "FAILED")
        write_jsonl(predictions_path, rows)
        meta = {"schema_version": PREDICTION_SCHEMA, **binding, "teacher_signature": canonical_hash(signatures),
                "teacher_metadata": identities, "predictions_sha256": file_hash(predictions_path),
                "status": status, "counts": dict(counts), "sources": by_source,
                "teacher_checks": inspections["teachers"], "started_at": started, "finished_at": utcnow(),
                "evidence_level": "TEACHER_PREDICTIONS_ONLY", "clinical_truth_created": False}
        atomic_json(meta_path, meta)
        report(status, force=True)
        return {**meta, "reused": False, "predictions_file": str(predictions_path), "metadata_file": str(meta_path)}
    except BaseException:
        report("INTERRUPTED_OR_FAILED", force=True)
        raise
