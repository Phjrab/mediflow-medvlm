"""Download pinned MediFlow classifiers and create a local, editable kiosk config.

This module uses only the Python standard library. It never imports a model
runtime, starts inference, or chooses an image's camera modality automatically.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.request
import uuid

from .common import REPO, SCHEMA, absolute_path, normalize_config
from ..runtime import file_hash


EYE_CLASSES = ["결막염 (Conjunctivitis)", "다래끼 (Eyelid)", "백내장 (Cataract)",
               "일반 (Normal)", "포도막염 (Uveitis)"]
SCALP_CLASSES = ["모낭사이홍반", "미세각질", "비듬", "탈모", "피지과다"]
DATASETS_REVISION = "d52bd95e4b6417ad39aa4550eaaf257dd40d29ed"
EYE_REVISION = "bfab34c79381d1603a5ada1e8a96bee7babcc284"
MODELS = {
    "eye": {
        "filename": "eye_effnet_b0.pth", "repository": "Phjrab/mediflow-kiosk-core",
        "revision": EYE_REVISION, "path": "models/Augmented_EffNet_V1_B0_best.pth",
        "sha256": "68c52d0b1f4848484e9546109ef9eb836bb56004fa9cd8f05ed87c3126ad5c15",
        "bytes": 16361799, "classes": EYE_CLASSES, "modality": "external_eye_rgb",
        "description": "외부 눈 단안 crop · EfficientNet-B0 · 5클래스",
    },
    "public6": {
        "filename": "scalp_public6.keras", "repository": "Phjrab/mediflow-datasets",
        "revision": DATASETS_REVISION, "path": "results/hair/selected_models/6class_performance_v1/hair_model.keras",
        "sha256": "5ee54f51e65d8656fc81e88de96ad82a6d119991aa74684f921e4cd7e8b7efbb",
        "bytes": 47234592, "classes": SCALP_CLASSES + ["양호"], "modality": "scalp_microscope",
        "description": "두피 공개 데이터 성능 후보 · 6클래스(양호 포함), 카메라 적응 모델과 별개",
    },
    "camera5": {
        "filename": "scalp_camera5.keras", "repository": "Phjrab/mediflow-datasets",
        "revision": DATASETS_REVISION, "path": "results/camera_adaptation_stage2_20261110/hair/best.keras",
        "sha256": "fa2c59ae6a0b1342b8c693014d0a951c59064a4c8439a313f2520e1ce8c224fd",
        "bytes": 27507790, "classes": SCALP_CLASSES, "modality": "scalp_microscope",
        "description": "두피 카메라 적응 Stage2 · 5클래스(양호/정상 없음)",
    },
}


def model_url(spec):
    return f"https://raw.githubusercontent.com/{spec['repository']}/{spec['revision']}/{spec['path']}"


def list_models():
    return {"status": "AVAILABLE", "models": [
        {"id": name, **deepcopy(spec), "url": model_url(spec)} for name, spec in MODELS.items()
    ], "note": "모델 파일과 계약 목록입니다. 실제 사진의 정확도를 검증한 결과가 아닙니다."}


def _outside_repo(value, kind):
    path = absolute_path(value, Path.cwd())
    if path.is_relative_to(REPO):
        raise ValueError(kind + " must be outside the git repository.")
    return path


def _verify_asset(path, spec):
    if not path.is_file() or path.stat().st_size != spec["bytes"] or file_hash(path) != spec["sha256"]:
        raise ValueError("Model is missing or its SHA-256/size differs: " + str(path))


def download_model(name, destination):
    """Publish only verified bytes; preserve any existing different file."""
    spec = MODELS[name]
    root = _outside_repo(destination, "Model directory")
    path = root / spec["filename"]
    if path.exists():
        _verify_asset(path, spec)
        return {"model": name, "status": "REUSED_VERIFIED", "path": str(path), "sha256": spec["sha256"]}
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / ("." + path.name + ".download-" + uuid.uuid4().hex)
    digest, received = hashlib.sha256(), 0
    request = urllib.request.Request(model_url(spec), headers={"User-Agent": "MediFlow-Kiosk-Setup/1"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("xb") as stream:
            while chunk := response.read(1024 * 1024):
                received += len(chunk)
                if received > spec["bytes"]:
                    raise ValueError("Download exceeds the pinned model size: " + name)
                digest.update(chunk)
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        if received != spec["bytes"] or digest.hexdigest() != spec["sha256"]:
            raise ValueError("Downloaded model SHA-256/size mismatch: " + name)
        # A same-directory hard link publishes atomically without replacing a
        # user file or a racing downloader's destination on Linux/Windows.
        try:
            os.link(temporary, path)
        except FileExistsError:
            _verify_asset(path, spec)
        return {"model": name, "status": "DOWNLOADED_VERIFIED", "path": str(path), "sha256": spec["sha256"]}
    finally:
        temporary.unlink(missing_ok=True)


def _new_json(path, value):
    """Atomic create, never replace an existing edited config."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def configure(args):
    config_path = _outside_repo(args.config, "Config file")
    if config_path.exists():
        raise ValueError("설정 파일이 이미 있습니다. 기존 내용을 보존합니다: " + str(config_path))
    if not args.eye_dir and not args.scalp_dir:
        raise ValueError("--eye-dir 또는 --scalp-dir 중 하나 이상을 지정하세요.")
    if args.eye_dir and args.eye_input != "single_eye_crop":
        raise ValueError("안구는 단안 crop 사진만 지원합니다. --eye-input single_eye_crop을 명시하세요.")
    if not args.eye_dir and (args.eye_input or args.eye_groups):
        raise ValueError("--eye-input/--eye-groups에는 --eye-dir가 필요합니다.")
    if bool(args.scalp_dir) != bool(args.scalp):
        raise ValueError("--scalp-dir와 --scalp public6|camera5를 함께 지정하세요.")
    if args.scalp_groups and not args.scalp_dir:
        raise ValueError("--scalp-groups에는 --scalp-dir가 필요합니다.")
    assets = _outside_repo(args.assets, "Model directory")
    data_root = absolute_path(args.data_root, Path.cwd())
    work_dir = absolute_path(args.work_dir, Path.cwd()) if args.work_dir else data_root / "kiosk_rgb"
    sources, teachers = [], {}
    choices = [("eye", "eye", args.eye_dir, args.eye_groups),
               ("scalp", args.scalp, args.scalp_dir, args.scalp_groups)]
    for source_id, choice, image_value, groups_value in choices:
        if not image_value:
            continue
        spec = MODELS[choice]
        image_dir = absolute_path(image_value, Path.cwd())
        if not image_dir.is_dir():
            raise ValueError("사진 폴더를 찾을 수 없습니다: " + str(image_dir))
        checkpoint = assets / spec["filename"]
        _verify_asset(checkpoint, spec)
        source = {"source_id": source_id, "image_dir": str(image_dir), "modality": spec["modality"],
                  "groups_csv": str(absolute_path(groups_value, Path.cwd())) if groups_value else None}
        if source["groups_csv"] and not Path(source["groups_csv"]).is_file():
            raise ValueError("그룹 CSV를 찾을 수 없습니다: " + source["groups_csv"])
        teacher = {"backend": "mediflow_eye" if source_id == "eye" else "keras",
                   "modality": spec["modality"], "checkpoint": str(checkpoint),
                   "expected_checkpoint_sha256": spec["sha256"], "class_names": list(spec["classes"]),
                   "input_size": [224, 224] if source_id == "eye" else [384, 384],
                   "preprocessing": "imagenet" if source_id == "eye" else "raw_0_255",
                   "color_order": "rgb", "output_type": "logits" if source_id == "eye" else "probabilities",
                   "device": "cpu", "image_processor": "pillow" if source_id == "eye" else "tensorflow_legacy",
                   "resize_method": "bilinear", "resize_mode": "stretch",
                   "source_repository": spec["repository"], "source_revision": spec["revision"],
                   "source_model_path": spec["path"], "model_note": spec["description"]}
        if source_id == "eye":
            source["input_scope"] = teacher["input_scope"] = "single_eye_crop"
        sources.append(source)
        teachers[source_id] = teacher
    config = {"schema_version": SCHEMA, "data_root": str(data_root), "work_dir": str(work_dir),
              "sources": sources, "teachers": teachers,
              "split": {"train": .8, "validation": .1, "test": .1, "seed": 20261126},
              "pseudo_labels": {"allow_train": True, "allow_validation": False, "min_confidence": .95,
                                "min_margin": .15, "note": "실험용 임계값이며 실제 정확도로 보정되지 않았습니다."},
              "training_template": str(absolute_path(args.training_template, Path.cwd())) if args.training_template else None,
              "initialization_adapter": str(absolute_path(args.initialization_adapter, Path.cwd())) if args.initialization_adapter else None,
              "initialization_manifest": None, "start_from_base": False,
              "_help": "안구는 일반 카메라의 단안 crop, 두피는 현미경 사진입니다. VLM 학습 전에 원본 training_template와 initialization_adapter를 지정하세요."}
    normalize_config(config, config_path)
    config.pop("_config_path", None)
    _new_json(config_path, config)
    return {"status": "CONFIG_CREATED", "config": str(config_path), "sources": [row["source_id"] for row in sources],
            "work_dir": str(work_dir), "training_paths_configured": bool(args.training_template and args.initialization_adapter),
            "eye_input_scope": "already_cropped_single_external_eye" if args.eye_dir else None,
            "automatic_eye_roi": False,
            "next": "kiosk_terminal.py --config <이 설정 파일> doctor → scan → label → review"}


def build_parser():
    parser = argparse.ArgumentParser(description="실제 MediFlow 분류 모델 다운로드와 로컬 설정 생성")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="모델 종류·클래스·원본 URL·SHA-256 확인")
    download = sub.add_parser("download", help="선택한 모델만 다운로드하고 SHA-256 검증")
    download.add_argument("--dest", required=True)
    download.add_argument("--eye", action="store_true")
    download.add_argument("--scalp", choices=["public6", "camera5"])
    config = sub.add_parser("configure", help="실제 사진·모델 경로로 새 설정 JSON 생성")
    for name in ("config", "assets", "data-root"):
        config.add_argument("--" + name, required=True)
    config.add_argument("--work-dir")
    config.add_argument("--eye-dir")
    config.add_argument("--eye-input", choices=["single_eye_crop"],
                        help="이미 눈 한쪽을 잘라 놓은 외부 눈 사진을 명시합니다. 자동 ROI 검출은 하지 않습니다.")
    config.add_argument("--eye-groups")
    config.add_argument("--scalp-dir")
    config.add_argument("--scalp", choices=["public6", "camera5"])
    config.add_argument("--scalp-groups")
    config.add_argument("--training-template")
    config.add_argument("--initialization-adapter")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "list":
            result = list_models()
        elif args.command == "configure":
            result = configure(args)
        else:
            choices = (["eye"] if args.eye else []) + ([args.scalp] if args.scalp else [])
            if not choices:
                raise ValueError("--eye 또는 --scalp public6|camera5를 지정하세요.")
            models = []
            for choice in choices:
                print("모델 파일 확인/다운로드: " + choice, file=sys.stderr, flush=True)
                models.append(download_model(choice, args.dest))
            result = {"status": "READY", "models": models}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except KeyboardInterrupt:
        print("다운로드/설정을 중단했습니다. 완료된 검증 파일은 보존합니다.", file=sys.stderr)
        return 130
    except (OSError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
