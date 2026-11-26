"""Manual foreground commands, a Korean terminal menu, and read-only results."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import signal
import socket
import subprocess
import sys
import time

from .common import SCHEMA, absolute_path, load_config, normalize_config, workspace
from ..runtime import atomic_json, utcnow

REPO = Path(__file__).resolve().parents[3]


def _read(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _run_name(name):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name or ""):
        raise ValueError("실행 이름은 영문/숫자로 시작하는 1~64자이며 영문, 숫자, _, -만 사용할 수 있습니다.")
    return name


@contextmanager
def _lease(path):
    """Cross-platform advisory lock; released by the OS after exit or a crash."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        try:
            if os.name == "nt":
                import msvcrt
                if path.stat().st_size == 0:
                    stream.write(b"0")
                    stream.flush()
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as exc:
            raise ValueError("다른 터미널이 같은 작업을 실행 중입니다: " + str(path)) from exc
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def initialize(args):
    """Create a local editable config. Never download models or start a worker."""
    path = Path(args.config).expanduser().resolve()
    if path.exists():
        raise ValueError("설정 파일이 이미 있습니다. 기존 파일을 보존합니다: " + str(path))
    image_dir, data_root = args.image_dir, args.data_root or os.environ.get("MEDVLM_DATA_ROOT")
    if sys.stdin.isatty():
        image_dir = image_dir or input("최신 사진 폴더의 절대경로: ").strip()
        data_root = data_root or input("기존 MEDVLM_DATA_ROOT 경로 (GPU 공용 잠금 사용): ").strip()
    if not image_dir or not data_root:
        raise ValueError("init에는 --image-dir 및 --data-root가 필요합니다.")
    image_dir = str(absolute_path(image_dir, Path.cwd()))
    data_root = str(absolute_path(data_root, Path.cwd()))
    work = str(absolute_path(args.work_dir, Path.cwd())) if args.work_dir else str(Path(data_root) / "kiosk_rgb")
    source = args.source_id
    cfg = {
        "schema_version": SCHEMA, "work_dir": work, "data_root": data_root,
        "sources": [{"source_id": source, "image_dir": image_dir, "modality": args.modality, "groups_csv": None}],
        "teachers": {source: {
            "backend": "keras", "modality": args.modality, "checkpoint": None,
            "class_names_file": None, "input_size": [224, 224],
            "preprocessing": None, "color_order": "rgb", "output_type": "probabilities", "device": "cpu",
            "_help": "checkpoint/class_names_file/입력 크기/전처리를 기존 분류 모델의 실제 계약에 맞게 지정하세요."
        }},
        "split": {"train": 0.8, "validation": 0.1, "test": 0.1, "seed": 20261126},
        "pseudo_labels": {"allow_train": True, "allow_validation": False, "min_confidence": 0.95,
                          "min_margin": 0.15, "note": "실험용 예시 임계값이며 실제 정확도로 보정되지 않았습니다."},
        "training_template": str(absolute_path(args.training_template, Path.cwd())) if args.training_template else None,
        "initialization_adapter": None, "initialization_manifest": None, "start_from_base": False,
        "_help": "경로는 이 설정 파일 기준입니다. validation/test 정답은 reviews.csv로 검토하세요. GPU 학습은 Linux/WSL에서 실행합니다."
    }
    normalize_config(cfg, path)
    cfg.pop("_config_path", None)
    atomic_json(path, cfg)
    return {"status": "CONFIG_CREATED", "config": str(path), "next": "설정의 teacher 및 training_template를 채운 뒤 doctor → scan → label → review를 실행하세요."}


def doctor(config):
    from .teacher import inspect_teachers
    checks = []
    for source in config["sources"]:
        checks.append({"name": "images:" + source["source_id"], "ok": Path(source["image_dir"]).is_dir(), "path": source["image_dir"]})
        if source.get("groups_csv"):
            checks.append({"name": "groups:" + source["source_id"], "ok": Path(source["groups_csv"]).is_file()})
    for module in ("PIL", "torch", "transformers", "peft"):
        checks.append({"name": "package:" + module, "ok": importlib.util.find_spec(module) is not None})
    checks.append({"name": "training_template", "ok": bool(config.get("training_template") and Path(config["training_template"]).is_file())})
    checks.append({"name": "native_training_platform", "ok": os.name == "posix", "detail": "GPU 학습은 Linux/WSL 필요; 점검/결과 조회는 Windows 가능"})
    teachers = inspect_teachers(config)
    return {"status": "CHECKED", "platform": platform.system(), "checks": checks, "teachers": teachers,
            "gpu_probe": "not_performed", "note": "이 명령은 모델 로드, 다운로드, 학습을 시작하지 않습니다."}


def _prepared(config, name):
    name = _run_name(name)
    record = _read(workspace(config) / "runs" / name / "prepared.json")
    if not record:
        raise ValueError("먼저 prepare --name " + name + " 명령으로 실행을 준비하세요.")
    return record


def _parse_events(payload):
    lines = payload.splitlines()
    values = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            values.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1 or payload.endswith("\n"):
                raise
    return values


def _last_events(path, limit=300):
    """Read a bounded tail; ignore only a final incomplete line during a write."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open("rb") as stream:
        size = stream.seek(0, 2)
        start = max(0, size - 262144)
        stream.seek(start)
        payload = stream.read().decode("utf-8", errors="replace")
    if start:
        payload = payload.partition("\n")[2]
    return _parse_events(payload)[-limit:]


def run_status(config, name):
    prepared = _prepared(config, name)
    output = Path(prepared["output_dir"])
    manifest = _read(output / "manifest.json", {})
    events = _last_events(output / "events.jsonl")
    update = next((event for event in reversed(events) if event.get("event") == "optimizer_update"), {})
    checkpoint = _read(output / "selection.json") or _read(output / "latest.json") or {}
    worker = _read(workspace(config) / "runs" / name / "terminal_worker.json", {})
    # Stored RUNNING can survive a crash. We display the receipt without claiming liveness.
    state = manifest.get("status", "PREPARED_NOT_RUN")
    return {"name": name, "status": state, "worker_record": worker,
            "live_worker_verified": False, "model_id": manifest.get("model_id"),
            "dataset_id": prepared.get("dataset_id"), "validation_truth": prepared.get("validation_truth"),
            "initialization_mode": prepared.get("initialization_mode"),
            "steps": update.get("step", manifest.get("optimizer_steps", 0)),
            "max_steps": _read(prepared["training_config"], {}).get("max_optimizer_steps"),
            "loss": update.get("loss"), "samples_processed": manifest.get("samples_processed", 0),
            "validation_before": (manifest.get("initial_validation") or {}).get("loss"),
            "best_validation": _read(output / "best.json", {}).get("validation"),
            "checkpoint": checkpoint.get("checkpoint"), "reason": manifest.get("reason_code"),
            "peak_gpu_gib": round(manifest["peak_cuda_allocated_bytes"] / 2**30, 3) if manifest.get("peak_cuda_allocated_bytes") is not None else None,
            "output_dir": str(output), "events_file": str(output / "events.jsonl"),
            "reload": _read(output / "reload.json"),
            "evaluations": {kind: _metric_summary(workspace(config) / "runs" / name / "evaluation" / kind) for kind in ("base", "tuned")},
            "note": "RUNNING은 저장된 상태입니다. 현재 프로세스 실행 여부를 뜻하지 않습니다." if state == "RUNNING" else None}


def _metric_summary(root):
    metrics = _read(Path(root) / "metrics.json")
    manifest = _read(Path(root) / "run_manifest.json", {})
    if metrics is None:
        return {"status": manifest.get("status", "NOT_RUN")}
    return {"status": manifest.get("status", "UNKNOWN"),
            **{key: metrics.get(key) for key in ("scheduled", "predictions", "cohorts", "human_review")}}


def _line(value):
    loss = value.get("loss")
    loss_text = "미측정" if loss is None else f"{loss:.5f}"
    return f"{value['name']} | 기록={value['status']} | step={value.get('steps', 0)}/{value.get('max_steps', '?')} | loss={loss_text} | 사유={value.get('reason') or '-'}"


def results(config, name=None, save=False, plot=False):
    from .dataset import status_data
    names = [_run_name(name)] if name else [p.parent.name for p in sorted((workspace(config) / "runs").glob("*/prepared.json"))]
    result = {"data": status_data(config), "runs": [run_status(config, n) for n in names]}
    if save or plot:
        if not name:
            raise ValueError("결과 저장에는 --name을 지정하세요.")
        prepared = _prepared(config, name)
        output = Path(prepared["output_dir"])
        report_dir = workspace(config) / "runs" / name / "report"
        report_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(report_dir / "summary.json", result)
        events = _parse_events((output / "events.jsonl").read_text(encoding="utf-8")) if (output / "events.jsonl").exists() else []
        updates = [e for e in events if e.get("event") == "optimizer_update"]
        with (report_dir / "training_loss.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=["step", "loss", "learning_rate", "timestamp"], extrasaction="ignore")
            writer.writeheader()
            writer.writerows({"step": e.get("step"), "loss": e.get("loss"),
                              "learning_rate": e.get("lr"), "timestamp": e.get("at")} for e in updates)
        result["report_dir"] = str(report_dir)
        if plot:
            if not updates:
                raise ValueError("실제 optimizer_update 기록이 없어 그래프를 만들 수 없습니다.")
            try:
                import matplotlib
                matplotlib.use("Agg")
                from matplotlib import pyplot as plt
            except ImportError as exc:
                raise ValueError("그래프에는 matplotlib이 필요합니다: python -m pip install -e '.[report]'") from exc
            figure, axis = plt.subplots(figsize=(9, 4.5))
            axis.plot([e["step"] for e in updates], [e["loss"] for e in updates], color="#087f8c", linewidth=1.4)
            axis.set(xlabel="Optimizer step", ylabel="Training loss", title="Observed training loss")
            axis.grid(alpha=0.2)
            figure.tight_layout()
            figure.savefig(report_dir / "training_loss.png", dpi=160)
            plt.close(figure)
            result["plot"] = str(report_dir / "training_loss.png")
    return result


def _python_argv(argv):
    argv = list(argv)
    argv[0] = sys.executable
    return argv


def _wait_foreground(process, *, interrupted=False):
    """Do not return to a menu while its foreground controller is shutting down."""
    while True:
        try:
            return process.wait()
        except KeyboardInterrupt:
            if not interrupted and os.name == "posix" and process.poll() is None:
                process.send_signal(signal.SIGINT)
            interrupted = True
            print("실행 종료 처리를 기다립니다. 종료 후 메뉴로 돌아갑니다.", flush=True)


def _execute(config, name, argv, *, success_file, success_statuses=("SUCCEEDED",), phase="train"):
    if os.name != "posix":
        raise ValueError("현재 GPU 실행 엔진은 Linux/WSL을 사용합니다. 해당 환경의 Python으로 동일 명령을 실행하세요.")
    record_path = workspace(config) / "runs" / name / "terminal_worker.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONUNBUFFERED"] = "1"
    with _lease(workspace(config) / ".manual-worker.lock"):
        process = subprocess.Popen(_python_argv(argv), cwd=REPO, env=env, shell=False, start_new_session=True)
        worker = {"pid": process.pid, "host": socket.gethostname(), "phase": phase,
                  "started_at": utcnow(), "finished_at": None, "exit_code": None}
        last = None
        interrupted = False
        try:
            atomic_json(record_path, worker)
            print(f"직접 실행 시작: {phase} (PID {process.pid}). Ctrl+C는 이 실행에 중단을 요청합니다.", flush=True)
            while process.poll() is None:
                try:
                    try:
                        current = (_line(run_status(config, name)) if phase in {"train", "resume"}
                                   else f"{name} | {phase} 실행 중 (PID {process.pid})")
                    except (OSError, ValueError, KeyError) as exc:
                        current = "진행 기록을 읽는 중: " + str(exc)
                    if current != last:
                        print(current, flush=True)
                        last = current
                    time.sleep(2)
                except KeyboardInterrupt:
                    if not interrupted:
                        interrupted = True
                        if process.poll() is None:
                            process.send_signal(signal.SIGINT)
                        print("이 worker에 중단을 요청했습니다. 체크포인트/종료 처리를 기다립니다.", flush=True)
                    else:
                        print("중단 처리 중입니다. 다른 실행에는 신호를 보내지 않습니다.", flush=True)
            code = process.wait()
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                _wait_foreground(process, interrupted=True)
            if process.poll() is not None:
                worker.update(finished_at=utcnow(), exit_code=process.returncode)
                atomic_json(record_path, worker)
        result = _read(success_file, {})
        status = result.get("status", "NO_RESULT")
        return {"status": status, "exit_code": code, "success": code == 0 and status in success_statuses,
                "result_file": str(success_file), "name": name, "phase": phase}


def run_action(config, args):
    name = _run_name(args.name)
    prepared = _prepared(config, name)
    # Revalidate immutable config/data/initialization before starting any worker.
    template = _read(prepared["training_config"])
    if not template:
        raise ValueError("Prepared training config is missing.")
    from ..runtime import canonical_hash
    if canonical_hash(template) != prepared.get("training_config_sha256"):
        raise ValueError("준비한 training config가 변경되었습니다. 설정 변경은 새 실행 이름으로 prepare하세요.")
    from ..training.config import load_config as load_training_config, load_frozen_data
    if args.command in {"train", "resume"}:
        from .dataset import validate_current_export
        validate_current_export(config, prepared["dataset"])
    load_frozen_data(load_training_config(prepared["training_config"]))
    output = Path(prepared["output_dir"])
    argv = [sys.executable, "-m", "mediflow_medvlm.training", "train", "--config", prepared["training_config"]]
    phase = args.command
    result_file = output / "manifest.json"
    if phase == "resume":
        checkpoint = args.checkpoint or (_read(output / "latest.json", {}) or {}).get("checkpoint")
        if not checkpoint:
            raise ValueError("재개할 체크포인트가 없습니다. --checkpoint를 지정하거나 train을 실행하세요.")
        argv += ["--resume", str(Path(checkpoint).expanduser().resolve())]
    elif phase == "reload":
        selected = _read(output / "selection.json") or _read(output / "latest.json") or {}
        checkpoint = args.checkpoint or selected.get("checkpoint")
        if not checkpoint:
            raise ValueError("재로드할 체크포인트가 없습니다.")
        argv = [sys.executable, "-m", "mediflow_medvlm.training", "reload", "--config", prepared["training_config"], "--checkpoint", str(Path(checkpoint).expanduser().resolve())]
        result_file = output / "reload.json"
    return _execute(config, name, argv, success_file=result_file, phase=phase)


def menu(config_path):
    items = [
        ("환경/설정 점검", "doctor"), ("사진 목록·그룹 분할", "scan"),
        ("기존 분류기로 라벨 후보 생성", "label"), ("검토 CSV 만들기", "review"),
        ("학습 데이터 내보내기", "export"), ("새 학습 실행 준비", "prepare"),
        ("학습 시작", "train"), ("체크포인트에서 재개", "resume"),
        ("저장 모델 재로드 확인", "reload"), ("평가 실행", "evaluate"),
        ("결과 목록 확인", "results"), ("진행 상황 보기", "watch"),
        ("Base·학습 모델 평가 비교", "compare"), ("저장 VLM으로 사진 한 장 실행", "predict")]
    while True:
        print("\nMediFlow — 수동 실행 터미널\n")
        for i, (title, _) in enumerate(items, 1):
            print(f"{i:2}. {title}")
        print(" 0. 종료")
        choice = input("번호: ").strip()
        if choice == "0":
            return 0
        if not choice.isdigit() or not 1 <= int(choice) <= len(items):
            print("메뉴 번호를 입력하세요.")
            continue
        command = items[int(choice) - 1][1]
        argv = [sys.executable, "-m", "mediflow_medvlm.kiosk.terminal", "--config", str(config_path), command]
        if command in {"prepare", "train", "resume", "reload", "evaluate", "watch", "compare", "predict"}:
            name = input("실행 이름 (예: scalp-pilot-01): ").strip()
            try:
                _run_name(name)
            except ValueError as exc:
                print(exc)
                continue
            argv += ["--name", name]
        if command == "evaluate":
            kind = input("평가 대상 [base/tuned, 기본 tuned]: ").strip() or "tuned"
            if kind not in {"base", "tuned"}:
                print("base 또는 tuned를 입력하세요.")
                continue
            argv += ["--kind", kind]
        if command == "predict":
            source = input("설정한 source_id (예: eye 또는 scalp): ").strip()
            photo = input("실행할 사진의 절대경로: ").strip()
            argv += ["--source-id", source, "--image", photo]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        # Each operation has its own process so TensorFlow memory is released before VLM training.
        process = subprocess.Popen(argv, cwd=REPO, env=env, shell=False, start_new_session=os.name == "posix")
        code = _wait_foreground(process)
        if code:
            print(f"명령 종료 코드: {code}")


def build_parser():
    parser = argparse.ArgumentParser(description="MediFlow 로컬 사진 라벨링·학습·결과 수동 실행")
    parser.add_argument("--config", default=os.environ.get("MEDVLM_KIOSK_CONFIG", str(REPO.parent / "medvlm-data" / "kiosk.local.json")))
    parser.add_argument("--json", action="store_true", help="결과를 JSON으로 표시")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("init", help="모델 실행 없이 로컬 설정 생성")
    p.add_argument("--image-dir"); p.add_argument("--data-root"); p.add_argument("--work-dir")
    p.add_argument("--source-id", default="eye")
    p.add_argument("--modality", choices=["external_eye_rgb", "scalp_microscope", "skin_rgb", "fundus", "other"], default="external_eye_rgb")
    p.add_argument("--training-template")
    for command in ("doctor", "scan", "label", "review", "export", "menu"):
        sub.add_parser(command)
    p = sub.add_parser("prepare"); p.add_argument("--name", required=True); p.add_argument("--max-steps", type=int)
    for command in ("train", "resume", "reload"):
        p = sub.add_parser(command); p.add_argument("--name", required=True)
        if command in {"resume", "reload"}: p.add_argument("--checkpoint")
    p = sub.add_parser("evaluate"); p.add_argument("--name", required=True)
    p.add_argument("--kind", choices=["base", "tuned"], default="tuned")
    p = sub.add_parser("compare"); p.add_argument("--name", required=True)
    p = sub.add_parser("predict", help="저장·재로드한 VLM으로 새 사진 한 장 실행")
    p.add_argument("--name", required=True); p.add_argument("--source-id", required=True)
    p.add_argument("--image", required=True); p.add_argument("--question")
    p = sub.add_parser("results"); p.add_argument("--name"); p.add_argument("--save", action="store_true"); p.add_argument("--plot", action="store_true")
    p = sub.add_parser("watch"); p.add_argument("--name", required=True); p.add_argument("--interval", type=float, default=3.0)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "init":
            result = initialize(args)
        else:
            config = load_config(args.config)
            if args.command == "menu":
                return menu(config["_config_path"])
            if args.command == "doctor":
                result = doctor(config)
            elif args.command == "results":
                result = results(config, args.name, args.save, args.plot)
            elif args.command == "watch":
                if not 0.2 <= args.interval <= 60:
                    raise ValueError("--interval은 0.2~60초 범위여야 합니다.")
                while True:
                    print(_line(run_status(config, args.name)), flush=True)
                    time.sleep(args.interval)
            elif args.command in {"train", "resume", "reload"}:
                result = run_action(config, args)
            elif args.command == "evaluate":
                from .training import prepare_evaluation
                evaluation = prepare_evaluation(config, _run_name(args.name), args.kind)
                result = _execute(config, args.name, evaluation["argv"], success_file=Path(evaluation["output_dir"]) / "run_manifest.json", phase="evaluate:" + args.kind)
            elif args.command == "compare":
                from ..evaluation.runner import compare
                root = workspace(config) / "runs" / _run_name(args.name) / "evaluation"
                result = compare(root / "base", root / "tuned", root / "comparison.json")
            elif args.command == "predict":
                from .inference import prepare_prediction
                with _lease(workspace(config) / ".preparation.lock"):
                    prediction = prepare_prediction(config, _run_name(args.name), args.image, args.source_id, args.question)
                result = _execute(config, args.name, prediction["argv"], success_file=Path(prediction["result_file"]), phase="predict")
                generated = _read(prediction["result_file"], {})
                result.update(answer=generated.get("answer"), finish_reason=generated.get("finish_reason"),
                              reason=generated.get("reason_code"), output_dir=prediction["output_dir"])
            else:
                with _lease(workspace(config) / ".preparation.lock"):
                    if args.command == "scan":
                        from .dataset import scan
                        result = scan(config)
                    elif args.command == "label":
                        from .teacher import label
                        def progress(value):
                            if not args.json:
                                print(f"라벨 생성: {value.get('done', 0)}/{value.get('total', '?')} | 성공={value.get('predicted', 0)} | 실패={value.get('failed', 0)}", flush=True)
                        result = label(config, progress=progress)
                    elif args.command == "review":
                        from .dataset import write_review_template
                        result = write_review_template(config)
                    elif args.command == "export":
                        from .dataset import export_dataset
                        result = export_dataset(config)
                    else:
                        from .training import prepare_training
                        result = prepare_training(config, _run_name(args.name), max_steps=args.max_steps)
        display(result, as_json=args.json, command=args.command)
        if isinstance(result, dict) and (result.get("success") is False or result.get("status") in {"FAILED", "PARTIAL", "BLOCKED", "FAILED_REPRODUCIBLE", "BLOCKED_HARDWARE"}):
            return 2
        return 0
    except KeyboardInterrupt:
        print("\n조회/입력을 종료했습니다. watch는 학습 프로세스를 종료하지 않습니다.", file=sys.stderr)
        return 130
    except (OSError, ValueError, KeyError, ImportError, EOFError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


def display(result, *, as_json=False, command=None):
    if as_json or not isinstance(result, dict):
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if command == "results":
        print("데이터 상태:")
        print(json.dumps(result["data"], ensure_ascii=False, indent=2))
        if not result["runs"]:
            print("준비/실행한 학습이 없습니다. export 후 prepare를 실행하세요.")
        for run in result["runs"]:
            print("\n" + _line(run))
            print("  시작 방식:", run.get("initialization_mode") or "기록 없음")
            print("  validation 정답 출처:", run.get("validation_truth") or "기록 없음")
            print("  체크포인트:", run.get("checkpoint") or "아직 없음")
            print("  기록 폴더:", run["output_dir"])
            print("  재로드:", (run.get("reload") or {}).get("status", "NOT_RUN"))
            for kind, metric in run["evaluations"].items():
                print("  평가", kind + ":", json.dumps(metric, ensure_ascii=False))
            if run.get("note"):
                print("  " + run["note"])
        if result.get("report_dir"):
            print("\n저장한 결과:", result["report_dir"])
        return
    if command == "doctor":
        for check in result["checks"]:
            print(("[OK] " if check["ok"] else "[설정 필요] ") + check["name"] + (" — " + check["detail"] if check.get("detail") else ""))
        print("분류 모델 점검:", json.dumps(result["teachers"], ensure_ascii=False, indent=2))
        print(result["note"])
        return
    if command == "predict":
        print("사진 추론:", result["status"])
        if result.get("answer") is not None:
            print("모델 답변:", result["answer"])
        if result.get("reason"):
            print("사유:", result["reason"])
        print("결과 파일:", result["result_file"])
        return
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
