"""Unattended orchestration; every model action runs in its existing CLI process."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

from .common import REPO, load_config, normalize_config, workspace
from .terminal import _lease, _run_name
from ..runtime import atomic_json, file_hash, utcnow


TRAIN_PROBE = """import torch, torchvision, transformers, peft, accelerate, PIL, matplotlib
if not torch.cuda.is_available():
    raise RuntimeError('CUDA is unavailable in the selected training environment')
print('CUDA:', torch.cuda.get_device_name(0), 'torch:', torch.__version__)
"""


def _python(value):
    # Do not resolve a venv's Python symlink: its invoked path selects the venv.
    expanded = os.path.expanduser(os.path.expandvars(value))
    path = shutil.which(expanded) if not os.path.isabs(expanded) else expanded
    if not path or not Path(path).is_file() or not os.access(path, os.X_OK):
        raise ValueError("Python executable not found: " + value)
    return os.path.abspath(path)


def _prerequisites(config, args):
    if os.name != "posix":
        raise ValueError("자동 VLM 학습은 Linux/WSL 환경에서 실행하세요.")
    _run_name(args.name)
    if args.max_steps is not None and args.max_steps < 10:
        raise ValueError("--max-steps는 10 이상이거나 생략해야 합니다.")
    template = config.get("training_template")
    if not template or not Path(template).is_file():
        raise ValueError("기존 training_template 파일을 먼저 지정하세요.")
    adapter = config.get("initialization_adapter")
    if not config["start_from_base"] and (not adapter or not Path(adapter).is_dir()):
        raise ValueError("기존 initialization_adapter 폴더를 먼저 지정하세요. 기본 모델로 자동 대체하지 않습니다.")
    if config.get("initialization_manifest") and not Path(config["initialization_manifest"]).is_file():
        raise ValueError("initialization_manifest 파일을 찾을 수 없습니다.")
    if any(row["modality"] not in {"external_eye_rgb", "scalp_microscope"} for row in config["sources"]):
        raise ValueError("이 자동 실행은 외부 눈/두피 사진만 지원합니다. 안저 사진은 포함하지 마세요.")
    if (workspace(config) / "runs" / args.name).exists():
        raise ValueError("기존 학습 run 이름을 재사용할 수 없습니다: " + args.name)
    if config["pseudo_labels"]["allow_validation"] and not args.allow_pseudo_validation:
        raise ValueError("자동 라벨 validation을 쓰려면 --allow-pseudo-validation을 명시하세요.")


class Pipeline:
    def __init__(self, args):
        self.args = args
        self.config = load_config(args.config)
        _prerequisites(self.config, args)
        self.label_python, self.train_python = _python(args.label_python), _python(args.train_python)
        self.root = workspace(self.config) / "autoruns" / args.name
        self.process = None
        self.cancel_signal = None
        self.state = None

    def _save(self):
        atomic_json(self.root / "state.json", self.state)

    def _interrupt(self, signum, _frame):
        already_requested = self.cancel_signal is not None
        self.cancel_signal = self.cancel_signal or signum
        if not already_requested and self.process is not None and self.process.poll() is None:
            # terminal._execute handles SIGINT and then waits for its own worker.
            # SIGTERM/kill on that controller could orphan the training worker.
            self.process.send_signal(signal.SIGINT)

    def _stage(self, name, argv, *, cpu=False, timeout=None):
        if self.cancel_signal:
            raise InterruptedError("종료 요청을 받았습니다.")
        if file_hash(self.snapshot) != self.state["config_sha256"]:
            raise ValueError("자동 실행 설정 snapshot이 변경되었습니다.")
        log = self.root / (name + ".log")
        stage = {"name": name, "argv": argv, "log": str(log), "status": "RUNNING", "started_at": utcnow()}
        self.state["stages"].append(stage)
        self.state.update(active_stage=name, status="RUNNING")
        self._save()
        print("자동 실행 단계: " + name + " | 로그: " + str(log), flush=True)
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env["PYTHONUNBUFFERED"] = "1"
        if cpu:
            env["CUDA_VISIBLE_DEVICES"] = "-1"
        timed_out = False
        started = time.monotonic()
        with log.open("x", encoding="utf-8") as stream:
            try:
                self.process = subprocess.Popen(argv, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                stage["pid"] = self.process.pid
                self._save()
                if self.cancel_signal:
                    self.process.send_signal(signal.SIGINT)
                while self.process.poll() is None:
                    try:
                        self.process.wait(timeout=0.5)
                    except subprocess.TimeoutExpired:
                        if timeout is not None and time.monotonic() - started > timeout:
                            # Only the direct import/CUDA probe has a timeout;
                            # it never starts the terminal's detached worker.
                            timed_out = True
                            self.process.send_signal(signal.SIGINT)
                            try:
                                self.process.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                self.process.kill()
                                self.process.wait()
                code = self.process.returncode
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                stage.update(status="CANCELLED" if self.cancel_signal else "FAILED",
                             reason=str(exc), finished_at=utcnow())
                self._save()
                raise
            finally:
                if self.process is not None and self.process.poll() is None:
                    self.process.send_signal(signal.SIGINT)
                    self.process.wait()  # Wait for controller and training cleanup.
                self.process = None
        stage.update(exit_code=code, finished_at=utcnow(),
                     status="CANCELLED" if self.cancel_signal else "FAILED" if code or timed_out else "SUCCEEDED")
        self._save()
        if self.cancel_signal:
            raise InterruptedError("종료 요청 후 실행 프로세스의 정리를 완료했습니다.")
        if timed_out or code:
            raise RuntimeError(name + " 단계 실패" + (" (환경 점검 시간 초과)" if timed_out else " (exit " + str(code) + ")") + ": " + str(log))

    def run(self):
        # One unattended controller per workspace, in addition to the existing
        # preparation, worker and shared GPU locks used by individual commands.
        with _lease(workspace(self.config) / ".autorun.lock"):
            self.root.mkdir(parents=True, exist_ok=False)
            self.snapshot = self.root / "config.json"
            source_hash = file_hash(self.config["_config_path"])
            self.config["pseudo_labels"]["allow_validation"] = bool(self.args.allow_pseudo_validation)
            normalize_config(self.config, self.snapshot)
            self.config.pop("_config_path", None)
            with self.snapshot.open("x", encoding="utf-8") as stream:
                json.dump(self.config, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            self.state = {"schema_version": "medvlm.kiosk-autorun.v1", "name": self.args.name,
                          "status": "PREPARED", "pid": os.getpid(), "host": socket.gethostname(),
                          "started_at": utcnow(), "finished_at": None, "active_stage": None, "stages": [],
                          "config": str(self.snapshot), "config_sha256": file_hash(self.snapshot),
                          "source_config_sha256": source_hash,
                          "pseudo_validation_allowed": self.args.allow_pseudo_validation,
                          "clinical_accuracy_evaluated": False,
                          "note": "자동 라벨은 teacher_pseudo입니다. pseudo validation은 독립 질환 정확도 평가가 아닙니다."}
            self._save()
            previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
            for sig in previous:
                signal.signal(sig, self._interrupt)
            try:
                self._stage("preflight", [self.train_python, "-c", TRAIN_PROBE], timeout=180)
                cli = str(REPO / "scripts" / "kiosk_terminal.py")
                for command in ("scan", "label", "review", "export", "prepare", "train", "reload", "results"):
                    cpu = command in {"scan", "label", "review", "export"}
                    argv = [self.label_python if cpu else self.train_python, cli, "--config", str(self.snapshot), command]
                    if not cpu:
                        argv += ["--name", self.args.name]
                    if command == "prepare" and self.args.max_steps is not None:
                        argv += ["--max-steps", str(self.args.max_steps)]
                    if command == "results":
                        argv += ["--save", "--plot"]
                    self._stage(command, argv, cpu=cpu)
                self.state.update(status="SUCCEEDED", active_stage=None)
                code = 0
            except InterruptedError as exc:
                self.state.update(status="CANCELLED", reason=str(exc))
                code = 128 + (self.cancel_signal or signal.SIGINT)
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                self.state.update(status="FAILED", reason=str(exc))
                if self.state["stages"] and self.state["stages"][-1]["status"] == "RUNNING":
                    self.state["stages"][-1].update(status="FAILED", reason=str(exc), finished_at=utcnow())
                code = 2
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
                self.state["finished_at"] = utcnow()
                self._save()
            print(json.dumps({"status": self.state["status"], "state_file": str(self.root / "state.json"),
                              "config": str(self.snapshot), "reason": self.state.get("reason")}, ensure_ascii=False), flush=True)
            return code


def build_parser():
    parser = argparse.ArgumentParser(description="로컬 사진 자동 라벨링 → VLM 학습 → 재로드 → 결과 저장")
    parser.add_argument("--config", required=True)
    parser.add_argument("--label-python", required=True)
    parser.add_argument("--train-python", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--allow-pseudo-validation", action="store_true",
                        help="검수 없는 탐색 실험용 validation을 명시적으로 허용합니다. 독립 정확도 평가가 아닙니다.")
    parser.add_argument("--max-steps", type=int)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return Pipeline(args).run()
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
