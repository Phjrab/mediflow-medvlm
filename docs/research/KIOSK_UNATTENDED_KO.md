# 로컬 사진 자동 라벨링 후 VLM 연속 학습

사용자가 자동 라벨링부터 학습 시작까지 승인한 경우의 실행 방법이다. 사진이 있는 컴퓨터에서 실행한다. 안구는 일반 카메라의 외부 눈 사진이며 안저 사진은 포함하지 않는다.

## Codex에 맡길 작업

[로컬 Codex 작업 프롬프트](KIOSK_LOCAL_CODEX_PROMPT_KO.txt)를 사진이 저장된 PC의 Codex에 전달한다. Codex는 실제 사진 폴더·가상환경·기존 VLM 체크포인트를 확인하고, 필요하면 단안 crop을 준비한 뒤 아래 실행기를 시작한다. 이 저장소를 내려받았다는 사실만으로 학습이 시작되지는 않는다.

실행기는 `scan → label → review → export → prepare → train → reload → results`를 순서대로 실행한다. `review`는 나중에 확인할 빈 검수 양식을 만드는 단계이며, 사람의 입력을 기다리거나 자동 예측을 검수 정답으로 바꾸지 않는다. 분류기와 VLM은 별도의 Python 환경을 사용할 수 있다.

## 준비 조건

- 실제 모델 다운로드와 사진 설정은 [로컬 사진 시작 안내](LOCAL_PHOTO_START_KO.md)를 따른다.
- 설정에 실제 `training_template`, `initialization_adapter`, 기존 학습의 공용 `data_root`를 넣는다. 기존 VLM에 이어서 학습하는 작업에서는 `start_from_base: false`를 유지한다.
- 안구 source는 이미 잘린 한쪽 눈 사진이어야 한다. 얼굴 원본이면 별도 ROI 준비가 선행되어야 한다.
- 분류기 환경에는 안구용 torch/torchvision, 두피용 TensorFlow/Keras가 있어야 한다. 학습 환경에는 기존 CUDA VLM 의존성과 결과 그래프용 matplotlib이 있어야 한다.
- 원본·같은 사람·세션에서 나온 사진은 가능한 범위에서 같은 그룹으로 유지한다. 그룹 근거가 없으면 환자 독립성을 확인했다고 기록하지 않는다.

## 검수 없이 진행하는 실험의 의미

기본 설정은 검수된 validation을 요구한다. 사람이 없는 동안 계속 학습하려면 `--allow-pseudo-validation`을 명시한다. 실행기는 원본 설정을 수정하지 않고 해당 실행의 복사본에만 `pseudo_labels.allow_validation: true`를 적용한다.

자동 라벨은 계속 `teacher_pseudo`로 기록한다. 이때 validation은 기존 분류기의 예측을 재현하는 정도를 확인하는 자료이며 독립적인 질환 정확도 평가가 아니다. 원래 test 분할은 보존하고, 검수된 test가 없으면 정확도 평가를 실행하지 않는다. 분류기 선택 기준과 점수 임계값이 현재 사진에서 보정되었다고 가정하지 않는다.

`label`의 부분 실패를 포함하여 어느 단계든 실패하면 다음 단계로 넘어가지 않는다. 모델·경로·의존성이 준비되어야 전체 흐름을 완료할 수 있다. 실제 가중치 로딩과 GPU 학습 성공은 해당 컴퓨터에서 확인해야 한다.

## 한 번에 실행

아래는 Bash 예시다. Python 두 경로와 설정 경로를 실제 컴퓨터의 값으로 바꾼다. 가상환경의 `bin/python` 심볼릭 링크를 `readlink -f`로 풀지 않는다. 가상환경을 선택하는 경로 자체를 사용한다.

```bash
python scripts/kiosk_autorun.py \
  --config "$HOME/mediflow-local/kiosk.local.json" \
  --label-python "$PWD/.venv-label/bin/python" \
  --train-python "/실제/VLM-가상환경/bin/python" \
  --name eye-scalp-auto-01 \
  --allow-pseudo-validation
```

기본 실행은 준비된 학습 자료를 한 번 통과하는 유한 실행이다. `--max-steps`를 지정하면 10 이상의 update 수를 사용하며 실제 자료가 그만큼 있어야 한다. 같은 이름의 자동 실행 또는 학습 run을 덮어쓰지 않는다.

시작 전 실제 학습 Python에서 CUDA와 주요 의존성을 확인한다. 설정 경로만 존재한다고 체크포인트가 호환된다고 단정하지 않으며, 기존 `prepare`와 학습 엔진의 검증도 그대로 적용한다.

## 터미널을 닫아도 계속 실행하기

먼저 모델·설정·환경 준비를 마친다. 저장소 루트에서 다음과 같이 실행할 수 있다.

```bash
LABEL_PYTHON="$PWD/.venv-label/bin/python"
TRAIN_PYTHON="/실제/VLM-가상환경/bin/python"
KIOSK_CONFIG="$HOME/mediflow-local/kiosk.local.json"
RUN_NAME="eye-scalp-auto-$(date +%Y%m%d-%H%M%S)"
LAUNCH_LOG_DIR="$HOME/mediflow-local/launch-logs"
mkdir -p "$LAUNCH_LOG_DIR"

nohup "$LABEL_PYTHON" scripts/kiosk_autorun.py \
  --config "$KIOSK_CONFIG" \
  --label-python "$LABEL_PYTHON" \
  --train-python "$TRAIN_PYTHON" \
  --name "$RUN_NAME" \
  --allow-pseudo-validation \
  > "$LAUNCH_LOG_DIR/$RUN_NAME.log" 2>&1 < /dev/null &

LAUNCH_PID=$!
printf '%s\n' "$LAUNCH_PID" > "$LAUNCH_LOG_DIR/$RUN_NAME.pid"
```

Ubuntu에서 `systemd-inhibit`가 지원되고 실행 권한이 확인되면 `nohup` 뒤에 `systemd-inhibit --what=sleep --mode=block --why="MediFlow labeling and training"`를 넣어 실행하는 동안 일반 절전을 막을 수 있다. 이 경우 셸에서 얻은 PID는 wrapper의 PID일 수 있으므로 실행기 자체 PID는 `state.json`을 확인한다.

컴퓨터 전원을 켜 두고 화면만 잠근다. 터미널 닫기와 PC 절전·로그아웃·재부팅은 다르다. 이 실행기는 전원 종료 후 자동 복구를 약속하지 않는다.

## 진행 상태와 결과

설정의 `work_dir` 아래에 다음 파일을 만든다.

| 위치 | 내용 |
| --- | --- |
| `autoruns/<이름>/config.json` | 해당 자동 실행의 설정 복사본 |
| `autoruns/<이름>/state.json` | 현재 단계, 단계별 상태, 실행기·자식 PID, 종료 사유 |
| `autoruns/<이름>/<단계>.log` | preflight/scan/label/review/export/prepare/train/reload/results 로그 |
| `runs/<이름>/report/` | 학습 결과 JSON, 실제 loss CSV와 PNG |

`RUNNING` 기록만으로 프로세스 생존을 단정하지 않는다. 로컬 Codex가 시작 직후 실행기 PID와 단계 로그의 실제 갱신을 확인하고, **실행 중**인지 **완료**인지 구분해서 보고하게 한다. `SUCCEEDED`는 모든 단계가 성공했을 때만 기록한다.

추가 결과 조회에는 원본 설정 대신 해당 실행의 설정 복사본을 사용한다.

```bash
"/실제/VLM-가상환경/bin/python" scripts/kiosk_terminal.py \
  --config "/실제/work_dir/autoruns/실행이름/config.json" \
  results --name 실행이름
```

중단하려면 해당 컴퓨터에서 `state.json`에 기록된 실행기 PID가 실제 실행 중인 명령과 일치하는지 확인한 뒤 SIGTERM 또는 SIGINT를 보낸다. 실행기는 기존 CLI에 SIGINT를 전달하고 학습 worker가 정리될 때까지 기다린다. 정리를 건너뛰는 강제 종료는 피한다.

실패 후 전체 자동 실행을 같은 이름으로 재실행하지 않는다. 실패 단계를 확인하고, 학습 중 중단이었다면 기존 `resume` 명령으로 지원되는 체크포인트에서 재개할 수 있다. 데이터나 라벨이 바뀌었다면 새 run을 준비한다.
