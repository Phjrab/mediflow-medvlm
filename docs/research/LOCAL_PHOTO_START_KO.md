# 실제 안구·두피 모델로 로컬 사진 처리 시작하기

이 안내는 **일반 카메라로 촬영한 눈 외부 사진**과 **현미경으로 촬영한 두피 사진**을 사용한다. 안저 데이터는 이 설정에 포함하지 않는다. 아래 명령은 사진이 저장된 **Ubuntu/WSL의 Bash 터미널**에서 실행한다.

먼저 기존 분류기로 라벨 후보를 만들고 검토한다. 기존 VLM 체크포인트를 시작점으로 학습하는 단계는 그 다음이다. 분류 모델과 학습할 VLM은 서로 다른 파일이다.

Codex가 환경과 스크립트를 준비하고 사용자가 터미널에서 직접 한 번 시작한 뒤 자리를 비워도 자동 라벨링부터 학습까지 이어지게 하려면 [연속 실행 안내](KIOSK_UNATTENDED_KO.md)와 [로컬 Codex 준비 프롬프트](KIOSK_LOCAL_CODEX_PROMPT_KO.txt)를 사용한다. Codex는 전체 실행을 대신 시작하지 않는다. 자동 라벨을 사용하는 탐색 실험이라는 기록은 유지한다.

## 확인한 실제 모델

| 용도 | 저장소의 가중치 | 클래스/입력 |
| --- | --- | --- |
| 안구 | `mediflow-kiosk-core/models/Augmented_EffNet_V1_B0_best.pth` | EfficientNet-B0, 224×224, 5개 클래스, PyTorch state dict |
| 두피 `public6` | `mediflow-datasets/results/hair/selected_models/6class_performance_v1/hair_model.keras` | EfficientNet-B1, 384×384, 양호 포함 6개 클래스 |
| 두피 `camera5` | `mediflow-datasets/results/camera_adaptation_stage2_20261110/hair/best.keras` | 카메라 적응 Stage2, EfficientNet-B1, 384×384, 5개 클래스 |

안구 출력 순서는 `결막염 (Conjunctivitis)`, `다래끼 (Eyelid)`, `백내장 (Cataract)`, `일반 (Normal)`, `포도막염 (Uveitis)`다. 두피는 `모낭사이홍반`, `미세각질`, `비듬`, `탈모`, `피지과다` 순서이며 `public6`에는 마지막에 `양호`가 추가된다.

모델 출처와 계약:

- [안구 config.py](https://github.com/Phjrab/mediflow-kiosk-core/blob/bfab34c79381d1603a5ada1e8a96bee7babcc284/config.py), [분류기 코드](https://github.com/Phjrab/mediflow-kiosk-core/blob/bfab34c79381d1603a5ada1e8a96bee7babcc284/modules/classifier.py), [가중치 SHA](https://github.com/Phjrab/mediflow-kiosk-core/blob/bfab34c79381d1603a5ada1e8a96bee7babcc284/models/SHA256SUMS)
- [두피 공개 데이터 선정 색인](https://github.com/Phjrab/mediflow-datasets/blob/d52bd95e4b6417ad39aa4550eaaf257dd40d29ed/results/PUBLIC_MODEL_SELECTION_20260928.json), [카메라 적응 색인](https://github.com/Phjrab/mediflow-datasets/blob/d52bd95e4b6417ad39aa4550eaaf257dd40d29ed/results/camera_adaptation_packages_20261111/package_index.json), [전처리 기준](https://github.com/Phjrab/mediflow-datasets/blob/d52bd95e4b6417ad39aa4550eaaf257dd40d29ed/results/MODEL_USAGE.md)

`camera5`에는 정상/양호가 없다. 정상 두피를 넣어도 다섯 항목 중 하나를 선택하므로, 정상도 섞인 자료를 처리하는 예제에서는 `public6`를 명시적으로 선택한다. 이는 별도의 공개 데이터 연구 후보이며 최신 카메라 사진에서 성능이 검증되었다는 뜻이 아니다. 촬영 환경 적응 5클래스 과제를 의도한 경우에만 아래 명령의 `public6`를 `camera5`로 바꾼다.

## 0. 사진 형태와 환경 확인

**안구 입력은 눈 한쪽이 이미 잘린 사진 폴더여야 한다.** 얼굴 전체, 양쪽 눈이 함께 크게 들어간 사진을 이 명령으로 자동 분류하지 않는다. 키오스크 운영 코드는 MediaPipe로 양쪽 눈을 각각 추출하지만, 이 준비 도구는 ROI 검출을 자동 실행하지 않는다. 얼굴 사진이면 먼저 원본·환자/세션 정보를 유지하며 단안 crop을 만들어야 한다. crop 실패 이미지를 얼굴 전체로 대신 처리하지 않는다.

이미지가 ZIP에 들어 있다면 사진 파일이 보이는 폴더로 먼저 압축을 푼다. 사진은 Git 저장소 밖의 원래 위치에 둘 수 있다.

터미널에서 환경을 확인한다.

```bash
python3 --version
uname -m
nvidia-smi
```

아래 라벨링 환경은 Python 3.12를 기준으로 한다. 3.11 또는 3.13도 해당 분류기 패키지 범위에 들어간다. GPU가 없어도 분류기 라벨링은 CPU로 실행할 수 있다. VLM 학습은 별도 CUDA 환경이 필요하다. Jetson이면 아래 PC용 PyTorch 설치 대신 해당 JetPack에 맞는 기존 환경을 사용한다.

## 1. 실행 코드 받기

새 작업 폴더를 만들고 실행 브랜치를 받는다. 이미 같은 폴더에 받았다면 다시 clone하지 않는다.

```bash
export MF_ROOT="$HOME/mediflow-local"
mkdir -p "$MF_ROOT"
cd "$MF_ROOT"

git clone --single-branch \
  --branch codex/kiosk-terminal-workflow-20261126 \
  https://github.com/Phjrab/mediflow-medvlm.git mediflow-medvlm

cd "$MF_ROOT/mediflow-medvlm"
```

이전 안내에서 이미 이 브랜치를 받았다면 해당 clone에서 아래 명령으로 이번 실제 모델 연결 코드를 받는다. 수정한 코드가 있으면 자동 덮어쓰기 대신 Git이 중단한다.

```bash
git status --short
git pull --ff-only
```

## 2. 분류기용 Python 환경 만들기

기존 VLM CUDA 환경과 별도로 만든다. 이 단계에 CPU PyTorch를 설치하므로 기존 학습용 환경에서 아래 설치 명령을 실행하지 않는다.

```bash
python3 -m venv .venv-label
source .venv-label/bin/activate
python -m pip install --upgrade pip

python -m pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cpu

python -m pip install -e '.[kiosk,report]' \
  'tensorflow==2.20.0' 'keras==3.13.2' \
  'numpy>=1.26,<3' 'pillow>=10,<13'
```

`python3 -m venv`에서 `ensurepip` 오류가 나면 Ubuntu의 해당 Python 버전에 맞는 venv 패키지가 필요하다. 배포판 기본 Python이면 `sudo apt install python3-venv` 후 다시 실행한다.

설치 확인:

```bash
CUDA_VISIBLE_DEVICES=-1 python - <<'PY'
import torch, torchvision, tensorflow as tf, keras
print('torch:', torch.__version__)
print('torchvision:', torchvision.__version__)
print('tensorflow:', tf.__version__)
print('keras:', keras.__version__)
PY
```

두피의 TensorFlow/Keras 버전은 원본 저장소의 [고정 의존성](https://github.com/Phjrab/mediflow-datasets/blob/d52bd95e4b6417ad39aa4550eaaf257dd40d29ed/pyproject.toml)에 맞춘다. CPU PyTorch 설치는 [공식 설치 안내](https://pytorch.org/get-started/locally/)의 CPU 경로다.

## 3. 실제 가중치만 다운로드

전체 키오스크 서버나 전체 데이터셋 저장소를 설치할 필요 없이, 지정한 commit의 두 모델 파일만 받는다.

```bash
python scripts/kiosk_setup_models.py download \
  --dest "$MF_ROOT/classifiers" \
  --eye \
  --scalp public6
```

모델 파일은 다음 이름으로 저장된다.

- `classifiers/eye_effnet_b0.pth`
- `classifiers/scalp_public6.keras`

다운로드한 파일의 크기와 SHA-256이 원본 기록과 맞아야 완료된다. 같은 파일은 재사용하고 다른 내용의 기존 파일은 덮어쓰지 않는다. `download`는 추론이나 학습을 시작하지 않는다.

두피만 진행하려면 `--eye`를 빼고, 안구만 진행하려면 `--scalp public6`를 뺀다. 촬영 환경 적응 모델을 의도적으로 선택할 때는 `--scalp camera5`를 사용한다.

## 4. 실제 사진 경로로 설정 만들기

파일 관리자에서 사진 폴더를 열어 `Ctrl+L`로 전체 경로를 확인한다. 아래 `read`의 질문에는 따옴표 없이 폴더 경로 자체를 붙여 넣는다. 명령에서는 변수에 따옴표를 사용하므로 경로에 공백/한글이 있어도 된다.

```bash
read -r -p "눈 한쪽씩 잘린 사진 폴더: " EYE_DIR
read -r -p "두피 현미경 사진 폴더: " SCALP_DIR

export MEDVLM_DATA_ROOT="${MEDVLM_DATA_ROOT:-$MF_ROOT/data}"
export MEDVLM_KIOSK_CONFIG="$MF_ROOT/kiosk.local.json"

python scripts/kiosk_setup_models.py configure \
  --config "$MEDVLM_KIOSK_CONFIG" \
  --assets "$MF_ROOT/classifiers" \
  --data-root "$MEDVLM_DATA_ROOT" \
  --work-dir "$MF_ROOT/work-eye-scalp" \
  --eye-dir "$EYE_DIR" \
  --eye-input single_eye_crop \
  --scalp-dir "$SCALP_DIR" \
  --scalp public6
```

`--eye-input single_eye_crop`은 사진이 실제로 단안 crop이라는 사용자 선언이다. 자동으로 눈 영역을 잘라준다는 옵션이 아니다.

두피만 먼저 하려면 `--eye-dir`, `--eye-input` 두 줄을 빼고 실행한다. 안구만 하려면 `--scalp-dir`, `--scalp` 두 줄을 뺀다. 기존에 생성한 설정 파일이 있으면 새 이름을 사용하거나 필요한 내용을 직접 편집한다.

클래스 목록, 모델 경로, 입력 크기, 전처리는 자동으로 채워진다.

- 안구: `.pth`를 정확한 EfficientNet-B0 구조로 strict 로드, 224×224 RGB/ImageNet 정규화, logits→softmax. 이 helper는 단안 crop의 분류기 입력을 만들며 키오스크 서버의 선행 MediaPipe/업로드 resize까지 재현하는 명령은 아니다.
- 두피: 원본 모델 계약대로 TensorFlow decode, no-EXIF, bilinear `antialias=False`, RGB float32 0–255. 외부 `/255`를 추가하지 않는다.

VLM의 사진 처리와 teacher의 이미지 전처리를 픽셀 단위로 동일하다고 가정하지 않는다. 각자의 모델 입력 계약과 사용한 사진 해시를 기록한다.

이 예제는 `MEDVLM_DATA_ROOT`가 이미 설정되어 있으면 그 값을 유지한다. **실제 VLM 학습 전에는 설정의 `data_root`가 원본 VLM 학습 설정의 공용 루트와 같은지 확인해야 한다.** 이번 작업의 `work_dir`는 그대로 유지할 수 있다.

## 5. 라벨 후보 생성과 검수

```bash
python scripts/kiosk_terminal.py doctor
python scripts/kiosk_terminal.py scan
python scripts/kiosk_terminal.py label
python scripts/kiosk_terminal.py review
```

`doctor`에서 `transformers`, `peft`, `training_template` 항목이 아직 미설정이어도 이번 분류기 라벨링 단계는 준비할 수 있다. 모델이 실제로 로드되어 score를 만드는 검사는 `label`에서 이루어진다. 이 환경은 CPU 분류기 환경이므로 사진이 많으면 시간이 걸린다.

같은 사람·촬영 세션의 사진이 여러 장이면 **scan 전에** 그룹 CSV를 넣는다. `configure`의 `--eye-groups`, `--scalp-groups` 또는 설정의 `groups_csv`를 사용한다. 형식은 [일반 터미널 안내의 그룹 분할 설명](KIOSK_TERMINAL_KO.md#4-사진-목록-분할-라벨-후보-검수)에 있다. 그룹을 모르면 사진/픽셀 중복만 자동으로 묶을 수 있고 환자 독립성은 확인되지 않는다.

생성된 `$MF_ROOT/work-eye-scalp/reviews.csv`를 LibreOffice나 CSV 편집기로 연다. 헤더와 이미지 식별 열을 유지하고 다음 열을 채운다.

| 열 | 검수 완료 행에 넣을 값 |
| --- | --- |
| `decision` | `approved`, 제외할 사진은 `rejected` |
| `label` | 확인한 실제 클래스명 |
| `label_source` | `human_reviewed` 또는 실제 정답 근거가 있는 `source_supported` |
| `reviewer` | 검수자 식별자 |
| `evidence` | `source_supported`이면 정답 근거 파일/문서/행 |

teacher_label은 자동 예측이다. 검수 없이 복사하여 독립 정답으로 표시하지 않는다. 기본적으로 train은 조건을 통과한 자동 라벨을 사용할 수 있지만, validation/test는 검수 자료가 필요하다.

```bash
python scripts/kiosk_terminal.py export
```

train 또는 validation이 비어 있으면 내보내기가 막히며 제외 사유가 표시된다. test가 아직 비어 있으면 학습 준비는 가능하지만 test 평가는 할 수 없다. 설정에 들어간 0.95/0.15는 실험용 채택 임계값이며 현재 사진의 정확도로 보정된 수치가 아니다.

## 6. 기존 VLM으로 추가 학습하기

여기서 쓰는 VLM은 위의 EfficientNet 분류 모델과 다르다. 로컬 설정에 다음 두 항목을 추가한다.

```json
{
  "training_template": "/실제/기존-VQA-run/config.json",
  "initialization_adapter": "/실제/기존-VQA-checkpoint-폴더"
}
```

이 두 줄은 전체 설정을 대체하는 JSON이 아니라 **기존 설정의 두 필드에 넣을 예시**다. 기존 native VQA 체크포인트 또는 완료·모델 준비 기록이 남은 V6 V1–V4 최종 모델을 사용한다. `data_root`도 원본 학습 설정과 동일하게 맞춘다. raw `.safetensors` 하나만 지정하는 방식은 지원하지 않는다.

라벨링 환경을 종료하고 기존 CUDA VLM 학습 환경을 활성화한다.

```bash
deactivate
```

그 다음 **본인이 VLM을 학습하던 가상환경/Conda 환경**을 활성화한다. 실제 환경 경로는 이 컴퓨터의 설치 위치에 따라 다르다. 같은 저장소 루트에서:

```bash
python -m pip install -e '.[kiosk,report]'

python - <<'PY'
import torch
print('torch:', torch.__version__)
print('CUDA 사용 가능:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
PY
```

CUDA 환경과 기존 VLM 경로가 준비되면 시험 학습을 실행한다.

```bash
python scripts/kiosk_terminal.py prepare --name eye-scalp-pilot-01 --max-steps 10
python scripts/kiosk_terminal.py train --name eye-scalp-pilot-01
python scripts/kiosk_terminal.py reload --name eye-scalp-pilot-01
```

기존 엔진의 smoke 기준은 최소 10 update다. 그만큼의 train 자료가 없으면 `--max-steps 10`을 생략하여 작은 전체 자료를 한 번 통과하는 실행을 준비한다. 성공 확인 후 새 이름으로 전체 학습한다.

```bash
python scripts/kiosk_terminal.py prepare --name eye-scalp-full-01
python scripts/kiosk_terminal.py train --name eye-scalp-full-01
python scripts/kiosk_terminal.py reload --name eye-scalp-full-01
```

## 7. 진행 상황, 결과, 새 사진 실행

```bash
python scripts/kiosk_terminal.py watch --name eye-scalp-full-01

python scripts/kiosk_terminal.py results \
  --name eye-scalp-full-01 --save --plot

python scripts/kiosk_terminal.py predict \
  --name eye-scalp-full-01 --source-id eye \
  --image "/실제/새로운-단안-사진.jpg"
```

`watch`는 별도 터미널에서 사용한다. 새 터미널에서는 같은 Python 환경과 `MEDVLM_KIOSK_CONFIG`를 다시 지정한다. 학습 터미널의 `Ctrl+C`는 해당 worker에 중단을 요청하고 종료를 기다린다. `watch`의 `Ctrl+C`는 조회만 끝낸다. 체크포인트에서 같은 run을 재개하려면 `resume --name eye-scalp-full-01`을 사용한다.

메뉴로도 동일하게 실행할 수 있다.

```bash
python scripts/kiosk_terminal.py menu
```

데이터·teacher·split을 바꾸면 기존 inventory를 덮어쓰지 말고 새 작업 폴더를 사용한다. 검수 라벨이나 pseudo 채택 정책을 바꾼 경우에는 다시 export하고 새 run 이름을 준비한다. 자세한 결과 파일 위치와 평가 방법은 [일반 터미널 안내](KIOSK_TERMINAL_KO.md)에 있다.
