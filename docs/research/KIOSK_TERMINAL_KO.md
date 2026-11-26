# 로컬 안구·두피 사진: 라벨링, 추가 학습, 결과 확인

다른 컴퓨터에 내려받은 최신 사진을 **그 컴퓨터의 터미널에서 직접 처리**하는 스크립트다. 기존 질환 분류기로 라벨 후보를 만들고, 검수한 자료와 구분하여 저장한 뒤, 기존 VQA VLM의 LoRA를 시작점으로 새 학습을 실행한다. Drive 검색이나 사진 업로드는 필요하지 않다.

스크립트는 명령을 입력할 때만 동작한다. 메뉴 열기·환경 점검·결과 조회는 학습이나 모델 다운로드를 시작하지 않는다. GPU 실행은 기존 엔진의 Linux/WSL 환경과 공용 GPU 잠금·사용량 기록을 사용한다.

**MediFlow 저장소의 실제 안구·두피 모델을 쓸 때는 [로컬 사진 시작 안내](LOCAL_PHOTO_START_KO.md)를 먼저 따른다.** 해당 모델만 다운로드하고 원본 클래스·전처리를 채운 설정을 만드는 `scripts/kiosk_setup_models.py`가 있다. 안구 `.pth`는 전용 `mediflow_eye` backend, 두피 `.keras`는 `tensorflow_legacy` 전처리 옵션으로 연결한다.

## 1. 준비할 로컬 파일

| 설정 | 넣을 값 |
| --- | --- |
| `sources[].image_dir` | 최신 사진 폴더. 안구와 두피를 서로 다른 source로 등록한다. |
| `sources[].modality` | 일반 카메라 눈 외부 사진은 `external_eye_rgb`, 두피 현미경 사진은 `scalp_microscope`. 안저 카메라 자료는 별도 `fundus`다. |
| `teachers.<source_id>` | 해당 사진 종류로 학습한 기존 질환 분류 모델과 실제 입력·출력 설정. |
| `training_template` | 기존 VQA 학습에 사용한 native `config.json`, 또는 V6 V1–V4의 실제 main-train `config.json`. campaign 계획 파일은 사용할 수 없다. |
| `initialization_adapter` | 기존 학습 모델의 체크포인트 폴더. 일반 native 체크포인트 또는 검증 기록이 있는 V6 최종 출력 폴더를 지원한다. |
| `data_root` | 기존 `MEDVLM_DATA_ROOT`와 같은 경로. GPU 잠금·예산을 다른 실행과 공유한다. |
| `work_dir` | 이번 라벨·검토·학습 결과를 저장할 새 전용 폴더. |

실제 설정 파일, 사진 폴더, `work_dir`는 git 저장소 밖에 둔다. 원본 사진 폴더와 작업 폴더는 서로 포함하지 않아야 한다. 파일은 원본 경로에서 읽으며 이미지를 학습 데이터 JSONL에 복사하지 않는다.

기존 VLM을 이어 학습하는 것이 기본이다. `initialization_adapter`를 비워 두면 준비 단계가 안내와 함께 멈춘다. 사전학습 base에서 새 LoRA를 시작하려는 경우에만 `start_from_base: true`를 명시하고 `initialization_adapter`를 비운다. **기존 adapter를 새 자료의 시작 가중치로 쓰는 동작**과 **중단한 동일 run의 optimizer·RNG를 복원하는 resume**은 별개다.

기존 체크포인트는 원본 `config.json`, manifest/완료 기록과 연결되어야 한다. 일반 native 형식은 원본 VQA 학습 JSONL도 해시 검증에 사용한다. V6는 원본 모델 준비 기록과 로컬 snapshot까지 검증한다. adapter 가중치 파일만 단독으로 복사한 경우에는 이 연결을 확인할 수 없다. 원본 run의 디렉터리 구조와 필요한 기록을 유지하고, 다른 컴퓨터로 옮길 때는 설정·기록에 저장된 경로도 실제로 접근 가능한지 확인한다.

## 2. 실행 환경과 설정 생성

먼저 기존 VLM을 학습·재로드할 수 있었던 Python 환경을 활성화한다. 저장소 루트에서 다음을 실행한다.

```bash
python -m pip install -e '.[kiosk,report]'

export MEDVLM_KIOSK_CONFIG="/absolute/path/to/medvlm-data/kiosk.local.json"

python scripts/kiosk_terminal.py init \
  --image-dir "/absolute/path/to/latest-photos/eye" \
  --source-id eye \
  --modality external_eye_rgb \
  --data-root "/absolute/path/to/medvlm-data" \
  --training-template "/absolute/path/to/existing-vqa-run/config.json"
```

`init`는 수정할 설정 파일만 만든다. 기존 파일은 덮어쓰지 않는다. 생성한 JSON에서 teacher 설정과 `initialization_adapter`를 채운다. 안구·두피를 같이 쓰려면 [설정 예제](../../configs/kiosk_terminal.example.json)의 두 source 구성을 참고한다. 예제의 경로, 224×224 입력 크기, backend는 실제 파일·모델에 맞게 바꿔야 한다.

새 GPU 환경이 필요하면 저장소의 [환경 안내](../../README.md#데이터와-실행)와 `configs/requirements-training.lock.txt`를 기준으로 준비한다. 학습에는 PyTorch, Transformers, PEFT와 실제 모델에 필요한 의존성이 있어야 한다. CPU 데이터 작업만 할 때에는 전체 VLM 런타임 설치가 필요하지 않다. Keras teacher는 그 모델을 실행하던 TensorFlow/Keras 환경에서 사용한다. TensorFlow와 VLM 환경이 다르면 아래의 외부 예측 import를 사용해 분리할 수 있다.

환경변수 대신 모든 명령에 `--config /path/kiosk.local.json`을 줄 수도 있다. `--config`와 `--json`은 **하위 명령 앞**에 둔다.

```bash
python scripts/kiosk_terminal.py --config "$MEDVLM_KIOSK_CONFIG" doctor
python scripts/kiosk_terminal.py --config "$MEDVLM_KIOSK_CONFIG" menu
```

설치 후에는 `medvlm-kiosk` 명령도 같다. Bash에서는 `bash scripts/kiosk_terminal.sh ...`, PowerShell에서는 `./scripts/kiosk_terminal.ps1 ...`도 사용할 수 있다. 특정 Python을 쓸 때는 `MEDVLM_PYTHON`에 실행 파일 경로를 지정한다. Windows에서 데이터 작업을 한 후 WSL로 이동하면 `D:/...` 등 경로를 해당 환경의 `/mnt/d/...` 경로로 설정하여 새 workspace에서 다시 준비한다.

## 3. 기존 분류 모델 연결

EfficientNet-B0 같은 모델 이름만으로 클래스 순서나 전처리를 결정하지 않는다. 기존 모델을 학습할 때 실제 사용한 값을 넣는다.

| 항목 | 계약 |
| --- | --- |
| `backend` | 전체 Keras 모델은 `keras`, TorchScript export는 `torchscript`, 기존 score JSONL은 `predictions`. 검증한 MediFlow 안구 state dict는 전용 `mediflow_eye`. |
| `checkpoint` | 신뢰하는 로컬 `.keras`/`.h5`/`.hdf5`, TorchScript `.pt`, 또는 `mediflow_eye` 전용의 검증한 `.pth`. 그 밖의 임의 state dict는 지원하지 않는다. |
| `class_names_file` | 모델 출력 인덱스 순서대로 된 JSON 문자열 배열. `class_names` 배열을 설정에 직접 넣어도 된다. |
| `input_size` | `[height, width]`. 모델 학습·추론 때 사용한 크기. |
| `preprocessing` | `raw_0_255`, `zero_one`, `minus_one_one`, `imagenet` 중 실제 모델 계약. 모델 안에 rescaling이 있으면 중복 적용하지 않는다. |
| `color_order` | `rgb` 또는 `bgr`. |
| `output_type` | 단일 클래스 분류의 `probabilities` 또는 `logits`. |
| `device` | teacher는 `cpu`를 사용한다. |

현재 teacher는 한 이미지의 단일 `[1, C]` 클래스 벡터를 지원한다. 두피 모델이 여러 증상별 head, multilabel 또는 sigmoid 구조이면 그 구조에 맞는 별도 adapter가 필요하다. 여러 증상의 확률을 단일 질환 softmax처럼 해석해 import하면 안 된다. 안저용 모델을 눈 외부 사진용 teacher로 등록하지 않는다.

기본 Pillow 입력은 EXIF 방향 적용 → RGB 변환 → 지정 크기 resize → 지정 색 순서·전처리 순서다. 두피 원본 계약용 `image_processor: tensorflow_legacy`는 TensorFlow decode·bilinear resize를 사용하고 EXIF 회전 및 외부 정규화를 하지 않는다. 자동 YOLO/MediaPipe 검출·ROI crop은 이 스크립트에 포함되어 있지 않다. 기존 모델이 crop을 요구한다면 동일 crop 과정을 거친 사진이나 기존 추론 파이프라인의 score를 사용한다.

### 다른 환경에서 계산한 분류 score 가져오기

먼저 `scan`을 실행하고 `work_dir/inventory.jsonl`의 `image_id`, `sha256`를 사용한다. 각 이미지의 **모든 클래스 score**를 다음 JSONL 형태로 내보낸다. 아래 값은 형식 예시다.

```json
{"image_id":"INVENTORY_IMAGE_ID","image_sha256":"INVENTORY_SHA256","scores":{"normal":0.97,"condition_a":0.03}}
```

teacher를 다음처럼 설정한 뒤 `label`을 실행한다. 클래스 이름은 기존 모델의 실제 클래스와 같아야 한다.

```json
{
  "backend": "predictions",
  "modality": "external_eye_rgb",
  "predictions_file": "/absolute/path/to/eye-scores.jsonl",
  "class_names": ["normal", "condition_a"],
  "output_type": "probabilities"
}
```

라벨 이름만 있는 파일, 클래스 불일치, NaN/Inf, 이미지 해시 불일치는 거부한다.

## 4. 사진 목록, 분할, 라벨 후보, 검수

```bash
python scripts/kiosk_terminal.py doctor
python scripts/kiosk_terminal.py scan
python scripts/kiosk_terminal.py label
python scripts/kiosk_terminal.py review
```

`doctor`는 경로·설정·설치 여부를 확인한다. 실제 모델을 로드하는 검사는 `label` 또는 사용자가 실행하는 GPU 명령에서 한다. `label`은 진행한 장수·성공·실패를 표시한다. 실패한 사진에 임의 라벨을 채우지 않으며 부분 실패도 결과에 기록한다.

`scan`은 실제 이미지를 열어 크기·파일 해시·방향을 적용한 픽셀 해시를 기록한다. 정확히 같은 픽셀의 중복은 함께 묶고, 고정 seed로 train/validation/test를 나눈다. **환자·촬영 세션을 알고 있다면 scan 전에 `groups_csv`를 제공한다.** 허용되는 열은 다음과 같다. `relative_path`는 필수이고 나머지는 확보한 정보만 넣는다.

```csv
relative_path,subject_id,session_id,parent_id,split
session_a/eye_01.jpg,subject_001,session_001,,train
session_a/eye_02.jpg,subject_001,session_001,,train
session_b/eye_03.jpg,subject_002,session_002,,test
```

그룹 ID는 가명 식별자를 사용한다. 동일 사람·세션·원본 사진에서 파생한 crop은 같은 그룹에 둔다. 폴더명을 질환 정답이나 환자 ID로 추정하지 않는다. 그룹 정보가 없으면 동일 파일/픽셀 중복은 막을 수 있지만 유사 사진이나 동일 사람의 다른 사진까지 판별할 수는 없다. 이 경우 결과의 환자 독립성은 검증되지 않은 상태다.

그룹 ID는 모든 source에 걸쳐 비교하므로, 서로 무관한 사람에게 `session_1` 같은 값을 재사용하지 않는다. 원본 사진과 파생 crop의 행에는 같은 `parent_id`를 넣는다. 같은 사람·세션을 의도적으로 묶는 경우에만 ID를 공유한다.

`review`가 만든 `work_dir/reviews.csv`를 열고 다음 열만 채운다. 기존 검수 파일은 자동으로 덮어쓰지 않는다.

| 열 | 입력 |
| --- | --- |
| `decision` | 검수 완료는 `approved`, 제외는 `rejected`, 미검수는 빈칸. |
| `label` | 검수하여 확인한 분류 항목. teacher_label을 그대로 복사했다는 이유로 정답이 되는 것은 아니다. |
| `label_source` | 직접 검수는 `human_reviewed`, 데이터 제공 출처의 정답을 확인한 경우 `source_supported`. |
| `reviewer` | 검수자/검수 절차의 식별자. |
| `evidence` | `source_supported`이면 정답 근거 문서/행/파일 참조가 필수. |

`image_id`, `image_sha256`, `source_id`, `split`과 CSV 헤더 순서는 유지한다. 검수가 필요한 행을 필터링하여 채울 수 있다. 초기화가 필요하면 기존 CSV를 다른 이름으로 보존한 뒤 `review`를 실행한다.

기본 정책에서 높은 score의 teacher 출력은 train에만 들어간다. 예제 임계값은 최고 score 0.95 이상, 1·2위 score 차이 0.15 이상이며 **검증된 정확도 수치가 아니다**. 모델과 실제 사진에 맞게 검토하여 바꾼다. validation에는 검수한 자료가 필요하고, test에는 항상 검수/출처 근거가 있는 자료만 들어간다. `allow_validation: true`를 명시하면 pseudo validation을 허용하지만, 결과에 teacher 기반 검증이라는 사실이 기록되며 독립 성능 평가가 되지는 않는다.

```bash
python scripts/kiosk_terminal.py export
```

비어 있는 train 또는 validation은 차단된다. 검수한 test가 아직 없으면 학습 준비는 가능하지만 test 평가는 실행할 수 없다. 제외 사유와 실제 채택 개수가 출력된다. export는 사진+짧은 분류 답의 `medical_vqa`를 만들며, 근거가 없는 병변 위치·상세 소견·치료 설명을 자동 생성하지 않는다.

## 5. 먼저 짧게 학습하고, 새 이름으로 전체 학습

```bash
python scripts/kiosk_terminal.py prepare --name eye-pilot-01 --max-steps 10
python scripts/kiosk_terminal.py train --name eye-pilot-01
python scripts/kiosk_terminal.py reload --name eye-pilot-01
python scripts/kiosk_terminal.py results --name eye-pilot-01
```

`prepare`는 모델·revision·원본 adapter·자료 해시를 검사하고 새 학습 설정을 저장한다. 모델을 로드하거나 GPU를 시작하지 않는다. `train`이 사용자가 명시적으로 시작하는 실제 foreground 학습이다. 초기 validation, optimizer step, loss, checkpoint가 실제 엔진에서 기록된다.

기존 엔진의 실제 자료 smoke 실행은 최소 10 optimizer update가 필요하다. 예를 들어 accumulation이 16이면 10 update를 만들 수 있는 train 자료가 최소 145장 필요하다. 자료가 이보다 적으면 `--max-steps`를 생략한 전체 one-pass 실행으로 준비한다. 이는 작은 자료로 가능한 실행을 뜻하며 성능이 검증되었다는 의미는 아니다.

짧은 실행을 확인한 뒤 전체 데이터 한 번 통과 학습은 별도 이름으로 준비한다.

```bash
python scripts/kiosk_terminal.py prepare --name eye-full-01
python scripts/kiosk_terminal.py train --name eye-full-01
python scripts/kiosk_terminal.py reload --name eye-full-01
```

이 full run도 설정한 원본 `initialization_adapter`에서 시작한다. pilot의 optimizer 상태를 자동으로 이어받지 않는다. 같은 run 이름에 다른 데이터·설정·시작 모델을 덮어쓸 수 없다. 검토 라벨 또는 pseudo-label 채택 정책이 달라지면 다시 `export`하고 새 run 이름으로 준비한다. 원본 사진·source·group CSV·split 또는 라벨 생성에 사용한 teacher 설정이 달라지면 새 `work_dir`에서 `scan → label → review → export`를 진행한다. 이전 작업 폴더와 결과는 보존한다.

V6에서 이어온 경우에도 **새 run**은 native 학습 엔진의 재개 가능한 checkpoint와 validation loss에 따른 선택 규칙을 사용한다. 원본 V6 final-only run의 완료 기록과 체크포인트 선택 결과는 그대로 보존한다.

### 중단, 재개, 별도 터미널에서 모니터링

학습을 실행한 터미널에서 `Ctrl+C`를 한 번 누르면 해당 worker에 중단을 요청하고 저장·종료 처리를 기다린다. 첫 checkpoint 이전에 중단되면 재개할 checkpoint가 없을 수 있다. 메뉴도 worker가 종료되기 전에 돌아오지 않는다.

```bash
python scripts/kiosk_terminal.py resume --name eye-full-01
```

`resume`은 같은 run의 마지막 checkpoint에서 optimizer·scheduler·RNG를 복원한다. 데이터나 설정을 바꿔 재개하지 않는다. 이전 V6 최종 adapter를 새 자료로 추가 학습할 때는 `initialization_adapter`와 새 run을 사용한다.

다른 터미널에서는 다음을 실행한다.

```bash
python scripts/kiosk_terminal.py watch --name eye-full-01
```

`watch`에서 `Ctrl+C`는 조회만 종료한다. 저장된 `RUNNING` 표시는 과거 상태일 수 있으므로 살아 있는 process라는 의미로 표시하지 않는다. 실제 worker 터미널의 종료 상태와 결과 기록을 함께 확인한다.

## 6. 결과 저장, test 평가, 새 사진 실행

```bash
python scripts/kiosk_terminal.py results --name eye-full-01 --save --plot
python scripts/kiosk_terminal.py evaluate --name eye-full-01 --kind base
python scripts/kiosk_terminal.py evaluate --name eye-full-01 --kind tuned
python scripts/kiosk_terminal.py compare --name eye-full-01
```

`results --save --plot`은 실제 event에 기록된 step/loss만 사용해 `summary.json`, `training_loss.csv`, `training_loss.png`를 저장한다. optimizer 기록이 없으면 그래프를 만들지 않는다. 출력 위치는 `work_dir/runs/<name>/report/`다.

test 평가는 동일한 고정 test와 같은 generation 설정을 사용한다. `base`는 해당 사전학습 base 모델, `tuned`는 이번 run에서 validation으로 선택하고 reload를 확인한 checkpoint다. 기존 초기화 adapter만 따로 비교하는 평가 모드는 포함되어 있지 않다. 따라서 이 비교만으로 이전 의료 adapter 대비 개선을 주장할 수는 없다. loss 감소나 lexical 정답 일치율만으로 실제 진단 정확도가 입증되는 것도 아니다.

저장 모델로 별도의 사진 한 장을 실행한다.

```bash
python scripts/kiosk_terminal.py predict --name eye-full-01 \
  --source-id eye --image "/absolute/path/to/new-eye-photo.jpg"
```

새 사진에는 정답 라벨이 필요하지 않다. 기본 질문은 해당 source의 학습 질문과 같다. `--question "질문"`으로 바꿀 수 있지만 이번 학습의 supervision은 짧은 분류 항목이며, 자유로운 설명의 정확도까지 검증한 상태는 아니다. 예측 답변과 사용한 모델·사진 해시는 run 아래에 저장된다.

## 7. 생성되는 주요 파일

| 위치 (`work_dir` 기준) | 내용 |
| --- | --- |
| `inventory.jsonl`, `inventory_meta.json` | 사진 목록, hash, 고정 split/group |
| `predictions.jsonl`, `predictions_meta.json` | teacher score, 파일·모델·설정 출처 |
| `label_progress.json` | 라벨 생성 진행/완료/실패 집계 |
| `reviews.csv` | 사용자가 편집하는 검수 결과 |
| `datasets/<dataset_id>/` | 변경 불가 학습/검증/test 자료와 freeze |
| `latest_dataset.json` | 현재 export 결과 위치와 진실값 종류 |
| `runs/<name>/training.json`, `prepared.json` | 새 run 설정과 준비 기록 |
| `runs/<name>/output/` | native 학습 manifest/event/checkpoint/reload |
| `runs/<name>/evaluation/` | 고정 test protocol, base/tuned 예측·평가·비교 |
| `runs/<name>/predictions/<id>/` | 사진 한 장 실행 요청, 모델 답변과 해시 기록 |
| `runs/<name>/report/` | 결과 요약, loss CSV/PNG |

최신 사진과 실제 모델 가중치가 없는 개발 환경에서는 합성 이미지로 파일 처리·검증·명령 연결을 테스트한다. 이 검사와 사용자의 실제 모델 추론·GPU 학습·질환별 정확도 검증은 서로 다른 단계다.
