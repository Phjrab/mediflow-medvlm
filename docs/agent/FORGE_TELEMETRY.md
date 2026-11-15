# Forge용 Mediflow telemetry 후보

`mediflow.telemetry.v1`은 기존 Mediflow 실행 경로에 scalar 관측만 추가한다. Forge가 학습·queue·GPU 소유자가 되지 않는다. 신규 코드의 GPU 실행은 TRAINING_OWNER가 별도 고정 hash 요청을 승인한 뒤 수행한다. CPU 시험이나 historical import는 실제 GPU 검증을 뜻하지 않는다.

## 실행 및 소유권

`MEDVLM_TELEMETRY_ROOT`가 없으면 계측을 끈다. 설정하면 기존 `campaign_main_train`의 본학습과 `campaign_gpu_smoke`의 고정 20-update 경로가 `<root>/<campaign>/<run>/<attempt>/`에 기록한다. `MEDVLM_TELEMETRY_CAMPAIGN`은 안전한 논리 ID, `MEDVLM_TELEMETRY_CODE_SHA`는 실행 후보의 40자리 commit SHA다. `MEDVLM_TELEMETRY_FSYNC_SECONDS` 기본값은 5초다.

- coordinator: manifest/events/status/summary 단일 작성자. 저장·fresh reload의 원래 검증 뒤에만 각각 성공을 표시한다. 평가를 실행하지 않으므로 evaluation_completed는 null이다.
- training child: steps/resources 단일 작성자. coordinator는 flush된 step을 250ms마다 관측하여 실제 update 이벤트와 상태를 갱신한다. event timestamp는 관측 시각이다.
- 각 stream은 별도 증가 seq이며 새 attempt의 배타적 파일 생성으로 중복 작성자를 거부한다. metadata snapshot은 원자 교체한다. 기존 checkpoint/optimizer/RNG resume 기능을 추가하지 않는다.

이미 계산된 `float(loss.detach())`, clip 이전 gradient norm, optimizer update 이전 LR와 scheduler 이후 LR를 사용한다. 마지막 batch는 실제 13개/11개로 나눈다. forward/backward, RNG draw, optimizer/scheduler 순서 및 샘플 순서는 바뀌지 않는다. update wall time은 기존 GPU 동기화를 추가하지 않은 host 구간이며 GPU 완료시간을 뜻하지 않는다. 현재 프로세스 CUDA allocator 값은 이미 초기화된 context에서만 조회한다.

각 update는 local JSONL append+flush한다. fsync는 설정 주기와 close 때 수행한다. final checkpoint 시 child가 종료되면서 flush/fsync되고 coordinator가 metrics와 step 수/표본/loss/LR/gradient/target·sequence token 합을 대조한다. logging 오류는 학습에 전파하지 않으며 health sidecar와 status의 sanitized reason/gaps/telemetry_complete=false로 남긴다. 디스크 자체가 쓰기를 거부하면 sidecar 기록도 보장할 수 없으므로 staging의 telemetry.health.json과 파일 부재도 확인한다. 저디스크에서 연구 산출물을 삭제하지 않는다. health의 write_seconds는 append/flush/fsync 비용이며 외부 owner probe의 on/off wall 비교와 함께 본다.

resource sampler는 5초 주기 stdlib thread다. nvidia-smi timeout은 2초, 실패 backoff는 최대 40초이며 gap을 명시한다. torch/NVML import나 CUDA context를 생성하지 않는다. 장치 전체 utilization/memory/power/temperature/clocks와 프로세스 RSS·CPU/host RAM을 분리한다. 센서 실패는 null+reason이며 GPU 부재 판정이 아니다. CPU percent의 첫 표본은 interval이 없어 null이다. 전력 적분/에너지 및 과거 자원 시계열을 만들지 않는다.

## CPU historical import

아래 경로 변수는 운영자가 지정하는 private alias다. 원본을 변경하지 않고 별도의 integration-owned root를 사용한다. CLI는 raw prediction 파일을 열지 않는다. metrics.json의 scores 배열도 export하지 않는다.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m mediflow_medvlm.telemetry_import --output-root "$TELEMETRY_ROOT" import-history --data-root "$MEDVLM_DATA_ROOT"
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m mediflow_medvlm.telemetry_import --output-root "$TELEMETRY_ROOT" export-freeze --freeze-directory "$COMMON_FREEZE_DIR"
```

`import-run --directory ... --campaign-id ... --expected-hashes hashes.json`는 명시적인 원본 파일명→sha256 receipt도 검증한다. native completion의 지원 artifact hash와 evaluator run_manifest의 metrics hash를 검증하며 mismatch는 거부한다. 외부 기대 hash가 없으면 계산한 source hash만 보존하고 receipt_verified를 null로 둔다. source bundle/hash로 immutable attempt ID를 만들고 같은 import는 idempotent다.

V6 example_losses는 update_example_counts의 실제 누적 경계로 복원한다. 고정 native20×16 smoke만 기존 recipe 근거로 경계를 사용한다. per-example sequence token·step duration·historical GPU series가 원본에 없으면 null이다. peak memory와 model load/audit/train/save/final-sync 전체 wall은 summary에만 둔다. legacy optimizer event의 lr은 scheduler 이후 값이므로 lr_next로만 표시하고 lr_used는 null+reason이다. 실제 event timestamp는 보존한다.

V1/V2/V4 native aggregate의 분모·metric 이름·VQA subgroup을 보존한다. frozen protocol을 canonical hash로 연결하여 split/version과 데이터 hash identity를 보존한다. 원래 evaluator가 생성한 paired bootstrap delta/CI도 별도 historical comparison attempt로 보존하며 재채점하지 않는다. 비교 receipt의 출처만 계산했을 때 independent verification을 주장하지 않는다. V3에는 training/aggregate artifact가 발견되지 않았고 V5는 run receipt만 있어 score가 미실행/미제공으로 남는다. partial/unknown identity는 직접 비교 가능으로 간주하지 않는다.

common freeze catalog는 text7,997/VQA1,979개 native metadata 행을 그대로 분리 export한다. source/revision/sample/group/split/license/restrictions/derivation/review/freeze 및 실제 이미지 hash를 보존한다. 원문/정답/이미지 경로는 제외한다. raw QA JSON의 local_asset_sha256을 image_sha256으로 부르지 않는다. 원본 Mediflow 재분할·변환·image preprocessing을 수행하지 않는다.

## Owner의 bounded GPU 검증

CPU 준비는 기존 CPU eligibility receipt를 읽어 T1/V1의 동일한 20-update, 320-example smoke config만 새 private 위치에 생성한다. `source_files`는 telemetry.py와 telemetry_probe.py를 포함한다. native CPU preflight나 model audit 전체를 다시 실행하지 않는다.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$TRAIN_PYTHON" -m mediflow_medvlm.telemetry_probe prepare --cell-id T1 --preflight-receipt "$T1_PREFLIGHT_RECEIPT" --output-dir "$PROBE_ROOT/T1-on" --config "$PROBE_ROOT/T1-on.json" --run-id forge-probe-T1-on
```

TRAINING_OWNER 전용 launch는 다음과 같다. hash와 후보 commit을 요청 receipt에 고정하고, T1-off → T1-on → V1-on 순서로 세 번만 실행한다. off는 별도 config/output/run ID이며 동일한 T1 science/selection/source를 쓴다.

```bash
# TRAINING_OWNER only; never execute from integration automation.
env -u MEDVLM_TELEMETRY_ROOT PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$TRAIN_PYTHON" -m mediflow_medvlm.telemetry_probe owner-run --config "$T1_OFF_CONFIG" --receipt "$T1_OFF_RECEIPT" --expected-sha256 "$T1_OFF_CONFIG_SHA256"
MEDVLM_TELEMETRY_ROOT="$TELEMETRY_ROOT" MEDVLM_TELEMETRY_CAMPAIGN=forge-integration-probes-v1 MEDVLM_TELEMETRY_CODE_SHA="$CANDIDATE_SHA" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$TRAIN_PYTHON" -m mediflow_medvlm.telemetry_probe owner-run --config "$T1_ON_CONFIG" --receipt "$T1_ON_RECEIPT" --expected-sha256 "$T1_ON_CONFIG_SHA256"
# Repeat the on command once for the pinned V1 config/receipt/hash.
```

helper는 기존 worker.lock과 medvlm_gpu.lock만 O_RDWR|O_NOFOLLOW로 연다. 파일 생성·별도 lock·새 supervisor·queue/state/schedule 변경이 없으며 동일 inode의 lease FD를 기존 train/reload child에 상속한다. 잠금 경합은 즉시 실패한다. 기존 supervisor progress env를 제거하여 old run의 progress를 덮어쓰지 않는다. GPU owner 검증에서는 두 개 이상의 실제 다른 step/loss, refresh 유지, checkpoint/fresh reload 분리, summary reconciliation, write→UI latency 및 on/off wall 차이를 측정한다. 캐시/warmup과 GPU 비결정성 때문에 차이를 순수 logging 비용이나 완전 동일 GPU 수치라고 주장하지 않는다.

원복은 신규 candidate 실행을 사용하지 않고 기존 immutable snapshot을 유지하는 것이다. 기존 history·checkpoint·data·receipt를 지우지 않는다. 계측 root 환경변수를 제거하면 신규 worker의 계측이 꺼진다. 실행 중 worker snapshot을 교체하지 않는다.

## CPU 검증

기존 training venv Python을 읽기 전용으로 사용하고 `CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ... -m pytest -p no:cacheprovider`로 tests/test_telemetry.py, test_telemetry_import.py, test_telemetry_smoke.py 및 기존 campaign_main_train/campaign_gpu_smoke 시험을 실행한다. 본학습 CPU on/off의 weights·optimizer state·RNG·loss·순서 exact equality, 13/11 partial divisor, 기존20-update smoke hook, stream health·resource null·privacy·receipt hash·large metadata를 확인한다. 이는 GPU/browser/배포 성공 증거를 대체하지 않는다.
