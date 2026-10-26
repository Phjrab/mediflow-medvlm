# 실행 인계

## 현재 후속 작업 — 2026-10-26

사용자 후속 지시로 v2 작업을 진행 중이다. 아래 상태는 중간 인계이며 최종 완료 실적이 아니다. 기존 `runtime/medvlm_state.json`과 `reports/public_summary/{RESULTS_KO.md,run_summary.json,stage_status.json}`은 **v1의 역사 상태**다. 후속 전체 작업의 최종 root 상태는 아직 작성 전이며 v1 상태 파일로 v2 완료 여부를 판단하지 않는다. 새 결과는 `reports/public_summary/v2/`에 구분한다.

- 현재 실행 정책: `configs/resource_policy.json`의 `budget_mode=unlimited`. `total_gpu_hours`, `per_model_gpu_hours`, `run_wall_seconds`, `job_timeout_seconds`, `campaign_wall_seconds`는 모두 `null`이다. `resolve_resource_policy()`가 해석한 현재 정책을 따른다. 이전 config/checkpoint의 24/6 제한을 되살리지 않으며 원장·model/run/phase/session 사용량과 checkpoint floor는 보존한다. 네트워크 요청 timeout·교착 진단·사용자 중단 처리는 유지한다.
- Jetson: `USER_DEFERRED`. 10/26 사용자 요청으로 접속·호스트키 확인·설치·변환·장치 추론을 범위에서 제외했다. 10/23 호스트키 미검증 이력은 아래에 보존하되 현재 재개 선행 조건이나 사용자 할 일로 취급하지 않는다.
- AI-Hub 71875: 실제 준비 완료 canonical 17,210개(train 15,297 / validation 956 / test 957). 사용자 제공 원본 경로를 사용하므로 API 키를 추가 필수 조건으로 만들지 않는다. 실제 수량·권리·제외·분할 근거는 v2 import manifest와 공개 요약을 따른다.
- AI-Hub 71762: 전달 파일 약 4.25GB에 대해 connector의 파일당 256MiB 제한이 있었고 브라우저 다운로드는 `ERR_BLOCKED_BY_CLIENT`로 차단됐다. 사용자의 로컬 원본 경로 응답을 기다린다. 승인 또는 인증 누락으로 단정하지 않고 경계를 우회하지 않는다. 원본 확보와 명시 pairing 근거가 없는 단계는 완료로 표시하지 않는다.
- MedMCQA: 공식 팀 MIT 답변과 원저자 ZIP을 확인해 원본 193,155행을 확보·감사했다. 선택적 영어 canonical train 20,000 / validation 2,816 / test 0을 별도 준비했다. KO/MIX 주 corpus와 기존 freeze에 넣지 않았으며 학습하지 않았다.
- 한국어 텍스트 실험: KO 4,000개(객관식1,500/단답1,300/서술1,200), MIX 같은 KO4,000+EN4,000과 공통 dev1,516을 동결했다. KorMedMCQA4,068 보호를 위한 lexical/semantic 후보 감사와 실제 native token 검사를 완료했다. KO250 update/243,394 target tokens, MIX500 update/1,137,306 target tokens의 한 pass이며 원본 Base에서 시작한다. 20-update smoke·새 process reload 후 본 실행, 동일 dev48개로 checkpoint를 선택한다. **구현 체크포인트 시점에는 학습은 VQA 종료를 기다리는 큐에 있으며 아직 완료하지 않았다.** 상태는 `v2/text/state.json`, 대기 프로세스는 `v2/text/queued-launch.json`, 불변 계획은 `v2/text/plan.json`이다.
- VQA v2: Qwen3.5-2B/MedGemma의 기존 Base·선정 v1 adapter를 재사용하며 새 VLM 학습은 하지 않는다. validation8개 실행으로 짧은 답변 형식과256-token 상한을 고정했고 전체SLAKE1061/VQA-RAD451 및 no-image32 평가를 진행 중이다. 상태는 `$MEDVLM_DATA_ROOT/v2/vqa/state.json`, 선언은 같은 폴더의 `plan.json`, 결정은 `validation-decision.json`, 로그·예측은 같은 v2 폴더에 둔다. 기존 v1 test를 새로운 blind test라고 부르지 않는다.

저장소 최상위에서 기존 환경과 data root를 사용한다. 먼저 현재 소유 worker/PID와 v2 상태·로그를 확인하고 **실행 중이면 두 번째 worker를 시작하지 않는다.** 다음은 진행 중 작업이 종료되거나 중단된 뒤의 재개 명령이다.

```bash
export MEDVLM_DATA_ROOT="$(realpath ../medvlm-data)"
export PYTHONPATH="$PWD/src"
export WANDB_DISABLED=true
# validation까지만 수행 또는 성공 결과 재사용
../.venv-medvlm/bin/python scripts/run_vqa_v2.py --data-root "$MEDVLM_DATA_ROOT" --validation-only
# 고정 validation 결정 후 선언된 SLAKE/VQA-RAD·이미지 제외 평가까지 계속
../.venv-medvlm/bin/python scripts/run_vqa_v2.py --data-root "$MEDVLM_DATA_ROOT"
# VQA와 대기 프로세스 종료 여부를 먼저 확인한 뒤 텍스트 행렬 재개
../.venv-medvlm/bin/python scripts/run_text_v2.py --data-root "$MEDVLM_DATA_ROOT" --phase all
# 완료 artifact만 검증·집계. 원래 v1 보고서는 재생성하지 않는다.
../.venv-medvlm/bin/python scripts/report_v2.py --data-root "$MEDVLM_DATA_ROOT" --output reports/public_summary/v2
# 기존 v1 예측의 CPU 재분석만 재생성
../.venv-medvlm/bin/python scripts/analyze_v1_followup.py --data-root "$MEDVLM_DATA_ROOT" --output reports/public_summary/v2
```

사용량 원장은 `$MEDVLM_DATA_ROOT/runtime/gpu_budget.json`이며 현재 정책과 별개로 보존한다. v1 freeze·checkpoint·source snapshot·prediction·config·기존 결과표를 덮어쓰지 않는다. 과거 미완료 run을 재개해야 한다면 검증된 resource-policy-only migration을 사용하며 identity 검사를 끄지 않는다. 정책 변경만으로 완료한 네 모델을 다시 학습하지 않는다. fallback 불필요 후보를 새 필수 셀로 만들지 않는다.

현재 사용자가 제공해야 할 입력은 **71762의 접근 가능한 로컬 원본 경로(및 제공 가능한 가이드·전처리 코드)**다. 이미 허용된 GPU 실행이나 MedGemma 접근, 제외한 Jetson 확인을 다시 요청하지 않는다. 새 실적과 최종 검증·commit/push 결과는 후속 통합 보고에서 갱신한다.

## 보존된 2026-10-23 v1 인계

다음 내용은 v1 종료 당시의 기록과 명령이다. 이후 갱신된 GPU 정책·Jetson 범위·AI-Hub/MedMCQA 상태는 위 10/26 항목이 우선한다. 아래의 v1 재학습·보고서 재생성·장치 명령은 현재 후속 작업을 시작하라는 지시가 아니다.

이번 v1 실행의 상태는 `runtime/medvlm_state.json`, 실제 근거는 Git 밖의 `MEDVLM_DATA_ROOT`에 있다. 최종 집계는 `reports/public_summary/RESULTS_KO.md`와 `run_summary.json`이다. 이 문서는 실행 가능한 명령과 보존 조건을 기록한다. 실행 중에는 상태 파일의 현재 프로세스를 먼저 확인한다.

최종 v1은 text2+VLM2의 실제 학습·새 process 재로드·주 paired 평가4개를 완료했다. Qwen3.5와 MedGemma의 VQA-RAD64건 paired 및 이미지 제외 진단도 완료했다. 남은 필수 GPU run은 없다. 예비 후보는 fallback 불필요로 미실행, Jetson·AIHub·외부 원문 권리와 전문가 검토는 각각 별도 상태로 남긴다. `stage_status.json`은 최종 단계 근거, 전달 사본의 `git_receipt.json`은 최종 commit/push/CI 근거다.

## 환경·데이터 위치

저장소 기준으로 전용 Python3.12 환경은 `../.venv-medvlm`, 데이터 루트는 `../medvlm-data`다. Python3.10 경량 검사 환경은 `../.venv-core310`다. 로컬 절대경로는 사용자에게 전달하는 별도 인계 사본과 Git 제외 상태에 기록한다.

```bash
export MEDVLM_DATA_ROOT="$(realpath ../medvlm-data)"
export PYTHONPATH="$PWD/src"
export WANDB_DISABLED=true
../.venv-medvlm/bin/python -m mediflow_medvlm status
../.venv-medvlm/bin/python -m mediflow_medvlm doctor --output "$MEDVLM_DATA_ROOT/doctor-recheck.json"
../.venv-core310/bin/python -m pytest
```

GPU와 네트워크는 이 desktop의 일반 sandbox에서 보이지 않았으며 승인된 실행 환경에서는 작동했다. `nvidia-smi`의 NVML 불일치를 GPU 부재로 간주하지 말고 doctor의 실제 forward/backward 증거를 확인한다. 시스템 드라이버를 변경하지 않는다. 가중치와 x86 환경을 Jetson에 그대로 복사하지 않는다.

학습 core freeze ID는 `1bf58dbc8baca7a3f5865accf5a86e9fba0b84f8abe3c986ed34b67d6c3d87c5`이고 파일 SHA256은 `af46823da1e4f4408480de16634460c20d2cc2b9c04b54583d20ebb1a875b934`다. 외부 VQA-RAD를 추가한 aggregate freeze ID는 `1562258c4b5b6f3b4c5c8948851fbb972da603cd63bff528aa7d278cf2253c86`이다. core 학습 파일을 aggregate 구축 과정에서 덮어쓰지 않았다. 모든 train/dev/test의 immutable JSONL은 `frozen/<split-sha256>/<split>.jsonl`이다.

## 재개 순서

1. 상태 및 실제 소유 PID를 확인한다. 동일 campaign을 중복 실행하지 않는다.
2. 남은 모델만 지정하거나 기본 유한 campaign을 실행한다. 검증된 성공 학습·재로드·예측은 재사용하며 실패를 덮어쓰지 않는다.
3. 최종 집계를 다시 생성한다. raw question/answer/prediction을 Git에 추가하지 않는다.

```bash
../.venv-medvlm/bin/python scripts/complete_v1.py --data-root "$MEDVLM_DATA_ROOT"
../.venv-medvlm/bin/python scripts/report_v1.py --data-root "$MEDVLM_DATA_ROOT" --output reports/public_summary --plot
```

기본 후보 순서는 Qwen3-1.7B, Qwen3-4B, Qwen3.5-2B, MedGemma1.5-4B다. 일반 VLM은 Qwen3.5-2B 실패 시 Qwen3-VL-2B→Qwen2.5-VL-3B로 진행한다. 의료 VLM은 MedGemma 실패 시 Gemma3-4B로 진행한다. SmolVLM500M은 아직 완료하지 못한 VLM 셀의 마지막 대안이며 두 모델로 중복 계산하지 않는다. Qwen2.5-VL-3B는 비상업 연구용 Research License이며 Apache 2.0으로 간주하지 않는다. 앞 후보가 성공하면 fallback은 `PREPARED_NOT_RUN / FALLBACK_NOT_NEEDED`다. 사용자 새 Hugging Face 계정으로 MedGemma 및 Gemma3 config 접근을 실제 확인했다. 기본 로그인 캐시를 읽은 뒤 모델 artifact 캐시를 분리한다. 토큰을 설정 파일·명령 인자·Git에 적지 않는다.

개별 실행의 실제 config는 `configs/<model-key>-<smoke|v1>-20261023.json`이다. config와 source/data/model hash가 바뀌면 같은 run을 재개하지 말고 별도 run_id를 사용한다. smoke20 실제 update와 fresh-process reload를 먼저 통과해야 본 학습으로 진행한다.

```bash
../.venv-medvlm/bin/python -m mediflow_medvlm.training train --config "$MEDVLM_DATA_ROOT/configs/qwen3_4b-v1-20261023.json"
../.venv-medvlm/bin/python -m mediflow_medvlm.training train --config <IMMUTABLE_RUN_CONFIG> --resume <COMPLETE_CHECKPOINT>
../.venv-medvlm/bin/python -m mediflow_medvlm.training reload --config <IMMUTABLE_RUN_CONFIG> --checkpoint <SELECTED_CHECKPOINT>
```

`<...>`는 각 run의 `latest.json` 또는 `selection.json`에 있는 값으로 채운다. 최초 Qwen3-1.7B는 초기 training code로 시작했으며 원래 7개 training/runtime 소스와 hash receipt가 run의 `source_snapshot/`에 보존됐다. 새 checkpoint identity로 이전 미완료 run을 이관할 때는 검증된 원본 소스가 필요하다. 이전 미완료 run에 한해 다음 명령을 사용한다.

```bash
../.venv-medvlm/bin/python -m mediflow_medvlm.training train --config <IMMUTABLE_RUN_CONFIG> --resume <COMPLETE_LEGACY_CHECKPOINT> --migrate-legacy-from <RUN_DIRECTORY>/source_snapshot
```

예산 원장은 `$MEDVLM_DATA_ROOT/runtime/gpu_budget.json`이다. checkpoint, source snapshot, config, freeze, 원장을 함께 보존한다. 모델 ID별6 GPU시간·전체24 GPU시간을 재개로 초기화하지 않는다. 신규 checkpoint의 예산 floor는 기존 원장과 max 병합하며 중복 합산하지 않는다. 강제 종료는 마지막 완전 checkpoint 이후 진행을 잃을 수 있고, stale active 예약은 보수적으로 과금된다.

## 평가와 원문 보관

평가 protocol은 `manifests/evaluation/*-v1.json`, 실제 예측은 `runs/<model-key>-test-<base|tuned>-20261023/predictions.jsonl`, 집계는 같은 폴더의 `metrics.json`이다. raw 예측은 공개 자료 기반이지만 제한된 로컬 artifact로만 유지한다.

```bash
../.venv-medvlm/bin/python -m mediflow_medvlm.evaluation evaluate --config <RUN_CONFIG> --protocol <FROZEN_PROTOCOL> --output-dir <BASE_RUN>
../.venv-medvlm/bin/python -m mediflow_medvlm.evaluation evaluate --config <RUN_CONFIG> --protocol <FROZEN_PROTOCOL> --output-dir <TUNED_RUN> --adapter <SELECTED_CHECKPOINT>
../.venv-medvlm/bin/python -m mediflow_medvlm.evaluation compare --base <BASE_RUN> --tuned <TUNED_RUN> --output <PAIRED_JSON>
```

원래 성공 artifact의 hash가 달라졌으면 기존 결과를 성공으로 재사용하지 않는다. 구현 오류 수정으로 재실행할 때는 이전 attempt를 보존하고 새 protocol/run identity로 두 조건을 함께 실행한다. 나쁜 점수를 이유로 test 문항이나 parser를 바꾸지 않는다. 전문가 검토 packet은 `review/<model-key>/`에만 생성하며 점수는 비어 있다. 검토를 완료하지 않았다면 `human_review=not_performed`를 유지한다.

## 데이터·장치 재개

소스별 데이터 명령과 권리는 `docs/research/DATA_PREPARATION.md`, `VISION_SOURCES.md`, `DATA_REPORT_KO.md`에 있다. 기존 원본과 freeze가 있으면 불필요하게 재구축하지 않는다.

```bash
../.venv-medvlm/bin/python -m mediflow_medvlm.data audit --data-root "$MEDVLM_DATA_ROOT"
../.venv-medvlm/bin/python -m mediflow_medvlm device inventory --output "$MEDVLM_DATA_ROOT/runs/device/new-inventory"
```

Jetson은 신뢰할 SSH 호스트 키가 등록되지 않아 접속을 보류했다. 관리자가 별도 경로로 장치 fingerprint를 확인하고 known_hosts에 신뢰 키를 등록한 뒤 inventory를 재개한다. 호스트 검증을 끄거나 키를 자동 신뢰하지 않는다. AIHub는 필요한 로그인·이용 자격과 공식 페어링 구조가 확인될 때만 재개한다. MedQA/MedMCQA는 외부 원문 파일에 적용되는 권리가 확인되어야 한다.

## Git 및 해석 범위

전용 branch `codex/medical-finetune-v1-20261023`만 사용한다. main merge, 서비스 배포, 모델 Hub 업로드는 수행하지 않는다. commit 및 remote push의 실제 결과는 최종 보고에 별도 기록한다.

모든 결과는 단일 seed 예비 연구다. PubMedQA/KorMedMCQA 정답률, 자유서술 lexical score, 이미지 의존 진단, 속도·메모리를 각각 해석한다. 전문가 검토·임상 검증·전력/TTFT 측정은 수행하지 않은 경우 수치나 완료 주장으로 채우지 않는다.
