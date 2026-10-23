# 실행 인계

이번 v1 실행의 상태는 `runtime/medvlm_state.json`, 실제 근거는 Git 밖의 `MEDVLM_DATA_ROOT`에 있다. 최종 집계는 `reports/public_summary/RESULTS_KO.md`와 `run_summary.json`이다. 이 문서는 실행 가능한 명령과 보존 조건을 기록한다. 실행 중에는 상태 파일의 현재 프로세스를 먼저 확인한다.

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

학습 core freeze ID는 `1bf58dbc8baca7a3f5865accf5a86e9fba0b84f8abe3c986ed34b67d6c3d87c5`이고 파일 SHA256은 `af46823da1e4f4408480de16634460c20d2cc2b9c04b54583d20ebb1a875b934`다. 외부 VQA-RAD를 추가한 aggregate freeze ID는 `4dc1e59694d90858b3c3bd81f0963ec798aba09180b014308e789471422e7b07`이다. core 학습 파일을 aggregate 구축 과정에서 덮어쓰지 않았다. 모든 train/dev/test의 immutable JSONL은 `frozen/<split-sha256>/<split>.jsonl`이다.

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
