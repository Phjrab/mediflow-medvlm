# 실행 인계

## 현재 V6/V7 — 2026-11-14 18:36 KST

고유8개 모델의 준비가 모두 성공했고, 다음 native CPU train 적격성 검사10개를 단일 supervisor에 등록했다. GPU smoke10개와 공통 freeze·본학습·평가는 아직 미시작이다. 새 source2ff65d4, 활성 manifest5c6d818f, 기존 실행·누적 실패 보존 및 다음 작업은 [V6/V7 전용 인계](V6_V7_HANDOFF_KO.md)를 따른다. 네이티브20분 예약 `medvlm-v6-v7`은 ACTIVE이며 정상 worker를 한 번 확인한 뒤 종료한다. 이 상태를 아래 역사 결과와 혼동하지 않는다.

> 현재 활성 범위(2026-11-14): [V6/V7 R2 감독 인계](V6_V7_HANDOFF_KO.md). 아래 v1~v5는 역사 기록이며 현재 queue·실행·상한을 대신하지 않는다.

## 현재 v5 후속 진단 — 2026-11-13 완료

사용자의 다음 단계 진행 요청으로 고정한 검증 전용384기록을 실제 GPU에서 모두 실행했다. 두 모델 Base/v4 Tuned ×기존/한국어 번역 형식 지시 ×16/256token ×24검증문항(세 source 각8개, KorMed 직군별2개)이다. test ID·정확 입력 제외, 최대입력562token, v4 train 정확 input 겹침0을 사전 확인했다. 새 학습·test 추론·기존 parser/점수 보정은 하지 않았다.

모든 worker가 성공했고 session9573은 종료, 전역 GPU active는 null이다. 기술실패0·기존 파일208개 hash 보존·전체 재채점·Base/Tuned native 가중치/template/dtype/prompt·greedy prefix192/192쌍을 검증했다. GPU 추가0.272409시간이다. source는commit3ea0fe8의9개 구현 파일과 byte hash가 일치한다. CPU255passed/7skipped/23subtests·6기존경고,focused4passed다.

두 Tuned는 네 지시/상한 조건 모두 유효 형식0/24였다. Base 기존지시 조건은1.7B22/24,4B21/24였다. 상한256 확대는 길이 종료를 줄였지만 형식 준수를 회복시키지 못했고, 한국어 번역의 효과도 모델마다 달랐다. 이는 작은 기존 validation의 진단이며 임상 능력·새 test 성능·무작위 QA 연결의 단독 인과 효과를 측정한 결과가 아니다. 기존 v4는 약한 참조 실험으로 유지하며 test에 맞춰 parser·prompt·seed·checkpoint를 조정하지 않는다. 성능에 따라 추가 epoch나 sweep을 자동 실행하지 않는다.

공개 결과는 `reports/public_summary/v5/READOUT_KO.md`, `RESULTS_KO.md`, `run_summary.json`, `verification.json`이다. 전체 v1~v5 공개 결과는 `reports/public_summary/INDEX_KO.md`로 연결된다. 코드·집계·그림은 전용 작업 브랜치에 commit/push하며 main merge·배포는 하지 않았다. 원본·가중치·전체 예측·A/B 키는 로컬 private data root에 있으며 GitHub만 복제하면 이 자료까지 복구되지 않는다. 사용자 전달 사본과 재개 명령·최종 CI 영수증은 workspace `outputs/v5/`에 기록한다. 전문의 검토는0건이며 기존 빈 검토표는 `outputs/v4/review/`다.

## 현재 후속 작업 — 2026-11-13 v4 완료

사용자가 2026-11-12 “71762는 제공처 방식으로 별도 실험 진행”을 명시했다. 이는 71762에 한해 카테고리·질환명·의도 안의 무작위 답변 연결 및 미제공 증강 lineage 한계를 허용한 예외다. 다른 source의 연결 기준은 바꾸지 않는다. 제공처 회신은 사용자 전달문이며 원본 메일 헤더를 인증한 것으로 표시하지 않는다. 공식 `make_train_data.py`는 미확보이며 제공처가 설명한 방식을 독립 구현했다. 정확한 고정 QA 매핑·parent 파일 확보를 현재 사용자 선행 작업으로 다시 요구하지 않는다.

2026-11-13 01:55:56 KST에 선언한 유한 v4 실험이 `SUCCEEDED`로 끝났고 전체 산출물 검증도 통과했다. 실행 소유 session79763이 종료됐으며 GPU 원장 active는 null이다. 원본 Base에서 독립 학습한 Qwen3-1.7B/4B 각각20update smoke·250update/1pass 본학습과 각 새 process 재로드를 완료했다. 두 본학습 모두4,000표본/정답1,093,637token을 처리했고 validation으로step250을 선택했다. test 선택·기존 EN/KO/MIX adapter 연속학습은 하지 않았다.

- 데이터: 학습4,000/검증105/파생평가256, 19질환 분류. 공식 Validation에서 hash로 dev/test를 파생했으며 공식 Test라고 부르지 않는다. KorMedMCQA 전체4,068 포함 고유5,589개 보호자료와 lexical/E5 의미 유사도를 검사했고 후보2,982개를 보수적으로 격리했다. 확정 의학/번역 중복 판정이나 augmentation parent 복원은 아니다. 동결 표본 source text·답변 section·cross-split exact hash를 재검증했다.
- 실행: 학습·재로드4단계, 새 평가6개/생성기록10,084건(조건별 반복 포함), 기존 공통 Base2개 검증. 새 공통 평가는 모델별4,530개, 무작위 참조는 Base/Tuned 각256개다. 모든 pair의 input/native prompt token hash와 생성 설정이 일치한다. 기술 실패0이며 기존 공개 보고서·선별 역사 파일77개 hash를 보존했다.
- 결과: 무작위 참조 lexical F1은1.7B 0.0864→0.0914,4B 0.1039→0.1148이다. 임상 개선 근거가 아니다. 양 모델 공통 객관식·분류형4,286개가 모두 invalid_format이며 KorMedMCQA0/3009는 형식 실패를 포함한 엄격한 점수다. 의학적 능력0이라고 해석하거나 parser·상한·프롬프트를 사후 변경하지 않는다. 한국어 자유서술 전용 학습·문체·길이·token 노출 차이로 무작위 연결 방식의 인과 효과를 분리할 수 없다.
- 검증: CPU251passed·7선택실행 제외·23subtests passed,6기존 Pillow 경고. 실제 GPU doctor의 forward/backward와4학습/fresh reload는 CPU 선택 검사와 별도의 근거다. GPU 추가7.644시간이며 unlimited/null 정책·누적 사용 원장·checkpoint floor는 보존했다.
- 미실행/한계: 공식 코드 실행, 전체 원본 학습, 추가seed, 새 MedGemma/VLM 학습, 전문가 임상 검토는 하지 않았다. 검토표2개·32비교행은 같은16문항이며 사람 점수는 모두 빈칸이다. Jetson은 USER_DEFERRED다. 실행 차단은 없으나 원본 parent·정확 QA 대응이 없으므로 lineage 격리와 임상 정답성을 보장하지 않는다.

공개 aggregate는 `reports/public_summary/v4/RESULTS_KO.md`, `run_summary.json`, `verification.json`, `figures/`다. 원본·모델·전체 예측·A/B 키는 private data root에 보존한다. 사용자 전달물은 workspace `outputs/v4/`이며 실제 checkpoint 경로는 `local_artifacts.json`, 재개 명령은 `RESUME_KO.md`, 최종 commit/push/CI 근거는 `git_receipt.json`에 기록한다. 아래 v1/v2/v3는 당시 기록으로 보존하며 현재 완료 상태를 대신하지 않는다.

재개는 기존 성공 산출물의 identity·hash를 검증 후 재사용한다. 이전 freeze·결과에 덮어쓰거나 결과에 따른 재추첨·추가seed를 자동 실행하지 않는다.

```bash
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_DISABLED=true
../.venv-medvlm/bin/python scripts/run_71762_v4.py --data-root ../medvlm-data --phase all
../.venv-medvlm/bin/python scripts/report_71762_v4.py --data-root ../medvlm-data --output reports/public_summary/v4
```

## 보존된 2026-10-28 인계 — 71762 로컬 원본

사용자가 두 로컬 ZIP을 제공했다. `120.초거대AI 사전학습용 헬스케어 질의응답 데이터.zip`이 71762, `09.필수의료 의학지식 데이터.zip`은 기존 71875와 동일 SHA다. 71762의 이전 `BLOCKED_DELIVERY`는 해소됐다. 현재 차단은 `UNRESOLVED_PAIRING`이며 원본 다운로드 실패나 API 키 누락이 아니다.

- 71762 4,248,340,940bytes, SHA `89584900411ac869817bd85c190ad4842f5839e9e0dc1fb85140113a148e99e4`. 내부 5 ZIP·7,028,863 파일 CRC/SHA 감사 완료. Training 질문1,143,131/답변1,966,902, Validation 질문141,292/답변245,404의 별도 구성요소3,496,729개를 정규화했다. TS/TL·VS/VL 내용은 같은 경로에서 전부 같고 중복 합산하지 않았다.
- private SQLite에 출처와 question 또는 answer 구조를 보존했다. 원본·split을 유지했고 source 문서를 QA로 만들지 않았다. 답변 ID 반복281,241그룹과 공식 split 간 동일 질문10,464그룹/28,930행을 관측했다. 전역 ID 고유성·증강 parent를 추정하지 않는다.
- 공식 HWP 설명서·가이드를 정상 브라우저 링크로 확보하고 본문을 읽었다. 가이드는 카테고리·의도 연결을 설명하나 사용자 지시는 이 방식의 join을 제외했다. 실제 Q→A foreign key·증강 parent 계약은 확인되지 않았다. QA쌍0, 새freeze·학습·평가는 미실행이다. 기존 v2 실험을 재실행하지 않았다.
- KorMedMCQA4,068개와 질문 본문의 정확 일치0. near/semantic·번역 감사는 paired train 후보가 생긴 뒤 필요하며 이번에는 미실행이다. 구성요소 문법·필드 검사를 임상적 검증이나 학습 적격성 확인으로 해석하지 않는다.
- 핵심 CPU 회귀241passed/7skipped/23subtests, 관련데이터21passed. v1/v2 공개 보고서38개·v2 실행소스14개·GPU원장 보존 검증 완료. 새 GPU 실행은 없고 unlimited 정책·Jetson USER_DEFERRED·human_review not_performed를 유지한다.

현재 보고서는 `reports/public_summary/v3/RESULTS_KO.md`, 실제 입력·감사·SQLite는 global data root의 `v3/`다. 데이터 처리 및 재개 명령은 `docs/research/DATA_AI_HUB_71762_KO.md`와 전달 폴더 `outputs/v3/RESUME_KO.md`에 있다. 기존 v2 결과/전달물은 당시 상태로 보존한다. 새 감사의 종료코드0은 component audit 성공이며 paired SFT 준비 완료가 아니다. 동일 audit를 동시에 실행하지 않는다.

사용자에게 필요한 입력은 71762 공식 소스·전처리 코드 또는 scoped 질문ID↔답변ID 및 증강원본parent 매핑 파일이다. 사용자가 다른 컴퓨터의 공식 페이지 ‘활용 AI 모델 및 코드’에서 파일을 확인할 수 있도록 안내했다. 이미 받은 ZIP·설명서·활용가이드·API 키·모델 가중치를 다시 요구하지 않는다. 자료가 생기면 실제 cardinality와 누수·token 검사를 거친 새 freeze로 이어가고 완료된 v2 corpus에 덧붙이지 않는다. 최종 commit/push/CI 근거는 `outputs/v3/git_receipt.json`에 별도 기록한다.

## 보존된 v2 종료 기록 — 2026-10-27 06:46 KST

실행 가능한 v2 선언 행렬은 완료했다. 전체 상태는 `COMPLETED_AVAILABLE_WORK_WITH_EXTERNAL_BLOCKER`이며 AI-Hub 71762는 `BLOCKED_DELIVERY`다. `reports/public_summary/v2/READOUT_KO.md`가 핵심 해석, `RESULTS_KO.md`가 전체 결과, `verification.json`이 실제 산출물 검증 근거다. `runtime/medvlm_state.json`은 변경하지 않은 v1 역사 상태이고, v2 인계는 `runtime/medvlm_state_v2.json`에 별도로 둔다.

- 현재 resource policy는 unlimited이며 total/model/run/job/campaign 상한은 모두 null이다. 사용량 원장·재개 identity·checkpoint floor는 보존했다. 시간 상한 없음이 추가 무한 실험을 뜻하지 않는다.
- 71875 실제 canonical 17,210개(train15,297/dev956/test957), KO4,000/MIX8,000, 공통 dev1,516을 준비·동결했다. KorMedMCQA 전체 dev/test4,068 보호를 위해 lexical/semantic 후보를 검사했고 후보 train/dev3,208행을 보수적으로 제외했다. 후보를 확정 중복이라고 부르지 않는다.
- 네 KO/MIX 본학습과 네 smoke 모두 성공했고 새 process 재로드를 통과했다. 원본 Base에서 독립 학습했으며 Qwen3-4B KO는 full250 update 중 validation step200, 나머지는 최종 step250/500을 선택했다. 텍스트9조건×4,530문항을 완료했다. MedGemma는 adapter 없는 native text Base다.
- 기존 VLM 가중치로 VQA validation8+본평가·진단12를 완료했다. VLM 재학습은 하지 않았다. MedGemma의 SLAKE dataset-level 사전학습 노출과 정확한 test 노출 미상 한계를 유지한다.
- MedMCQA는 공식 MIT 근거로 원본193,155행 확보·감사, optional canonical22,816행을 준비했다. 주 KO/MIX에는 미포함이고 추가 학습하지 않았다. v1 예측의 형식/길이/OPEN-CLOSED/8,450→8,400 재분석은 완료했다.
- 익명 검토표4개·64비교행은 동일16문항이며 사람의 검토는0건이다. 전달 사본만 편집하고 private key는 별도 보존한다. 추가seed는 NOT_SELECTED, 예비모델은 fallback 불필요다.
- 71762 약4.25GB 파일은 링크를 받았지만 connector256MiB 한도와 브라우저 ERR_BLOCKED_BY_CLIENT로 원본0bytes다. 실제 pairing은 UNRESOLVED_PAIRING이다. 필요한 외부 입력은 로컬 원본 경로와 제공 가능한 가이드다. 이미 받은71875에 API 키를 요구하지 않는다.
- Jetson은 USER_DEFERRED다. 과거 호스트키 미검증 이력은 아래에 보존하며 현재 사용자 할 일로 요청하지 않는다.

완료 평가29개(validation포함), 학습·재로드8개, 검토표4개를 검증했다. CPU234passed·7skipped·23subtests와 별도 native41passed·20subtests를 구분한다. v1 공개 파일18개와 실행 소스 스냅샷14개의 보존을 재확인했다. 실행 소유 worker가 종료됐으며 재개 전에는 상태와 실제 PID를 다시 확인한다. 공개 결과는 원문·예측·가중치·비밀값을 포함하지 않는다.

저장소에서 다음 명령을 사용한다. 성공 결과는 검증·재사용하며 v1 보고서는 재생성하지 않는다.

```bash
export MEDVLM_DATA_ROOT="$(realpath ../medvlm-data)"
export PYTHONPATH="$PWD/src"
export WANDB_DISABLED=true
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
../.venv-medvlm/bin/python scripts/run_vqa_v2.py --data-root "$MEDVLM_DATA_ROOT"
../.venv-medvlm/bin/python scripts/run_text_v2.py --data-root "$MEDVLM_DATA_ROOT" --phase all
../.venv-medvlm/bin/python scripts/report_v2.py --data-root "$MEDVLM_DATA_ROOT" --output reports/public_summary/v2
MPLCONFIGDIR=/tmp/medvlm-v2-mpl-cache ../.venv-medvlm/bin/python scripts/plot_v2.py --summary reports/public_summary/v2/run_summary.json --output reports/public_summary/v2/figures
```

전용 작업 branch `codex/medical-finetune-v1-20261023`만 사용한다. 최종 commit/push/CI 영수증은 작업공간의 `outputs/v2/git_receipt.json`에 별도 기록하여 commit 자기참조를 피한다. 데이터·모델·비밀값은 Git 밖에 보존한다. main merge·배포·유료 cloud·OS/드라이버 교체는 수행하지 않는다.

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
