# V6 첫 seed 본학습 실행부 — 2026-11-15

공통 train 동결은 실제 완료됐으며, 첫 seed 본학습 실행부는 구현·검증 후 기존 단일 supervisor에 등록됐다. **등록 완료·본학습 미실행 상태다.** 실제 실행 여부는 개별 worker의 final checkpoint·fresh-process reload receipt와 supervisor 상태로 판단한다.

## 실행 범위

고유 8개 모델의 텍스트 6개·VLM 4개 셀에 seed `20261023`을 적용한다. 각 셀은 지정된 원본 Base에서 독립적으로 시작한다. GPU smoke와 과거 학습 adapter는 초기화에 재사용하지 않는다. 추가 seed와 V7 조건을 이 단계에 포함하지 않는다.

텍스트 7,997개·VQA 1,979개의 실제 공통 freeze, 원본 행·정답·모델별 native encoding과 첫 seed 순서를 그대로 사용한다. 전체 정답의 token 평균 loss를 예제별로 구한 뒤 해당 업데이트의 실제 예제 수로 평균한다. 텍스트는 500 updates·마지막 13개, VQA는 124 updates·마지막 11개다. 표본 반복·누락·추가 제외·정답 자르기를 허용하지 않는다.

NF4 double quantization·BF16 연산, LoRA rank 8/alpha 16/dropout 0.05, 검증된 언어 projection 범위, microbatch 1·유효 batch 16, context 2,048과 gradient checkpointing을 유지한다. AdamW betas 0.9/0.999·eps 1e-8·weight decay 0, 선형 감소·warmup 0·gradient clip 1이며 텍스트 LR 2e-4/VQA LR 1e-4다. 세부 identity와 모델별 token 합계는 [실제 공통 동결](common_train_freeze_results.json)에 연결한다.

## 완료와 재개 판단

공통 train 전량을 한 pass 처리하고 정확한 updates·표본 순서·token 수·adapter 변화가 확인된 checkpoint만 `final`로 인정한다. 기존 minimum-validation-loss `selection.json`은 수정하지 않는다. 학습 child가 종료된 뒤 별도 process에서 같은 Base revision·저장 adapter·processor를 재로드하고 실제 추론을 확인한다. 재로드 확인에 학습 표본을 사용하면 학습 표본 probe로 표시하며 독립 평가나 성능 검증으로 해석하지 않는다.

한 worker는 한 셀만 수행한다. 단일 supervisor가 10개 작업의 queue와 최초 시도 이후 최대 2회 기술 복구를 누적 관리한다. 한 셀의 실패가 다른 셀의 실행을 막지 않도록 독립 의존성을 두고, 마지막 CPU 집계는 성공·실패·차단을 구별한다. 정상 worker를 시간 경과나 낮은 품질 때문에 중단하지 않는다.

다른 identity의 checkpoint를 이어 학습하거나 부분 adapter를 optimizer/RNG resume checkpoint로 부르지 않는다. 기존 성공 산출물의 재사용과 중단된 시도의 재실행은 별도로 검증하며 이전 시도 기록을 보존한다.

## 후속 범위

첫 seed 본학습 이후 validation 출력 상한 보정, 전체 공통 dev, 서로 다른 family 선정, 추가 seed 최대 8조건, test·보고를 순서대로 진행한다. V6 종료 후 V7 P0/P1/R0/R1 네 조건을 별도로 실행한다. 현재 단계는 dev/test 점수나 family 선택을 수행하지 않는다.

GPU 누적 시간 상한은 unlimited/null이고 Jetson은 미실행이다. 원본 데이터·새 가중치·비공개 예측은 공개하지 않는다. 이 문서는 실행부의 범위를 설명하며 실제 학습 성능이나 임상 검증을 주장하지 않는다.

## 실제 등록·검증 — 2026-11-15 08:01 KST

source commit `cab4a59887d42845909988d066c238574a275b4e`, worker source hash `b6fcf38c8c1cbc48abf5fe85797dc15fe416ac9f2dac6e165523878fbdf95634`, aggregate source hash `a58d62fe7d6c3b6be359dff9f7a0cd4b608f7edbbe141ca90e5ef5da1ab22e0b`, 활성 manifest `6282505feec5c200f498abba8e127cc469020bf19db6e56a87395d66a2a9c6d1`다. `v6-first-seed-T1`~`T6` 및 `V1`~`V4`의 독립 GPU 작업 10개와 `v6-first_seed_final` CPU 집계를 한 번의 `refine-first-seed`로 등록했다. 현재53개 작업은 SUCCEEDED31/PENDING11/IMPLEMENTATION_PENDING11이고 실행 중 worker는 없다. 다음 대상은 `v6-first-seed-T1`이다. 기존561파일과 다른42개 state record·모든 누적 시도/실패/복구 횟수를 보존했다.

설치본 전체830passed/52skipped/32subtests/기존Pillow경고6을 확인했다. 이후 완료 receipt의 불필요한 재쓰기를 막는 작은 수정에 대해 정확한 최종 설치본114passed/4optional-torch-skipped와 독립 Astra native CPU185passed를 확인했다. 전체 검사가 마지막 수정 전이라는 점을 구분한다. 실제10개 config는 공통 freeze의 표본·순서·target/sequence token 합계와 일치하며 계획 검증에 torch를 import하지 않았다. CPU 수치 검증은 실제 마지막1/13/11개 평균으로 독립 AdamW 기준과 일치했다. 이 결과로 장시간 CUDA 메모리 적합성·최종 재로드 성공을 미리 주장하지 않는다.

한 셀의 실제 실패와 실패 원인은 집계에 남긴다. FAILED/BLOCKED 셀은 등록 당시 고정 설정을 검증하며, 실패 원인이 된 model cache 접근을 성공 필수조건처럼 재요구하지 않는다. SUCCEEDED 셀은 실제 final artifact·receipt·재로드 검증을 통과해야 한다. 실패를 숨겨 전체 성공으로 바꾸지 않는다.

이번 heartbeat는 기존 감독 tick1회만 호출했고 등록 뒤 추가 tick/GPU 실행은0이다. 기존 coordinator `1602f35`와 OS launcher를 유지했다. 네이티브 예약 `medvlm-v6-v7`은20분 설정 ACTIVE이며 앱의 실행 시각은 활성 채팅 턴 때문에 지연될 수 있다. 최신 시간은 workspace scheduler receipt를 따른다. [기계 판독 집계](first_seed_implementation.json), [최신 인계·재개 명령](../../../docs/agent/V6_V7_HANDOFF_KO.md)을 참고한다.
