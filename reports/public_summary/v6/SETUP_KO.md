# V6/V7 초기 설정 — 2026-11-14

2026-11-14 22:52 KST 갱신: T1 native GPU는 학습 전 검증 오류로2회 실패했다. PEFT 경로축약 검증을 수정하고 미시작9개에만 새source를 등록했다. T1 마지막 복구는 준비 상태다. [오류·수정·검증 집계](native_gpu_source_recovery.json) 및 [최신 인계](../../../docs/agent/V6_V7_HANDOFF_KO.md)를 우선한다. 실제 GPU 성공·본학습 완료를 의미하지 않는다.

R2 통합 지시와 사용자의 모델·추론 강도 위임을 적용했다. 기존 실행은 실제 호스트의 process identity, GPU 사용, lock과 원장으로 확인했으며 시작 시 진행 중인 의료 모델 worker는 없었다. 기존 완료 v1~v5 및 11/13 Release를 보존한다.

이번 구현은 독립 worker, 단일 supervisor, 영속 queue·복구 횟수, immutable source/config/data/model identity, 성공 artifact 검증, 미시작 단계의 versioned activation을 제공한다. 최초 시도 이후 복구는 최대2회이며 2회 실패 후 강한 모델의 실제 분석 근거 없이 마지막 시도를 시작하지 않는다. 살아 있는 worker는 heartbeat가 늦다는 이유로 중단하지 않는다.

첫 GPU job은 기존 Qwen3-1.7B 경로의 새 engineering smoke(320예제/20update)다. 전체 V6 공통 데이터 동결이나 본학습 결과가 아니며 이 adapter로 본학습을 초기화하지 않는다. 완료는 실제 final checkpoint와 fresh-process reload receipt로 판정한다. 고유 8개 모델 준비가 뒤따른다.

V6 6 text/4 VLM 셀, 최초10조건과 validation으로 선정하는 추가seed 최대8조건, V6 종료 뒤 V7 4조건을 선언했다. 미구현 scientific gate는 `IMPLEMENTATION_PENDING`이며 완료로 세지 않는다. [아키텍처 감사](../../../docs/research/V6_V7_ARCHITECTURE_AUDIT_KO.md)에 남긴 native LoRA 범위·마스킹·공통 train 교집합·final 선택·likelihood scorer·평가 순서 보완을 예약 작업이 이어서 구현한다. 한 모델 실패가 다른 모델의 준비나 전체 보고를 막지 않게 terminal barrier를 사용한다.

예약은 로컬 프로젝트에 연결된 네이티브 heartbeat `medvlm-v6-v7`, 요청 간격20분이다. OS 서비스는 worker를 독립 실행하는 launcher이며 별도20분 타이머로 표시하지 않는다. 실제 활성화·등록된 다음 시각·worker PID/boot/start ticks는 workspace `outputs/v6_v7/`의 배포·예약 receipt에 기록한다. 로컬 예약에는 이 컴퓨터와 Codex 앱이 켜져 있어야 한다.

CPU 전체 검사 및 독립 launcher 생존·동시 실행 배제·복구 횟수·PID 재사용·artifact 변조·final report/CI gate 검사를 수행한다. 실제 CUDA 연산은 RTX3060 12GB에서 통과했다. 학습 완료나 평가 성능은 초기 설정 검사로 대신하지 않는다.

[현재 재개 방법](../../../docs/agent/V6_V7_HANDOFF_KO.md), [실제 에이전트 모델 선택 기록](orchestration_log.json), [채택된 사용자 R2](../../../docs/research/V6_V7_R2_INSTRUCTIONS.txt)를 함께 보존한다. 상태 수집은 로컬 Python, pipeline 구현은 Sol/high, 아키텍처 감사는 Astra/high로 수행했다. 메인 채팅 모델은 변경하지 않았다.

71762는 제공처가 설명한 category·질환명·의도 내 random 연결 실험이다. 개별 QA의 임상 정합성·증강 lineage는 검증되지 않았다. GPU 시간 상한은 모두 unlimited/null이고, Jetson은 계획만 있으며 실행하지 않는다. 새로운 원본·V6/V7 가중치·비공개 예측은 공개하지 않는다. 최종 보고와 실제 최종 HEAD CI를 검증한 후에만 해당 campaign 예약을 중지한다.

초기 등록 검증: 2026-11-14 13:39 KST, 독립 worker PID236828 및 학습 PID236836의 생존과 실제 GPU 점유를 확인했다. 실행 소스 commit은 `1602f3557c09ce4b3347362075c4088030ab368d`이며 복구0회다. 네이티브 예약은 ACTIVE, 다음 등록 시각은 2026-11-14 13:59:43 KST다. CPU298passed/7skipped/23subtests passed. 본학습·평가 완료를 기다리지 않고 예약으로 이어간다. 이 시각 이후 상태는 private supervisor receipt가 기준이다.

2026-11-14 14:00 KST 첫 예약 점검: engineering smoke 20update/320예제/46,179 target tokens, 제외0·adapter 변화·fresh reload 모두 성공을 artifact SHA로 확인했다. GPU126.561초, 기술 실패0회이며 V6 본학습 성공으로 집계하지 않는다. supervisor가 prepare-T1로 진행했다. 전달 HEAD1cad6f1의 실제 CI43557824314는 일반 wheel 설치에서 source 경로를 checkout처럼 계산한 문제로 실패했다. 실행 snapshot은 보존하고 다음 source의 설치 형태별 identity 처리를 보완한다.

CI 보완: 실제 import된 package bytes를 고정 logical source key로 hash하고 planner에는 명시적 checkout/snapshot 경로를 지원했다. source fallback은 필요한 경우에만 계산하며 runner/smoke hash를 일치시켰다. 일반 wheel 설치 전체303passed/7skipped/23subtests, 추가 회귀검사5개 통과. 실행 중 snapshot·공유 환경·동결 config는 변경하지 않았다. 새 commit의 원격 CI는 다음 실제 조회로 별도 확인한다.

2026-11-14 18:20 KST 점검: 고유 8개 모델의 준비 결과가 모두 `SUCCEEDED`다. Qwen3.5는 native shard 파일명 지원을 보완했고 SmolVLM2는 필수 processor 의존성 두 개를 추가한 뒤 각각 마지막 허용 복구에서 성공했다. 두 모델의 이전 실패2회와 총 시도3회는 그대로 보존했다. [집계와 고정 revision](model_preparation.json)에 준비 receipt hash를 기록한다. 이 결과는 가중치·native asset 준비 범위이며, 전체 train 적격성·10셀 CUDA smoke·본학습 성공을 의미하지 않는다.

현재 본학습 완료 수는0이다. 다음 범위는 텍스트8,000개와 SLAKE train1,979개의 셀별 native 입력·정답 mask·2048 context 적격성 검사다. 이후 지원 가능한 셀의20-update GPU smoke를 확인하고 축별 공통 train을 동결한다. 준비된 모델 수가 늘었다는 이유로 예전96-target Qwen3.5 adapter나 minimum-validation-loss checkpoint를 V6 결과로 재해석하지 않는다.

18:36 KST에 검증한 source2ff65d4의 CPU preflight10개를 supervisor에 등록했다. GPU smoke10개는 별도 미구현 작업이며 이번 턴에는 추가 worker를 실행하지 않았다. [구현·CPU 검증 집계](native_preflight_implementation.json)와 [최신 인계](../../../docs/agent/V6_V7_HANDOFF_KO.md)를 참고한다. 전체 train scan과 본학습 결과는 아직 없다.

2026-11-14 22:00 KST 점검: native CPU preflight10개가 모두 최초 시도에 성공했다. 실제 train 전체의 인코딩·정답 mask·이미지 tensor 검사 결과는 [CPU 결과](NATIVE_PREFLIGHT_KO.md)와 [집계 JSON](native_preflight_results.json)에 보존했다. 길이 초과를 제외한 잠정 교집합은 text7,997/VQA1,979다. 공통 학습셋 동결은 GPU smoke 이후이며, CPU 검사를 본학습·평가 성공으로 세지 않는다.

2026-11-14 22:17 KST 등록: [GPU smoke 실행부 검증](native_gpu_smoke_implementation.json)을 완료하고 미시작10개를 기존 supervisor에 등록했다. source`9634b0f`, manifest`30d28fbfbb110c1f5c577ec2bc84bddb0cb8aaca7220da302c8b0bc0d24a3b87`. 전체 설치523passed/24skipped/32subtests, nativeCPU runtime9/worker29, Astra/high 검토 통과. worker 없음·GPU native 실제실행0이며 다음22:19:58 KST 예약이 T1을 시작한다. 이전파일262개와state33개 보존. 부분resume 없이 같은config의 허용된 복구는원본Base에서시작하며이전attempt를보존한다. GPU시간상한null·Jetson미실행·새가중치비공개를유지한다.
