# V6/V7 초기 설정 — 2026-11-14

R2 통합 지시와 사용자의 모델·추론 강도 위임을 적용했다. 기존 실행은 실제 호스트의 process identity, GPU 사용, lock과 원장으로 확인했으며 시작 시 진행 중인 의료 모델 worker는 없었다. 기존 완료 v1~v5 및 11/13 Release를 보존한다.

이번 구현은 독립 worker, 단일 supervisor, 영속 queue·복구 횟수, immutable source/config/data/model identity, 성공 artifact 검증, 미시작 단계의 versioned activation을 제공한다. 최초 시도 이후 복구는 최대2회이며 2회 실패 후 강한 모델의 실제 분석 근거 없이 마지막 시도를 시작하지 않는다. 살아 있는 worker는 heartbeat가 늦다는 이유로 중단하지 않는다.

첫 GPU job은 기존 Qwen3-1.7B 경로의 새 engineering smoke(320예제/20update)다. 전체 V6 공통 데이터 동결이나 본학습 결과가 아니며 이 adapter로 본학습을 초기화하지 않는다. 완료는 실제 final checkpoint와 fresh-process reload receipt로 판정한다. 고유 8개 모델 준비가 뒤따른다.

V6 6 text/4 VLM 셀, 최초10조건과 validation으로 선정하는 추가seed 최대8조건, V6 종료 뒤 V7 4조건을 선언했다. 미구현 scientific gate는 `IMPLEMENTATION_PENDING`이며 완료로 세지 않는다. [아키텍처 감사](../../../docs/research/V6_V7_ARCHITECTURE_AUDIT_KO.md)에 남긴 native LoRA 범위·마스킹·공통 train 교집합·final 선택·likelihood scorer·평가 순서 보완을 예약 작업이 이어서 구현한다. 한 모델 실패가 다른 모델의 준비나 전체 보고를 막지 않게 terminal barrier를 사용한다.

예약은 로컬 프로젝트에 연결된 네이티브 heartbeat `medvlm-v6-v7`, 요청 간격20분이다. OS 서비스는 worker를 독립 실행하는 launcher이며 별도20분 타이머로 표시하지 않는다. 실제 활성화·등록된 다음 시각·worker PID/boot/start ticks는 workspace `outputs/v6_v7/`의 배포·예약 receipt에 기록한다. 로컬 예약에는 이 컴퓨터와 Codex 앱이 켜져 있어야 한다.

CPU 전체 검사 및 독립 launcher 생존·동시 실행 배제·복구 횟수·PID 재사용·artifact 변조·final report/CI gate 검사를 수행한다. 실제 CUDA 연산은 RTX3060 12GB에서 통과했다. 학습 완료나 평가 성능은 초기 설정 검사로 대신하지 않는다.

[현재 재개 방법](../../../docs/agent/V6_V7_HANDOFF_KO.md), [실제 에이전트 모델 선택 기록](orchestration_log.json), [채택된 사용자 R2](../../../docs/research/V6_V7_R2_INSTRUCTIONS.txt)를 함께 보존한다. 상태 수집은 로컬 Python, pipeline 구현은 Sol/high, 아키텍처 감사는 Astra/high로 수행했다. 메인 채팅 모델은 변경하지 않았다.

71762는 제공처가 설명한 category·질환명·의도 내 random 연결 실험이다. 개별 QA의 임상 정합성·증강 lineage는 검증되지 않았다. GPU 시간 상한은 모두 unlimited/null이고, Jetson은 계획만 있으며 실행하지 않는다. 새로운 원본·V6/V7 가중치·비공개 예측은 공개하지 않는다. 최종 보고와 실제 최종 HEAD CI를 검증한 후에만 해당 campaign 예약을 중지한다.

초기 등록 검증: 2026-11-14 13:39 KST, 독립 worker PID236828 및 학습 PID236836의 생존과 실제 GPU 점유를 확인했다. 실행 소스 commit은 `1602f3557c09ce4b3347362075c4088030ab368d`이며 복구0회다. 네이티브 예약은 ACTIVE, 다음 등록 시각은 2026-11-14 13:59:43 KST다. CPU298passed/7skipped/23subtests passed. 본학습·평가 완료를 기다리지 않고 예약으로 이어간다. 이 시각 이후 상태는 private supervisor receipt가 기준이다.
