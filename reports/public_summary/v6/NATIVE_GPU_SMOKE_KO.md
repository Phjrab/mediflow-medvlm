# V6 native GPU smoke 결과 — 2026-11-15

고유 **8개 모델의 텍스트 6개·VLM 4개 셀 모두 실제 GPU smoke와 fresh-process reload를 통과**했다. 마지막 V4는 2026-11-15 06:00:15 KST에 완료됐다. 각 셀은 원본 Base에서 시작해 train 320표본으로 20 optimizer updates를 수행했고, 학습 중 제외 0개·유한 gradient·대상 projection별 adapter 변화·저장한 processor·새 프로세스 재로드를 검증했다.

총 22시도 중 성공 10회·실패 12회다. 이전 실패와 셀별 누적 복구 횟수는 보존했다. 성공 실행의 표본 노출 합계 3,200은 고유 질문 수가 아니다. 모델별 전체 train 적격성 결과와 실행 identity·성공 receipt SHA는 [집계 JSON](native_smoke_results.json)에 있다.

| 셀 | 모델 | 시도 / 실패 | 성공 학습 시간(분) | 성공 학습 peak reserved(GiB) |
|---|---|---:|---:|---:|
| T1 | Qwen/Qwen3-1.7B | 3 / 2 | 1.94 | 7.61 |
| T2 | Qwen/Qwen3-4B | 1 / 0 | 3.79 | 7.92 |
| T3 | google/gemma-3-4b-it | 3 / 2 | 3.96 | 8.54 |
| T4 | google/medgemma-1.5-4b-it | 3 / 2 | 3.96 | 8.54 |
| T5 | microsoft/Phi-4-mini-instruct | 1 / 0 | 2.92 | 10.19 |
| T6 | HuggingFaceTB/SmolLM3-3B | 1 / 0 | 3.36 | 7.38 |
| V1 | Qwen/Qwen3.5-2B | 1 / 0 | 5.47 | 7.93 |
| V2 | google/gemma-3-4b-it | 3 / 2 | 10.21 | 8.53 |
| V3 | google/medgemma-1.5-4b-it | 3 / 2 | 10.22 | 8.53 |
| V4 | HuggingFaceTB/SmolVLM2-2.2B-Instruct | 3 / 2 | 17.84 | 7.03 |

시간과 메모리는 각 성공 시도의 학습 구간만 나타내며 로딩·재로드·과거 실패 비용을 모두 포함하지 않는다. 모델별 native template와 토큰 길이도 다르므로 동일 처리량 벤치마크로 해석하지 않는다. 전체 사용량은 비공개 공유 GPU 원장에 누적한다. GPU 시간 상한은 계속 unlimited/null이다.

T1의 PEFT selector 축약 검증, T3/T4의 전정답 loss 메모리, V2/V3의 학습–재로드 CUDA lifetime, V4의 native projected image token 결합 문제는 실제 마지막 허용 복구에서 해결됐다. [복구 구현 기록](native_runtime_recovery.json)은 등록 당시의 역사 기록이며, 수정 후 실제 GPU 성공은 이번 결과에 기록한다. seed·정답·native processor·recipe는 보존했고 smoke adapter를 본학습 초기값으로 쓰지 않는다.

현재 완료 범위는 native 경로의 학습·저장·재로드에 대한 기술 검증이다. 집계 worker도 2026-11-15 06:29:23 KST에 준비 8개·CPU 10개·GPU 10개의 증거 검증을 마쳤고, 06:51 점검에서 supervisor가 성공을 확인했다. [집계 실행 결과](preflight_gate_results.json)에 실제 receipt와 실행 identity를 기록했다. **공통 train 동결, V6 본학습·평가·추가 seed, V7 P0/P1/R0/R1은 아직 미시작**이며, 공통 train과 본학습 recipe를 다음 단계에서 별도로 동결한다. 임상 성능·모델 순위·사람 검토 결과를 주장하지 않으며 실제 사람 검토는 0건이다.

원본 데이터·새 V6/V7 가중치·비공개 예측은 공개하지 않았다. Jetson은 향후 계획만 유지한다. [최신 실행 인계와 재개 방법](../../../docs/agent/V6_V7_HANDOFF_KO.md)을 따른다.
