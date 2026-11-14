# Mediflow 실험 결과 모음

공개 코드·집계 결과·그림은 `codex/medical-finetune-v1-20261023` 작업 브랜치에 보관한다. main 병합은 하지 않았다. 각 보고서는 해당 실행 당시의 결과이며 후속 실험이 이전 파일을 덮어쓰지 않는다.

2026-11-13 사용자의 LoRA 공개 업로드 추가 승인에 따른 전체 보존 결과는 [GitHub Release](https://github.com/Phjrab/mediflow-medvlm/releases/tag/medvlm-results-20261113)에서 받는다. **평가 기록 62,998건(63파일), 학습 이벤트 3,566건(20실행), 보존 중인 LoRA 체크포인트 22개**를 포함한다. 생성 응답·문항별 점수와 실패 기록도 포함하며, 평가 기록 수는 고유 질문 수가 아니다. [배포 범위·복구 방법·미실행 항목](RELEASE_20261113_KO.md)과 [파일별 SHA256 명세](../release_exports/medvlm-results-20261113/manifest.json)를 함께 확인한다.

| 실행 | 실제 수행 범위 | 결과 |
| --- | --- | --- |
| v1 | 텍스트2종·VLM2종 실제 학습·새 process 재로드·주 평가 및 진단 | [결과](RESULTS_KO.md) |
| v2 | 한국어 KO/MIX 학습·공통 텍스트 평가·VQA 확대 평가·기존 예측 재분석 | [핵심 해석](v2/READOUT_KO.md), [전체 결과](v2/RESULTS_KO.md) |
| v3 | 전달된71762 원본·별도 질문/답변 구성요소 감사. 당시에는 QA 연결을 만들지 않음 | [감사 결과](v3/RESULTS_KO.md) |
| v4 | 사용자 예외 승인에 따라 제공처 설명의 category-random 방식 독립 구현·두 텍스트 모델 학습·평가 | [결과](v4/RESULTS_KO.md), [검증](v4/verification.json), [그림](v4/figures) |
| v6/v7 | 고유8모델 준비·native CPU 적격성 검사10개 완료. GPU smoke·본학습·평가는 후속 단계 | [CPU 검사 결과](v6/NATIVE_PREFLIGHT_KO.md), [설정 및 미구현 범위](v6/SETUP_KO.md) |
| v5 | 24검증문항의384건 형식 진단. 두 모델 Base/Tuned·기존/한국어 지시·16/256token 비교 | [해석](v5/READOUT_KO.md), [전체 결과](v5/RESULTS_KO.md), [검증](v5/verification.json) |

v4의 실행 성공은 성능 개선 입증을 뜻하지 않는다. 두 모델의 분류형 출력이 요구 형식을 지키지 못했고, 무작위 참조와의 문자열 일치는 임상 정확도가 아니다. 증강 parent와 고정된 문항별 QA 대응은 미제공이다. 자세한 조건과 남은 한계는 각 보고서에 기록했다.

원본 데이터·원본 전문가 검토표·비공개 A/B 키·기본 모델 가중치·optimizer/RNG 상태는 로컬에 유지한다. 공개 Release에는 생성 응답, 원본을 노출하지 않는 평가 점수와 실행 기록, 학습된 LoRA를 담았다. 원본 질문·문맥·이미지·자유서술 정답 본문은 재배포하지 않는다. Git 복제 후 Release를 별도로 내려받아야 하며, 원본 데이터와 기본 모델은 각 제공처에서 확보해야 한다. 과거 보고서의 공개 범위 설명은 당시 기준이고, 이번 추가 승인과 배포 범위는 이 문서 및 Release 명세가 갱신한다.

v5는 [검증셋 전용 형식 진단 계획](../../docs/research/FORMAT_DIAGNOSTIC_V5_KO.md)에 따라 완료했다. 두 Tuned는 모든 조건에서 형식0/24였으며, 기존 v4 test 점수를 바꾸지 않았다. 추가학습·새 test 성능 개선을 수행하거나 입증한 결과가 아니다.
