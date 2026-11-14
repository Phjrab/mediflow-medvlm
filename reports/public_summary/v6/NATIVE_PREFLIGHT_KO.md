# V6 native CPU 사전 검사 결과 — 2026-11-14

고유 8개 모델의 텍스트 6개·VLM 4개 경로에서 기존 train 전체의 native 인코딩 검사를 완료했다. 각 작업은 최초 시도에 성공했고 supervisor가 완료 artifact SHA를 검증했다. 실제 가중치의 GPU 학습·재로드·본평가 결과는 아직 없다.

| 셀 | train 후보 | 2048 이내 적격 | 제외 | 적격 표본의 최대 native token |
| --- | ---: | ---: | ---: | ---: |
| T1 | 8,000 | 8,000 | 0 | 2,036 |
| T2 | 8,000 | 8,000 | 0 | 2,036 |
| T3 | 8,000 | 7,999 | 1 | 2,016 |
| T4 | 8,000 | 7,999 | 1 | 2,016 |
| T5 | 8,000 | 8,000 | 0 | 2,009 |
| T6 | 8,000 | 7,997 | 3 | 2,037 |
| V1 | 1,979 | 1,979 | 0 | 1,093 |
| V2 | 1,979 | 1,979 | 0 | 321 |
| V3 | 1,979 | 1,979 | 0 | 321 |
| V4 | 1,979 | 1,979 | 0 | 1,481 |

제외된 텍스트 표본은 native 입력과 전체 정답을 합친 길이가 2,048을 초과했다. 답변을 자르거나 요약하지 않았다. T3/T4는 같은 1개를, T6는 3개를 제외했고, 텍스트 6개 경로의 잠정 교집합은 7,997개다. VLM 4개 경로는 SLAKE train 1,979개가 모두 적격이다. 이 교집합 수량은 CPU 관측치이며, GPU smoke 결과와 지원 가능한 경로를 확인한 뒤 공통 학습셋을 동결한다.

native template의 날짜는 2026-11-14로 고정했다. prompt·이미지·padding은 loss에서 제외하고 정답 전체의 감독 token을 확인했다. 각 VLM의 pixel tensor와 native 부가 tensor를 유지했다. dev/test는 읽지 않았고, 모델 가중치를 메모리에 올리거나 CUDA를 초기화하지 않았다. 따라서 이 검사는 gradient·LoRA 변화·optimizer update·fresh reload 성공의 근거가 아니다.

다음 단계는 셀마다 독립 Base에서 20 updates/320 train 표본의 GPU smoke를 수행하고 실제 adapter 저장·새 프로세스 재로드를 검증하는 것이다. 이 smoke adapter를 본학습 초기값으로 사용하지 않는다. 본학습·추가 seed·V7은 미시작이다.

[기계 판독 집계와 artifact SHA](native_preflight_results.json), [구현 및 CPU/meta 검증](native_preflight_implementation.json), [전체 설정](SETUP_KO.md). 개별 표본 ID·원문·이미지·새 가중치는 private 경로에 보존한다.
