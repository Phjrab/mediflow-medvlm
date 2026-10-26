# v2 후속 결과

## 현재 단계

기준 시각: 2026-10-26T21:45:55.750228+00:00. 실제 파일을 읽은 시점의 상태다. 실행 중 항목의 중간 예측은 완료 점수에 포함하지 않았다.

| 단계 | 완료·무결성 검증 | 예정 | 상태 |
|---|---:|---:|---|
| VQA 본평가·이미지 진단 | 12 | 12 | {"SUCCEEDED": 12} |
| KO/MIX smoke·본학습·재로드 | 8 | 8 | {"SUCCEEDED": 8} |
| 텍스트 평가 조건 | 9 | 9 | {"SUCCEEDED": 9} |

VQA validation 결정: **SUCCEEDED**, 선택된 출력 상한 256 tokens. Validation의 길이 종료 규칙으로 128→256을 한 번 검토했다. 테스트로 상한을 선택하지 않았다.

## 이번에 한 일

원본 v1 가중치·어댑터를 재사용한 VQA 평가와 새 KO/MIX 텍스트 조건을 분리했다. 아래 수치는 완료 파일의 예측 SHA, protocol·입력·정답·이미지 SHA, config·model revision, 기록된 native 모델 digest, adapter byte hash를 검증한 뒤 다시 집계한 값이다. base 가중치 전체 재해시는 하지 않았다.

| 데이터/작업 | 실제 확보·준비 | 상태 |
|---|---|---|
| AI-Hub 71875 | {"test": 957, "train": 15297, "validation": 956} | SUCCEEDED |
| KO/MIX corpus | {"ko": 4000, "mix": 8000} | SUCCEEDED |
| MedMCQA | {"test": 0, "train": 20000, "validation": 2816}; KO/MIX 미포함 | PREPARED_NOT_RUN |
| AI-Hub 71762 | 확인된 다운로드 0 bytes; 입력 receipt 기준 | BLOCKED_DELIVERY |
| Jetson | 사용자 보류; v2 접속·배포·benchmark 미실행 | USER_DEFERRED |
| 전문가 검토표 준비 | 4/4 packet, 64개 답변 쌍; q_type 2/3 각 8개/packet | PREPARED_NOT_RUN |
| 전문가 검토 수행 | 검토 0건·점수 미수집 | NOT_PERFORMED |

검토표 준비와 사람의 검토는 별도다. 준비된 표는 파일·프로토콜·출처 예측 해시와 점수 칸 공란을 확인한다. 원문이 있는 CSV 및 A/B 대응 private key는 로컬 데이터 경로에만 두며 공개 보고서에는 싣지 않는다.

71875 수집/제외 수: {"non_qa_documents": 52126, "q_type_1": 14028, "q_type_2": 1634, "q_type_3": 1618, "raw_official_split_train": 15360, "raw_official_split_validation": 1920, "raw_qa": 17280}; 제외: {"EXACT_INPUT_DUPLICATE": 1, "MCQ_ANSWER_TEXT_MISMATCH": 48, "MCQ_EMPTY_OR_DUPLICATE_OPTION": 3, "MCQ_INVALID_ANSWER_NUMBER": 1, "MCQ_OPTION_BLOCK_AMBIGUOUS": 13, "QUESTION_GROUP_OVERLAP_WITH_OFFICIAL_HELDOUT": 4}. 최종 corpus manifest가 가리키는 source audit 기준이다.

71762는 connector 256 MiB 다운로드 제한과 정상 브라우저의 ERR_BLOCKED_BY_CLIENT로 막혔다. 약 4.25 GB 원본은 확보되지 않았으며 QA pairing도 확인되지 않았다. 수집된 원본이 없는 reader 구현을 데이터 준비 완료로 계산하지 않는다.

의미 유사도 검사: KorMedMCQA 4068문항을 보호하고 E5 cosine ≥0.92, top-5에서 3396행/8278쌍을 후보로 찾았다. 이는 확정 중복이나 번역 동등성 판정이 아니다. lexical exact/near 검사와 별도다.
split별 후보 처리: {"test": {"canonical": 957, "semantic_candidate_rows": 188, "eligible_after_quarantine": 957, "test_preserved_candidates_only_flagged": true}, "train": {"canonical": 15297, "semantic_candidate_rows": 3010, "eligible_after_quarantine": 12287, "test_preserved_candidates_only_flagged": false}, "validation": {"canonical": 956, "semantic_candidate_rows": 198, "eligible_after_quarantine": 758, "test_preserved_candidates_only_flagged": false}}. train/dev 후보는 보수적으로 제외하고 test 전체 957개는 보존하며 후보 표시만 남겼다.

최종 KO q_type 선택은 {"1": 1500, "2": 1300, "3": 1200}; MIX는 같은 KO에 {"medquad": 3550, "pubmedqa_labeled": 450}를 더한다. 공통 dev는 1516개(AI-Hub 758 + EN 758)다. 최초 균등 q_type quota와 dev 1,912개에서 바뀐 이유는 고정 threshold의 후보 격리로 long/train과 dev가 줄었기 때문이다. 평가 성능으로 선택하거나 threshold를 바꾸지 않았다.

KO4,000(250 update), MIX8,000(500 update)는 seed 20261023 한 pass를 계획한다. 각각 독립 smoke 20 update와 새 프로세스 재로드를 먼저 확인한다. LR 2e-4, LoRA rank 8/alpha 16/dropout 0.05, effective batch 16이다. KO/MIX checkpoint 비교에는 동일 공통 dev 파일에서 source별 균형으로 고른 48개를 쓴다. q_type별 층화 표본을 보장하지 않는다.

| 학습 run | 상태 | 실제 update | 실제 처리 문항 | supervised tokens | 선택 update | 재로드 확인 |
|---|---|---:|---:|---:|---:|---|
| qwen3_17b-ko-smoke-v2-20261026 | SUCCEEDED | 20 | 320 | 16245 | 20 | True |
| qwen3_17b-ko-full-v2-20261026 | SUCCEEDED | 250 | 4000 | 243394 | 250 | True |
| qwen3_17b-mix-smoke-v2-20261026 | SUCCEEDED | 20 | 320 | 46179 | 20 | True |
| qwen3_17b-mix-full-v2-20261026 | SUCCEEDED | 500 | 8000 | 1137306 | 500 | True |
| qwen3_4b-ko-smoke-v2-20261026 | SUCCEEDED | 20 | 320 | 16245 | 20 | True |
| qwen3_4b-ko-full-v2-20261026 | SUCCEEDED | 250 | 4000 | 243394 | 200 | True |
| qwen3_4b-mix-smoke-v2-20261026 | SUCCEEDED | 20 | 320 | 46179 | 20 | True |
| qwen3_4b-mix-full-v2-20261026 | SUCCEEDED | 500 | 8000 | 1137306 | 500 | True |

실제 처리량은 전체 run의 실행량이며, 평가는 validation으로 선택한 checkpoint를 사용한다. 선택 시점의 문항·token·loss는 run_summary.json에 별도로 기록한다. 각 run의 최신·선택 checkpoint와 검증 이벤트를 보존하며, 선택되지 않은 중간 checkpoint 파일은 기존 보관 정책에 따라 정리한다. 이전 v1 실험은 변경하지 않는다.

## 확인 결과

KorMedMCQA는 동일 전체 3,009문항의 micro/macro subject 정답률, coverage, invalid, 유효 응답 정답률을 함께 표시한다. 무응답·실패는 전체 분모에 남긴다. [직군별 세부 지표](metrics.csv)는 source_id=kormedmcqa와 subject 열로 확인할 수 있다.

| 모델/조건 | n | 전체 micro | subject macro | coverage | invalid 형식 | 유효 응답 정답률 |
|---|---:|---:|---:|---:|---:|---:|
| qwen3_17b/base | 3009 | 19.01% | 18.86% | 54.90% | 45.10% | 34.62% |
| qwen3_17b/en_v1 | 3009 | 39.08% | 37.70% | 99.80% | 0.20% | 39.16% |
| qwen3_17b/ko | 3009 | 37.99% | 36.93% | 100.00% | 0.00% | 37.99% |
| qwen3_17b/mix | 3009 | 39.32% | 38.07% | 100.00% | 0.00% | 39.32% |
| qwen3_4b/base | 3009 | 43.67% | 41.90% | 80.76% | 19.24% | 54.07% |
| qwen3_4b/en_v1 | 3009 | 54.44% | 53.16% | 99.83% | 0.17% | 54.53% |
| qwen3_4b/ko | 3009 | 51.31% | 50.02% | 100.00% | 0.00% | 51.31% |
| qwen3_4b/mix | 3009 | 54.07% | 52.79% | 100.00% | 0.00% | 54.07% |
| medgemma15_4b/base | 3009 | 23.89% | 22.57% | 57.06% | 42.94% | 41.88% |

완료된 텍스트 평가가 없으면 위 표는 비어 있다. MedGemma 텍스트 조건은 원본 Base이며 VQA adapter를 사용하지 않는다. 검증된 text 실행은 원문 입력과 native pixel tensor 기록에서 영상 입력이 없음을 확인했다.

| 모델/조건 | AI-Hub q_type | n | 전체 MCQ 정답률 또는 lexical EM | lexical F1 | coverage | 잘림 |
|---|---|---:|---:|---:|---:|---:|
| qwen3_17b/base | MCQ | 777 | 22.39% | 해당 없음 | 52.90% | 134 |
| qwen3_17b/base | short | 83 | 0.00% | 0.31% | 100.00% | 48 |
| qwen3_17b/base | long | 97 | 0.00% | 12.90% | 100.00% | 95 |
| qwen3_17b/en_v1 | MCQ | 777 | 55.34% | 해당 없음 | 99.87% | 0 |
| qwen3_17b/en_v1 | short | 83 | 2.41% | 2.57% | 100.00% | 41 |
| qwen3_17b/en_v1 | long | 97 | 0.00% | 6.86% | 100.00% | 76 |
| qwen3_17b/ko | MCQ | 777 | 62.03% | 해당 없음 | 100.00% | 0 |
| qwen3_17b/ko | short | 83 | 3.61% | 7.59% | 100.00% | 0 |
| qwen3_17b/ko | long | 97 | 0.00% | 15.92% | 100.00% | 26 |
| qwen3_17b/mix | MCQ | 777 | 60.10% | 해당 없음 | 100.00% | 0 |
| qwen3_17b/mix | short | 83 | 6.02% | 12.01% | 100.00% | 0 |
| qwen3_17b/mix | long | 97 | 0.00% | 17.31% | 100.00% | 19 |
| qwen3_4b/base | MCQ | 777 | 58.04% | 해당 없음 | 87.00% | 16 |
| qwen3_4b/base | short | 83 | 0.00% | 0.58% | 100.00% | 77 |
| qwen3_4b/base | long | 97 | 0.00% | 14.10% | 100.00% | 97 |
| qwen3_4b/en_v1 | MCQ | 777 | 67.95% | 해당 없음 | 99.61% | 0 |
| qwen3_4b/en_v1 | short | 83 | 0.00% | 3.02% | 100.00% | 9 |
| qwen3_4b/en_v1 | long | 97 | 0.00% | 11.92% | 100.00% | 54 |
| qwen3_4b/ko | MCQ | 777 | 72.46% | 해당 없음 | 100.00% | 0 |
| qwen3_4b/ko | short | 83 | 8.43% | 15.93% | 100.00% | 0 |
| qwen3_4b/ko | long | 97 | 0.00% | 20.45% | 100.00% | 6 |
| qwen3_4b/mix | MCQ | 777 | 71.81% | 해당 없음 | 100.00% | 0 |
| qwen3_4b/mix | short | 83 | 6.02% | 12.79% | 100.00% | 1 |
| qwen3_4b/mix | long | 97 | 0.00% | 22.08% | 100.00% | 11 |
| medgemma15_4b/base | MCQ | 777 | 44.27% | 해당 없음 | 72.59% | 201 |
| medgemma15_4b/base | short | 83 | 0.00% | 0.03% | 100.00% | 83 |
| medgemma15_4b/base | long | 97 | 0.00% | 0.43% | 100.00% | 97 |

short/long의 EM/F1는 문자열 일치도이며 임상 정확도가 아니다. MCQ q_type 1, short 2, long 3을 따로 집계했다. 잘림은 사전에 고정한 생성 상한에 도달한 건수이며 기술 실패와 구분한다. 잘린 응답도 전체 평가 분모에 유지하므로, 특히 자유서술 점수는 답변 길이 제한과 함께 해석한다.

| 모델/조건 | 영어 retention source | n | 전체 정답률/lexical EM | lexical F1 | coverage | 잘림 |
|---|---|---:|---:|---:|---:|---:|
| qwen3_17b/base | medquad | 64 | 0.00% | 24.17% | 100.00% | 48 |
| qwen3_17b/base | pubmedqa_labeled | 500 | 41.80% | 해당 없음 | 100.00% | 0 |
| qwen3_17b/en_v1 | medquad | 64 | 3.12% | 33.54% | 100.00% | 15 |
| qwen3_17b/en_v1 | pubmedqa_labeled | 500 | 74.40% | 해당 없음 | 100.00% | 0 |
| qwen3_17b/ko | medquad | 64 | 0.00% | 8.91% | 100.00% | 0 |
| qwen3_17b/ko | pubmedqa_labeled | 500 | 33.80% | 해당 없음 | 100.00% | 0 |
| qwen3_17b/mix | medquad | 64 | 3.12% | 31.51% | 100.00% | 17 |
| qwen3_17b/mix | pubmedqa_labeled | 500 | 70.60% | 해당 없음 | 100.00% | 0 |
| qwen3_4b/base | medquad | 64 | 0.00% | 25.67% | 100.00% | 56 |
| qwen3_4b/base | pubmedqa_labeled | 500 | 35.40% | 해당 없음 | 91.80% | 41 |
| qwen3_4b/en_v1 | medquad | 64 | 3.12% | 34.88% | 100.00% | 15 |
| qwen3_4b/en_v1 | pubmedqa_labeled | 500 | 76.80% | 해당 없음 | 100.00% | 0 |
| qwen3_4b/ko | medquad | 64 | 0.00% | 13.16% | 100.00% | 0 |
| qwen3_4b/ko | pubmedqa_labeled | 500 | 51.00% | 해당 없음 | 100.00% | 0 |
| qwen3_4b/mix | medquad | 64 | 3.12% | 34.18% | 100.00% | 15 |
| qwen3_4b/mix | pubmedqa_labeled | 500 | 76.80% | 해당 없음 | 100.00% | 0 |
| medgemma15_4b/base | medquad | 64 | 0.00% | 21.09% | 100.00% | 64 |
| medgemma15_4b/base | pubmedqa_labeled | 500 | 47.60% | 해당 없음 | 93.40% | 33 |

VQA: 두 모델의 Base/Tuned는 같은 짧은 답변 prompt와 선택된 생성 상한을 쓴다. SLAKE 1,061개, VQA-RAD 451개 전체와 고정 32개 no-image 진단을 구분한다.

| 모델/조건 | source/이미지 | OPEN/CLOSED | n | strict EM | v2 EM | v2 F1 | 잘림 | 출력 단어 중앙값 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| qwen35_2b/base | slake/image | ALL | 1061 | 52.40% | 52.40% | 56.40% | 0 | 1 |
| qwen35_2b/base | slake/image | OPEN | 645 | 36.43% | 36.43% | 42.69% | 0 | 1 |
| qwen35_2b/base | slake/image | CLOSED | 416 | 77.16% | 77.16% | 77.64% | 0 | 1.0 |
| qwen35_2b/tuned_v1 | slake/image | ALL | 1061 | 70.50% | 70.50% | 73.85% | 0 | 1 |
| qwen35_2b/tuned_v1 | slake/image | OPEN | 645 | 61.09% | 61.09% | 66.59% | 0 | 1 |
| qwen35_2b/tuned_v1 | slake/image | CLOSED | 416 | 85.10% | 85.10% | 85.10% | 0 | 1.0 |
| qwen35_2b/base | vqa_rad/image | ALL | 451 | 47.67% | 47.89% | 51.13% | 0 | 1 |
| qwen35_2b/base | vqa_rad/image | OPEN | 179 | 16.76% | 17.32% | 25.11% | 0 | 1 |
| qwen35_2b/base | vqa_rad/image | CLOSED | 272 | 68.01% | 68.01% | 68.26% | 0 | 1.0 |
| qwen35_2b/tuned_v1 | vqa_rad/image | ALL | 451 | 48.78% | 48.78% | 52.16% | 0 | 1 |
| qwen35_2b/tuned_v1 | vqa_rad/image | OPEN | 179 | 12.29% | 12.29% | 20.81% | 0 | 1 |
| qwen35_2b/tuned_v1 | vqa_rad/image | CLOSED | 272 | 72.79% | 72.79% | 72.79% | 0 | 1.0 |
| qwen35_2b/base | slake/no_image | ALL | 32 | 25.00% | 25.00% | 25.78% | 0 | 1.0 |
| qwen35_2b/base | slake/no_image | OPEN | 17 | 5.88% | 5.88% | 7.35% | 0 | 1 |
| qwen35_2b/base | slake/no_image | CLOSED | 15 | 46.67% | 46.67% | 46.67% | 0 | 1 |
| qwen35_2b/tuned_v1 | slake/no_image | ALL | 32 | 37.50% | 37.50% | 37.50% | 0 | 1.0 |
| qwen35_2b/tuned_v1 | slake/no_image | OPEN | 17 | 11.76% | 11.76% | 11.76% | 0 | 1 |
| qwen35_2b/tuned_v1 | slake/no_image | CLOSED | 15 | 66.67% | 66.67% | 66.67% | 0 | 1 |
| medgemma15_4b/base | slake/image | ALL | 1061 | 20.08% | 36.38% | 46.23% | 49 | 1 |
| medgemma15_4b/base | slake/image | OPEN | 645 | 13.02% | 13.49% | 29.24% | 48 | 2 |
| medgemma15_4b/base | slake/image | CLOSED | 416 | 31.01% | 71.88% | 72.58% | 1 | 1.0 |
| medgemma15_4b/tuned_v1 | slake/image | ALL | 1061 | 69.37% | 69.37% | 72.76% | 0 | 1 |
| medgemma15_4b/tuned_v1 | slake/image | OPEN | 645 | 61.55% | 61.55% | 66.89% | 0 | 1 |
| medgemma15_4b/tuned_v1 | slake/image | CLOSED | 416 | 81.49% | 81.49% | 81.85% | 0 | 1.0 |
| medgemma15_4b/base | vqa_rad/image | ALL | 451 | 24.39% | 43.24% | 49.96% | 4 | 1 |
| medgemma15_4b/base | vqa_rad/image | OPEN | 179 | 5.03% | 8.38% | 24.48% | 4 | 3 |
| medgemma15_4b/base | vqa_rad/image | CLOSED | 272 | 37.13% | 66.18% | 66.73% | 0 | 1.0 |
| medgemma15_4b/tuned_v1 | vqa_rad/image | ALL | 451 | 48.34% | 48.34% | 51.62% | 0 | 1 |
| medgemma15_4b/tuned_v1 | vqa_rad/image | OPEN | 179 | 12.85% | 12.85% | 20.74% | 0 | 1 |
| medgemma15_4b/tuned_v1 | vqa_rad/image | CLOSED | 272 | 71.69% | 71.69% | 71.94% | 0 | 1.0 |
| medgemma15_4b/base | slake/no_image | ALL | 32 | 12.50% | 12.50% | 13.42% | 3 | 3.5 |
| medgemma15_4b/base | slake/no_image | OPEN | 17 | 5.88% | 5.88% | 5.88% | 2 | 9 |
| medgemma15_4b/base | slake/no_image | CLOSED | 15 | 20.00% | 20.00% | 21.97% | 1 | 1 |
| medgemma15_4b/tuned_v1 | slake/no_image | ALL | 32 | 34.38% | 34.38% | 34.38% | 0 | 1.0 |
| medgemma15_4b/tuned_v1 | slake/no_image | OPEN | 17 | 35.29% | 35.29% | 35.29% | 0 | 1 |
| medgemma15_4b/tuned_v1 | slake/no_image | CLOSED | 15 | 33.33% | 33.33% | 33.33% | 0 | 1 |

strict는 기존 NFKC·casefold·공백 정규화이고, v2는 문장 끝 .!?。！？만 추가 제거한다. 숫자·단위·부정과 내부 구두점은 유지한다. 정답 구절 포함률을 정확도로 사용하지 않았다.

| 모델 | 비교 | source/단계 | n/groups | 정답률 또는 lexical EM 차이 | 관측 group bootstrap 95% CI |
|---|---|---|---:|---:|---|
| qwen35_2b | tuned_v1 − base | slake/slake | 1061/71 | +18.10%p | [+14.68, +21.63]%p |
| qwen35_2b | tuned_v1 − base | vqa_rad/vqarad | 451/203 | +0.89%p | [-2.94, +4.85]%p |
| qwen35_2b | tuned_v1 − base | slake/noimage | 32/26 | +12.50%p | [+2.94, +24.14]%p |
| medgemma15_4b | tuned_v1 − base | slake/slake | 1061/71 | +32.99%p | [+29.59, +36.45]%p |
| medgemma15_4b | tuned_v1 − base | vqa_rad/vqarad | 451/203 | +5.10%p | [+1.08, +9.03]%p |
| medgemma15_4b | tuned_v1 − base | slake/noimage | 32/26 | +21.88%p | [+9.38, +36.67]%p |
| qwen3_17b | en_v1 − base | aihub_71875/text | 777/777 | +32.95%p | [+29.09, +36.68]%p |
| qwen3_17b | en_v1 − base | aihub_71875/text | 180/180 | +1.11%p | [+0.00, +2.78]%p |
| qwen3_17b | en_v1 − base | kormedmcqa/text | 3009/3009 | +20.07%p | [+18.28, +22.00]%p |
| qwen3_17b | en_v1 − base | medquad/text | 64/59 | +3.12%p | [+0.00, +8.06]%p |
| qwen3_17b | en_v1 − base | pubmedqa_labeled/text | 500/500 | +32.60%p | [+27.40, +38.00]%p |
| qwen3_17b | ko − base | aihub_71875/text | 777/777 | +39.64%p | [+35.65, +43.63]%p |
| qwen3_17b | ko − base | aihub_71875/text | 180/180 | +1.67%p | [+0.00, +3.89]%p |
| qwen3_17b | ko − base | kormedmcqa/text | 3009/3009 | +18.98%p | [+16.92, +21.04]%p |
| qwen3_17b | ko − base | medquad/text | 64/59 | +0.00%p | [+0.00, +0.00]%p |
| qwen3_17b | ko − base | pubmedqa_labeled/text | 500/500 | -8.00%p | [-12.00, -4.00]%p |
| qwen3_17b | mix − base | aihub_71875/text | 777/777 | +37.71%p | [+33.59, +41.83]%p |
| qwen3_17b | mix − base | aihub_71875/text | 180/180 | +2.78%p | [+0.56, +5.56]%p |
| qwen3_17b | mix − base | kormedmcqa/text | 3009/3009 | +20.31%p | [+18.25, +22.37]%p |
| qwen3_17b | mix − base | medquad/text | 64/59 | +3.12%p | [+0.00, +8.06]%p |
| qwen3_17b | mix − base | pubmedqa_labeled/text | 500/500 | +28.80%p | [+23.80, +34.00]%p |
| qwen3_4b | en_v1 − base | aihub_71875/text | 777/777 | +9.91%p | [+6.95, +12.87]%p |
| qwen3_4b | en_v1 − base | aihub_71875/text | 180/180 | +0.00%p | [+0.00, +0.00]%p |
| qwen3_4b | en_v1 − base | kormedmcqa/text | 3009/3009 | +10.77%p | [+9.17, +12.36]%p |
| qwen3_4b | en_v1 − base | medquad/text | 64/59 | +3.12%p | [+0.00, +8.06]%p |
| qwen3_4b | en_v1 − base | pubmedqa_labeled/text | 500/500 | +41.40%p | [+35.80, +46.80]%p |
| qwen3_4b | ko − base | aihub_71875/text | 777/777 | +14.41%p | [+11.07, +17.63]%p |
| qwen3_4b | ko − base | aihub_71875/text | 180/180 | +3.89%p | [+1.11, +6.67]%p |
| qwen3_4b | ko − base | kormedmcqa/text | 3009/3009 | +7.64%p | [+5.92, +9.41]%p |
| qwen3_4b | ko − base | medquad/text | 64/59 | +0.00%p | [+0.00, +0.00]%p |
| qwen3_4b | ko − base | pubmedqa_labeled/text | 500/500 | +15.60%p | [+12.00, +19.40]%p |
| qwen3_4b | mix − base | aihub_71875/text | 777/777 | +13.77%p | [+10.55, +16.86]%p |
| qwen3_4b | mix − base | aihub_71875/text | 180/180 | +2.78%p | [+0.56, +5.56]%p |
| qwen3_4b | mix − base | kormedmcqa/text | 3009/3009 | +10.40%p | [+8.81, +12.13]%p |
| qwen3_4b | mix − base | medquad/text | 64/59 | +3.12%p | [+0.00, +8.06]%p |
| qwen3_4b | mix − base | pubmedqa_labeled/text | 500/500 | +41.40%p | [+36.20, +46.60]%p |
| qwen35_2b | base_no_image − base_image | slake/image_diagnostic | 32/26 | -43.75%p | [-60.61, -27.78]%p |
| qwen35_2b | tuned_v1_no_image − tuned_v1_image | slake/image_diagnostic | 32/26 | -40.62%p | [-59.38, -22.58]%p |
| medgemma15_4b | base_no_image − base_image | slake/image_diagnostic | 32/26 | -34.38%p | [-51.43, -16.13]%p |
| medgemma15_4b | tuned_v1_no_image − tuned_v1_image | slake/image_diagnostic | 32/26 | -43.75%p | [-62.50, -22.58]%p |

이미지 진단은 같은 32개 ID의 image/no-image 결과만 비교한다. native no-image 실행 실패는 지원을 가정하거나 blank image로 바꾸지 않는다. VQA-RAD 외부 source 성능은 임상 일반화의 증명이 아니다.

| 평가 run | 상태 | 전체 E2E latency 중앙값(초) | peak allocated/reserved(GiB) |
|---|---|---:|---:|
| qwen35_2b-slake-base-v2-20261026 | SUCCEEDED | 0.2408 | 1.995/2.055 |
| qwen35_2b-slake-tuned-v2-20261026 | SUCCEEDED | 0.2730 | 2.015/2.074 |
| qwen35_2b-vqarad-base-v2-20261026 | SUCCEEDED | 0.3932 | 2.039/2.172 |
| qwen35_2b-vqarad-tuned-v2-20261026 | SUCCEEDED | 0.4504 | 2.059/2.193 |
| qwen35_2b-noimage-base-v2-20261026 | SUCCEEDED | 0.1007 | 1.829/1.863 |
| qwen35_2b-noimage-tuned-v2-20261026 | SUCCEEDED | 0.1305 | 1.849/1.883 |
| medgemma15_4b-slake-base-v2-20261026 | SUCCEEDED | 0.5620 | 3.151/3.307 |
| medgemma15_4b-slake-tuned-v2-20261026 | SUCCEEDED | 0.5722 | 3.207/3.357 |
| medgemma15_4b-vqarad-base-v2-20261026 | SUCCEEDED | 0.5377 | 3.153/3.309 |
| medgemma15_4b-vqarad-tuned-v2-20261026 | SUCCEEDED | 0.5704 | 3.208/3.342 |
| medgemma15_4b-noimage-base-v2-20261026 | SUCCEEDED | 0.2474 | 3.059/3.125 |
| medgemma15_4b-noimage-tuned-v2-20261026 | SUCCEEDED | 0.1271 | 3.090/3.188 |
| qwen3_17b-base-text-v2-20261026 | SUCCEEDED | 0.0999 | 1.393/2.232 |
| qwen3_17b-en_v1-text-v2-20261026 | SUCCEEDED | 0.1010 | 1.438/2.264 |
| qwen3_17b-ko-text-v2-20261026 | SUCCEEDED | 0.1012 | 1.438/2.264 |
| qwen3_17b-mix-text-v2-20261026 | SUCCEEDED | 0.1011 | 1.438/2.264 |
| qwen3_4b-base-text-v2-20261026 | SUCCEEDED | 0.1526 | 2.703/2.830 |
| qwen3_4b-en_v1-text-v2-20261026 | SUCCEEDED | 0.1906 | 2.767/2.914 |
| qwen3_4b-ko-text-v2-20261026 | SUCCEEDED | 0.1908 | 2.767/2.914 |
| qwen3_4b-mix-text-v2-20261026 | SUCCEEDED | 0.1912 | 2.767/2.914 |
| medgemma15_4b-base-text-v2-20261026 | SUCCEEDED | 0.1816 | 3.213/3.373 |

위 latency는 실제 측정된 모든 예측의 E2E 중앙값이다. 전처리·생성을 포함하고 모델 load·warmup을 제외한다. 정확한 source/subject별 측정 분모와 분포는 JSON에 있다. Peak memory는 전체 run의 측정값이며 decode-only 처리량이나 에너지를 뜻하지 않는다.

[v1 예측 재분석](V1_REANALYSIS_KO.md): KorMedMCQA 개선에는 invalid→valid 전환이 포함된다. both-valid 변화와 분리해 봐야 한다. MedGemma Base의 v1 SLAKE EM 0%는 출력 형태·32-token 상한의 영향을 받으며 의료 능력 0%를 뜻하지 않는다. 정답 구절 포함은 모순·부정에도 생길 수 있다. v1의 준비 8,450→처리 8,400은 50개 CONTEXT_OVERFLOW 제외였으며 시간·step 상한 때문이 아니다.

| 동일 문항 교집합 | 조건 | OPEN/CLOSED | n | v1 strict EM | v2 strict EM | v2 normalized EM |
|---|---|---|---:|---:|---:|---:|
| qwen35_2b/slake | base | ALL | 256 | 34.38% | 57.03% | 57.03% |
| qwen35_2b/slake | base | OPEN | 154 | 27.27% | 44.81% | 44.81% |
| qwen35_2b/slake | base | CLOSED | 102 | 45.10% | 75.49% | 75.49% |
| qwen35_2b/slake | tuned_v1 | ALL | 256 | 72.27% | 70.31% | 70.31% |
| qwen35_2b/slake | tuned_v1 | OPEN | 154 | 64.94% | 62.99% | 62.99% |
| qwen35_2b/slake | tuned_v1 | CLOSED | 102 | 83.33% | 81.37% | 81.37% |
| qwen35_2b/vqarad | base | ALL | 64 | 29.69% | 42.19% | 42.19% |
| qwen35_2b/vqarad | base | OPEN | 22 | 0.00% | 9.09% | 9.09% |
| qwen35_2b/vqarad | base | CLOSED | 42 | 45.24% | 59.52% | 59.52% |
| qwen35_2b/vqarad | tuned_v1 | ALL | 64 | 48.44% | 42.19% | 42.19% |
| qwen35_2b/vqarad | tuned_v1 | OPEN | 22 | 0.00% | 0.00% | 0.00% |
| qwen35_2b/vqarad | tuned_v1 | CLOSED | 42 | 73.81% | 64.29% | 64.29% |
| medgemma15_4b/slake | base | ALL | 256 | 0.00% | 20.70% | 35.94% |
| medgemma15_4b/slake | base | OPEN | 154 | 0.00% | 14.29% | 14.29% |
| medgemma15_4b/slake | base | CLOSED | 102 | 0.00% | 30.39% | 68.63% |
| medgemma15_4b/slake | tuned_v1 | ALL | 256 | 69.92% | 70.70% | 70.70% |
| medgemma15_4b/slake | tuned_v1 | OPEN | 154 | 64.29% | 62.99% | 62.99% |
| medgemma15_4b/slake | tuned_v1 | CLOSED | 102 | 78.43% | 82.35% | 82.35% |
| medgemma15_4b/vqarad | base | ALL | 64 | 0.00% | 20.31% | 43.75% |
| medgemma15_4b/vqarad | base | OPEN | 22 | 0.00% | 0.00% | 13.64% |
| medgemma15_4b/vqarad | base | CLOSED | 42 | 0.00% | 30.95% | 59.52% |
| medgemma15_4b/vqarad | tuned_v1 | ALL | 64 | 46.88% | 40.62% | 40.62% |
| medgemma15_4b/vqarad | tuned_v1 | OPEN | 22 | 4.55% | 4.55% | 4.55% |
| medgemma15_4b/vqarad | tuned_v1 | CLOSED | 42 | 69.05% | 59.52% | 59.52% |

위 표는 v1과 v2의 동일 ID·입력·정답 교집합만 재집계한 것이다. v1 부분집합과 v2 전체집합을 직접 빼지 않는다. prompt·생성 상한·정규화가 함께 바뀐 프로토콜 묶음 효과이며 특정 변수 하나의 원인으로 단정하지 않는다.

현재 resource policy: **unlimited** — 누적 GPU 시간 상한 없음. total/model/run/job/campaign 시간 상한은 {"total_gpu_hours": null, "per_model_gpu_hours": null, "run_wall_seconds": null, "job_timeout_seconds": null, "campaign_wall_seconds": null}. v1의 24 GPU-hour/모델별 6-hour는 과거 실행 당시 설정이다.

실제 누적 정산 사용 14.8306시간; v1 기준 5.5687시간; 정산 증가 9.2618시간. 진행 중 미정산 세션의 읽기 시점 경과 0.0000시간은 누적 정산값과 분리했다. 사용량 ledger를 초기화하거나 수정하지 않았다.

학습 seed는 하나다. KO/MIX는 총 문항·token·update 수가 달라 언어 구성만의 효과를 분리할 수 없다. MedGemma의 SLAKE dataset-level 사전학습 노출은 알려져 있으나 정확한 test 노출 여부는 미상이다. 일반 모델들의 과거 노출도 미상으로 남긴다. v1에서 본 test를 포함하므로 새로운 blind 결과가 아니다. 자유서술 결과에 전문가 정확도 점수는 없다.

## 다음 단계

선언된 GPU 실행 행렬과 검토표 준비를 완료했다. 아래 명령은 복구·무결성 확인을 위한 재개 절차이며, 검증된 완료 학습·평가는 재사용한다.

```bash
export MEDVLM_DATA_ROOT="$(realpath ../medvlm-data)"
PYTHONPATH=src ../.venv-medvlm/bin/python scripts/run_vqa_v2.py --data-root "$MEDVLM_DATA_ROOT" --validation-only
PYTHONPATH=src ../.venv-medvlm/bin/python scripts/run_vqa_v2.py --data-root "$MEDVLM_DATA_ROOT"
PYTHONPATH=src ../.venv-medvlm/bin/python scripts/run_text_v2.py --data-root "$MEDVLM_DATA_ROOT" --phase all
PYTHONPATH=src ../.venv-medvlm/bin/python scripts/report_v2.py --data-root "$MEDVLM_DATA_ROOT" --output reports/public_summary/v2
```

현재 캠페인이 실행 중이면 중복 실행하지 않는다. v2/vqa/state.json과 v2/text/state.json의 완료·실패 상태를 확인한 뒤 재개한다. verification.json의 실패가 있으면 해당 artifact를 조사하고 점수를 사용하지 않는다.

## 사용자가 할 일

현재 허용된 나머지 작업은 계속 진행할 수 있다. AI-Hub 71762를 이어가려면 정상적으로 확보한 원본의 로컬 경로가 필요하다. 접근 제한을 우회하지 않았고 71875 준비에 새 API key는 필요하지 않았다. Jetson은 사용자 보류 상태로 유지한다.
