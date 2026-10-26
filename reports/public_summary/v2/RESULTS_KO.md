# v2 후속 결과

## 현재 단계

기준 시각: 2026-10-26T13:06:23.113944+00:00. 실제 파일을 읽은 시점의 상태다. 실행 중 항목의 중간 예측은 완료 점수에 포함하지 않았다.

| 단계 | 완료·무결성 검증 | 예정 | 상태 |
|---|---:|---:|---|
| VQA 본평가·이미지 진단 | 5 | 12 | {"SUCCEEDED": 5, "NOT_RUN": 6, "RUNNING": 1} |
| KO/MIX smoke·본학습·재로드 | 0 | 8 | {"NOT_RUN": 8} |
| 텍스트 평가 조건 | 0 | 9 | {"NOT_RUN": 9} |

VQA validation 결정: **SUCCEEDED**, 선택된 출력 상한 256 tokens. Validation의 길이 종료 규칙으로 128→256을 한 번 검토했다. 테스트로 상한을 선택하지 않았다.

## 이번에 한 일

원본 v1 가중치·어댑터를 재사용한 VQA 평가와 새 KO/MIX 텍스트 조건을 분리했다. 아래 수치는 완료 파일의 예측 SHA, protocol·입력·정답·이미지 SHA, config·model revision, 기록된 native 모델 digest, adapter byte hash를 검증한 뒤 다시 집계한 값이다. base 가중치 전체 재해시는 하지 않았다.

| 데이터/작업 | 실제 확보·준비 | 상태 |
|---|---|---|
| AI-Hub 71875 | {"test": 957, "train": 15297, "validation": 956} | SUCCEEDED |
| KO/MIX corpus | {"ko": 4000, "mix": 8000} | PREPARED_NOT_RUN |
| MedMCQA | {"test": 0, "train": 20000, "validation": 2816}; KO/MIX 미포함 | PREPARED_NOT_RUN |
| AI-Hub 71762 | 확인된 다운로드 0 bytes; 입력 receipt 기준 | BLOCKED_DELIVERY |
| Jetson | 사용자 보류; v2 접속·배포·benchmark 미실행 | USER_DEFERRED |
| 전문가 검토표 준비 | 0/4 packet, 0개 답변 쌍; q_type 2/3 각 8개/packet | INCOMPLETE_PREPARATION |
| 전문가 검토 수행 | 검토 0건·점수 미수집 | NOT_PERFORMED |

검토표 준비와 사람의 검토는 별도다. 준비된 표는 파일·프로토콜·출처 예측 해시와 점수 칸 공란을 확인한다. 원문이 있는 CSV 및 A/B 대응 private key는 로컬 데이터 경로에만 두며 공개 보고서에는 싣지 않는다.

71875 수집/제외 수: {"non_qa_documents": 52126, "q_type_1": 14028, "q_type_2": 1634, "q_type_3": 1618, "raw_official_split_train": 15360, "raw_official_split_validation": 1920, "raw_qa": 17280}; 제외: {"EXACT_INPUT_DUPLICATE": 1, "MCQ_ANSWER_TEXT_MISMATCH": 48, "MCQ_EMPTY_OR_DUPLICATE_OPTION": 3, "MCQ_INVALID_ANSWER_NUMBER": 1, "MCQ_OPTION_BLOCK_AMBIGUOUS": 13, "QUESTION_GROUP_OVERLAP_WITH_OFFICIAL_HELDOUT": 4}. 최종 corpus manifest가 가리키는 source audit 기준이다.

71762는 connector 256 MiB 다운로드 제한과 정상 브라우저의 ERR_BLOCKED_BY_CLIENT로 막혔다. 약 4.25 GB 원본은 확보되지 않았으며 QA pairing도 확인되지 않았다. 수집된 원본이 없는 reader 구현을 데이터 준비 완료로 계산하지 않는다.

의미 유사도 검사: KorMedMCQA 4068문항을 보호하고 E5 cosine ≥0.92, top-5에서 3396행/8278쌍을 후보로 찾았다. 이는 확정 중복이나 번역 동등성 판정이 아니다. lexical exact/near 검사와 별도다.
split별 후보 처리: {"test": {"canonical": 957, "semantic_candidate_rows": 188, "eligible_after_quarantine": 957, "test_preserved_candidates_only_flagged": true}, "train": {"canonical": 15297, "semantic_candidate_rows": 3010, "eligible_after_quarantine": 12287, "test_preserved_candidates_only_flagged": false}, "validation": {"canonical": 956, "semantic_candidate_rows": 198, "eligible_after_quarantine": 758, "test_preserved_candidates_only_flagged": false}}. train/dev 후보는 보수적으로 제외하고 test 전체 957개는 보존하며 후보 표시만 남겼다.

최종 KO q_type 선택은 {"1": 1500, "2": 1300, "3": 1200}; MIX는 같은 KO에 {"medquad": 3550, "pubmedqa_labeled": 450}를 더한다. 공통 dev는 1516개(AI-Hub 758 + EN 758)다. 최초 균등 q_type quota와 dev 1,912개에서 바뀐 이유는 고정 threshold의 후보 격리로 long/train과 dev가 줄었기 때문이다. 평가 성능으로 선택하거나 threshold를 바꾸지 않았다.

KO4,000(250 update), MIX8,000(500 update)는 seed 20261023 한 pass를 계획한다. 각각 독립 smoke 20 update와 새 프로세스 재로드를 먼저 확인한다. LR 2e-4, LoRA rank 8/alpha 16/dropout 0.05, effective batch 16이다. KO/MIX checkpoint 비교에는 동일 공통 dev 파일에서 source별 균형으로 고른 48개를 쓴다. q_type별 층화 표본을 보장하지 않는다.

| 학습 run | 상태 | 실제 update | 실제 처리 문항 | supervised tokens | 재로드 확인 |
|---|---|---:|---:|---:|---|
| qwen3_17b-ko-smoke-v2-20261026 | NOT_RUN | 미실행 | 미실행 | 미실행 | False |
| qwen3_17b-ko-full-v2-20261026 | NOT_RUN | 미실행 | 미실행 | 미실행 | False |
| qwen3_17b-mix-smoke-v2-20261026 | NOT_RUN | 미실행 | 미실행 | 미실행 | False |
| qwen3_17b-mix-full-v2-20261026 | NOT_RUN | 미실행 | 미실행 | 미실행 | False |
| qwen3_4b-ko-smoke-v2-20261026 | NOT_RUN | 미실행 | 미실행 | 미실행 | False |
| qwen3_4b-ko-full-v2-20261026 | NOT_RUN | 미실행 | 미실행 | 미실행 | False |
| qwen3_4b-mix-smoke-v2-20261026 | NOT_RUN | 미실행 | 미실행 | 미실행 | False |
| qwen3_4b-mix-full-v2-20261026 | NOT_RUN | 미실행 | 미실행 | 미실행 | False |

## 확인 결과

KorMedMCQA는 동일 전체 3,009문항의 micro/macro subject 정답률, coverage, invalid, 유효 응답 정답률을 함께 표시한다. 무응답·실패는 전체 분모에 남긴다.

| 모델/조건 | n | 전체 micro | subject macro | coverage | invalid 형식 | 유효 응답 정답률 |
|---|---:|---:|---:|---:|---:|---:|

완료된 텍스트 평가가 없으면 위 표는 비어 있다. MedGemma 텍스트 조건은 원본 Base이며 VQA adapter를 사용하지 않는다.

| 모델/조건 | AI-Hub q_type | n | 전체 MCQ 정답률 또는 lexical EM | lexical F1 |
|---|---|---:|---:|---:|

short/long의 EM/F1는 문자열 일치도이며 임상 정확도가 아니다. MCQ q_type 1, short 2, long 3을 따로 집계했다.

| 모델/조건 | 영어 retention source | n | 전체 정답률/lexical EM | lexical F1 |
|---|---|---:|---:|---:|

VQA: 두 모델의 Base/Tuned는 같은 짧은 답변 prompt와 선택된 생성 상한을 쓴다. SLAKE 1,061개, VQA-RAD 451개 전체와 고정 32개 no-image 진단을 구분한다.

| 모델/조건 | source/이미지 | OPEN/CLOSED | n | strict EM | v2 EM | v2 F1 | 잘림 | 출력 단어 중앙값 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| qwen35_2b/base | slake/image | ALL | 1061 | 52.40% | 52.40% | 56.40% | 0 | 1 |
| qwen35_2b/base | slake/image | OPEN | 645 | 36.43% | 36.43% | 42.69% | 0 | 1 |
| qwen35_2b/base | slake/image | CLOSED | 416 | 77.16% | 77.16% | 77.64% | 0 | 1.0 |
| qwen35_2b/tuned_v1 | slake/image | ALL | 1061 | 70.50% | 70.50% | 73.85% | 0 | 1 |
| qwen35_2b/tuned_v1 | slake/image | OPEN | 645 | 61.09% | 61.09% | 66.59% | 0 | 1 |
| qwen35_2b/tuned_v1 | slake/image | CLOSED | 416 | 85.10% | 85.10% | 85.10% | 0 | 1.0 |
| qwen35_2b/base | slake/no_image | ALL | 32 | 25.00% | 25.00% | 25.78% | 0 | 1.0 |
| qwen35_2b/base | slake/no_image | OPEN | 17 | 5.88% | 5.88% | 7.35% | 0 | 1 |
| qwen35_2b/base | slake/no_image | CLOSED | 15 | 46.67% | 46.67% | 46.67% | 0 | 1 |
| qwen35_2b/tuned_v1 | slake/no_image | ALL | 32 | 37.50% | 37.50% | 37.50% | 0 | 1.0 |
| qwen35_2b/tuned_v1 | slake/no_image | OPEN | 17 | 11.76% | 11.76% | 11.76% | 0 | 1 |
| qwen35_2b/tuned_v1 | slake/no_image | CLOSED | 15 | 66.67% | 66.67% | 66.67% | 0 | 1 |
| medgemma15_4b/base | slake/image | ALL | 1061 | 20.08% | 36.38% | 46.23% | 49 | 1 |
| medgemma15_4b/base | slake/image | OPEN | 645 | 13.02% | 13.49% | 29.24% | 48 | 2 |
| medgemma15_4b/base | slake/image | CLOSED | 416 | 31.01% | 71.88% | 72.58% | 1 | 1.0 |

strict는 기존 NFKC·casefold·공백 정규화이고, v2는 문장 끝 .!?。！？만 추가 제거한다. 숫자·단위·부정과 내부 구두점은 유지한다. 정답 구절 포함률을 정확도로 사용하지 않았다.

| 모델 | 비교 | source/단계 | n/groups | 정답률 또는 lexical EM 차이 | 관측 group bootstrap 95% CI |
|---|---|---|---:|---:|---|
| qwen35_2b | tuned_v1 − base | slake/slake | 1061/71 | +18.10%p | [+14.68, +21.63]%p |
| qwen35_2b | tuned_v1 − base | slake/noimage | 32/26 | +12.50%p | [+2.94, +24.14]%p |
| qwen35_2b | base_no_image − base_image | slake/image_diagnostic | 32/26 | -43.75%p | [-60.61, -27.78]%p |
| qwen35_2b | tuned_v1_no_image − tuned_v1_image | slake/image_diagnostic | 32/26 | -40.62%p | [-59.38, -22.58]%p |

이미지 진단은 같은 32개 ID의 image/no-image 결과만 비교한다. native no-image 실행 실패는 지원을 가정하거나 blank image로 바꾸지 않는다. VQA-RAD 외부 source 성능은 임상 일반화의 증명이 아니다.

| 평가 run | 상태 | 전체 E2E latency 중앙값(초) | peak allocated/reserved(GiB) |
|---|---|---:|---:|
| qwen35_2b-slake-base-v2-20261026 | SUCCEEDED | 0.2408 | 1.995/2.055 |
| qwen35_2b-slake-tuned-v2-20261026 | SUCCEEDED | 0.2730 | 2.015/2.074 |
| qwen35_2b-vqarad-base-v2-20261026 | NOT_RUN | 미측정 | 미측정 |
| qwen35_2b-vqarad-tuned-v2-20261026 | NOT_RUN | 미측정 | 미측정 |
| qwen35_2b-noimage-base-v2-20261026 | SUCCEEDED | 0.1007 | 1.829/1.863 |
| qwen35_2b-noimage-tuned-v2-20261026 | SUCCEEDED | 0.1305 | 1.849/1.883 |
| medgemma15_4b-slake-base-v2-20261026 | SUCCEEDED | 0.5620 | 3.151/3.307 |
| medgemma15_4b-slake-tuned-v2-20261026 | RUNNING | 미측정 | 미측정 |
| medgemma15_4b-vqarad-base-v2-20261026 | NOT_RUN | 미측정 | 미측정 |
| medgemma15_4b-vqarad-tuned-v2-20261026 | NOT_RUN | 미측정 | 미측정 |
| medgemma15_4b-noimage-base-v2-20261026 | NOT_RUN | 미측정 | 미측정 |
| medgemma15_4b-noimage-tuned-v2-20261026 | NOT_RUN | 미측정 | 미측정 |
| qwen3_17b-base-text-v2-20261026 | NOT_RUN | 미측정 | 미측정 |
| qwen3_17b-en_v1-text-v2-20261026 | NOT_RUN | 미측정 | 미측정 |
| qwen3_17b-ko-text-v2-20261026 | NOT_RUN | 미측정 | 미측정 |
| qwen3_17b-mix-text-v2-20261026 | NOT_RUN | 미측정 | 미측정 |
| qwen3_4b-base-text-v2-20261026 | NOT_RUN | 미측정 | 미측정 |
| qwen3_4b-en_v1-text-v2-20261026 | NOT_RUN | 미측정 | 미측정 |
| qwen3_4b-ko-text-v2-20261026 | NOT_RUN | 미측정 | 미측정 |
| qwen3_4b-mix-text-v2-20261026 | NOT_RUN | 미측정 | 미측정 |
| medgemma15_4b-base-text-v2-20261026 | NOT_RUN | 미측정 | 미측정 |

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
| medgemma15_4b/slake | base | ALL | 256 | 0.00% | 20.70% | 35.94% |
| medgemma15_4b/slake | base | OPEN | 154 | 0.00% | 14.29% | 14.29% |
| medgemma15_4b/slake | base | CLOSED | 102 | 0.00% | 30.39% | 68.63% |

위 표는 v1과 v2의 동일 ID·입력·정답 교집합만 재집계한 것이다. v1 부분집합과 v2 전체집합을 직접 빼지 않는다. prompt·생성 상한·정규화가 함께 바뀐 프로토콜 묶음 효과이며 특정 변수 하나의 원인으로 단정하지 않는다.

현재 resource policy: **unlimited**. total/model/run/job/campaign 시간 상한은 {"total_gpu_hours": null, "per_model_gpu_hours": null, "run_wall_seconds": null, "job_timeout_seconds": null, "campaign_wall_seconds": null}. v1의 24 GPU-hour/모델별 6-hour는 과거 실행 당시 설정이다.

실제 누적 정산 사용 6.1852시간; v1 기준 5.5687시간; 정산 증가 0.6165시간. 진행 중 미정산 세션의 읽기 시점 경과 0.0500시간은 누적 정산값과 분리했다. 사용량 ledger를 초기화하거나 수정하지 않았다.

학습 seed는 하나다. KO/MIX는 총 문항·token·update 수가 달라 언어 구성만의 효과를 분리할 수 없다. 모델 사전학습의 test 노출 여부는 미상이다. v1에서 본 test를 포함하므로 새로운 blind 결과가 아니다. 자유서술 결과에 전문가 정확도 점수는 없다.

## 다음 단계

미완료 조건은 동일 선언과 artifact를 사용해 재개하고, 모두 끝난 뒤 이 보고서를 다시 생성한다. 완료 조건을 새로 학습하는 명령이 아니다.

```bash
export MEDVLM_DATA_ROOT="$(realpath ../medvlm-data)"
PYTHONPATH=src python scripts/run_vqa_v2.py --data-root "$MEDVLM_DATA_ROOT" --validation-only
PYTHONPATH=src python scripts/run_vqa_v2.py --data-root "$MEDVLM_DATA_ROOT"
PYTHONPATH=src python scripts/run_text_v2.py --data-root "$MEDVLM_DATA_ROOT" --phase all
PYTHONPATH=src python scripts/report_v2.py --data-root "$MEDVLM_DATA_ROOT" --output reports/public_summary/v2
```

현재 캠페인이 실행 중이면 중복 실행하지 않는다. v2/vqa/state.json과 v2/text/state.json의 완료·실패 상태를 확인한 뒤 재개한다. verification.json의 실패가 있으면 해당 artifact를 조사하고 점수를 사용하지 않는다.

## 사용자가 할 일

현재 허용된 나머지 작업은 계속 진행할 수 있다. AI-Hub 71762를 이어가려면 정상적으로 확보한 원본의 로컬 경로가 필요하다. 접근 제한을 우회하지 않았고 71875 준비에 새 API key는 필요하지 않았다. Jetson은 사용자 보류 상태로 유지한다.
