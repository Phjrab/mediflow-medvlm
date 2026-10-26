# v1 예측 후속 재분석

기존 성공 예측과 동결 프로토콜을 읽어 재집계했다. 새 학습·추론은 하지 않았다. 원문·예측 본문은 공개 보고서에 포함하지 않는다.

## KorMedMCQA: 형식 유효성과 정답 변화

| 모델 | 구분 | support | Base 정답 | Tuned 정답 | 개선 | 퇴보 | 전체 정답률 기여 |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen3_17b | both_valid | 142 | 55 | 53 | 18 | 20 | -0.78%p |
| qwen3_17b | invalid_to_valid | 113 | 0 | 49 | 49 | 0 | 19.14%p |
| qwen3_17b | valid_to_invalid | 1 | 1 | 0 | 0 | 1 | -0.39%p |
| qwen3_17b | both_invalid | 0 | 0 | 0 | 0 | 0 | 0.00%p |
| qwen3_4b | both_valid | 207 | 102 | 115 | 20 | 7 | 5.08%p |
| qwen3_4b | invalid_to_valid | 48 | 0 | 26 | 26 | 0 | 10.16%p |
| qwen3_4b | valid_to_invalid | 1 | 0 | 0 | 0 | 0 | 0.00%p |
| qwen3_4b | both_invalid | 0 | 0 | 0 | 0 | 0 | 0.00%p |

유효 응답은 기존 parser의 answered 상태와 parsed answer를 기준으로 한다. invalid에는 형식 오류·응답 보류·실패가 포함되며 상세 상태 수는 JSON에 있다. both_valid의 변화는 형식 전환 없이 정답이 바뀐 부분이다. invalid→valid에서 맞힌 수는 출력 유효성 회복과 함께 발생한 변화이며 내용 지식과의 인과적 분리는 아니다.

## VQA: 답 길이·정답 구절·잘림·OPEN/CLOSED

| 모델/source | 조건 | 구분 | n | lexical EM | 구절 포함 | 잘림 | 출력 단어 중앙값 |
|---|---|---|---:|---:|---:|---:|---:|
| medgemma15_4b/slake | base | ALL | 256 | 0.00% | 55.08% | 72 | 14.0 |
| medgemma15_4b/slake | base | OPEN | 154 | 0.00% | 45.45% | 35 | 12.0 |
| medgemma15_4b/slake | base | CLOSED | 102 | 0.00% | 69.61% | 37 | 16.0 |
| medgemma15_4b/slake | tuned | ALL | 256 | 69.92% | 70.70% | 0 | 1.0 |
| medgemma15_4b/slake | tuned | OPEN | 154 | 64.29% | 65.58% | 0 | 1.0 |
| medgemma15_4b/slake | tuned | CLOSED | 102 | 78.43% | 78.43% | 0 | 1.0 |
| qwen35_2b/vqa_rad | base | ALL | 64 | 29.69% | 50.00% | 5 | 1.0 |
| qwen35_2b/vqa_rad | base | OPEN | 22 | 0.00% | 13.64% | 4 | 8.5 |
| qwen35_2b/vqa_rad | base | CLOSED | 42 | 45.24% | 69.05% | 1 | 1.0 |
| qwen35_2b/vqa_rad | tuned | ALL | 64 | 48.44% | 48.44% | 0 | 1.0 |
| qwen35_2b/vqa_rad | tuned | OPEN | 22 | 0.00% | 0.00% | 0 | 1.5 |
| qwen35_2b/vqa_rad | tuned | CLOSED | 42 | 73.81% | 73.81% | 0 | 1.0 |
| medgemma15_4b/vqa_rad | base | ALL | 64 | 0.00% | 37.50% | 19 | 16.0 |
| medgemma15_4b/vqa_rad | base | OPEN | 22 | 0.00% | 22.73% | 4 | 14.0 |
| medgemma15_4b/vqa_rad | base | CLOSED | 42 | 0.00% | 45.24% | 15 | 18.5 |
| medgemma15_4b/vqa_rad | tuned | ALL | 64 | 46.88% | 46.88% | 0 | 1.0 |
| medgemma15_4b/vqa_rad | tuned | OPEN | 22 | 4.55% | 4.55% | 0 | 2.0 |
| medgemma15_4b/vqa_rad | tuned | CLOSED | 42 | 69.05% | 69.05% | 0 | 1.0 |

구절 포함은 NFKC·대소문자·공백 정규화 후 단어 경계가 있는 연속 정답 구절의 출현이다. 부정·대안 나열·인용도 포함될 수 있으므로 정확도가 아니다. 숫자·단위·부정을 지우지 않았다. MedGemma Base의 EM 0%는 32-token 생성 상한과 답변 형태의 영향을 받으며 의료 능력 0%를 의미하지 않는다. VQA-RAD의 향상은 주로 CLOSED에서 발생했고 OPEN 22개는 매우 작은 표본이다. 자유서술 일반화 개선을 입증하지 않는다.

## 8,450 준비 문항과 실제 8,400 처리 문항

| 모델 | source | 준비/선택 | 제외 | 처리 | supervised tokens |
|---|---|---:|---:|---:|---:|
| qwen3_17b | medquad | 8000/8000 | 50 | 7950 | 1982439 |
| qwen3_17b | pubmedqa_labeled | 450/450 | 0 | 450 | 1350 |
| qwen3_4b | medquad | 8000/8000 | 50 | 7950 | 1982439 |
| qwen3_4b | pubmedqa_labeled | 450/450 | 0 | 450 | 1350 |

두 모델 모두 native template 적용 후 max_length=2048을 넘는 50문항을 통째로 제외했다. 남은 8,400개를 16개 effective batch로 525회 업데이트했고 미처리 usable 문항은 0개다. 1,000-step 상한이나 GPU 시간 상한으로 50개가 누락된 것이 아니다. 질문·답변을 잘라서 학습하지 않았다.

## 무결성과 해석 범위

예측/기존 metrics의 기록 SHA, config hash, protocol·입력·reference·이미지 byte hash, run/model revision과 기록된 native 모델 digest, adapter bytes를 검증했다. sampler 순서·preflight 제외·최종 checkpoint 처리 위치·optimizer event의 표본 및 token 합계를 대조했다. base 모델 가중치를 새로 로드하거나 전체 바이트를 재해시하지 않았다. 검증 receipt는 JSON에 있다. 전문가 검토는 수행하지 않았다.

재생성: `PYTHONPATH=src python scripts/analyze_v1_followup.py --data-root ../medvlm-data --output reports/public_summary/v2`
