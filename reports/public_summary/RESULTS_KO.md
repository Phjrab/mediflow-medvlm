# 공개 의료 QA/VQA 로컬 학습 실행 보고서

생성 시각: 2026-10-23T13:58:09.728975+00:00. 실제 artifact에서 자동 집계. 단일 학습 seed 20261023의 예비 연구 결과다.

## 실제 데이터 확보·정제

정제 canonical 19,866개. 감사 critical 오류 0개. aggregate freeze `1562258c4b5b6f3b4c5c8948851fbb972da603cd63bff528aa7d278cf2253c86`.

| 소스 | 원본 QA 행 | 학습 | 검증 | 시험 | 사용 조건 |
|---|---:|---:|---:|---:|---|
| medquad | 47441 | 8000 | 1494 | 1301 | CC-BY-4.0 |
| pubmedqa_labeled | 1000 | 450 | 50 | 500 | MIT repository distribution; underlying abstract rights retained |
| kormedmcqa | 7489 | 0 | 1059 | 3009 | CC-BY-NC-2.0 |
| slake | 14028 | 1979 | 512 | 1061 | CC-BY-4.0 author-linked dataset distribution |
| vqa_rad | 2248 | 0 | 0 | 451 | OSF project CC0-1.0 |

외부 VQA-RAD 평가를 준비할 때 이미지 해시 필드 누락을 발견해 provenance를 수정하고 새 aggregate freeze를 만들었다. 원본316파일과451개 영상 참조를 재검증했다. 질문·정답·ID·이미지·core 학습 분할은 그대로이며 이전 freeze도 보존했다. 수정 전 외부 추론·점수는 없었다. 상세 변경 이력은 `vqarad_image_provenance_repair.json`이다.

학습용 코퍼스로 MedQuAD/PubMedQA와 SLAKE를 준비했다. 실제 사용·완료 여부는 모델별 실행 표를 따른다. KorMedMCQA는 한국어 원문 평가 전용, VQA-RAD는 외부 평가 전용이다. 합성·번역 의료 학습 데이터는 추가하지 않았다. 환자 독립성은 확인할 수 없으며, 이미지 유사 후보를 보수적으로 같은 그룹에 묶어 학습 데이터를 제외했다. 소스별 정확한 제외 수·권리·revision·분할 해시는 DATA_REPORT_KO.md와 공개 manifest를 참조한다.

## 실제 학습 및 재로드

| 모델 | v1 학습 상태 / 전체 셀 상태 | 실제 update | 처리 sample 횟수 | target token | 재로드 |
|---|---|---:|---:|---:|---|
| Qwen/Qwen3-1.7B | SUCCEEDED / SUCCEEDED paired_evaluation | 525 | 8400 | 1983789 | SUCCEEDED |
| Qwen/Qwen3-4B | SUCCEEDED / SUCCEEDED paired_evaluation | 525 | 8400 | 1983789 | SUCCEEDED |
| Qwen/Qwen3.5-2B | SUCCEEDED / SUCCEEDED paired_evaluation | 124 | 1979 | 7693 | SUCCEEDED |
| Qwen/Qwen3-VL-2B-Instruct | PREPARED_NOT_RUN / PREPARED_NOT_RUN FALLBACK_NOT_NEEDED | — | — | — | PREPARED_NOT_RUN |
| Qwen/Qwen2.5-VL-3B-Instruct | PREPARED_NOT_RUN / PREPARED_NOT_RUN FALLBACK_NOT_NEEDED | — | — | — | PREPARED_NOT_RUN |
| google/medgemma-1.5-4b-it | SUCCEEDED / SUCCEEDED paired_evaluation | 124 | 1979 | 7252 | SUCCEEDED |
| google/gemma-3-4b-it | PREPARED_NOT_RUN / PREPARED_NOT_RUN FALLBACK_NOT_NEEDED | — | — | — | PREPARED_NOT_RUN |
| HuggingFaceTB/SmolVLM-500M-Instruct | PREPARED_NOT_RUN / PREPARED_NOT_RUN FALLBACK_NOT_NEEDED | — | — | — | PREPARED_NOT_RUN |

학습 횟수와 평가 완료 수는 별개다. smoke와 본 학습도 별도 run으로 보존했다. 기본 설정은 NF4 QLoRA, SmolVLM은 BF16 LoRA이며 vision encoder/projector는 동결했다. 모델별 실제 설정·모듈·메모리·adapter digest는 ADAPTER_CARDS.md와 run_summary.json에 보존했다. 선택은 고정 validation loss에만 근거하고 test는 선택에 사용하지 않았다.

## 동일 조건 Base/Tuned 시험 평가

| 모델 | source / 언어 / 직군 | 조건 | 예정/시도/완료/유효 | 주 지표 | coverage | 형식 오류 | 기술 실패 | E2E 중앙값(초) |
|---|---|---|---|---:|---:|---:|---:|---:|
| Qwen/Qwen3-1.7B | kormedmcqa / ko / dentist | base | 64/64/64/34 | 21.88% | 53.12% | 46.88% | 0.00% | 0.081 |
| Qwen/Qwen3-1.7B | kormedmcqa / ko / doctor | base | 64/64/64/44 | 21.88% | 68.75% | 31.25% | 0.00% | 0.102 |
| Qwen/Qwen3-1.7B | kormedmcqa / ko / nurse | base | 64/64/64/31 | 20.31% | 48.44% | 51.56% | 0.00% | 0.120 |
| Qwen/Qwen3-1.7B | kormedmcqa / ko / pharm | base | 64/64/64/34 | 23.44% | 53.12% | 46.88% | 0.00% | 0.100 |
| Qwen/Qwen3-1.7B | medquad / en | base | 64/64/64/64 | 0.00% | 100.00% | 0.00% | 0.00% | 2.320 |
| Qwen/Qwen3-1.7B | pubmedqa_labeled / en | base | 500/500/500/500 | 41.80% | 100.00% | 0.00% | 0.00% | 0.103 |
| Qwen/Qwen3-1.7B | kormedmcqa / ko / dentist | tuned | 64/64/64/63 | 35.94% | 98.44% | 1.56% | 0.00% | 0.106 |
| Qwen/Qwen3-1.7B | kormedmcqa / ko / doctor | tuned | 64/64/64/64 | 26.56% | 100.00% | 0.00% | 0.00% | 0.120 |
| Qwen/Qwen3-1.7B | kormedmcqa / ko / nurse | tuned | 64/64/64/64 | 59.38% | 100.00% | 0.00% | 0.00% | 0.096 |
| Qwen/Qwen3-1.7B | kormedmcqa / ko / pharm | tuned | 64/64/64/64 | 37.50% | 100.00% | 0.00% | 0.00% | 0.107 |
| Qwen/Qwen3-1.7B | medquad / en | tuned | 64/64/64/64 | 3.12% | 100.00% | 0.00% | 0.00% | 3.059 |
| Qwen/Qwen3-1.7B | pubmedqa_labeled / en | tuned | 500/500/500/500 | 74.40% | 100.00% | 0.00% | 0.00% | 0.150 |
| Qwen/Qwen3-4B | kormedmcqa / ko / dentist | base | 64/64/64/54 | 25.00% | 84.38% | 15.62% | 0.00% | 0.150 |
| Qwen/Qwen3-4B | kormedmcqa / ko / doctor | base | 64/64/64/43 | 29.69% | 67.19% | 32.81% | 0.00% | 0.226 |
| Qwen/Qwen3-4B | kormedmcqa / ko / nurse | base | 64/64/64/57 | 68.75% | 89.06% | 10.94% | 0.00% | 0.144 |
| Qwen/Qwen3-4B | kormedmcqa / ko / pharm | base | 64/64/64/54 | 35.94% | 84.38% | 15.62% | 0.00% | 0.151 |
| Qwen/Qwen3-4B | medquad / en | base | 64/64/64/64 | 0.00% | 100.00% | 0.00% | 0.00% | 3.072 |
| Qwen/Qwen3-4B | pubmedqa_labeled / en | base | 500/500/500/459 | 35.40% | 91.80% | 8.20% | 0.00% | 0.252 |
| Qwen/Qwen3-4B | kormedmcqa / ko / dentist | tuned | 64/64/64/63 | 42.19% | 98.44% | 1.56% | 0.00% | 0.191 |
| Qwen/Qwen3-4B | kormedmcqa / ko / doctor | tuned | 64/64/64/64 | 46.88% | 100.00% | 0.00% | 0.00% | 0.225 |
| Qwen/Qwen3-4B | kormedmcqa / ko / nurse | tuned | 64/64/64/64 | 78.12% | 100.00% | 0.00% | 0.00% | 0.160 |
| Qwen/Qwen3-4B | kormedmcqa / ko / pharm | tuned | 64/64/64/64 | 53.12% | 100.00% | 0.00% | 0.00% | 0.195 |
| Qwen/Qwen3-4B | medquad / en | tuned | 64/64/64/64 | 3.12% | 100.00% | 0.00% | 0.00% | 4.365 |
| Qwen/Qwen3-4B | pubmedqa_labeled / en | tuned | 500/500/500/500 | 76.80% | 100.00% | 0.00% | 0.00% | 0.302 |
| Qwen/Qwen3.5-2B | slake / en | base | 256/256/256/256 | 34.38% | 100.00% | 0.00% | 0.00% | 0.491 |
| Qwen/Qwen3.5-2B | slake / en | tuned | 256/256/256/256 | 72.27% | 100.00% | 0.00% | 0.00% | 0.285 |
| Qwen/Qwen3-VL-2B-Instruct | — | base | PREPARED_NOT_RUN | N/A | N/A | N/A | N/A | N/A |
| Qwen/Qwen3-VL-2B-Instruct | — | tuned | PREPARED_NOT_RUN | N/A | N/A | N/A | N/A | N/A |
| Qwen/Qwen2.5-VL-3B-Instruct | — | base | PREPARED_NOT_RUN | N/A | N/A | N/A | N/A | N/A |
| Qwen/Qwen2.5-VL-3B-Instruct | — | tuned | PREPARED_NOT_RUN | N/A | N/A | N/A | N/A | N/A |
| google/medgemma-1.5-4b-it | slake / en | base | 256/256/256/256 | 0.00% | 100.00% | 0.00% | 0.00% | 1.053 |
| google/medgemma-1.5-4b-it | slake / en | tuned | 256/256/256/256 | 69.92% | 100.00% | 0.00% | 0.00% | 0.588 |
| google/gemma-3-4b-it | — | base | PREPARED_NOT_RUN | N/A | N/A | N/A | N/A | N/A |
| google/gemma-3-4b-it | — | tuned | PREPARED_NOT_RUN | N/A | N/A | N/A | N/A | N/A |
| HuggingFaceTB/SmolVLM-500M-Instruct | — | base | PREPARED_NOT_RUN | N/A | N/A | N/A | N/A | N/A |
| HuggingFaceTB/SmolVLM-500M-Instruct | — | tuned | PREPARED_NOT_RUN | N/A | N/A | N/A | N/A | N/A |

PubMedQA·KorMedMCQA 주 지표는 전체 예정 문항 기준 정답률이다. MedQuAD·SLAKE 주 지표는 보수적인 문자열 exact match이며 임상 정답률이 아니다. lexical token F1, 유효 응답 기준 지표, 잘림과 실패 분모는 CSV/JSON에 함께 보존했다. 평가 대상이 없거나 실행하지 않은 셀을 0점으로 표시하지 않았다.

**MedGemma 기본 모델의 SLAKE EM 0/256은 임상 정답률 0%라는 뜻이 아니다.** 사후 형식 진단에서 기본 응답은 중앙값14단어, 정답과 tuned 응답은1단어였다. 기본72건이32-token 생성 상한에 도달했고 정상 종료184건도 엄격한 EM은0이었다. 기본141건에는 gold 문자열이 단어 경계를 지키는 구절로 포함됐지만, 포함 여부는 정답 판정이 아니다. 이미지 입력·해시·동일 prompt 연결에서 명확한 구현 오류는 발견하지 못했다. 형식과 답 길이의 불일치가 큰 영향을 준 증거이며, 내용 오류와 형식의 기여도를 분리한 수치는 없다. 따라서 0→69.92% lexical EM 변화를 같은 크기의 의학적 능력 향상으로 해석하지 않는다. 고정 NF4/32-token 조건은 공식 BF16/장문 생성 예시의 성능 재현도 아니다. 설정·parser·평가 문항은 결과를 본 뒤 변경하지 않았다. 집계 정의와 근거 해시는 `medgemma_format_diagnostic.json`에 있다.

| 모델 | source | 문항 / 관측 그룹 | Tuned − Base | 그룹 bootstrap 95% CI |
|---|---|---:|---:|---|
| Qwen/Qwen3-1.7B | kormedmcqa | 256 / 256 | 17.97%p | 11.33%p ~ 24.61%p |
| Qwen/Qwen3-1.7B | medquad | 64 / 59 | 3.12%p | 0.00%p ~ 8.06%p |
| Qwen/Qwen3-1.7B | pubmedqa_labeled | 500 / 500 | 32.60%p | 27.40%p ~ 38.00%p |
| Qwen/Qwen3-4B | kormedmcqa | 256 / 256 | 15.23%p | 10.16%p ~ 20.70%p |
| Qwen/Qwen3-4B | medquad | 64 / 59 | 3.12%p | 0.00%p ~ 8.06%p |
| Qwen/Qwen3-4B | pubmedqa_labeled | 500 / 500 | 41.40%p | 36.20%p ~ 46.80%p |
| Qwen/Qwen3.5-2B | slake | 256 / 69 | 37.89%p | 30.80%p ~ 44.98%p |
| google/medgemma-1.5-4b-it | slake | 256 / 69 | 69.92%p | 64.42%p ~ 75.42%p |

| 모델 | source | 개선 | 동일 | 퇴보 |
|---|---|---:|---:|---:|
| Qwen/Qwen3-1.7B | medquad | 2 | 62 | 0 |
| Qwen/Qwen3-1.7B | pubmedqa_labeled | 194 | 275 | 31 |
| Qwen/Qwen3-1.7B | kormedmcqa | 67 | 168 | 21 |
| Qwen/Qwen3-4B | medquad | 2 | 62 | 0 |
| Qwen/Qwen3-4B | pubmedqa_labeled | 240 | 227 | 33 |
| Qwen/Qwen3-4B | kormedmcqa | 46 | 203 | 7 |
| Qwen/Qwen3.5-2B | slake | 105 | 143 | 8 |
| google/medgemma-1.5-4b-it | slake | 179 | 77 | 0 |

개선·퇴보는 같은 문항의 이산 정답/lexical EM 변화에 대한 사후 집계다. 예시 opaque ID와 상태 변화는 run_summary.json에 보존하고 원문·응답은 로컬 제한 artifact에만 둔다. 한국어 성능 변화에는 정답 내용과 출력 형식 준수의 변화가 함께 포함된다. VQA lexical 개선에도 답변 길이·형식과 생성상한 잘림 감소가 영향을 줄 수 있으므로 시각적 추론 능력만의 변화로 단정하지 않는다. 잘림 수는 CSV/JSON에서 확인할 수 있다.

CI는 관측된 문서/이미지 그룹의 문항 표본 불확실성만 반영한다. 학습 seed 변동성·새 환자 임상 일반화는 측정하지 않았다. MedGemma의 SLAKE dataset-level 사전학습 노출은 알려져 있으나 정확한 test 노출 여부는 미상이다. 이번 실행 일반 모델들의 과거 노출도 미상으로 남겼다.

## VQA 하위집단

| 모델 | source | 조건 | 구분 | 값 | 예정/유효 | lexical EM | token F1 |
|---|---|---|---|---|---:|---:|---:|
| Qwen/Qwen3.5-2B | slake | base | answer_type | CLOSED | 102/102 | 45.10% | 46.74% |
| Qwen/Qwen3.5-2B | slake | base | answer_type | OPEN | 154/154 | 27.27% | 33.35% |
| Qwen/Qwen3.5-2B | slake | base | modality | CT | 114/114 | 28.95% | 30.77% |
| Qwen/Qwen3.5-2B | slake | base | modality | MRI | 56/56 | 42.86% | 46.42% |
| Qwen/Qwen3.5-2B | slake | base | modality | X-Ray | 86/86 | 36.05% | 44.14% |
| Qwen/Qwen3.5-2B | slake | tuned | answer_type | CLOSED | 102/102 | 83.33% | 84.10% |
| Qwen/Qwen3.5-2B | slake | tuned | answer_type | OPEN | 154/154 | 64.94% | 71.52% |
| Qwen/Qwen3.5-2B | slake | tuned | modality | CT | 114/114 | 70.18% | 72.72% |
| Qwen/Qwen3.5-2B | slake | tuned | modality | MRI | 56/56 | 71.43% | 75.07% |
| Qwen/Qwen3.5-2B | slake | tuned | modality | X-Ray | 86/86 | 75.58% | 82.54% |
| google/medgemma-1.5-4b-it | slake | base | answer_type | CLOSED | 102/102 | 0.00% | 1.85% |
| google/medgemma-1.5-4b-it | slake | base | answer_type | OPEN | 154/154 | 0.00% | 5.02% |
| google/medgemma-1.5-4b-it | slake | base | modality | CT | 114/114 | 0.00% | 2.68% |
| google/medgemma-1.5-4b-it | slake | base | modality | MRI | 56/56 | 0.00% | 2.62% |
| google/medgemma-1.5-4b-it | slake | base | modality | X-Ray | 86/86 | 0.00% | 5.91% |
| google/medgemma-1.5-4b-it | slake | tuned | answer_type | CLOSED | 102/102 | 78.43% | 78.43% |
| google/medgemma-1.5-4b-it | slake | tuned | answer_type | OPEN | 154/154 | 64.29% | 70.07% |
| google/medgemma-1.5-4b-it | slake | tuned | modality | CT | 114/114 | 67.54% | 68.86% |
| google/medgemma-1.5-4b-it | slake | tuned | modality | MRI | 56/56 | 71.43% | 73.88% |
| google/medgemma-1.5-4b-it | slake | tuned | modality | X-Ray | 86/86 | 72.09% | 79.11% |
| Qwen/Qwen3.5-2B | vqa_rad | base | answer_type | CLOSED | 42/42 | 45.24% | 46.20% |
| Qwen/Qwen3.5-2B | vqa_rad | base | answer_type | OPEN | 22/22 | 0.00% | 14.91% |
| Qwen/Qwen3.5-2B | vqa_rad | base | modality | 미상 | 64/64 | 29.69% | 35.44% |
| Qwen/Qwen3.5-2B | vqa_rad | tuned | answer_type | CLOSED | 42/42 | 73.81% | 73.81% |
| Qwen/Qwen3.5-2B | vqa_rad | tuned | answer_type | OPEN | 22/22 | 0.00% | 12.32% |
| Qwen/Qwen3.5-2B | vqa_rad | tuned | modality | 미상 | 64/64 | 48.44% | 52.67% |
| google/medgemma-1.5-4b-it | vqa_rad | base | answer_type | CLOSED | 42/42 | 0.00% | 1.22% |
| google/medgemma-1.5-4b-it | vqa_rad | base | answer_type | OPEN | 22/22 | 0.00% | 7.05% |
| google/medgemma-1.5-4b-it | vqa_rad | base | modality | 미상 | 64/64 | 0.00% | 3.22% |
| google/medgemma-1.5-4b-it | vqa_rad | tuned | answer_type | CLOSED | 42/42 | 69.05% | 69.05% |
| google/medgemma-1.5-4b-it | vqa_rad | tuned | answer_type | OPEN | 22/22 | 4.55% | 15.11% |
| google/medgemma-1.5-4b-it | vqa_rad | tuned | modality | 미상 | 64/64 | 46.88% | 50.51% |

질문 유형과 영상 종류는 원본 metadata를 사용했다. 서로 겹치는 구분이므로 행들의 예정 수를 합산하지 않는다. 작은 하위집단의 차이는 임상 일반화를 뜻하지 않는다.

## 추가 영상 진단·외부 평가

| 모델 | 조건 | 예정 문항 | 차이와 해석 |
|---|---|---:|---|
| qwen35_2b-base | 같은 모델의 이미지 포함 − 제외 | 32 | lexical EM 34.38%p; 미지원 0개 |
| qwen35_2b-tuned | 같은 모델의 이미지 포함 − 제외 | 32 | lexical EM 40.62%p; 미지원 0개 |
| medgemma15_4b-base | 같은 모델의 이미지 포함 − 제외 | 32 | lexical EM 0.00%p; 미지원 0개 |
| medgemma15_4b-tuned | 같은 모델의 이미지 포함 − 제외 | 32 | lexical EM 40.62%p; 미지원 0개 |
| Qwen/Qwen3.5-2B | VQA-RAD 외부 test, Tuned − Base | 64 | lexical EM 18.75%p, CI 9.23%p ~ 29.51%p |
| google/medgemma-1.5-4b-it | VQA-RAD 외부 test, Tuned − Base | 64 | lexical EM 46.88%p, CI 33.85%p ~ 60.29%p |

| 모델 | VQA-RAD 조건 | 예정/시도/완료/유효 | lexical EM | token F1 | 기술 실패 |
|---|---|---:|---:|---:|---:|
| Qwen/Qwen3.5-2B | base (SUCCEEDED) | 64/64/64/64 | 29.69% | 35.44% | 0.00% |
| Qwen/Qwen3.5-2B | tuned (SUCCEEDED) | 64/64/64/64 | 48.44% | 52.67% | 0.00% |
| google/medgemma-1.5-4b-it | base (SUCCEEDED) | 64/64/64/64 | 0.00% | 3.22% | 0.00% |
| google/medgemma-1.5-4b-it | tuned (SUCCEEDED) | 64/64/64/64 | 46.88% | 50.51% | 0.00% |

VQA-RAD 외부 평가는 OSF `phrase_type=test_freeform/test_para`에서 사전 정책으로 선택한64건이다. MedGemma 공식 카드가 보고한 balanced split 성적과 같은 시험이라고 주장하지 않는다.

이미지 제외 조건에는 빈 이미지를 넣지 않는다. 동일 응답만으로 시각 정보가 무시됐다고 단정하지 않는다. 추가 진단은 본 평가 및 남은 모델 예산을 우선한 뒤 실행한다. 전문가 검토표는 로컬에만 준비하며 실제 검토 점수는 비어 있다.

## 실행 비용·환경

RTX 3060 12GB에서 실제 CUDA 연산으로 학습 가능성을 확인했다. NVML kernel/userspace mismatch 때문에 nvidia-smi는 실패했지만 PyTorch forward/backward는 성공했다. 드라이버는 변경하지 않았다.

| 기반 모델 | 누적 GPU 점유(시간) |
|---|---:|
| Qwen/Qwen3-1.7B | 1.1594 |
| Qwen/Qwen3-4B | 2.1989 |
| Qwen/Qwen3.5-2B | 0.7501 |
| google/medgemma-1.5-4b-it | 1.4602 |

GPU 예산은 load/smoke/train/reload/evaluation을 포함한 단일 worker 점유 시간으로 계측한다. 상한은 전체24시간·기반모델별6시간이다. TTFT와 전력·에너지는 측정하지 않아 null이다. tokens/E2E는 전처리와 생성의 합산 처리량이며 순수 decode throughput이 아니다.

## 실제 수행·미실행·차단

- 실제 수행: 공개 원본 확보, 정제·중복/누수 감사, immutable split freeze, CPU 계약 검사, 위 표에 명시된 GPU 학습·재로드·평가.
- 접근 재확인: 사용자 새 계정과 추가 승인으로 MedGemma 1.5와 Gemma 3 접근 모두 성공. 과거401/403 시도는 이력으로 보존. 일반 VLM은 Qwen3.5→Qwen3-VL→Qwen2.5-VL, 의료 VLM은 MedGemma→Gemma3 순으로 대체한다. Smol은 아직 완료하지 못한 VLM 셀의 마지막 대안이며 두 셀로 중복 계산하지 않는다.
- 접근 차단: AIHub 71875/71762 인증 및 허용 범위 미확보.
- 권리 확인 보류: MedQA/MedMCQA의 외부 시험·교재 파일 적용 권리 미확정. ROCOv2/PMC-VQA는 등록·권리 조사만 수행.
- 장치 차단: Jetson SSH 호스트 키 검증 자료 없음. 호스트 검증을 우회하지 않았으며 장치 identity·실제 추론 결과는 없다.
- 미실행: 전문가 사실성·안전성 검토, 추가 seed, 임상 검증, 서비스 배포. 자동 점수를 전문가 판정으로 대체하지 않았다.

핵심 Python3.10 검사139passed/6skipped/6subtests와 별도 native CPU16passed 기록은 `verification.json`에 있다. 주 평가4쌍·외부 평가2쌍의 실제 예측4,560건에 대한 파일SHA, 고정 문항, 동일 pair 입력·생성 조건, 선정 adapter 연결의 최종 검증은 `final_artifact_verification.json`에 있다. 네 모델 모두 실제 의료 데이터 smoke20 update와 별도 process 재로드를 통과했다. 미실행 검사는 성공 수에 포함하지 않았다.

전용 branch는 `codex/medical-finetune-v1-20261023`이다. 최종 전달 commit·remote SHA 일치·CI 결과는 전달 폴더의 `git_receipt.json`에 별도 보존한다. 각 학습/평가 run의 code commit과 source snapshot은 당시 실행 코드의 근거이며 최종 문서 commit과 구분한다.

후속 연구의 우선순위는 준비한 익명 전문가 검토표의 사실성·안전성 판정, 그리고 별도 validation에서 응답 길이·형식을 사전 고정한 새 VQA 실험이다. 현재 결과를 보고 기존 test에 맞춘 설정 변경으로 이번 점수를 대체하지 않는다. 추가 seed는 선택적 후속 실험이며 이번 v1 완료의 선행 조건이 아니다.

## 재개

정확한 로컬 경로와 명령은 HANDOFF.md에 기록했다. 공개 저장소에서는 다음과 같이 기존 환경을 재사용한다.

```bash
export MEDVLM_DATA_ROOT="$(realpath ../medvlm-data)"
PYTHONPATH=src ../.venv-medvlm/bin/python -m mediflow_medvlm status
PYTHONPATH=src ../.venv-medvlm/bin/python scripts/complete_v1.py --data-root "$MEDVLM_DATA_ROOT"
PYTHONPATH=src ../.venv-medvlm/bin/python scripts/report_v1.py --data-root "$MEDVLM_DATA_ROOT" --output reports/public_summary --plot
```

완료한 run은 동일 config/data/model identity 확인 후 재사용한다. 진행 중인 소유 프로세스가 있으면 중복 실행하지 않는다. 예산 ledger와 checkpoint·source snapshot을 함께 보존한다.

## 실제 시험 추론 계측

| 모델 | 조건 | 계측 시도 수 | 생성 token | 합산 E2E(초) | token / 합산 E2E | E2E p95(초) | peak allocated / reserved bytes |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen/Qwen3-1.7B | base | 820 | 10751 | 241.199 | 44.573 | 2.307 | 1495727104 / 2394947584 |
| Qwen/Qwen3-1.7B | tuned | 820 | 6782 | 298.004 | 22.758 | 2.180 | 1543703552 / 2430599168 |
| Qwen/Qwen3-4B | base | 820 | 10997 | 384.215 | 28.622 | 3.063 | 2902580224 / 3038773248 |
| Qwen/Qwen3-4B | tuned | 820 | 6765 | 448.899 | 15.070 | 2.496 | 2971190272 / 3128950784 |
| Qwen/Qwen3.5-2B | base | 256 | 2923 | 131.713 | 22.192 | 1.029 | 2135234560 / 2204106752 |
| Qwen/Qwen3.5-2B | tuned | 256 | 1286 | 110.217 | 11.668 | 0.813 | 2158662144 / 2227175424 |
| google/medgemma-1.5-4b-it | base | 256 | 5275 | 276.342 | 19.089 | 1.429 | 3378230784 / 3527409664 |
| google/medgemma-1.5-4b-it | tuned | 256 | 634 | 156.544 | 4.050 | 0.735 | 3437835776 / 3588227072 |

E2E p95는 계측된 문항 지연의 nearest-rank 값이다. model load와 validation warmup은 제외하고 전처리·생성·parsing은 포함한다. peak memory는 run 전체 단위이며 source별 별도 peak로 해석하지 않는다.
