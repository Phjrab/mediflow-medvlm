# 후속 실험 읽기

## 현재 단계

2026-10-27 06:46 KST 기준으로 실행 가능한 선언 행렬은 모두 완료했다. AI-Hub 71762 원본 전달은 여전히 차단되어 전체 데이터 소스가 완료된 것은 아니다.

## 이번에 한 일

- 누적 GPU 시간 상한을 해제하고 finite 정책·원장·재개 무결성을 유지했다.
- AI-Hub 71875를 실제 확보·정제하여 canonical 17,210개를 만들고, KorMedMCQA 전체 dev/test 4,068개를 보호하는 lexical/semantic 후보 감사를 거쳐 KO 4,000개와 MIX 8,000개를 동결했다.
- Qwen3-1.7B/4B 각각 KO/MIX를 원본 Base에서 한 pass 학습했다. 네 smoke와 네 본학습 모두 adapter 갱신·새 process 재로드를 통과했다.
- 텍스트 9조건 × 동일 4,530문항 = 40,770개 예측, VQA validation 8개와 본평가·진단 12개 = 6,432개 예측을 완료했다. 47,202는 반복 조건을 합한 예측 수이며 독립 문항 수가 아니다.
- 기존 v1 예측을 재분석했고, MedMCQA 원본 확보·정제와 익명 검토표 네 개를 준비했다. VLM은 재학습하지 않았다.

## 확인 결과

| 모델 | 조건 | KorMedMCQA 전체 정답률 | Kor 유효 응답률 | AI-Hub MCQ 정답률 | PubMedQA 정답률 | 전체 잘림 |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-1.7B | base | 19.01% | 54.90% | 22.39% | 41.80% | 855 |
| Qwen3-1.7B | en_v1 | 39.08% | 99.80% | 55.34% | 74.40% | 132 |
| Qwen3-1.7B | ko | 37.99% | 100.00% | 62.03% | 33.80% | 26 |
| Qwen3-1.7B | mix | 39.32% | 100.00% | 60.10% | 70.60% | 36 |
| Qwen3-4B | base | 43.67% | 80.76% | 58.04% | 35.40% | 375 |
| Qwen3-4B | en_v1 | 54.44% | 99.83% | 67.95% | 76.80% | 78 |
| Qwen3-4B | ko | 51.31% | 100.00% | 72.46% | 51.00% | 6 |
| Qwen3-4B | mix | 54.07% | 100.00% | 71.81% | 76.80% | 27 |
| MedGemma-1.5-4B | base | 23.89% | 57.06% | 44.27% | 47.60% | 1677 |

분모는 각각 KorMedMCQA 3,009, AI-Hub 객관식 777, PubMedQA 500이다. AI-Hub 단답 83·서술 97과 MedQuAD 64의 문자열 점수, 직군별 micro/macro·invalid·answered accuracy는 [전체 결과](RESULTS_KO.md)와 [세부 지표](metrics.csv)에 있다. 기술 오류와 형식 무효·잘림은 다른 상태다. MedGemma는 VQA adapter 없이 원본 Base의 native text 경로를 사용했고, 전체 텍스트 조건에서 이미지·pixel tensor 입력이 없음을 검증했다.

텍스트 9조건은 NF4·greedy로 평가했으며 출력 상한은 객관식/문맥 QA 16 tokens, 자유서술 256 tokens로 동일하게 고정했다. 특히 낮은 유효 응답률과 많은 잘림이 있는 조건의 전체 정답률은 이 출력 형식·길이 제약을 포함한 결과다. 낮은 점수를 모델의 의료 능력 전체에 대한 판정으로 바꾸지 않는다.

MedGemma Base는 KorMedMCQA 719/3,009개 정답, 유효 응답 1717/3,009개였다. 전체 잘림은 1677/4,530개이며 AI-Hub 단답·서술 180개와 MedQuAD 64개는 모두 출력 상한에 도달했다. 이 조건의 자유서술 점수는 이러한 길이 제약과 함께 해석한다. test 결과를 본 뒤 출력 상한이나 parser를 바꾸지 않았다.

KO 단독 학습은 두 Qwen 모델에서 AI-Hub 객관식 점수를 높였지만, KorMedMCQA에서는 EN-v1보다 낮았고 영어 PubMedQA 성능도 떨어졌다. MIX는 KO보다 영어 성능을 더 잘 유지했다. KorMedMCQA의 MIX–EN 차이는 1.7B에서 +7문항, 4B에서 −11문항에 불과하다. 단일 seed이며 학습량이 달라 MIX의 우월성이나 한국어 데이터만의 효과를 입증한 결과로 볼 수 없다.

Base에서 SFT 조건으로 갈 때 유효 응답률도 크게 바뀐다. 전체 정답률 상승을 전부 내용 지식 향상으로 해석하지 않는다. v1의 양쪽 유효 응답·invalid→valid 전환은 [기존 예측 재분석](V1_REANALYSIS_KO.md)에 분리했다.

VQA의 SLAKE normalized EM은 Qwen3.5 52.40→70.50%, MedGemma 36.38→69.37%였다. VQA-RAD는 각각 47.89→48.78%, 43.24→48.34%였다. Qwen3.5 VQA-RAD OPEN은 17.32→12.29%로 낮아졌으므로 전체 점수의 작은 상승이 자유서술 개선을 뜻하지 않는다. MedGemma의 SLAKE 데이터셋 사전학습 노출은 알려져 있고 정확한 test 노출은 미상이다. 두 결과 모두 임상적 유효성의 증명이 아니다.

[텍스트 비교 그림](figures/text_v2.png) · [VQA 비교 그림](figures/vqa_v2.png). 그림은 검증된 집계만 사용하며 원문·예측을 포함하지 않는다.

KO/MIX 본학습은 각각 250/500 updates, 243,394/1,137,306 supervised tokens를 처리했다. Qwen3-4B KO는 공통 validation 48개로 step 200을 선택해 평가했다(선택 시점 3,200문항·196,330 tokens). 나머지는 마지막 step 250/500을 선택했다. 모든 test 결과는 학습·prompt·parser·seed 선택에 사용하지 않았다.

KorMedMCQA와 의미 유사한 AI-Hub 후보 3,396행은 확정 중복 판정이 아니다. 후보 train/dev 3,208행을 보수적으로 제외했으며 test 후보 188행은 원래 test 957개 안에 유지했다. 미탐지 오염과 원본 검수 수준의 불확실성은 남아 있다.

현재 누적 GPU 시간 상한 없음. 원장 정산 누적 14.8306시간, v1 이후 증가 9.2618시간이다. 이는 GPU 세션의 벽시계 점유시간이며 활용률이나 전력 측정이 아니다.

검증은 완료 평가 29개(validation 포함), 학습·재로드 8개, 검토표 4개에 대해 통과했다. CPU 회귀 234 passed·7 skipped·23 subtests passed와 별도 native 검사 41 passed·20 subtests passed를 구분한다. 해시 검증·독립 지표 재집계를 수행했으며 base 가중치 전체를 다시 해시한 것은 아니다. [실행 검증](verification.json), [구현 검증](implementation_verification.json), [보존 확인](preservation_verification.json)을 참조한다.

## 다음 단계

| 구분 | 상태 | 재개 또는 후속 조건 |
|---|---|---|
| 선언된 KO/MIX·텍스트·VQA 행렬 | 완료 | 재학습할 남은 필수 셀 없음. 재개 wrapper는 성공 파일을 검증·재사용 |
| AI-Hub 71762 | BLOCKED_DELIVERY / UNRESOLVED_PAIRING | 로컬 원본과 실제 pairing 근거를 확인한 뒤 별도 새 freeze로 준비 |
| 전문가 검토 | NOT_PERFORMED | 네 표·64비교행은 같은 16문항. 점수 칸 공란, 실제 검토 0건 |
| MedMCQA 추가 학습 | PREPARED_NOT_RUN | 확보·정제 완료, 주 KO/MIX 실험에는 미포함 |
| 추가 seed | NOT_SELECTED | 선언 행렬을 확대하지 않음 |
| Jetson | USER_DEFERRED | 이번 범위에서 제외 |
| 예비 모델·교재·대용량 VQA 확장 | 미실행 | fallback 불필요 또는 이번 범위에서 제외 |

전체 실행·복구 명령은 저장소 HANDOFF와 전달 폴더의 RESUME_KO.md에 기록했다. 과거 v1 설정·산출물은 보존하며, 새 데이터가 생겨도 완료된 v2 freeze에 덧붙이지 않는다. 최종 commit/push와 CI 증거는 전달 폴더의 git_receipt.json에 둔다.

## 사용자가 할 일

71762를 이어서 처리하려면 접근 가능한 로컬 ZIP·전체 분할 조각·원본 폴더 경로와 제공 가능한 가이드가 필요하다. Drive 링크는 이미 받았지만 4.25GB 파일은 connector 256MiB 한도와 브라우저 ERR_BLOCKED_BY_CLIENT로 확보하지 못했다. API 키를 추가 필수 조건으로 만들지 않는다. 사람 평가가 필요하면 의료 검토자를 지정하여 전달 사본을 채점하면 된다. 이미 승인된 GPU·모델 접근이나 제외된 Jetson 확인을 다시 요청하지 않는다.
