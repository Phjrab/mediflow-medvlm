# 의료 QA·VQA 데이터 확보 및 검증 결과

기준일: **2026-10-23, Asia/Seoul**. 이 문서는 실제 source/build/audit/freeze manifest와 `reports/public_summary/{data_sources,data_quality,data_freeze,vision_sources}.json`을 대조한 데이터 단계 보고다. 원문 질문·답변·이미지는 포함하지 않는다.

**우선 데이터 4종의 준비를 완료했고, 추가 VQA-RAD 외부 평가 자료도 확보·검증했다.** 최종 aggregate는 통합본 복제를 제외한 **19,866개 canonical sample**이다. 원본 수집 합계 72,206행과 학습·평가에 사용하는 정제 자료 수량은 서로 다르다. 데이터 단계는 `SUCCEEDED / CPU_TESTED`이며, 이 결과만으로 모델 학습·의료 성능 평가가 끝났다고 의미하지 않는다. 실제 학습에 선택된 행·토큰·step과 모델 평가는 각 run manifest의 결과를 따른다.

## 1. 실제 확보량과 최종 사용량

| 원천 | 실제 원본 QA 행 | 최종 train | validation | test | 최종 합계 | v1 역할 |
|---|---:|---:|---:|---:|---:|---|
| MedQuAD | 47,441 | 8,000 | 1,494 | 1,301 | 10,795 | 공개 답변 SFT·자체 held-out 평가 |
| PubMedQA PQA-L | 1,000 | 450 | 50 | 500 | 1,000 | 문맥 기반 yes/no/maybe |
| KorMedMCQA | 7,489 | 0 | 1,059 | 3,009 | 4,068 | 한국어 평가 전용 |
| SLAKE | 14,028 | 1,979 | 512 | 1,061 | 3,552 | 영어 영상 QA SFT·평가 |
| VQA-RAD | 2,248 | 0 | 0 | 451 | 451 | 추가 외부 평가 전용 |
| **합계** | **72,206** | **10,429** | **3,115** | **6,322** | **19,866** | 원천별 canonical을 한 번씩 계산 |

원본 72,206행에는 MedQuAD의 답변 제거 행, KorMedMCQA fewshot 중복, SLAKE 중국어, VQA-RAD의 평가 외 문항도 포함한다. 독립 학습 문항 72,206개라는 뜻이 아니다. 최종 19,866행은 sample ID가 모두 고유하고, 최종 감사에서 동일 normalized input 중복도 없었다.

SLAKE 실제 참조 이미지 **642개**, VQA-RAD 공식 이미지 **315개**를 내려받아 모두 decode하고 SHA-256을 확인했다. VQA-RAD의 전체 QA는 이미지 314개를 참조하며, 외부 test 451 QA는 203개 이미지를 사용한다. 영상 QA 준비 여부는 경로 문자열이 아닌 실제 파일과 decode 결과로 확인했다. 원본 파일 manifest의 바이트 합은 **276,039,874 bytes**이며, 압축 해제본·Git metadata·모델·실험 출력까지 포함한 전체 디스크 사용량은 아니다.

`processed/text`는 MedQuAD와 PQA-L을 합친 **8,450 / 1,544 / 1,801행**의 편의용 통합본이다. 위 합계에 다시 더하지 않는다. KorMedMCQA와 VQA-RAD는 SFT에 포함하지 않는다. SLAKE 학습은 텍스트 SFT와 분리한다.

## 2. 두 freeze의 범위와 동일성

| 구분 | 원천과 목적 | canonical 수 | freeze ID |
|---|---|---:|---|
| **core v1 freeze** | MedQuAD·PQA-L·KorMedMCQA·SLAKE. 기존 학습·평가가 참조하는 고정본 | **19,415** | `1bf58dbc8baca7a3f5865accf5a86e9fba0b84f8abe3c986ed34b67d6c3d87c5` |
| **추가 aggregate freeze** | core 4종에 VQA-RAD 외부 test 451행만 추가 | **19,866** | `1562258c4b5b6f3b4c5c8948851fbb972da603cd63bff528aa7d278cf2253c86` |

두 manifest를 직접 비교한 결과, **core에 있던 모든 source split과 text 통합 split의 파일 SHA-256이 aggregate에서도 동일**하다. VQA-RAD 추가로 기존 train/dev/test를 다시 나누거나 기존 core bytes를 변경하지 않았다. core의 train/validation/test 합계는 10,429 / 3,115 / 5,871이며, aggregate에서는 test만 451행 늘어난다.

각 freeze는 `manifests/freeze-{전체ID}.json`과 `frozen/{파일SHA256}/{split}.jsonl`의 불변 사본을 보존한다. `current_freeze.json`은 현재 aggregate를 가리키는 편의 포인터이며, 기존 실행을 재현할 때 고정 core manifest 대신 무조건 최신 포인터를 사용하면 안 된다. 원본 revision, ordered sample IDs, input/original/group/image digest, transformation code SHA, seed와 감사 SHA도 manifest에 들어 있다.

`data_quality.json`은 core freeze `1bf58…`에 대한 텍스트 길이·구성 보고다. `data_freeze.json`의 최종 전체 수량은 aggregate `4dc1…` 기준이다. 서로 다른 범위의 문서를 같은 시점의 동일 데이터셋으로 해석하지 않는다.

주요 파일 SHA-256은 다음과 같다. 전체 source별 train/dev/test와 group/image digest는 freeze manifest에 보존한다.

| 고정 파일 | 행 수 | SHA-256 |
|---|---:|---|
| text/train | 8,450 | `b9455396d86410bef87415ab8f1b30b1fa4169e5c8a0585fcf080768fc62c2b8` |
| text/validation | 1,544 | `49a662ca502771ef0daceb1886ad07a07ebfa09cbf6aeca59f264c8a64ca6ae0` |
| text/test | 1,801 | `aed7188ac53f5833c606e1af27004bd700630c7041e1c53756da6f68bafb069e` |
| SLAKE/train | 1,979 | `1810b04cd4a435d58a7464366a3c0d74f7ece6d621c69f3830039d17657d3418` |
| SLAKE/validation | 512 | `24a116369a69805616c67539c7e1f5f99810e630a11db1602696f598f54ea1c6` |
| SLAKE/test | 1,061 | `7a41364e82af44ac6cac76ba5254f5012f8ef160b9ec56bb0dc0b9c02f59919f` |
| KorMedMCQA/test | 3,009 | `5c4567bf8770dac8f1d111421baf47627a2ea1d696995dd4c878d76722bbc971` |
| VQA-RAD/test | 451 | `0ab3919e0afd4ac5ade7599c12bc4db680a33c8f854f4eecc980d43b5b6cc3a7` |

## 3. 원천별 정제와 분할 결정

### MedQuAD

README 소개 수량 47,457 대신, 고정 revision의 XML에서 실제 읽은 **47,441행**을 분모로 사용한다. 저작권상 답변이 제거된 `10_MPlus_ADAM_QA`, `11_MPlusDrugs_QA`, `12_MPlusHerbsSupplements_QA`의 31,029행과 나머지 빈 답변 5행을 제외했다. 남은 공개 nonempty 답변은 16,407행이다.

동일 입력·다른 정답 2,283행을 격리하고 동일 입력·동일 정답 20행을 제거했다. 보수적인 label conflict 기준이므로 표현만 다른 유효한 답변도 제외될 수 있다. 문서와 exact-input 연결 요소를 기준으로 seed=20261023의 80/10/10 hash 분할을 적용하고, train 8,000 상한을 맞추기 위해 전체 group 단위로 3,309행을 제외했다. 최종은 **8,000 / 1,494 / 1,301**이다. 행을 임의 이동해 정확한 비율만 맞추지 않았다.

원본 qid가 여러 XML 조각에서 재사용된 사례는 파일/row 위치를 sample identity에 포함해 구분하고 문서 group은 유지했다. 제외된 답변을 원 사이트에서 재수집하거나 모델로 복구하지 않았다. LiveQA 자료는 raw 보존 대상으로만 두고 학습 XML glob에 포함하지 않았다.

### PubMedQA PQA-L

PQA-L 1,000행을 모두 사용했다. 저자 split script의 **seed=0, fold=0**을 비파괴 구현했으며, 공식 함수 부분의 실행 결과와 PMID 집합이 일치하는 **train 450 / validation 50 / test 500**을 확인했다. 이 seed는 MedQuAD의 연구 seed=20261023과 다른 공식 알고리즘 값이다. 원본 script의 기존 디렉터리 삭제 부분은 실행하지 않았다.

입력에는 question과 허용된 context만 들어간다. `long_answer`, `final_decision`, reasoning/classifier prediction은 입력과 분리했다. 결론 label 또는 정답/질문과 동일한 context segment를 제외하는 방어 규칙을 구현했으며, 이번 원본에서 해당 사유로 제거된 context는 0개다. PQA-A와 PQA-U는 확보·학습하지 않았다. 전체 label 분포는 yes 552 / no 338 / maybe 110이며, 이는 모든 split 합계다.

### KorMedMCQA

공식 parquet 16개와 카드를 확보했다. 7,489행에서 실제 original ID와 normalized input을 각각 검사하여 **7,469개 고유 문항**과 dev↔fewshot **20행 중복**을 확인했다. 원본 train 3,401행과 fewshot 20행은 보존·감사에만 사용하고 SFT/zero-shot 채점에서 제외했다. 공식 `dev`는 canonical `validation`으로 매핑하되 원래 이름을 보존한다. 답 1~5는 A~E로 명시 변환했고 `cot`를 입력에 넣지 않았다.

| 공식 배포 직군 | 원본 train | dev → validation | test | fewshot |
|---|---:|---:|---:|---:|
| doctor | 1,890 | 164 | 435 | 5 |
| nurse | 582 | 291 | 878 | 5 |
| pharm | 632 | 300 | 885 | 5 |
| dentist | 297 | 304 | 811 | 5 |

원문 metadata에는 `pharm`과 `pharmacy` 두 표기가 관찰된다. 사용되는 dev/test 합계에서 각각 914행과 271행이며, 공식 pharm config의 1,185행에 해당한다. 원문을 변조하지 않고 보존했으므로 직군별 micro/macro 평가에서는 두 표기를 하나의 약사 직군으로 묶는 집계 규칙이 필요하다. 다섯 직군이 존재하는 것으로 해석하지 않는다.

### SLAKE

공식 원본 split QA는 9,835 / 2,099 / 2,094행, 영어 subset은 **4,919 / 1,053 / 1,061행**이다. 실제 이미지 642개를 검증했고, 서로 다른 원본 이미지 파일 사이의 exact decoded-pixel 중복은 0쌍이다. 그러나 **같은 원본 image ID 298개가 공식 split을 넘는다**. 서로 다른 이미지 파일의 exact 중복과 같은 파일의 split 간 재사용은 다른 검사다.

test 우선, 그다음 validation 우선으로 다음 train/dev 행을 제외했다.

| 제외 사유 | QA 행 수 |
|---|---:|
| 같은 원본 image ID의 split 중복 | 1,627 |
| perceptual 후보 component 연결로 추가 제외 | 1,853 |
| 영어 빈 답변 | 1 |
| 영어 cohort 밖의 중국어 | 6,995 |

최종 **1,979 / 512 / 1,061행**이며, 영어 공식 test 1,061행은 모두 유지한다. 원본 영어 train/dev 5,972행 중 3,481행을 제외하여 2,491행만 남겼다. 따라서 학습 자료 감소가 크고 원래 공개 benchmark의 학습 프로토콜과 같지 않다.

perceptual 검사는 EXIF 방향 보정→RGB→grayscale L→9×8 LANCZOS resize→가로 인접 픽셀 비교의 64-bit dHash이며 Hamming distance ≤4를 후보로 삼았다. 937 후보쌍을 connected component로 연결해 364개 component를 얻었고, 가장 큰 것은 MRI 영상 **86개**, 다음은 CT **34/23/22/16개**다. 비슷한 해부학적 모양과 절편이 연결될 수 있으므로 이 후보 전체를 실제 중복 또는 동일 환자로 확정하지 않았다. false positive로 유효 학습 행을 잃는 비용을 감수한 보수적 연구 분할이다.

보고 프로토콜은 `official_split_preserved_with_lower_priority_overlap_and_conflict_exclusion`이다. 공식 test를 학습으로 이동하지 않았고, 원래 benchmark와 동일 조건의 leaderboard 수치로 주장하지 않는다. 원본 `qid/img_id/img_name/answer_type/content_type/modality/q_lang` 및 외부 지식 관련 metadata를 보존한다. 환자·study 독립성은 미확인이다.

### VQA-RAD

웹 열람 도구의 OSF 페이지 응답은 403이었으나, 실행 호스트에서 공식 public API와 실제 다운로드는 성공했다. 원본 QA 2,248행은 freeform 1,206 / para 591 / test_freeform 308 / test_para 143이다. **원본 test 유형 451행만** 외부 평가 canonical로 만들고 나머지 1,797행은 SFT에 넣지 않았다. 315개 이미지 파일과 QA JSON의 공개 SHA-256/OSF version을 확인했다.

SLAKE 642개와 VQA-RAD 315개 이미지 사이의 exact pixel 중복은 **0개**다. VQA-RAD test의 동일 입력 중복·정답 충돌도 0개다. 다만 source 간 perceptual 동일 환자/절편 확인은 수행하지 않았다. 추가 자료의 확보 완료와 실제 모델이 451문항을 평가한 여부는 구분한다.

## 4. 권리 근거와 고정 출처

| 원천 | 고정 revision/배포 식별자 | 확인한 권리와 적용 범위 |
|---|---|---|
| MedQuAD | `577bd37b96c02d1833b2c9eed2de9f96964e96cb` | [저자 LICENSE](https://github.com/abachaa/MedQuAD/blob/577bd37b96c02d1833b2c9eed2de9f96964e96cb/LICENSE.txt)의 CC-BY-4.0 공개 배포 답변 범위. 제거된 subset 답변은 사용하지 않고 원문 source URL을 유지한다. |
| PubMedQA | `1cbae8e92f72f20c8d3747cbb3bf5bc53554d997` | [저장소 MIT](https://github.com/pubmedqa/pubmedqa/blob/1cbae8e92f72f20c8d3747cbb3bf5bc53554d997/LICENSE)와 논문 초록 원저작권을 구분한다. 초록 전체가 MIT로 재허가되었다고 주장하지 않으며 PMID·원문 권리를 보존한다. |
| KorMedMCQA | `79efd6f91edfc8036330d7a4daa88b9f2deb9a82` | [원저자 카드](https://huggingface.co/datasets/sean0042/KorMedMCQA/blob/79efd6f91edfc8036330d7a4daa88b9f2deb9a82/README.md)의 CC-BY-NC-2.0. 출처 표시와 비상업 연구 범위이며 원 시험 문제의 상업용 키오스크 허가를 뜻하지 않는다. |
| SLAKE | `a9083ce6c34ac3ffb17671a605962924d8a8f9e9` | [저자 사이트](https://www.med-vqa.com/slake/)가 직접 연결한 [공식 카드](https://huggingface.co/datasets/BoKelvin/SLAKE/blob/a9083ce6c34ac3ffb17671a605962924d8a8f9e9/README.md)의 CC-BY-4.0. 개별 원본 이미지 권리는 배포에 별도로 열거되지 않았다. |
| VQA-RAD | `osf-89kps-459aaec1e7e2b92a23f506a0d3df65b84f2d953486fac6d4a5406c1d2aacbea2` | [OSF node](https://api.osf.io/v2/nodes/89kps/)에 연결된 [CC0 1.0](https://api.osf.io/v2/licenses/563c1cf88c5e4a3877f9e96c/)과 파일별 version/hash 확인. MedPix 원문 case URL도 유지한다. |

원저자 데이터 배포 조건, repository 코드 license, 원문 기사·시험·영상 권리를 서로 같은 것으로 처리하지 않았다. 원문은 Git 밖의 연구 data root에 보존하며 Git에는 코드·설정·비식별 집계와 manifest를 남긴다. 이번 데이터 준비 과정에서 새로운 전문가 정답이나 reasoning을 생성하지 않았고, 채택 corpus의 canonical derivation은 모두 `original`이다. 이는 모든 원천의 문장을 동일한 수준의 전문가 gold로 재검증했다는 뜻은 아니다.

## 5. 품질 감사와 해석 범위

aggregate 19,866행에 대한 최종 감사 결과는 다음과 같다.

| 검사 | 결과 |
|---|---:|
| critical schema/ID/옵션/known-group 오류 | 0 |
| 동일 입력 중복 group | 0 |
| 정답 충돌 group | 0 |
| 알려진 문서·PMID·문항·이미지 group의 split 침범 | 0 |
| source 간 이미지 byte/pixel group의 split 침범 | 0 |
| 텍스트 near 후보 생성·검사 쌍 | 927,288 |
| Jaccard ≥0.95인 텍스트 후보 | 2쌍, 모두 같은 split |
| 생성된 후보쌍 전체 검사 상한 도달 | 아니오 |
| 환자 수준 독립성 검증 | 미수행·식별자 부족 |

텍스트 near 검사는 질문 word-trigram Jaccard 후보 탐색이다. hash signature, 단어 수 bucket, bucket당 비교 대상 100개 cap을 사용하므로 의미가 같은 모든 재서술을 찾는 exhaustive 검사는 아니다. 후보 2쌍은 임상적으로 동일하다고 확정하지 않았다. 비교용 Unicode/공백 정규화와 원문 저장을 분리하고, 숫자·단위·부정·좌우 정보를 임의로 지우지 않았다. VQA는 같은 질문 문구라도 이미지가 다르면 서로 다른 입력으로 취급한다.

원문/정답은 canonical의 `input`과 `supervision`에 분리했다. 평가 입력 renderer는 question/context/options/images allowlist만 읽으며, 정답·해설은 추론 입력에 붙이지 않는다. parent 및 문서/image group을 기준으로 분할 관계를 검사한다. source에서 없는 patient ID나 임상 검토 상태를 만들어 채우지 않는다.

텍스트 통합 train에서 행 수 기준 MedQuAD 비중은 **94.67%**, PQA-L은 **5.33%**다. 이것은 동결 union의 구성이지 실제 학습 sampler 또는 supervised token 비중이 아니다. MedQuAD 정답 길이는 문자 기준 median 870 / p95 3,115 / max 27,910이어서 tokenizer별 예산에 따라 학습 가능 행이 달라질 수 있다. 문자 수를 토큰 수로 바꾸어 추정하지 않았으며, 정답 target 보존·truncation·실제 토큰 수는 학습 renderer/run manifest에서 확인해야 한다.

학습 이전 모델의 데이터 노출과 이번 SFT split 누수는 별개다. MedGemma의 SLAKE dataset-level 학습 노출은 알려져 있으나 정확한 공식 test 노출은 미확인이다. 다른 모델과 원천도 근거 없이 비노출이라고 표시하지 않는다. 이번 데이터 준비를 새로운 환자에 대한 진단·치료 성능 또는 한국어 임상 상담 전반의 검증으로 확대 해석하지 않는다.

## 6. 미실행·차단 원천

| 원천 | 실행 상태 | 실제 확보 QA | 이번에 완료한 것 | 재개에 필요한 것 |
|---|---|---:|---|---|
| MedQA | BLOCKED_LICENSE | 0 | 공식 repo/Drive 경로와 repo MIT 확인, US/4_options reader 준비 | 외부 archive의 시험 QA에 적용되는 데이터 이용 조건 확인. 교과서가 함께 있는 archive를 무조건 허용된 자료로 간주하지 않음 |
| MedMCQA | BLOCKED_LICENSE | 0 | repo MIT와 HF 카드 Apache-2.0 구분, index/single-choice reader CPU 검사 | 실제 수집 artifact의 데이터 권리 범위 해결, `cop` 기준 확인. 비공개 official test gold에 dummy 정확도 계산 금지 |
| AI-Hub 71875 | BLOCKED_ACCESS | 0 | 공식 version 1.1·LLM 생성 표시·필드와 reader 준비 | 현재 승인된 계정/API key 또는 적법하게 확보한 원본과 적용 조건 |
| AI-Hub 71762 | BLOCKED_ACCESS | 0 | 공식 version 1.1 조사, 위험한 pairing 거절 검사 | 승인 원본·활용 가이드·명시 matching ID. 현 상태는 `UNRESOLVED_PAIRING` |
| ROCOv2 | PREPARED_NOT_RUN | 0 | 기존 record보다 최신 [10821435](https://zenodo.org/records/10821435), version 2.0.1·CC-BY-NC-4.0·6.36GB 규모 확인 | 별도 caption 확장 범위와 개별 이미지 license/PMCID 확인. v1 기본은 registry까지만 |
| PMC-VQA | PREPARED_NOT_RUN | 0 | [공식 배포](https://huggingface.co/datasets/RadGenome/PMC-VQA) revision·CC BY-SA/원문 CC0·CC BY·archive 크기 확인 | 별도 대용량 QA 확장, PMCID별 권리와 version/test/test_clean 구분 |

이 원천들은 실제 학습 corpus에 섞지 않았다. 한국어 SFT와 로컬 번역 진단 세트도 이 데이터 보고의 완료 항목에 포함하지 않는다. AI-Hub 접근 실패가 공개 영어 자료·KorMedMCQA 한국어 평가·SLAKE 작업을 막는 선행 조건이 되지 않도록 분리했다. source별 상세 URL·revision·이용 조건 증거는 `vision_sources.json`과 `VISION_SOURCES.md`에 있다.

## 7. 증거 위치와 재개 명령

명령은 저장소 최상위에서 실행하며 원문 data root의 기본 배치는 그 옆 `../medvlm-data`다. 다른 위치에서는 `MEDVLM_DATA_ROOT`를 해당 위치로 설정한다. source manifest는 data root의 `manifests/sources`, 변환 결과와 제외 목록은 `manifests/build`, 영상 감사는 `manifests/slake_image_audit.json` 및 `manifests/vqa_rad_image_inventory.json`에 있다. 두 freeze의 전체 digest가 이 문서에 있으므로 재실행에서 사용한 범위를 구분할 수 있다.

기존 프로젝트 가상환경에서 다음 명령을 사용한다. audit은 현재 source별 자료를 검사한다. fetch/build가 이미 성공한 원본을 이유 없이 다시 수집할 필요는 없다.

```bash
export MEDVLM_DATA_ROOT="$(realpath ../medvlm-data)"
PYTHONPATH=src python -m mediflow_medvlm.data audit --data-root "$MEDVLM_DATA_ROOT"
```

파일 손상·새 환경 등으로 원본 재확보가 필요한 경우의 명령은 다음과 같다. 기존 원본을 hash로 검사하고, 가능한 경우 재사용한다.

```bash
PYTHONPATH=src python -m mediflow_medvlm.data fetch --source all --data-root "$MEDVLM_DATA_ROOT"
PYTHONPATH=src python -m mediflow_medvlm.data build --source all --data-root "$MEDVLM_DATA_ROOT"
PYTHONPATH=src python -m mediflow_medvlm.data.slake --data-root "$MEDVLM_DATA_ROOT" --fetch
PYTHONPATH=src python -m mediflow_medvlm.data.vqarad --data-root "$MEDVLM_DATA_ROOT" --fetch
```

이 CLI의 `all`은 MedQuAD/PQA-L/KorMedMCQA 3종이다. 영상 source는 별도 명령으로 실행한다. 새 자료 변경 후에만 `python -m mediflow_medvlm.data freeze --data-root "$MEDVLM_DATA_ROOT"`를 실행해 새 freeze를 만들며, 기존 core/aggregate manifest와 불변 사본은 보존한다.

이미 성공한 공개 데이터 작업에 추가 사용자 승인은 필요하지 않다. 사용자 또는 외부 제공자의 조치가 필요한 항목은 AI-Hub의 실제 승인 자격/원본, MedQA·MedMCQA의 미해결 데이터 이용 범위다. 대용량 caption/QA 확장과 한국어 추가 학습은 이번 v1 데이터 완료 수량에 포함되지 않는다.

## 외부 VQA-RAD 영상 provenance 수정

2026-10-23 21:18 KST, 최초 외부 평가 protocol 생성에서 JPEG hash 대신 QA JSON hash가 참조되는 계약 불일치를 발견했다. 원본316파일 및451개 test 행의 실제 영상 hash는 모두 유지됐다. `provenance.image_sha256`과 transformation version만 보완하고 새 aggregate freeze를 만들었다. 질문·정답·ID·그룹·이미지·core 학습 split은 변경하지 않았다. 이전 aggregate `4dc1e59694d90858b3c3bd81f0963ec798aba09180b014308e789471422e7b07`와 test snapshot `f70e35d9273fd9ea5e2a653be6b6e5cfef654962d11e3bf0fa5a80a612a9ccfe`는 보존했다. 수정 전 외부 추론·점수는 없었으며, 기존 cap64 선택 정책을 유지했다. 상세 증거는 `reports/public_summary/vqarad_image_provenance_repair.json`이다.
