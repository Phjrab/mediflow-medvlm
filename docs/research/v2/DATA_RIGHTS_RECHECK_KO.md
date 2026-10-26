# 후속 원천 권리·확보 재검토

확인일: 2026-10-26. v1 보고서는 당시 판단을 보존한다. 아래 상태는 후속 조사와 실제 로컬 확보 결과이며, 기존 freeze·학습·예측을 변경하지 않았다. 원문은 Git 밖에만 보관한다.

## MedMCQA: 보류 해소, 공식 원본 확보와 선택적 정제 완료

[공식 팀의 댓글 1329300257](https://github.com/medmcqa/medmcqa/issues/2#issuecomment-1329300257)은 팀 논의를 거쳐 **현재 데이터셋에 MIT를 적용**한다고 명시한다. 공개 GitHub API에서 댓글 ID, MEMBER 작성자 `monk1337`, 작성·수정일 2022-11-28을 직접 확인했다. [고정 revision의 LICENSE.md](https://github.com/medmcqa/medmcqa/blob/c59ef14ca1990266c4107c7864b45a20fd93e5e0/LICENSE.md)와 함께 읽으면, 이전 HF Apache 표기와의 차이만으로 원저자 데이터 사용을 보류할 사유는 해소된다. MIT의 저작권·허가 고지를 유지한다. 배포자 설명은 개별 기출문제·교재 원저작물의 권리를 항목별로 열거한 것은 아니므로, 이번에는 저자 배포 데이터의 로컬 연구 사용으로 제한하고 원문을 재배포하지 않는다.

[공식 README](https://github.com/medmcqa/medmcqa/blob/c59ef14ca1990266c4107c7864b45a20fd93e5e0/README.md)가 연결한 [저자 Drive 파일](https://drive.google.com/uc?export=download&id=15VkJdq5eyWIkfb_aoD3oS8i4tScbHYky)을 사용했다. 제3자 HF 재가공본을 원저자 파일로 표시하지 않았다. Drive의 정상 다운로드 확인 폼을 거쳐 ZIP 55,285,460 bytes를 받았고 CRC 전체 검사를 통과했다. ZIP SHA-256은 `16c1fbc6f47d548d2af7837b18e893aa45f45c0be9bda0a9adfff3c625bf9262`다. repo revision은 `c59ef14ca1990266c4107c7864b45a20fd93e5e0`이며 외부 ZIP은 자체 SHA로 별도 고정한다. 라이선스 댓글·LICENSE·README·저자 loader의 원본 및 hash도 로컬에 보관했다.

| 구분 | 원본 행 | 단일 정답 후보 | 정제 후 전체 적격 | 이번 canonical |
|---|---:|---:|---:|---:|
| train | 182,822 | 120,765 | 120,728 | 20,000 |
| dev → validation | 4,183 | 2,816 | 2,816 | 2,816 |
| 공식 test | 6,150 | 4,134 | 채점 대상 아님 | 0 |

원본 합계는 193,155행이다. 원저자 [dataset.py](https://github.com/medmcqa/medmcqa/blob/c59ef14ca1990266c4107c7864b45a20fd93e5e0/dataset.py)는 `cop-1`로 label을 읽는다. 실제 train/dev `cop`는 모두 1–4 범위이고, 이를 A–D로 변환했다. **test 6,150행에는 cop가 없다.** test를 임의 정답으로 변환하거나 정확도를 산출하지 않는다.

train의 `choice_type=multi` 62,057행과 dev 1,367행은 이번 single-choice reader 범위에서 제외했다. 질문·선택지 순서에 무관한 exact lexical key로 train 내부 동일 입력 35행과 답변 내용 충돌 2행을 제외했다. 공식 dev/test 입력과 일치하는 train은 0행이었다. 숫자·단위·부정을 삭제하지 않는다. 원문은 모두 ZIP에 남아 있으며, 표현이 다른 의미 중복·번역 중복은 이번 optional cohort에서 검증하지 않았다.

적격 train 120,728개 중 seed=20261023과 원본 ID의 hash 순서로 20,000개를 고정했다. 나머지 100,728개는 손상·권리 문제로 폐기한 것이 아니라 **선택적 영어 추가 실험의 유한 범위 제한**이다. 이 데이터는 한국어 KO/MIX 주 corpus에 포함하지 않았으며 학습도 수행하지 않았다. 전체 MedMCQA를 학습했다고 표시하지 않는다. 설명이 없는 행의 rationale은 비워두고 생성하지 않았다.

실제 canonical 위치는 `$MEDVLM_DATA_ROOT/v2/optional/medmcqa/<ZIP-SHA>/processed/`다. 별도 manifest와 public summary `reports/public_summary/v2/medmcqa_source.json`에 member hash, 정제 수량, source별 subject 수와 canonical hash를 기록했다. 기존 `processed/` 및 `current_freeze.json`에 자동 편입하지 않는다.

```bash
export MEDVLM_DATA_ROOT="$(realpath ../medvlm-data)"
PYTHONPATH=src python -m mediflow_medvlm.data.medmcqa \
  --data-root "$MEDVLM_DATA_ROOT" \
  --archive "$MEDVLM_DATA_ROOT/raw/medmcqa_v2/data.zip" --train-cap 20000
PYTHONPATH=src python -m pytest tests/test_medmcqa.py tests/test_v1_reanalysis.py
```

원본이 없는 새 호스트에서는 위 공식 Drive 파일을 정상 경로로 받아 ZIP hash를 먼저 검증한다. reader는 예상 ZIP hash, member 목록·중복·CRC·확장 크기를 확인하고 압축 안의 JSONL을 직접 읽는다. 경로를 추출하지 않으므로 archive member의 경로 탈출을 허용하지 않는다. raw+images 25GiB와 남은 디스크 20GiB/15% 중 큰 값의 공간 보호를 유지한다.

## MedQA: QA와 교재를 분리해 한정 조사

[저자 저장소](https://github.com/jind11/MedQA)는 code and data 프로젝트이고 MIT LICENSE를 포함한다. [공식 README Data 절](https://github.com/jind11/MedQA#data)은 동일 Drive 배포에 QA와 교재가 함께 있음을 밝힌다. QA는 지역별 JSONL과 공식 train/dev/test, US 4_options 변형이고, 교재는 영어·중국어 파일로 구분된다.

이번 후속 핵심 실험은 한국어 SFT와 기존 VLM 평가이므로 MedQA를 추가 수집하거나 학습하지 않았다. 불명확한 교재는 기본 제외한다. QA-only를 후속 추가할 때 남은 확인점은 **공식 외부 US/4_options JSONL의 정확한 파일·version·hash 및 그 파일에 대한 배포 조건**이며, repo의 코드 라이선스를 교재 전체 권리로 확대 해석하지 않는다. 새 승인 요구나 무기한 권리 재조사 대기 조건으로 만들지 않는다. 현재 상태는 `PREPARED_NOT_RUN / OPTIONAL_QA_ONLY_NOT_SELECTED`, 교재는 `EXCLUDED_UNRESOLVED_UNDERLYING_RIGHTS`다.

## AI-Hub 공통 이용 근거

[AI-Hub 공식 이용정책](https://www.aihub.or.kr/intrcn/guid/usagepolicy.do?currMenu=151&topMenu=105)의 **AI 허브 개방 데이터** 조건을 적용한다. 이는 MIT/CC로 바꿔 쓸 수 없는 별도 조건이다. 연구·개발 활용과 모델 학습 목적, NIA 사업결과 출처표시, 승인 없는 제3자 제공 금지, 국외 이용·반출에 대한 별도 합의 조건을 기록한다. 사용자 제공 Drive/로컬 원본은 전달 경로이며 공식 원천은 AI-Hub다. 정상 취득한 제공 자료로 로컬 작업을 진행할 수 있고 API 키를 추가 선행 조건으로 만들지 않는다. 이 문서는 실제 원본 수신·변환 성공을 대신하지 않으며 수량과 split은 별도 import manifest를 따른다.

### 71875

[공식 데이터 페이지](https://www.aihub.or.kr/aihubdata/data/view.do?aihubDataSe=data&dataSetSn=71875)에서 version 1.1(2025-06-30), 생성 방식 LLM, q_type 1=객관식·2=단답형·3=서술형을 확인했다. schema 예시의 qa_id 앞에는 U+FEFF가 보이며, qa_id 설명은 기관 번호와 문항 고유 ID를 혼용하므로 기관을 환자·문서 그룹으로 추정하지 않는다. 원천 말뭉치와 QA 라벨은 별개이며 말뭉치를 임의 QA로 만들지 않는다.

같은 공식 페이지의 활용 AI 모델 설명은 본 과제 QA로 학습하고 `sean0042/KorMedMCQA`로 test한다고 명시한다. 동봉 모델·평가 파일을 train glob에 넣지 않으며 KorMedMCQA 전체 dev/test에 대한 오염 검사를 별도로 수행해야 한다. 소개 통계 QA 19,201과 품질 표 19,210이 다르므로 실제 다운로드 행 수를 우선한다. [설명서 링크](https://www.aihub.or.kr/file/down.do?dataMnlFileSn=25375&dataSetSn=71875&fileSn=25375)와 [활용가이드 링크](https://www.aihub.or.kr/file/down.do?cnstcPrcuseFileSn=25376&dataSetSn=71875&fileSn=25376)는 공식 페이지에서 확인했다.

### 71762

[공식 데이터 페이지](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71762)는 version 1.1(2024-10-30)이며 질문 1,431,243개와 답변 2,461,960개를 제시한다. 공개 schema의 질문은 `fileName`·question, 답변은 `fileName`·answer의 intro/body/conclusion 등으로 표현된다. 질문은 BART 의미 유지 증강, 답변은 의료진 및 의료문서 기반 자료를 포함한다. 이 수량 차이만으로 1:1 또는 특정 1:N 연결을 추론하지 않는다.

공개 표의 disease/category/intention은 연결 키의 근거가 아니다. 명시 parent/ID와 cardinality는 받은 원본·전처리 코드·가이드에서 확인한 부분만 사용해야 한다. [설명서](https://www.aihub.or.kr/file/down.do?dataMnlFileSn=21553&dataSetSn=71762&fileSn=21553)·[활용가이드](https://www.aihub.or.kr/file/down.do?cnstcPrcuseFileSn=21554&dataSetSn=71762&fileSn=21554) 공개 URL의 직접 요청은 오류 HTML을 반환했고, 정상 페이지 세션·Referer를 사용한 한 차례 재시도 역시 오류 HTML이었다. 가이드 내용을 읽었다고 주장하지 않는다. 이 제한은 사용자 제공 원본 또는 동봉 가이드의 처리 가능성을 막지 않으며, 연결 근거가 없는 행만 별도 보류한다.
