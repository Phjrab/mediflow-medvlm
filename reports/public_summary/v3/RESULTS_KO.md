# AI-Hub 71762 로컬 원본 후속 결과

## 현재 단계

2026-10-28T09:51:15.576438+00:00 기준. 원본 전달 차단은 해소됐고 전체 파일의 CRC·JSON 스키마 감사와 개별 구성요소 정규화를 완료했다. 질문↔답변 연결은 `UNRESOLVED_PAIRING`이다. 이 문서는 새 모델 성능 보고가 아니다.

## 이번에 한 일

| 사용자 파일 | 데이터셋 | 확인 |
|---|---|---|
| 120.초거대AI 사전학습용 헬스케어 질의응답 데이터.zip | [71762](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71762) | 4,248,340,940 bytes; SHA-256 `89584900411ac869817bd85c190ad4842f5839e9e0dc1fb85140113a148e99e4` |
| 09.필수의료 의학지식 데이터.zip | 71875 | 이전 확보본과 SHA-256 `530abb197345d03e92ebabecf7e4fd2e0a6a186c1b2043749478003e2bf8bc25` 일치 |

71875는 동일 바이트이므로 다시 수집·정제·학습하지 않았다. 71762는 원본 ZIP을 유지하고 내부 TS/TL/VS/VL/Other ZIP 5개를 별도 private 경로에 설치했다. 내부 파일 7,028,863개를 전부 읽어 CRC와 개별 SHA를 검사했다. 원천·라벨 중복을 학습량으로 두 번 세지 않았다.

| 공식 제공 split | 질문 JSON | 답변 JSON | 구성요소 합계 |
|---|---:|---:|---:|
| train | 1,143,131 | 1,966,902 | 3,110,033 |
| validation | 141,292 | 245,404 | 386,696 |

총 3,496,729개 라벨 JSON이 문법·필수 필드 검사를 통과했다. 숫자는 QA 쌍 수가 아니라 질문·답변 파일 수다. 공식 제공 Training/Validation split을 유지했고 별도 Test ZIP은 이 배포 파일에서 관측되지 않았다.

TS↔TL 및 VS↔VL은 동일 상대경로의 파일 목록과 실제 내용 SHA가 모두 일치했다(누락·내용 차이 0건). 따라서 source와 label은 독립 QA 표본으로 합치지 않는다. Other는 기사 TXT 35,404개와 XLSX 1개이며 임의 QA로 변환하지 않았다. answer.intro/body/conclusion은 원문 문자열과 구조를 유지했다. 참여자 인구통계는 정규화한 구성요소에 넣지 않았고 participantID를 환자·parent ID로 사용하지 않았다.

공식 설명서와 활용가이드는 일반 HTTP 요청에서는 HTML이 반환됐지만 정상 공식 브라우저 링크로 HWP 원본을 확보했다. 파일·본문 추출 SHA를 JSON에 기록했다. 가이드 본문은 OLE BodyText의 PARA_TEXT로 읽었고 원본 HWP를 보존했다.

## 확인 결과

전수 검사한 질문의 최상위 키는 fileName/participantsInfo/disease_category/disease_name/intention/question/entities/num_of_words이고, 답변은 fileName/disease_category/disease_name/department/intention/answer/num_of_words다. 질문 ID는 HC-Q, 답변 ID는 HC-A 형식이며 직접 Q→A 연결 키와 증강 원본 parent 키는 확인되지 않았다. entities.id는 문장 내부 엔티티 식별자로 취급한다.

가이드의 데이터 동기화 절은 별도 수집한 질문·답변을 공통 카테고리·의도로 연결하는 방식을 설명한다. 설명서에는 1:1 학습셋 구성 문구가 있지만, 어느 질문 ID와 어느 답변 ID를 연결하는지와 증강 parent를 정의하는 매핑은 확보하지 못했다. 사용자가 질환·의도만의 연결을 명시적으로 제외했으므로 문서의 일반 설명을 임의 pairing의 허가로 해석하지 않았다.

답변 fileName 중 281,241개 ID가 여러 경로에서 반복된다. 이를 전역 고유 ID로 가정하면 자료를 잃거나 잘못 연결할 수 있어, 현재 inventory는 archive·전체 상대경로를 별도 보존한다. 반복 ID를 곧바로 동일 답변이나 같은 환자로 해석하지 않는다.

| NFC·공백 정규화한 구성요소 텍스트 | 중복 그룹 | 초과 행 | split 간 동일 텍스트 그룹 / 행 |
|---|---:|---:|---:|
| 질문 | 44,913 | 64,137 | 10,464 / 28,930 |
| 답변(intro/body/conclusion 결합 기준) | 35 | 35 | 5 / 10 |

공식 split 간 중복이 있어 향후 paired corpus 구축 시 parent·텍스트 그룹을 함께 보호해야 한다. 원본과 split은 감사 과정에서 변경하지 않았다. KorMedMCQA 전체 dev/test 4,068문항과 질문 본문의 정확 일치는 0건이었다. near/semantic·번역 중복 검사는 paired train 후보가 생긴 뒤 수행해야 하며 이번에는 미실행이다. 오염이 없음을 보증하는 결과로 해석하지 않는다.

CPU 회귀검사는 241 passed·7 skipped·23 subtests passed, 관련 데이터 검사는 21 passed다. ZIP·SQLite·실행 소스 SHA와 독립 DB 건수 집계를 재검증했다. 기존 v1/v2 공개 보고서와 v2 실행 소스 14개, GPU 원장이 보존됐다. 원문·HWP·SQLite·모델 가중치는 Git에 올리지 않는다. [집계 JSON](aihub_71762_data.json), [검증 근거](verification.json)를 참조한다.

## 다음 단계

| 항목 | 상태 |
|---|---|
| 원본 전달·내부 CRC 검증 | 완료 |
| JSON 스키마·개별 질문/답변 정규화·중복 감사 | 완료 |
| 질문↔답변 관계·증강 parent 확인 | UNRESOLVED_PAIRING |
| 신규 paired canonical / freeze / supervised token 감사 | 미실행: pairing 의존 |
| 71762를 사용하는 신규 학습·재로드·평가 | 미실행: pairing 의존 |
| 기존 v2 KO/MIX·텍스트·VQA | 완료 결과 보존, 재실행하지 않음 |
| 전문가 검토 / Jetson | 미수행 / 사용자 보류 |

pairing 근거가 생기면 실제 ID 범위·중복·1:1/1:N/N:1·증강 parent를 먼저 확정하고, 전체 KorMedMCQA와 기존 held-out 보호 감사를 수행한 새 freeze로 진행한다. 기존 완료 v2 corpus와 결과에 덧붙이지 않는다. 성능을 이유로 모델·seed sweep을 자동 확대하지 않는다. 현재 데이터 감사의 재개 명령은 전달 폴더 RESUME_KO.md에 있다.

## 사용자가 할 일

다른 컴퓨터에서 71762 페이지의 ‘활용 AI 모델 및 코드’ → ‘AI 모델 다운로드’를 확인하고, 소스코드·전처리 코드 또는 Q→A 매핑 CSV/JSON이 있으면 로컬 경로를 알려주면 된다. 코드는 직접 분석할 필요 없다. 필요한 근거는 scoped 질문 ID↔답변 ID 연결과 증강 원본 parent 관계다. 제공 파일이 없다면 데이터 제공처에 이 매핑의 제공 여부를 확인해야 한다. 이미 확보한 데이터 ZIP·설명서·활용가이드나 API 키를 다시 제공할 필요는 없다. 모델 가중치도 이 연결 확인에는 필요하지 않다.
