# AI-Hub 71762 로컬 원본 처리

## 2026-11-12 제공처 회신과 현재 결정

사용자가 대화에 전달한 제공처 답변을 근거로 아래 사항을 추가한다. 별도 매핑 파일의 존재를 가정하거나 계속 찾아야 할 사용자 입력으로 요구하지 않는다.

- 질문별 특정 답변이 고정 1:1로 정해져 있지 않으며, 파일 일련번호에 대응 관계가 없다고 설명했다.
- 카테고리 폴더 또는 JSON의 질환명·의도 태그를 이용해 같은 분류 안의 답변을 무작위 선택하도록 안내했다. 예시는 감염성질환/HIV감염/검진이다.
- 별도 질문↔답변 매핑 자료와 증강 원본 식별자는 제공하지 않는다고 답했다.
- AI 모델 패키지의 `1. 모델 소스코드` 내 `make_train_data.py`가 split별 무작위 질문–답변 JSONL을 생성한다고 안내했다. 코드 내용, 실제 split 처리 및 난수 제어 방식은 아직 검토하지 않았다. 회신을 전달받은 것과 메일 헤더·공식 코드 원본을 직접 확인한 것을 구분한다.

이 설명은 기존 전수 검사에서 직접 foreign key와 parent가 관측되지 않았던 사실 및 활용가이드의 카테고리·의도 연결 설명과 일관된다. 그러나 무작위 쌍을 실제 대응 정답 쌍으로 해석하거나 각 개별 쌍의 의학적 적합성이 확인됐다고 표시할 근거는 아니다. 증강 원본 그룹과 그 split 격리도 보장할 수 없다.

현재 사용자 계약은 2026-10-26 후속 지시의 “파일 순서, 질환, 질문 의도만으로 연결하거나 Cartesian join하지 마라”와 “증강 질문은 원문 parent와 같은 그룹으로 유지한다”이다. 제공처의 안내를 이 조건에 대한 사용자 예외 승인으로 자동 채택하지 않는다. 사용자에게 **기존 조건 유지/71762 학습 제외**와 **71762만 별도 무작위 연결 실험 허용** 중 선택을 요청했다. 요청 후 사용자가 2026-11-12 “71762는 제공처 방식으로 별도 실험 진행”을 명시했다. 이 데이터에 한한 예외를 반영해 별도 v4 데이터·실험을 진행한다. 미제공 증강 lineage 한계도 함께 명시하며 다른 source에는 이 예외를 적용하지 않는다.

회신 직후에는 해석과 계약 대조만 수행했다. 이후 사용자 예외 허용으로 별도 pairing과 canonical freeze를 생성하고 GPU smoke를 시작했다. 완료 실적은 별도 v4 상태·보고서에 기록한다. 완료된 71875 및 다른 데이터의 v1/v2 학습·평가와 기존 v3 감사 결과는 변경하지 않는다.

### 사용자가 허용한 별도 실험 조건

2026-11-12 사용자 선택에 따라 다음 조건으로 진행한다.

1. `category_random_pairing`처럼 방식을 명시한 별도 corpus와 실험으로 관리한다. 정확한 원래 Q→A·증강 parent 매핑을 복구한 것으로 표시하지 않는다.
2. 질환 분류·질환명·의도와 원래 split을 모두 보존한다. 질문·답변의 전체 상대경로 및 내용 hash를 출처로 남기며 파일 일련번호와 전역 고유 ID를 가정하지 않는다. 응답 구조·의미·부정·수치·단위를 보존한다.
3. 후보 순서·난수 seed·선택 규칙·재사용 횟수와 결정적 표본을 고정한다. 모든 가능한 조합을 만드는 Cartesian join을 하지 않는다. 무작위 재매핑을 각 epoch마다 바꾸거나 성적에 맞춰 고르지 않는다.
4. train/validation 간 기존 정확 중복과 near/semantic 중복, KorMedMCQA 전체 보호셋과의 겹침을 검사한다. 미제공 parent를 추정 ID로 대체해 그룹 보존에 성공했다고 주장하지 않는다. 원본 lineage 부재와 남은 누수 한계를 명시한다.
5. 유한한 표본·token·update·평가 조건을 선언하고 별도 freeze를 만든 뒤 실제 smoke·학습·새 process 재로드·동일 protocol 평가로 진행한다. 무작위로 선택한 답변과의 문자열 일치를 임상 정답률이나 정답이 검증된 QA 성능으로 보고하지 않는다.

공식 코드는 구현 예시를 확인하는 데 유용하지만, 제공처가 없다고 설명한 고정 매핑이나 parent를 복원하는 자료로 기대하지 않는다. 코드가 확보되지 않았다는 이유만으로 기존 완료 실험을 반복하지 않는다.

## 보존된 2026-10-28 감사 기록

2026-10-28 후속. 이전 v2 결과는 당시 실행 기록으로 보존한다. 현재 집계는 [v3 결과](../../reports/public_summary/v3/RESULTS_KO.md), 기계 판독 상태는 [stage_status.json](../../reports/public_summary/v3/stage_status.json)이다.

`120.초거대AI 사전학습용 헬스케어 질의응답 데이터.zip`은 71762다. 4,248,340,940 bytes, SHA-256 `89584900411ac869817bd85c190ad4842f5839e9e0dc1fb85140113a148e99e4`로 고정했다. `09.필수의료 의학지식 데이터.zip`은 기존 71875 확보본과 SHA `530abb197345d03e92ebabecf7e4fd2e0a6a186c1b2043749478003e2bf8bc25`가 같으므로 다시 준비·학습하지 않았다.

71762는 TS/TL/VS/VL/Other 내부 ZIP 5개다. 전체 파일 7,028,863개를 CRC·SHA로 검사했고, label JSON 3,496,729개의 필수 구조를 검사했다. 원천·라벨은 같은 상대경로에서 내용 SHA가 모두 같다. 같은 자료를 두 번 학습량에 합치지 않는다. 공식 Training/Validation을 유지했고 원본과 증강 구성요소를 삭제하지 않았다.

`aihub71762.py`는 대규모 per-member JSON 목록을 Python 메모리에 모으지 않고 SQLite에 기록한다. 한 번에 하나의 ZIP central directory와 작은 insert batch를 처리하며 원본·정규화 내용·감사 결과를 Git 밖에 둔다. SQLite에는 question 또는 answer.intro/body/conclusion과 출처를 별도로 보존하며 demographic fields를 옮기지 않는다. 수치·단위·부정을 변경하거나 문서를 임의 QA로 만들지 않는다. 재개 identity는 원본·실행 코드·보호셋·권리 근거에 묶이고, 완료 archive 영수증을 재사용하며 중단된 archive의 파생 행만 다시 처리한다. 동일 audit를 동시에 실행하지 않는다.

## 현재 연결 한계

공식 설명서와 활용가이드는 정상 브라우저 링크로 HWP 원본을 확보했고, 파일·추출 본문 SHA를 private receipt와 공개 집계에 남겼다. 일반 HTTP의 HTML 응답을 가이드 본문으로 읽었다고 표시하지 않는다.

가이드는 별도 수집한 질문·답변을 공통 카테고리·의도로 연결하는 설명을 제공한다. 설명서의 1:1 학습셋 문구와 파일 serial 규칙만으로 실제 Q→A foreign key나 증강 parent 관계가 확정되지는 않는다. 사용자 지시는 파일 순서·질환·의도만의 연결을 제외하므로 이 방식으로 학습 쌍을 만들지 않았다. 전수 JSON에서 직접 연결 키도 관측되지 않았다. 답변 ID는 전역 고유하지 않아 전체 상대경로의 scope를 보존해야 한다.

현재 상태는 원본 전달·개별 구성요소 감사 `SUCCEEDED`, pairing `UNRESOLVED_PAIRING`, paired canonical 0, 새로운 freeze·학습·평가는 `NOT_RUN_UNRESOLVED_PAIRING`다. 구성요소 감사의 CLI 종료 코드 0은 paired SFT 준비 완료를 뜻하지 않는다. 기존 v2의 학습·재로드·평가는 보존한다.

정확 텍스트 중복과 split 간 겹침을 집계했다. KorMedMCQA 공식 dev/test 4,068개와 질문 본문의 정확 일치는 0개지만 near/semantic·번역 동등성은 paired train 후보가 생기기 전이라 검사하지 않았다. 학습 적격성이나 임상적 정확성을 확인한 자료로 표시하지 않는다.

## 재현·재개

저장소 최상위에서 기존 전용 환경을 사용한다. private 원본과 SQLite를 유지하며 성공 감사는 SHA 확인 후 재사용한다. 다음 경로는 실제 완료 receipt다. `MEDVLM_DATA_ROOT`는 기존 global data root다.

```bash
export MEDVLM_DATA_ROOT="$(realpath ../medvlm-data)"
export PYTHONPATH="$PWD/src"
# LOCAL_71762/LOCAL_71875는 제공 원본의 실제 경로다. 전달 RESUME_KO.md에 로컬 값을 기록했다.
../.venv-core310/bin/python -m mediflow_medvlm.data aihub-import \
  --data-root "$MEDVLM_DATA_ROOT/v3" --dataset-id 71762 --version 1.1 \
  --input "$LOCAL_71762"
../.venv-core310/bin/python -m mediflow_medvlm.data aihub-build \
  --data-root "$MEDVLM_DATA_ROOT/v3" \
  --import-manifest "$MEDVLM_DATA_ROOT/v3/manifests/imports/0b1ad83d92bd51d305e5ad90967e8bbf5f61efd7a4c1fcf6d2736dcd1647698a.json" \
  --protected-freeze "$MEDVLM_DATA_ROOT/manifests/freeze-1bf58dbc8baca7a3f5865accf5a86e9fba0b84f8abe3c986ed34b67d6c3d87c5.json" \
  --license-evidence 'https://www.aihub.or.kr/intrcn/guid/usagepolicy.do?currMenu=151'
../.venv-core310/bin/python scripts/report_aihub71762.py \
  --data-root "$MEDVLM_DATA_ROOT" \
  --audit-manifest "$MEDVLM_DATA_ROOT/v3/prepared/aihub_71762/ef07a830a830126d8f00669db250381f5355bfa3510c0e5ac9c9f2e39f8f00eb/audit.json" \
  --local-71875 "$LOCAL_71875" \
  --output reports/public_summary/v3
```

새 데이터나 전처리 코드를 기존 audit/source snapshot/freeze에 덮어쓰지 않는다. 실제 ID 매핑과 parent 계약이 생기면 scope·cardinality·다중 참조·미연결·증강 그룹을 먼저 검증한다. 이후 누수·native token 검사와 새 유한 corpus freeze로 진행한다. 현재 없는 `--pairing-contract` 옵션이나 완료하지 않은 새 학습 명령을 가정하지 않는다.

## 당시 요청했던 다른 컴퓨터 입력 — 현재는 위 회신이 우선

[공식 71762 페이지](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71762)의 ‘활용 AI 모델 및 코드’ → ‘AI 모델 다운로드’에서 소스코드·전처리 코드, 또는 별도 질문↔답변 매핑 CSV/JSON이 있는지 확인한다. 코드를 사용자가 직접 분석할 필요는 없다. 파일의 로컬 경로를 제공하면 실제 연결 근거를 조사할 수 있다.

핵심은 어느 scoped 질문 ID가 어느 scoped 답변 ID와 연결되는지, 증강 문항의 원본 parent는 무엇인지다. 해당 자료가 없다면 데이터 제공처에 이 관계를 정의한 파일·규칙의 제공 여부를 확인해야 한다. 데이터 ZIP·설명서·활용가이드는 이미 확보했으므로 다시 요구하지 않는다. API 키나 모델 가중치도 이 확인의 필수 입력이 아니다.

## 별도 v4 실험 — 데이터 동결과 실행

공식 코드를 내려받아 실행한 것이 아니라 제공처가 설명한 category-random 방식을 별도 구현했다. 정규화된 질환 분류·질환명·의도와 split을 모두 일치시킨다. 카테고리별 질문4개/답변8개 hash priority reservoir를 만든 뒤 질문당 답변 하나를 seed20261023으로 선택한다. 전체 원본에 대한 균일 추출이 아니라 명시된 유한 표본이며 답변 재사용을 기록한다. serial 대응·Cartesian 조합·epoch별 재추첨은 없다.

- 원본 label 구성요소 전체에서 split 간 동일 질문28,930행과 답변10행을 제외했다. 추가 동일 질문45,671행·답변30행을 하나로 줄였다. 기존 SQLite는 읽기 전용이다.
- 19개 질환 분류를 포함한 train 후보12,000 및 공식 Validation에서 exact-text hash parity로 파생한 dev/test 후보각512개를 만들었다. 공식 Test를 받은 것으로 표시하지 않는다.
- 두 Qwen의 native token 경계를 확인하고, KorMedMCQA 전체4,068 및 기존 공통 평가 자료를 합친 고유5,589개와 lexical/의미 유사도 검사를 수행했다. train 대 dev/test도 검사했다. 고정E5 .92/top5 후보2,982개를 보수적으로 격리했으며 확정 중복·의학적 동등성으로 부르지 않는다.
- 최종 train4,000/dev105/test256을 별도 freeze에 고정했다. train 정답토큰1,093,637/전체sequence1,298,700은 두 모델에서 동일하다. 양쪽 모두 원본 Base에서 독립 SFT하며, smoke20과 본학습250updates/1pass 및 각각의 새 process 재로드를 선언했다. validation48개로 checkpoint를 고른다.
- 새 파생평가256개는 Base/Tuned에512-token 생성 상한을 사전에 고정한다. 무작위 참조 답변과의 일치는 의학적 정답률이 아니다. 기존 공통4,530개는 원래 v2 protocol/config를 그대로 재사용하며 예전 Base 결과를 검증 후 대조한다. 추가 seed·curriculum·전체 원본 학습은 선언하지 않았다.

실제 private 파일은 global data root의 `v4/`에 있으며 실행 명령은 다음과 같다. source·승인·freeze hash가 달라지면 성공 파일에 덮어쓰지 않는다.

```bash
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_DISABLED=true
../.venv-medvlm/bin/python scripts/prepare_71762_v4.py --data-root ../medvlm-data
../.venv-medvlm/bin/python scripts/run_71762_v4.py --data-root ../medvlm-data --phase all
```

GPU 정책은 unlimited/null이고 사용 원장·checkpoint는 보존한다. 성공한 학습·재로드·평가는 검증 후 재사용한다. 원본 parent가 없으므로 누수 완전 배제나 독립 표본·임상 정답성을 보장하지 않는다.

## v4 실제 종료 — 2026-11-13 01:55 KST

선언한 두 모델의 smoke20/full250update·새 process 재로드4단계와 새 평가6개를 완료했고 전체 검증을 통과했다. 파생 평가256개씩의 lexical F1은1.7B 0.0864→0.0914,4B 0.1039→0.1148이다. 길이·문체·무작위 참조의 영향을 포함하므로 임상 정확도나 검증된 QA 적합성 지표가 아니다. 공통 객관식·분류형4,286개/모델은 모두 형식 미준수로 엄격한 점수0이며 임상능력0으로 해석하지 않는다. 조건·parser·문항·checkpoint를 결과에 맞춰 바꾸지 않았다.

생성기록10,084건, 기술실패0, 기존 Base 대조2개와 native prompt 일치 검증, 역사 파일77개 hash 보존을 확인했다. 사람 검토0, 공식 코드 미확보·독립 구현, 미제공 parent와 기반모델 과거노출 미상 한계를 유지한다. 결과·그림·전달물은 v4에만 저장하고 이전 v1/v2/v3를 재생성하지 않는다. 자세한 결과와 재개 명령은 `reports/public_summary/v4/RESULTS_KO.md`에 있다.
