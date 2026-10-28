# AI-Hub 71762 로컬 원본 처리

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

## 다른 컴퓨터에서 확인할 입력

[공식 71762 페이지](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71762)의 ‘활용 AI 모델 및 코드’ → ‘AI 모델 다운로드’에서 소스코드·전처리 코드, 또는 별도 질문↔답변 매핑 CSV/JSON이 있는지 확인한다. 코드를 사용자가 직접 분석할 필요는 없다. 파일의 로컬 경로를 제공하면 실제 연결 근거를 조사할 수 있다.

핵심은 어느 scoped 질문 ID가 어느 scoped 답변 ID와 연결되는지, 증강 문항의 원본 parent는 무엇인지다. 해당 자료가 없다면 데이터 제공처에 이 관계를 정의한 파일·규칙의 제공 여부를 확인해야 한다. 데이터 ZIP·설명서·활용가이드는 이미 확보했으므로 다시 요구하지 않는다. API 키나 모델 가중치도 이 확인의 필수 입력이 아니다.
