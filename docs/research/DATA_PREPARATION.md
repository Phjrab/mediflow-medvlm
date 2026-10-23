# 실제 의료 데이터 준비 및 재현

이번 자료는 공개 자료를 이용한 비상업 오프라인 비교 연구용이다. 원문·영상·모델 응답은 Git에 올리지 않는다. 코드와 비식별 집계만 공개 요약 디렉터리에 둔다. 데이터 상태의 `SUCCEEDED / CPU_TESTED`는 데이터 처리와 검사가 성공했다는 뜻이며 모델의 의료 성능을 뜻하지 않는다.

## 실제 확보한 텍스트 자료

| 자료 | 고정 revision | 실제 원본 QA 수 | 최종 train / validation / test |
| --- | --- | ---: | ---: |
| [MedQuAD](https://github.com/abachaa/MedQuAD) | `577bd37b96c02d1833b2c9eed2de9f96964e96cb` | 47,441 | 8,000 / 1,494 / 1,301 |
| [PubMedQA PQA-L](https://github.com/pubmedqa/pubmedqa) | `1cbae8e92f72f20c8d3747cbb3bf5bc53554d997` | 1,000 | 450 / 50 / 500 |
| [KorMedMCQA](https://huggingface.co/datasets/sean0042/KorMedMCQA) | `79efd6f91edfc8036330d7a4daa88b9f2deb9a82` | 7,489행, 7,469 고유 문항 | 0 / 1,059 / 3,009 |

MedQuAD README의 47,457은 실제 파일을 읽은 수량과 다르다. 12개 하위 디렉터리 XML에서 읽은 수량은 47,441이며, 제거 대상으로 지정된 세 MedlinePlus subset 31,029행과 그 외 빈 답변 5행을 제외했다. 실제 공개 nonempty 답변 16,407행 중 동일 입력·상이한 정답 2,283행을 격리하고 동일 입력·동일 정답 20행을 제거했다. 문서 및 동일 입력 연결 요소 단위로 seed=20261023의 80/10/10 hash 분할을 적용한 뒤 train cap 때문에 3,309행을 제외했다. 같은 문서의 XML 조각에서 document/qid가 재사용되는 사례는 원본 XML 파일과 row index를 sample identity에 포함하여 구분하고 document group은 유지했다. 제거된 답변을 재수집하거나 생성하지 않았다. LiveQA archive는 raw snapshot에 보존하지만 training XML glob에는 포함하지 않는다.

MedQuAD [원저자 CC-BY-4.0](https://github.com/abachaa/MedQuAD/blob/577bd37b96c02d1833b2c9eed2de9f96964e96cb/LICENSE.txt)의 공개 답변 범위를 사용하며 각 NIH source URL을 보존한다. 자유서술 답변이 다르다는 이유만으로 같은 입력을 독립 문제로 간주하지 않으므로 label conflict 격리는 보수적이다. 서로 다른 표현의 유효한 답변도 격리될 수 있다.

PubMedQA는 원저자 `preprocess/split_dataset.py`의 seed=0, fold=0 알고리즘을 비파괴 구현했다. 실제 PQA-L에서 공식 파일의 함수 부분만 AST로 분리하여 실행한 결과와 train/dev/test PMID 집합이 정확히 일치함을 확인했다. 원본 스크립트의 디렉터리 삭제 부분은 실행하지 않았다. 입력은 QUESTION와 CONTEXTS뿐이며 LONG_ANSWER·final_decision·reasoning classifier prediction을 입력에서 제외한다. CONCLUSIONS label 또는 질문·LONG_ANSWER와 동일한 context segment가 있을 때 제외하는 방어 코드도 테스트했다. 실제 원본에는 해당 필터로 제외된 segment가 없었다.

PubMedQA [저장소 MIT 배포](https://github.com/pubmedqa/pubmedqa/blob/1cbae8e92f72f20c8d3747cbb3bf5bc53554d997/LICENSE)와 원문 초록의 권리를 구분한다. PMID별 원문 출처를 보존하고 연구용 로컬 처리에 한정하며 초록이 public domain이거나 MIT로 재허가되었다고 주장하지 않는다. 원문 내용은 재배포하지 않는다.

KorMedMCQA의 원저자 데이터 카드 `cc-by-nc-2.0`를 현재 API와 README에서 확인했다. 16개 공식 parquet를 내려받아 직접 검사했으며 fewshot 20행이 dev에 포함된 동일 입력임을 확인했다. train 3,401행과 fewshot 20행은 원본 보존·감사에만 사용하고 v1 SFT 및 zero-shot 채점에서 제외한다. `dev`를 canonical `validation`에 매핑하고 원래 split 이름을 유지한다. 답 1~5를 A~E로 변환하고 `cot`는 model input에 넣지 않는다. 이 비상업 연구 허용을 시험 문제의 상업용 키오스크 이용 허가로 해석하지 않는다.

| KorMedMCQA 직군 | 원본 train | dev | test | fewshot |
| --- | ---: | ---: | ---: | ---: |
| doctor | 1,890 | 164 | 435 | 5 |
| nurse | 582 | 291 | 878 | 5 |
| pharm | 632 | 300 | 885 | 5 |
| dentist | 297 | 304 | 811 | 5 |

## 저장·감사·동결 계약

- `raw/{source}/`: 원본 불변 snapshot. source manifest에 revision, 파일 bytes와 SHA-256을 기록한다. build 전에 모든 원본 checksum을 검사한다.
- `processed/{source}/{train,validation,test}.jsonl`: canonical source별 자료. input과 supervision을 분리한다.
- `processed/text/{train,validation,test}.jsonl`: MedQuAD와 PQA-L의 통합본. 현재 8,450 / 1,544 / 1,801행이며 KorMedMCQA와 영상 자료는 포함하지 않는다.
- `manifests/sources/{source}.json`, `manifests/build/{source}.json`: 권리·수집·변환 기록과 제외 목록. 텍스트 원문을 로그에 출력하지 않는다.
- `manifests/audit.json`: schema, sample ID, 옵션, exact input/label conflict, known-group split 침범, 영상 파일 존재 및 hash 감사.
- `manifests/freeze-{digest}.json`: ordered sample IDs, source revisions, input/original/group/image digest, code SHA와 seed를 포함한 불변 manifest.
- `frozen/{file_sha256}/{split}.jsonl`: freeze가 참조한 bytes의 불변 사본. `current_freeze.json`은 편의용 포인터다. 이전 freeze 파일과 해당 bytes를 유지한다.

정확한 입력과 알려진 문서·PMID·원본 문항·이미지 그룹의 누수는 critical error이며 하나라도 있으면 freeze를 거부한다. 서로 다른 source의 이미지 byte/pixel hash도 공통 group으로 검사한다. VQA 문구가 같아도 이미지가 다르면 동일 입력으로 병합하지 않는다.

텍스트 near-duplicate 검사는 질문 word-trigram Jaccard>=0.95 후보를 생성한다. Unicode NFC·공백만 비교용으로 정규화하고 원문을 보존한다. 숫자·단위·부정·좌우를 지우지 않는다. 후보는 자동으로 같은 의미라고 확정하지 않는다. 인덱스의 hash signature와 단어 수 bucket, bucket당 100개 비교 cap을 사용하므로 exhaustive semantic deduplication이 아니다. 생성된 후보쌍 전체에 대한 실제 검사 수와 cap 도달 여부는 audit에 남긴다. 최초 네 source 감사에서 927,288쌍을 검사하여 동일 split 내 후보 2쌍을 발견했으며 cross-split 후보는 없었다. 추가 source 이후 최종 수치는 `reports/public_summary/data_freeze.json` 및 로컬 audit를 기준으로 한다.

환자 식별자가 제공되지 않아 `patient_group_verified=false`다. exact 누수 0은 환자 독립성·완전한 사전학습 비노출·모든 재서술 탐지를 보장하지 않는다. 실제 tokenizer의 truncation과 assistant target 보존은 training renderer가 별도로 검사한다. source 원문 char length를 token 수라고 표기하지 않는다.

## 재개 명령

저장소 최상위에서 기존 프로젝트 가상환경을 활성화하고 실행한다. 아래 루트에는 이미 확보된 원본이 있으므로 fetch는 기존 snapshot을 재사용하고 build는 checksum을 확인한다.

```bash
export MEDVLM_DATA_ROOT="$(realpath ../medvlm-data)"
python -m mediflow_medvlm.data discover
python -m mediflow_medvlm.data fetch --source all --data-root "$MEDVLM_DATA_ROOT"
python -m mediflow_medvlm.data build --source all --data-root "$MEDVLM_DATA_ROOT"
python -m mediflow_medvlm.data audit --data-root "$MEDVLM_DATA_ROOT"
python -m mediflow_medvlm.data freeze --data-root "$MEDVLM_DATA_ROOT"
```

이 CLI의 fetch/build `all`은 MedQuAD, PQA-L, KorMedMCQA의 세 우선 텍스트 source다. SLAKE와 추가 source의 확보·변환 명령은 해당 source 문서를 따른다. audit/freeze는 `processed` 아래 모든 source별 JSONL을 합쳐 검사한다. source별 실패는 다른 source 처리를 중단하지 않는다. HTTP timeout과 제한된 재시도, `.part` 기반 HTTP 재개, 파일 크기 cap, 원본 SHA, atomic rename을 사용한다. Git clone이 중단되면 `repository.part`를 보존하고 자동 삭제하지 않는다.

```bash
python -m mediflow_medvlm.data.forge_export --data-root "$MEDVLM_DATA_ROOT" --freeze-manifest "$MEDVLM_DATA_ROOT/manifests/freeze-1bf58dbc8baca7a3f5865accf5a86e9fba0b84f8abe3c986ed34b67d6c3d87c5.json" --rows 4
```
