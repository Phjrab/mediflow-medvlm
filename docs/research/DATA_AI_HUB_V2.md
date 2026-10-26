# AI-Hub 한국어 후속 데이터 준비

기준일 2026-10-26. v1 원본·processed·freeze·예측을 유지하고 `${MEDVLM_DATA_ROOT}/v2`에 별도로 준비했다. 실행 계약은 `v2/manifests/text-corpora-v2.json`, 실제 파일 검증은 `text-corpora-verification-v2.json`이다. 공개 집계는 [v2 데이터 보고서](../../reports/public_summary/v2/aihub_v2_data.json)에 있다. 이 문서의 완료 범위는 데이터이며 학습 성공을 의미하지 않는다.

## 실제 확보와 reader

71875 사용자 제공 Drive ZIP 136,478,457 bytes의 SHA-256은 `530abb197345d03e92ebabecf7e4fd2e0a6a186c1b2043749478003e2bf8bc25`다. ZIP 17개와 파일 멤버 69,422개를 CRC·SHA로 검사했다. 원천 문서 52,126개는 QA를 생성하는 데 쓰지 않았다. 공식 Training/Validation 라벨 17,280개를 확인했다. `qa_id/domain/q_type`은 정수, 질문·답변은 문자열이다. 파일 BOM 17,280개와 key BOM 0개를 확인했으며 key 정규화 충돌은 별도로 거절한다.

q_type은 공식 정의대로 객관식 1, 단답형 2, 서술형 3이다. 객관식은 줄 시작의 연속 번호 선택지와 정답 번호/내용을 분리한다. 번호와 내용 불일치 48, 선택지 블록 불명확 13, 빈/중복 선택지 3, 범위 밖 번호 1행을 격리했다. 이후 동일 입력 중복 1행과 공식 held-out에 같은 질문 그룹이 있는 train 4행을 제외했다. 기관이나 qa_id를 환자·문서 ID로 간주하지 않았다.

| 구분 | 원래 train | 원래 validation | 정제 후 train | 새 dev | 새 test |
|---|---:|---:|---:|---:|---:|
| QA 수 | 15,360 | 1,920 | 15,297 | 956 | 957 |

제공된 공식 validation을 질문 내용 그룹과 seed 20261023의 고정 hash로 dev/test로 나눴다. 새 test의 q_type별 수는 객관식 777, 단답 83, 서술 97이다. 공식 test를 받은 것으로 표시하지 않으며, 기반 모델의 사전학습 비노출이나 환자 독립성을 보장하지 않는다.

71762의 입력 링크와 4,248,340,940-byte 파일 metadata는 수신했다. 연결 도구의 256MiB 제한(413)과 정상 브라우저 다운로드의 `ERR_BLOCKED_BY_CLIENT` 때문에 실제 확보량은 0 bytes이며 수집 상태는 `BLOCKED_DELIVERY`다. 링크 미제공이나 AI-Hub 승인 거절로 분류하지 않는다. 실제 schema·pairing은 아직 미검증(`UNRESOLVED_PAIRING`)이다. 명시 ID·외래키·가이드 근거가 있는 1:1/1:N/N:1 join 함수만 합성 fixture로 준비했다. 파일 순서·질환·의도 기반 연결은 금지하며, 실제 71762 변환·학습 성공으로 계산하지 않는다.

## 권리와 오염 후보 감사

71875는 사용자 정상 취득·로컬 연구 학습 승인과 [AI-Hub 이용정책](https://www.aihub.or.kr/intrcn/guid/usagepolicy.do?currMenu=151&topMenu=105)을 근거로 사용했다. CC/MIT가 아닌 AI-Hub custom terms이며 NIA 출처표시와 원문 제3자 제공 제한을 유지한다. [공식 71875 설명](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71875)의 v1.1 및 LLM 생성 표시를 기록하며 모든 답변을 전문가 직접 작성이라고 표현하지 않는다. 공식 페이지의 제공 모델 설명에 KorMedMCQA 평가 사용이 있으므로 동봉 평가·모델 경로를 train에서 제외하고 전체 보호셋을 대조했다.

KorMedMCQA 공식 dev 1,059 + test 3,009 = **4,068개 전부**의 frozen 바이트를 검증했다. 숫자·단위·부정·구두점을 보존한 질문·선택지 exact, 선택지 순서 변경, 같은 질문 stem 검사와 문자 5-gram Jaccard ≥0.90 후보는 0개였다. AI-Hub train과 새 held-out 간 exact/문자 5-gram ≥0.95 후보도 0개였다.

재서술·번역 후보를 위해 MIT [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small/tree/614241f622f53c4eeff9890bdc4f31cfecc418b3)을 493,292,184 bytes로 확보했다. CPU float32, 양쪽 `query:` 접두, masked mean pooling·L2, cosine ≥0.92, 상위 5개 규칙을 **검색 전에 고정**했다. 정답 없이 질문과 원래 순서의 선택지만 사용했다. AI-Hub 17,210 + EN 8,450와 KorMedMCQA 4,068의 총 29,728개 입력은 모두 512토큰 안에 들어와 잘림이 없었다. 원문을 외부 서비스에 전송하거나 GPU를 사용하지 않았다.

AI-Hub **3,396행/8,278쌍**이 유사도 후보였다. 이들은 확정 중복이 아니다. train 3,010과 dev 198행은 보수적으로 격리했다. test 188행은 원래 957개 안에 보존하고 `aihub-heldout-v2.json`에 별도 flag를 남겼다. EN 후보는 0개였다. 낮은 유사도는 번역·의학적 동등성 부재를 입증하지 않으며, 높은 유사도도 동등성을 확정하지 않는다.

## 고정된 학습 코퍼스

처음 제안한 q_type별 1,334/1,333/1,333개는 후보 격리 후 서술형 후보가 1,240개만 남아 유지할 수 없었다. 성능 평가 전에 객관식 **1,500**, 단답 **1,300**, 서술 **1,200**으로 조정했다. cutoff는 바꾸지 않았고 결정 이력은 `text-corpus-selection-v2.json`에 보존했다. 가능한 domain 균형과 질문/문서 그룹 전체를 유지하며 seed 20261023으로 결정적으로 선택했다.

| 코퍼스 | 실제 train 구성 | train 수 | 공통 validation | 1-pass update/batch16 | supervised tokens |
|---|---|---:|---:|---:|---:|
| KO | AI-Hub 71875 | 4,000 | 1,516 | 250 | 243,394 |
| MIX | 같은 KO 4,000 + MedQuAD 3,550 + PubMedQA 450 | 8,000 | 1,516 | 500 | 1,137,306 |

두 Qwen의 tokenizer 및 native template SHA가 같음을 확인했다. 실제 trainer와 같은 template 인코딩·assistant 경계를 CPU로 검사했고 2,048토큰을 넘는 EN train 50개와 EN dev 12개를 선택 대상에서 제외했다. AI-Hub 후보는 모두 적합했다. token 수에는 학습 대상 종료 토큰이 포함되고 prompt는 포함되지 않는다.

공통 validation은 AI-Hub 758 + EN 758(PubMedQA 50, MedQuAD 708)로 두 조건에서 **파일과 SHA가 같다**. KO의 영어 validation 사용은 공통 유지 성능 기반 checkpoint 선정이며 영어 train 추가가 아니다. MIX는 KO보다 표본과 supervised token이 많으므로 효과를 언어 구성만의 인과 효과로 해석하지 않는다.

KO freeze는 `03714513944265854b7663d1725fafd2a6193b1a55a4d0a5acd06590c3e36aec`, MIX는 `2b1e5b382209d8ccd1795a89c1afa6f403b7f518f582de1995fde634a3014ace`다. source 변환 스냅샷은 최종 pool `9b5469c6b67bded6f9d4280eef95443d918780612d587aad6c432104b17d5767/source_snapshot/aihub.py`에 보존했다. 이후 ZIP 폴더 재개 보강은 기존 frozen bytes를 변경하지 않았다.

## 실행·재개

완료된 freeze를 다시 만들 필요는 없다. 최종 corpus manifest를 학습 실행기가 읽고 성공 artifact를 검증해 재사용한다. 원본 import를 재검증하려면 저장소에서 다음을 실행한다.

```bash
export MEDVLM_DATA_ROOT="$(realpath ../medvlm-data)"
export PYTHONPATH="$PWD/src"
../.venv-medvlm/bin/python -m mediflow_medvlm.data aihub-import \
  --data-root "$MEDVLM_DATA_ROOT/v2" --dataset-id 71875 --version 1.1 \
  --input "$MEDVLM_DATA_ROOT/v2/raw/drive/aihub_71875_delivery.zip" \
  --delivery-receipt "$MEDVLM_DATA_ROOT/v2/raw/drive/aihub_71875_delivery.receipt.json"
../.venv-core310/bin/python -m pytest tests/test_data_aihub.py tests/test_supplemental.py
```

완성 ZIP, 원본 폴더, 순번이 연속된 `.zip.part001`/`.zip.001`/`.zip.partaa` 입력을 지원한다. 조각은 삭제하지 않고 완성 ZIP을 재병합하지 않는다. 표준 multi-disk `.z01` 포맷은 임의로 연결하지 않는다. 원본 압축의 루트 `/` 표시는 스트리밍 멤버 이름으로만 처리하며 실제 파일경로로 추출하지 않는다. 경로 traversal·심볼릭 링크·중복 멤버·CRC 오류는 거절한다. 25GiB 원본 cap과 20GiB/15% 디스크 여유 기준은 유지한다.

`aihub-build`와 `aihub-corpora`의 필수 인자는 `python -m mediflow_medvlm.data --help`에 있다. 같은 corpus를 재개할 때 기존 source/token/semantic/selection receipts와 frozen SHA를 검증하며, 변경된 입력을 기존 성공으로 덮어쓰지 않는다. 71762에는 실제 로컬 원본 또는 사용 가능한 전달 경로와 명시 pairing 근거가 추가로 필요하다.
