# 영상·보충 데이터 확보와 제한

확인일: 2026-10-23 (Asia/Seoul). 실제 source별 수량·hash·revision은 `reports/public_summary/vision_sources.json`과 Git 밖 data root의 `manifests/sources/*.json`을 기준으로 한다. 데이터 원문·이미지·제외 ID 목록은 Git 밖에 보존한다.

## SLAKE: 실제 확보·정제 완료

[저자 프로젝트](https://www.med-vqa.com/slake/)가 직접 연결한 [BoKelvin/SLAKE](https://huggingface.co/datasets/BoKelvin/SLAKE)를 사용했다. revision은 `a9083ce6c34ac3ffb17671a605962924d8a8f9e9`, [해당 데이터 카드](https://huggingface.co/datasets/BoKelvin/SLAKE/blob/a9083ce6c34ac3ffb17671a605962924d8a8f9e9/README.md)의 데이터 라이선스는 CC-BY-4.0이다. 개별 원본 영상의 권리가 별도로 열거되어 있지는 않다. 저자 배포 범위의 오프라인 연구로 사용하며 원문은 Git에 재배포하지 않는다.

원본 QA는 train 9,835 / validation 2,099 / test 2,094, 총 14,028개다. 그중 영어는 4,919 / 1,053 / 1,061개다. 212,343,373-byte `imgs.zip`의 공식 LFS SHA-256 `44eb7d9214e1ac5b7946e237b669401866df9114fd9a3618c5a84fbffcded0b0`을 검증했고, 참조하는 642개 이미지를 모두 실제로 decode했다. 서로 다른 이미지 파일 사이의 exact decoded-pixel 중복은 0쌍이었다. 이는 같은 image ID가 여러 split에서 재사용되지 않는다는 뜻은 아니다.

공식 split에서 같은 image ID가 298개 겹친다. test를 우선 보존하고 그다음 validation을 보존하여 동일 원본 이미지 때문에 train/dev 1,627 QA를 제외했다. 추가로 보수적인 perceptual 후보 component 때문에 1,853 QA를 제외했다. 빈 답변 1개와 중국어 6,995개를 제외한 연구용 canonical은 **train 1,979 / validation 512 / test 1,061**이다. test의 영어 1,061 QA는 모두 보존했다.

perceptual 후보 생성은 EXIF 방향 보정→RGB→Pillow grayscale L→9×8 LANCZOS resize→인접 가로 픽셀 left>right의 64-bit dHash이며 Hamming distance ≤4이다. 937쌍의 후보를 connected component로 묶었다. 전체 364 components이고 최대는 MRI 영상 86개다. 다음 큰 component는 CT 34, 23, 22, 16개다. 의료 영상의 유사한 절편·모양만으로도 연결될 수 있으므로 **937쌍 모두 실제 중복이라고 판단한 결과가 아니다**. 민감도 중심의 후보 묶음이 많은 학습 행을 제거한 영향이 있으며, 영상 전문가의 동일 환자·절편 검증은 수행하지 않았다.

따라서 이 연구의 이름은 `official_split_preserved_with_lower_priority_overlap_and_conflict_exclusion`이다. 원래 benchmark의 학습 프로토콜과 동일하다고 주장하지 않는다. patient/study 식별자가 없어 환자 독립성은 미확인이다. 상세 component·modality·제외 사유는 `manifests/slake_image_audit.json`에 기록했다. 모델 입력은 질문과 실제 이미지 참조만 읽고 정답·knowledge triple은 metadata/supervision에 분리한다.

재개/재검증:

```bash
PYTHONPATH=src python -m mediflow_medvlm.data.slake --data-root ../medvlm-data --fetch
PYTHONPATH=src python -m pytest tests/test_slake.py tests/test_supplemental.py tests/test_vqarad.py -q
```

`--fetch`는 기존 원본이 hash/size와 일치하면 다운로드를 생략한다. source allowlist, timeout, 최대 3회 재시도, HTTP Range 재개, atomic rename, zip traversal/symlink/용량 검사를 구현했다. SLAKE 다운로드는 공통 redirect/range/disk guard를 사용하고 압축 해제와 VQA-RAD 다운로드도 raw+images 합계 25GiB 및 남은 디스크 20GiB/15% 중 큰 값의 예산을 검사한다. archive의 기존 파일은 새 공간으로 이중 예약하지 않으며 기존 destination/partial symlink의 바깥 경로 탈출을 거절한다. 오류가 없는 원본은 변경하지 않는다. 13개 테스트로 이미지·split·충돌 정답과 보충 source 계약, 다운로드 전 예산 거절, 기존 symlink 탈출 방지를 확인했다. 이 재개 코드 보강은 이미 동결한 원본/processed bytes를 변경하지 않았다.

## VQA-RAD: 공식 OSF 외부 평가 후보

웹 열람 도구에서 [OSF 페이지](https://osf.io/89kps/)가 403이었지만, 이번 실행 호스트의 **공식 OSF public API와 원본 download 링크는 실제로 접근 가능**했다. 이를 영구 접근 불가로 기록하지 않는다. [OSF node metadata](https://api.osf.io/v2/nodes/89kps/)의 license 관계가 [CC0 1.0 Universal](https://api.osf.io/v2/licenses/563c1cf88c5e4a3877f9e96c/)을 가리키는 것을 확인했다. MedPix case URL, OSF file ID/version/SHA-256을 보존한다.

QA JSON은 2,248행이며 `phrase_type`이 freeform 1,206 / para 591 / test_freeform 308 / test_para 143이다. 공식 이미지 315개를 모두 확보하여 공개 SHA-256과 실제 decode를 검증했다. 원본 QA는 그중 314개 이미지를 참조하고, 외부 평가 canonical 451 QA는 203개 이미지를 참조한다. 이 source는 SFT에 넣지 않는다. 원본 설명의 평가 태그 `evaluation`, qid/linked ID/문항 유형/영상 기관/MedPix case도 보존한다. SLAKE 642개 이미지와 VQA-RAD 315개 이미지 사이의 exact decoded-pixel 중복은 0개다. source 간 perceptual 동일 환자·절편 검토는 수행하지 않았다.

```bash
PYTHONPATH=src python -m mediflow_medvlm.data.vqarad --data-root ../medvlm-data --fetch
```

공식 API 파일 목록에서 400개/128 MiB 상한을 먼저 점검하고 최대 2개 다운로드만 동시 수행한다. 최초 실행은 4개로 시작한 설정 실수가 있었고, 이미 시작한 요청은 완료시키되 재개 코드의 상한을 2로 수정했다. 원본 파일마다 공개 SHA-256과 version을 검증한다. 외부 source 이미지와 SLAKE 이미지의 exact pixel 동일성도 최종 전체 freeze 감사에서 확인한다. 모델의 upstream exposure는 별도 기록한다.

## 이번 v1에서 실제 수집하지 않은 자료

| 원천 | 상태 | 확인 결과와 재개 조건 |
|---|---|---|
| MedQA | BLOCKED_LICENSE | [공식 repo](https://github.com/jind11/MedQA) HEAD `27b02f66aac217933c9648a06f82e9f720377925`, MIT는 repo에 있으나 저자가 연결한 외부 `data_clean.zip`에는 시험 QA와 교과서가 함께 있다. [공식 Drive 링크](https://drive.google.com/file/d/1ImYUSLk9JbgHXOemfvyiDiirluZHPeQw/view)는 접근 가능하지만 그 외부 QA 파일에 적용할 데이터 권리 범위를 이번 조사로 해결하지 못했다. 원문/교과서를 자동 수집하지 않았다. 데이터 조건이 확인되면 US/4_options만 reader로 변환한다. |
| MedMCQA | BLOCKED_LICENSE | [공식 repo](https://github.com/medmcqa/medmcqa) HEAD `c59ef14ca1990266c4107c7864b45a20fd93e5e0`의 MIT와 [배포 카드](https://huggingface.co/datasets/openlifescienceai/medmcqa/blob/91c6572c454088bf71b679ad90aa8dffcd0d5868/README.md)의 Apache-2.0 적용 범위를 구분했다. 카드 licensing 본문은 미기입 상태이며 official repo의 배포 링크는 별도 Drive이다. 원자료의 시험/책 권리 조건은 미해결이다. 공식 test gold는 공개하지 않는다고 명시되어 있어 dummy label 정확도를 만들지 않는다. |
| AI-Hub 71875 | BLOCKED_ACCESS | [공식 페이지](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71875)는 version 1.1, 생성 방식 LLM을 표시한다. 이번 환경에 승인된 다운로드 자격/API key/원본 파일이 설정되지 않았다. 전체를 전문가 직접 작성으로 간주하지 않는다. 승인된 파일·조건이 제공되면 qa_id/question/answer/q_type/domain reader를 실행한다. |
| AI-Hub 71762 | BLOCKED_ACCESS | [공식 페이지](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71762)는 version 1.1이다. 원본·활용 가이드의 질문↔답변 ID 계약을 확인하지 못해 pairing은 `UNRESOLVED_PAIRING`이다. 파일 순서·질환·의도로 join하지 않는다. 승인 원본·가이드·명시 ID가 있어야 재개한다. |
| ROCOv2 | PREPARED_NOT_RUN | [pack의 Zenodo 8333645](https://zenodo.org/records/8333645)는 현재 최신본 사용을 요청한다. 공식 latest 관계의 [10821435](https://zenodo.org/records/10821435), version 2.0.1, CC-BY-NC-4.0을 현재 확인했다. 총 파일 6,360,329,403 bytes, `license_information.csv` 9,446,130 bytes. caption task이며 v1는 metadata/크기 조사만 완료한다. |
| PMC-VQA | PREPARED_NOT_RUN | [저자 연결 배포](https://huggingface.co/datasets/RadGenome/PMC-VQA) revision `b56ae594f794867893143b337b4118a835794647`. 최종 데이터 CC BY-SA와 원문 CC0/CC BY를 구분한다. `images.zip` 18,945,102,275 bytes, `images_2.zip` 2,206,255,503 bytes이므로 v1 metadata/크기만 조사했다. 실제 이름은 `train_2.csv`, `test_2.csv`이며 카드의 설명형 이름과 차이가 있다. PMCID별 권리 파일도 후속 수집에 필요하다. |

보충 reader는 `data/supplemental.py`에 구현했다. MedQA는 variant를 US/4_options로 고정하고 동일 영어 question의 variant를 같은 group으로 연결하되 번역 연결은 미확인으로 남긴다. MedMCQA는 실제 파일별 `cop_base=0` 또는 `1`을 명시해야 하며 single-choice만 처리하고 없는 해설을 생성하지 않는다. 이 reader들은 fixture 수준의 CPU 검사를 거쳤고 **실제 MedQA/MedMCQA/AI-Hub canonical 변환은 수행하지 않았다**.

권리/접근 조건이 해결되지 않은 자료에 성공 status나 확보 건수를 넣지 않는다. 신규 약관 동의·계정 생성·유료 서비스·우회 mirror·외부 업로드는 하지 않았다. 확보된 SLAKE와 별도 공개 텍스트 데이터의 학습·평가는 이 보류 항목에 의존하지 않는다.
