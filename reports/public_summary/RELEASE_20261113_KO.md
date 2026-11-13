# 전체 보존 결과·LoRA 공개 배포 — 2026-11-13

사용자가 공개 GitHub 저장소임을 확인한 질문에 **“LoRA 가중치도 포함”**으로 응답하여, 기존 가중치 공개 제외 지시를 이번 LoRA에 한해 갱신했다. 기존 실행 파일은 읽기만 하고 공개용 사본을 만들었다. 새 학습·추론·점수 보정은 수행하지 않았다.

결과는 [Release medvlm-results-20261113](https://github.com/Phjrab/mediflow-medvlm/releases/tag/medvlm-results-20261113)에 첨부한다. 코드와 명세는 작업 브랜치에 보관하며 main에 병합하지 않는다. 각 압축 파일을 다시 읽어 보관 원본과 SHA256을 대조했고, 업로드 파일의 GitHub SHA256 digest와 크기도 검증한다. 실제 원격 검증 영수증은 이 배포의 `upload_verification.json`에 기록한다.

| 첨부 파일 | 실제 포함 범위 |
| --- | --- |
| `medvlm-results-v1-v5.tar.gz` | 63개 예측 파일의 62,998개 생성 결과와 문항별 점수, 오류·중단 상태, 20개 학습 실행의 3,566개 이벤트, 설정·선택·재로드·실행·검증 기록, 기존 공개 보고서 54파일 스냅샷 |
| `lora-qwen3-1.7b.tar.gz` | 보존 체크포인트 8개: v1 2개·v2 4개·v4 2개 |
| `lora-qwen3-4b.tar.gz` | 보존 체크포인트 9개: v1 2개·v2 5개·v4 2개 |
| `lora-qwen3.5-2b.tar.gz` | v1 보존 체크포인트 3개 |
| `lora-medgemma-1.5-4b-it.tar.gz` | v1 보존 체크포인트 2개와 HAI-DEF 약관·금지사용 정책·고지·변경 설명 |
| `manifest.json`, `SHA256SUMS`, `README.md` | 원본/공개 사본 해시, 파일별 수량·크기, 제외 항목 및 복구 안내 |

LoRA 원본 22파일은 합계 **1,059,378,696바이트**다. smoke, 본 학습, 검증 손실에 따라 선택된 체크포인트, 남아 있는 최종 체크포인트를 모두 포함한다. `CHECKPOINTS.json`과 각 `RELEASE_METADATA.json`의 `smoke`, `selected_in_run`, `selected_for_main_evaluation`, `optimizer_steps`를 구분해서 사용한다. 삭제된 과거 중간 체크포인트를 복원한 것은 아니다. 선택 여부는 해당 실행의 동결된 선택 포인터이며 성능 추천이 아니다.

62,998건에는 test·validation·이미지 제거·출력 형식 진단이 함께 들어 있다. 동일 문항을 여러 모델·조건으로 실행하므로 고유 질문 수나 독립 test 수가 아니다. 기존 parser/점수를 보존했고 v5는 원래 사용한 `score_rows`를 각 검증 조건에 그대로 적용했다. `case_key`/`group_key`는 원본 식별자의 SHA256으로 실행 간 대응을 유지한다. 이는 비가역 익명화 보증이 아니다. 자유서술 참조 정답 대신 기존 문자열 점수만 공개하며, 분류 정답은 A–E/yes/no/maybe 레이블만 담는다. `raw_output`과 `parsed_answer`는 모델이 생성한 연구 출력이며 정답이나 의료 조언이 아니다.

원본 질문·문맥·선택지 본문·이미지·자유서술 정답, 원본 데이터 ZIP, 기본 모델·tokenizer·processor, optimizer/RNG 상태 22개, 전문가 검토용 원본 표와 비공개 A/B 키, 비밀값과 임의 stdout/stderr는 포함하지 않는다. 설정의 로컬 경로는 `${MEDVLM_DATA_ROOT}` 등으로 변환한다. 이 Release는 승인된 연구 산출물 보관이며 전체 개인 데이터 디렉터리 백업은 아니다.

Qwen LoRA에는 각 기본 모델의 고정 revision에서 받은 Apache 2.0 라이선스 사본과 변경 고지를 동봉한다. MedGemma LoRA에는 [HAI-DEF 이용약관](https://developers.google.com/health-ai-developer-foundations/terms)의 재배포 조건에 따라 약관 전체와 §3.2 사용 제한, [금지사용 정책](https://developers.google.com/health-ai-developer-foundations/prohibited-use-policy), Notice 및 변경 고지를 동봉한다. 저장소 코드의 Apache 선언이 MedGemma 가중치의 조건을 대체하지 않는다. 기본 모델은 공식 제공처의 정상 접근 절차와 이용 조건에 따라 별도로 받는다.

## 다운로드·재로드

```bash
gh release download medvlm-results-20261113 --repo Phjrab/mediflow-medvlm --dir medvlm-artifacts
cd medvlm-artifacts
sha256sum -c SHA256SUMS
mkdir results adapters
tar -xzf medvlm-results-v1-v5.tar.gz -C results
tar -xzf lora-qwen3-1.7b.tar.gz -C adapters
```

`adapters/CHECKPOINTS.json`에서 목적에 맞는 체크포인트를 선택한다. 각 체크포인트의 `RELEASE_METADATA.json`에 기본 모델 ID와 정확한 revision이 있고, 결과 묶음의 `metadata/`에 실제 라이브러리 버전·quantization·dtype와 모델/template 해시가 있다. 실행 당시 기록된 버전을 재현해야 한다. 저장소 `pyproject.toml`은 버전 범위를 선언하며 완전한 환경 lock은 아니다. 아래는 GPU 환경에서 저장소의 native loader로 재로드하는 예시다. 이전 재로드 검증의 공개 사본은 각 학습 실행의 `metadata/.../reload.json`에서 확인한다.

```python
import json
from pathlib import Path
from mediflow_medvlm.training.models import load_model

adapter = Path("adapters") / "runs/qwen3_17b-v1-20261023/checkpoints/step-000525-c28d685d"
info = json.loads((adapter / "RELEASE_METADATA.json").read_text())
model, processor, receipt = load_model({
    "model_id": info["model_id"], "revision": info["base_revision"],
    "device": "cuda", "dtype": "bfloat16", "quantization": "nf4",
    "local_files_only": False,
}, training=False, adapter_path=adapter)
```

기본 모델을 미리 받은 환경에서는 `local_files_only=True`와 새 머신의 `cache_dir`을 지정한다. MedGemma에는 해당 기본 모델의 접근 권한이 필요하다. 이 안내는 저장된 LoRA 재로드용이며 제외한 optimizer/RNG 상태와 원본 데이터 없이 이전 학습을 정확히 이어갈 수는 없다.

## 실제 수행·미실행·차단 및 재개

| 구분 | 상태·다음 작업 |
| --- | --- |
| 실제 수행 | v1 텍스트·VLM 학습/재로드, v2 한국어 KO/MIX 학습 및 확대 평가, v3 데이터 구성 감사, v4 71762 제공처 설명에 따른 별도 무작위 매핑 실험, v5 검증 전용 형식 진단. 세부 결과는 [전체 색인](INDEX_KO.md) 참조 |
| 이번 실제 수행 | 공개용 변환·전량 보존 LoRA 패키징·해시 대조·작업 브랜치 업로드·Release 첨부 공개 |
| 미실행 | 의료 전문가 검토 0건, 임상 검증 미수행, Jetson 현장 평가는 사용자 보류. 이번에 추가 학습이나 새 test 추론은 없음 |
| 외부 차단/제한 | 71762 문항별 고정 QA 연결과 증강 parent 식별자 미제공. 제공처 공식 `make_train_data.py` 미확보이므로 v4는 설명 기반 독립 구현. 실제 임상 성능이나 원인 효과를 입증하지 않음 |
| 재개 | 먼저 위 명령으로 공개 결과·LoRA 복구. 로컬 원본이 있는 기존 환경의 v4/v5 재검증 명령은 아래와 같음. 완료된 실행을 검증/보고하며, 새 실험은 새 계획으로 분리해야 함 |

```bash
export MEDVLM_DATA_ROOT=/path/to/private/medvlm-data
PYTHONPATH=src python scripts/run_71762_v4.py --data-root "$MEDVLM_DATA_ROOT" --phase all
PYTHONPATH=src python scripts/run_format_diagnostic_v5.py --data-root "$MEDVLM_DATA_ROOT"

# 원본이 있는 환경에서 공개용 묶음을 다시 만들 때: 출력은 비어 있는 새 폴더
PYTHONPATH=src python -m mediflow_medvlm.publication \
  --data-root "$MEDVLM_DATA_ROOT" --repo "$PWD" \
  --licenses /path/to/official-license-copies --output /path/to/new-release-output
```

공개 묶음만으로 데이터 라이선스·접근 권한과 전문가 검토를 대신할 수 없다. 기존 보고서의 결과·한계·동결 해시는 바꾸지 않는다.
