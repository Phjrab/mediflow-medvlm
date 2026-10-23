# Mediflow MedVLM

공개 의료 QA·VQA 데이터의 정제, 고정 분할, 로컬 LoRA/QLoRA 학습과 학습 전후 평가를
수행하는 연구용 파이프라인이다. 기존 VQA·보고서 초안·시계열 영상 비교 계약에
이미지 없는 TEXT_QA를 추가했다.

## 현재 범위

- 비식별화된 study ID와 검증된 영상 참조를 사용하는 공통 데이터 계약
- TEXT_QA·VQA·보고서·시계열 비교 작업 타입
- 원본/정답/모델 입력 분리, source manifest, 중복·이미지·split 감사
- native Transformers/PEFT 학습, assistant-only loss, adapter 저장·재로드·strict resume
- frozen 평가, 실패를 포함한 분모, baseline/tuned paired 비교
- 검증된 SSH 호스트에 대한 별도 Jetson 조사·제한된 추론 명령

구현·실제 확보·실제 학습·실제 평가 상태는 각각 기록한다. 현재 실행 결과는
`reports/public_summary`와 `docs/agent/HANDOFF.md`를 확인한다. DICOM 수집·임상 배포는 포함하지 않는다.
의료전문가의 검토, 기관의 검증, 관련 규제 검토 없이 진단·치료 결정에 사용해서는 안 된다.

## 개발 환경

Python 3.10 이상이 필요하다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
```

외부 개발 의존성을 설치하지 않은 상태에서는 다음 표준 라이브러리 검사도 사용할 수 있다.

```bash
PYTHONPATH=src:tests python3 -m unittest test_domain test_validation test_workflows -v
```

## 데이터와 실행

core는 Python3.10 이상이며, 검증한 GPU 학습 profile은 Python3.12다. 학습 의존성은
`configs/requirements-training.lock.txt`에 실제 설치 버전을 기록했다. Jetson에는 x86용
환경을 복사하지 않는다. 대용량 원본·가중치·예측은 Git 밖에 둔다.

```bash
export MEDVLM_DATA_ROOT="$(realpath ../medvlm-data)"
python -m pip install -e '.[dev,data]'
medvlm doctor --output "$MEDVLM_DATA_ROOT/host_inventory.json"
medvlm data discover
medvlm data fetch --data-root "$MEDVLM_DATA_ROOT"
medvlm data build --data-root "$MEDVLM_DATA_ROOT"
medvlm data audit --data-root "$MEDVLM_DATA_ROOT"
medvlm data freeze --data-root "$MEDVLM_DATA_ROOT"
```

SLAKE/VQA-RAD 명령과 source별 권리·차단 조건은 [데이터 준비](docs/research/DATA_PREPARATION.md),
[영상 자료](docs/research/VISION_SOURCES.md), [실제 데이터 결과](docs/research/DATA_REPORT_KO.md)에 있다.

```bash
medvlm models probe --model-id Qwen/Qwen3-1.7B --output "$MEDVLM_DATA_ROOT/model-probe.json" --tokenizer-check
medvlm train --config "$MEDVLM_DATA_ROOT/configs/qwen3_17b-v1-20261023.json"
medvlm reload --config "$MEDVLM_DATA_ROOT/configs/qwen3_17b-v1-20261023.json" --checkpoint <SELECTED_CHECKPOINT>
medvlm evaluate --config <RUN_CONFIG> --protocol <FROZEN_EVAL_PROTOCOL> --output-dir <NEW_BASE_RUN>
medvlm compare --base <BASE_RUN> --tuned <TUNED_RUN> --output <PAIRED_SUMMARY_JSON>
medvlm device inventory --output "$MEDVLM_DATA_ROOT/runs/device/new-inventory"
medvlm status
```

`<...>` 항목은 해당 run artifact에서 채워야 하는 placeholder다. 실제 로컬 설정을 준비한 뒤
`PYTHONPATH=src python scripts/complete_v1.py --data-root "$MEDVLM_DATA_ROOT"`로
성공한 단계는 재사용하면서 고정된 4개 모델 셀을 순서대로 처리한다. 실행 중인 자기
worker를 확인하며 동일 GPU의 두 번째 학습을 만들지 않는다. 인증/권리/하드웨어 차단은
그 셀에 기록하고 다른 셀을 계속한다. 일반적인 명시적 argv 작업 목록은
`medvlm run --plan <PLAN_JSON>` 및 같은 인자의 `medvlm resume`로 관리한다.

평가에는 원문을 포함하는 제한된 로컬 예측을 보존하고 Git에는 집계만 남긴다.
자동 lexical score나 loss 감소는 임상 정확도가 아니다. 데이터별 사전학습 노출 가능성과
환자 독립성 미확인, 전문가 미검토 상태를 결과에 함께 기록한다.

## 개인정보와 데이터

환자 이름, MRN, 생년월일, 원본 DICOM UID 등 식별자를 요청·프롬프트·로그에 넣지 않는다.
`study_id`에는 기관 내부에서만 역매핑 가능한 가명 식별자를 사용한다. 자세한 구현 및
운영 기준은 다음 문서를 확인한다.

- [아키텍처](ARCHITECTURE.md): 모듈 경계와 모델 어댑터 계약
- [데이터 거버넌스](DATA_GOVERNANCE.md): PHI 처리와 로그 정책
- [평가 프로토콜](EVALUATION.md): 모델·프롬프트 변경 전 검증 기준
