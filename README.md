# Mediflow MedVLM

연구·판독 보조 목적의 의료 멀티모달/비전 언어 모델(Medical MLLM/VLM) 워크플로 골격이다.
VQA, 보고서 초안, 시계열 영상 비교를 **구조화된 입력·출력과 안전 검증** 아래에서
구현하는 것을 목표로 한다.

## 현재 범위

- 비식별화된 study ID와 검증된 영상 참조를 사용하는 공통 데이터 계약
- VQA·보고서·시계열 비교 작업 타입
- 모델 제공자에 독립적인 어댑터 및 워크플로 확장 지점

현재는 연구용 초기 구현이며, 실제 모델 호출·DICOM 수집·임상 배포는 포함하지 않는다.
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
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## 개인정보와 데이터

환자 이름, MRN, 생년월일, 원본 DICOM UID 등 식별자를 요청·프롬프트·로그에 넣지 않는다.
`study_id`에는 기관 내부에서만 역매핑 가능한 가명 식별자를 사용한다. 자세한 구현 및
운영 기준은 다음 문서를 확인한다.

- [아키텍처](ARCHITECTURE.md): 모듈 경계와 모델 어댑터 계약
- [데이터 거버넌스](DATA_GOVERNANCE.md): PHI 처리와 로그 정책
- [평가 프로토콜](EVALUATION.md): 모델·프롬프트 변경 전 검증 기준
