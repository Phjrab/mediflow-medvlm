# 현재 저장소 조사 — 2026-10-23

원래 pack의 02 파일은 과거 스냅샷이다. 이번에는 아래 remote를 새 clone/fetch하여 실제 revision을 확인했다. 참고 저장소의 원본 코드와 운영환경은 수정하지 않았다.

| 저장소/branch | 이번 확인 SHA | 읽은 핵심 경로 |
|---|---|---|
| mediflow-medvlm main | 87df64d03acb79eb86f432456141aa9a518239b5 | domain.py, validation.py, workflows.py, README, ARCHITECTURE, DATA_GOVERNANCE, EVALUATION |
| forge-finetune-dashboard main | d9a4656a5bee38cc9903e8caf5c4b680709423d2 | train_worker.py, docs/adapter-contract.md |
| llm-cluster-benchmark main | 49bda2e866d8fa2f25280f6cf761fd09bfcd952f | CONTRIBUTING.md, 연구 근거 계약 |
| llm-cluster-benchmark codex/operational-observability-hardening | 186dc69474610f0436d693dd339da155b83e7ae1 | docs/followups/operational-hardening/F01-integration.md |
| mediflow-kiosk-core main | bfab34c79381d1603a5ada1e8a96bee7babcc284 | docs/AI_EXPERIMENT_PROTOCOL.md, utils/runtime_receipt.py |
| mediflow-datasets main | ffd71eca68ff1e21c145ed56158f4cfa77f7f037 | AGENTS.md, docs/research/EVALUATION_REPRODUCTION_AUDIT_20260927.md |

cluster의 작업 branch는 pack의 `0a17c9d...`에서 바뀌었다. 현재 F01은 코드통합과 검증을 보고하지만 배포·하드웨어 검증을 주장하지 않는다. 이번 연구에도 원칙만 참고하며 기존 formal campaign 규칙은 가져오지 않는다.

주 저장소에는 적용 AGENTS.md가 없다. datasets의 AGENTS는 그 저장소 내부 작업의 규칙이며 읽기 전용 참조에 사용했다. 원본/결과 보존, split 누수 감사, validation 선택 후 test, 실제/미실행 구분 및 한국어 5항목 보고 원칙을 계승한다. 그 저장소의 기존 장비 통합 보류를 이번 사용자 승인 연구의 gate로 삼지 않는다.

Forge의 assistant span과 GPU worker/atomic checkpoint를 참고하되, canonical 데이터 계보를 유지하고 이미지 processor tensor를 보존한다. Kiosk의 E0 분류기/E3 텍스트 설명을 새 의료 QA/VQA 정답으로 사용하지 않는다. 주 저장소 baseline 테스트는 Python3.12에서8개 통과했고, 3.10호환 문제는 코드에 실제 존재했다.
