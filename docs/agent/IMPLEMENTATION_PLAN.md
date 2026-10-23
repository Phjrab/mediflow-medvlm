# 고정 v1 계획

1. 현재 저장소/호스트 검증, core 3.10 및 TEXT_QA, optional CLI 환경.
2. 실제 MedQuAD 공개 답변, PubMedQA PQA-L fold0, KorMedMCQA 평가 전용, SLAKE 영어 이미지. 확장 source는 권리와 접근이 확인되는 경우만 포함.
3. group/duplicate/conflict 감사, immutable source manifest와 split freeze. seed 20261023, MedQuAD train 상한8000, 전체 text20000/VLM5000.
4. Qwen3 1.7B/4B 및 일반 VLM/MedGemma 또는 지정 fallback: 실제 GPU 연산 확인 후 optimizer smoke20회, 새 process adapter reload, 최대1epoch/1000updates. 모델 누적6h/총24h, 단일GPU worker. 실행 가능한 각 모델만 진행한다.
5. validation으로 프로토콜과 checkpoint 선택 후 paired test. preliminary single seed. MedQuAD test 최대64, PubMedQA test500, KorMedMCQA test 직군별 최대64(총256), SLAKE test 최대256을 성능을 보기 전에 고정한다. 결정적 SHA256(seed+sample_id) 순서로 선택한다. dev는 source별 최대32, 한국어는 직군별16. 다른 task 점수를 하나의 정확도로 합치지 않는다. 생성 상한은 자유QA128, 문맥분류16, MCQ16, VQA32. 동일 pair 정책을 유지한다.
6. 지정 Jetson 읽기 전용 inventory 후 가능한 경우 별도 추론2warmup+10측정. 최대2h, 시스템 변경 없음.
7. 상태/근거/정확한 재개 명령, Korean report, 실제 검증, commit/push 확인.

완료는 실행 상태와 evidence_level을 분리한다. synthetic 성공은 의료 성능이 아니다. 실행하지 않은 baseline/tuned는 N/A로 남긴다. 동일 오류 진단수정2회/일시 네트워크 재시도3회 상한, test 결과로 prompt/data를 바꾸지 않는다.
