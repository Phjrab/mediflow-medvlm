# 작업 기록

- 2026-10-23: 사용자 pack 전체 계약을 읽음. 현재 repository SHA는 snapshot과 일치. initial unit tests 8/8 (Python3.12). 새 branch 생성.
- sandbox DNS 제한은 승인된 network escalation으로 해결, 저장소 clone 성공. 시스템 변경 없이 CUDA 직접 초기화 성공; NVML mismatch를 GPU 부재로 단정하지 않도록 조사 수정.
- 실제 원본과 실행 artifact는 Git 외부 data root, feature branch에는 코드/설정/비식별 요약만 유지. 사용자 승인에 따라 독립 source와 CPU 구현 분업 시작.
- 2026-10-23 17:38 KST: 사용자 새 Hugging Face 계정 연결 지시에 따라 gated 접근 재확인. 프로젝트 HF_HOME에는 인증이 없어401이었으나 기본 로그인 캐시의 기존 인증을 사용하자 MedGemma1.5-4B config 접근 성공. Gemma3-4B는403 유지. 인증값은 출력·복사하지 않음. MedGemma를 본 VLM 셀로 복귀하고 SmolVLM은 실패 시 fallback으로 보존. 기존 freeze/평가 샘플 정책은 변경 없음. MedGemma 고정revision 다운로드 시작(maxworkers2).
- 2026-10-23 17:46 KST: GPU 학습 PID를 그대로 유지하고 자식 없는 완료 대기 wrapper만 재시작. 새 wrapper는 validation generation→test protocol freeze→paired test→유한한 추가 진단을 실행한다. Pre-training validation loss는 학습 전에 측정했고, unadapted base의 validation generation은 학습 종료 후 동일 고정 base를 새로 로드해 측정한다. 이 시간 순서 차이를 pre-training generation으로 표기하지 않는다.
- core Python3.10 검사72개+subtest3 통과(native GPU/LoRA 테스트 opt-in1 제외). 평가 입력·부모모델 해시·no-image·전문가 빈 검토표 focused 검사48개 통과. CPU synthetic LoRA는 실제 의료 성능 근거와 분리.
- 2026-10-23 17:54 KST: 사용자 Gemma3 추가 승인 후 같은 계정으로 config 접근 및 Gemma3Processor text/multiturn/padding 검사 성공. 과거401/403 attempt는 별도 보존. Gemma3도 접근 차단 해소. MedGemma→Gemma3→Smol 순서로 실패 시에만 fallback하며 본 목표는 text2+VLM2 유지. Gemma3 전체 가중치는 필요 시 다운로드한다.

- 2026-10-23 18:16 KST: Qwen3-1.7B 본 학습525 update/8,400 sample pass/1,983,789 target token과 fresh-process reload 완료. 동일 frozen test820문항 base/tuned 각각 실제 완료. PubMedQA209/500→372/500, KorMedMCQA56/256→102/256. 한국어 유효응답143/256→255/256이므로 전체 개선에는 형식 준수 개선도 포함한다. 단일 seed 예비 결과이며 test에 맞춰 parser/문항/설정을 변경하지 않았다. Qwen3-4B smoke20 update와 재로드 후 본 학습 시작.
- feature branch 최초 구현 commit cff65471c426856007e58adf62c8078101a63441의 remote push를 ls-remote로 확인했고 GitHub Actions Test run39929597344가 성공했다. 원본·모델·raw 예측·인증값은 commit하지 않았다.
