# 현재 작업 맥락

2026-10-23 Asia/Seoul. 공개 자료로 수행하는 오프라인 의료 QA/VQA 연구다. 사용자 제공 `mediflow_medvlm_codex_pack_20261023`의 01, 03~09, 11을 실행 계약으로 채택했다. 공개 데이터/모델 확보, 전용 환경, 로컬 GPU 학습, 제한된 Jetson 검증, 전용 branch commit/push는 승인됐다. 기본 branch merge, 유료 서비스, Hub 업로드, 시스템 드라이버 교체는 범위 밖이다.

최초 확인한 main HEAD `87df64d03acb79eb86f432456141aa9a518239b5`는 사전 스냅샷과 일치한다. 새 clone에서 사용자 dirty 파일은 없었고 적용 AGENTS.md는 없었다. 작업 branch는 `codex/medical-finetune-v1-20261023`. 기준 Python 3.12 unittest 8개는 통과했다. Python 3.10 StrEnum 호환성과 TEXT_QA 지원을 수정했고 Python3.10 실제 계약 검사도 통과했다.

실행 자료는 `MEDVLM_DATA_ROOT`의 raw/processed/manifests/models/runs에 저장한다. 로컬 실제 경로는 Git 제외 runtime 상태에 기록한다. 공개 요약에 원문/이미지/예측/인증값을 넣지 않는다. GPU는 PCI에서 RTX 3060 확인. NVML은 kernel/library version mismatch이나 직접 CUDA cuInit은 성공했다. PyTorch CUDA forward/backward가 실제 성공했고 Qwen3-1.7B는 실제525 update·재로드·820문항 base/tuned 비교를 마쳤다. Qwen3-4B는 smoke·재로드를 통과해 본 학습 중이다. NVML 오류를 근거로 드라이버를 변경하지 않았다.

분업: text_data는 공통 canonical·audit·freeze와 MedQuAD/PubMedQA/KorMedMCQA, vision_sources는 SLAKE와 확장 source, training_core는 core 호환성과 training/runtime, 주 작업자는 평가·장치·통합·인계를 소유한다. 다른 참고 저장소는 읽기 전용이다.

데이터 canonical 19,866개, aggregate freeze 4dc1e59694d90858b3c3bd81f0963ec798aba09180b014308e789471422e7b07. 학습은 기존 core freeze 1bf58dbc8baca7a3f5865accf5a86e9fba0b84f8abe3c986ed34b67d6c3d87c5를 유지한다. 17:38 KST 사용자 새계정으로 MedGemma 접근 성공, Gemma3은 첫 재확인403이었으나 사용자 추가 승인 후17:54KST 접근 및 processor 검사 성공. MedGemma는 원래 VLM셀로 복귀하고 Smol은 fallback이다. Jetson은 현재 호스트키 미검증으로 차단됐다.
