# 현재 작업 맥락

2026-10-23 Asia/Seoul. 공개 자료로 수행하는 오프라인 의료 QA/VQA 연구다. 사용자 제공 `mediflow_medvlm_codex_pack_20261023`의 01, 03~09, 11을 실행 계약으로 채택했다. 공개 데이터/모델 확보, 전용 환경, 로컬 GPU 학습, 제한된 Jetson 검증, 전용 branch commit/push는 승인됐다. 기본 branch merge, 유료 서비스, Hub 업로드, 시스템 드라이버 교체는 범위 밖이다.

최초 확인한 main HEAD `87df64d03acb79eb86f432456141aa9a518239b5`는 사전 스냅샷과 일치한다. 새 clone에서 사용자 dirty 파일은 없었고 적용 AGENTS.md는 없었다. 작업 branch는 `codex/medical-finetune-v1-20261023`. 기준 Python 3.12 unittest 8개는 통과했다. Python 3.10 StrEnum 호환성과 TEXT_QA 지원을 수정했고 Python3.10 실제 계약 검사도 통과했다.

실행 자료는 `MEDVLM_DATA_ROOT`의 raw/processed/manifests/models/runs에 저장한다. 로컬 실제 경로는 Git 제외 runtime 상태에 기록한다. 공개 요약에 원문/이미지/예측/인증값을 넣지 않는다. RTX 3060에서 Qwen3-1.7B/4B 각각525update·fresh-process 재로드·820문항 base/tuned 평가를 완료했다. Qwen3.5-2B/MedGemma1.5-4B는 각각 SLAKE1,979건/124update 학습 후 validation으로step100/124를 선택하고 fresh-process 재로드·SLAKE256문항씩·VQA-RAD64문항씩 base/tuned 평가·no-image32문항씩을 완료했다. 총 GPU 점유5.5687시간, 필수 text2+VLM2 목표 달성. NVML kernel/library mismatch에도 실제 CUDA forward/backward가 성공했으며 드라이버를 변경하지 않았다.

분업: text_data는 공통 canonical·audit·freeze와 MedQuAD/PubMedQA/KorMedMCQA, vision_sources는 SLAKE와 확장 source, training_core는 core 호환성과 training/runtime, 주 작업자는 평가·장치·통합·인계를 소유한다. 다른 참고 저장소는 읽기 전용이다.

데이터 canonical 19,866개, aggregate freeze 1562258c4b5b6f3b4c5c8948851fbb972da603cd63bff528aa7d278cf2253c86. 학습은 기존 core freeze 1bf58dbc8baca7a3f5865accf5a86e9fba0b84f8abe3c986ed34b67d6c3d87c5를 유지한다. 사용자 새 계정으로 MedGemma 및 Gemma3 접근을 확인했고 Gemma3 실제 weight HEAD도 성공했다. Gemma3/Qwen3-VL/Qwen2.5-VL/SmolVLM은 CPU 준비만 완료했으며 fallback 불필요로 GPU 학습하지 않았다. Jetson은 호스트키 미검증, AIHub는 접근 자격, MedQA/MedMCQA는 외부 원문 권리 미확정으로 별도 차단 상태다. 전문가 검토·임상 검증·추가 seed는 미실행이다.

MedGemma 기본 SLAKE EM0은 장문 응답과 엄격한 단답 lexical 점수의 불일치 영향을 크게 받았다. 실제 이미지 입력·hash·pair alignment는 검증됐으나 임상 정답률0 또는 +69.92%p 의학적 능력 향상으로 해석하지 않는다. 최종 결과·실행 상태·형식 진단·재개 절차는 reports/public_summary와 HANDOFF.md에 보존한다.
