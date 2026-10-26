# 현재 작업 맥락

## 2026-10-26 후속 범위와 현재 상태

현재 사용자는 기존 v1 결과를 보존하면서 GPU 시간 상한 해제, 한국어 의료 SFT, 기존 예측 재분석과 VQA v2를 요청했다. 기준 branch는 `codex/medical-finetune-v1-20261023`, 후속 시작 시 HEAD는 `e16f8c6df3244a78cd9c30ba5ce7e43d5b8f45a3`였다. 코드·데이터 준비·로컬 GPU 실행·오류 수정·전용 branch commit/push는 승인됐고 main merge·배포·유료 서비스·시스템/드라이버 교체·원문/비밀값/가중치 공개 업로드는 제외한다.

현재 정책은 `configs/resource_policy.json`의 unlimited이며 누적 전체/모델별 GPU 시간 및 run/job/campaign 시간 상한은 `null`이다. 기록·재개 무결성·usage floor는 유지한다. 과거 v1의 24/6 상한과 1-pass/1,000-step recipe는 실행 당시 기록으로 남기고 기존 config나 checkpoint를 소급 수정하지 않는다. 실행 범위는 유한한 선언 행렬이며 성능이 좋아질 때까지 반복하지 않는다.

Jetson은 사용자 요청으로 `USER_DEFERRED`다. v1 호스트키 미검증 이력은 보존하지만 현재 접속·검증·변환·장치 추론은 하지 않는다. `runtime/medvlm_state.json`은 v1 역사 상태이며 v2 전체 root 최종 상태는 아직 없다. 후속 결과는 `reports/public_summary/v2/`에 저장한다.

AI-Hub 71875는 실제 canonical 17,210개(train 15,297 / validation 956 / test 957)를 준비했다. 71762 약 4.25GB 전달 파일은 connector 256MiB 상한과 브라우저 `ERR_BLOCKED_BY_CLIENT` 때문에 아직 확보하지 못했고 로컬 원본 경로 문의가 대기 중이다. 이것을 API 키 필수나 사용자 승인 실패로 바꾸지 않는다. 정상 제공 원본부터 계속 처리하며 명시 pairing이 확인되지 않은 부분은 분리한다. MedMCQA는 공식 MIT 근거로 원본 193,155행 확보·감사와 optional canonical 22,816행을 완료했으나 한국어 주 실험에 넣지 않았다.

Qwen3-1.7B/4B의 KO-v2 4,000개(객관식1,500/단답1,300/서술1,200)와 MIX-v2 8,000개(같은 KO4,000+영어4,000), 공통 validation1,516을 동결했다. KorMedMCQA dev/test4,068 보호를 위한 lexical 검사와 E5 의미 유사도 후보 감사를 완료했으며 후보 train/dev3,208개를 보수적으로 제외했다. 후보를 확정 중복으로 부르지 않고 test957은 보존한다. KO250/MIX500 update 한 pass를 원본 Base에서 시작하며 각각20-update smoke와 fresh-process reload를 먼저 수행한다. 현재 텍스트 실행은 VQA 종료를 기다리는 큐에 있고 학습·재로드·평가는 아직 미완료다. 기존 Qwen3.5-2B/MedGemma Base·선정 adapter의 VQA validation8개가 완료되어256-token 상한을 고정했고 전체SLAKE1,061/VQA-RAD451 및no-image32 평가를 진행 중이다. 재개 진입점은 `scripts/run_vqa_v2.py`와 `scripts/run_text_v2.py --phase all`; 상태는 각 `v2/{vqa,text}/state.json`이다. 소유 worker와 대기 queue가 살아 있으면 중복 실행하지 않는다. 실제 완료 artifact만 v2 보고서에 반영한다.

기존 예측 재분석은 10개 실행의 파일·protocol·model/adapter identity를 검증한 뒤 완료했다. 8,450→8,400은 양 모델 모두 MedQuAD 50개가 native context 길이를 초과해 제외된 결과이고 나머지는 525 updates로 모두 처리했다. KorMedMCQA의 형식 유효성 전환과 both-valid 정답 변화, MedGemma의 길이·구절 포함·잘림, VQA-RAD OPEN/CLOSED를 분리했다. 구절 포함이나 문자열 점수를 임상 정확도로 표현하지 않는다. 전문가 검토는 `not_performed`다.

## 보존된 2026-10-23 v1 맥락

아래는 당시 기록이다. 현재 승인 범위·정책·차단 상태는 위 10/26 후속 항목이 우선한다.

2026-10-23 Asia/Seoul. 공개 자료로 수행하는 오프라인 의료 QA/VQA 연구다. 사용자 제공 `mediflow_medvlm_codex_pack_20261023`의 01, 03~09, 11을 실행 계약으로 채택했다. 공개 데이터/모델 확보, 전용 환경, 로컬 GPU 학습, 제한된 Jetson 검증, 전용 branch commit/push는 승인됐다. 기본 branch merge, 유료 서비스, Hub 업로드, 시스템 드라이버 교체는 범위 밖이다.

최초 확인한 main HEAD `87df64d03acb79eb86f432456141aa9a518239b5`는 사전 스냅샷과 일치한다. 새 clone에서 사용자 dirty 파일은 없었고 적용 AGENTS.md는 없었다. 작업 branch는 `codex/medical-finetune-v1-20261023`. 기준 Python 3.12 unittest 8개는 통과했다. Python 3.10 StrEnum 호환성과 TEXT_QA 지원을 수정했고 Python3.10 실제 계약 검사도 통과했다.

실행 자료는 `MEDVLM_DATA_ROOT`의 raw/processed/manifests/models/runs에 저장한다. 로컬 실제 경로는 Git 제외 runtime 상태에 기록한다. 공개 요약에 원문/이미지/예측/인증값을 넣지 않는다. RTX 3060에서 Qwen3-1.7B/4B 각각525update·fresh-process 재로드·820문항 base/tuned 평가를 완료했다. Qwen3.5-2B/MedGemma1.5-4B는 각각 SLAKE1,979건/124update 학습 후 validation으로step100/124를 선택하고 fresh-process 재로드·SLAKE256문항씩·VQA-RAD64문항씩 base/tuned 평가·no-image32문항씩을 완료했다. 총 GPU 점유5.5687시간, 필수 text2+VLM2 목표 달성. NVML kernel/library mismatch에도 실제 CUDA forward/backward가 성공했으며 드라이버를 변경하지 않았다.

분업: text_data는 공통 canonical·audit·freeze와 MedQuAD/PubMedQA/KorMedMCQA, vision_sources는 SLAKE와 확장 source, training_core는 core 호환성과 training/runtime, 주 작업자는 평가·장치·통합·인계를 소유한다. 다른 참고 저장소는 읽기 전용이다.

데이터 canonical 19,866개, aggregate freeze 1562258c4b5b6f3b4c5c8948851fbb972da603cd63bff528aa7d278cf2253c86. 학습은 기존 core freeze 1bf58dbc8baca7a3f5865accf5a86e9fba0b84f8abe3c986ed34b67d6c3d87c5를 유지한다. 사용자 새 계정으로 MedGemma 및 Gemma3 접근을 확인했고 Gemma3 실제 weight HEAD도 성공했다. Gemma3/Qwen3-VL/Qwen2.5-VL/SmolVLM은 CPU 준비만 완료했으며 fallback 불필요로 GPU 학습하지 않았다. Jetson은 호스트키 미검증, AIHub는 접근 자격, MedQA/MedMCQA는 외부 원문 권리 미확정으로 별도 차단 상태다. 전문가 검토·임상 검증·추가 seed는 미실행이다.

MedGemma 기본 SLAKE EM0은 장문 응답과 엄격한 단답 lexical 점수의 불일치 영향을 크게 받았다. 실제 이미지 입력·hash·pair alignment는 검증됐으나 임상 정답률0 또는 +69.92%p 의학적 능력 향상으로 해석하지 않는다. 최종 결과·실행 상태·형식 진단·재개 절차는 reports/public_summary와 HANDOFF.md에 보존한다.
