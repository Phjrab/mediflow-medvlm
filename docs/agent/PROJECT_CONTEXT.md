# 현재 작업 맥락

## 현재 후속 작업 — 2026-10-27 06:46 KST

실행 가능한 v2 선언 행렬은 완료했다. 전체 상태는 `COMPLETED_AVAILABLE_WORK_WITH_EXTERNAL_BLOCKER`이며 AI-Hub 71762는 `BLOCKED_DELIVERY`다. `reports/public_summary/v2/READOUT_KO.md`가 핵심 해석, `RESULTS_KO.md`가 전체 결과, `verification.json`이 실제 산출물 검증 근거다. `runtime/medvlm_state.json`은 변경하지 않은 v1 역사 상태이고, v2 인계는 `runtime/medvlm_state_v2.json`에 별도로 둔다.

- 현재 resource policy는 unlimited이며 total/model/run/job/campaign 상한은 모두 null이다. 사용량 원장·재개 identity·checkpoint floor는 보존했다. 시간 상한 없음이 추가 무한 실험을 뜻하지 않는다.
- 71875 실제 canonical 17,210개(train15,297/dev956/test957), KO4,000/MIX8,000, 공통 dev1,516을 준비·동결했다. KorMedMCQA 전체 dev/test4,068 보호를 위해 lexical/semantic 후보를 검사했고 후보 train/dev3,208행을 보수적으로 제외했다. 후보를 확정 중복이라고 부르지 않는다.
- 네 KO/MIX 본학습과 네 smoke 모두 성공했고 새 process 재로드를 통과했다. 원본 Base에서 독립 학습했으며 Qwen3-4B KO는 full250 update 중 validation step200, 나머지는 최종 step250/500을 선택했다. 텍스트9조건×4,530문항을 완료했다. MedGemma는 adapter 없는 native text Base다.
- 기존 VLM 가중치로 VQA validation8+본평가·진단12를 완료했다. VLM 재학습은 하지 않았다. MedGemma의 SLAKE dataset-level 사전학습 노출과 정확한 test 노출 미상 한계를 유지한다.
- MedMCQA는 공식 MIT 근거로 원본193,155행 확보·감사, optional canonical22,816행을 준비했다. 주 KO/MIX에는 미포함이고 추가 학습하지 않았다. v1 예측의 형식/길이/OPEN-CLOSED/8,450→8,400 재분석은 완료했다.
- 익명 검토표4개·64비교행은 동일16문항이며 사람의 검토는0건이다. 전달 사본만 편집하고 private key는 별도 보존한다. 추가seed는 NOT_SELECTED, 예비모델은 fallback 불필요다.
- 71762 약4.25GB 파일은 링크를 받았지만 connector256MiB 한도와 브라우저 ERR_BLOCKED_BY_CLIENT로 원본0bytes다. 실제 pairing은 UNRESOLVED_PAIRING이다. 필요한 외부 입력은 로컬 원본 경로와 제공 가능한 가이드다. 이미 받은71875에 API 키를 요구하지 않는다.
- Jetson은 USER_DEFERRED다. 과거 호스트키 미검증 이력은 아래에 보존하며 현재 사용자 할 일로 요청하지 않는다.

완료 평가29개(validation포함), 학습·재로드8개, 검토표4개를 검증했다. CPU234passed·7skipped·23subtests와 별도 native41passed·20subtests를 구분한다. v1 공개 파일18개와 실행 소스 스냅샷14개의 보존을 재확인했다. 실행 소유 worker가 종료됐으며 재개 전에는 상태와 실제 PID를 다시 확인한다. 공개 결과는 원문·예측·가중치·비밀값을 포함하지 않는다.

저장소에서 다음 명령을 사용한다. 성공 결과는 검증·재사용하며 v1 보고서는 재생성하지 않는다.

```bash
export MEDVLM_DATA_ROOT="$(realpath ../medvlm-data)"
export PYTHONPATH="$PWD/src"
export WANDB_DISABLED=true
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
../.venv-medvlm/bin/python scripts/run_vqa_v2.py --data-root "$MEDVLM_DATA_ROOT"
../.venv-medvlm/bin/python scripts/run_text_v2.py --data-root "$MEDVLM_DATA_ROOT" --phase all
../.venv-medvlm/bin/python scripts/report_v2.py --data-root "$MEDVLM_DATA_ROOT" --output reports/public_summary/v2
MPLCONFIGDIR=/tmp/medvlm-v2-mpl-cache ../.venv-medvlm/bin/python scripts/plot_v2.py --summary reports/public_summary/v2/run_summary.json --output reports/public_summary/v2/figures
```

전용 작업 branch `codex/medical-finetune-v1-20261023`만 사용한다. 최종 commit/push/CI 영수증은 작업공간의 `outputs/v2/git_receipt.json`에 별도 기록하여 commit 자기참조를 피한다. 데이터·모델·비밀값은 Git 밖에 보존한다. main merge·배포·유료 cloud·OS/드라이버 교체는 수행하지 않는다.

## 보존된 2026-10-23 v1 맥락

아래는 당시 기록이다. 현재 승인 범위·정책·차단 상태는 위 10/26 후속 항목이 우선한다.

2026-10-23 Asia/Seoul. 공개 자료로 수행하는 오프라인 의료 QA/VQA 연구다. 사용자 제공 `mediflow_medvlm_codex_pack_20261023`의 01, 03~09, 11을 실행 계약으로 채택했다. 공개 데이터/모델 확보, 전용 환경, 로컬 GPU 학습, 제한된 Jetson 검증, 전용 branch commit/push는 승인됐다. 기본 branch merge, 유료 서비스, Hub 업로드, 시스템 드라이버 교체는 범위 밖이다.

최초 확인한 main HEAD `87df64d03acb79eb86f432456141aa9a518239b5`는 사전 스냅샷과 일치한다. 새 clone에서 사용자 dirty 파일은 없었고 적용 AGENTS.md는 없었다. 작업 branch는 `codex/medical-finetune-v1-20261023`. 기준 Python 3.12 unittest 8개는 통과했다. Python 3.10 StrEnum 호환성과 TEXT_QA 지원을 수정했고 Python3.10 실제 계약 검사도 통과했다.

실행 자료는 `MEDVLM_DATA_ROOT`의 raw/processed/manifests/models/runs에 저장한다. 로컬 실제 경로는 Git 제외 runtime 상태에 기록한다. 공개 요약에 원문/이미지/예측/인증값을 넣지 않는다. RTX 3060에서 Qwen3-1.7B/4B 각각525update·fresh-process 재로드·820문항 base/tuned 평가를 완료했다. Qwen3.5-2B/MedGemma1.5-4B는 각각 SLAKE1,979건/124update 학습 후 validation으로step100/124를 선택하고 fresh-process 재로드·SLAKE256문항씩·VQA-RAD64문항씩 base/tuned 평가·no-image32문항씩을 완료했다. 총 GPU 점유5.5687시간, 필수 text2+VLM2 목표 달성. NVML kernel/library mismatch에도 실제 CUDA forward/backward가 성공했으며 드라이버를 변경하지 않았다.

분업: text_data는 공통 canonical·audit·freeze와 MedQuAD/PubMedQA/KorMedMCQA, vision_sources는 SLAKE와 확장 source, training_core는 core 호환성과 training/runtime, 주 작업자는 평가·장치·통합·인계를 소유한다. 다른 참고 저장소는 읽기 전용이다.

데이터 canonical 19,866개, aggregate freeze 1562258c4b5b6f3b4c5c8948851fbb972da603cd63bff528aa7d278cf2253c86. 학습은 기존 core freeze 1bf58dbc8baca7a3f5865accf5a86e9fba0b84f8abe3c986ed34b67d6c3d87c5를 유지한다. 사용자 새 계정으로 MedGemma 및 Gemma3 접근을 확인했고 Gemma3 실제 weight HEAD도 성공했다. Gemma3/Qwen3-VL/Qwen2.5-VL/SmolVLM은 CPU 준비만 완료했으며 fallback 불필요로 GPU 학습하지 않았다. Jetson은 호스트키 미검증, AIHub는 접근 자격, MedQA/MedMCQA는 외부 원문 권리 미확정으로 별도 차단 상태다. 전문가 검토·임상 검증·추가 seed는 미실행이다.

MedGemma 기본 SLAKE EM0은 장문 응답과 엄격한 단답 lexical 점수의 불일치 영향을 크게 받았다. 실제 이미지 입력·hash·pair alignment는 검증됐으나 임상 정답률0 또는 +69.92%p 의학적 능력 향상으로 해석하지 않는다. 최종 결과·실행 상태·형식 진단·재개 절차는 reports/public_summary와 HANDOFF.md에 보존한다.
