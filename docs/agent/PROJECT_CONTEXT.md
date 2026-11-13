# 현재 작업 맥락

## 현재 v5 후속 진단 — 2026-11-13 완료

사용자의 다음 단계 진행 요청으로 고정한 검증 전용384기록을 실제 GPU에서 모두 실행했다. 두 모델 Base/v4 Tuned ×기존/한국어 번역 형식 지시 ×16/256token ×24검증문항(세 source 각8개, KorMed 직군별2개)이다. test ID·정확 입력 제외, 최대입력562token, v4 train 정확 input 겹침0을 사전 확인했다. 새 학습·test 추론·기존 parser/점수 보정은 하지 않았다.

모든 worker가 성공했고 session9573은 종료, 전역 GPU active는 null이다. 기술실패0·기존 파일208개 hash 보존·전체 재채점·Base/Tuned native 가중치/template/dtype/prompt·greedy prefix192/192쌍을 검증했다. GPU 추가0.272409시간이다. source는commit3ea0fe8의9개 구현 파일과 byte hash가 일치한다. CPU255passed/7skipped/23subtests·6기존경고,focused4passed다.

두 Tuned는 네 지시/상한 조건 모두 유효 형식0/24였다. Base 기존지시 조건은1.7B22/24,4B21/24였다. 상한256 확대는 길이 종료를 줄였지만 형식 준수를 회복시키지 못했고, 한국어 번역의 효과도 모델마다 달랐다. 이는 작은 기존 validation의 진단이며 임상 능력·새 test 성능·무작위 QA 연결의 단독 인과 효과를 측정한 결과가 아니다. 기존 v4는 약한 참조 실험으로 유지하며 test에 맞춰 parser·prompt·seed·checkpoint를 조정하지 않는다. 성능에 따라 추가 epoch나 sweep을 자동 실행하지 않는다.

공개 결과는 `reports/public_summary/v5/READOUT_KO.md`, `RESULTS_KO.md`, `run_summary.json`, `verification.json`이다. 전체 v1~v5 공개 결과는 `reports/public_summary/INDEX_KO.md`로 연결된다. 코드·집계·그림은 전용 작업 브랜치에 commit/push하며 main merge·배포는 하지 않았다. 원본·가중치·전체 예측·A/B 키는 로컬 private data root에 있으며 GitHub만 복제하면 이 자료까지 복구되지 않는다. 사용자 전달 사본과 재개 명령·최종 CI 영수증은 workspace `outputs/v5/`에 기록한다. 전문의 검토는0건이며 기존 빈 검토표는 `outputs/v4/review/`다.

## 현재 후속 작업 — 2026-11-13 v4 완료

사용자가 2026-11-12 “71762는 제공처 방식으로 별도 실험 진행”을 명시했다. 이는 71762에 한해 카테고리·질환명·의도 안의 무작위 답변 연결 및 미제공 증강 lineage 한계를 허용한 예외다. 다른 source의 연결 기준은 바꾸지 않는다. 제공처 회신은 사용자 전달문이며 원본 메일 헤더를 인증한 것으로 표시하지 않는다. 공식 `make_train_data.py`는 미확보이며 제공처가 설명한 방식을 독립 구현했다. 정확한 고정 QA 매핑·parent 파일 확보를 현재 사용자 선행 작업으로 다시 요구하지 않는다.

2026-11-13 01:55:56 KST에 선언한 유한 v4 실험이 `SUCCEEDED`로 끝났고 전체 산출물 검증도 통과했다. 실행 소유 session79763이 종료됐으며 GPU 원장 active는 null이다. 원본 Base에서 독립 학습한 Qwen3-1.7B/4B 각각20update smoke·250update/1pass 본학습과 각 새 process 재로드를 완료했다. 두 본학습 모두4,000표본/정답1,093,637token을 처리했고 validation으로step250을 선택했다. test 선택·기존 EN/KO/MIX adapter 연속학습은 하지 않았다.

- 데이터: 학습4,000/검증105/파생평가256, 19질환 분류. 공식 Validation에서 hash로 dev/test를 파생했으며 공식 Test라고 부르지 않는다. KorMedMCQA 전체4,068 포함 고유5,589개 보호자료와 lexical/E5 의미 유사도를 검사했고 후보2,982개를 보수적으로 격리했다. 확정 의학/번역 중복 판정이나 augmentation parent 복원은 아니다. 동결 표본 source text·답변 section·cross-split exact hash를 재검증했다.
- 실행: 학습·재로드4단계, 새 평가6개/생성기록10,084건(조건별 반복 포함), 기존 공통 Base2개 검증. 새 공통 평가는 모델별4,530개, 무작위 참조는 Base/Tuned 각256개다. 모든 pair의 input/native prompt token hash와 생성 설정이 일치한다. 기술 실패0이며 기존 공개 보고서·선별 역사 파일77개 hash를 보존했다.
- 결과: 무작위 참조 lexical F1은1.7B 0.0864→0.0914,4B 0.1039→0.1148이다. 임상 개선 근거가 아니다. 양 모델 공통 객관식·분류형4,286개가 모두 invalid_format이며 KorMedMCQA0/3009는 형식 실패를 포함한 엄격한 점수다. 의학적 능력0이라고 해석하거나 parser·상한·프롬프트를 사후 변경하지 않는다. 한국어 자유서술 전용 학습·문체·길이·token 노출 차이로 무작위 연결 방식의 인과 효과를 분리할 수 없다.
- 검증: CPU251passed·7선택실행 제외·23subtests passed,6기존 Pillow 경고. 실제 GPU doctor의 forward/backward와4학습/fresh reload는 CPU 선택 검사와 별도의 근거다. GPU 추가7.644시간이며 unlimited/null 정책·누적 사용 원장·checkpoint floor는 보존했다.
- 미실행/한계: 공식 코드 실행, 전체 원본 학습, 추가seed, 새 MedGemma/VLM 학습, 전문가 임상 검토는 하지 않았다. 검토표2개·32비교행은 같은16문항이며 사람 점수는 모두 빈칸이다. Jetson은 USER_DEFERRED다. 실행 차단은 없으나 원본 parent·정확 QA 대응이 없으므로 lineage 격리와 임상 정답성을 보장하지 않는다.

공개 aggregate는 `reports/public_summary/v4/RESULTS_KO.md`, `run_summary.json`, `verification.json`, `figures/`다. 원본·모델·전체 예측·A/B 키는 private data root에 보존한다. 사용자 전달물은 workspace `outputs/v4/`이며 실제 checkpoint 경로는 `local_artifacts.json`, 재개 명령은 `RESUME_KO.md`, 최종 commit/push/CI 근거는 `git_receipt.json`에 기록한다. 아래 v1/v2/v3는 당시 기록으로 보존하며 현재 완료 상태를 대신하지 않는다.

재개는 기존 성공 산출물의 identity·hash를 검증 후 재사용한다. 이전 freeze·결과에 덮어쓰거나 결과에 따른 재추첨·추가seed를 자동 실행하지 않는다.

```bash
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_DISABLED=true
../.venv-medvlm/bin/python scripts/run_71762_v4.py --data-root ../medvlm-data --phase all
../.venv-medvlm/bin/python scripts/report_71762_v4.py --data-root ../medvlm-data --output reports/public_summary/v4
```

## 보존된 2026-10-28 맥락 — 71762 로컬 원본

사용자가 두 로컬 ZIP을 제공했다. `120.초거대AI 사전학습용 헬스케어 질의응답 데이터.zip`이 71762, `09.필수의료 의학지식 데이터.zip`은 기존 71875와 동일 SHA다. 71762의 이전 `BLOCKED_DELIVERY`는 해소됐다. 현재 차단은 `UNRESOLVED_PAIRING`이며 원본 다운로드 실패나 API 키 누락이 아니다.

- 71762 4,248,340,940bytes, SHA `89584900411ac869817bd85c190ad4842f5839e9e0dc1fb85140113a148e99e4`. 내부 5 ZIP·7,028,863 파일 CRC/SHA 감사 완료. Training 질문1,143,131/답변1,966,902, Validation 질문141,292/답변245,404의 별도 구성요소3,496,729개를 정규화했다. TS/TL·VS/VL 내용은 같은 경로에서 전부 같고 중복 합산하지 않았다.
- private SQLite에 출처와 question 또는 answer 구조를 보존했다. 원본·split을 유지했고 source 문서를 QA로 만들지 않았다. 답변 ID 반복281,241그룹과 공식 split 간 동일 질문10,464그룹/28,930행을 관측했다. 전역 ID 고유성·증강 parent를 추정하지 않는다.
- 공식 HWP 설명서·가이드를 정상 브라우저 링크로 확보하고 본문을 읽었다. 가이드는 카테고리·의도 연결을 설명하나 사용자 지시는 이 방식의 join을 제외했다. 실제 Q→A foreign key·증강 parent 계약은 확인되지 않았다. QA쌍0, 새freeze·학습·평가는 미실행이다. 기존 v2 실험을 재실행하지 않았다.
- KorMedMCQA4,068개와 질문 본문의 정확 일치0. near/semantic·번역 감사는 paired train 후보가 생긴 뒤 필요하며 이번에는 미실행이다. 구성요소 문법·필드 검사를 임상적 검증이나 학습 적격성 확인으로 해석하지 않는다.
- 핵심 CPU 회귀241passed/7skipped/23subtests, 관련데이터21passed. v1/v2 공개 보고서38개·v2 실행소스14개·GPU원장 보존 검증 완료. 새 GPU 실행은 없고 unlimited 정책·Jetson USER_DEFERRED·human_review not_performed를 유지한다.

현재 보고서는 `reports/public_summary/v3/RESULTS_KO.md`, 실제 입력·감사·SQLite는 global data root의 `v3/`다. 데이터 처리 및 재개 명령은 `docs/research/DATA_AI_HUB_71762_KO.md`와 전달 폴더 `outputs/v3/RESUME_KO.md`에 있다. 기존 v2 결과/전달물은 당시 상태로 보존한다. 새 감사의 종료코드0은 component audit 성공이며 paired SFT 준비 완료가 아니다. 동일 audit를 동시에 실행하지 않는다.

사용자에게 필요한 입력은 71762 공식 소스·전처리 코드 또는 scoped 질문ID↔답변ID 및 증강원본parent 매핑 파일이다. 사용자가 다른 컴퓨터의 공식 페이지 ‘활용 AI 모델 및 코드’에서 파일을 확인할 수 있도록 안내했다. 이미 받은 ZIP·설명서·활용가이드·API 키·모델 가중치를 다시 요구하지 않는다. 자료가 생기면 실제 cardinality와 누수·token 검사를 거친 새 freeze로 이어가고 완료된 v2 corpus에 덧붙이지 않는다. 최종 commit/push/CI 근거는 `outputs/v3/git_receipt.json`에 별도 기록한다.

## 보존된 v2 종료 기록 — 2026-10-27 06:46 KST

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
