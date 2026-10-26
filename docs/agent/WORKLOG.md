# 작업 기록

## 2026-10-27 06:46 KST — v2 선언 행렬 최종 완료

텍스트9조건·VQA20조건, KO/MIX smoke/full8개와 fresh-process reload, 검토표4개 준비를 모두 검증했다. 전체47,202개 예측을 완료했고 기술 실패는 최종 verification과 metrics로 확인했다. 보고서·그림을 최종 summary로 생성했고 v1공개18파일·실행소스14파일 보존을 확인했다. CPU234passed·7skipped·23subtests, native41passed·20subtests 근거를 보존했다. AIHub71762 BLOCKED_DELIVERY/UNRESOLVED_PAIRING, Jetson USER_DEFERRED, human_review not_performed0건, 추가seed NOT_SELECTED, MedMCQA 추가학습 PREPARED_NOT_RUN을 분리했다. 전달 위치는 작업공간 outputs/v2이며 commit/push/CI 결과는 같은 폴더 git_receipt.json에 기록한다.

## 2026-10-27 06:03 KST — Qwen 공통 평가·검토표 준비 완료

- Qwen3-4B Base/EN-v1/KO-v2/MIX-v2 각각4,530문항 평가 완료. KorMedMCQA3,009 정답은1,314/1,638/1,544/1,627(coverage80.76%/99.83%/100%/100%). PubMedQA500 정답은177/384/255/384이며 Base coverage91.8%, 나머지100%다. AI-Hub MCQ777 정답은451/528/563/558.
- MIX는 이 단일 seed에서 EN과 PubMedQA 정답 수가 같고, KorMedMCQA는11문항 적었다. AI-Hub MCQ는30문항 많았다. KO/MIX의 데이터·token·update 노출량이 달라 언어 혼합의 인과 효과나 임상적 우월성을 주장하지 않는다. 1.7B의 PubMedQA coverage도 네 조건 모두100%임을 확인했다.
- 두 Qwen의8개 조건, VQA20개, 새 학습·재로드8개와 익명 검토표4개 검증 오류0. 텍스트36,240개 예측에서 vision_ingested=false 및 native pixel tensor 없음 확인. 검토표는 동일16문항을 공유하는64개 비교행이며 사람 검토0건이다.
- 실제 native MedGemma Base 텍스트 추론 시작. VQA adapter는 사용하지 않으며 마지막4,530문항 평가는 아직 진행 중이다.
- 최종 보고의 native text 검증을 강화했다. hash가 일치하더라도 픽셀 입력이 섞인 text 결과를 거부하는 회귀검사를 추가했다. Core Python3.10 전체234passed/7skipped/6warnings/23subtests passed. 알려진 MedGemma SLAKE dataset-level 노출과 정확한 test 노출 미상은 결과 보고와 VQA 도표에 함께 유지했다.
- 사용자 Drive 폴더 재조회 결과 기존71875/71762 ZIP 두 개만 보이며 추가 가이드·분할 archive는 없었다. 원본 다운로드를 재시도하거나 차단 경계를 우회하지 않았다.

## 2026-10-27 03:43 KST — Qwen3-1.7B 공통 평가 완료

- Base/EN-v1/KO-v2/MIX-v2 각각4,530문항을 평가하고 실제 예측·protocol·adapter·점수 재계산 검증을 통과했다. KorMedMCQA3,009 정답 수는572/1,176/1,143/1,183, coverage는54.90%/99.80%/100%/100%다. 전체 Base 대비 차이에는 형식 준수 개선이 포함된다.
- PubMedQA500 정답은209/372/169/353이다. KO의 영어 성능 하락을 그대로 보존했다. AI-Hub MCQ777 정답은174/430/482/467. MIX와 EN의 KorMedMCQA 차이는7문항으로, 학습량 차이와 단일 seed를 고려하지 않고 우월성을 주장하지 않는다. test 결과로 학습량·prompt·parser를 바꾸지 않았다.
- Base–KO/Base–MIX 익명 검토표 각16행을 실제 생성하고 빈 사람 점수와 hash를 확인했다. 동일16문항을 공유하며 human_review=not_performed다.
- 완료 학습·재로드8개, 평가24개(VQA20+text4), 검토표2개의 보고서 검증 오류0. Qwen3-4B Base 평가가 시작됐으며 나머지5개 텍스트 조건은 아직 완료되지 않았다.

## 2026-10-27 02:04 KST — 한국어 KO/MIX 학습·재로드 완료

- Qwen3-1.7B/4B 각각 KO와 MIX를 원본 Base에서 별도로 학습했다. 네 smoke20update와 네 full one-pass의 adapter 갱신·새 process reload가 모두 성공했다(8/8). KO 각각4,000samples/250updates/243,394target tokens, MIX 각각8,000samples/500updates/1,137,306target tokens.
- validation 선택은 Qwen3-1.7B KO250/MIX500, Qwen3-4B KO200/MIX500이다. Qwen3-4B KO의 선택 checkpoint 노출은3,200samples/196,330target tokens이며 전체 실행량과 구분한다. test로 checkpoint를 선택하지 않았다.
- 9개 공통 텍스트 조건의 실제 평가가 시작됐다. 각 조건4,530문항(KorMedMCQA3,009/AI-Hub957/PubMedQA500/MedQuAD64). 실행 시점에는 평가와 익명 검토표 준비가 아직 완료되지 않았다.
- CPU 전체 회귀검사 재실행 종료코드0. 완료된7개 학습·재로드와20개 VQA 평가에 대한 보고서 검증 오류0(마지막 학습 완료 전 집계). 최종 집계는 텍스트 평가 후 갱신한다.

## 2026-10-26 22:35 KST — VQA v2 완료·한국어 학습 시작

- VQA validation8+본평가/진단12개 전체 성공,20개 산출물 identity/SHA/점수 재계산 검증 통과. SLAKE1061개 normalized EM: Qwen3.5 Base52.40%→adapter70.50%, MedGemma36.38%→69.37%. VQA-RAD451개는47.89%→48.78%,43.24%→48.34%다. 문자열 점수이며 임상 정확도가 아니다.
- 구현 ec0d1433f526f4b9d896e0c4f79fc5db8c024a13 push와 CI40327855285 success 확인. 원문·가중치·인증값은 Git에 포함하지 않았다.
- VQA가 GPU를 반환한 뒤13:32 UTC에 텍스트 worker1502009 시작. Qwen3-1.7B KO smoke 실제20update/320samples/16245supervised tokens, adapter_changed=true와 fresh-process reload 성공. 원본 Base에서 별도 KO full250update 실행 중. 나머지 셀/공통 텍스트 평가/익명 검토표는 아직 미완료다.
- 데이터 감사 함수 source snapshot과 재현 절차를 문서화했고 완료된 aggregate만 사용하는 그림을 추가했다.

- 2026-10-26 22:00 KST 구현 체크포인트: AIHub71875 실제17,280 QA→17,210 canonical, Kor4068 전부 보호. 고정multilingual E5 .92/top5의3,396후보는확정중복이아니며train3010/dev198격리,test188flag. KO4000=1500/1300/1200, MIX8000=같은KO+EN4000, 공통dev1516 동결. 실제token243394/1137306, 250/500updates 선언; 새학습은아직GPU대기. 원래v1 보고서18개SHA불변. VQA validation8개완료후256-token프로토콜고정, Qwen3.5 SLAKE1061 Base556→Tuned748 lexical EM 및 no-image완료, MedGemma평가진행중. 전체입력4530개CPU native인코딩검사오버플로0. 실제실행·예측은Git밖에보존하며최종결과로과장하지않음.

- 2026-10-23: 사용자 pack 전체 계약을 읽음. 현재 repository SHA는 snapshot과 일치. initial unit tests 8/8 (Python3.12). 새 branch 생성.
- sandbox DNS 제한은 승인된 network escalation으로 해결, 저장소 clone 성공. 시스템 변경 없이 CUDA 직접 초기화 성공; NVML mismatch를 GPU 부재로 단정하지 않도록 조사 수정.
- 실제 원본과 실행 artifact는 Git 외부 data root, feature branch에는 코드/설정/비식별 요약만 유지. 사용자 승인에 따라 독립 source와 CPU 구현 분업 시작.
- 2026-10-23 17:38 KST: 사용자 새 Hugging Face 계정 연결 지시에 따라 gated 접근 재확인. 프로젝트 HF_HOME에는 인증이 없어401이었으나 기본 로그인 캐시의 기존 인증을 사용하자 MedGemma1.5-4B config 접근 성공. Gemma3-4B는403 유지. 인증값은 출력·복사하지 않음. MedGemma를 본 VLM 셀로 복귀하고 SmolVLM은 실패 시 fallback으로 보존. 기존 freeze/평가 샘플 정책은 변경 없음. MedGemma 고정revision 다운로드 시작(maxworkers2).
- 2026-10-23 17:46 KST: GPU 학습 PID를 그대로 유지하고 자식 없는 완료 대기 wrapper만 재시작. 새 wrapper는 validation generation→test protocol freeze→paired test→유한한 추가 진단을 실행한다. Pre-training validation loss는 학습 전에 측정했고, unadapted base의 validation generation은 학습 종료 후 동일 고정 base를 새로 로드해 측정한다. 이 시간 순서 차이를 pre-training generation으로 표기하지 않는다.
- core Python3.10 검사72개+subtest3 통과(native GPU/LoRA 테스트 opt-in1 제외). 평가 입력·부모모델 해시·no-image·전문가 빈 검토표 focused 검사48개 통과. CPU synthetic LoRA는 실제 의료 성능 근거와 분리.
- 2026-10-23 17:54 KST: 사용자 Gemma3 추가 승인 후 같은 계정으로 config 접근 및 Gemma3Processor text/multiturn/padding 검사 성공. 과거401/403 attempt는 별도 보존. Gemma3도 접근 차단 해소. MedGemma→Gemma3→Smol 순서로 실패 시에만 fallback하며 본 목표는 text2+VLM2 유지. Gemma3 전체 가중치는 필요 시 다운로드한다.

- 2026-10-23 18:16 KST: Qwen3-1.7B 본 학습525 update/8,400 sample pass/1,983,789 target token과 fresh-process reload 완료. 동일 frozen test820문항 base/tuned 각각 실제 완료. PubMedQA209/500→372/500, KorMedMCQA56/256→102/256. 한국어 유효응답143/256→255/256이므로 전체 개선에는 형식 준수 개선도 포함한다. 단일 seed 예비 결과이며 test에 맞춰 parser/문항/설정을 변경하지 않았다. Qwen3-4B smoke20 update와 재로드 후 본 학습 시작.
- feature branch 최초 구현 commit cff65471c426856007e58adf62c8078101a63441의 remote push를 ls-remote로 확인했고 GitHub Actions Test run39929597344가 성공했다. 원본·모델·raw 예측·인증값은 commit하지 않았다.

- 2026-10-23 20:29 KST: Qwen3-4B 본 학습525 update/8,400 sample pass/1,983,789 target token(6,667.135초)과 fresh-process reload 완료. 시험820문항씩 실제 완료. PubMedQA177/500→384/500(유효459→500), KorMedMCQA102/256→141/256(유효208→255). 두 source 모두 출력 형식 개선도 포함한다. 모델별 기존 test protocol은 변경하지 않았다.
- 2026-10-23 20:31 KST: Qwen3.5-2B 실제 이미지 smoke20 update와 adapter 변화·fresh-process reload 통과. 최적화 causal_conv1d/flash-linear-attention 패키지가 없어 설치된 Transformers의 reference PyTorch kernel을 사용했다. 이 실행 환경을 유지하고 본 학습 시작.
- 2026-10-23 21:07 KST: Qwen3.5-2B SLAKE1,979건 전체1epoch/124update/7,693target token 학습(2,131.828초) 및 새 PID 재로드 추론 성공. validation loss는 초기1.655138→step100 0.337728→step124 0.342843으로 최소값인step100을 선택했다. test_used=false이며 실제 test 결과는 아직 미측정. peak CUDA allocated7,582,394,368/reserved11,655,970,816bytes. 선택adapter digest6e7eb2fe1e9f88c95d1ab7fe416ea805dfdeb160fd4f7854942ad13c9724d5be.
- 2026-10-23 21:12 KST: Qwen3.5-2B SLAKE 고정256건 base/tuned 완료, lexical EM88/256→185/256(34.375%→72.265625%), token F1 0.386867→0.765309. 양쪽 유효256/기술실패0, 생성상한 잘림35→0. 관측이미지69그룹 bootstrap2000회 delta37.890625%p, CI30.8~44.978166%p. CLOSED102건46→85, OPEN154건42→100. 보고 지표는 임상 정확도가 아닌 문자열 일치이며 단일seed와 기반모델 과거노출 한계를 유지한다. 이미지 제외 및 외부VQA-RAD 진단 시작.
- 2026-10-23 21:18 KST: Qwen3.5 no-image32건씩 성공(이미지 포함/제외 EM:base11/0,tuned24/11). VQA-RAD 첫 외부 protocol freeze는 provenance 영상hash 누락으로 QA JSONhash와 JPEG를 비교해 실패했다. 원본316파일·451행 영상hash 모두 보존됨을 확인했고 source generator에 명시적 image_sha256만 추가했다. transformation version v2, 새 aggregate1562258c4b5b6f3b4c5c8948851fbb972da603cd63bff528aa7d278cf2253c86. 이전 aggregate/snapshot과 실패진단을 보존했고 질문·정답·ID·그룹·SFT/core bytes는 동일하다. 영상hash 검증을 우회하거나 학습 renderer를 변경하지 않았다. 재감사critical0, 외부cap64 protocol 최초freeze 성공. MedGemma가 GPU를 반환한 뒤 Qwen3.5 외부진단만 재개할 예정이다.
- 2026-10-23 21:20 KST: MedGemma 실제 smoke20 update·adapter변화·새 PID reload 통과, 본SLAKE 학습 시작. VQA-RAD 수정 후 core Python3.10 139passed/6skipped/6subtests passed; Pillow deprecation6warnings. 검사로그SHA와 실제 CI39932678838 성공 근거를 verification.json에 보존했다.
- 2026-10-23 22:29 KST: MedGemma1.5-4B SLAKE1,979건 전체1epoch/124update/7,252target token 학습(4,308.618초)과 새 PID1007831 재로드 추론 성공(학습PID1002834). validation 기준step124 선택, peak CUDA allocated7,894,200,832/reserved8,264,876,032bytes. 텍스트2종+VLM2종 실제 학습·재로드 목표 충족. MedGemma validation generation→고정SLAKE test→no-image/VQA-RAD 순으로 평가를 계속한다.

- 2026-10-23 22:42 KST: MedGemma SLAKE256건 base/tuned 완료, lexical EM0/256→179/256, F1 0.0375465→0.734003. 이미지 입력512건·native 파일·prompt token hash·선정 adapter를 확인했다. 기본 응답 중앙값14단어와 정답1단어의 불일치,72건 생성상한 도달이 있어 EM0을 임상 정확도로 해석하지 않는다. 사후 형식 진단은 공개 집계에만 기록했고 평가 설정/parser는 변경하지 않았다. no-image32건씩 완료(이미지 포함/제외 EM:base0/0,tuned25/12). VQA-RAD64건 base0→tuned30, 기술실패0.
- 2026-10-23 22:45 KST: 수정된 VQA-RAD provenance로 Qwen3.5 외부 진단 재개 완료. 기존 학습·재로드·SLAKE 평가를 검증 후 재사용했으며 재학습하지 않았다. 외부64건 lexical EM19→31, F1 0.354450→0.526711, 기술실패0, delta18.75%p(관측53그룹 CI9.23~29.51%p). 첫 실패와 이전 freeze는 보존했다. 전체 네 모델의 주 paired4개 및 외부 paired2개, no-image 진단4개 완료. GPU 예산20,047.434초(5.5687h), active 예약과 campaign PID 없음.
- 최종 읽기 전용 artifact 검증: 주/외부 예측 총4,560건의 파일SHA·protocol sample ID·pair별 input/prompt token·native base/precision/generation·선정adapter 연결 통과. MedGemma 형식 진단은 재계산해 일치 확인. 예비 모델4종은 CPU 준비 완료/FALLBACK_NOT_NEEDED, Jetson·AIHub·MedQA/MedMCQA는 별도 차단, 전문가 검토/추가seed는 미실행. 기존 f4462de commit의 remote push 및 CI39957026311 success 확인. 최종 전달 commit/push/CI는 별도 receipt에 기록한다.


## 2026-10-26 후속 작업 — 진행 중

- 사용자 후속 지시로 전체 24시간/모델별 6시간 및 run 6시간/campaign 36시간 제한을 해제하는 unlimited/null 정책을 도입했다. 사용량 원장과 model/run/phase/session·checkpoint floor는 보존한다. 이전 v1 config·checkpoint·freeze·실적을 소급 변경하지 않는다. 새 행렬은 유한한 corpus/seed/학습량으로 선언한다.
- Jetson은 사용자 요청에 따라 `USER_DEFERRED`로 범위에서 제외했다. 과거 호스트키 미검증 차단은 역사 기록으로 남기며 현재 접속·확인·변환·장치 추론은 하지 않는다.
- AI-Hub 71875의 실제 canonical 17,210개(train 15,297 / validation 956 / test 957)를 준비했다. 전달 경로는 사용자 제공 원본이며 API 키를 추가 선행 조건으로 요구하지 않는다. 71762 약 4.25GB 파일은 connector 파일당256MiB 제한과 브라우저 `ERR_BLOCKED_BY_CLIENT`에 막혔다. 경계를 우회하지 않았고 접근 가능한 로컬 원본 경로 문의가 대기 중이다.
- MedMCQA 공식 팀 댓글의 데이터셋 MIT 적용과 고정 LICENSE/README/loader를 확인했다. 원저자 ZIP55,285,460bytes에서193,155행을 실제 확보·감사했다. optional train20,000/validation2,816/test0을 별도 저장했고 KO/MIX corpus에는 넣지 않았다. 공식 test는 정답이 없어 채점하지 않는다. MedQA QA/교재 분리 조사와 AI-Hub 조건·schema 근거는 `docs/research/v2/DATA_RIGHTS_RECHECK_KO.md`에 기록했다.
- v1 예측10개 실행을 재사용해 파일SHA·동결protocol·모델/config/adapter identity를 검증하고 한국어 유효성 전환, MedGemma 길이/구절/잘림, VQA-RAD OPEN/CLOSED를 재분석했다. 두 텍스트 모델의 8,450→8,400은 MedQuAD50개 context overflow 제외이며 나머지는525updates로 모두 처리됨을 preflight/checkpoint/event로 대조했다. 재분석/MedMCQA focused5tests 통과; 새 학습·추론 없이 `reports/public_summary/v2/`에 별도 보고했다.
- KO-v2 4,000개/MIX-v2 8,000개 한국어 텍스트 실험은 계획 단계이며 이 기록 시점 새 학습·재로드·평가는 미실행이다. Qwen3.5-2B/MedGemma의 기존 weight/선정adapter를 재사용하는 VQA v2 validation이 실행 중이다. 상태는 `$MEDVLM_DATA_ROOT/v2/vqa/state.json`, 진입점은 `scripts/run_vqa_v2.py --data-root ...` 및 `--validation-only`다.
- 인계/현재 맥락 문서를 후속 범위로 갱신하고 원래 v1 문서를 역사 구간에 보존했다. `runtime/medvlm_state.json`은 v1 역사 상태이며 후속 전체 root 최종 상태·최종 실적은 아직 없다. `report_v1.py`는 향후 생성 시 과거24/6 제약과 `resolve_resource_policy()`의 현재 정책·기록 사용량을 구분하도록 변경했다. 기존 생성된 v1 보고서는 재생성하지 않았다. 최종 실행·검증·commit/push 실적은 후속 통합 단계에서 추가한다.
