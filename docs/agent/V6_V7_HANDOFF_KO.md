# V6/V7 R2 감독 인계

2026-11-14 사용자 채택 지시: `docs/research/V6_V7_R2_INSTRUCTIONS.txt` 전체 및 같은 날 모델·추론 강도 자율 위임. 기존 v1~v5와 2026-11-13 Release는 보존한다. 이 문서는 초기 설정 인계이며 V6/V7 완료 보고가 아니다.

## 최신 등록 — 2026-11-14 18:36 KST

고유8개 모델의 준비가 모두 성공했다. V1/V4는 각각 시도3회·실패2회·복구2회를 그대로 유지한다. 이번 18:19 예약의 tick은 한 번뿐이며 V4 성공을 검증한 뒤 worker가 없는 미구현 gate에서 다음 코드를 구현했다. 새 작업을 이 턴에 실행하지 않았다.

`refine-preflight`로 미시작 gate를 세분화했다. 활성 manifest hash는 `5c6d818fdf9a43d2fe4c74ed80f499986d8098c272126949a9f62895e7e9746c`, 새 source commit은 `2ff65d4d6435e14f0c5052044265308c399bf147`, source hash는 `19acab068e17abef54e9983d30d7b3e562e61b051f43f67d4fb91f8bb1146054`다. 기존23개 state record와 기존 파일131개의 bytes를 보존했다. 전체43 jobs 중 새 CPU10개는 `PENDING`, 새 GPU smoke10개와 aggregate gate는 `IMPLEMENTATION_PENDING`이다. 고유8모델/텍스트6·VLM4, 본학습10+추가최대8, V7별도4 범위는 변하지 않았다.

다음 단일 tick의 대상은 `v6-native-preflight-T1`이다. 기존 OS coordinator를 교체하지 않았다. 각 새 job은 위 immutable snapshot의 `campaign_preflight`를 사용하고, worker/queue/복구 횟수는 같은 supervisor가 소유한다. 등록을 반복하거나 별도 worker를 시작하지 말고 아래의 기존 점검 명령을 한 번 사용한다.

CPU worker는 기존 train 텍스트8,000개 또는 SLAKE1,979개 전체를 native processor로 검사하며 정답을 자르지 않는다. template 날짜는 `2026-11-14`, context 적격성은2048로 고정한다. 파일·모델·소스·환경 version을 확인하고 적격/제외 IDs와 source/task/label별 이유를 private에 저장한다. 알려지지 않은 encoder 오류·파일 변조·mask 위반은 셀 실패다. 완료 directory marker와 artifact SHA가 맞으면 receipt 작성 직전 중단도 동일 identity로 복구한다. CPU/meta 성공은 weights/CUDA/gradient/update/reload/본학습 성공을 뜻하지 않는다.

실제 native8모델 meta 구조·template와 SmolLM3 고정 날짜20사례를 검증했다. native contract12검사, CPU worker28검사, supervisor refinement89검사를 통과했고, 일반 wheel 설치 전체495passed/14optional-skipped/30subtests를 확인했다. Astra/high의 최종 검토와 source bytes가 일치한다. private 근거는 workspace `outputs/v6_v7/preflight/`의 `native_contracts_cpu_20261114.json`, `astra_review_20261114.json`, `noneditable-test-receipt.json`, `deployment-preflight.json`, `registration-receipt.json`이다. 공개 집계는 `reports/public_summary/v6/native_preflight_implementation.json`이다.

**다음 미시작 구현:** CPU jobs 이후 GPU smoke10개의 실행부를 개별 활성화한다. 새 strict projection 경로와 frozen native template를 실제 quantized loader에 적용하고, 모든 계열의 finite gradient·adapter 변화·20 updates·fresh reload를 검증해야 한다. 지금 역사 training loader/runner는 바꾸지 않았다. 모든 native eligibility와 지원 가능한 smoke 결과 뒤 공통 train을 동결한다. 기존 `v6-preflight_smoke` aggregate는 자식 결과를 검증하는 미구현 gate이며, CPU 작업 성공만으로 자동 성공시키지 않는다. first-seed/final checkpoint/평가/추가seed/V7은 여전히 미시작이다.

20분 네이티브 예약은 ACTIVE이며 등록 확인 당시 다음 시각은18:39:58 KST였다. 최신 시간은 scheduler receipt를 우선한다. 기존 전달 HEAD780cfc2의 실제 CI43576060020은 성공했다. 새 전달 HEAD CI는 `outputs/v6_v7/git_receipt.json`에서 별도 확인한다.

## 실제 경로와 단일 소유자

- 작업 저장소: `/home/chosun/Documents/Codex/2026-10-23/01-master-prompt-md-03-09/work/mediflow-medvlm`
- private data root: `/home/chosun/Documents/Codex/2026-10-23/01-master-prompt-md-03-09/work/medvlm-data`
- 감독 상태: data root의 `v6_v7/{manifest.json,state.json}`. 활성화 뒤에는 `state.manifest_path`가 가리키는 versioned manifest를 사용하며 최초 manifest.json은 보존한다. queue·다음 단계·누적 실패 횟수는 `campaign_supervisor`만 변경한다.
- 네이티브 예약 ID: `medvlm-v6-v7`, 현재 채팅 ID `01a167b3-5c1b-7d52-8c69-d59fda6ba373`. 20분 간격의 실제 활성화·다음 시각은 `outputs/v6_v7/scheduler_receipt.json` 및 앱 저장 상태로 확인한다.
- OS 서비스: `medvlm-v6-v7-supervisor.service`. 독립 worker를 시작하는 oneshot이며 예약 타이머가 아니다. `KillMode=process`로 launcher 종료가 정상 worker를 종료하지 않게 한다. CPU 테스트에서도 launcher 종료 후 worker 생존을 검증한다.
- worker의 실제 source snapshot, Python, argv, cwd, config/data/model/source hash는 각 job 및 `attempts/<job>/<attempt>/launch.json`에 저장된다. 실행 중 source·config·환경은 수정하지 않는다.

호스트에서 다음 명령을 한 번 호출한다. 기본 shell sandbox에서는 호스트 PID·systemd bus가 보이지 않을 수 있으므로 필요한 실행 권한으로 실제 호스트에서 확인한다.

```bash
systemctl --user start medvlm-v6-v7-supervisor.service
```

`outputs/v6_v7/deployment_receipt.json`의 `status_argv`는 읽기 전용 상태 조회, `tick_argv`는 같은 supervisor의 직접 재개 명령이다. 두 경로가 같은 lock을 사용한다. `state.json` 직접 편집, 별도 학습 launcher, 예약별 새 queue 생성은 금지한다. 상태·PID만으로 성공을 주장하지 않고 artifact receipt SHA까지 확인한다.

## 이번 초기 실행과 다음 미시작 범위

초기 GPU job은 `v6-engineering-t1-smoke-20261114`다. 기존 MIX8000 중 seed 고정 320개/validation8, 20 updates, accumulation16, 같은 Qwen3-1.7B base로 기존 엔진의 독립 실행·저장·새 process reload를 확인한다. 신규 run/config를 사용하며 기존 결과를 덮어쓰지 않는다. **V6 공통 train 동결·본학습 또는 전체 10개 native smoke 통과로 세지 않는다.** 성공 receipt에는 실제 step20·320표본·제외0·adapter 변화·final reload가 필요하다.

그 뒤 고유 8개 모델의 revision·권한·native tokenizer/processor·파일 hash를 준비한다. 한 모델의 실패가 다른 준비 job을 막지 않는다. 실패 모델을 성공 모델로 바꾸거나 모델 수를 줄여 완료했다고 쓰지 않는다.

나머지는 `IMPLEMENTATION_PENDING` gate로 명시돼 있다. 예약 점검은 정상 worker가 있으면 한 번 확인 후 끝낸다. worker가 없고 미구현 gate에 도달하면 다음 미시작 범위만 구현·검증한 후 새 source snapshot으로 등록한다. 구체적 우선순위는 `docs/research/V6_V7_ARCHITECTURE_AUDIT_KO.md`다.

1. Phi fused projection, Qwen3.5 linear attention, native multimodal 경로, SmolLM3 날짜·no-thinking 결정성, 전체 정답 mask와 gradient/adapter 검증.
2. 보호셋·group 경계 확인 및 모든 지원 모델의 2048 input+target 적격 교집합. 본학습 중 truncation/추가 제외 금지. 각 native 경로의 20-update smoke.
3. 공통 train·recipe 동결 → 10개 first-seed 본학습 → validation cap 보정 한 번 → 전체 공통 dev → 서로 다른 계열 2개씩 selection 동결 → 추가 seed 최대8조건 → test·보고. 검증 전에 test를 조회해 selection하지 않는다.
4. final checkpoint·실제 fresh final reload, full-token likelihood scorer, 실패 포함 예정 분모, paired/group bootstrap 및 resource 기록을 구현한다. 기존 minimum-validation-loss selection을 새 final 선택으로 재해석하지 않는다.
5. V6 terminal 후 V7 P0/P1/R0/R1 독립 base 학습·평가. core/M 동일, replay20%, N 공식 및 최소800, 기존 v4 random 연결 고정, 71762 원본 parent unknown·개별 QA 임상 정합성 미검증 유지.

`activate --manifest <새 manifest>`는 미시작 pending job의 실행 부분만 허용한다. job/cell/run/phase/dependencies 및 이미 시작된 identity를 보존하며 revision audit를 남긴다. 세부 실험은 기존 선언된 8모델/10셀, V6 최대18본학습, V7 최대4조건 안에서만 추가한다. 미구현을 FAILED나 완료로 바꿔 예약을 종료하면 안 된다. 현재 미래 job은 단계별 placeholder이므로 실제 10셀/추가8조건을 등록하기 전 supervisor가 조건별 queue와 복구 횟수까지 소유하도록 미시작 gate의 세분화 등록을 구현·검증한다. 단일 단계 worker 안에 별도 queue owner를 만들지 않는다.

## 복구 및 추론 모델

최초 시도+기술 복구 최대2회는 영속 state에 누적한다. PID뿐 아니라 boot ID와 process start ticks를 검사한다. 늦은 heartbeat만으로 살아 있는 worker를 종료하지 않는다. 완료되지 않은 checkpoint·바뀐 identity에서 resume하지 않는다.

2회 실패 후에는 `ANALYSIS_REQUIRED`로 보류하고 실제 Astra 분석 후 `authorize-recovery --job-id ... --analysis-receipt ...`로 남은 1회만 허용한다. 분석 receipt는 실제 호출 모델·강도·원인·결과·누적 실패 수와 근거 artifact SHA를 기록한다. 실패 횟수를 초기화하지 않는다.

### 2026-11-14 Qwen3.5 준비 복구 인계

`prepare-V1`의 두 worker exit는 모두 실패다. Hub 접근·고정 revision 확인은 성공했지만 `model.safetensors-00001-of-00001.safetensors`를 기존 파일명 allowlist가 인식하지 못했다. 기존 failure의 `resolve_pin` 단계명은 실제 실패 위치보다 앞선 값이다. Astra medium 분석과 로컬 CPU native config/tokenizer/processor 확인을 완료했다. 6개 성공 준비 결과, 모델 pin, 기존 snapshot·attempt는 보존한다.

수정 코드와 검증 결과는 Git 이력에 보존하고, 실제 등록용 후보는 작업 폴더의 `outputs/v6_v7/recovery/prepared_recovery.json`에 저장한다. 이 후보 파일의 존재는 등록·복구 성공을 뜻하지 않는다. 이번 점검에서는 추가 tick이나 마지막 시도를 실행하지 않았다. 다음 예약의 **단일 tick**이 두 번째 실패를 영속 상태에 반영하면 V1은 `ANALYSIS_REQUIRED`가 되고 나머지 준비 job은 계속 진행할 수 있다. 정상 worker가 있으면 통상 규칙대로 해당 점검을 끝낸다.

worker가 없는 점검에서 V1의 `failure_count=2`, `attempts=2`, `ANALYSIS_REQUIRED` 및 후보의 원본 job/manifest hash가 현재 상태와 일치하는지 확인한다. 후보에 기록된 수정 snapshot의 supervisor로 `authorize-recovery --job-id prepare-V1 --analysis-receipt <검증된 분석 receipt> --manifest <수정 manifest>`를 실행한다. 이 연산은 해당 실패 job의 `cwd/source_files/source_hash`만 변경하며 나머지 job, argv, config/data/model identity와 누적 횟수를 보존한다. 이전 manifest와 실행 snapshot을 수정하지 않는다. 기존 OS launcher는 versioned manifest를 읽으므로 서비스 교체는 필요 없다. 이 등록 자체는 worker를 실행하지 않으며 다음 예약의 tick이 남은 마지막 시도를 시작한다. 후보·분석이 변경됐거나 검증되지 않았으면 등록하지 않는다.

**2026-11-14 08:28 UTC 갱신:** V1 수정 snapshot은 08:21 UTC에 supervisor API로 등록됐다. 현재 versioned manifest는 `f5025fc060467924da8ceafb9874cec6a1d402f08d732a43f36d7b977dd6eacb`, V1 source commit은 `f45378c847498682132de00300f367dedd480fb4`다. `outputs/v6_v7/recovery/prepare-V1-registration-receipt.json`과 `prepared_recovery.json`은 `REGISTERED_NOT_STARTED`를 기록한다. 위 준비 절차를 다시 실행하지 말고 현재 state와 등록 receipt를 우선 확인한다. 두 기존 시도·실패는 유지됐고 마지막 시도는 다음 단일 tick의 대상이다.

V4도 두 번의 `native_assets` ImportError 뒤 보류됐다. Astra medium은 SmolVLMProcessor가 필수로 요구하는 `num2words` 누락을 확인했다. worker가 없는 상태에서 공식 PyPI SHA256과 대조한 `num2words==0.5.14`, `docopt==0.6.2`만 추가했고 기존 배포판은 변경·삭제하지 않았다. 전체 환경 전후 목록, 설치 artifact hash, native processor·합성 이미지 CPU 검증, V1 native 재확인은 `outputs/v6_v7/recovery/v4/`에 있다. 기존 모델·source snapshot·argv·config/data identity는 유지한다. 저장소 train extra와 training lock은 재현을 위해 추가 의존성을 선언한다.

V4 최종 Astra 검토 후 08:27 UTC에 남은 1회 복구를 supervisor API로 등록했다. `outputs/v6_v7/recovery/v4/registration-receipt.json`을 확인한다. V1/V4 모두 `PENDING`, `attempts=2`, `failure_count=2`이며 실제 세 번째 실행은 아직 시작하지 않았다. 이번 점검에서 tick은 한 번뿐이며 별도 launcher는 없다. 기존 queue 순서대로 V1 다음 V4가 진행된다. 후속 정상 worker는 한 번 확인하고 종료한다. 이 환경 의존성 수정과 native CPU 검증을 본학습·GPU smoke·full weights 검증 성공으로 세지 않는다.

상태 수집은 local Python, 필요 시 Luna low/medium. 일반 구현은 Sol medium, 학습 pipeline은 Sol high. 아키텍처·누수·반복 오류는 Astra medium/high, 복잡한 종합 설계는 Astra high/xhigh. 현재 채팅 모델을 바꿨다고 주장하지 않는다. 실제 subagent 호출 모델/effort와 결과는 `reports/public_summary/v6/orchestration_log.json`에 누적한다. 네이티브 heartbeat의 모델/effort override는 현재 도구가 노출하지 않는다.

## 종료 조건과 공개 범위

GPU 시간 상한 5개는 모두 null/unlimited다. 유한 모델·데이터·seed·1 pass 범위만 제한한다. Jetson은 `docs/research/JETSON_DEPLOYMENT_VALIDATION_PLAN_KO.md` 계획만 있으며 접속·실행 금지다. 사람 검토는 실제 없으면0건이다.

코드·집계 보고서만 기존 feature branch에 commit/push하며 main 병합은 하지 않는다. 원본 데이터·새 V6/V7 가중치·optimizer/RNG·비공개 예측을 공개하지 않는다. 11/13 승인 Release는 보존하며 새 가중치 공개 승인으로 확대하지 않는다.

모든 선언 셀의 실제 terminal 사유, V6/V7 최종 보고서·재현 명령·보존 검증 및 최종 HEAD의 실제 GitHub CI가 있어야 `finalize --receipt ...`를 호출한다. `ready_to_stop_scheduler=true`를 확인한 뒤 **이 campaign의** 네이티브 예약만 PAUSED로 바꾸고 실제 상태를 재확인한다. 서비스는 worker가 없음을 확인한 뒤 정리한다. 미실행·차단·실제 수행을 최종 보고에서 분리한다.
