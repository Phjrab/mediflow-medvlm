# V6/V7 R2 감독 인계

2026-11-14 사용자 채택 지시: `docs/research/V6_V7_R2_INSTRUCTIONS.txt` 전체 및 같은 날 모델·추론 강도 자율 위임. 기존 v1~v5와 2026-11-13 Release는 보존한다. 이 문서는 초기 설정 인계이며 V6/V7 완료 보고가 아니다.

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

상태 수집은 local Python, 필요 시 Luna low/medium. 일반 구현은 Sol medium, 학습 pipeline은 Sol high. 아키텍처·누수·반복 오류는 Astra medium/high, 복잡한 종합 설계는 Astra high/xhigh. 현재 채팅 모델을 바꿨다고 주장하지 않는다. 실제 subagent 호출 모델/effort와 결과는 `reports/public_summary/v6/orchestration_log.json`에 누적한다. 네이티브 heartbeat의 모델/effort override는 현재 도구가 노출하지 않는다.

## 종료 조건과 공개 범위

GPU 시간 상한 5개는 모두 null/unlimited다. 유한 모델·데이터·seed·1 pass 범위만 제한한다. Jetson은 `docs/research/JETSON_DEPLOYMENT_VALIDATION_PLAN_KO.md` 계획만 있으며 접속·실행 금지다. 사람 검토는 실제 없으면0건이다.

코드·집계 보고서만 기존 feature branch에 commit/push하며 main 병합은 하지 않는다. 원본 데이터·새 V6/V7 가중치·optimizer/RNG·비공개 예측을 공개하지 않는다. 11/13 승인 Release는 보존하며 새 가중치 공개 승인으로 확대하지 않는다.

모든 선언 셀의 실제 terminal 사유, V6/V7 최종 보고서·재현 명령·보존 검증 및 최종 HEAD의 실제 GitHub CI가 있어야 `finalize --receipt ...`를 호출한다. `ready_to_stop_scheduler=true`를 확인한 뒤 **이 campaign의** 네이티브 예약만 PAUSED로 바꾸고 실제 상태를 재확인한다. 서비스는 worker가 없음을 확인한 뒤 정리한다. 미실행·차단·실제 수행을 최종 보고에서 분리한다.
