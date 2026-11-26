# V7 foreground terminal handoff

V7의 고정 모델은 `Qwen/Qwen3-1.7B`, 조건은 `P0`, `P1`, `R0`, `R1`이다. 이 클라이언트는 저장된 canonical operator의 `status_argv`와 `tick_argv`를 그대로 사용한다. 별도 worker, queue, 서비스, 예약을 만들지 않는다. 이미 실행 중인 V6 작업은 조회만 한다.

현재 V7 train/evaluation/report worker와 동결된 readiness 증거가 없으므로 V7 실행 준비가 완료된 상태가 아니다. 기존 V7 placeholder나 `IMPLEMENTATION_PENDING`을 완료로 표시하지 않는다. 실제 배포 영수증·handoff·검증된 명령은 TRAINING_OWNER가 별도로 전달하며, 준비가 끝나기 전에는 scope token을 발급하지 않는다.

## 명령

기본 명령은 읽기 전용 `status`이다. `preflight`는 실제 prerequisites와 고정된 파일을 검사하고 미충족 시 종료 코드 2를 반환한다. `watch`는 재접속용 조회이며 dispatch하지 않는다. 준비가 미완료여도 blocker와 현재 V6 진행을 계속 보여준다. `run`만 명시적인 foreground 실행 요청으로 canonical tick을 호출한다.

```text
<reviewed-python> <archive>/scripts/run_campaign_terminal.py \
  --deployment <saved-wrapper-deployment.json> --deployment-sha256 <SHA256> \
  --handoff <handoff.json> --handoff-sha256 <SHA256> preflight
```

위 명령은 경로·hash를 실제 영수증으로 대체해야 하는 형식 예시다. 실행이 준비되면 같은 인자로 `watch` 또는 `run`을 사용한다. `run`은 사용자가 받은 token을 환경 변수 `MEDVLM_V7_SCOPE_TOKEN`으로 명시해야 한다. token은 argv·로그·배포 영수증에 저장하지 않는다. `status`, `preflight`, `watch`는 주변 환경의 token을 전달하지 않는다.

`--interval`은 1~60초, 기본 5초다. `--json`은 의료 원문 없이 scalar 상태를 출력한다. `--max-updates N`으로 foreground 조회 횟수를 제한할 수 있으며, 완료 전에 한도에 도달하면 종료 코드 3을 반환한다.

## 실행과 재접속

`run`은 네 조건의 동결된 train/evaluation job들과 report를 manifest 순서와 native dependency에 따라 하나씩 진행한다. job 선택·상태 변경·worker 실행은 같은 canonical supervisor가 담당한다. 살아 있는 worker가 있으면 상태·진행률만 읽는다. 확인된 exit 뒤의 reconcile/다음 dispatch도 canonical tick에 맡긴다.

Ctrl-C는 클라이언트를 분리하며 worker에 신호를 보내지 않는다. 이미 요청한 canonical tick은 atomic 처리를 마치며 worker를 시작할 수도 있다. 분리 뒤 추가 tick을 요청하지 않는다. `watch`로 dispatch 없이 다시 붙을 수 있고, `run` 재시작은 성공 receipt와 artifact를 검증해 완료된 작업을 중복 실행하지 않는다. 실패·repair 대기·차단 상태는 0이 아닌 코드로 종료한다.

네 조건의 train과 evaluation, 최종 report가 모두 실제 `SUCCEEDED`이고 receipt/artifact 검증을 통과해야만 V7 완료를 표시한다. 등록만 완료된 상태, 학습만 완료된 상태, telemetry 파일만 존재하는 상태는 전체 완료가 아니다.

## 진행률과 Forge

최초 및 dispatch 경계의 canonical 검증은 오래 걸릴 수 있다. 이때 검증 진행 중임을 즉시 표시하고 ETA를 꾸미지 않는다. 정상 worker를 모니터링하는 동안에는 고정 manifest/state, producer progress와 같은 Forge telemetry 파일을 로컬에서 읽어 매번 전체 canonical 검증을 반복하지 않는다.

표시 항목은 step/total, loss, learning rate, gradient norm, 처리량, GPU 사용량·메모리, RAM, ETA다. 값이 없거나 센서를 지원하지 않으면 null과 이유를 표시한다. ETA를 계산할 수 있는 경우 최신 실측 sample rate와 남은 sample 수로 만든 추정치임을 명시한다. GPU 센서를 새로 실행하거나 모델을 로드하지 않는다.

현재 attempt의 worker/child가 종료되고 launch identity와 일치하는 exit 증거가 있으면 `EXITED_AWAITING_VERIFICATION`과 exit code를 표시한다. 원래 `native_state=RUNNING`은 유지되며, exit code 0만으로 성공을 선언하지 않는다. 종료 증거가 없거나 identity가 다르면 `ACTIVITY_UNVERIFIED`로 표시한다. likelihood처럼 학습이 아닌 작업은 identity가 일치하는 progress의 `rows_completed`, `scheduled_text_rows`, `technical_failures`를 표시하며 학습 지표를 만들어 내지 않는다.

`run`에서 살아 있는 worker identity가 없으면 exit 파일이 없어도 canonical tick에 reconcile을 요청한다. launch race와 worker/GPU lease는 기존 supervisor가 판단한다. `watch`는 이 경우에도 tick을 호출하지 않는다.

Telemetry는 native launch identity와 scientific producer의 `attempt.json` UUID를 연결한 경우에만 사용한다. 이전 attempt나 중복된 producer 증거는 현재 실측값으로 표시하지 않는다. LF로 완료된 최신 stream record만 읽고, 의료 질문·정답·생성문·traceback·원시 로그를 출력하지 않는다. 이 화면은 telemetry 원본을 수정하거나 Forge cache를 새로 만들지 않는다.

클라이언트 subprocess 환경은 문서화된 최소 host 환경(`PATH`, `HOME`, 사용자/locale/terminal/runtime 경로)과 저장된 배포의 명시적 환경을 합친다. 주변 `PYTHONPATH`, `PYTHONHOME`, GPU 선택, preload, thread 설정은 저장된 배포에 명시된 경우에만 전달한다. 저장된 cwd와 Forge 경로는 그대로 유지하며, `run`의 명시적 scope token만 추가한다.

## Handoff 검증

`mediflow.v7-terminal-handoff.v1`은 manifest의 raw SHA256와 canonical hash, 고정 모델/네 조건, train/evaluation/report의 정확한 job 목록, per-job config/source/model/data binding, 준비 검증 영수증, prior deployment, guard source, Forge contract와 token hash를 고정한다. guard는 canonical supervisor lock 안에서 같은 조건을 다시 확인한다. 학습·평가 worker가 미구현이거나 V6 report가 검증되지 않았으면 `preflight`와 `run`은 `BLOCKED_PREREQUISITE`로 끝난다.

CPU fixture 테스트는 실제 GPU 실행이나 V7 전체 준비 완료를 뜻하지 않는다.
