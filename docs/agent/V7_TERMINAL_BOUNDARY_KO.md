# V7 수동 실행 경계

V7은 Qwen/Qwen3-1.7B의 P0/P1/R0/R1 네 조건이다. 자동 예약은 V6만 진행하며 V7은 사용자가 터미널 run을 실행할 때만 시작한다. 기존 단일 Supervisor와 물리 supervisor/worker/GPU 잠금·누적 사용 ledger를 그대로 사용한다. 별도 queue·예약·worker launcher는 없다.

`campaign_control_terminal_guard.py`는 SHA로 고정한 이전 deployment/control 객체를 사용한다. 자동 tick 안에서만 원본 manifest를 검증한 뒤 V7을 제외한 메모리상 dispatch eligibility view를 제공한다. 새 manifest를 기록하거나 원본 job/state/history/attempt를 직접 편집하지 않는다. status는 원본 전체 상태를 반환한다. 현재 자동 경계는 기존 다섯 V7 placeholder의 정확한 집합을 기대하며, 향후 V7 세분화 시 새 owner binding과 독립 검토가 필요하다.

터미널 scope는 동일 Supervisor 잠금 안에서 다시 검증한다. V6 terminal 및 성공한 V6 보고서, 전체 V7 train/evaluation/report job 목록, 원본 Base·조건·source/config/data/model hash와 실제 완료 receipt/artifact, 정확한 readiness closure 및 foreground token이 필요하다. token은 미준비 상태에 발급하지 않는다. `status/watch`는 worker를 시작하지 않는다. Ctrl-C는 터미널 모니터만 분리하며 worker를 signal하지 않는다. 실제 재개는 원본 Base에서 새 attempt를 만드는 검증된 복구 경로이며 부분 optimizer/RNG resume를 주장하지 않는다.

2026-11-26 구현 시 V6 final test/report, V7 subset/core/shared MCQ 및 최신 보호셋 audit freeze, R의 전체 target/masking receipt, V7 worker와 native test protocol이 미완료다. 현재 CLI/경계의 CPU 검증은 V7 학습 성공이나 GPU 실행 검증이 아니다. 준비 부족은 nonzero BLOCKED_PREREQUISITE로 반환한다. 기존 미구현 다섯 placeholder를 runnable로 바꾸지 않는다.

Forge와 터미널은 같은 `mediflow.telemetry.v1` attempt를 읽는다. 측정되지 않은 값은 null/이유로 남긴다. ETA는 실측 처리량과 남은 표본 수가 있을 때만 클라이언트 추정값으로 표시한다. 모든 조건은 동일 seed의 실험이며 독립 seed 반복이나 71762 random reference의 임상 정답성을 주장하지 않는다.

검증 명령: `PYTHONPATH=src python -m pytest tests/test_campaign_control_terminal_guard.py tests/test_campaign_terminal.py`
