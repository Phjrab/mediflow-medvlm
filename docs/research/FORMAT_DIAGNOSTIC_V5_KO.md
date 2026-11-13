# 71762 후속 형식 진단 — v5

2026-11-13 사용자의 다음 단계 진행 요청에 따라, 기존 test 점수를 보존한 검증셋 전용 진단을 진행한다. v4의 두 모델에서 객관식·분류형4,286개/모델이 모두 형식 미준수였다. 기술 실패나 임상 능력0으로 해석하지 않는다.

실행 전에 private `v5/plan.json`에 유한 행렬·코드·입력·adapter hash를 고정했다. Qwen3-1.7B/4B × 원본 Base/v4 Tuned × 기존 지시/한국어 번역 지시 ×16/256 생성 token ×24검증문항 =384생성 기록이다. 세 source(KorMedMCQA, AIHub71875 객관식, PubMedQA) 각8개이며 KorMed는 직군별2개다. seed20261023의 sample-ID hash 순서로 선택하고 기존 test ID·동일 입력을 제외했다. 두 tokenizer에서48개 prompt씩 입력 길이를 확인했으며 최대562token이었다. 선택 입력과 v4 train의 정확 input hash 겹침은0이다.

입력 질문·선택지·context를 유지하고 마지막 출력 형식 지시문만 의미가 대응하는 한국어로 번역한다. 정답·source 태그·metadata는 predictor에 전달하지 않는다. native template, NF4/BF16, greedy 및 thinking=false는 기존 경로를 사용한다. strict parser는 수정하지 않는다. 16/256 pair의 native prompt와 생성 token prefix를 대조하고, Base/Tuned의 native 가중치·template·dtype·prompt 일치를 확인한다.

새 학습·checkpoint 선택·test 추론·기존 점수 보정은 하지 않는다. 기존 test 출력의 스타일·길이·형식 전환은 관측 후 기술 통계이며, 느슨한 정답 추출을 새 정확도로 보고하지 않는다. 검증24문항은 작고 기존에 이용 가능한 자료이므로 blind test·일반화 성능·임상 검증이 아니다. 번역의 효과와 의료 추론 능력, 학습 과제·노출량·무작위 QA 연결의 인과 효과를 혼동하지 않는다.

GPU worker는 하나씩 실행하며 기존 전역 GPU lock과 unlimited 정책·사용량 원장을 유지한다. 성공한 worker는 source/input/adapter identity 및 prediction hash 확인 후 재사용하고, 한 조건이 막혀도 다른 조건을 계속한다. 기존 v4와 모든 공개 보고서의 hash를 실행 전 보관하고 완료 시 대조한다. 현재 실행 상태는 private `v5/state.json`, 실제 예측과 로그는 `v5/runs/`다.

```bash
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_DISABLED=true
../.venv-medvlm/bin/python scripts/run_format_diagnostic_v5.py --data-root ../medvlm-data
```

공개 요약은 완료 후 `reports/public_summary/v5/`에 저장한다. 원본·전체 예측·가중치·비공개 A/B 키는 Git에 올리지 않는다. 사람의 임상 검토는 미실행, Jetson은 사용자 보류다.
