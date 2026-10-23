# 실제 학습 및 재개 계약

이 저장소는 native Transformers 모델과 PEFT LoRA/QLoRA를 사용한다. Core는 Python 3.10 이상이며 torch를 import하지 않는다. 실제 학습 전용 환경은 별도로 설치한다. 이번 환경의 정확한 패키지 버전과 실제 GPU 결과는 실행 보고서와 각 run manifest를 기준으로 읽는다. 아래 예제 설정이 있다는 사실은 의료 학습 완료를 뜻하지 않는다.

## 입력, 모델과 마스킹

`training.rendering`은 canonical `input.question/context/options/images`만 user prompt에 넣는다. `supervision.answer` 또는 MCQ `option_id`는 마지막 assistant target에만 들어간다. rationale, provenance, source metadata, split 및 gold label은 user prompt에 넣지 않는다. TEXT_QA는 이미지가 필요 없으며 기존 VQA, REPORT, longitudinal 이미지 요구 조건은 그대로 유지한다. TEXT_QA workflow 결과는 answer/status/limitations와 MCQ option_id를 사용한다. 기존 VQA answer/evidence/limitations 계약은 바꾸지 않는다.

`revision`은 immutable Hub commit으로 고정한다. 모델 config의 `architectures`에 해당하는 설치된 native class로 로드하며 원격 코드는 실행하지 않는다. Qwen thinking은 template 인자 `enable_thinking=False`로 끈다. 모델/processor가 반환한 pixel values, image grid 등 추가 tensor를 유지한다.

마스킹은 generation prompt의 실제 token ID가 전체 학습 sequence의 정확한 prefix인지 검증한다. 마지막 assistant 앞은 모두 -100이며 padding은 attention mask로 제외한다. EOS와 PAD가 같아도 실제 답의 EOS는 남긴다. 이미지 특수 token은 해당 model config와 tokenizer의 현재 vocabulary에서 찾는다. 경계 불일치나 shift 이후 target token 0개는 실패한다. context 초과 record는 질문/답을 자르지 않고 sample ID와 제외 사유를 기록한다. VLM은 실제 image hash/order/dimensions 및 생성된 image token 수를 mask audit에 기록한다.

언어 projection의 전체 module 경로를 LoRA 대상으로 고정한다. vision/visual/projector/merger module은 제외하며 모든 base parameter의 freeze flag와 gradient 부재를 검사한다. q_proj/v_proj 이름만 보고 vision encoder를 함께 학습하지 않는다.

## 실행 순서

저장소 루트에서 다음 명령을 실행한다. `DATA`는 Git 밖의 사용 가능한 data root, `PY`는 학습 전용 Python이다. 실제 frozen manifest와 원문 split 파일이 먼저 존재해야 한다.

```bash
export MEDVLM_DATA_ROOT="$(pwd)/../medvlm-data"
export HF_HOME="$MEDVLM_DATA_ROOT/models/hub"
export PYTHONPATH="$PWD/src"
PY="../.venv-medvlm/bin/python"
"$PY" -m mediflow_medvlm.training probe \
  --model-id Qwen/Qwen3-1.7B \
  --cache-dir "$HF_HOME" \
  --tokenizer-check \
  --output "$MEDVLM_DATA_ROOT/manifests/models/qwen3_17b.json"
```

`configs/training.*.example.json`을 data root의 `configs/`에 복사하고 실행 위치에 맞춰 경로를 확정한다. `freeze_manifest`의 `current_freeze.json`은 실행 전에 동일 내용의 `freeze-<digest>.json` 경로로 바꾼다. 이후 같은 run의 config를 수정하지 않는다. 예제의 상대 경로는 저장소 루트에서 실행하는 경우를 기준으로 한다. `--config`가 받은 JSON 자체가 run output에 immutable snapshot으로 저장된다.

```bash
"$PY" -m mediflow_medvlm.training train \
  --config "$MEDVLM_DATA_ROOT/configs/qwen3-17b-smoke.json"

# 위 process가 끝난 뒤 manifest/selection.json에 기록된 정확한 경로 사용
"$PY" -m mediflow_medvlm.training reload \
  --config "$MEDVLM_DATA_ROOT/configs/qwen3-17b-smoke.json" \
  --checkpoint "$MEDVLM_DATA_ROOT/runs/qwen3-17b-smoke/checkpoints/step-000020-EXACT_ID"

# smoke와 별도 config/출력 경로에서 처음부터 본 학습
"$PY" -m mediflow_medvlm.training train \
  --config "$MEDVLM_DATA_ROOT/configs/qwen3-17b-v1.json"
```

실제 smoke 예제는 microbatch=1, accumulation=1, 실제 optimizer update 20회다. 본 학습 예제는 accumulation=16, 최대 1 epoch 또는 1000 optimizer updates다. Smoke는 gradient accumulation microstep을 optimizer step으로 계산하지 않는다. 초과 길이 record를 제외한 뒤 필요한 smoke 표본 수를 채우되 원본 학습 split을 한 번 이상 순회하지 않는다. 충분한 표본이 없으면 조용히 성공으로 처리하지 않는다.

Validation은 seed와 sample ID hash로 정렬한 뒤 source 간 round robin으로 고정 32개를 뽑는다. 두 text source가 충분하면 16+16이며 manifest에 실제 sample IDs와 source counts를 남긴다. 학습 전 validation loss를 기록하고 checkpoint 선택도 validation loss만 사용한다. Test 파일은 열지 않고 frozen manifest의 test hash 참조만 남긴다. loss는 같은 모델 checkpoint 선택용이며 tokenizer가 다른 모델의 성능 순위로 사용하지 않는다.

## 체크포인트, 예산과 결과

완성되지 않은 `.pending-*` 체크포인트는 재개 대상으로 쓰지 않는다. 완성본은 adapter/processor, optimizer/scheduler/scaler, Python·torch·CUDA RNG, 다음 sample 위치, sample ordering hash, config/data/model/template/training-source identity를 저장한다. 실제 학습한 base와 processor 파일의 hash도 run model metadata에 기록한다. 자체 run의 latest/best 이외 recovery checkpoint만 정리하며 다른 run과 원본 데이터는 건드리지 않는다.

```bash
"$PY" -m mediflow_medvlm.training train \
  --config "$MEDVLM_DATA_ROOT/configs/qwen3-17b-v1.json" \
  --resume "$MEDVLM_DATA_ROOT/runs/qwen3-17b-v1/checkpoints/step-NNNNNN-EXACT_ID"
```

재개는 digest를 검증한 이 도구의 로컬 `training_state.pt`만 읽는다. 다른 config/data/base revision/template/target modules/quantization/training code는 strict resume 오류이며 새 run이 필요하다. 외부에서 받은 pickle checkpoint를 이 경로에 넣지 않는다. `selection.json`의 최종 adapter는 새 process의 `reload.json` 성공으로 검증한다. 같은 training PID에서 reload 호출은 거부한다. raw 생성 내용은 restricted local artifact에만 저장한다.

`data_root/runtime/medvlm_gpu.lock`은 OS advisory lock이며 다른 GPU worker를 종료하지 않는다. `gpu_budget.json`은 canonical full model ID별 occupancy seconds를 모든 revision/task/run/retry/probe/train/reload/eval에 합산한다. 기본 상한은 모델당 6시간, 전체 24시간이며 더 작은 config 값도 적용한다. 충돌은 RESOURCE_BUSY, CUDA 부재는 GPU_NOT_AVAILABLE, OOM은 OOM, 예산 종료는 SKIPPED_BUDGET으로 기록한다. GPU가 없을 때 큰 모델 CPU 학습을 자동으로 시작하지 않는다.

`manifest.json`, `events.jsonl`, `mask_audit.jsonl`, `excluded.json`, `best.json`, `latest.json`, `selection.json`, `reload.json`을 확인한다. checkpoint 단계의 finite loss/gradient, nonzero targets와 adapter 전후 digest 변화는 학습 기능 검증이다. 의료 성능은 별도 고정 baseline/tuned 평가를 거쳐야 한다. 합성 fixture 결과는 SYNTHETIC_PIPELINE_ONLY이며 의료 학습이나 GPU 검증으로 집계하지 않는다.

## 확인한 테스트와 공식 근거

가벼운 unit tests는 enum 문자열/JSON/identity, TEXT_QA와 VQA 차이, gold 입력 차단, resume identity, GPU lease, 누적 예산과 source-balanced validation을 검사한다. Python 3.10 syntax 검사는 실제 Python 3.10 실행과 구분한다. 별도 native fixture는 임의 초기화한 tiny Qwen3에서 실제 CPU LoRA update, parameter 변화, EOS/PAD mask, atomic checkpoint, 새 process reload, optimizer/RNG/sampler resume를 실행한다.

```bash
PYTHONPATH=src ../.venv-medvlm/bin/python -m unittest discover -s tests -v
MEDVLM_RUN_TRAINING_TESTS=1 PYTHONPATH=src \
  ../.venv-medvlm/bin/python -m unittest discover -s tests -p 'test_training*.py' -v
```

공식 참고: [Qwen3 모델 카드](https://huggingface.co/Qwen/Qwen3-1.7B), [native Qwen3 지원](https://huggingface.co/docs/transformers/en/model_doc/qwen3), [PEFT checkpoint 계약](https://huggingface.co/docs/peft/en/developer_guides/checkpoint). 모델별 현재 revision/access/native processor 검사 결과는 `manifests/models/`에 별도 기록한다.
