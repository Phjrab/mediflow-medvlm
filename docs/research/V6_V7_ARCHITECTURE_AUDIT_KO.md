# V6/V7 초기 architecture 감사

기준: 2026-11-14, 사용자 R2 전체 지시. 이 문서는 구현 완료나 학습 완료 보고가 아니다. 감사 담당 모델/강도는 위임 설정 `gpt-6-astra/high`이며 사용자 권장 xhigh와 구분한다. 이 감사에서 GPU 실행·모델 가중치 다운로드·서비스 변경·commit은 하지 않았다. CPU tokenizer/마스킹 점검 328건 성공, 기술 실패 0건이다.

## 초기 시작 판단

기존 Qwen3-1.7B 경로로 **새로운 독립 20-update engineering smoke**를 시작할 수 있다. 기존 MIX train의 고정 320개와 validation 8개의 실제 native 인코딩을 CPU에서 검사했다. 이 결과는 V6 전체 준비, 공통 train 동결, 신규 계열 지원, 본학습 성공을 뜻하지 않는다. 주 실행자가 GPU 소유권·환경·소스 snapshot·독립 worker 수명을 검증한 뒤 시작한다. smoke adapter는 본학습에 이어 붙이지 않는다.

기존 완료 smoke를 덮어쓰거나 재개하지 않는다. 새 config/run/pair/experiment/output identity를 만든다. 검사에 사용한 기존 config는 private data root 기준 `v2/text/configs/qwen3_17b-mix-smoke-v2-20261026.json`이다.

| 항목 | 검증된 값 / 새 smoke 설정 |
| --- | --- |
| model / revision | `Qwen/Qwen3-1.7B` / `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| native config / tokenizer | `Qwen3ForCausalLM` / `Qwen2Tokenizer` |
| train file | `v2/frozen/387f45412b6984bc0d195997ec6822b139e23c019df071b9a262c71147147dac/train.jsonl` |
| validation file | `v2/frozen/871b03ebc3a8c1c8ecbf7013babfa31e1af69e53aa3bd7eb0348a86704bb06c1/validation.jsonl` |
| freeze | `v2/manifests/freeze-2b1e5b382209d8ccd1795a89c1afa6f403b7f518f582de1995fde634a3014ace.json` |
| freeze file SHA256 | `eb17d6d07d02a69f8e4657b119cd173bce6e24e6bbe9ee17854fec59754d7f41` |
| selection | 기존 runner와 동일한 seed 20261023 shuffle의 앞 320개, `train_samples_cap=320` |
| updates / batch | `mode=smoke`, `max_optimizer_steps=20`, microbatch 1, accumulation 16 |
| recipe | schema `medvlm.training.v2`, recipe `one_pass`, LR 2e-4, NF4 double quant, r8/alpha16/dropout0.05, BF16, max_length2048 |
| validation / checkpoints | `validation_samples=8`, `checkpoint_every_steps=20` |
| cache / network | 기존 `models/hub`, `local_files_only=true`; 모델 추가 다운로드 불필요 |

private data root의 현재 위치는 저장소 옆 `../medvlm-data`이며, worker에서는 주 실행자가 검증한 절대경로를 사용한다. 위 train/validation 파일은 각각 8,000/1,516행이다. 원본 파일은 그대로 두고 cap/선택 규칙을 새 config에 기록한다.

| CPU 검사 | train | validation |
| --- | ---: | ---: |
| 검사 / 적격 / 실패 | 320 / 320 / 0 | 8 / 8 / 0 |
| 최대 input+target tokens | 1,663 | 459 |
| supervised tokens 합 | 46,179 | 295 |
| AIHub / MedQuAD / PubMedQA | 172 / 129 / 19 | 3 / 3 / 2 |

train 순서 SHA는 `53aef176a2dfd07a990ab6a9462586448c6adde259665d6c67fe48c392b792f2`, validation 순서 SHA는 `a1647436189c2d2d0191d24083738694e2eb8375f6ce34fc26f1479231340b85`다. `CUDA_VISIBLE_DEVICES=''`, offline 모드로 tokenizer/config만 읽었다. CUDA forward/backward·저장·새 process reload는 이후 worker가 실제 증거를 만들어야 한다.

기존 CLI는 `python -m mediflow_medvlm.training train --config <NEW_CONFIG>`와 별도 process의 `python -m mediflow_medvlm.training reload --config <NEW_CONFIG> --checkpoint <VERIFIED_STEP20>`이다. 저장소의 `src`를 그대로 사용하기보다 동결한 실행 snapshot의 `src`를 `PYTHONPATH`로 지정한다. checkpoint는 새 run의 `latest.json`으로 찾고 artifact hash·step20·320행·adapter 변화·finite gradients를 검증한다. 기존 runner는 smoke 성공에 10 updates만 요구하므로 supervisor가 정확한 20 updates를 추가 검증해야 한다.

checkpoint 간격을 20으로 두면 이 smoke의 유일한 정상 checkpoint가 step20이므로 기존 best-selection과 마지막 checkpoint가 같다. 이 특수성이 본학습의 final 선택 문제를 해결한 것은 아니다.

## 읽은 현재 기록과 유지할 제약

R2 §1의 HANDOFF, WORKLOG, DATA_GOVERNANCE, 공개 INDEX/RELEASE, v2 READOUT/RESULTS, v3 RESULTS, v4 RESULTS, v5 READOUT/RESULTS 및 현재 loader/rendering/trainer/evaluator/runtime을 확인했다. 적용 경로에서 AGENTS.md는 발견되지 않았다.

v1~v5는 완료된 역사이며 옛 BLOCKED_DELIVERY, UNRESOLVED_PAIRING, 6/24시간 한도를 현 상태로 되살리지 않는다. 71762 원본 확보·제공처 방식의 독립 random 연결 실험은 이미 완료됐다. 공식 코드 동일성, 개별 QA의 임상 정합성, 증강 parent 복원은 검증되지 않았다. v4 형식 실패와 v5 0/24 결과는 의료 능력 0 또는 random pairing 단독 원인의 증거가 아니다. 원문·이미지·optimizer/RNG·private 검토키는 private로 유지한다. 기존 Release 승인 범위와 새 V6/V7 공개 범위를 혼동하지 않는다. 사람 검토 0건, Jetson USER_DEFERRED를 유지한다.

## 우선 구현할 gate

### P0 — 언어 decoder의 실제 projection 범위

`training/models.py`의 현재 기본 suffix는 q/k/v/o, gate/up/down이다. 대상이 하나라도 있으면 통과하므로 새로운 계열에 대한 충분한 검증이 아니다. installed Transformers 5.20.2의 실제 modeling 소스와 공식 model config를 읽었다. 아래 수량은 구조상 예상이며 실제 로딩 후 `named_modules` 전체 경로로 확정해야 한다.

| 셀 | 현 상태 / 필요한 검사 |
| --- | --- |
| T1/T2 Qwen3 | 기존 경로를 보존하되 decoder block 내부의 7 projection 그룹을 층별 확인. embeddings/lm_head 제외. |
| T3/T4 및 V2/V3 Gemma/MedGemma | native `Gemma3ForConditionalGeneration` 계열과 AutoProcessor 경로를 사용한다. text에서는 pixel/image tensor가 없어야 하고 VQA에서는 실제 이미지가 있어야 한다. 같은 base cache를 공유하되 text/VQA adapter/run은 독립. |
| T5 Phi-4-mini | 공식 config의 native class는 `Phi3ForCausalLM`. `self_attn.qkv_proj/o_proj`, `mlp.gate_up_proj/down_proj`가 필요하다. 32층×4=128개 예상. 현 suffix는 o/down 64개만 잡을 수 있어 fused QKV와 gate-up을 빠뜨린다. |
| T6 SmolLM3 | 해당 revision의 native class·language block 범위를 확인한다. 한국어는 주요 지원 언어 밖의 전이 탐색으로 해석한다. thinking/date 항목은 아래 별도 gate. |
| V1 Qwen3.5 | 24층 중 18 linear-attention과 6 full-attention. 현재 96개에 더해 `linear_attn.in_proj_qkv/in_proj_z/in_proj_b/in_proj_a/out_proj`의 90개가 필요해 총186개 예상. convolution·norm·비선형 상태 parameter는 선형 projection 규약 밖으로 남긴다. 확장 뒤 기존 adapter를 같은 처방으로 재사용할 수 없다. |
| V4 SmolVLM2 | 공식 native class `SmolVLMForConditionalGeneration`; text decoder는 SmolLM2 기반 Llama 구조다. SmolLM3와 동일 decoder가 아니다. processor의 crops/tiles/pixel tensor를 유지하며 vision와 connector/projector를 동결한다. |

global suffix 추가만으로 끝내지 않는다. 확인된 language decoder 하위 경로를 허용하고 실제 module 종류·층·projection별 누락/중복을 검사한다. 특히 SmolVLM의 `connector`는 현재 VISION_MARKERS에 없으므로 all-linear로 확장하면 잘못 포함할 위험이 있다. 실제 전체 target 경로, 그룹별 finite/nonzero gradient 및 adapter 변화 근거를 저장한다. 초기 LoRA A의 gradient가 0일 수 있어 모든 parameter가 첫 backward부터 비영이어야 한다고 요구하지 않는다.

현재 parameter metadata는 trainable/frozen의 단순 numel 합이고 언어/vision/projector 분리가 없다. NF4 packed storage의 numel을 원래 dense parameter 수로 오해하지 않도록 원래 shape 기준 parameter 수와 실제 저장 byte 수를 분리한다.

공식 근거: [Phi 카드](https://huggingface.co/microsoft/Phi-4-mini-instruct), [Phi config](https://huggingface.co/microsoft/Phi-4-mini-instruct/raw/main/config.json), [Qwen3.5 config](https://huggingface.co/Qwen/Qwen3.5-2B/raw/main/config.json), [Gemma 카드](https://huggingface.co/google/gemma-3-4b-it), [MedGemma 카드](https://huggingface.co/google/medgemma-1.5-4b-it), [SmolVLM2 카드](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct). 온라인 main은 조사 자료이며 실제 실행 revision은 별도로 SHA 고정한다.

### P0 — thinking, native mask, 시간에 따른 template 변화

기존 `encode_sample`은 full token IDs의 prefix가 generation prompt와 정확히 일치하는지 검사하고 prompt/image/pad를 -100으로 마스킹한다. 정답 전체 보존과 overflow reject는 좋은 기반이다. 신규 native template의 assistant 종료 token, 멀티턴, 빈/긴 target, EOS=PAD, 실제 이미지 경로를 각각 검사한다.

SmolLM3 공식 template는 `enable_thinking=False`를 실제 지원하며 system `/think` 또는 `/no_think`가 kwarg보다 우선한다. 비활성 경로의 빈 think span은 generation prefix에 포함돼야 한다. 또한 template가 `strftime_now`로 오늘 날짜를 넣는다. 현 코드는 template byte hash만 고정하므로 여러 날의 학습/재로드/Base/Tuned 입력이 달라질 수 있다. 실행 전에 native 의미를 보존하는 결정적 날짜 context 또는 명시적으로 version/hash를 붙인 frozen template 사본을 구현하고 rendered prompt hash까지 검증해야 한다. 단순히 kwarg를 전달한 사실을 비활성 검증으로 삼지 않는다. [공식 카드와 비활성 예제](https://huggingface.co/HuggingFaceTB/SmolLM3-3B), [공식 template](https://huggingface.co/HuggingFaceTB/SmolLM3-3B/raw/main/chat_template.jinja).

### P0 — 공통 train 동결과 final checkpoint

현재 runner는 모델별 overflow를 제외하고 남은 예제로 업데이트한다. V6에는 모든 지원 후보의 native train eligibility를 먼저 모아 축별 공통 IDs를 동결하고, 본학습 중 추가 제외가 생기면 해당 run을 실패로 처리하는 gate가 필요하다. validation/test는 입력만으로 사전 적합성을 검사하고 실패도 예정 분모에 남긴다. 기존 source/group split을 유지한다.

현 runner의 selection은 `minimum_validation_loss`다. V6는 별도 `final_checkpoint.json`에 정상 완료, N개 정확히 한 pass, `ceil(N/16)` updates, 마지막 sample position, checkpoint digest, 실제 final reload 결과를 기록해야 한다. 중단된 latest는 final이 아니다. old selection을 수정하지 않는다.

현재 `reload_checkpoint`는 final을 실제로 로드해도 기존 best selection과 경로가 다르면 manifest의 reload_verified를 true로 바꾸지 않는다. 새 final receipt는 `reload.json`의 실제 checkpoint 경로·status와 identity를 검증해야 하며 run-level boolean 하나로 final reload를 판정하면 안 된다.

기존 AdamW는 LR/weight_decay만 명시하고 betas/eps는 환경 기본값을 사용한다. 현재 설치된 PyTorch의 AdamW signature를 CPU에서 읽어 betas=(0.9,0.999),eps=1e-8,amsgrad=false를 확인했다. V6에서 이 값과 역사 training_state의 param_groups를 대조해 명시한다. 현재 objective는 예제별 target-token 평균 loss를 accumulation group의 예제 수로 나누며 마지막 작은 group을 반복 채우지 않는다. 이를 유지한다.

### P1 — 전체 validation, 선택 gate, likelihood

기존 48개 validation loss와 8개 smoke validation은 V6 전체 dev 선택을 대신하지 않는다. 단계는 preflight/smoke → 공통 freeze → 첫 seed 본학습 → Base/Tuned validation 길이 calibration → 전체 dev → family 및 추가 seed 선택 receipt → 추가 seed → test다. 어떤 V6 test도 selection receipt 전에 실행하지 않는다.

새 protocol에서 MCQ/분류64, 자유서술/VQA256, eval context4096을 시작값으로 둔다. 첫 seed validation에서 task별 truncation>1%이면 해당 task 전체 모델에 한 번만128/512로 올리고 validation을 동일 설정으로 다시 생성한다. 기존16token 예측을 같은 규약의 결과로 재사용하지 않는다.

likelihood scorer는 아직 구현돼 있지 않다. 모든 후보 label token의 conditional log-prob 합, native assistant prefix/BPE 경계, EOS 정책, 후보 순서와 점수를 독립 artifact로 저장한다. multi-token label fixture와 수작업 작은 tensor 사례를 검증한다. scorer 실패는 생성 평가와 별도 상태로 관리한다.

추가 seed는 Tuned 절대 validation 지표로 family 대표를 고른 다음 서로 다른 두 family에서 선택한다. Qwen 두 크기와 Gemma/MedGemma를 각각 다른 family로 세지 않는다. 첫 seed10조건, 추가 최대8조건, V7별도4조건의 유한 범위를 유지한다.

### P1 — 비용 및 실패 상태

기존 평가에는 elapsed/peak memory가 있으나 TTFT는 null이다. whole-generation tokens/s를 decode-only로 바꾸어 부르지 않는다. 동일 장비/고정 subset/warmup/synchronization/반복 수와 문자량을 함께 기록하고 group-paired bootstrap을 추가한다. 미측정 값은 null로 유지한다.

기술 복구는 모델·단계별 최초 이후 최대2회이며 실패 attempt를 보존한다. GPU worker와 queue owner는 각 하나다. lock, heartbeat와 last_progress 분리, PID/start time/boot identity, immutable snapshot, stale checkpoint 해시를 확인한다. 정상 validation·저장 시간을 hang으로 판정하지 않는다.

## GPU 시간 상한 없음의 실제 근거

`configs/resource_policy.json`은 unlimited와 total/model/run/job/campaign 다섯 null이다. `runtime.ResourcePolicy`는 unlimited에 유한값이 있으면 거부하고 null deadline을 반환한다. `BudgetLedger.can_continue`는 unlimited일 때 시간 때문에 중단하지 않는다. `resolve_resource_policy`는 budget_mode 없는 옛6/24시간 필드로 현재 정책을 덮어쓰지 않는다. 이번 CPU 점검에서 실제 smoke source config도 동일 unlimited/null로 해석됐다. usage ledger와 checkpoint floor는 계속 보존한다. OS service에도 별도 runtime limit을 만들면 안 된다.

## V7에 넘길 계약

V6의 T1 revision/loader/LoRA/mask/final/eval protocol을 검증한 뒤 P0/P1/R0/R1 네 조건을 각각 Base에서 시작한다. P는 71875 short/long 각1,000, R은 기존 v4 고정2,000쌍, M은 동일 MCQ400개가 기본이다. V6 공통 dev까지 보호 경계를 추가 검사하며 71762 원본 전체 감사를 반복하지 않는다. 필요시 성능을 보기 전에 R2의80배수 공식으로 공통 N을 줄이고 N<800이면 V7만 terminal로 종료한다. 20% MCQ는 기존 예제 예산을 대체하며 updates를 추가하지 않는다. 각 조건의 target tokens·style·source 차이를 보고하고 random pairing의 순수 인과 효과로 해석하지 않는다.
