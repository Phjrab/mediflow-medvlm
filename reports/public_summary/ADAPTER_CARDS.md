# 모델·adapter 실행 카드

공개 의료 데이터의 오프라인 연구용이다. 모델이나 adapter 가중치를 공개 업로드하지 않았다. 자동 lexical 점수와 객관식 결과는 임상 검증이 아니며 전문가 검토는 미실행이다. 각 adapter에는 기반 모델 및 학습 데이터의 이용 조건이 함께 적용된다. `${MEDVLM_DATA_ROOT}`는 실제 로컬 data root다.

## Qwen/Qwen3-1.7B

학습 상태 `SUCCEEDED`, 재로드 `SUCCEEDED`, 전체 셀 `SUCCEEDED`.

| 항목 | 실제 값 또는 사전 설정 |
|---|---|
| Base revision | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| 로컬 native weights 파일 | SUCCEEDED (2/2 files, 4063515592 bytes); 로드 성공과 별개 |
| Base native files digest | `4d457421c1028aeb7bf5d539ace87ab5687adda2a02f6164a29efbb41bb531c0` |
| Processor / loader | Qwen2Tokenizer / Qwen3ForCausalLM |
| Native chat template SHA256 | `41d5929bf73796beb66809ac700b2cf3ff81694f933e5c14d52b7fd6963c947d` |
| Adapter 경로 | `${MEDVLM_DATA_ROOT}/runs/qwen3_17b-v1-20261023/checkpoints/step-000525-c28d685d` |
| Adapter digest / parent | `0a1d2fbcf96b01b9fd84c6b4377930e0f042ba5f6b4f0432bdef9a932e83a0d9` / `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| 학습 source / config hash | `717b44dce9a170c3c68a083bb595ea5e2b3a0fd964884dc39dfa396da1730397` / `e026913f3cae9b0b7bd1ec66842247eb81d4c201468f9093b6c5d58abd644691` |
| 설정 seed / rank / alpha / dropout | 20261023 / 8 / 16 / 0.05 |
| 설정 dtype / quantization / max length | bfloat16 / nf4 / 2048 |
| 실제 학습 / 동결 parameter 수 | 8716288 / 1015931904 |
| 실제 target module 수 / SHA256 | 196 / `1e57874a63c23dab76d3731b219dea34c3e6e0f164760b44d564c74cdbd6ff2c` |
| 실제 sample pass / target token / update | 8400 / 1983789 / 525 |
| 실제 epoch / effective batch | 0.9940828402366864 / 16 |
| 학습 wall seconds / GPU seconds | 3338.4511403010692 / 3338.4511403010692 |
| Peak CUDA allocated / reserved bytes | 6951442432 / 11075059712 |
| 선언 license / 원본 근거 | apache-2.0 / https://huggingface.co/Qwen/Qwen3-1.7B |

원본 설정, 정확한 target module 목록, 환경 버전, train/dev digest, validation 선정과 source snapshot은 해당 로컬 run의 `config.json`, `manifest.json`, `selection.json`, `reload.json`에 보존한다. 동결 parameter 수는 로드한 quantized tensor의 numel 합으로, 원본 모델 규모와 다를 수 있다. 미실행 모델의 null은 측정값이 아니다.

## Qwen/Qwen3-4B

학습 상태 `SUCCEEDED`, 재로드 `SUCCEEDED`, 전체 셀 `SUCCEEDED`.

| 항목 | 실제 값 또는 사전 설정 |
|---|---|
| Base revision | `1cfa9a7208912126459214e8b04321603b3df60c` |
| 로컬 native weights 파일 | SUCCEEDED (3/3 files, 8044982000 bytes); 로드 성공과 별개 |
| Base native files digest | `e2f061ac45cee943a3360c21c7713caef4bbe58dda2cdf92098e2a349e48f5a1` |
| Processor / loader | Qwen2Tokenizer / Qwen3ForCausalLM |
| Native chat template SHA256 | `41d5929bf73796beb66809ac700b2cf3ff81694f933e5c14d52b7fd6963c947d` |
| Adapter 경로 | `${MEDVLM_DATA_ROOT}/runs/qwen3_4b-v1-20261023/checkpoints/step-000525-3c6161ba` |
| Adapter digest / parent | `5781b132ee742c376b99dd879348519b649f2b1bab18e973a7beb4554088dee4` / `1cfa9a7208912126459214e8b04321603b3df60c` |
| 학습 source / config hash | `92acc1d170ddd7e6c666b6dc3da2126411b93213018ae7e267d520f0639e9104` / `43f1cce38e73481a5cfe5a48ec40d77ae6ebde0e7c61f7380333a234eb6098b6` |
| 설정 seed / rank / alpha / dropout | 20261023 / 8 / 16 / 0.05 |
| 설정 dtype / quantization / max length | bfloat16 / nf4 / 2048 |
| 실제 학습 / 동결 parameter 수 | 16515072 / 2205810176 |
| 실제 target module 수 / SHA256 | 252 / `631bb032cfdb9bd207f490e5d968932d9916e2b6646745083d4c58b7eb1df69a` |
| 실제 sample pass / target token / update | 8400 / 1983789 / 525 |
| 실제 epoch / effective batch | 0.9940828402366864 / 16 |
| 학습 wall seconds / GPU seconds | 6667.1352707440965 / 6665.283004283905 |
| Peak CUDA allocated / reserved bytes | 9002012672 / 11863588864 |
| 선언 license / 원본 근거 | apache-2.0 / https://huggingface.co/Qwen/Qwen3-4B |

원본 설정, 정확한 target module 목록, 환경 버전, train/dev digest, validation 선정과 source snapshot은 해당 로컬 run의 `config.json`, `manifest.json`, `selection.json`, `reload.json`에 보존한다. 동결 parameter 수는 로드한 quantized tensor의 numel 합으로, 원본 모델 규모와 다를 수 있다. 미실행 모델의 null은 측정값이 아니다.

## Qwen/Qwen3.5-2B

학습 상태 `SUCCEEDED`, 재로드 `SUCCEEDED`, 전체 셀 `SUCCEEDED`.

| 항목 | 실제 값 또는 사전 설정 |
|---|---|
| Base revision | `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| 로컬 native weights 파일 | SUCCEEDED (1/1 files, 4548221488 bytes); 로드 성공과 별개 |
| Base native files digest | `9f30f5743130cf4b209ef381c4f017be0e8b648d6aa9c9e20d7061a09b60bf47` |
| Processor / loader | Qwen3VLProcessor / Qwen3_5ForConditionalGeneration |
| Native chat template SHA256 | `e9cb98e65623809a8a297af444986a3b417ecae1a19d519c0952eb1192abc5c1` |
| Adapter 경로 | `${MEDVLM_DATA_ROOT}/runs/qwen35_2b-v1-20261023/checkpoints/step-000100-9f8b3683` |
| Adapter digest / parent | `6e7eb2fe1e9f88c95d1ab7fe416ea805dfdeb160fd4f7854942ad13c9724d5be` / `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| 학습 source / config hash | `92acc1d170ddd7e6c666b6dc3da2126411b93213018ae7e267d520f0639e9104` / `205464c60638d5987cda24b6f1964dc4ae649f9f9bb1d5b08e3a889b460d0ddc` |
| 설정 seed / rank / alpha / dropout | 20261023 / 8 / 16 / 0.05 |
| 설정 dtype / quantization / max length | bfloat16 / nf4 / 2048 |
| 실제 학습 / 동결 parameter 수 | 5455872 / 1363305280 |
| 실제 target module 수 / SHA256 | 96 / `45dd4c9b621aee2b4610698b5830ce331fd9e0e2ffbdb13e6cdb85b5a1324616` |
| 실제 sample pass / target token / update | 1979 / 7693 / 124 |
| 실제 epoch / effective batch | 1.0 / 16 |
| 학습 wall seconds / GPU seconds | 2131.82813272113 / 2130.2282421588898 |
| Peak CUDA allocated / reserved bytes | 7582394368 / 11655970816 |
| 선언 license / 원본 근거 | apache-2.0 / https://huggingface.co/Qwen/Qwen3.5-2B |

원본 설정, 정확한 target module 목록, 환경 버전, train/dev digest, validation 선정과 source snapshot은 해당 로컬 run의 `config.json`, `manifest.json`, `selection.json`, `reload.json`에 보존한다. 동결 parameter 수는 로드한 quantized tensor의 numel 합으로, 원본 모델 규모와 다를 수 있다. 미실행 모델의 null은 측정값이 아니다.

## Qwen/Qwen3-VL-2B-Instruct

학습 상태 `PREPARED_NOT_RUN`, 재로드 `PREPARED_NOT_RUN`, 전체 셀 `PREPARED_NOT_RUN`.

| 항목 | 실제 값 또는 사전 설정 |
|---|---|
| Base revision | `89644892e4d85e24eaac8bacfd4f463576704203` |
| 로컬 native weights 파일 | PREPARED_NOT_RUN (0/총수 미확인 files, 0 bytes); 로드 성공과 별개 |
| Base native files digest | `미측정` |
| Processor / loader | Qwen3VLProcessor / None |
| Native chat template SHA256 | `None` |
| Adapter 경로 | `미생성` |
| Adapter digest / parent | `None` / `None` |
| 학습 source / config hash | `None` / `None` |
| 설정 seed / rank / alpha / dropout | 20261023 / 8 / 16 / 0.05 |
| 설정 dtype / quantization / max length | bfloat16 / nf4 / 2048 |
| 실제 학습 / 동결 parameter 수 | None / None |
| 실제 target module 수 / SHA256 | None / `None` |
| 실제 sample pass / target token / update | None / None / None |
| 실제 epoch / effective batch | None / None |
| 학습 wall seconds / GPU seconds | None / None |
| Peak CUDA allocated / reserved bytes | None / None |
| 선언 license / 원본 근거 | apache-2.0 / https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct |

추가 이용 조건 근거: Apache License 2.0. Apache 2.0 terms apply; retain required notices. [apache-2.0](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/blob/89644892e4d85e24eaac8bacfd4f463576704203/README.md).

원본 설정, 정확한 target module 목록, 환경 버전, train/dev digest, validation 선정과 source snapshot은 해당 로컬 run의 `config.json`, `manifest.json`, `selection.json`, `reload.json`에 보존한다. 동결 parameter 수는 로드한 quantized tensor의 numel 합으로, 원본 모델 규모와 다를 수 있다. 미실행 모델의 null은 측정값이 아니다.

## Qwen/Qwen2.5-VL-3B-Instruct

학습 상태 `PREPARED_NOT_RUN`, 재로드 `PREPARED_NOT_RUN`, 전체 셀 `PREPARED_NOT_RUN`.

| 항목 | 실제 값 또는 사전 설정 |
|---|---|
| Base revision | `66285546d2b821cf421d4f5eb2576359d3770cd3` |
| 로컬 native weights 파일 | PREPARED_NOT_RUN (0/총수 미확인 files, 0 bytes); 로드 성공과 별개 |
| Base native files digest | `미측정` |
| Processor / loader | Qwen2_5_VLProcessor / None |
| Native chat template SHA256 | `None` |
| Adapter 경로 | `미생성` |
| Adapter digest / parent | `None` / `None` |
| 학습 source / config hash | `None` / `None` |
| 설정 seed / rank / alpha / dropout | 20261023 / 8 / 16 / 0.05 |
| 설정 dtype / quantization / max length | bfloat16 / nf4 / 2048 |
| 실제 학습 / 동결 parameter 수 | None / None |
| 실제 target module 수 / SHA256 | None / `None` |
| 실제 sample pass / target token / update | None / None / None |
| 실제 epoch / effective batch | None / None |
| 학습 wall seconds / GPU seconds | None / None |
| Peak CUDA allocated / reserved bytes | None / None |
| 선언 license / 원본 근거 | None / https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct |

추가 이용 조건 근거: Qwen Research License Agreement (2024-09-19). Research/evaluation only under non-commercial grant; commercial use requires an Alibaba Cloud license. [qwen-research](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/66285546d2b821cf421d4f5eb2576359d3770cd3/LICENSE).

원본 설정, 정확한 target module 목록, 환경 버전, train/dev digest, validation 선정과 source snapshot은 해당 로컬 run의 `config.json`, `manifest.json`, `selection.json`, `reload.json`에 보존한다. 동결 parameter 수는 로드한 quantized tensor의 numel 합으로, 원본 모델 규모와 다를 수 있다. 미실행 모델의 null은 측정값이 아니다.

## google/medgemma-1.5-4b-it

학습 상태 `SUCCEEDED`, 재로드 `SUCCEEDED`, 전체 셀 `SUCCEEDED`.

| 항목 | 실제 값 또는 사전 설정 |
|---|---|
| Base revision | `91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b` |
| 로컬 native weights 파일 | SUCCEEDED (2/2 files, 8600277880 bytes); 로드 성공과 별개 |
| Base native files digest | `33733078d018113212f156467c2de6137bebe8de691ad5f28c90359c33156aa7` |
| Processor / loader | Gemma3Processor / Gemma3ForConditionalGeneration |
| Native chat template SHA256 | `a5c992cca048607f8a84c3d6ca4eb14bacc19bc1b4b2b3adfe3b987a4569f714` |
| Adapter 경로 | `${MEDVLM_DATA_ROOT}/runs/medgemma15_4b-v1-20261023/checkpoints/step-000124-a020e6f5` |
| Adapter digest / parent | `8f40335b936fd587873a23a2cfeab2b1ae3bf4b52d52a44a1bc27a8aafb1c31f` / `91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b` |
| 학습 source / config hash | `92acc1d170ddd7e6c666b6dc3da2126411b93213018ae7e267d520f0639e9104` / `1578134a46eb626c73b2cf9a529b5b7e0e298203e6c77334f405b269e1f54a01` |
| 설정 seed / rank / alpha / dropout | 20261023 / 8 / 16 / 0.05 |
| 설정 dtype / quantization / max length | bfloat16 / nf4 / 2048 |
| 실제 학습 / 동결 parameter 수 | 14901248 / 2490222960 |
| 실제 target module 수 / SHA256 | 238 / `b6b439af0788f986fa98920f7cafd7a46389e3591c00e49de9797d6aca4faa1a` |
| 실제 sample pass / target token / update | 1979 / 7252 / 124 |
| 실제 epoch / effective batch | 1.0 / 16 |
| 학습 wall seconds / GPU seconds | 4308.618213267066 / 4306.9445815086365 |
| Peak CUDA allocated / reserved bytes | 7894200832 / 8264876032 |
| 선언 license / 원본 근거 | other / https://huggingface.co/google/medgemma-1.5-4b-it |

추가 이용 조건 근거: Health AI Developer Foundations Terms of Use. Model-card declaration names Health AI Developer Foundations Terms; existing authenticated gated access was verified. This receipt does not infer new permissions. [health-ai-developer-foundations](https://huggingface.co/google/medgemma-1.5-4b-it/blob/91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b/README.md).

원본 설정, 정확한 target module 목록, 환경 버전, train/dev digest, validation 선정과 source snapshot은 해당 로컬 run의 `config.json`, `manifest.json`, `selection.json`, `reload.json`에 보존한다. 동결 parameter 수는 로드한 quantized tensor의 numel 합으로, 원본 모델 규모와 다를 수 있다. 미실행 모델의 null은 측정값이 아니다.

## google/gemma-3-4b-it

학습 상태 `PREPARED_NOT_RUN`, 재로드 `PREPARED_NOT_RUN`, 전체 셀 `PREPARED_NOT_RUN`.

| 항목 | 실제 값 또는 사전 설정 |
|---|---|
| Base revision | `093f9f388b31de276ce2de164bdc2081324b9767` |
| 로컬 native weights 파일 | PREPARED_NOT_RUN (0/총수 미확인 files, 0 bytes); 로드 성공과 별개 |
| Base native files digest | `미측정` |
| Processor / loader | Gemma3Processor / None |
| Native chat template SHA256 | `None` |
| Adapter 경로 | `미생성` |
| Adapter digest / parent | `None` / `None` |
| 학습 source / config hash | `None` / `None` |
| 설정 seed / rank / alpha / dropout | 20261023 / 8 / 16 / 0.05 |
| 설정 dtype / quantization / max length | bfloat16 / nf4 / 2048 |
| 실제 학습 / 동결 parameter 수 | None / None |
| 실제 target module 수 / SHA256 | None / `None` |
| 실제 sample pass / target token / update | None / None / None |
| 실제 epoch / effective batch | None / None |
| 학습 wall seconds / GPU seconds | None / None |
| Peak CUDA allocated / reserved bytes | None / None |
| 선언 license / 원본 근거 | gemma / https://huggingface.co/google/gemma-3-4b-it |

원본 설정, 정확한 target module 목록, 환경 버전, train/dev digest, validation 선정과 source snapshot은 해당 로컬 run의 `config.json`, `manifest.json`, `selection.json`, `reload.json`에 보존한다. 동결 parameter 수는 로드한 quantized tensor의 numel 합으로, 원본 모델 규모와 다를 수 있다. 미실행 모델의 null은 측정값이 아니다.

## HuggingFaceTB/SmolVLM-500M-Instruct

학습 상태 `PREPARED_NOT_RUN`, 재로드 `PREPARED_NOT_RUN`, 전체 셀 `PREPARED_NOT_RUN`.

| 항목 | 실제 값 또는 사전 설정 |
|---|---|
| Base revision | `a7da5b986cb59b408707209984f360a5f4ad7e47` |
| 로컬 native weights 파일 | SUCCEEDED (1/1 files, 1015025832 bytes); 로드 성공과 별개 |
| Base native files digest | `미측정` |
| Processor / loader | Idefics3Processor / None |
| Native chat template SHA256 | `None` |
| Adapter 경로 | `미생성` |
| Adapter digest / parent | `None` / `None` |
| 학습 source / config hash | `None` / `None` |
| 설정 seed / rank / alpha / dropout | 20261023 / 8 / 16 / 0.05 |
| 설정 dtype / quantization / max length | bfloat16 / none / 2048 |
| 실제 학습 / 동결 parameter 수 | None / None |
| 실제 target module 수 / SHA256 | None / `None` |
| 실제 sample pass / target token / update | None / None / None |
| 실제 epoch / effective batch | None / None |
| 학습 wall seconds / GPU seconds | None / None |
| Peak CUDA allocated / reserved bytes | None / None |
| 선언 license / 원본 근거 | apache-2.0 / https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct |

원본 설정, 정확한 target module 목록, 환경 버전, train/dev digest, validation 선정과 source snapshot은 해당 로컬 run의 `config.json`, `manifest.json`, `selection.json`, `reload.json`에 보존한다. 동결 parameter 수는 로드한 quantized tensor의 numel 합으로, 원본 모델 규모와 다를 수 있다. 미실행 모델의 null은 측정값이 아니다.
