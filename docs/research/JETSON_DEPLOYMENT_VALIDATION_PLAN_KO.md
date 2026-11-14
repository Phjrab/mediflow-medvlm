# 향후 Jetson 배포 검증 계획

상태: `USER_DEFERRED / NOT_RUN`. 2026-11-14 R2에 따라 계획만 준비했다. Jetson에 접속·파일 전송·학습·변환·재플래싱·JetPack 변경·벤치마크를 하지 않았다. IP, 계정, RAM, 모듈, OS를 추정하지 않는다. 과거 다른 프로젝트의 Orin NX를 이번 Orin Nano로 간주하지 않는다.

## 장치 조사와 범위 고정

향후 실제 접속·실행이 승인되면 먼저 읽기 전용 조사로 다음을 기록한다.

| 구분 | 확인할 값 | 현재 값 |
| --- | --- | --- |
| 장치 | 정확한 Orin Nano 모듈,4GB/8GB,Developer Kit/Super 여부 | 미확인 |
| runtime | OS/aarch64,JetPack,L4T,CUDA,TensorRT,driver,backend 버전 | 미확인 |
| 자원 | 저장장치 여유,RAM/SWAP,UI·카메라·음성 상주 메모리 | 미확인 |
| 전력·열 | nvpmodel,전원,냉각,clocks,thermal/throttling | 미확인 |
| 용도 | 사전 선언 모델 비교 또는 최종 제품 후보 회귀 | 미확정 |

Super 명칭이 RAM 증가를 뜻한다고 가정하지 않는다. Tegra의 CPU/GPU 메모리는 같은 물리 DRAM을 사용하므로 표시 RAM 전체를 전용 GPU VRAM으로 계산하지 않는다. [NVIDIA CUDA for Tegra](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html).

학습 장치가 달라졌다는 이유만으로 의료 SFT를 반복할 필요는 없다. 데스크톱에서 검증한 정확한 Base revision+adapter와 예측을 기준으로 보존하고 실제 변환 artifact/runtime을 검증한다. 모든 V6 셀을 Jetson에 올리는 작업을 선행 조건으로 만들지 않는다.

## 모델별 compatibility manifest 초안

아래 표는 성공 지원표가 아니라 향후 확인 대상이다. runtime 자체의 모델 목록, 정확한 model_type/크기, converter, 양자화, LoRA, 이미지 처리, Jetson 플랫폼 지원을 각각 확인한다. 한 Qwen의 성공을 다른 계열에 확대하지 않는다.

| model ID | 작업 | llama.cpp/GGUF | TensorRT-LLM | TensorRT Edge-LLM | 실제 장치 |
| --- | --- | --- | --- | --- | --- |
| Qwen/Qwen3-1.7B | text | UNVERIFIED | UNVERIFIED | UNVERIFIED | NOT_RUN |
| Qwen/Qwen3-4B | text | UNVERIFIED | UNVERIFIED | UNVERIFIED | NOT_RUN |
| google/gemma-3-4b-it | text/VQA 별도 adapter | UNVERIFIED | UNVERIFIED | UNVERIFIED | NOT_RUN |
| google/medgemma-1.5-4b-it | text/VQA 별도 adapter | UNVERIFIED | UNVERIFIED | UNVERIFIED | NOT_RUN |
| microsoft/Phi-4-mini-instruct | text | UNVERIFIED | UNVERIFIED | UNVERIFIED | NOT_RUN |
| HuggingFaceTB/SmolLM3-3B | text | UNVERIFIED | UNVERIFIED | UNVERIFIED | NOT_RUN |
| Qwen/Qwen3.5-2B | VQA | UNVERIFIED | UNVERIFIED | UNVERIFIED | NOT_RUN |
| HuggingFaceTB/SmolVLM2-2.2B-Instruct | VQA | UNVERIFIED | UNVERIFIED | UNVERIFIED | NOT_RUN |

2026-11-14에 공식 문서 위치를 확인했다. 실제 선택 시 해당 backend commit/release와 장치 버전에 맞추어 다시 확인한다: [llama.cpp multimodal 문서](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md), [TensorRT-LLM support matrix](https://nvidia.github.io/TensorRT-LLM/reference/support-matrix.html), [TensorRT Edge-LLM supported models](https://nvidia.github.io/TensorRT-Edge-LLM/user_guide/getting_started/supported-models.html). 현재 문서의 목록 등재만으로 해당 LoRA와 Orin Nano에서 실행됐다고 판정하지 않는다. Phi-4-multimodal 지원은 Phi-4-mini-instruct 지원과 별개다.

## artifact 변환 및 품질 확인 순서

1. A: 검증된 desktop Base+LoRA의 checkpoint/processor/prompt/예측 hash를 보존한다.
2. B: 선택 runtime의 adapter 직접 지원을 확인한다. 병합이 필요하면 정확한 원본 Base에 병합하고 데스크톱 회귀를 수행한다.
3. C: 지원되는 GGUF/INT4 등으로 변환한다. calibration이 필요하면 train/dev만 사용한다. 변환 도구·commit·명령·설정·입력/출력 SHA를 별도로 기록한다.
4. D: 최종 artifact를 실제 Jetson backend에서 평가한다. VLM은 tokenizer/chat template/vision encoder/projector/crops/해상도/시각 token과 실제 pixel 입력까지 확인한다.

NF4 QLoRA와 GGUF Q4_K_M/AWQ INT4는 같은 포맷이나 연산이 아니다. 저장 bit 수만으로 품질 동일성을 주장하지 않는다. x86 RTX에서 만든 TensorRT engine을 Jetson에 그대로 복사하면 된다고 가정하지 않는다. engine은 플랫폼·GPU·버전·plugin 호환 제약을 확인해야 한다. [TensorRT 공식 호환성](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html).

변환 manifest 필수 필드: model_id/base_revision, adapter_sha, original_config_sha, processor/template_sha, merge_tool/version, converter/version, quantization/calibration_ids_hash, artifact_files_sha, backend/version/build_flags, device/runtime_identity, eval_protocol_hash, status/error, created_at. 아직 실행하지 않은 값은 null이며 추정값으로 채우지 않는다.

## 품질 회귀와 선택의 분리

동일 artifact/precision의 장치 이동은 고정 사례의 기능·형식·출력 회귀와 task 점수를 확인한다. 병합·양자화·kernel/runtime/processor 변경은 사전에 고정한 충분한 의료 평가분을 사용한다. 가능하면 기존 전체 고정 benchmark를 평가하고, 불가능하면 사전 선언 subset과 실제 분모를 명시한다. 기존 공개 benchmark를 새 blind test라고 부르지 않는다.

숫자·단위·부정·선택지·한국어·길이 종료 및 VQA OPEN/CLOSED를 별도로 보고한다. 누락/OOM/미지원도 예정 분모에 유지한다. 양자화 설정을 성능으로 선택했다면 그 평가의 selection 역할을 기록하고 최종 확인 평가와 구분한다. bitwise 동일성만을 통과 기준으로 요구하지 않으며 임상 검증 완료로 확대하지 않는다.

## 성능 시험 설계

- cold load/첫 요청과 warm 상태를 구분한다. batch/concurrency1,고정 context·입출력 상한·이미지 processor를 사용한다.
- TTFT,decode 시간/토큰 간 지연,end-to-end p50/p95,실제 출력 native tokens와 문자/UTF-8 byte 수를 측정한다.
- RAM/SWAP/peak/OOM 및 입력 context·이미지 해상도별 변화를 기록한다.
- 실제 tegrastats 등의 계측으로 온도,전력 rail,주파수,throttling을 수집한다. rail 전력과 콘센트 전체 전력을 구분한다.
- 최초 안정성 시험은 사전 선언한 30~60분 고정 요청 반복 및 UI/카메라 동시 실행을 계획한다. 이는 시험 시나리오이며 GPU 누적 시간 상한이 아니다.
- 응답시간 목표는 사용자 요구 또는 측정 전 선언값을 사용한다. 근거 없이 tokens/s 예상치를 확정하지 않는다.

향후 산출물은 장치 inventory,compatibility manifest,변환 lineage,동결 평가 cohort,품질 회귀 및 성능 측정표,OOM/미지원 목록이다. 현재 이 문서 외 실제 Jetson artifact나 측정값은 없다.
