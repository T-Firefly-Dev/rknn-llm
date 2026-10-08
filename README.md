# LLM Deployment Guide Based on RKLLM SDK (AIBOX-3588)

RK3588 is one of the mainstream platforms for on-device large language model deployment today. For developers, the motivation to move LLM inference from the cloud to an edge device falls into three areas:

1. **Data locality.** Conversations, images and documents are all processed locally on the NPU, which satisfies the privacy and compliance requirements of industrial sites, intranet offices and classified environments. This is exactly how the official RKLLM Quickstart positions the SDK: "all processing happens locally on your device — your data never leaves it."
2. **Predictable cost.** Cloud APIs are billed per token, so the cost of running high-frequency task-oriented inference (log summarization, ticket classification, OCR post-processing) grows linearly with call volume; an edge device is a one-off hardware investment with a typical power draw of only 2.64W.
3. **Ecosystem maturity.** Since its open-source release in 2024, the RKLLM SDK has maintained steady iteration, keeping pace with mainstream model families (the full Qwen and Gemma lineups, DeepSeek-R1-Distill, and others), and ships with an officially maintained pre-converted model repository and an OpenAI-compatible service example, keeping integration effort low.

## RKLLM Software Stack

The stack is organized into three layers:

```
┌──────────────────────────────────────────────────┐
│  Host PC (x86_64 Linux)                          │
│  RKLLM-Toolkit (Python)                          │
│  HuggingFace/GGUF model → quantization → .rkllm  │
└───────────────────────┬──────────────────────────┘
                        │ adb / scp transfer
┌───────────────────────▼──────────────────────────┐
│  Board (AIBOX-3588)                              │
│  RKLLM Runtime (C/C++ API, librkllmrt.so)        │
│  Load .rkllm → invoke NPU driver for inference   │
├──────────────────────────────────────────────────┤
│  RKNPU kernel driver (requires ≥ v0.9.8)         │
└──────────────────────────────────────────────────┘
```

- **RKLLM-Toolkit**: runs on the host PC and handles model conversion and quantization. It supports loading HuggingFace-format models (`load_huggingface`) as well as GGUF models (`load_gguf`, q4_0/fp16).
- **RKLLM Runtime**: the on-board C/C++ inference library, which streams inference results through predefined callback functions. Rockchip also provides a packaged Flask service (OpenAI-compatible API) and a Gradio web UI.
- **RKNPU driver**: already merged into the mainline Rockchip kernel tree; on the board side only the driver version needs to be verified.

## Model Ecosystem

The following is based on version v1.3.1.

### Supported Model Families

| Category | Models |
| --- | --- |
| General LLMs | Qwen2 / Qwen2.5 / **Qwen3 / Qwen3.5**, LLaMA, TinyLLaMA, Phi2/Phi3, ChatGLM3-6B, Gemma2 / **Gemma3 / Gemma3n / Gemma4**, InternLM2, MiniCPM3 / MiniCPM4, TeleChat2, DeepSeek-R1-Distill, Spark-X2.5, RWKV7 |
| Multimodal VLMs | **Qwen2-VL / Qwen3-VL**, MiniCPM-V-2_6, Janus-Pro-1B, InternVL2-1B / InternVL3-1B, SmolVLM, SmolLM3, **LFM2.5-VL-450M** |
| OCR-specific | **DeepSeekOCR** (3B, 570M activated parameters) |

Compared with the v1.1.x/v1.2.x releases that most early community tutorials were based on, the current version adds support for Qwen3/Qwen3.5, Gemma3/3n/4, MiniCPM4, InternVL3, DeepSeekOCR, RWKV7 and others, and introduces a multimodal audio input interface (Gemma4 supports simultaneous image and audio input).

### Official Performance Data (RK3588, w8a8)

Test conditions: CPU and NPU both locked at maximum frequency (official frequency-setting script), models converted with `optimization_level=0`, 128 input tokens and 64 generated tokens.

| Model | Size | TTFT | Generation speed | Runtime memory |
| --- | --- | --- | --- | --- |
| LFM2.5-VL | 450M | 0.10s | **62.2** tok/s | 437 MB |
| MiniCPM4 | 0.5B | 0.14s | 45.3 tok/s | 535 MB |
| Qwen2 | 0.5B | 0.15s | 41.6 tok/s | 670 MB |
| Qwen3 | 0.6B | 0.20s | 32.9 tok/s | 791 MB |
| **Qwen3.5** | **0.8B** | 0.59s | **27.1** tok/s | 1.0 GB |
| Qwen2.5 | 1.5B | 0.38s | 16.7 tok/s | 1.7 GB |
| Qwen3.5 | 2B | 0.78s | 13.6 tok/s | 2.1 GB |
| Qwen3-VL (multimodal) | 2B | 0.38s | 15.0 tok/s | 1.9 GB |
| Gemma2 | 2B | 0.60s | 10.4 tok/s | 2.8 GB |
| Qwen3.5 | 4B | 2.07s | 6.2 tok/s | 4.6 GB |
| Phi3 | 3.8B | 1.02s | 7.5 tok/s | 3.8 GB |
| DeepSeekOCR | 3B(A570M) | 0.70s | 31.6 tok/s | 3.1 GB |
| ChatGLM3 | 6B | 1.35s | 5.0 tok/s | 6.0 GB |

Full-pipeline multimodal (VLM) data (RK3588, w8a8):

| Model | Image encoding (448×448) | Prefill | Decode |
| --- | --- | --- | --- |
| Qwen3-VL-2B | 2.08s | 649ms | 14.9 tok/s |
| DeepSeekOCR-3B | 2.09s | 696ms | 31.8 tok/s |
| Qwen3.5-0.8B (vision) | 0.69s | 1.56s | 27.0 tok/s |
| Qwen2.5-VL-3B | 2.93s (392×392) | 1120ms | 8.7 tok/s |

## Quick Evaluation

Rockchip maintains the **pre-converted model repository rkllm_model_zoo**, which provides `.rkllm` files for mainstream models together with matching demo executables. When there is no need to set up a conversion environment on the PC, this is the recommended path:

1. Download: [rkllm_model_zoo](https://console.box.lenovo.com/l/l0tXb8), fetch code `rkllm` (the quickstart directory contains both the demo and the models).
2. Transfer to the board:

```bash
adb push ./demo_Linux_aarch64 /data
adb push model.rkllm /data/demo_Linux_aarch64
adb push model.rknn /data/demo_Linux_aarch64   # multimodal vision encoder
```

3. Run (visual Q&A with Qwen3-VL-2B as the example):

```bash
adb shell
cd /data/demo_Linux_aarch64
export LD_LIBRARY_PATH=./lib
# usage: ./demo <image> <vision_rknn> <audio> <audio_rknn> <llm_rkllm> \
#              <max_new_tokens> <max_context> <npu_cores> <platform> [tokens...]
./demo demo.jpg ./qwen3-vl-2b_vision_rk3588.rknn "" "" \
       ./qwen3-vl-2b-instruct_w8a8_rk3588.rkllm 2048 4096 3 rk3588 \
       "<|vision_start|>" "<|vision_end|>" "<|image_pad|>"
```

The vision encoder (`.rknn`, FP16) and the language model (`.rkllm`) work together: the image is first encoded into a token sequence by RKNN and then passed to RKLLM for generation, with the whole pipeline running on the local NPU. Passing `""` for an image or audio modality disables that input; special tokens such as `img_start` must be checked against the chat implementation in the model source (for example, InternVL3's `modeling_internvl_chat.py`).

## Custom Model Conversion and Deployment

Use this path when you need to convert a model or quantization profile of your own choosing. The overall flow is: **calibration on the PC → conversion and export → cross-compilation → execution on the board**.

### RKLLM-Toolkit Installation

- OS: Linux PC (x86_64, Ubuntu 22.04 or later recommended)
- **Python: 3.10 / 3.11 / 3.12** (note: v1.3.x no longer supports Python 3.8 as used in earlier tutorials; before installing in a Python 3.12 environment, run `export BUILD_CUDA_EXT=0`)
- SDK download: [RKLLM_SDK](https://console.zbox.filez.com/l/RJJDmB), fetch code `rkllm`

```bash
# Check whether miniforge3 is installed with conda -V; install it first if not
conda create -n RKLLM-Toolkit python=3.10
conda activate RKLLM-Toolkit
# Use the actual filename of the downloaded wheel
pip3 install rkllm_toolkit-1.3.1-cp310-cp310-linux_x86_64.whl
```

Verify the installation:

```bash
python
>>> from rkllm.api import RKLLM
>>> # No errors means the installation succeeded
```

### Generating Quantization Calibration Data

Taking DeepSeek-R1-Distill-Qwen-1.5B as an example, first clone the complete model repository from HuggingFace, then run:

```bash
cd rknn-llm/examples/rkllm_api_demo/export
python3 generate_data_quant.py -m ~/DeepSeek-R1-Distill-Qwen-1.5B
# Produces data_quant.json, formatted as: [{"input": "Human: Hello!\nAssistant: ", "target": "..."}]
```

The quality of the calibration data directly affects post-quantization accuracy, so use a corpus that matches the target application.

### Conversion and Export

Modify the key parameters in `export_rkllm.py` (the actual parameters of the current version are shown below):

```python
modelpath = '/path/to/DeepSeek-R1-Distill-Qwen-1.5B'
llm = RKLLM()

# Loading: device accepts 'cpu'/'cuda'; dtype accepts float32/float16/bfloat16
# GGUF is also supported: llm.load_gguf(model=modelpath)
ret = llm.load_huggingface(model=modelpath, model_lora=None,
                           device='cuda', dtype="float32",
                           custom_config=None, load_weight=True)

dataset = "./data_quant.json"
target_platform    = "RK3588"   # "RK3576" for RK3576
optimization_level = 0          # Official benchmarks are based on 0 (enables runtime optimization)
quantized_dtype    = "W8A8"     # or w4a16 / w4a16_g128 / w8a8_g256, etc.
quantized_algorithm = "normal"  # normal is recommended for w8a8; grq for w4a16
num_npu_core       = 3          # 3 cores for RK3588; use 2 for the dual-core RK3576

ret = llm.build(do_quantization=True, optimization_level=optimization_level,
                quantized_dtype=quantized_dtype, quantized_algorithm=quantized_algorithm,
                target_platform=target_platform, num_npu_core=num_npu_core,
                extra_qparams=None, dataset=dataset, hybrid_rate=0, max_context=4096)

ret = llm.export_rkllm(f"./{os.path.basename(modelpath)}_{quantized_dtype}_{target_platform}.rkllm")
```

Typical output of a successful conversion:

```
INFO: rkllm-toolkit version: 1.2.3
Building model: 100%|████████| 371/371 [00:05<00:00]
Optimizing model: 100%|████████| 28/28 [00:26<00:00]
WARNING: The bos token has two ids: 151646 and 151643, please ensure that the
         bos token ids in config.json and tokenizer_config.json are consistent!
INFO: Setting token_id of bos to 151646
INFO: Setting max_context_limit to 4096
INFO: Exporting the model, please wait ....
[=================================================>] 597/597 (100%)
INFO: Model has been saved to ./DeepSeek-R1-Distill-Qwen-1.5B_W8A8_RK3588.rkllm!
```

(The bos/eos token id WARNING is a common message with Qwen-family models; the Toolkit handles it automatically and no manual intervention is required.)

### Board-Side Environment Requirements

**NPU driver version ≥ v0.9.8**:

```bash
root@firefly:/# cat /sys/kernel/debug/rknpu/version
RKNPU driver: v0.9.8
```

If the version is lower than v0.9.8, download the latest firmware from the official Firefly firmware page and upgrade.

**Cross-compilation toolchain**: on Linux, [gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu](https://pan.baidu.com/s/1HOB492BcduU4LFHJM2-Y3w?pwd=1234) is recommended; on Android, use NDK r21e.

```bash
cd rknn-llm/examples/rkllm_api_demo/
# Set the cross-compiler path in build-linux.sh
GCC_COMPILER_PATH=~/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu
./build-linux.sh        # use ./build-android.sh on Android
```

### Transferring to the Board and Running

```bash
# Build output
adb push install/demo_Linux_aarch64 /data
# Converted model
adb push rknn-llm/examples/rkllm_api_demo/export/DeepSeek-R1-Distill-Qwen-1.5B_W8A8_RK3588.rkllm /data/demo_Linux_aarch64
# Frequency-setting script (locks the maximum frequency; mandatory before performance testing)
adb push rknn-llm/scripts/fix_freq_rk3588.sh /data/demo_Linux_aarch64
```

Run on the board:

```bash
adb shell
cd /data/demo_Linux_aarch64
export LD_LIBRARY_PATH=./lib
sh ./fix_freq_rk3588.sh       # Lock the maximum operating frequency
export RKLLM_LOG_LEVEL=1      # Emit inference performance and memory usage logs
./llm_demo ./DeepSeek-R1-Distill-Qwen-1.5B_W8A8_RK3588.rkllm 2048 4096
```

The interactive interface after a successful run (official Firefly record):

```
rkllm init start
rkllm init success

**********可输入以下问题对应序号获取回答/或自定义输入********

[0] 现有一笼子，里面有鸡和兔子若干只，数一数，共有头14个，腿38条，求鸡和兔子各有多少只？
[1] 有28位小朋友排成一行,从左边开始数第10位是学豆,从右边开始数他是第几位?

************************************************************

user:
```

(The prompt list built into the demo is in Chinese; arbitrary custom input is also accepted.)

> Note: with `RKLLM_LOG_LEVEL=1`, the logs report per-step inference latency and memory usage. Combined with `scripts/eval_perf_watch_cpu.sh` and `eval_perf_watch_npu.sh`, they also capture CPU/NPU utilization. This is the officially recommended standard method for performance measurement and is suitable for compiling performance reports.
