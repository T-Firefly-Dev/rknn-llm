# 基于 RKLLM SDK 的大语言模型部署指南（AIBOX-3588）

RK3588 是当前端侧大模型落地的主流平台之一。对开发人员而言，将 LLM 推理由云端迁移至边缘设备，主要基于以下三方面考量：

1. **数据本地化**。对话内容、图像与文档均在本地 NPU 上完成推理，可满足工业现场、内网办公、涉密环境等隐私合规场景的要求。RKLLM 官方 Quickstart 对此的定位即 "all processing happens locally on your device — your data never leaves it"。
2. **成本可控**。云端 API 按 token 计费，长期运行高频任务型推理（日志摘要、工单分类、OCR 后处理）时成本随调用量线性增长；边缘设备为一次性硬件投入，典型功耗仅 2.64W。
3. **生态成熟度**。RKLLM SDK 自 2024 年开源以来保持稳定迭代，主流模型家族（Qwen 全系、Gemma 全系、DeepSeek-R1-Distill 等）基本保持同步适配，并配套官方维护的预转换模型仓库与 OpenAI 兼容服务示例，集成成本较低。

## RKLLM 软件栈

三层结构如下：

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

- **RKLLM-Toolkit**：运行于 PC 端，负责模型转换与量化。支持 HuggingFace 格式加载（`load_huggingface`），亦支持 GGUF 格式（`load_gguf`，q4_0/fp16）。
- **RKLLM Runtime**：板端 C/C++ 推理库，通过预定义回调函数以流式方式返回推理结果；官方另提供封装好的 Flask 服务（OpenAI 兼容 API）与 Gradio Web UI。
- **RKNPU 驱动**：已并入 Rockchip 内核代码主线，板端仅需确认驱动版本。

## 模型生态

以下基于 v1.3.1 版本。

### 支持的模型家族

| 类别 | 模型 |
| --- | --- |
| 通用 LLM | Qwen2 / Qwen2.5 / **Qwen3 / Qwen3.5**、LLaMA、TinyLLaMA、Phi2/Phi3、ChatGLM3-6B、Gemma2 / **Gemma3 / Gemma3n / Gemma4**、InternLM2、MiniCPM3 / MiniCPM4、TeleChat2、DeepSeek-R1-Distill、Spark-X2.5、RWKV7 |
| 多模态 VLM | **Qwen2-VL / Qwen3-VL**、MiniCPM-V-2_6、Janus-Pro-1B、InternVL2-1B / InternVL3-1B、SmolVLM、SmolLM3、**LFM2.5-VL-450M** |
| OCR 专用 | **DeepSeekOCR**（3B，激活参数 570M） |

相较于社区早期教程普遍采用的 v1.1.x/v1.2.x 版本，当前版本新增了 Qwen3/Qwen3.5、Gemma3/3n/4、MiniCPM4、InternVL3、DeepSeekOCR、RWKV7 等模型支持，并引入多模态音频输入接口（Gemma4 支持图像与音频同时输入）。

### 官方性能数据（RK3588，w8a8）

测试条件：CPU/NPU 均锁定最高频率（官方定频脚本），`optimization_level=0` 转换，输入 128 token、生成 64 token。

| 模型 | 规模 | TTFT | 生成速度 | 运行内存 |
| --- | --- | --- | --- | --- |
| LFM2.5-VL | 450M | 0.10s | **62.2** tok/s | 437 MB |
| MiniCPM4 | 0.5B | 0.14s | 45.3 tok/s | 535 MB |
| Qwen2 | 0.5B | 0.15s | 41.6 tok/s | 670 MB |
| Qwen3 | 0.6B | 0.20s | 32.9 tok/s | 791 MB |
| **Qwen3.5** | **0.8B** | 0.59s | **27.1** tok/s | 1.0 GB |
| Qwen2.5 | 1.5B | 0.38s | 16.7 tok/s | 1.7 GB |
| Qwen3.5 | 2B | 0.78s | 13.6 tok/s | 2.1 GB |
| Qwen3-VL（多模态） | 2B | 0.38s | 15.0 tok/s | 1.9 GB |
| Gemma2 | 2B | 0.60s | 10.4 tok/s | 2.8 GB |
| Qwen3.5 | 4B | 2.07s | 6.2 tok/s | 4.6 GB |
| Phi3 | 3.8B | 1.02s | 7.5 tok/s | 3.8 GB |
| DeepSeekOCR | 3B(A570M) | 0.70s | 31.6 tok/s | 3.1 GB |
| ChatGLM3 | 6B | 1.35s | 5.0 tok/s | 6.0 GB |

多模态（VLM）完整链路数据（RK3588, w8a8）：

| 模型 | 视觉编码（448×448） | Prefill | Decode |
| --- | --- | --- | --- |
| Qwen3-VL-2B | 2.08s | 649ms | 14.9 tok/s |
| DeepSeekOCR-3B | 2.09s | 696ms | 31.8 tok/s |
| Qwen3.5-0.8B（视觉版） | 0.69s | 1.56s | 27.0 tok/s |
| Qwen2.5-VL-3B | 2.93s（392×392） | 1120ms | 8.7 tok/s |

## 快速验证

RK 官方维护了**预转换模型仓库 rkllm_model_zoo**，提供主流模型的 `.rkllm` 文件及配套 demo 可执行文件。无需在 PC 端搭建转换环境时，建议采用该路径：

1. 下载地址：[rkllm_model_zoo](https://console.box.lenovo.com/l/l0tXb8)，提取码 `rkllm`（quickstart 目录内含 demo 与模型）。
2. 推送至板端：

```bash
adb push ./demo_Linux_aarch64 /data
adb push model.rkllm /data/demo_Linux_aarch64
adb push model.rknn /data/demo_Linux_aarch64   # 多模态视觉编码器
```

3. 运行（以 Qwen3-VL-2B 图文问答为例）：

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

视觉编码器（`.rknn`，FP16）与语言模型（`.rkllm`）协同工作：图像先经 RKNN 编码为 token 序列，再交由 RKLLM 完成生成，全流程均在本地 NPU 上执行。图像/音频模态传入 `""` 即可禁用对应输入；`img_start` 等特殊 token 需依据模型源码中的 chat 实现核对（如 InternVL3 的 `modeling_internvl_chat.py`）。

## 自定义模型转换与部署

需要转换自行选定的模型或量化档位时采用该路径。整体流程：**PC 端标定 → 转换导出 → 交叉编译 → 板端运行**。

### RKLLM-Toolkit 安装

- 系统：Linux PC（x86_64，建议 Ubuntu 22.04 及以上）
- **Python：3.10 / 3.11 / 3.12**（注意：v1.3.x 已不再支持早期教程中的 Python 3.8；Python 3.12 环境安装前需执行 `export BUILD_CUDA_EXT=0`）
- SDK 下载：[RKLLM_SDK](https://console.zbox.filez.com/l/RJJDmB)，提取码 `rkllm`

```bash
# 通过 conda -V 确认 miniforge3 已安装，未安装则先完成安装
conda create -n RKLLM-Toolkit python=3.10
conda activate RKLLM-Toolkit
# whl 文件名以实际下载版本为准
pip3 install rkllm_toolkit-1.3.1-cp310-cp310-linux_x86_64.whl
```

安装验证：

```bash
python
>>> from rkllm.api import RKLLM
>>> # 无报错即安装成功
```

### 生成量化标定数据

以 DeepSeek-R1-Distill-Qwen-1.5B 为例，先从 HuggingFace 克隆完整模型仓库，随后执行：

```bash
cd rknn-llm/examples/rkllm_api_demo/export
python3 generate_data_quant.py -m ~/DeepSeek-R1-Distill-Qwen-1.5B
# 生成 data_quant.json，格式：[{"input": "Human: 你好！\nAssistant: ", "target": "..."}]
```

标定数据的质量直接影响量化后模型精度，建议使用与目标应用场景一致的语料。

### 转换与导出

修改 `export_rkllm.py` 中的关键参数（当前版本实际参数如下）：

```python
modelpath = '/path/to/DeepSeek-R1-Distill-Qwen-1.5B'
llm = RKLLM()

# 加载：device 可选 'cpu'/'cuda'；dtype 可选 float32/float16/bfloat16
# 亦支持 GGUF: llm.load_gguf(model=modelpath)
ret = llm.load_huggingface(model=modelpath, model_lora=None,
                           device='cuda', dtype="float32",
                           custom_config=None, load_weight=True)

dataset = "./data_quant.json"
target_platform    = "RK3588"   # RK3576 为 "RK3576"
optimization_level = 0          # 官方 benchmark 即基于 0（启用运行时优化）
quantized_dtype    = "W8A8"     # 或 w4a16 / w4a16_g128 / w8a8_g256 等
quantized_algorithm = "normal"  # w8a8 系推荐 normal；w4a16 系推荐 grq
num_npu_core       = 3          # RK3588 为 3 核；RK3576 双核填 2

ret = llm.build(do_quantization=True, optimization_level=optimization_level,
                quantized_dtype=quantized_dtype, quantized_algorithm=quantized_algorithm,
                target_platform=target_platform, num_npu_core=num_npu_core,
                extra_qparams=None, dataset=dataset, hybrid_rate=0, max_context=4096)

ret = llm.export_rkllm(f"./{os.path.basename(modelpath)}_{quantized_dtype}_{target_platform}.rkllm")
```

转换成功时的典型输出：

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

（bos/eos token id 的 WARNING 为 Qwen 系模型的常见提示，Toolkit 会自动处理，无需手动干预。）

### 板端环境要求

**NPU 驱动版本 ≥ v0.9.8**：

```bash
root@firefly:/# cat /sys/kernel/debug/rknpu/version
RKNPU driver: v0.9.8
```

若版本低于 v0.9.8，请前往 Firefly 官方固件页面下载最新固件进行升级。

**交叉编译工具链**：Linux 平台推荐 [gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu](https://pan.baidu.com/s/1HOB492BcduU4LFHJM2-Y3w?pwd=1234)；Android 平台使用 NDK r21e。

```bash
cd rknn-llm/examples/rkllm_api_demo/
# 修改 build-linux.sh 中的交叉编译器路径
GCC_COMPILER_PATH=~/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu
./build-linux.sh        # Android 平台执行 ./build-android.sh
```

### 推送至板端并运行

```bash
# 编译产物
adb push install/demo_Linux_aarch64 /data
# 转换完成的模型
adb push rknn-llm/examples/rkllm_api_demo/export/DeepSeek-R1-Distill-Qwen-1.5B_W8A8_RK3588.rkllm /data/demo_Linux_aarch64
# 定频脚本（锁定最高频率，性能测试前必须执行）
adb push rknn-llm/scripts/fix_freq_rk3588.sh /data/demo_Linux_aarch64
```

板端执行：

```bash
adb shell
cd /data/demo_Linux_aarch64
export LD_LIBRARY_PATH=./lib
sh ./fix_freq_rk3588.sh       # 锁定最高运行频率
export RKLLM_LOG_LEVEL=1      # 输出推理性能与内存占用日志
./llm_demo ./DeepSeek-R1-Distill-Qwen-1.5B_W8A8_RK3588.rkllm 2048 4096
```

运行成功后的交互界面（Firefly 官方记录）：

```
rkllm init start
rkllm init success

**********可输入以下问题对应序号获取回答/或自定义输入********

[0] 现有一笼子，里面有鸡和兔子若干只，数一数，共有头14个，腿38条，求鸡和兔子各有多少只？
[1] 有28位小朋友排成一行,从左边开始数第10位是学豆,从右边开始数他是第几位?

************************************************************

user:
```

> 说明：`RKLLM_LOG_LEVEL=1` 时日志将输出逐步推理耗时与内存占用，配合 `scripts/eval_perf_watch_cpu.sh` / `eval_perf_watch_npu.sh` 可采集 CPU/NPU 占用率。此为 RK 官方推荐的标准性能测量方法，适用于编制性能报告。
