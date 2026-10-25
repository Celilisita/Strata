# Titan V 小模型手动入口

这两个入口使用主代码 `F:\Codex\Strata_moe` 中新增的 `strata-dense-engine` target 和
`tools/dense_server.py`。从该 repo 的 `launch` 目录手动运行 BAT；启动时从 BAT 所在位置
解析 repo 根目录。共享 tokenizer、frontend 和 GGUF reader 保留主代码现有版本，
MiniCPM5 的 tokenizer 支持由可选的 `dense_tokenizer.py` 适配层提供。

## 当前状态

- **MiniCPM5-2B 是纯文本模型**。该入口不配置图像编码器，不接受图片。
- **Qwen3.5-0.8B 已完成本轮真实图像验证**：ENC 单图 283/256 rows、双图
  549/512 rows，以及单图回答、多图回答和图片后文本恢复均已运行。
  主要颜色、形状和相对位置正确，仍存在物体计数错误，具体限制见下文验证记录。
- 两入口默认启用 PP half cache、device greedy 和同输入投影合并。
  Qwen 额外启用固定页内存异步回读及仅用于预填充的 GDN 归一化融合；单 token 生成
  仍用原归一化路径。Mini 异步回读只测得约 0.6% 收益，未达到门槛，默认关闭。
  完整文本、HTTP 采样、取消/恢复和 Qwen 图像功能已经随该版本完成验证。
  **2× 性能目标尚未达到**；比较使用各项最快有效参考，并保留实际自然生成工作量。
- 每个服务只驻留一个 decoder、一个请求 slot，请求串行处理；Qwen 图像入口额外驻留
  一个编码器。入口提供 API，未包含 WebUI。
- TC attention 实验通过了 20 个独立算子数值/内存场景，但完整模型批次的端到端 PP 性能
  全面回退，因此实验实现已从主代码撤出，不作为交付路径，也不在 BAT 中启用。

## 手动启动

在 repo 根目录的 Windows 命令行中选择一个入口：

```bat
launch\START-MINICPM5-2B-TITAN-V.bat
```

```bat
launch\START-QWEN35-08B-VISION-TITAN-V.bat
```

默认地址均为 `http://127.0.0.1:8096`，两个入口使用同一个默认端口。
服务直接 bind 该地址；端口已占用时会报错退出。BAT 不下载、编译、自动注册服务或停止
其他进程。退出当前手动服务使用该终端的 Ctrl+C，子进程由本服务清理。

启动前需已有 Python 3.11+、`regex`、`jinja2`，以及本批匹配的 engine、GGML DLL 和
Qwen 视觉 helper。默认构建位置如下；DLL 需保留在对应 exe 附近。BAT 为当前进程
显式加入 CUDA 12.8 的 `bin` 和 `G:\llama.cpp\build\bin\Release` 到 PATH。

| 项目 | 默认值 |
|---|---|
| GPU UUID | `GPU-4d92f8d5-d57a-68de-67a4-6a0cfa30844c` |
| native build | `G:\temp\strata-minicpm-v47-titan-20261014\small-dense\build-main-final` |
| vision build | `G:\temp\strata-minicpm-v47-titan-20261014\small-dense\build-dense-vision` |
| decoder exe（相对 native build） | `Release\strata-dense-engine.exe` |
| encoder exe（相对 vision build） | `bin\Release\strata-dense-vision.exe` |
| 总 decoder 预算 | `12000 MiB`，不是额外缓存预算 |
| MiniCPM5 context / 输出上限 | `131072 / 65536` tokens |
| Qwen context / 输出上限 | `262144 / 65536` tokens |
| PP chunk | `512` |
| Qwen 每图 token 上限 | `1024` |

### 本批构建依赖身份

已核实 `G:\llama.cpp\build\bin\Release` 中实际 DLL 为 **b10656 / 732707dff**；
`G:\llama.cpp` 的 74ce157 checkout 头文件和原 import library 已陈旧。
本批使用任务内匹配头文件及从实际 DLL exports 生成的 import library。
以下 `$run` 为 `G:\temp\strata-minicpm-v47-titan-20261014\small-dense`：

| 构建参数 | 本批匹配路径或值 |
|---|---|
| native `STRATA_DENSE_GGML_SOURCE` | `$run\llama-headers-732707dff\ggml` |
| native `STRATA_DENSE_GGML_LIB_DIR` | `$run\backend-imports-732707dff\ggml\src` |
| native `STRATA_DENSE_GGML_BIN_DIR` | `G:\llama.cpp\build\bin\Release` |
| helper `STRATA_LLAMA_SOURCE` | `$run\llama-headers-732707dff` |
| helper `STRATA_LLAMA_BUILD` | `$run\backend-imports-732707dff` |
| helper `STRATA_LLAMA_RUNTIME_DIR` | `G:\llama.cpp\build\bin\Release` |
| helper `STRATA_VISION_LINKAGE` | `SHARED` |

BAT 只运行已构建的 exe；替换构建目录时，也须保留上述 headers、import libraries 和
DLL 的同源身份。

## 覆盖路径和配置

BAT 读取以下环境变量，未设置时使用默认值。构建目录可指向原主代码的现有构建目录，
无需复制源码或改动旧 35B 启动入口。

| 环境变量 | 含义 |
|---|---|
| `STRATA_DENSE_BUILD_DIR` | native build，含 `Release\strata-dense-engine.exe` |
| `STRATA_DENSE_VISION_BUILD_DIR` | Qwen vision build，含 `bin\Release\strata-dense-vision.exe` |
| `STRATA_DENSE_LOG_DIR` | 当前模型的日志、图片工件及缓存根目录 |
| `STRATA_DENSE_PYTHON` | Python 可执行文件名称或完整路径，不附加命令行参数 |
| `STRATA_DENSE_PORT` | HTTP 端口，默认 `8096` |
| `STRATA_DENSE_CONTEXT` | 当前入口 context |
| `STRATA_DENSE_MAX_OUTPUT` | 每次请求输出上限；还需满足完整输入加输出不超 context |
| `STRATA_DENSE_PROJECTION_FUSION` | 同输入投影合并，默认 `1`；不重新量化或复制 GPU 权重 |
| `STRATA_DENSE_COMPACT_ASYNC` | 固定页 host packet 异步回读，Qwen 默认 `1`、Mini 默认 `0`；不可用时同步回退 |
| `STRATA_DENSE_NORM_FUSION` | 仅预填充的 GDN RMS/MUL 融合，Qwen 默认 `1`、Mini 默认 `0` |
| `STRATA_MINICPM5_MODEL` | 默认 `G:\LM_Models\openbmb\MiniCPM5-2B-GGUF\MiniCPM5-2B-Q4_K_M.gguf` |
| `STRATA_QWEN35_08B_MODEL` | 默认 `G:\LM_Models\unsloth\Qwen3.5-0.8B-GGUF\Qwen3.5-0.8B-Q8_0.gguf` |
| `STRATA_QWEN35_08B_MMPROJ` | 默认同 Qwen 模型目录的 `mmproj-F32.gguf` |

例如：

```bat
set "STRATA_DENSE_BUILD_DIR=G:\temp\strata-minicpm-v47-titan-20261014\small-dense\build-main-final"
set "STRATA_DENSE_PORT=8096"
launch\START-MINICPM5-2B-TITAN-V.bat
```

默认日志分别位于任务根目录下的 `small-dense\manual-minicpm5-2b` 和
`small-dense\manual-qwen35-08b-vision`。覆盖的 `STRATA_DENSE_LOG_DIR` 也必须位于
`G:\temp\strata-minicpm-v47-titan-20261014` 内，服务会校验。
BAT 将工作目录、TEMP/TMP、Python bytecode 和 CUDA cache 都放在该任务目录内。
引擎 stdout/stderr 和 HTTP 访问日志有界写入日志目录；前端启动信息显示在当前终端。

## 模型模板与 API

使用 `/health`、`/v1/models` 和 `/v1/chat/completions`。支持非流式和 SSE。
模型名分别为 `minicpm5-2b-strata`、`qwen3.5-0.8b-strata`。

MiniCPM5 保留实际模型模板的默认思考行为。Qwen3.5-0.8B 的实际模板默认不思考，
需要时在请求中显式开启；BAT 不向模板强制传入 `false`。

```json
{
  "model": "qwen3.5-0.8b-strata",
  "messages": [{"role": "user", "content": "计算17×19，给出结果。"}],
  "max_tokens": 512,
  "temperature": 0,
  "chat_template_kwargs": {"enable_thinking": true}
}
```

也支持顶层 `enable_thinking: true`；若同时提供两种写法，两者必须一致。
实际思考文本通过 `reasoning_content` 返回。非零 temperature 或重复惩罚使用完整采样路径，
由 native engine 决定是否使用 device greedy。

Qwen 图片请求使用 OpenAI `image_url` 内容项，支持 PNG/JPEG/BMP 的 base64 data URL，
或任务目录内的绝对文件路径；不读取远程 HTTP 图片 URL。多张图片按消息中的顺序处理。
真实图像编码、SDV/DVI 文件和 decoder 消耗均纳入请求过程。本轮验证包括实际图像回答，
输入、输出及质量限制按预定的图像检查记录，见下文。

## 最近一次验证（2026-10-21）

当前主代码版本已完成两模型的预定文本请求及 HTTP 采样、取消和恢复检查；
Qwen 还完成了真实单图、多图和图片后文本状态重置检查。每模型同次加载先做短答，
再在相同物理权重上运行 ABBA 对照，通过后继续完整任务，不抑制 EOS 或使用 teacher forcing。

投影合并将相同输入、相同量化类型的矩阵放到一个原始字节布局中，TG 用一次 matmul，
PP 继续使用原矩阵视图及 half cache。未重复驻留 GPU 权重。相对原稳定版本，投影阶段
同驻留 TG 中位提升 Mini 约 7.7%、Qwen 约 11.3%；完整任务结果单独记录。

贪心输出保留原 argmax 和全 logits 的 abs/sum 有限性检查，仍回传 8 字节。
异步路径将 graph 和两次 scalar D2H 放在同一 backend stream 上，最终一次等待后验证。
`compute_ms` 在异步模式表示提交耗时；比较完整执行应使用 `execute_readback_ms` 或 TG/请求耗时。
固定页 packet 类型经过实际核对，分配失败时同步回退；非贪心采样继续完整 logits 路径。

GDN 归一化在 512-token 的三个算子场景快约 22%，数值检查全部通过；单 token
场景未通过所有计时门槛，因此本次只选择预填充，生成路径保持原 SCALE。
常量新增实际 GPU 分配 512 B，已计入总预算。图计数表示融合资格，未声称直接观测每个 kernel。

图像主要颜色、形状和相对位置回答正确，第二张图片仍有物体计数错误。Mini 长文任务
仍存在重复思考与字数不足，Qwen 长文翻译仍有遗漏和语义错误；性能改善没有清除这些质量限制。
TC attention 实验的 PP 全面回退证据继续保留，实验实现已从主代码撤出。

当前尚未达到 2× 性能目标。早期参考测试遗漏了官方默认关闭、标为 experimental 的
`--backend-sampling` 开关，已有结果保留为 CPU 采样诊断基线。开启该开关的参考测试现已
完成，11 次请求的最终设置和源码适用条件均已核验；默认日志过滤了注册 INFO，逐 token
GPU 执行仍未直接观察，未记录 WARN 回退。最终比较保留每项最快的有效参考。
Nsight 插桩结果仅用于成本诊断，不作为正式端到端性能结果。

在普通完整 repo 构建中，使用 `-DSTRATA_BUILD_DENSE=ON` 加入 dense targets；
独立 dense 构建使用 `-DSTRATA_DENSE_ONLY=ON`。原有综合 vision 项目通过
`-DSTRATA_BUILD_DENSE_VISION=ON` 加入 dense helper；复用既有 backend 库的独立
helper 项目入口为 `tools/vision/dense`。

新增 dense targets 显式使用 C++17，兼容主项目的 C++20 默认标准；默认路径不启用这些 targets。
