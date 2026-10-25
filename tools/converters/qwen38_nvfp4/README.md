# Qwen3.8 27B Uncensored Q4_K_M → NVFP4

本目录是独立的模型资产转换工具，不属于默认运行时构建目标。入口为
`convert_q4_k_m_to_nvfp4.py`。本次主线合并逐字节保留冻结发布包的全部十个
实现文件，只新增本路径说明；未执行转换、CUDA 编译、模型加载或推理。
新参数化入口此前只有静态检查资格，不能把历史转换成绩当成它的新实测。

`README.txt`、`conversion-provenance.json`、`conversion-dependencies.json`
保留原发布包说明、真实来源和历史身份。其中 `converters/` 路径指原发布包，
在当前主线中改用 `tools/converters/qwen38_nvfp4/`。原说明中的任务路径是历史
证据，当前工具通过显式参数接收输入、输出与工作目录。

## 职责与固定数学

| 模块 | 职责 |
| --- | --- |
| `convert_q4_k_m_to_nvfp4.py` | 参数检查、工作目录约束、按顺序调用主转换与 head 转换 |
| `main_conversion.py` | 主层矩阵选择、metadata 原记录复制、未转换 tensor 原字节复制 |
| `head_conversion.py` | 独立三 tensor head；CPU 两遍流式处理，全 head 唯一 global |
| `quantize_nvfp4.py` | Q4_K/Q6_K 解码、三邻域 scale 搜索、E2M1 RNE 与磁盘打包 |
| `gguf_reader.py`、`conversion_io.py` | GGUF 目录、extent、身份、独占发布与 receipt |
| `gpu_conversion.py`、`gpu/` | 可选主矩阵 GPU 后端与完整、固定版本的 codec/include 闭包 |

主转换只处理主层 `blk.i.*` 的二维 Q4_K/Q6_K 矩阵，要求 `i < block_count -
nextn_predict_layers` 且 GGUF `shape[1] >= 256`。已记录输入对应 400 个矩阵。
embedding、原输出 head、MTP、norm、convolution 和其他 tensor 复制原字节。
每个矩阵一个 F32 global = `amax / (6 * 448)`（全零时为 1）；每 16 个数按
中心、中心−1、中心+1 三个合法 E4M3 code 重新编码并比较实际解码 SSE，严格
更小时替换，E2M1 使用 round-to-nearest-even。没有 SF127 穷举或校准变更。

独立 head 从主转换保留的 Q6_K `output.weight [5120,248320]` 产生。
第一遍计算整个 head 唯一 global 并 hash，第二遍固定 global 进行相同量化，
两遍源 hash 必须一致。输出按 `output.weight`、`output.scale`、
`output_norm.weight` 排列，norm 复制原字节。

磁盘 type40 布局每 64 值 36 字节（4 个 block16 scale 字节 + 32 个 packed
E2M1 字节），另有独立 F32 global。运行时的 layout/repack 读取由 Dense
后端维护；本目录不会把设备 repack 当成再次量化，也不重复实现运行时权重
所有权或 MoE 专家目录。

## 显式调用

当前已发布资产位于 `L:\models\Qwen3.8-27B-Uncensored-NVFP4-Strata`，主模型
为 `Qwen3.8-27B-Uncensored-Q4_K_M-to-NVFP4.gguf`，独立 head 为
`Qwen3.8-27B-Uncensored-Head-NVFP4.gguf`。本次合并复用它们，不重新转换或
复制权重。现有 223 冷 head 绑定有来源与几何约束，新转换完成不自动获得
运行时质量与性能资格，不可伪填旧 SHA 绕过检查。

下列命令仅作后续显式重建说明。输出使用全新 `-Rebuilt` 目录，不可覆盖已
发布资产；入口拒绝覆盖已有输出/receipt 不完整或身份不匹配的文件。

CPU 需要 64 位 Windows Python 和 NumPy。先将进程 TEMP/TMP 与工作目录
设到任务目录，再调用入口（PowerShell 示例，参数均为绝对路径）：

```powershell
$taskRoot = 'Y:\temp\qwen38-uncensored-conversion'
New-Item -ItemType Directory -Force $taskRoot | Out-Null
$env:TEMP = $taskRoot
$env:TMP = $taskRoot
Set-Location $taskRoot
python -B Y:\Strata_moe\tools\converters\qwen38_nvfp4\convert_q4_k_m_to_nvfp4.py `
  --source H:\LLM\JonathanColetti\Qwen3.8-27B-Uncensored-GGUF\Qwen3.8-27B-Uncensored-Q4_K_M.gguf `
  --output L:\models\Qwen3.8-27B-Uncensored-NVFP4-Strata-Rebuilt\Qwen3.8-27B-Uncensored-Q4_K_M-to-NVFP4.gguf `
  --head-output L:\models\Qwen3.8-27B-Uncensored-NVFP4-Strata-Rebuilt\Qwen3.8-27B-Uncensored-Head-NVFP4.gguf `
  --work-dir $taskRoot --backend cpu
```

GPU 仅替换主矩阵后端，head 仍使用 CPU 两遍。需 SM120、匹配的 CUDA 运行库
及显式构建的 DLL；历史工具链为 CUDA 13.0 + MSVC x64 v14.38。不会隐式
编译、切换其他算法或启动测试。每个矩阵至多 200M 元素，分配后保留 4 GiB
GPU 余量。`sfp4_convert`/`sfp4_error` 是转换 DLL 的独立 ABI，不链接模型
runtime、ggml、llama 或服务程序。

在已初始化的 x64 MSVC/CUDA 命令环境中，以下构建配方将产物与编译临时文件
留在 `Y:\temp`；本次没有执行它：

```powershell
$converterRoot = 'Y:\Strata_moe\tools\converters\qwen38_nvfp4'
$taskRoot = 'Y:\temp\qwen38-uncensored-conversion'
New-Item -ItemType Directory -Force $taskRoot | Out-Null
$env:TEMP = $taskRoot
$env:TMP = $taskRoot
$env:CUDA_CACHE_PATH = "$taskRoot\cuda-cache"
Set-Location $taskRoot
nvcc -std=c++17 -O3 -arch=sm_120 --shared --cudart=shared `
  -Xcompiler=/MD,/EHsc,/utf-8,/DNOMINMAX -Xptxas=-v `
  "-I$converterRoot\gpu\include" "-I$converterRoot\gpu" `
  "$converterRoot\gpu\convert_gpu.cu" -o "$taskRoot\convert_gpu.dll"
```

然后使用同样的完整转换命令，将末尾参数改为
`--backend gpu --gpu-library Y:\temp\qwen38-uncensored-conversion\convert_gpu.dll`。
主线默认 CMake 不构建此 DLL；它是用户显式转换时的可选工具依赖。

所有 `.partial`、状态、receipt 暂存、cache 都在 `--work-dir` 下，程序在导入
NumPy/CUDA 前将 TEMP/TMP/cwd 约束到 `Y:\temp`。`L:\models` 是持久输出，
不作临时目录。跨卷发布采用独占复制、fsync、完整 SHA 校验，成功后才发布
完成 receipt；失败保留工作文件供明确恢复操作。此工具不修改 router、启动
入口或生产配置。
