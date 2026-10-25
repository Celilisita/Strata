# Qwen MoE / MiniCPM 已验收源码维护说明

本目录整理的是 TITAN RTX 上已验收的 Qwen3.6-35B-A3B 与 MiniCPM-V-4.7-35B-A3B 原生 MoE 路径。
本次从最初任务快照到 P20 的完整变更是 95 份生产源码：58 新增、37 修改、无删除。
源码保留原相对路径、原始字节与 SHA256；不是仅复制最后 P15→P20 的 16 份文件。

## 目录职责

- `include/strata/qwen35moe/`：模型配置、布局、专家存储与指标接口。
- `src/qwen35moe/`：PP/TG 调度、专家目录、缓存、KV/状态及生成入口。
- `src/qwen35moe/cuda/`：attention 内核、选择条件及已有回退。
- `src/prefill/`、`include/strata/prefill/`：native Q8、grouped MMQ、GU/Down 打包及 provider。
- `src/kernels/cuda/native_*`、对应 `include/strata/kernels/`：原生量化、GDN 与 TG 融合定义。
- `cmake/strata_titanrtx.cmake`：唯一生产注册入口；在顶层既有 target 定义之后包含。
- `cmake/strata_titanrtx/`：按原有效顺序排列的 20 个生产注册/策略模块。

注册入口只包含生产内容，不导入历史 payload、DEFER 链、算子资格程序或基准程序。
没有 Qwen35MoE Runtime target 的配置会直接返回；有该 target 时需要原 CUDA 与 DeepSeek-v4 GGML 依赖。
这次整理沿用实际 P20 的依赖和 target 契约，没有扩大 CUDA/HIP/SYCL 后端兼容性声明。

## 需要保留的实现边界

Strata 的专家调度、目录、epoch/lease、内存所有权和状态管理保持原实现。
旧 EarlyRows64、Rows32 以及其他默认关闭的 provider 仍有选择或回退引用，不能按文件名视为废弃文件删除。
Qpad 的共享数学主体继续由宽上下文与 early TU 共用，不另复制计算实现。
Runtime attention CU 保留 CUDA20/no-fastmath；原 `attention_tc.cu` 保留其 CUDA17/fastmath 策略。
native MMQ/GDN CU 保留各自原 target/source 层面的选项与私有 GGML ABI，不在这里改全局 flags。

## 已验收的 MiniCPM P20 配置

以下四项是实际 MiniCPM 正常请求配置的选择值；本次整理没有改用户的运行配置或自动开启开关。

```text
STRATA_QWEN35_ATTN_ROWS64_QPAD8=1
STRATA_QWEN35_ATTN_ROWS64_EARLY=1
STRATA_QWEN35_PP_DOWN_DIRECT_PACKED=1
STRATA_QWEN35_PP_GU_WARP_REPACK=1
```

其余 provider、线程、采样及预算继承原验收配置：context 40960、slot 1、F16 KV、21000 MiB 显存预算、
64 GiB host reserve、4 个 native CPU worker 加 1 个 host 线程；每项任务独立 reset，按自然 EOS 生成。
三个纯计时变量 `STRATA_QWEN35_CUDA_EVENTS`、`STRATA_QWEN35_DENSE_EVENTS`、`STRATA_MINICPM_PP_PROFILE`
在正常请求验收中未设置；`STRATA_QWEN35_TG_PHASE_VERIFY=0`，必需的 key/epoch/lease/状态 guard 保留。

历史实际 P20 正常请求结果为：8192 tokens PP 2360.432104 tok/s（公平 llama 的 1.111489×），
30000 tokens PP 2140.540619 tok/s（1.137061×）；10/10 项质量及状态检查通过。
该证据对应原 P20 二进制与原构建注册。整理后 95 份源码逐字节相同；新的生产注册布局只做静态文件、
哈希与路径检查，没有运行 CMake、编译、模型加载或测试，因此不能把上述历史数字当作新主目录构建的重测结果。

## 整理与恢复

完整路径/旧新 SHA、适用基底、备份位置及交付 ZIP 身份记录在独立的 `code-cleanup-manifest.json`。
原始主目录被替换文件的备份与整理辅助文件保存在 `Y:\temp` 的本任务子目录，主源码树不放测试输入、
结果、临时脚本或历史源码副本。已验收的任务源码、旧构建目录与历史证据保留，便于追溯。
其他任务在 shared sampler、native35/dense 等路径的修改不属于本次变更，保持原样。
用户维护的 router、启动入口、运行配置与既有可执行文件保持原状。

今后另行构建时应使用这个生产注册入口，并清除该构建配置中旧的外部 `CMAKE_PROJECT_INCLUDE` payload 链，
避免同时注册两套入口。本次没有修改或重配置任何既有构建缓存。
