Qwen3.8-27B uncensored：Q4_K_M → Strata NVFP4 转换说明

本说明配合本目录的新参数化工具及两个 provenance/dependency JSON 使用。
此轮整理只进行了文件读取、源码整理、清单/哈希检查；未运行新转换器、编译、
加载模型或推理。既有冻结转换器和既有 main/head 模型未被修改。

1. 真实来源与命名

原始输入记录：
H:\LLM\JonathanColetti\Qwen3.8-27B-Uncensored-GGUF\Qwen3.8-27B-Uncensored-Q4_K_M.gguf
字节数：16,810,714,528；历史 mtime_ns：1786812291364302100。
本地原始检查记录有866个 tensor，65个 block（包括1个 nextN/MTP block），
隐藏维5120、词表248320、context262144；GGUF general.name 为“Abl 27b”。
原文件名/目录明确含 Uncensored，这是本次交付 uncensored 标签的来源；
JonathanColetti 是记录中的目录命名空间，此处不将其推断成已认证的在线发布者。
原文件完整SHA已在远端仅文件读取时计算：
4c5e2db039e9325ac7724c8846c71356a24ad1cdfa28002d73ecb6be645f9675
证据为uncensored-model-files-result.json；不是从路径/mtime/目录哈希推算。

GGUF只是容器；此转换从真实 Q4_K/Q6_K 量化 payload 解码后再次量化，
并不能恢复未量化 BF16 原权重。它不是另行下载的 NVIDIA 官方 NVFP4 模型。

2. 应当一起保留的已采用资产

Main：原 Y:\temp\strata-27b-perf-20261018\native-fp4 下
Qwen3.8-27B-original-derived-NVFP4.gguf
  bytes 15740217984
  SHA256 0611180fd2368fa914739b4037435dad27cff76118b8036b0d770814a2079d46

Head：原同任务目录 head_fp4\head_fp4_fixture.gguf
  bytes 715183104
  SHA256 58a359c0c3b6f99ea67c31df507236c1c362867ce1c7e4ba18159cd40f7c5be0

两件合计16,455,401,088 bytes。Main中原 output.weight 仍是 Q6_K，
优化后的输出投影通过独立三 tensor NVFP4 head 覆盖它。
当前普通入口用 STRATA_NVFP4_HEAD_FILE 显式指定 head 路径。
需要独立 head 的原因是已采用的原始转换选择范围与已资格运行时绑定，
并非 GGUF 格式原则上只能分成两个文件。

已发布的实际目录与文件为：
L:\models\Qwen3.8-27B-Uncensored-NVFP4-Strata\
  Qwen3.8-27B-Uncensored-Q4_K_M-to-NVFP4.gguf
  Qwen3.8-27B-Uncensored-Head-NVFP4.gguf
两件目标完整SHA已核实；旧Y路径保留为symlink，不保留第二份大payload。
仅改变外部文件名/路径，不能为改名重写 GGUF metadata 或 tensor；这样保持
完整文件 SHA 不变，现有冷 head 验证仍能检查原始嵌入来源记录。
本工具不修改用户维护的 strata-router.py、启动入口或现有持久配置。

不要把 head-sf127/head-sf127-abi 当成交付 head。它们虽降低权重 SSE，
其实际原生长生成/TG 未取得采用资格，不能替换上面的58a359...已采用 head。
EAGLE 外部 draft 权重也不属于本转换资产。

3. Main tensor 选择：保留混合模型，而非把一切都重编码

冻结选择条件为：
  main_layers = qwen35.block_count - qwen35.nextn_predict_layers
  仅 blk.i.* 中 i < main_layers 的二维矩阵；
  原类型12(Q4_K)或14(Q6_K)，且原 GGUF shape[1] >= 256。
实际输入满足该条件的400个矩阵转换为 GGUF type40 NVFP4，
各自增加一个 tensor-name.scale 的F32全局尺度。

tokenizer、chat template及其余metadata保留；旧general.file_type删除以免
把混合输出继续误标为纯Q4_K_M。Embedding、原输出head、nextN/MTP、norm、
convolution和不满足阈值的小矩阵逐字节复制，不暗改其精度或数学。
转换输出使用 GGUF v3，并重新计算目录offset/alignment。
通用GGUF reader/runtime必须支持type40及本工具的独立全局scale约定；
不能仅凭文件后缀声称所有外部运行时都兼容。

4. 固定的量化数学与磁盘布局

每个转换矩阵先按原Q4_K/Q6_K格式解码到F32。
全矩阵仅一个F32 global：amax/(6*448)，全零时global=1。
每16个权重构成一组：先将 amax/(6*global) 就近编码为有限E4M3尺度，
依次评估中心、中心-1、中心+1三个合法code，范围0..126。
每个候选都重新编码E2M1值并计算真实解码SSE；只有严格更小才替换，
相同SSE保留先遇到的候选。E2M1采用round-to-nearest-even。

重建值 = E2M1(code) * E4M3(block-scale) * F32(global)。
每64个值占36字节：前4字节是4组E4M3尺度，后32字节为E2M1。
一组16值中，前8值占packed byte低nibble，后8值占对应高nibble。
另有每矩阵4字节F32 global，不包含在36字节block内。
磁盘有效权重负载约4.5 bits/value加每矩阵global；设备执行视图可无损重排，
但不应将设备repack当成再次重量化。

这套既有算法只搜索三个相邻SF，不是全127合法SF穷举或训练/校准。
本轮参数化整理应保持量化数学不变，不能暗中加入未采用的SF127候选。

5. 独立 head

head源来自Main中逐字节保留的 output.weight：Q6_K [5120,248320]，
共1,271,398,400值，原payload1,042,944,000字节。
原Q6 head payload SHA：
2155d6e1f7c5549077438c4760e6e150945dd5e406484b29162dff7b03d05da0

CPU两遍流式处理：第一遍求整个head唯一global并hash源，第二遍用固定global
执行同样的block16/三邻SF/E2M1规则；源head两遍hash必须相同。
输出只包含以下三tensor，并保持顺序：
  output.weight       [5120,248320] type40，715161600字节，offset0
  output.scale        [1]          F32，4字节，offset715161600
  output_norm.weight  [5120]       F32，20480字节，offset715161632
这些offset相对GGUF data-start；对已采用旧head，data-start为992。
output_norm从原模型逐字节复制。
已采用head packed payload SHA：
b3fb7bfff152ee5dc311701fe79450a752666190c75a19427be5630acff8fef3
旧head对原Q6解码值relativeL2约0.0862817348，是格式重编码误差记录，
不是声称恢复原BF16或推导生成接受率。

6. 依赖与资源

CPU：64-bit Python、NumPy、本目录GGUF reader及量化模块。
GPU主转换另需Windows x64、SM120、已显式构建的转换DLL及匹配的CUDA运行库。
历史环境为CUDA13.0、MSVC x64 v14.38；源依赖Strata codec.cuh/quant.hpp，
CUDA fp16/fp4/fp8头。GPU DLL每次只处理一个矩阵，最多200M元素，
检查原块+F32解码+新payload等临时申请后仍保留4GiB GPU余量。
它不需要加载整个模型，也不需要native service/路由器进程。

历史主模型转换已完成：400矩阵，conversion31.8039645秒
（仅主模型转换，不含CPU独立head转换或最终文件复制）。
独立head历史CPU转换另计约188.0436秒，不是此轮新工具的实测耗时。
层0三个FFN矩阵的完整CPU/GPU GGUF fixture SHA相同：
61607c8e836cf5b9535bf0ed0444315e320f2d3fa57beef08a8284555de1193e
该记录证明当时fixture的格式/payload一致，不应泛化成任意输入、CPU/GPU
所有SSE归约tie都保证同字节，更不要求跨框架自由生成IDs相同。

7. 路径、完成状态与兼容边界

所有任务TEMP/TMP、cache、chunks、.partial及工作日志放Y:\temp子目录。
L:\models为最终持久资产目的地，不能用它堆积临时文件。
新输出必须独占创建；先在工作目录完成、fsync、检查GGUF目录和tensor extent、
hash全文件，再发布并写完成receipt。Y盘到L盘跨卷复制不能假装是原子rename。
不可静默覆盖旧资产；失败partial由明确的恢复操作处理，不猜测为已完成。

当前223冷head绑定刻意限制已资格main/head几何和来源SHA。新通用转换器对
任意新输入应记录真实来源，不能伪填旧来源SHA以绕过运行时检查；
“转换文件完成”不等于“该新资产已经过完整原生质量/状态/性能验证”。
为兼容已知223绑定，只有完整main061118...、header289085...、原Q6head2155...
及normb3ba...四个SHA同时匹配时，新head工具才保留历史logical来源标签；
真实外部路径始终写在receipt，其余输入使用真实source.name，不伪造旧来源。
此次发布既有main/head只需文件操作/大小/哈希，不需要重新转换、编译或推理。

8. 新参数化命令（只作说明，此轮没有执行）

从交付根目录运行。入口明确指定source、main、head、work-dir，不能省略后
悄悄使用旧模型位置；GPU库也不自动编译。

以下为明确的-Rebuilt新目录重建示例，不是已发布资产路径，也没有执行。
不要直接运行来覆盖已交付模型；本轮交付复用既有main/head，不需要重建。

CPU完整转换示例（head同样为CPU，可能耗时较长）：
python -B converters/convert_q4_k_m_to_nvfp4.py --source "H:\LLM\JonathanColetti\Qwen3.8-27B-Uncensored-GGUF\Qwen3.8-27B-Uncensored-Q4_K_M.gguf" --output "L:\models\Qwen3.8-27B-Uncensored-NVFP4-Strata-Rebuilt\Qwen3.8-27B-Uncensored-Q4_K_M-to-NVFP4.gguf" --head-output "L:\models\Qwen3.8-27B-Uncensored-NVFP4-Strata-Rebuilt\Qwen3.8-27B-Uncensored-Head-NVFP4.gguf" --work-dir "Y:\temp\qwen38-uncensored-conversion" --backend cpu

GPU主矩阵转换示例（head仍固定CPU whole-head两遍，不换量化算法）：
python -B converters/convert_q4_k_m_to_nvfp4.py --source "H:\LLM\JonathanColetti\Qwen3.8-27B-Uncensored-GGUF\Qwen3.8-27B-Uncensored-Q4_K_M.gguf" --output "L:\models\Qwen3.8-27B-Uncensored-NVFP4-Strata-Rebuilt\Qwen3.8-27B-Uncensored-Q4_K_M-to-NVFP4.gguf" --head-output "L:\models\Qwen3.8-27B-Uncensored-NVFP4-Strata-Rebuilt\Qwen3.8-27B-Uncensored-Head-NVFP4.gguf" --work-dir "Y:\temp\qwen38-uncensored-conversion" --backend gpu --gpu-library "converters\bin\convert_gpu.dll"

--backend 只允许cpu/gpu。GPU库默认路径为converters/bin/convert_gpu.dll，
也可通过--gpu-library给出已显式构建且身份记录清楚的DLL。
没有该DLL或依赖CUDA运行库时，应明确失败，不自动改用别的量化实现。
上述-Rebuilt输出仅作新输出命名示例。已发布资产固定在
L:\models\Qwen3.8-27B-Uncensored-NVFP4-Strata，实际两文件名见第2节。
若目标文件已存在，不要重复执行转换或覆盖它，用发布receipt/大小/哈希核实。
新工具不会自动编译、运行算子测试、加载模型或推理；它只在用户显式调用时转换。

9. 证据文件与后续维护

conversion-provenance.json：来源记录、已采用main/head大小与SHA、三tensor目录、
原转换receipt证据文件SHA及uncensored命名依据。
conversion-dependencies.json：CPU/GPU最小依赖、固定数学/ABI、历史工具限制、
路径及发布约束。工具源码和最终发布资产的完整清单由release manifest记录。
SOURCE_README.txt和identity/source/说明当前223源码身份，和模型转换receipt是
不同的证据；二者均应保留，不能用一个GGUF hash代替运行时源码/二进制身份。

原始冻结脚本如单独归档到legacy，应保持其原字节；新参数化工具是独立入口，
去掉硬编码文件路径/mtime，保留原量化规则和不满足条件tensor的原字节复制。
本轮未验证新工具在真实权重上的输出或性能，不把AST/路径检查当成转换完成。
