MoE NVFP4 converter: Qwen3.6-35B-A3B / MiniCPM-V4.7-35B-A3B

This is an offline weight conversion tool. It does not load a model/runtime,
start a server, build CUDA automatically, or qualify model quality/performance.
It does not modify the original GGUF or another chat's converter package.

Input contract
  general.architecture = qwen35moe
  block_count-nextn_predict_layers =40, expert_count256
  embedding_length2048, expert_feed_forward_length512
  Qwen has41 blocks/nextN1; Mini has40 main blocks. MTP stays unchanged.
  Exactly the120 main-stack tensors matching
    blk.i.ffn_{gate,up,down}_exps.weight, i=0..39
  convert from original GGUF type12/13/14. GU shape[2048,512,256],
  Down shape[512,2048,256]. The original block formats are decoded to F32;
  this does not recover unquantized BF16 weights.

Output contract, version1
  Each selected tensor keeps its original name/shape and becomes GGUF type40.
  ExpertId is original dimension2; its matrix remains contiguous in the payload.
  One expert has1048576 values and589824 NVFP4 bytes.
  Every64 values occupy36 bytes: first4 bytes are four positive finite E4M3
  scales, then32 bytes are E2M1 data. Each16-value group stores first8 values
  in low nibbles and next8 in high nibbles, not adjacent-element nibble pairs.
  Reconstructed weight = E2M1(code) * E4M3(block16Scale) * F32(global[expert]).
  Every expert has its own global =maxabs(original decoded expert)/(6*448),
  or1 for an all-zero expert. There is no shared global across256 experts.
  The companion tensor name is original name without '.weight' plus '.scale':
    blk.0.ffn_gate_exps.scale, F32 shape[256], global[expertId].
  This companion lives in the SAME GGUF. The JSON receipt is diagnostic,
  never the only copy of a scale needed to interpret the payload.
  Per-block quantization preserves the existing27B three-neighbor E4M3
  search and E2M1 round-to-nearest-even algorithm. It is not SF127 search.

Metadata
  strata.moe.nvfp4.version           U32=1
  strata.moe.nvfp4.scale_axis        U32=2
  strata.moe.nvfp4.scope             STRING=main_routed_experts
  strata.moe.nvfp4.scale_suffix      STRING=.scale
  strata.moe.nvfp4.algorithm        STRING=the versioned converter algorithm
  strata.source.gguf.sha256         STRING=one actual full input hash
  strata.source.gguf.header_sha256  STRING=actual original header+padding hash
  strata.source.gguf.bytes          U64=actual source file size
  strata.source.gguf.name           STRING=actual source basename
  Original metadata records remain byte-identical except general.file_type
  is removed, because the output is mixed precision. Tokenizer/template remain.
  Every unselected tensor keeps its original type, shape and payload bytes,
  including embedding/head/shared FFN/norm/convolution and nextN/MTP.
  Offsets/alignment are recalculated. Generic GGUF readers must understand
  the separate per-expert global contract; the file extension alone is not
  a claim of compatibility with a runtime that ignores these companions.

Resource / identity contract
  CPU uses two bounded passes per expert with1024 K-block chunks.
  GPU reads/converts/writes one expert at a time, at most1048576 elements;
  temporary GPU buffers total about6MiB/expert and retain a4GiB free guard.
  No whole3D tensor or whole model is expanded to F32 in host RAM.
  Source gets one full SHA at the start, hashes while actual payloads are
  processed, and a final size/mtime check; no second full input hash.
  New output gets one full SHA. Same-volume publication uses an exclusive
  hardlink; unsupported/cross-volume publication uses exclusive copy+verify.
  Existing/incomplete output or receipt is never silently overwritten.
  Disk preflight allows complete partial plus copy fallback, about2*output.
  D:\temp stores work/cache/partials; D:\StrataModels stores final output.
  The receipt carries original/converted tensor and per-expert hashes,
  companion-scale hashes and measured quantization errors. It explicitly
  records runtime_qualified=false and full_model_loads=0.

Build only the independent converter DLL, no model build/load
  Include gpu/add_nvfp4_converter.cmake after CUDA is enabled, or the root's
  equivalent target. This directory's target is moe-nvfp4-convert, output
  moe_nvfp4_convert.dll. Only convert_moe_nvfp4.cu/source_codec.cuh compile;
  they need CUDA12.8 SM120 fp4/fp8 headers and shared cudart, no Strata/ggml.
  ABI: moe_nvfp4_abi_v1()==1; moe_nvfp4_convert_expert_v1 with a40-byte Result.
  The versioned name prevents accidentally loading the older Q4/Q6-only DLL.

Before full conversion, one real codec check, no model load
  python -B check_gpu.py --mini D:\StrataModels\Mini...gguf
    --qwen D:\StrataModels\Qwen...gguf --gpu-library D:\temp\...\moe_nvfp4_convert.dll
    --output D:\temp\...\expert-codec-check.json
  It reads Mini Q4 GU/Q5 Down/Q4 Down and Qwen Q6 Down expert0, checks all
  payload codes/SFs against CPU RNE and per-block minimum SSE, and records
  hashes/error. CPU/GPU tied SF choices are not forced to have equal bytes.
  The caller must require exit0 before the two full conversions.

Full conversion, explicit paths and backend
  python -B convert_moe_nvfp4.py --source D:\StrataModels\Mini...Q4_K_S.gguf
    --output D:\StrataModels\Mini-NVFP4-Strata\Mini...NVFP4.gguf
    --work-dir D:\temp\nvfp4-conversion-mini --backend gpu
    --gpu-library D:\temp\...\moe_nvfp4_convert.dll
  Use a separate work/output for Qwen; --backend cpu is an explicit reference
  alternative. This tool never changes to another backend after a CUDA error.
  New persistent assets are not runtime-ready until native operator, state,
  task quality and complete PP/TG gates pass under the intended resource budget.

Borrowed sources
  Codec search and streaming publication ideas derive from the local27B
  release-uncensored/converters package. The400 dense matrix selection,
  whole-matrix global scale, independent head conversion and Ytemp restriction
  are deliberately not used as MoE policy. Q5_K decode and per-expert I/O are new.
