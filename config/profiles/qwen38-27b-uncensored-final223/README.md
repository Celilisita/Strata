# Qwen3.8-27B-Uncensored final223 source profile

This profile records the accepted SM120 slice integrated on 2026-10-21. It is an explicit source-level profile; it does not activate a service, select a production executable, or replace a launcher.

The public backend is `strata --family qwen35_dense --variant qwen38_27b_uncensored_nvfp4`. It owns one Dense model/weight inventory, the Native FP4 provider and separate head binding. The original `qwen38_27b` variant remains accepted. Dense weights are independent of the MoE expert directory.

`runtime-profile.json` preserves all 69 effective environment settings and five explicit unset keys from the accepted release, including the exact model/head identities. `native-config-fragment.json` adapts those settings to the common native family service. Complete the listed executable, tokenizer, frontend, device and ABI bindings using the newly built mainline binary before use. Its status deliberately remains build/runtime pending.

Use MTP2, temperature 1, top_k 20, top_p .95, logical context 262144, physical GPU rows 40960, selected rows 8192, keep queries 3 and reserve 4096 MiB. Complete RAM history and the original full-history fallback remain in the runtime. Physical rows do not redefine the model's logical context. The profile is scoped to SM120; it is not a V100 budget.

The service keeps normal max_new/EOS, penalty/RNG state, independent request reset and subsequent requests. Cancellation is cooperative at command boundaries; it does not preempt a running GPU prefill. Frozen service benchmark/canary output has been removed from the production protocol.

## Reproducible affected-target build (not executed during source merge)

Run from an existing authorized VS2022 x64 command environment using MSVC 14.38 and CUDA 13.0. All build and cache directories must be under `Y:\temp`. Preserve the current production build and dependency trees.

```bat
set TEMP=Y:\temp\strata-final223-merge-20261021
set TMP=%TEMP%
cd /d Y:\temp\strata-final223-merge-20261021
cmake -S Y:\Strata_moe -B Y:\temp\strata-final223-merge-20261021\build-dense -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared -DSTRATA_ENABLE_CUDA=ON -DSTRATA_QWEN35_DENSE=ON -DSTRATA_QWEN35_DENSE_FINAL223=ON -DSTRATA_QWEN35_MOE=OFF -DSTRATA_DEEPSEEK_V4=OFF -DSTRATA_BUILD_TESTS=OFF -DSTRATA_PRODUCTION_ONLY=ON -DSTRATA_CUTLASS_DIR=Y:\temp\strata-5090-unified-validation-20261017\deps\cutlass
cmake --build Y:\temp\strata-final223-merge-20261021\build-dense --target strata --parallel 2
```

This command isolates the affected public Dense target for a first integration build. A combined Flash/D4/MoE build must retain its existing ABI-matched reference dependencies and enable its own options; no source or default for those families is removed by this slice. If a pinned ggml source is needed, set `STRATA_GGML_DIR` to that existing identity-matched checkout, rather than allowing a different version into the dependency closure.

The CUDA 13/SM120 final223 option is explicit. Its native FP4 object library is compiled for 120a; the general Dense code uses the configured 120 architecture. Unsupported older-toolkit Dense builds fail configuration clearly instead of linking the historical incomplete `nvfp4_unavailable.cpp`. MoE-only builds retain their original sampler. Combined builds select a single short-prefix sampler implementation, one NVFP4 Runtime and one GEMM implementation per interface. No old private objects, archive substitutions or benchmark executable are production dependencies.

The optional model converter lives in `tools\converters\qwen38_nvfp4`. It is independent of default CMake targets and its optional CUDA DLL ABI is separate from the model runtime. Conversion does not belong to runtime layout loading.

## Validation boundary

Stage A performed file/hash/path, syntax, dependency, registration and ownership checks only. It did not compile, convert weights, load a model, run inference or measure performance. The historical PE SHA `75c6c07c79c38107e1db9ebba95b5393aa7600b7c6da7f0e6f5cf3e88a69a138` and its 8K/30K results describe the frozen pre-merge binary, never a new merged binary. GPU-round merge and EAGLE4 remained rejected. Production executables, router and launch entries were not replaced.
