#pragma once
#include "ggml-backend.h"
#include <cstddef>
#include <cstdint>
#include "device_cache_trace.hpp"
struct ggml_backend_cuda_context;
struct ggml_tensor;
bool strata_device_cache_lookup_cuda(ggml_backend_cuda_context&, ggml_tensor*);
bool strata_device_cache_guard_cuda(ggml_backend_cuda_context&, ggml_tensor*);
bool strata_device_cache_reserve_cuda(ggml_backend_t, size_t, bool);
bool strata_device_cache_trace_cuda(ggml_backend_t, strata::deepseek_v4::CacheGraphTrace*);

bool strata_device_cache_hybrid_weight_cuda(ggml_backend_cuda_context&,ggml_tensor*);
// Explicit zero-model fixture only; never called by production lookup.
bool strata_glm53_probe_cold_gpu_copy(ggml_backend_t,uintptr_t,int32_t*,uint64_t,int);
bool strata_glm53_probe_cold_gpu_copy4(ggml_backend_t,uintptr_t,int32_t*,uint64_t,int);
