#pragma once
#include "ggml-backend.h"
#include <cstddef>
#include "device_cache_trace.hpp"
struct ggml_backend_cuda_context;
struct ggml_tensor;
bool strata_device_cache_lookup_cuda(ggml_backend_cuda_context&, ggml_tensor*);
bool strata_device_cache_guard_cuda(ggml_backend_cuda_context&, ggml_tensor*);
bool strata_device_cache_reserve_cuda(ggml_backend_t, size_t, bool);
bool strata_device_cache_trace_cuda(ggml_backend_t, strata::deepseek_v4::CacheGraphTrace*);
