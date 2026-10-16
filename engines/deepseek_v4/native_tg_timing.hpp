#pragma once
#include "ggml-backend.h"
#include <cstdint>
namespace strata::deepseek_v4 {
struct NativeTgTimingResult { uint64_t layers=0, expert_ns=0, whole_gpu_ns=0, expert_compute_ns=0, join_ns=0; };
bool native_tg_timing_begin(ggml_backend_t);
bool native_tg_timing_read(ggml_backend_t,NativeTgTimingResult*);
}
