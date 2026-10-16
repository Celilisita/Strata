#pragma once
#include "ggml-backend.h"
#include <cstdint>
#include <array>
#include <vector>
namespace strata::deepseek_v4 {
struct NativeRouteRecord {int32_t ids[6];uint32_t missing=0,epoch=0;};
static_assert(sizeof(NativeRouteRecord)==32);
struct NativePhaseClock {uint64_t publish=0,join_begin=0,cpu_ready=0,join_end=0;};
static_assert(sizeof(NativePhaseClock)==32);
struct NativeRouteTrace {uint32_t epochs=0;std::vector<NativeRouteRecord> records;std::vector<NativePhaseClock> phase_clocks;
    std::array<uint64_t,4> missing_unique_weight_buckets{};};
bool native_tg_trace_read(ggml_backend_t,NativeRouteTrace*,bool payload);
}
