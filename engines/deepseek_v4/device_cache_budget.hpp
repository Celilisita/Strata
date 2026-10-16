#pragma once
#include "ggml-backend.h"
#include <cstdint>
namespace strata::deepseek_v4 {
struct DeviceMemoryBudget {
    uint64_t cuda_free=0,cuda_total=0,local_budget=0,local_usage=0,nonlocal_usage=0;
    bool wddm_available=false;
};
DeviceMemoryBudget device_memory_budget(ggml_backend_t);
void trace_device_memory_budget(ggml_backend_t,const char* phase,int cache_slots);
int pp_cache_slots_with_headroom(int slots);
}
