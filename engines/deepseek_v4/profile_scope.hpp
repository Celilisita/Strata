#pragma once

#ifdef STRATA_DEEPSEEK_CUDA_PROFILER
#include "graph_diag.hpp"
#include <cuda_profiler_api.h>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#endif

#ifdef STRATA_DEEPSEEK_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

namespace strata::deepseek_v4 {

class ProfileScope {
public:
    explicit ProfileScope(const char* name, bool capture = true) {
#ifdef STRATA_DEEPSEEK_CUDA_PROFILER
        const char* enabled = std::getenv("STRATA_DSV4_PROFILE");
        active_ = capture && enabled && std::strcmp(enabled, "1") == 0;
        if (active_ && cudaProfilerStart() != cudaSuccess)
            throw std::runtime_error("request-scoped CUDA profiler start failed");
        graph_diag_start();
#endif
#ifndef STRATA_DEEPSEEK_CUDA_PROFILER
        (void) capture;
#endif
#ifdef STRATA_DEEPSEEK_NVTX
        nvtxRangePushA(name);
#else
        (void) name;
#endif
    }
    ~ProfileScope() {
#ifdef STRATA_DEEPSEEK_NVTX
        nvtxRangePop();
#endif
#ifdef STRATA_DEEPSEEK_CUDA_PROFILER
        graph_diag_end();
        if (active_) cudaProfilerStop();
#endif
    }
    ProfileScope(const ProfileScope&) = delete;
    ProfileScope& operator=(const ProfileScope&) = delete;
private:
    bool active_ = false;
};

} // namespace strata::deepseek_v4
