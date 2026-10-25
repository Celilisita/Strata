#pragma once
#include "strata/prefill/native_grouped_mmq.hpp"
#include "strata/prefill/moe_mmq.hpp"
#ifdef STRATA_BLACKWELL_MMQ
#include "strata/prefill/native_blackwell_mmq.hpp"
#endif
#include "native_grouped_mmq_plan.hpp"
#include "strata/prefill/native_code_sum_support.hpp"
#include "native_ordinary_grouped_mmq.hpp"
#include "native_down_direct_packed.hpp"
#include "native_gu_warp_repack.hpp"
#include <limits>
#include <cuda_runtime.h>
#include <array>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
namespace strata::prefill::mmq {
namespace native_grouped_detail {
constexpr size_t control_capacity = 256 * sizeof(NativeMmqMatrix) + 65536 * 8 + 1280 * sizeof(NativeMmqTile);
inline void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
inline bool ordinary_enabled() {
    const char* flag=std::getenv("STRATA_QWEN35_PP_GROUPED_ORDINARY_MMA");
    if(!flag||std::strcmp(flag,"0")==0)return false;
    if(std::strcmp(flag,"1")!=0)throw std::invalid_argument("PP_GROUPED_ORDINARY_MMA must be0 or1");
    return true;
}
inline bool q5_down_enabled() {
    const char* flag=std::getenv("STRATA_QWEN35_PP_Q5_DOWN_MMA");
    if(!flag||std::strcmp(flag,"0")==0)return false;
    if(std::strcmp(flag,"1")!=0)throw std::invalid_argument("PP_Q5_DOWN_MMA must be0 or1");
    return true;
}
struct Lease {
    uint8_t* bytes = nullptr;
    cudaEvent_t done{};
    bool used = false;
    Lease() {
        ck(cudaMallocHost(reinterpret_cast<void**>(&bytes), control_capacity));
        try { ck(cudaEventCreateWithFlags(&done, cudaEventDisableTiming)); }
        catch (...) { cudaFreeHost(bytes); throw; }
    }
    ~Lease() {
        if (used) cudaEventSynchronize(done);
        cudaEventDestroy(done); cudaFreeHost(bytes);
    }
};
}
using namespace native_grouped_detail;
struct NativeGroupedMmqProvider::Impl {
    const bool gu_warp;
    cudaStream_t stream;
    std::array<Lease, 4> leases;
    size_t next = 0;
    bool active_tiles = false;
    bool gu_int8 = false, gu_i32 = false, gu_i64 = false, down_int8 = false, down_fused = false;
    explicit Impl(void* s) : gu_warp(gu_warp_repack_flag(std::getenv("STRATA_QWEN35_PP_GU_WARP_REPACK"))), stream(static_cast<cudaStream_t>(s)) {
        if (!stream) throw std::invalid_argument("grouped MMQ requires explicit stream");
        if (const char* raw=std::getenv("STRATA_QWEN35_GU_GROUP_INT8")) {
            if (std::strcmp(raw,"0") && std::strcmp(raw,"1"))
                throw std::invalid_argument("GU group INT8 requires0 or1");
            gu_int8=!std::strcmp(raw,"1");
        }
        if (const char* raw=std::getenv("STRATA_QWEN35_GU_GROUP_INT8_I32")) {
            if (std::strcmp(raw,"0") && std::strcmp(raw,"1"))
                throw std::invalid_argument("GU I32 requires0 or1");
            gu_i32=!std::strcmp(raw,"1");
        }
        if (const char* raw=std::getenv("STRATA_QWEN35_GU_GROUP_INT8_I64")) {
            if (std::strcmp(raw,"0") && std::strcmp(raw,"1"))
                throw std::invalid_argument("GU I64 requires0 or1");
            gu_i64=!std::strcmp(raw,"1");
        }
        if (const char* raw=std::getenv("STRATA_QWEN35_DOWN_GROUP_INT8")) {
            if (std::strcmp(raw,"0") && std::strcmp(raw,"1"))
                throw std::invalid_argument("Down group INT8 requires0 or1");
            down_int8=!std::strcmp(raw,"1");
        }
        if (const char* raw=std::getenv("STRATA_QWEN35_DOWN_INPUT_FUSED")) down_fused=!std::strcmp(raw,"1");
        if (const char* s = std::getenv("STRATA_NATIVE_MMQ_ACTIVE_TILES")) {
            if (*s && std::strcmp(s,"0") && std::strcmp(s,"1"))
                throw std::invalid_argument("STRATA_NATIVE_MMQ_ACTIVE_TILES must be0 or1");
            active_tiles = !std::strcmp(s,"1");
        }
    }
};
} // namespace strata::prefill::mmq
