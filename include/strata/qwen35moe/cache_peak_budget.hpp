#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace strata::qwen35moe {
inline bool cache_peak_flag(const char* value) {
    if(!value||!std::strcmp(value,"0"))return false;
    if(!std::strcmp(value,"1"))return true;
    throw std::invalid_argument("CACHE_PEAK requires0 or1");
}
inline bool cache_peak_enabled() {
    static const bool enabled=cache_peak_flag(std::getenv("STRATA_QWEN35_CACHE_PEAK"));
    return enabled;
}
// Exact current allocate_batch ownership for TC+scan+tile, large phase arena,
// without legacy expert_stage. Views own zero bytes; Gemm/providers are separate
// already-live allocations and are deliberately not credited by this policy.
inline size_t cache_peak_prepared_bytes(int capacity) {
    if(capacity<=1024||capacity>8192)throw std::invalid_argument("CACHE_PEAK large batch extent");
    return size_t(capacity)*171372+(size_t(6)<<20)+65536;
}
template<class Batch>
bool cache_peak_ready(const Batch& b,int desired,bool single,bool mtp,
    bool tc,bool scan,bool tile,bool healthy,bool sampler_ready,void* stream,bool nvfp4=false) {
    if(!healthy||!sampler_ready||!single||mtp||!tc||!scan||!tile||desired<=1024||desired>b.capacity||
       b.capacity<=1024||b.capacity>8192||b.expert_stage.data()||
       !b.gemm.scratch()||b.gemm.scratch_elems()!=int64_t(8192)*2048||b.gemm.stream()!=stream)
        return false;
    // Typed FP4 owns its host-only control leases and borrows the same
    // phase buffers. Its two old K-quant providers are intentionally absent.
    // Mixed/unknown providers retain the conservative reservation.
    if(nvfp4 ? bool(b.down_mmq)||bool(b.routed_tc) : !b.down_mmq||!b.routed_tc)
        return false;
    // Unknown future workspace layouts fail closed to the old reservation.
    return b.phase_arena.owned_bytes()==size_t(b.capacity)*28672*4&&
        b.owned_bytes()==cache_peak_prepared_bytes(b.capacity);
}
inline size_t cache_peak_future_bytes(int kv,size_t inventory,int logical) {
    if(kv<1||kv>logical||logical>196608||!inventory||inventory>(std::numeric_limits<size_t>::max)()/8)
        throw std::invalid_argument("CACHE_PEAK request/inventory extent");
    // One canonical full-prompt residual: count<=kv and embedding_width=2048.
    // This is NOT KV. reserve_context additional separately accounts actual KV
    // and attention scratch. Sampler/CUB must already exist for a GPU head.
    // Both possible history buffers may grow to logical context; existing bytes
    // are not credited. Retain the old 128MiB CUDA/allocator opaque guard.
    const size_t fixed=size_t(kv)*8192+size_t(logical)*8+(size_t(128)<<20);
    const size_t ages=inventory*8; // Full future dynamic ages; old bytes not credited.
    if(ages>(std::numeric_limits<size_t>::max)()-fixed)
        throw std::overflow_error("CACHE_PEAK reserve overflow");
    return fixed+ages;
}
inline bool cache_peak_reuse_workspace() {
    const char* value=std::getenv("STRATA_QWEN35_LONG_WORKSPACE");
    return !value||!std::strcmp(value,"reuse"); // Control owns two residuals: legacy policy.
}
} // namespace strata::qwen35moe
