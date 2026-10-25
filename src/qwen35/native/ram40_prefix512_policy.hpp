#pragma once
#include "ram40_triple_wave_policy.hpp"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strata::qwen35::native {
struct Ram40Prefix512Resources {
    int registers, blocks_per_sm, shared_bytes, local_bytes, binary_version;
};
Ram40Prefix512Resources ram40_prefix512_resources();
struct Ram40Prefix512Stats {
    bool requested=false, admitted=false, role_admitted=false;
    std::uint64_t host_calls=0, fullsort_fallback_calls=0;
    std::uint64_t epoch_bound=0, phase_epoch=0;
    std::uintptr_t owner=0;
    int last_tokens=0, last_pages=0;
    Ram40Prefix512Resources resource{};
};
inline Ram40Prefix512Stats& ram40_prefix512_state() {
    static thread_local Ram40Prefix512Stats s;
    return s;
}
inline Ram40Prefix512Stats ram40_prefix512_stats() {
    return ram40_prefix512_state();
}
inline bool ram40_prefix512_requested() {
    static const bool enabled=[] {
        const char* p=std::getenv("STRATA_DENSE_RAM40_PREFIX512");
        if(p&&std::strcmp(p,"0")&&std::strcmp(p,"1"))
            throw std::invalid_argument("RAM40 prefix512 flag must be 0 or 1");
        return p&&p[0]=='1';
    }();
    return enabled;
}
inline bool& ram40_prefix512_closed_call() {
    static thread_local bool closed=false;
    return closed;
}
// Only the closed AttentionLayer call without visual/query observers creates this lease.
// It lasts through enqueue/capture, and is not read by graph replay or GPU kernels.
class Ram40Prefix512ClosedScope {
    bool previous_;
public:
    explicit Ram40Prefix512ClosedScope(bool closed_no_observer)
        : previous_(ram40_prefix512_closed_call()) {
        ram40_prefix512_closed_call()=closed_no_observer;
    }
    ~Ram40Prefix512ClosedScope() { ram40_prefix512_closed_call()=previous_; }
    Ram40Prefix512ClosedScope(const Ram40Prefix512ClosedScope&)=delete;
    Ram40Prefix512ClosedScope& operator=(const Ram40Prefix512ClosedScope&)=delete;
};
// PREPARE runs after releasing old graphs. No resource query occurs during capture.
inline void configure_ram40_prefix512_request(bool allowed) {
    auto& s=ram40_prefix512_state();
    s={};
    s.requested=ram40_prefix512_requested();
    if(!s.requested||!allowed)return;
    int device=0;
    check_cuda(cudaGetDevice(&device),"prefix512 cold device");
    const auto c=capabilities(device);
    if(c.major!=12||c.minor!=0)return;
    s.resource=ram40_prefix512_resources();
    if(s.resource.binary_version!=120||s.resource.local_bytes||s.resource.blocks_per_sm<1)
        throw std::runtime_error("prefix512 resource gate");
    s.admitted=true;
}
inline bool use_ram40_prefix512(bool triple, RamKvView v, int tokens, int heads,
                               int parts, int shift, size_t bytes) {
    auto& s=ram40_prefix512_state();
    if(!s.admitted)return false;
    const auto t=ram40_triple_wave_stats();
    // The caller already ran both existing admission functions exactly once.
    // Reusing their owner/epoch receipt must not increment their counters again.
    const bool use=triple&&ram40_prefix512_closed_call()&&t.role_admitted&&
        t.owner&&t.phase_epoch&&t.last_tokens==3&&t.last_pages==2560&&
        tokens==3&&heads==24&&parts==64&&!shift&&bytes>=TripleBorrowedBytes&&
        v.pages==2560&&v.slots==2560&&v.resident==40960&&v.host_capacity==40960&&
        v.heads==4&&v.selected_tokens==8192&&v.selected_slots==512&&v.keep_queries==3&&
        v.key==v.host_key&&v.value==v.host_value;
    if(use) {
        s.role_admitted=true;
        ++s.host_calls;
        s.epoch_bound=t.epoch_bound;
        s.phase_epoch=t.phase_epoch;
        s.owner=t.owner;
        s.last_tokens=tokens;
        s.last_pages=v.pages;
    } else {
        ++s.fullsort_fallback_calls;
    }
    return use;
}
} // namespace strata::qwen35::native
