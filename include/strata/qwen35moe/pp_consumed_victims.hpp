#pragma once
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>
namespace strata::qwen35moe::pp_consumed_victims {
inline bool enabled() {
    const char* value=std::getenv("STRATA_QWEN35_PP_CONSUMED_VICTIMS");
    if(!value || !std::strcmp(value,"0")) return false;
    if(!std::strcmp(value,"1")) return true;
    throw std::invalid_argument("PP_CONSUMED_VICTIMS must be0 or1");
}
inline int rank(int32_t key,size_t layer) {
    if(key<0) return 0;
    return layer>=2 && size_t(key)/256<layer-1 ? 1 : 2;
}
// Eligibility is established by the caller: resident/current/previous stay
// protected. Only order existing legal victims; retain the original tie break.
struct Order {
    const std::vector<int32_t>& slots;
    const std::vector<uint64_t>& ages;
    size_t layer;
    bool consumed_first;
    bool operator()(size_t a,size_t b) const {
        if(consumed_first) {
            const int ar=rank(slots[a],layer),br=rank(slots[b],layer);
            if(ar!=br) return ar<br;
        }
        const uint64_t aa=slots[a]<0?0:ages[a],bb=slots[b]<0?0:ages[b];
        return aa!=bb ? aa<bb : a<b;
    }
};
} // namespace strata::qwen35moe::pp_consumed_victims
