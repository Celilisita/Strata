#pragma once
#include "critical_intervals.hpp"

namespace strata::glm5_next {
// One CTA per serial join invocation, not a sum across parallel CTAs. The
// observed poll lies within the complete join kernel and omits other CTAs,
// launch/scheduling delay, mapped contribution loads and weighted products.
// Thus it is a sampled residency lower bound, not isolated CPU compute time.
// GPU globaltimer can also span WDDM descheduling; no CPU-only attribution.
template<bool Enabled>
__device__ inline void record_critical_join(unsigned long long* counters,uint64_t elapsed) {
    if constexpr(Enabled) {
        if(blockIdx.x==0&&blockIdx.y==0&&blockIdx.z==0&&threadIdx.x==0) {
            atomicAdd(counters+critical_join_sum,static_cast<unsigned long long>(elapsed));
            atomicMax(counters+critical_join_max,static_cast<unsigned long long>(elapsed));
            atomicAdd(counters+critical_join_count,1ull);
        }
    }
}
} // namespace strata::glm5_next
