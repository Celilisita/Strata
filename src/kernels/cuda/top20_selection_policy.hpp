#pragma once
#include <cstddef>
#include <cstdint>
#include <ostream>

namespace strata::kernels {
// Process-lifetime host submissions, including graph capture; never replay counts.
// Capability is queried once by the first sampler State constructor, outside capture.
struct Top20SelectionStats {
    bool requested=false,capable=false,admitted=false;
    uint64_t host_calls=0,fallback_calls=0;
    int last_vocabulary=0,last_top_k=0;
    int binary_version=0,blocks_per_sm=0,registers=0;
    size_t local_bytes=0;
};
Top20SelectionStats top20_selection_stats();
inline std::ostream& operator<<(std::ostream& out,const Top20SelectionStats& s) {
    return out<<" top20_selection_requested="<<int(s.requested)
        <<" top20_selection_capable="<<int(s.capable)
        <<" top20_selection_admitted="<<int(s.admitted)
        <<" top20_selection_host_calls="<<s.host_calls
        <<" top20_selection_fallback_calls="<<s.fallback_calls
        <<" top20_selection_last_vocabulary="<<s.last_vocabulary
        <<" top20_selection_last_top_k="<<s.last_top_k
        <<" top20_selection_binary_version="<<s.binary_version
        <<" top20_selection_blocks_per_sm="<<s.blocks_per_sm
        <<" top20_selection_registers="<<s.registers
        <<" top20_selection_local_bytes="<<s.local_bytes;
}
} // namespace strata::kernels
