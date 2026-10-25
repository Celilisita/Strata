#pragma once
#include <cstddef>
#include <cstdint>
#include <ostream>
namespace strata::kernels {
struct Top20PrepareFusionStats {
    bool requested=false,capable=false,admitted=false;
    uint64_t host_calls=0,fallback_calls=0;
    int last_vocabulary=0,last_top_k=0;
    int binary_version=0,blocks_per_sm=0,registers=0;size_t local_bytes=0;
};
// Process-lifetime accepted host eager/capture calls, never Graph replay counters.
Top20PrepareFusionStats top20_prepare_fusion_stats();
inline std::ostream &operator<<(std::ostream &out,const Top20PrepareFusionStats &state){
    return out<<" top20_prepare_fusion_requested="<<int(state.requested)
        <<" top20_prepare_fusion_capable="<<int(state.capable)
        <<" top20_prepare_fusion_admitted="<<int(state.admitted)
        <<" top20_prepare_fusion_host_calls="<<state.host_calls
        <<" top20_prepare_fusion_fallback_calls="<<state.fallback_calls
        <<" top20_prepare_fusion_last_vocabulary="<<state.last_vocabulary
        <<" top20_prepare_fusion_last_top_k="<<state.last_top_k
        <<" top20_prepare_fusion_binary_version="<<state.binary_version
        <<" top20_prepare_fusion_blocks_per_sm="<<state.blocks_per_sm
        <<" top20_prepare_fusion_registers="<<state.registers
        <<" top20_prepare_fusion_local_bytes="<<state.local_bytes;
}
}
