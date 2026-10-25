#pragma once
#include "attention_compensated_pp_k1024.hpp"

namespace strata::gemma4::native {
// Independent precision candidate: HH + 2^-12(HL + LH); LL is omitted.
// Same frozen324 encoding/layout/borrowed owner lease; no Model hook.
// Relative error has no universal bound under cancellation: real NRMS gate required.
void three_product_physical_pp_333_attention(const float*,float*,CacheView,const int*,
    int past,int heads,int tokens,AttentionWorkspace,CompensatedPpK1024Lease,
    CompensatedPpK1024Binding live);
}
