#pragma once
#include <cstdlib>
#include <cstring>
namespace strata::deepseek_v4 {
// Native policy experiment for the large dense Q8 matrices in two-position
// MTP verification. The immutable quantized weights and model math stay fixed.
inline bool dense_pair_tensor_core(const ggml_tensor* weight,int64_t tokens,int cc) {
    if(tokens!=2||weight->type!=GGML_TYPE_Q8_0||weight->ne[0]<1024||weight->ne[1]<4096||
       !blackwell_mma_available(cc))return false;
    const char* enabled=std::getenv("STRATA_DSV4_DENSE_PAIR_MMQ");
    return enabled&&!std::strcmp(enabled,"1");
}
}
