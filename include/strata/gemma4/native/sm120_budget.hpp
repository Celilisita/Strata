#pragma once
#include "strata/gemma4/native/sm120_projection_plan.hpp"
#include <cstddef>
namespace strata::gemma4::native {
struct Sm120Budget {
    std::size_t weights=0,encoded_input=0,provider=0,ffn=0,model_io=0;
    // Conservative selected-MMQ pool allowance, separate from exact owned buffers.
    std::size_t vendor_pool_bound=0;
    std::size_t static_owned=0,constructor_peak=0,kv=0,attention=0,request=0;
    std::size_t total_owned()const{return static_owned+request;}
};
Sm120Budget estimate_sm120_budget(const GgufFile&,const Spec&,int batch,int total=0);
// Additional allocation is compared with actual free GPU bytes and the existing
// reserve. This never shrinks batch, context, matrices or retained KV history.
void admit_sm120_budget(std::size_t additional,const char* phase);
}
