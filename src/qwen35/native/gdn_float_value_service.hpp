#include "gdn_long_chunk_policy.hpp"
#pragma once
#include "gdn_float_value.hpp"
#include "gdn_register_prepare.hpp"
#include "kreg_policy.hpp"
#include "strata/qwen35/native/ffn.hpp"
#include <cstdlib>
namespace strata::qwen35::native {
inline bool gdn_float_value_admitted(GdnShape s) {
    return s.key_dim==128 && s.value_dim==128 && s.query_heads==16 && s.value_heads==48 &&
           s.tokens==8192;
}
inline size_t service_gdn_float_workspace(GdnShape s) {
    return gdn_float_value_admitted(s) ? gdn_float_value_workspace(s) : gdn_vectorized_workspace_bytes(s);
}
inline void service_gdn_float_value(GdnInputs x,GdnShape s,float *out,float *state,
                                   void *scratch,size_t bytes,cudaStream_t stream) {
    if(gdn_float_value_admitted(s)){
        if(gdn_register_requested()&&use_gdn_kreg(s))service_gdn_kreg(x,s,out,state,scratch,bytes,stream);
        else if(gdn_register_requested())gdn_register_run(x,s,out,state,scratch,bytes,stream);
        else gdn_float_value_run(x,s,out,state,scratch,bytes,stream);
    }
    else gdn_recurrence_vectorized(x,s,out,state,scratch,bytes,stream);
}
inline bool use_gdn_float_value(int prompt) {
    const char *control=std::getenv("STRATA_DENSE_F32_U_GDN");
    if(!gdn_fast_request(prompt) || (control && control[0]=='0'))return false;
    int device=0;check_cuda(cudaGetDevice(&device),"F32 U GDN device");
    const auto caps=capabilities(device);
    return caps.major==12 && caps.minor==0;
}
} // namespace strata::qwen35::native
