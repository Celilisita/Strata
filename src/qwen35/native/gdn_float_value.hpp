#pragma once
#include "gdn_chunk.hpp"
#include "strata/qwen35/native/gdn_vectorized.hpp"
#include <stdexcept>
namespace strata::qwen35::native {
struct GdnFloatValueWork {__half *w;float *u;__half *b;float *prefix;};
void gdn_float_value_prepare_configure();
void gdn_float_value_scan_configure();
void gdn_float_value_prepare(GdnInputs,GdnShape,GdnFloatValueWork,cudaStream_t);
void gdn_float_value_scan(GdnInputs,GdnShape,GdnFloatValueWork,float*,float*,cudaStream_t);
inline size_t gdn_float_value_workspace(GdnShape s){
    return size_t((s.tokens+31)/32)*s.value_heads*(32*128*2+32*128*4+32*32*2+32*4);
}
inline GdnFloatValueWork gdn_float_value_work(void *storage,GdnShape s){
    const size_t blocks=size_t((s.tokens+31)/32)*s.value_heads;
    auto *w=static_cast<__half*>(storage);auto *u=reinterpret_cast<float*>(w+blocks*32*128);
    auto *b=reinterpret_cast<__half*>(u+blocks*32*128);auto *prefix=reinterpret_cast<float*>(b+blocks*32*32);
    return {w,u,b,prefix};
}
inline void gdn_float_value_run(GdnInputs x,GdnShape s,float *out,float *state,void *storage,size_t bytes,cudaStream_t stream){
    if(!x.q||!x.k||!x.v||!x.log_decay||!x.beta||!x.state||!out||!state||!storage||
       s.key_dim!=128||s.value_dim!=128||s.query_heads<1||s.value_heads<1||s.value_heads>256||
       s.value_heads%s.query_heads||s.tokens<1||s.tokens>8192)
        throw std::invalid_argument("F32value GDN geometry/operands");
    if(bytes<gdn_float_value_workspace(s))throw std::invalid_argument("F32value GDN workspace");
    auto work=gdn_float_value_work(storage,s);
    gdn_float_value_prepare(x,s,work,stream);gdn_float_value_scan(x,s,work,out,state,stream);
}
}
