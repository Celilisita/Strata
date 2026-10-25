#pragma once
#include "attention_verify_online.hpp"
#include "service_profile.hpp"
namespace strata::completed_v100::qwen35::native {
template<int MinimumCapacity>
void verify_online_dispatch(const float* q,const void* k,const void* v,float* out,
    void* work,size_t bytes,PrefillAttentionShape shape,int parts,cudaStream_t stream){
    if(shape.tokens==3&&shape.capacity>=MinimumCapacity&&(shape.query_heads/shape.kv_heads)%6==0)
        verify_attention_online_q8(q,k,v,out,work,bytes,shape,parts,stream);
    else service_verify(q,k,v,out,work,bytes,shape,parts,stream);
}
}
