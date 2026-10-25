#pragma once
#include "attention_online_t3.hpp"
#include "service_profile.hpp"
#include <cstdint>
namespace strata::completed_v100::qwen35::native::q36_online {
struct Enqueues { uint64_t online=0,parent=0; };
// One Dense service slot/host command owner. Counts host eager/capture enqueues;
// Graph replay does not re-enter this dispatch and is counted by NativeMtp.
inline Enqueues enqueues;
inline void dispatch(const float* q,const void* k,const void* v,float* out,
    void* work,size_t bytes,PrefillAttentionShape s,int parts,cudaStream_t stream){
    if(s.q8&&s.dim==256&&s.query_heads==24&&s.kv_heads==4&&s.tokens==3){
        ++enqueues.online;attention(q,k,v,out,work,bytes,s,parts,stream);
    }else{
        ++enqueues.parent;service_verify(q,k,v,out,work,bytes,s,parts,stream);
    }
}
inline bool install(DenseModel& model,bool enabled){
    enqueues={};
    if(enabled){
        if(!model.q8_cache()||model.kv_window()!=0||model.kv_resident()!=0)
            throw std::invalid_argument("Q36online403 requires complete contiguous GPU Q8 history");
        configure(); // no hot allocation/copy/sync or provider lifetime mutation
        model.attention_work().use_verify_attention_kernel(dispatch);
    }else model.attention_work().use_verify_attention_kernel(service_verify);
    return enabled;
}
}
