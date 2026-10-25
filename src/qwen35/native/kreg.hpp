#pragma once
#include "gdn_register_prepare.hpp"
#include <cstdint>
#include <limits>
namespace strata::qwen35::native {
// Task-only K-register cache; the unchanged W/U/B/prefix layout borrows the existing decoded-weight arena.
constexpr size_t KregLeaseBytes=size_t(5120)*17408*4;
GdnRegisterResources kreg_prepare_resources();
void kreg_prepare_configure();
void kreg_prepare(GdnInputs,GdnShape,GdnFloatValueWork,cudaStream_t);
inline size_t kreg_workspace(GdnShape s){return gdn_float_value_workspace(s);}
inline bool kreg_overlap(const void *a,size_t an,const void *b,size_t bn){
    const auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);
    if(x>std::numeric_limits<uintptr_t>::max()-an||y>std::numeric_limits<uintptr_t>::max()-bn)
        throw std::invalid_argument("kreg pointer range overflow");
    return x<y+bn&&y<x+an;
}
inline void kreg_validate(GdnInputs x,GdnShape s,float *out,float *next,void *arena,size_t bytes){
    if(s.key_dim!=128||s.value_dim!=128||s.query_heads!=16||s.value_heads!=48||(s.tokens!=8192&&s.tokens!=5424))
        throw std::invalid_argument("kreg fixed synthetic actual-shape scope");
    const void *inputs[]={x.q,x.k,x.v,x.log_decay,x.beta,x.state};
    if(!arena||!out||!next)throw std::invalid_argument("kreg storage");
    if(reinterpret_cast<uintptr_t>(arena)%32||bytes<kreg_workspace(s)||bytes>KregLeaseBytes)
        throw std::invalid_argument("kreg exact borrowed lease capacity/alignment");
    const size_t q=size_t(s.tokens)*16*128*4,v=size_t(s.tokens)*48*128*4,g=size_t(s.tokens)*48*4,st=size_t(48)*128*128*4;
    const size_t counts[]={q,q,v,g,g,st};
    for(int i=0;i<6;++i){
        if(!inputs[i]||reinterpret_cast<uintptr_t>(inputs[i])%4)throw std::invalid_argument("kreg input alignment/storage");
        if(kreg_overlap(arena,bytes,inputs[i],counts[i])||kreg_overlap(out,v,inputs[i],counts[i]))
            throw std::invalid_argument("kreg live input arena/output alias");
        if(kreg_overlap(next,st,inputs[i],counts[i])&&!(i==5&&next==x.state))
            throw std::invalid_argument("kreg final-state partial/live-input alias");
    }
    if(reinterpret_cast<uintptr_t>(out)%4||reinterpret_cast<uintptr_t>(next)%4||kreg_overlap(arena,bytes,out,v)||
       kreg_overlap(arena,bytes,next,st)||kreg_overlap(out,v,next,st))throw std::invalid_argument("kreg output/lease alias/alignment");
}
inline void kreg_run(GdnInputs x,GdnShape s,float *out,float *next,void *arena,size_t bytes,cudaStream_t stream){
    kreg_validate(x,s,out,next,arena,bytes);
    auto work=gdn_float_value_work(arena,s);
    kreg_prepare(x,s,work,stream);gdn_delta_half(x,s,work,out,next,stream);
}
}
