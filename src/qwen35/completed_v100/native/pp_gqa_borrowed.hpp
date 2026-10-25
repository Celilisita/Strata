#pragma once
#include "pp_gqa.hpp"
namespace strata::completed_v100::qwen35::native {
// A per-service adapter using the existing borrowed DenseGemm lease.
// It adds no fields to AttentionLayer and owns no model/KV/weight state.
class PpGqaBorrowed final:public DenseGemm {
public:
    PpGqaBorrowed(DenseGemm& service,void* arena,size_t bytes):service_(service),arena_(arena),bytes_(bytes){
        if(!arena||!bytes)throw std::invalid_argument("PP GQA borrowed arena");
    }
    bool eligible(PrefillAttentionShape s)const{
        return s.q8&&!s.graph_start&&s.dim==256&&s.query_heads==24&&s.kv_heads==4&&
               s.tokens>=6144&&s.valid>=30720&&s.valid<=32768;
    }
    void attention(const float* q,const void* k,const void* v,float* out,PrefillAttentionShape s,cudaStream_t stream){
        if(!eligible(s))throw std::invalid_argument("PP GQA borrowed selection");
        pp_gqa::run(service_,q,k,v,out,arena_,bytes_,s,128,stream);++recorded_;
    }
    void reset(){recorded_=0;}
    unsigned long long recorded()const{return recorded_;}
    void project(const void* w,const void* x,float* y,int rows,int cols,int tokens,cudaStream_t s)override{
        service_.project(w,x,y,rows,cols,tokens,s);
    }
    void batched_fp16(const void* a,const void* b,float* c,GemmBatch p,cudaStream_t s)override{
        service_.batched_fp16(a,b,c,p,s);
    }
    size_t workspace_bytes()const override{return service_.workspace_bytes();}
    unsigned long long calls()const override{return service_.calls();}
private:
    DenseGemm& service_;void* arena_;size_t bytes_;unsigned long long recorded_=0;
};
inline bool try_pp_gqa_borrowed(DenseGemm* service,const float* q,const void* k,const void* v,float* out,
                               PrefillAttentionShape s,cudaStream_t stream){
    auto* adapter=dynamic_cast<PpGqaBorrowed*>(service);
    if(!adapter||!adapter->eligible(s))return false;
    adapter->attention(q,k,v,out,s,stream);return true;
}
}
