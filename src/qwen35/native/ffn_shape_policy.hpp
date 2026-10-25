#pragma once
#include "strata/qwen35/native/dense_gemm.hpp"
namespace strata::qwen35::native {
std::unique_ptr<DenseGemm> make_partial16_gemm(DenseGemm &,cudaStream_t);
// Cold service choice from a measured shape crossover; all kernels/ownership
// stay with the existing fused provider and its admitted paired fallback.
class FfnShapePolicy final:public DenseGemm {
    DenseGemm &fused_;int pair_min_tokens_;DenseGemm *long_fused_,*long_linear_;
public:
    FfnShapePolicy(DenseGemm &fused,int pair_min_tokens,DenseGemm *long_fused=nullptr,DenseGemm *long_linear=nullptr)
        :fused_(fused),pair_min_tokens_(pair_min_tokens),long_fused_(long_fused),long_linear_(long_linear){
        if(pair_min_tokens<128)throw std::invalid_argument("invalid measured FFN crossover");
    }
    bool can_fuse_swiglu(int r,int k,int t)const override{return t<pair_min_tokens_?fused_.can_fuse_swiglu(r,k,t):
        long_fused_&&long_fused_->can_fuse_swiglu(r,k,t);}
    void project_swiglu(const void *w,const void *x,void *y,float *logical,float *gate,float *up,
                        int r,int k,int t,cudaStream_t s)override{
        auto &provider=t>=pair_min_tokens_&&long_fused_?*long_fused_:fused_;
        provider.project_swiglu(w,x,y,logical,gate,up,r,k,t,s);
    }
    void project(const void *w,const void *x,float *y,int r,int k,int t,cudaStream_t s)override{
        auto &provider=t>=pair_min_tokens_&&r>=256&&long_linear_?*long_linear_:fused_;
        provider.project(w,x,y,r,k,t,s);
    }
    bool project_pair(const void *w,const void *x,float *y,int r,int k,int t,cudaStream_t s)override{
        auto &provider=t>=pair_min_tokens_&&r>=128&&long_linear_?*long_linear_:fused_;
        return provider.project_pair(w,x,y,r,k,t,s);
    }
    void batched_fp16(const void *a,const void *b,float *c,GemmBatch p,cudaStream_t s)override{fused_.batched_fp16(a,b,c,p,s);}
    size_t workspace_bytes()const override{return fused_.workspace_bytes();}
    unsigned long long calls()const override{return fused_.calls();}
    GemmWorkspace borrowed_workspace()const override{return fused_.borrowed_workspace();}
};
}
