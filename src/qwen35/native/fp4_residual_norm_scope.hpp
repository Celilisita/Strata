#pragma once
#include <cuda_runtime.h>
#include <cstdlib>
#include <string>
#include <cmath>
#include <stdexcept>
namespace strata::qwen35::native {
class ResidualNormScope;
inline ResidualNormScope *&residual_norm_pending(){static thread_local ResidualNormScope *p=nullptr;return p;}
// A synchronous host lease. Kernel parameters copy these owned device pointers;
// the graph never stores an address of this descriptor.
class ResidualNormScope {
    ResidualNormScope *prior_=nullptr;bool deferred_=false,consumed_=false;
    int features_,tokens_;cudaStream_t stream_;
public:
    const float *a,*b,*gamma;float *residual,*normalized;float epsilon;bool reference_fused;
    ResidualNormScope(bool provider,const float *x,const float *y,const float *w,float *r,float *n,
            int features,int tokens,float eps,cudaStream_t stream,bool fused)
        :features_(features),tokens_(tokens),stream_(stream),a(x),b(y),gamma(w),residual(r),normalized(n),epsilon(eps),reference_fused(fused){
        if(!provider||features!=5120||tokens!=8192)return;
        const char *flag=std::getenv("STRATA_NVFP4_RESIDUAL_NORM");
        if(flag&&std::string(flag)!="0"&&std::string(flag)!="1")throw std::invalid_argument("residual norm mode");
        if(flag&&flag[0]=='0')return;
        if(!a||!b||!gamma||!residual||!normalized||a==residual||b==residual||residual==normalized||!(eps>0)||!std::isfinite(eps))
            throw std::invalid_argument("residual norm scoped operand lease");
        prior_=residual_norm_pending();residual_norm_pending()=this;deferred_=true;
    }
    ~ResidualNormScope(){if(deferred_)residual_norm_pending()=prior_;}
    ResidualNormScope(const ResidualNormScope&)=delete;
    bool deferred()const{return deferred_;}
    void validate(const float *input,int features,int tokens,cudaStream_t stream)const{
        if(!deferred_||consumed_||residual_norm_pending()!=this||input!=normalized||features!=features_||tokens!=tokens_||stream!=stream_)
            throw std::logic_error("residual norm stale/cross-stream/shape lease");
    }
    void consume(){if(consumed_)throw std::logic_error("residual norm duplicate consumption");consumed_=true;}
    void verify()const{if(deferred_&&!consumed_)throw std::logic_error("residual norm was not prepared by authoritative runtime");}
};
}
