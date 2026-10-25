#pragma once
#include "inputnorm_policy.hpp"
#include "strata/qwen35/native/ops.hpp"
#include "strata/qwen35/native/nvfp4_runtime.hpp"
#include <array>
#include <cmath>
#include <initializer_list>
#include <limits>
namespace strata::qwen35::native {
class InputNormScope;
inline InputNormScope *&inputnorm_pending(){static thread_local InputNormScope *scope=nullptr;return scope;}
// Synchronous host lease only. Kernels/Graphs copy pointers into model-owned GPU storage.
class InputNormScope {
    InputNormScope *prior_=nullptr;bool deferred_=false,consumed_=false;int features_,tokens_;cudaStream_t stream_;
public:
    const float *hidden,*gamma;float *normalized;void *fp16;float epsilon;
    InputNormScope(bool provider,const float *x,const float *w,float *n,void *half,int features,int tokens,float eps,cudaStream_t stream)
        :features_(features),tokens_(tokens),stream_(stream),hidden(x),gamma(w),normalized(n),fp16(half),epsilon(eps){
        if(!provider||!inputnorm_eligible(features,tokens))return;
        if(!hidden||!gamma||!normalized||hidden==normalized||!(eps>0)||!std::isfinite(eps))
            throw std::invalid_argument("input RMS scoped source/materialization lease");
        prior_=inputnorm_pending();inputnorm_pending()=this;deferred_=true;
    }
    ~InputNormScope(){if(deferred_)inputnorm_pending()=prior_;}
    InputNormScope(const InputNormScope&)=delete;
    bool deferred()const{return deferred_;}
    void validate(const float *input,int features,int tokens,cudaStream_t stream)const{
        if(!deferred_||consumed_||inputnorm_pending()!=this||input!=normalized||features!=features_||tokens!=tokens_||stream!=stream_)
            throw std::logic_error("input RMS stale/cross-stream/shape lease");
    }
    void consume(){if(consumed_)throw std::logic_error("input RMS duplicate consumption");consumed_=true;}
    void verify()const{if(deferred_&&!consumed_)throw std::logic_error("input RMS was not consumed by ordered projection group");}
};
inline bool inputnorm_overlap(const void *a,size_t an,const void *b,size_t bn){
    const auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);
    if(x>std::numeric_limits<uintptr_t>::max()-an||y>std::numeric_limits<uintptr_t>::max()-bn)
        throw std::invalid_argument("input RMS pointer range overflow");
    return x<y+bn&&y<x+an;
}
inline bool inputnorm_prepare_set(const float *input,const std::array<const Nvfp4View*,4> &views,
        std::initializer_list<Nvfp4Projection> projections,Nvfp4Input encoded,int tokens,cudaStream_t stream){
    auto *scope=inputnorm_pending();if(!scope)return false;
    const int columns=projections.begin()->weight.columns;scope->validate(input,columns,tokens,stream);
    const size_t floats=size_t(tokens)*columns*4;bool eligible=inputnorm_request_active()&&inputnorm_eligible(columns,tokens);
    const float scale=views[0]->input_scale;size_t index=0;
    for(const auto &p:projections){const auto &w=*views[index++];
        eligible=eligible&&w.columns==5120&&w.input_scale==scale; // At8192 packed and planar weights both use Runtime kind0.
        const size_t bytes=size_t(tokens)*w.rows*4;
        if(inputnorm_overlap(p.output,bytes,scope->hidden,floats)||inputnorm_overlap(p.output,bytes,scope->gamma,5120*4)||
           (scope->fp16&&inputnorm_overlap(p.output,bytes,scope->fp16,floats/2)))
            throw std::invalid_argument("input RMS projection aliases live source/gamma/FP16 lease");
    }
    if(eligible){
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12080
        inputnorm_fused(scope->hidden,scope->gamma,encoded,scope->fp16,tokens,scope->epsilon,scale,stream);
        ++inputnorm_host_submissions();scope->consume();return true;
#endif
    }
    // Flag-off/mixed-group fallback must materialize the old view before the original codec reads it.
    rms_norm(scope->hidden,scope->gamma,scope->normalized,columns,tokens,scope->epsilon,stream);
    if(scope->fp16)to_half(scope->normalized,scope->fp16,size_t(tokens)*columns,stream);
    scope->consume();return false;
}
}
