#pragma once
#include "strata/qwen35/native/nvfp4_runtime.hpp"
#include "nvfp4_output_lease.hpp"

namespace strata::qwen35::native {
class DownAddScope;
inline DownAddScope *&down_add_pending(){static thread_local DownAddScope *scope=nullptr;return scope;}
// Synchronous host-only operand lease. Graphs retain device pointers, never this descriptor.
class DownAddScope {
    bool deferred_=false,consumed_=false;int tokens_;cudaStream_t stream_;
public:
    const float *const residual;float *const destination;float *const original;
    DownAddScope(bool enabled,const float *c,float *d,float *o,int tokens,cudaStream_t stream)
        :tokens_(tokens),stream_(stream),residual(c),destination(d),original(o){
        if(!enabled||tokens!=8192)return;
        const size_t bytes=size_t(tokens)*5120*4;
        if(down_add_pending()||!nvfp4_float_aligned(c)||!nvfp4_float_aligned(d)||!nvfp4_float_aligned(o)||
           nvfp4_span_overlap(c,bytes,d,bytes)||nvfp4_span_overlap(c,bytes,o,bytes)||nvfp4_span_overlap(d,bytes,o,bytes))
            throw std::invalid_argument("Down add scoped ownership/alignment/nesting");
        down_add_pending()=this;deferred_=true;
    }
    ~DownAddScope(){if(deferred_)down_add_pending()=nullptr;}
    DownAddScope(const DownAddScope&)=delete;DownAddScope& operator=(const DownAddScope&)=delete;
    bool deferred()const{return deferred_;}
    bool consumed()const{return consumed_;}
    void validate_output(const float *out,int features,int tokens,cudaStream_t stream)const{
        if(!deferred_||down_add_pending()!=this||consumed_||out!=original||features!=5120||tokens!=tokens_||stream!=stream_)
            throw std::logic_error("Down add stale/consumed/output/shape/stream lease");
    }
    void validate(Nvfp4View w,Nvfp4Input x,float *out,int tokens,cudaStream_t stream)const{
        validate_output(out,w.rows,tokens,stream);validate_nvfp4_add_output(w,x,residual,destination,tokens);
        const size_t bytes=size_t(tokens)*5120*4;
        const void *operands[]={w.values,w.scales,x.values,x.scales};
        const size_t spans[]={nvfp4_value_bytes(w.rows,w.columns),nvfp4_scale_bytes(w.rows,w.columns),
            nvfp4_value_bytes(tokens,w.columns),nvfp4_scale_bytes(tokens,w.columns)};
        for(int k=0;k<4;++k)if(nvfp4_span_overlap(original,bytes,operands[k],spans[k]))
            throw std::invalid_argument("Down add original output overlaps encoded operand");
    }
    void consume(){
        if(!deferred_||down_add_pending()!=this||consumed_)throw std::logic_error("Down add duplicate consumption");
        consumed_=true;
    }
    // Unconsumed is a normal fallback: the caller must issue the old residual add.
    void verify()const{
        if(deferred_&&down_add_pending()!=this)throw std::logic_error("Down add scope identity changed");
    }
};
class DownAddPolicy {
    bool enabled_=false;DownAddCounters counters_;
public:
    void configure(bool enabled){
        if(down_add_pending())throw std::logic_error("Down add configure with active host lease");
        enabled_=enabled;counters_={};
    }
    DownAddCounters counters()const{return counters_;}
    DownAddScope *admit(float *original,int features,int tokens,cudaStream_t stream,bool eligible){
        auto *scope=down_add_pending();if(!scope)return nullptr;
        scope->validate_output(original,features,tokens,stream);
        if(!enabled_||!eligible){++counters_.fallback;return nullptr;}
        return scope;
    }
    void project(Nvfp4Gemm &gemm,Nvfp4View w,Nvfp4Input x,float *out,int tokens,cudaStream_t stream,DownAddScope *scope){
        if(!scope){gemm.project(w,x,out,tokens,stream);return;}
        scope->validate(w,x,out,tokens,stream);
        gemm.project_add(w,x,scope->residual,scope->destination,tokens,stream);
        scope->consume();++counters_.fused;
    }
};
}
