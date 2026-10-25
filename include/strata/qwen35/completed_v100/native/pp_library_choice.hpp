#pragma once
#include "strata/qwen35/completed_v100/native/attention_gemm.hpp"
namespace strata::completed_v100::qwen35::native {
// Request owner configures a borrowed service/arena before graph capture.
class PpLibraryChoice {
public:
    using Kernel=void(*)(DenseGemm&,const float*,const void*,const void*,float*,void*,size_t,
                         PrefillAttentionShape,int,cudaStream_t);
    void configure(Kernel kernel,DenseGemm* service,void* arena,size_t bytes,int minimum,int maximum){
        if(kernel&&(!service||!arena||!bytes||minimum<1||maximum<minimum))
            throw std::invalid_argument("PP library provider/arena lease");
        kernel_=kernel;service_=service;arena_=arena;bytes_=bytes;minimum_=minimum;maximum_=maximum;recorded_=0;
    }
    bool eligible(PrefillAttentionShape s)const{
        return kernel_&&s.q8&&!s.graph_start&&s.tokens>=128&&s.valid>=minimum_&&s.valid<=maximum_;
    }
    void run(const float* q,const void* k,const void* v,float* out,PrefillAttentionShape s,cudaStream_t stream){
        if(!eligible(s))throw std::invalid_argument("PP library shape selection");
        kernel_(*service_,q,k,v,out,arena_,bytes_,s,128,stream);++recorded_;
    }
    unsigned long long recorded()const{return recorded_;}
private:
    Kernel kernel_=nullptr;DenseGemm* service_=nullptr;void* arena_=nullptr;size_t bytes_=0;
    int minimum_=0,maximum_=0;unsigned long long recorded_=0;
};
}
