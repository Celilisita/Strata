#include "strata/gemma4/native/decoder_ops.hpp"
#include <cmath>
namespace strata::gemma4::native {
namespace {
// Same cursor update as adopted head.cu, independent of greedy sampling.
__global__ void advance_position(int* position,int tokens,int capacity,unsigned* invalid){
    if(*position<0||*position>capacity-tokens){atomicExch(invalid,1u);return;}*position+=tokens;
}
__global__ void checked_cap(float* logits,std::size_t count,float threshold,unsigned* invalid){
    const auto i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count){const float raw=logits[i];if(!isfinite(raw))atomicOr(invalid,1u);
        logits[i]=threshold*tanhf(raw/threshold);}
}
}
void advance(int* position,int tokens,int capacity,unsigned* invalid,cudaStream_t stream){
    if(!position||!invalid||tokens<1||tokens>capacity)throw std::invalid_argument("Gemma device position extent");
    advance_position<<<1,1,0,stream>>>(position,tokens,capacity,invalid);
    primitive::check_cuda(cudaGetLastError(),"Gemma device position advance");
}
void softcap_logits_checked(float* logits,std::size_t count,float threshold,unsigned* invalid,cudaStream_t stream){
    if(!logits||!count||!invalid||!(threshold>0)||!std::isfinite(threshold))
        throw std::invalid_argument("Gemma checked head softcap");
    checked_cap<<<unsigned((count+255)/256),256,0,stream>>>(logits,count,threshold,invalid);
    primitive::check_cuda(cudaGetLastError(),"Gemma finite logits and final softcap");
}
}
