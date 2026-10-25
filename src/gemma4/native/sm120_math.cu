// GELU/residual/softcap math extracted unchanged from adopted Gemma ops.cu.
#include "strata/gemma4/native/decoder_ops.hpp"
#include <cmath>
namespace strata::gemma4::native {
namespace {
__global__ void gelu(const float* g,const float* u,float* out,size_t count){
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=count)return;
    const float x=g[i];const float t=0.7978845608028654f*(x+0.044715f*x*x*x);
    out[i]=(0.5f*x*(1.f+tanhf(t)))*u[i];
}
__global__ void combine(const float* post,const float* residual,const float* scale,float* out,size_t count){
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count)out[i]=__fmul_rn(__fadd_rn(residual[i],post[i]),*scale);
}
__global__ void cap(float* logits,size_t count,float threshold){
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count)logits[i]=threshold*tanhf(logits[i]/threshold);
}
}
void gelu_product(const float* g,const float* u,float* out,size_t count,cudaStream_t stream){
    if(!g||!u||!out||g==out||u==out||!count)throw std::invalid_argument("Gemma4 GELU operand lease");
    gelu<<<unsigned((count+255)/256),256,0,stream>>>(g,u,out,count);
    primitive::check_cuda(cudaGetLastError(),"Gemma4 GELU tanh product");
}
void residual_scale(const float* post,const float* residual,const float* scale,float* out,size_t count,cudaStream_t stream){
    if(!post||!residual||!scale||!out||!count)throw std::invalid_argument("Gemma4 residual/scale span");
    combine<<<unsigned((count+255)/256),256,0,stream>>>(post,residual,scale,out,count);
    primitive::check_cuda(cudaGetLastError(),"Gemma4 post FFN residual and layer scale");
}
void softcap_logits(float* logits,size_t count,float threshold,cudaStream_t stream){
    if(!logits||!count||!std::isfinite(threshold)||threshold<=0)throw std::invalid_argument("Gemma4 softcap bounds");
    cap<<<unsigned((count+255)/256),256,0,stream>>>(logits,count,threshold);
    primitive::check_cuda(cudaGetLastError(),"Gemma4 final logit softcap");
}
}
