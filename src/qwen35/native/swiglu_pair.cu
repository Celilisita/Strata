#include "strata/qwen35/native/dense_gemm.hpp"
#include "strata/qwen35/native/ffn.hpp"
#include "math.cuh"
#include <cuda_fp16.h>

namespace strata::qwen35::native {
namespace {
__global__ void activate_pair(const float *pair,half2 *out,int rows,size_t n){
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=n/4)return;
    const size_t token=(i*4)/rows,feature=(i*4)%rows,start=token*2*rows+feature;
    const auto g=*reinterpret_cast<const float4 *>(pair+start);
    const auto u=*reinterpret_cast<const float4 *>(pair+start+rows);
    out[2*i]=__floats2half2_rn(swiglu_value(g.x,u.x),swiglu_value(g.y,u.y));
    out[2*i+1]=__floats2half2_rn(swiglu_value(g.z,u.z),swiglu_value(g.w,u.w));
}
__global__ void materialize_pair(const float *pair,float *gate,float *up,float *activation,half *out,int rows,size_t n){
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=n)return;
    const size_t token=i/rows,feature=i%rows,start=token*2*rows+feature;
    const float g=pair[start],u=pair[start+rows],a=swiglu_value(g,u);
    gate[i]=g;up[i]=u;activation[i]=a;out[i]=__float2half_rn(a);
}
}
void swiglu_pair_half(const float *pair,void *out,int rows,int tokens,cudaStream_t stream){
    if(!pair||!out||rows<1||rows%4||tokens<1)throw std::invalid_argument("paired SwiGLU vector geometry invalid");
    const size_t n=size_t(rows)*tokens;
    activate_pair<<<unsigned((n/4+127)/128),128,0,stream>>>(pair,static_cast<half2 *>(out),rows,n);
    check_cuda(cudaGetLastError(),"Strata paired vector SwiGLU half output");
}
void split_swiglu_pair(const float *pair,float *gate,float *up,float *activation,void *out,int rows,int tokens,cudaStream_t stream){
    if(!pair||!gate||!up||!activation||!out||rows<1||tokens<1||pair==gate||pair==up)
        throw std::invalid_argument("paired diagnostic requires nonaliasing copied source");
    const size_t n=size_t(rows)*tokens;
    materialize_pair<<<unsigned((n+255)/256),256,0,stream>>>(pair,gate,up,activation,static_cast<half *>(out),rows,n);
    check_cuda(cudaGetLastError(),"Strata paired diagnostic materialization");
}
}
