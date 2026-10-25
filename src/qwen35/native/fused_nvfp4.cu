#include "fused_nvfp4.hpp"
#include "strata/qwen35/native/ffn.hpp"
#include "nvfp4_layout.cuh"
#include "math.cuh"
#include <cmath>
namespace strata::qwen35::native {
namespace {
template<bool Half> __device__ float2 input_pair(const void *data,size_t pair){
    if constexpr(Half)return __half22float2(reinterpret_cast<const __half2 *>(data)[pair]);
    else return reinterpret_cast<const float2 *>(data)[pair];
}
template<bool Diagnostic,bool Half> __global__ void fused(const void *gate,const void *up,float *activation,
    unsigned char *q,unsigned char *sf,int tokens,int columns,float global){
    const size_t pair=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(pair>=size_t(tokens)*columns/2)return;
    const auto g=input_pair<Half>(gate,pair),u=input_pair<Half>(up,pair);
    const float x=swiglu_value(g.x,u.x),y=swiglu_value(g.y,u.y);
    if constexpr(Diagnostic)reinterpret_cast<float2 *>(activation)[pair]=make_float2(x,y);
    float maximum=fmaxf(fabsf(x),fabsf(y));unsigned bad=unsigned(!isfinite(x)||!isfinite(y));
    for(int d=4;d;d>>=1){maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,d,8));bad|=__shfl_xor_sync(0xffffffff,bad,d,8);}
    const unsigned char scale=bad?127:__nv_cvt_float_to_fp8(maximum/(6.f*global),__NV_SATFINITE,__NV_E4M3);
    const float restored=nvfp4_detail::scale_float(scale)*global,inverse=restored>0?1.f/restored:0;
    const int row=int(2*pair/columns),column=int(2*pair%columns);
    if((threadIdx.x&7)==0)sf[nvfp4_detail::scale_index(row,column/16,columns)]=scale;
    q[pair]=__nv_cvt_float2_to_fp4x2(make_float2(x*inverse,y*inverse),__NV_E2M1,cudaRoundNearest);
}
template<bool Half> void launch(const void *gate,const void *up,float *diagnostic,Nvfp4Input x,int t,int k,float global,cudaStream_t stream){
    if(!gate||!up||!x.values||!x.scales||t<1||k<64||k%64||t>x.capacity_rows||k>x.capacity_columns||!(global>0)||!std::isfinite(global))
        throw std::invalid_argument("NVFP4 fused SwiGLU storage/geometry");
    const unsigned blocks=unsigned((size_t(t)*k/2+255)/256);
    if(diagnostic)fused<true,Half><<<blocks,256,0,stream>>>(gate,up,diagnostic,static_cast<unsigned char *>(x.values),static_cast<unsigned char *>(x.scales),t,k,global);
    else fused<false,Half><<<blocks,256,0,stream>>>(gate,up,nullptr,static_cast<unsigned char *>(x.values),static_cast<unsigned char *>(x.scales),t,k,global);
    check_cuda(cudaGetLastError(),"NVFP4 fused SwiGLU quantization");
}
__global__ void expand(const __half *input,float *output,size_t n){const size_t i=size_t(blockIdx.x)*256+threadIdx.x;if(i<n)output[i]=__half2float(input[i]);}
}
void nvfp4_fused_swiglu(const float *g,const float *u,float *a,Nvfp4Input x,int t,int k,float global,cudaStream_t s){launch<false>(g,u,a,x,t,k,global,s);}
void nvfp4_fused_swiglu_half(const void *g,const void *u,float *a,Nvfp4Input x,int t,int k,float global,cudaStream_t s){launch<true>(g,u,a,x,t,k,global,s);}
void nvfp4_expand_half(const void *input,float *output,size_t n,cudaStream_t s){
    if(!input||!output||!n)throw std::invalid_argument("NVFP4 half diagnostic storage");
    expand<<<unsigned((n+255)/256),256,0,s>>>(static_cast<const __half *>(input),output,n);check_cuda(cudaGetLastError(),"NVFP4 half diagnostic materialization");
}
}
