#pragma once
#include <cuda_fp16.h>
namespace strata::gemma4::native {
namespace {
// The first256 bytes of the existing cuBLAS workspace are exclusively ours;
// cuBLAS receives only the aligned remainder. No new device allocation.
__global__ void initialize_input_scalars(float* s){s[0]=0;s[1]=1;s[2]=1;s[3]=1.f/4096.f;s[4]=0;s[5]=1;}
__global__ void input_maximum(const float* input,size_t count,float* scalar){
    __shared__ float warps[8];const int tid=threadIdx.x,lane=tid%32;
    const size_t i=size_t(blockIdx.x)*256+tid;float x=i<count?fabsf(input[i]):0.f;
    for(int d=16;d;d/=2)x=fmaxf(x,__shfl_xor_sync(0xffffffff,x,d));
    if(!lane)warps[tid/32]=x;__syncthreads();
    if(tid<32){x=lane<8?warps[lane]:0.f;
        for(int d=16;d;d/=2)x=fmaxf(x,__shfl_xor_sync(0xffffffff,x,d));
        if(!lane)atomicMax(reinterpret_cast<unsigned*>(scalar),__float_as_uint(x));}
}
__global__ void choose_input_scale(float* s){
    const float maximum=s[0];float scale=1.f;
    if(isfinite(maximum)&&maximum>256.f){
        const int exponent=int((__float_as_uint(maximum)>>23)&255);
        scale=__uint_as_float(unsigned(261-exponent)<<23);
    }
    s[1]=scale;s[2]=1.f/scale;s[3]=(1.f/4096.f)/scale;
}
__global__ void input_planes_scaled(const float* input,half* output,size_t count,const float* scalars){
    const size_t i=size_t(blockIdx.x)*256+threadIdx.x;
    if(i<count){const float x=input[i]*scalars[1];const half first=__float2half_rn(x);
        output[i]=first;output[count+i]=__float2half_rn((x-__half2float(first))*4096.f);}
}
}
}
