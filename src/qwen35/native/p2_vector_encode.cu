#include "p2_vector_encode.hpp"
#include "p2_scale.cuh"
#include "strata/qwen35/native/ffn.hpp"
#include "math.cuh"
#include <cuda_fp16.h>
#include <math_constants.h>
#include <cstdint>
namespace strata::qwen35::native {
namespace n=strata::qwen35::native;
namespace {
__device__ __forceinline__ unsigned raw(half2 v){
    return unsigned(__half_as_ushort(__low2half(v)))|(unsigned(__half_as_ushort(__high2half(v)))<<16);
}
template<bool Gated>__global__ void vector_encode(const float* x,const float* sg,const float* su,
    uint2* out,int rows,size_t vectors){
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=vectors)return;
    const size_t base=i*4;float4 values;
    if constexpr(Gated){
        const size_t at=(base/rows)*size_t(2*rows)+base%rows;
        const auto g=*reinterpret_cast<const float4*>(x+at),u=*reinterpret_cast<const float4*>(x+at+rows);
        const float a=*sg,b=*su;
        values={n::swiglu_value(__fmul_rn(g.x,a),__fmul_rn(u.x,b)),n::swiglu_value(__fmul_rn(g.y,a),__fmul_rn(u.y,b)),
            n::swiglu_value(__fmul_rn(g.z,a),__fmul_rn(u.z,b)),n::swiglu_value(__fmul_rn(g.w,a),__fmul_rn(u.w,b))};
    }else values=reinterpret_cast<const float4*>(x)[i];
    float maximum=fmaxf(fmaxf(fabsf(values.x),fabsf(values.y)),fmaxf(fabsf(values.z),fabsf(values.w)));
    const unsigned mask=__activemask();
    for(int d=2;d;d/=2)maximum=fmaxf(maximum,__shfl_xor_sync(mask,maximum,d,4));
    const bool invalid=!isfinite(values.x)||!isfinite(values.y)||!isfinite(values.z)||!isfinite(values.w)||
        fabsf(values.x)>65024.f||fabsf(values.y)>65024.f||fabsf(values.z)>65024.f||fabsf(values.w)>65024.f;
    const unsigned bad_lanes=__ballot_sync(mask,invalid);
    const bool bad=(bad_lanes&(0xfu<<(threadIdx.x&28)))!=0;
    const int p=p2_scale::power(maximum);const float s=p2_scale::scale(p);
    const float q0=float(p2_scale::code(values.x,p))*s,q1=float(p2_scale::code(values.y,p))*s;
    const float q2=float(p2_scale::code(values.z,p))*s,q3=float(p2_scale::code(values.w,p))*s;
    const auto low=__floats2half2_rn(bad?CUDART_NAN_F:q0,bad?CUDART_NAN_F:q1);
    const auto high=__floats2half2_rn(bad?CUDART_NAN_F:q2,bad?CUDART_NAN_F:q3);
    out[i]=make_uint2(raw(low),raw(high));
}
void lease(const float* x,void* out,size_t count){
    if(!x||!out||x==out||!count||count%64||reinterpret_cast<uintptr_t>(x)%16||reinterpret_cast<uintptr_t>(out)%8)
        throw std::invalid_argument("P2 vector aligned group16 span/lease");
}
}
void q8p2_encode_vector(const float* x,void* out,size_t count,cudaStream_t stream){
    lease(x,out,count);const size_t vectors=count/4;
    vector_encode<false><<<unsigned((vectors+127)/128),128,0,stream>>>(x,nullptr,nullptr,static_cast<uint2*>(out),0,vectors);
    n::check_cuda(cudaGetLastError(),"P2 vector vector4 group16 encode");
}
void swiglu_encode_vector(const float* x,const float* sg,const float* su,void* out,int rows,int tokens,cudaStream_t stream){
    if(rows<64||rows%64||tokens<1||tokens>8192||!sg||!su)throw std::invalid_argument("P2 vector packed FFN geometry");
    const size_t count=size_t(rows)*tokens;lease(x,out,count);
    const size_t vectors=count/4;
    vector_encode<true><<<unsigned((vectors+127)/128),128,0,stream>>>(x,sg,su,static_cast<uint2*>(out),rows,vectors);
    n::check_cuda(cudaGetLastError(),"P2 vector vector4 SwiGLU/P2");
}
}
