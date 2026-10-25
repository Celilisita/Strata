#pragma once
#include "fp4_compensated.hpp"
#include "nvfp4_layout.cuh"
namespace strata::qwen35::native::compensated_codec_detail {
__device__ __forceinline__ float decode_fp4(unsigned char q){
    const unsigned m=q&7;
    const unsigned magnitude=m<2?(m?0x3f000000u:0u):(((m/2+126u)<<23)|((m&1)<<22));
    return __uint_as_float(magnitude|((unsigned(q)&8)<<28));
}
template<bool Scalar> __device__ __forceinline__ float decode(unsigned char q){
    if constexpr(Scalar)return decode_fp4(q);else return nvfp4_detail::fp4_float(q);
}
struct LowCandidate {unsigned char bits,scale;float square_error,peak;};
template<int Stride,bool Scalar> __device__ LowCandidate low_candidate(float x,float y,unsigned char sf,float global,unsigned mask){
    const float scale=nvfp4_detail::scale_float(sf)*global,inv=scale>0?1.f/scale:0.f;
    const unsigned char bits=__nv_cvt_float2_to_fp4x2(make_float2(x*inv,y*inv),__NV_E2M1,cudaRoundNearest);
    const float ex=__fsub_rn(x,__fmul_rn(decode<Scalar>(bits&15),scale));
    const float ey=__fsub_rn(y,__fmul_rn(decode<Scalar>(bits>>4),scale));
    float square=__fadd_rn(__fmul_rn(ex,ex),__fmul_rn(ey,ey)),peak=fmaxf(fabsf(ex),fabsf(ey));
    for(int d=4*Stride;d>=Stride;d>>=1){square+=__shfl_xor_sync(mask,square,d,Stride==1?8:32);peak=fmaxf(peak,__shfl_xor_sync(mask,peak,d,Stride==1?8:32));}
    return {bits,sf,square,peak};
}
template<int Stride,bool Scalar> __device__ void encode_pair(float x,float y,CompensatedFp4Input out,int row,int col,int columns,float global,unsigned mask){
    const size_t pair=(size_t(row)*columns+col)/2;
    unsigned bad=unsigned(!isfinite(x)||!isfinite(y));
    float maximum=fmaxf(fabsf(x),fabsf(y));
    for(int d=4*Stride;d>=Stride;d>>=1){maximum=fmaxf(maximum,__shfl_xor_sync(mask,maximum,d,Stride==1?8:32));bad|=__shfl_xor_sync(mask,bad,d,Stride==1?8:32);}
    const float input_peak=maximum;
    const unsigned char hs=bad?127:__nv_cvt_float_to_fp8(maximum/(6.f*global),__NV_SATFINITE,__NV_E4M3);
    const float high_scale=nvfp4_detail::scale_float(hs)*global,high_inv=high_scale>0?1.f/high_scale:0.f;
    const unsigned char high=__nv_cvt_float2_to_fp4x2(make_float2(x*high_inv,y*high_inv),__NV_E2M1,cudaRoundNearest);
    static_cast<unsigned char*>(out.high.values)[pair]=high;
    const float rx=bad?0.f:__fsub_rn(x,__fmul_rn(decode<Scalar>(high&15),high_scale));
    const float ry=bad?0.f:__fsub_rn(y,__fmul_rn(decode<Scalar>(high>>4),high_scale));
    maximum=fmaxf(fabsf(rx),fabsf(ry));
    for(int d=4*Stride;d>=Stride;d>>=1)maximum=fmaxf(maximum,__shfl_xor_sync(mask,maximum,d,Stride==1?8:32));
    const float low_global=__fmul_rn(global,CompensatedLowRatio);
    float energy=bad?0.f:__fadd_rn(__fmul_rn(x,x),__fmul_rn(y,y));
    for(int d=4*Stride;d>=Stride;d>>=1)energy+=__shfl_xor_sync(mask,energy,d,Stride==1?8:32);
    const auto classic=__nv_cvt_float_to_fp8(maximum/(6.f*low_global),__NV_SATFINITE,__NV_E4M3);
    LowCandidate best=low_candidate<Stride,Scalar>(rx,ry,classic,low_global,mask);
    const float peak_limit=.02f*input_peak;
    if(best.square_error>.0001f*energy||best.peak>peak_limit){
        const float target=maximum/(4.5f*low_global);
        const unsigned char near=__nv_cvt_float_to_fp8(target,__NV_SATFINITE,__NV_E4M3);
        const int floor=int(near)-(nvfp4_detail::scale_float(near)>target?1:0);
        #pragma unroll
        for(int i=0;i<2;++i){
            const unsigned char sf=static_cast<unsigned char>(max(0,min(126,floor+i)));
            const auto trial=low_candidate<Stride,Scalar>(rx,ry,sf,low_global,mask);
            const bool valid=trial.peak<=peak_limit,previous=best.peak<=peak_limit;
            if((valid&&(!previous||trial.square_error<best.square_error))||(!valid&&!previous&&trial.peak<best.peak))best=trial;
        }
    }
    const unsigned char ls=bad?127:best.scale;
    static_cast<unsigned char*>(out.low.values)[pair]=bad?0:best.bits;
    if(((threadIdx.x/Stride)&7)==0){const auto index=nvfp4_detail::scale_index(row,col/16,columns);
        static_cast<unsigned char*>(out.high.scales)[index]=hs;static_cast<unsigned char*>(out.low.scales)[index]=ls;}
}
}
