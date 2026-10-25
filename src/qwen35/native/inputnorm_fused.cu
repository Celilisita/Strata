#include "inputnorm.hpp"
#include <nvfp4_layout.cuh>
#include <strata/qwen35/native/ffn.hpp>
#include <cmath>
#include <cstdint>
#include <limits>
namespace strata::qwen35::native {
namespace {
__device__ float sum_warp(float value){
    for(int d=16;d;d/=2)value=__fadd_rn(value,__shfl_xor_sync(0xffffffff,value,d));return value;
}
__global__ void fused_input_norm(const float *input,const float *gamma,Nvfp4Input encoded,
                                 __half *half,int features,float epsilon,float global){
    __shared__ float partial[32];
    __shared__ __align__(16) float normalized[5120];
    const size_t base=size_t(blockIdx.x)*5120;float value[5],square=0;
    #pragma unroll
    for(int j=0;j<5;++j){const int col=threadIdx.x+1024*j;
        value[j]=input[base+col];square=__fmaf_rn(value[j],value[j],square);}
    square=sum_warp(square);
    if(threadIdx.x%32==0)partial[threadIdx.x/32]=square;
    __syncthreads();square=sum_warp(partial[threadIdx.x%32]);
    const float inverse=rsqrtf(__fmaf_rn(square,__fdividef(1.f,float(features)),epsilon));
    #pragma unroll
    for(int j=0;j<5;++j){const int col=threadIdx.x+1024*j;
        const float scaled=__fmul_rn(__fmul_rn(value[j],inverse),gamma[col]);normalized[col]=scaled;
        if(half)half[base+col]=__float2half_rn(scaled);
    }
    __syncthreads();
    for(int pair=threadIdx.x;pair<2560;pair+=1024){
        const float x=normalized[2*pair],y=normalized[2*pair+1];
        float peak=fmaxf(fabsf(x),fabsf(y));unsigned bad=unsigned(!isfinite(x)||!isfinite(y));
        for(int d=4;d;d>>=1){peak=fmaxf(peak,__shfl_xor_sync(0xffffffff,peak,d,8));bad|=__shfl_xor_sync(0xffffffff,bad,d,8);}
        const unsigned char sf=bad?127:__nv_cvt_float_to_fp8(peak/(6.f*global),__NV_SATFINITE,__NV_E4M3);
        const float scale=nvfp4_detail::scale_float(sf)*global,inverse_scale=scale>0?1.f/scale:0.f;
        if((threadIdx.x&7)==0)static_cast<unsigned char*>(encoded.scales)[nvfp4_detail::scale_index(blockIdx.x,pair/8,5120)]=sf;
        static_cast<unsigned char*>(encoded.values)[size_t(blockIdx.x)*2560+pair]=
            __nv_cvt_float2_to_fp4x2(make_float2(x*inverse_scale,y*inverse_scale),__NV_E2M1,cudaRoundNearest);
    }
}
bool overlap(const void *a,size_t an,const void *b,size_t bn){
    const auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);
    if(x>std::numeric_limits<uintptr_t>::max()-an||y>std::numeric_limits<uintptr_t>::max()-bn)
        throw std::invalid_argument("input norm pointer range overflow");
    return x<y+bn&&y<x+an;
}
}
PackedFp4Resources inputnorm_resources(){
    cudaFuncAttributes a{};int blocks=0;
    check_cuda(cudaFuncGetAttributes(&a,fused_input_norm),"input RMS FP4 resources");
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,fused_input_norm,1024,0),"input RMS FP4 occupancy");
    return {a.numRegs,blocks,a.sharedSizeBytes,a.localSizeBytes};
}
void inputnorm_fused(const float *input,const float *gamma,Nvfp4Input encoded,void *optional_fp16,
                     int tokens,float epsilon,float global,cudaStream_t stream){
    if(!input||!gamma||!encoded.values||!encoded.scales||!inputnorm_eligible(5120,tokens)||
       encoded.capacity_rows<tokens||encoded.capacity_columns<5120||
       !(epsilon>0)||!std::isfinite(epsilon)||!(global>0)||!std::isfinite(global))
        throw std::invalid_argument("input RMS FP4 exact shape/storage/scales");
    const size_t floats=size_t(tokens)*5120*4,halves=floats/2,values=nvfp4_value_bytes(tokens,5120),scales=nvfp4_scale_bytes(tokens,5120);
    const void *outputs[]={encoded.values,encoded.scales,optional_fp16};const size_t bytes[]={values,scales,halves};
    if(reinterpret_cast<uintptr_t>(input)%4||reinterpret_cast<uintptr_t>(gamma)%4)
        throw std::invalid_argument("input RMS FP4 source alignment");
    for(int i=0;i<3;++i)if(outputs[i]){
        if(reinterpret_cast<uintptr_t>(outputs[i])%16||overlap(outputs[i],bytes[i],input,floats)||overlap(outputs[i],bytes[i],gamma,5120*4))
            throw std::invalid_argument("input RMS FP4 live source/output alias or alignment");
        for(int j=0;j<i;++j)if(outputs[j]&&overlap(outputs[i],bytes[i],outputs[j],bytes[j]))
            throw std::invalid_argument("input RMS FP4 output planes alias");
    }
    fused_input_norm<<<tokens,1024,0,stream>>>(input,gamma,encoded,static_cast<__half*>(optional_fp16),5120,epsilon,global);
    check_cuda(cudaGetLastError(),"input RMS FP4 with optional FP16");
}
}
