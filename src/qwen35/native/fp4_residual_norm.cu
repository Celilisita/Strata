#include "fp4_residual_norm.hpp"
#include <nvfp4_layout.cuh>
#include "strata/qwen35/native/ffn.hpp"
#include <cmath>
namespace strata::qwen35::native {
namespace {
__device__ float norm_sum(float value){
    for(int d=16;d;d/=2)value=__fadd_rn(value,__shfl_xor_sync(0xffffffff,value,d));
    return value;
}
template<bool Debug>__global__ void fused_norm(const float *a,const float *b,const float *gamma,
        float *residual,float *debug,Nvfp4Input encoded,int features,float epsilon,float global){
    __shared__ float partial[32];
    __shared__ __align__(16) float normalized[5120];
    const size_t base=size_t(blockIdx.x)*5120;float value[5],square=0;
    #pragma unroll
    for(int j=0;j<5;++j){const int col=threadIdx.x+1024*j;
        value[j]=__fadd_rn(a[base+col],b[base+col]);square=__fmaf_rn(value[j],value[j],square);}
    square=norm_sum(square);
    if(threadIdx.x%32==0)partial[threadIdx.x/32]=square;
    __syncthreads();square=norm_sum(partial[threadIdx.x%32]);
    const float inverse=rsqrtf(__fmaf_rn(square,__fdividef(1.f,float(features)),epsilon));
    #pragma unroll
    for(int j=0;j<5;++j){const int col=threadIdx.x+1024*j;
        residual[base+col]=value[j];
        const float scaled=__fmul_rn(__fmul_rn(value[j],inverse),gamma[col]);normalized[col]=scaled;
        if constexpr(Debug)debug[base+col]=scaled;
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
}
PackedFp4Resources fp4_residual_norm_resources(){
    cudaFuncAttributes a{};int blocks=0;
    check_cuda(cudaFuncGetAttributes(&a,fused_norm<false>),"residual RMS FP4 resources");
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,fused_norm<false>,1024,0),"residual RMS FP4 occupancy");
    return {a.numRegs,blocks,a.sharedSizeBytes,a.localSizeBytes};
}
void fp4_residual_norm(const float *a,const float *b,const float *gamma,float *residual,float *debug,
        Nvfp4Input encoded,int tokens,float epsilon,float global,cudaStream_t stream){
    if(!a||!b||!gamma||!residual||!encoded.values||!encoded.scales||a==residual||b==residual||
        debug==residual||tokens!=8192||encoded.capacity_rows<tokens||encoded.capacity_columns<5120||
        !(epsilon>0)||!std::isfinite(epsilon)||!(global>0)||!std::isfinite(global))
        throw std::invalid_argument("fused residual RMS FP4 shape/lease");
    if(debug)fused_norm<true><<<tokens,1024,0,stream>>>(a,b,gamma,residual,debug,encoded,5120,epsilon,global);
    else fused_norm<false><<<tokens,1024,0,stream>>>(a,b,gamma,residual,nullptr,encoded,5120,epsilon,global);
    check_cuda(cudaGetLastError(),"fused residual RMS FP4");
}
}
