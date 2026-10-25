#include "gdn_output_elision.hpp"
#include <nvfp4_layout.cuh>
#include "strata/qwen35/native/ffn.hpp"
#include <math.cuh>
#include <cmath>
#include <cstdint>
namespace strata::qwen35::native {
namespace {
__device__ float gdn_output_elision_sum(float value){
    for(int d=16;d;d/=2)value=__fadd_rn(value,__shfl_xor_sync(0xffffffff,value,d));
    return value;
}
__global__ void dead_output_norm_encode(const float *attention,const float *gamma,const float *gate,
                                 Nvfp4Input encoded,float epsilon,float global){
    // Exactly the existing RMS128 256-thread reduction, followed by the same
    // FP4 block16 pair codec. Neither the dead F32 store nor its reread remains.
    __shared__ float partial[32];
    __shared__ __align__(16) float normalized[128];
    const size_t base=size_t(blockIdx.x)*128;float square=0;
    for(int i=threadIdx.x;i<128;i+=blockDim.x){
        const float value=attention[base+i];square=__fmaf_rn(value,value,square);
    }
    square=gdn_output_elision_sum(square);
    if(threadIdx.x%32==0)partial[threadIdx.x/32]=square;
    __syncthreads();
    square=gdn_output_elision_sum(threadIdx.x%32<blockDim.x/32?partial[threadIdx.x%32]:0.0f);
    const float scale=rsqrtf(__fmaf_rn(square,__fdividef(1.0f,float(128)),epsilon));
    for(int i=threadIdx.x;i<128;i+=blockDim.x){
        float value=__fmul_rn(__fmul_rn(attention[base+i],scale),gamma[i]);
        value=__fmul_rn(value,swiglu_value(gate[base+i],1.0f));
        normalized[i]=value;
    }
    __syncthreads();
    if(threadIdx.x<64){
        const int pair=threadIdx.x;const float x=normalized[2*pair],y=normalized[2*pair+1];
        float peak=fmaxf(fabsf(x),fabsf(y));unsigned bad=unsigned(!isfinite(x)||!isfinite(y));
        for(int d=4;d;d>>=1){
            peak=fmaxf(peak,__shfl_xor_sync(0xffffffff,peak,d,8));
            bad|=__shfl_xor_sync(0xffffffff,bad,d,8);
        }
        const unsigned char sf=bad?127:__nv_cvt_float_to_fp8(peak/(6.f*global),__NV_SATFINITE,__NV_E4M3);
        const float factor=nvfp4_detail::scale_float(sf)*global,inverse=factor>0?1.f/factor:0.f;
        const int row=int(blockIdx.x)/48,col=(int(blockIdx.x)%48)*128+2*pair;
        if((pair&7)==0)static_cast<unsigned char*>(encoded.scales)[nvfp4_detail::scale_index(row,col/16,6144)]=sf;
        static_cast<unsigned char*>(encoded.values)[size_t(blockIdx.x)*64+pair]=
            __nv_cvt_float2_to_fp4x2(make_float2(x*inverse,y*inverse),__NV_E2M1,cudaRoundNearest);
    }
}
bool overlap(const void *a,size_t n,const void *b,size_t m){
    const auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);
    return x<=y?y-x<n:x-y<m;
}
}
void gdn_output_elision_validate(const float *attention,const float *gamma,const float *gate,float *gated,
                   Nvfp4Input encoded,int tokens,float epsilon,float global){
    if(!attention||!gamma||!gate||!gated||!encoded.values||!encoded.scales||
       (tokens!=8192&&tokens!=5424)||tokens>encoded.capacity_rows||encoded.capacity_columns<6144||
       !(epsilon>0)||!std::isfinite(epsilon)||!(global>0)||!std::isfinite(global))
        throw std::invalid_argument("GDN output elision fixed PP geometry/storage/global");
    const void *pointers[]={attention,gamma,gate,gated,encoded.values,encoded.scales};
    for(const void *p:pointers)
        if(reinterpret_cast<uintptr_t>(p)%16)throw std::invalid_argument("GDN output elision alignment");
    const size_t n=size_t(tokens)*6144*4,q=nvfp4_value_bytes(tokens,6144),sf=nvfp4_scale_bytes(tokens,6144);
    const size_t spans[]={n,128*4,n,n,q,sf};
    for(int i=0;i<6;++i)if(spans[i]>UINTPTR_MAX-reinterpret_cast<uintptr_t>(pointers[i]))throw std::invalid_argument("dead output span overflow");
    const void *writers[]={gated,encoded.values,encoded.scales};const size_t sizes[]={n,q,sf};
    const void *readers[]={attention,gamma,gate};const size_t reads[]={n,128*4,n};
    for(int i=0;i<3;++i){
        for(int j=0;j<3;++j)if(overlap(writers[i],sizes[i],readers[j],reads[j]))
            throw std::invalid_argument("GDN output elision output aliases a live input");
        for(int j=0;j<i;++j)if(overlap(writers[i],sizes[i],writers[j],sizes[j]))
            throw std::invalid_argument("GDN output elision output ranges overlap");
    }
}
GdnOutputElisionResources gdn_output_elision_resources(){
    cudaFuncAttributes a{};int blocks=0;
    check_cuda(cudaFuncGetAttributes(&a,dead_output_norm_encode),"GDN output elision resources");
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,dead_output_norm_encode,256,0),"GDN output elision occupancy");
    return {a.numRegs,blocks,a.binaryVersion,a.sharedSizeBytes,a.localSizeBytes};
}
void gdn_output_elision_encode(const float *attention,const float *gamma,const float *gate,float *gated,
                 Nvfp4Input encoded,int tokens,float epsilon,float global,cudaStream_t stream){
    gdn_output_elision_validate(attention,gamma,gate,gated,encoded,tokens,epsilon,global);
    dead_output_norm_encode<<<tokens*48,256,0,stream>>>(attention,gamma,gate,encoded,epsilon,global);
    check_cuda(cudaGetLastError(),"GDN output elision gated RMS128 and FP4 encode");
}
}
