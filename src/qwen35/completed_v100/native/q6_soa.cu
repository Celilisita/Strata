#include "q6_soa.hpp"
#include "vector_fragment.cuh"
#include <cuda_fp16.h>
namespace strata::completed_v100::qwen35::native {
namespace {
__global__ void encode(const float* x,int* codes,int groups,int tokens) {
    const int group=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x&31,t=blockIdx.y;
    if(group>=groups)return;
    const float value=x[size_t(t)*groups*32+group*32+lane];float maximum=fabsf(value);
    for(int d=16;d;d/=2)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,d));
    const float delta=__fmul_rn(maximum,1.f/127.f);
    const int code=maximum?int(roundf(__fdividef(value,delta))):0;
    unsigned word=unsigned(code)&255;
    #pragma unroll
    for(int i=1;i<4;++i)word|=(unsigned(__shfl_down_sync(0xffffffff,code,i))&255)<<(8*i);
    if(!(lane&3))codes[size_t(t)*groups*8+group*8+lane/4]=int(word);
    if(!lane) {
        auto* scales=reinterpret_cast<float4*>(codes+3*groups*8)+group;
        reinterpret_cast<float*>(scales)[t]=__half2float(__float2half_rn(delta));
        if(!t)scales->w=0.f;
    }
}
__device__ __forceinline__ float4 scales(const float4* p) {
    float4 x;asm("ld.global.nc.v4.f32 {%0,%1,%2,%3}, [%4];":"=f"(x.x),"=f"(x.y),"=f"(x.z),"=f"(x.w):"l"(p));return x;
}
template<int Tokens> __global__ void multiply_soa080(Matrix m,const int* input,float* out) {
    __shared__ float partial[Tokens][128];
    const int row=blockIdx.x,tid=threadIdx.x,lane=tid&31,pair=lane/8;
    const int group=(pair/2)*4+pair%2,position=(lane&7)*4;
    const int groups=m.columns/32,blocks=m.columns/256;
    const auto* a=static_cast<const unsigned char*>(m.data)+size_t(row)*blocks*210;
    const auto* delta=reinterpret_cast<const float4*>(input+3*groups*8);
    float sum[Tokens]{};
    for(int block=tid/32;block<blocks;block+=4) {
        const auto* p=a+size_t(block)*210;
        const auto w=detail::packed_pair<Quant::Q6K>(p,group,position,nullptr);
        const int g=block*8+group,offset=g*8+position/4;
        const float4 d0=scales(delta+g),d1=scales(delta+g+2);
        const int s0=int(reinterpret_cast<const signed char*>(p+192)[group*2+position/16]);
        const int s1=int(reinterpret_cast<const signed char*>(p+192)[(group+2)*2+position/16]);
        const float wd=__half2float(*reinterpret_cast<const half*>(p+208));
        #pragma unroll
        for(int t=0;t<Tokens;++t) {
            const int* x=input+size_t(t)*groups*8+offset;
            const int dot0=__dp4a(w.first,__ldg(x),0),dot1=__dp4a(w.second,__ldg(x+16),0);
            const float x0=t==0?d0.x:t==1?d0.y:d0.z,x1=t==0?d1.x:t==1?d1.y:d1.z;
            float scaled=__fmaf_rn(x0,float(s0*dot0),0.f);
            scaled=__fmaf_rn(x1,float(s1*dot1),scaled);
            sum[t]=__fmaf_rn(wd,scaled,sum[t]);
        }
    }
    #pragma unroll
    for(int t=0;t<Tokens;++t)partial[t][tid]=sum[t];
    __syncthreads();
    if(tid<32) {
        #pragma unroll
        for(int t=0;t<Tokens;++t) {
            float value=partial[t][lane];
            #pragma unroll
            for(int stripe=1;stripe<4;++stripe)value=__fadd_rn(value,partial[t][stripe*32+lane]);
            for(int d=16;d;d/=2)value=__fadd_rn(value,__shfl_xor_sync(0xffffffff,value,d));
            if(!lane)out[size_t(t)*m.rows+row]=value;
        }
    }
}
__global__ void compare(const detail::ActivationBlock* old,const int* codes,int groups,int tokens,unsigned* errors) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=tokens*groups*32)return;
    const int t=i/(groups*32),g=(i/32)%groups,c=i%32;
    if(old[t*groups+g].codes[c]!=reinterpret_cast<const signed char*>(codes)[t*groups*32+g*32+c])atomicAdd(errors,1);
    if(!c) {
        const auto* s=reinterpret_cast<const float4*>(codes+3*groups*8)+g;
        if(__float_as_uint(__half2float(old[t*groups+g].delta))!=__float_as_uint(reinterpret_cast<const float*>(s)[t]))atomicAdd(errors,1);
    }
}
}
size_t q6_soa_bytes(int columns){if(columns<256||columns%256)throw std::invalid_argument("Q6SoA080columns");return size_t(columns/32)*112;}
void q6_soa_prepare(const float* x,void* y,int columns,int tokens,cudaStream_t s) {
    q6_soa_bytes(columns);if(!x||!y||(tokens!=2&&tokens!=3))throw std::invalid_argument("Q6SoA080input");
    encode<<<dim3(unsigned((columns/32+3)/4),unsigned(tokens)),128,0,s>>>(x,static_cast<int*>(y),columns/32,tokens);
    check_cuda(cudaGetLastError(),"Q6SoA080encode");
}
void q6_soa_project(Matrix m,const void* x,float* y,int tokens,cudaStream_t s) {
    matrix_bytes(m);q6_soa_bytes(m.columns);
    if(m.quant!=Quant::Q6K||m.layout!=BlockLayout::Gguf||!x||!y||(tokens!=2&&tokens!=3))throw std::invalid_argument("Q6SoA080weight/inputlease");
    if(tokens==2)multiply_soa080<2><<<m.rows,128,0,s>>>(m,static_cast<const int*>(x),y);
    else multiply_soa080<3><<<m.rows,128,0,s>>>(m,static_cast<const int*>(x),y);
    check_cuda(cudaGetLastError(),"Q6SoA080project");
}
void q6_soa_compare_encoding(const void* old,const void* x,int columns,int tokens,unsigned* errors,cudaStream_t s) {
    compare<<<unsigned((size_t(tokens)*columns+255)/256),256,0,s>>>(static_cast<const detail::ActivationBlock*>(old),static_cast<const int*>(x),columns/32,tokens,errors);
    check_cuda(cudaGetLastError(),"Q6SoA080payloadcheck");
}
}
