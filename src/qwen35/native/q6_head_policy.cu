#include "q6_head_policy.hpp"
#include "vector_fragment.cuh"
#include <cuda_fp16.h>
namespace strata::qwen35::native {
namespace {
thread_local int selected_warps=0;
__device__ __forceinline__ float4 scales(const float4* p) {
 float4 x;asm("ld.global.nc.v4.f32 {%0,%1,%2,%3}, [%4];":"=f"(x.x),"=f"(x.y),"=f"(x.z),"=f"(x.w):"l"(p));return x;
}
template<int Warps,int Tokens> __global__ void multiply_head404(Matrix m,const int* input,float* out) {
    __shared__ float partial[Tokens][128];
    const int tid=threadIdx.x,lane=tid&31,pair=lane/8;
    const int group_row=(tid/32)/Warps, local_tid=tid%(Warps*32);
    const int row=int(blockIdx.x)*(4/Warps)+group_row;
    const int group=(pair/2)*4+pair%2,position=(lane&7)*4;
    const int groups=m.columns/32,blocks=m.columns/256;
    const auto* a=static_cast<const unsigned char*>(m.data)+size_t(row)*blocks*210;
    const auto* delta=reinterpret_cast<const float4*>(input+3*groups*8);
    float sum[Tokens]{};
    for(int block=local_tid/32;block<blocks;block+=Warps) {
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
    if(local_tid<32) {
        #pragma unroll
        for(int t=0;t<Tokens;++t) {
            float value=partial[t][group_row*Warps*32+lane];
            #pragma unroll
            for(int stripe=1;stripe<Warps;++stripe)value=__fadd_rn(value,partial[t][group_row*Warps*32+stripe*32+lane]);
            for(int d=16;d;d/=2)value=__fadd_rn(value,__shfl_xor_sync(0xffffffff,value,d));
            if(!lane)out[size_t(t)*m.rows+row]=value;
        }
    }
}
}
int q6_head_warps(){return selected_warps;}
void set_q6_head_warps(int warps){
 if(warps!=0&&warps!=1&&warps!=2)throw std::invalid_argument("head404 policy0/1/2");
 selected_warps=warps;
}
void project_q6_head(Matrix m,const void* x,float* y,int tokens,int warps,cudaStream_t s){
 matrix_bytes(m);
 if(m.quant!=Quant::Q6K||m.layout!=BlockLayout::Gguf||m.columns!=5120||m.rows!=248320||
    !x||!y||tokens!=3||(warps!=1&&warps!=2))throw std::invalid_argument("head404 originalQ36T3 lease");
 if(warps==1)multiply_head404<1,3><<<m.rows/4,128,0,s>>>(m,static_cast<const int*>(x),y);
 else multiply_head404<2,3><<<m.rows/2,128,0,s>>>(m,static_cast<const int*>(x),y);
 check_cuda(cudaGetLastError(),"head404 Q6 projection");
}
}
