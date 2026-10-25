#include "raw_q8_wide.cuh"
#include "raw_q8_wide_codec.cuh"
#include "strata/qwen35/native/ffn.hpp"
#include <cuda_fp16.h>
namespace strata::qwen35::native {
namespace {
struct alignas(16) AoS {int4 code;float scale;unsigned invalid,padding[2];};
static_assert(sizeof(AoS)==32,"Existing056 activation ABI");
__device__ __forceinline__ float4 scale4(const float4* p) {
    float4 s;asm("ld.global.nc.v4.f32 {%0,%1,%2,%3}, [%4];":"=f"(s.x),"=f"(s.y),"=f"(s.z),"=f"(s.w):"l"(p));return s;
}
__device__ __forceinline__ int dot(int4 w,int4 x) {
    int a=__dp4a(w.x,x.x,0);a=__dp4a(w.y,x.y,a);a=__dp4a(w.z,x.z,a);return __dp4a(w.w,x.w,a);
}
template<int Tokens> __global__ void wide_head(const unsigned char* raw,const void* encoded,float* output,int rows,int columns) {
    const int row=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x&31;if(row>=rows)return;
    const int groups=columns/16;const auto* weight=raw+size_t(row)*(columns/32)*34;
    float sum[Tokens]{};
    for(int group=lane;group<groups;group+=32) {
        const auto* block=weight+(group/2)*34;
        const int4 w=q8wide::codes16(block+2+(group&1)*16);
        const float delta=__half2float(__ldg(reinterpret_cast<const half*>(block)));
        float4 scales{};
        if constexpr(Tokens==3)scales=scale4(reinterpret_cast<const float4*>(static_cast<const int4*>(encoded)+3*groups)+group);
        #pragma unroll
        for(int t=0;t<Tokens;++t) {
            int4 x;float scale;
            if constexpr(Tokens==3) {
                x=__ldg(static_cast<const int4*>(encoded)+size_t(t)*groups+group);
                scale=t==0?scales.x:t==1?scales.y:scales.z;
            }else {
                const auto* p=static_cast<const AoS*>(encoded)+size_t(t)*groups+group;
                x=__ldg(&p->code);scale=__ldg(&p->scale);
            }
            sum[t]=fmaf(float(dot(w,x))*scale,delta,sum[t]);
        }
    }
    #pragma unroll
    for(int t=0;t<Tokens;++t) {
        #pragma unroll
        for(int d=16;d;d/=2)sum[t]+=__shfl_down_sync(0xffffffff,sum[t],d);
        if(!lane)output[size_t(t)*rows+row]=sum[t];
    }
}
}
void raw_q8_wide_project(const void* raw,const void* input,float* output,int rows,int columns,int tokens,cudaStream_t stream) {
    if(!raw||!input||!output||rows<1||columns<64||columns%64||tokens<1||tokens>3||static_cast<const void*>(output)==input||static_cast<const void*>(output)==raw)
        throw std::invalid_argument("Wide rawQ8 originalpayload/input extent or alias");
    const dim3 grid(unsigned((rows+3)/4));
    if(tokens==1)wide_head<1><<<grid,128,0,stream>>>(static_cast<const unsigned char*>(raw),input,output,rows,columns);
    else if(tokens==2)wide_head<2><<<grid,128,0,stream>>>(static_cast<const unsigned char*>(raw),input,output,rows,columns);
    else wide_head<3><<<grid,128,0,stream>>>(static_cast<const unsigned char*>(raw),input,output,rows,columns);
    check_cuda(cudaGetLastError(),"Wide rawQ8 originalweight projection");
}
}
