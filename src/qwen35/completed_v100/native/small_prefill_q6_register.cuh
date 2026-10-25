#pragma once
#include "small_prefill_mma.cuh"
#include "small_prefill_half_input.cuh"

namespace strata::completed_v100::qwen35::native::detail {
struct RegisterQ6Shared {
    unsigned input[32][CompactHalfOperandPitch];
    float delta[8][32];
    float2 coefficient[8][64];
    float weight_delta[64];
};
static_assert(sizeof(RegisterQ6Shared)==13824,"register Q6 shared extent");
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800
__device__ __forceinline__ void register_q6_mma(uint4 a,uint4 b,float (&sum)[8]) {
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7},{%8,%9},{%10,%11},{%0,%1,%2,%3,%4,%5,%6,%7};"
                 : "+f"(sum[0]),"+f"(sum[1]),"+f"(sum[2]),"+f"(sum[3]),
                   "+f"(sum[4]),"+f"(sum[5]),"+f"(sum[6]),"+f"(sum[7])
                 : "r"(a.x),"r"(a.y),"r"(b.x),"r"(b.y));
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7},{%8,%9},{%10,%11},{%0,%1,%2,%3,%4,%5,%6,%7};"
                 : "+f"(sum[0]),"+f"(sum[1]),"+f"(sum[2]),"+f"(sum[3]),
                   "+f"(sum[4]),"+f"(sum[5]),"+f"(sum[6]),"+f"(sum[7])
                 : "r"(a.z),"r"(a.w),"r"(b.z),"r"(b.w));
}
#endif
// Four warps own64 rows*32 tokens. Each loaded A operand serves both N16
// fragments. The borrowed byte-code inventory is never materialized to shared.
__global__ __launch_bounds__(128,4)
void register_q6_project(Matrix weight,const HalfInputBlock *input,float *output) {
    #if __CUDA_ARCH__ >= 800
    return;
    #else
    __shared__ __align__(16) RegisterQ6Shared s;
    const int tid=threadIdx.x,warp=tid/32,lane=tid%32,first=int(blockIdx.x)*64;
    const int blocks=weight.columns/256,matrix=(lane%16)/4,local=(lane%4)+(lane>=16?4:0);
    const int row=first+warp*16+(matrix/2)*8+local;
    const auto *base=static_cast<const unsigned char *>(weight.data);
    float result[2][8]{};
    for(int block=0;block<blocks;++block) {
        if(tid<64) {
            const auto *meta=base+size_t(weight.rows)*blocks*256+(size_t(first+tid)*blocks+block)*18;
            #pragma unroll
            for(int g=0;g<8;++g) {
                const unsigned bits=__ldg(reinterpret_cast<const unsigned short *>(meta)+g);
                s.coefficient[g][tid]=make_float2(float(int(static_cast<signed char>(bits&255u))),
                                                 float(int(static_cast<signed char>(bits>>8))));
            }
            s.weight_delta[tid]=__half2float(__ushort_as_half(__ldg(reinterpret_cast<const unsigned short *>(meta)+8)));
        }
        for(int i=tid;i<256;i+=128)s.delta[i/32][i%32]=input[block].delta[i/32][i%32];
        for(int chunk=0;chunk<2;++chunk) {
            for(int token=warp;token<32;token+=4)
                *reinterpret_cast<uint2 *>(s.input[token]+lane*2)=__ldg(reinterpret_cast<const uint2 *>(input[block].codes[token]+chunk*64)+lane);
            __syncthreads();
            float weight_delta[2];
            #pragma unroll
            for(int r=0;r<2;++r)weight_delta[r]=s.weight_delta[result_row(warp*2,lane,r*2)];
            #pragma unroll
            for(int pair=0;pair<2;++pair) {
                float scaled[2][8]{};
                #pragma unroll
                for(int parity=0;parity<2;++parity) {
                    const int group=chunk*4+pair*2+parity,g=pair*2+parity;
                    float dot[2][2][8]{};
                    #pragma unroll
                    for(int part=0;part<2;++part) {
                        const auto *p=base+(size_t(row)*blocks+block)*256+group*32+part*16;
                        const uint4 raw=__ldg(reinterpret_cast<const uint4 *>(p));
                        #pragma unroll
                        for(int step=0;step<2;++step) {
                            const unsigned x=step?raw.z:raw.x,y=step?raw.w:raw.y;
                            const uint4 a=make_uint4(half_codes(x,true,0),half_codes(x,true,2),half_codes(y,true,0),half_codes(y,true,2));
                            #pragma unroll
                            for(int n=0;n<2;++n) {
                                const int token=n*16+(matrix%2)*8+local;
                                const uint4 b=input_operand4<true>(&s.input[0][0],token,g*16+part*8+step*4);
                                register_q6_mma(a,b,dot[n][part]);
                            }
                        }
                    }
                    float2 coefficient[2];
                    #pragma unroll
                    for(int r=0;r<2;++r)coefficient[r]=s.coefficient[group][result_row(warp*2,lane,r*2)];
                    #pragma unroll
                    for(int n=0;n<2;++n) {
                        float delta[4];
                        #pragma unroll
                        for(int t=0;t<4;++t)delta[t]=s.delta[group][result_token(warp*2+n,lane,(t&1)+(t/2)*4)];
                        #pragma unroll
                        for(int i=0;i<8;++i) {
                            const int r=(i/2)%2,t=(i&1)+(i/4)*2;
                            const float weighted=__fmaf_rn(coefficient[r].x,dot[n][0][i],__fmul_rn(coefficient[r].y,dot[n][1][i]));
                            scaled[n][i]=__fmaf_rn(delta[t],weighted,scaled[n][i]);
                        }
                    }
                }
                #pragma unroll
                for(int n=0;n<2;++n) {
                    #pragma unroll
                    for(int i=0;i<8;++i)result[n][i]=__fmaf_rn(weight_delta[(i/2)%2],scaled[n][i],result[n][i]);
                }
            }
            __syncthreads();
        }
    }
    #pragma unroll
    for(int n=0;n<2;++n) {
        #pragma unroll
        for(int i=0;i<8;++i)output[size_t(result_token(warp*2+n,lane,i))*weight.rows+first+result_row(warp*2+n,lane,i)]=__fadd_rn(result[n][i],0.0f);
    }
    #endif
}
}
