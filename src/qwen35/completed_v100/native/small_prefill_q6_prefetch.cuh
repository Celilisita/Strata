#pragma once
#include "small_prefill_mma.cuh"
#include "small_prefill_half_input.cuh"
#include "small_prefill_values.cuh"
#include "small_prefill_q6_raw_words.cuh"

namespace strata::completed_v100::qwen35::native::detail {
struct PrefetchQ6Shared {
    unsigned weight[64][HalfOperandPitch];
    unsigned input[32][CompactHalfOperandPitch];
    float delta[8][32];
    float2 coefficient[8][64];
    float weight_delta[64];
};
static_assert(sizeof(PrefetchQ6Shared)==31232,"prefetch Q6 shared extent");
// Keep eight warps per64x32 tile; issue eight independent row loads before
// converting/storing their payload. The same compact inventory serves all rows.
template<bool Raw=false,int Partitions=1,bool Planes=false,bool BytePlanes=false>
__global__ __launch_bounds__(256,2)
void prefetch_q6_project(Matrix weight,const HalfInputBlock *input,float *output) {
    #if __CUDA_ARCH__ >= 800
    return;
    #else
    __shared__ __align__(16) PrefetchQ6Shared s;
    static_assert(!BytePlanes||Planes);
    constexpr int PlaneCodeBytes=BytePlanes?256:192,PlaneTileBytes=64*(PlaneCodeBytes+18);
    const int tid=threadIdx.x,warp=tid/32,lane=tid%32,first=int(blockIdx.x)*64,blocks=weight.columns/256;
    const auto *base=static_cast<const unsigned char *>(weight.data);
    float result[8]{};
    static_assert(Partitions==1||(Raw&&Partitions==2));
    const int begin=int(blockIdx.z)*blocks/Partitions,end=int(blockIdx.z+1)*blocks/Partitions;
    for(int block=begin;block<end;++block) {
        if(tid<64) {
            const size_t index=size_t(first+tid)*blocks+block;
            const size_t tile=(size_t(first+tid)/64)*blocks+block;
            const auto *meta=Planes?base+tile*PlaneTileBytes+64*PlaneCodeBytes+(tid%64)*2:
                Raw?base+index*210+192:base+size_t(weight.rows)*blocks*256+index*18;
            #pragma unroll
            for(int g=0;g<8;++g) {
                const unsigned bits=__ldg(reinterpret_cast<const unsigned short *>(meta)+(Planes?g*64:g));
                s.coefficient[g][tid]=make_float2(float(int(static_cast<signed char>(bits&255u))),
                                                 float(int(static_cast<signed char>(bits>>8))));
            }
            s.weight_delta[tid]=__half2float(__ushort_as_half(__ldg(reinterpret_cast<const unsigned short *>(meta)+(Planes?512:8))));
        }
        s.delta[tid/32][tid%32]=input[block].delta[tid/32][tid%32];
        float weight_delta[2];
        for(int chunk=0;chunk<2;++chunk) {
            for(int token=warp;token<32;token+=8)
                *reinterpret_cast<uint2 *>(s.input[token]+lane*2)=__ldg(reinterpret_cast<const uint2 *>(input[block].codes[token]+chunk*64)+lane);
            unsigned packed[8];
            if constexpr(BytePlanes){
                const size_t tile=size_t(first/64)*blocks+block;
                #pragma unroll
                for(int r=0;r<8;++r)
                    packed[r]=__ldg(reinterpret_cast<const unsigned *>(base+tile*PlaneTileBytes+(warp+r*8)*256+chunk*128)+lane);
            }else if constexpr(Planes){
                const size_t tile=size_t(first/64)*blocks+block;
                #pragma unroll
                for(int r=0;r<8;++r)
                    packed[r]=raw_q6_codes<true>(base+tile*PlaneTileBytes+(warp+r*8)*192,chunk*4+lane/8,(lane%8)*4);
            }else if constexpr(Raw){
                // All eight row indices in a warp share the same alignment.
                const bool aligned=((size_t(first+warp)*blocks+block)&1u)==0;
                if(aligned){
                    #pragma unroll
                    for(int r=0;r<8;++r){const size_t index=size_t(first+warp+r*8)*blocks+block;
                        packed[r]=raw_q6_codes<true>(base+index*210,chunk*4+lane/8,(lane%8)*4);
                    }
                }else{
                    #pragma unroll
                    for(int r=0;r<8;++r){const size_t index=size_t(first+warp+r*8)*blocks+block;
                        packed[r]=raw_q6_codes<false>(base+index*210,chunk*4+lane/8,(lane%8)*4);
                    }
                }
            }else{
                #pragma unroll
                for(int r=0;r<8;++r){const size_t index=size_t(first+warp+r*8)*blocks+block;
                    packed[r]=__ldg(reinterpret_cast<const unsigned *>(base+index*256+chunk*128)+lane);
                }
            }
            #pragma unroll
            for(int r=0;r<8;++r)
                *reinterpret_cast<uint2 *>(s.weight[warp+r*8]+lane*2)=make_uint2(half_codes(packed[r],true,0),half_codes(packed[r],true,2));
            __syncthreads();
            if(!chunk) {
                #pragma unroll
                for(int r=0;r<2;++r)weight_delta[r]=s.weight_delta[result_row(warp,lane,r*2)];
            }
            #pragma unroll
            for(int pair=0;pair<2;++pair) {
                float scaled[8]{};
                #pragma unroll
                for(int parity=0;parity<2;++parity) {
                    const int local_group=pair*2+parity,group=chunk*4+local_group;
                    float one[8]{},two[8]{};
                    code_dot16<64,true>(&s.weight[0][0],&s.input[0][0],local_group,0,warp,lane,one);
                    code_dot16<64,true>(&s.weight[0][0],&s.input[0][0],local_group,1,warp,lane,two);
                    float2 coefficient[2];float delta[4];
                    #pragma unroll
                    for(int r=0;r<2;++r)coefficient[r]=s.coefficient[group][result_row(warp,lane,r*2)];
                    #pragma unroll
                    for(int t=0;t<4;++t)delta[t]=s.delta[group][result_token(warp,lane,(t&1)+(t/2)*4)];
                    #pragma unroll
                    for(int i=0;i<8;++i) {
                        const int r=(i/2)%2,t=(i&1)+(i/4)*2;
                        const float weighted=__fmaf_rn(coefficient[r].x,one[i],__fmul_rn(coefficient[r].y,two[i]));
                        scaled[i]=__fmaf_rn(delta[t],weighted,scaled[i]);
                    }
                }
                #pragma unroll
                for(int i=0;i<8;++i)result[i]=__fmaf_rn(weight_delta[(i/2)%2],scaled[i],result[i]);
            }
            __syncthreads();
        }
    }
    #pragma unroll
    for(int i=0;i<8;++i)output[(size_t(blockIdx.z)*32+result_token(warp,lane,i))*weight.rows+first+result_row(warp,lane,i)]=__fadd_rn(result[i],0.0f);
    #endif
}
}
