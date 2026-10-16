#include "small_prefill_blocked.cuh"
#include "strata/qwen35/native/ffn.hpp"
#include <cuda_fp16.h>

namespace strata::qwen35::native {
namespace {
__global__ void encode_vector(const float *input,detail::VectorBlock *output,int columns,int tokens){
    const int group=int(blockIdx.x)*16+threadIdx.x/8,lane=threadIdx.x%8,token=int(blockIdx.y);
    if(group>=columns/32)return;
    const float4 v=token<tokens?reinterpret_cast<const float4 *>(input+size_t(token)*columns)[group*8+lane]:make_float4(0,0,0,0);
    float maximum=fmaxf(fmaxf(fabsf(v.x),fabsf(v.y)),fmaxf(fabsf(v.z),fabsf(v.w)));
    float sum=__fadd_rn(__fadd_rn(__fadd_rn(v.x,v.y),v.z),v.w);
    for(int offset=4;offset;offset/=2){
        maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,offset));
        sum=__fadd_rn(sum,__shfl_xor_sync(0xffffffff,sum,offset));
    }
    const float reciprocal=maximum?__fdividef(127.0f,maximum):0;
    const float delta=maximum?__fdividef(1.0f,reciprocal):0;
    const char4 codes=make_char4(static_cast<signed char>(roundf(__fmul_rn(v.x,reciprocal))),
                               static_cast<signed char>(roundf(__fmul_rn(v.y,reciprocal))),
                               static_cast<signed char>(roundf(__fmul_rn(v.z,reciprocal))),
                               static_cast<signed char>(roundf(__fmul_rn(v.w,reciprocal))));
    auto &block=output[(group/8)*((tokens+31)/32)+token/32];
    block.codes[(group%8)*8+lane][token%32]=*reinterpret_cast<const int *>(&codes);
    if(!lane){
        block.delta[group%8][token%32]=delta;
        block.ds[group%8][token%32]=unsigned(__half_as_ushort(__float2half_rn(delta)))|
                                 (unsigned(__half_as_ushort(__float2half_rn(sum)))<<16);
    }
}
}
size_t small_prefill_vector_bytes(int columns,int tokens){
    if(columns<=0||columns%256||tokens<2||tokens>=64)throw std::invalid_argument("blocked small PP shape");
    return size_t(columns/256)*((tokens+31)/32)*sizeof(detail::VectorBlock);
}
void small_prefill_vector_quantize(const float *input,void *output,int columns,int tokens,cudaStream_t stream){
    small_prefill_vector_bytes(columns,tokens);
    encode_vector<<<dim3((columns/32+15)/16,((tokens+31)/32)*32),128,0,stream>>>(
        input,static_cast<detail::VectorBlock *>(output),columns,tokens);
    check_cuda(cudaGetLastError(),"owned blocked input quantization");
}
}
