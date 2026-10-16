#include "small_prefill_half_input.cuh"
#include "small_prefill_half_raw.hpp"
#include "small_prefill_half_group.cuh"
#include "strata/qwen35/native/ffn.hpp"
#include <cuda_fp16.h>

namespace strata::qwen35::native {
namespace {
template<class Block,bool Affine=false>
__global__ void encode_half(const float *input,Block *output,int columns,int tokens){
    const int group=int(blockIdx.x)*16+threadIdx.x/8,lane=threadIdx.x%8,token=int(blockIdx.y);
    if(group>=columns/32)return;
    const float4 v=token<tokens?reinterpret_cast<const float4 *>(input+size_t(token)*columns)[group*8+lane]:
                              make_float4(0,0,0,0);
    detail::encode_half_group<Block,Affine>(v,output,group,token,(tokens+31)/32,lane);
}
}
size_t small_prefill_half_input_bytes(int columns,int tokens){
    if(columns<=0||columns%256||tokens<2||tokens>=64)
        throw std::invalid_argument("prepared half small PP shape");
    return size_t(columns/256)*size_t((tokens+31)/32)*sizeof(detail::HalfInputBlock);
}
namespace {
void validate_half_input(const float *input,void *output,int columns,int tokens){
    const size_t bytes=small_prefill_half_input_bytes(columns,tokens);
    const size_t source_bytes=size_t(columns)*size_t(tokens)*sizeof(float);
    const uintptr_t source=reinterpret_cast<uintptr_t>(input),destination=reinterpret_cast<uintptr_t>(output);
    if(!input||!output||source%16||destination%16||
       (source<=destination?destination-source<source_bytes:source-destination<bytes))
        throw std::invalid_argument("prepared half input requires aligned distinct buffers");
}
}
void small_prefill_half_quantize(const float *input,void *output,int columns,int tokens,cudaStream_t stream){
    validate_half_input(input,output,columns,tokens);
    encode_half<detail::HalfInputBlock><<<dim3((columns/32+15)/16,((tokens+31)/32)*32),128,0,stream>>>(
        input,static_cast<detail::HalfInputBlock *>(output),columns,tokens);
    check_cuda(cudaGetLastError(),"owned prepared half input quantization");
}
void small_prefill_half_raw_quantize(const float *input,void *output,Quant type,int columns,int tokens,cudaStream_t stream){
    validate_half_input(input,output,columns,tokens);
    if(type==Quant::Q6K){small_prefill_half_quantize(input,output,columns,tokens,stream);return;}
    if(type!=Quant::Q4K)throw std::invalid_argument("unsupported prepared-input quant format");
    encode_half<detail::HalfAffineInputBlock,true><<<dim3((columns/32+15)/16,((tokens+31)/32)*32),128,0,stream>>>(
        input,static_cast<detail::HalfAffineInputBlock *>(output),columns,tokens);
    check_cuda(cudaGetLastError(),"owned affine half input quantization");
}
}
