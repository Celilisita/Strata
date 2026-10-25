#include "strata/gemma4/native/vector_striped_q6_down.hpp"
#include "small_prefill_values.cuh"
#include <cuda_fp16.h>
#include <cstdint>
#include <limits>

namespace strata::gemma4::native {
namespace {
constexpr int Rows=5376,Columns=21504,Stride=224,Threads=128,RowsPerBlock=4;
constexpr size_t WeightBytes=size_t(Rows)*(Columns/256)*Stride;
constexpr size_t InputBytes=size_t(Columns/32)*sizeof(SingleQ8Block);
bool aligned(const void* p,size_t n){return p&&reinterpret_cast<std::uintptr_t>(p)%n==0;}
struct Span {const void* data;size_t bytes;};
void validate_span(Span s){
    if(!s.data||!s.bytes||s.bytes>std::numeric_limits<std::uintptr_t>::max()-
       reinterpret_cast<std::uintptr_t>(s.data))
        throw std::invalid_argument("Gemma262 span extent");
}
bool overlap(Span a,Span b){
    const auto x=reinterpret_cast<std::uintptr_t>(a.data),y=reinterpret_cast<std::uintptr_t>(b.data);
    return x<y+b.bytes&&y<x+a.bytes;
}
// Exact216 per-group signed scales, integer products and F32 terms. Only
// independent K terms and the final warp reduction are regrouped.
__device__ float term(const unsigned char* raw,const SingleQ8Block& input,int group,float prior){
    int first=0,second=0;
    #pragma unroll
    for(int word=0;word<8;++word){
        const int weight=primitive::detail::weight_codes<primitive::Quant::Q6K,true>(raw,group,word*4);
        const int code=reinterpret_cast<const int*>(input.code)[word];
        if(word<4)first=__dp4a(weight,code,first);
        else second=__dp4a(weight,code,second);
    }
    const auto* scales=reinterpret_cast<const signed char*>(raw+192)+group*2;
    const int dot=int(scales[0])*first+int(scales[1])*second;
    const float product=__fmaf_rn(input.d,float(dot),0.f);
    return __fmaf_rn(__half2float(*reinterpret_cast<const half*>(raw+208)),product,prior);
}
__global__ void multiply(AlignedQ6View matrix,const SingleQ8Block* input,float* output){
    const int row=int(blockIdx.x)*RowsPerBlock+threadIdx.x/32;
    const int lane=threadIdx.x%32,group=lane%8,stripe=lane/8;
    float value=0.f;
    for(int k=stripe;k<Columns/256;k+=4){
        const auto* raw=static_cast<const unsigned char*>(matrix.data)+
            (size_t(row)*(Columns/256)+k)*Stride;
        value=term(raw,input[k*8+group],group,value);
    }
    #pragma unroll
    for(int distance=4;distance;distance/=2)
        value=__fadd_rn(value,__shfl_xor_sync(0xffffffff,value,distance));
    value=__fadd_rn(value,__shfl_xor_sync(0xffffffff,value,16));
    value=__fadd_rn(value,__shfl_xor_sync(0xffffffff,value,8));
    if(!lane)output[row]=value;
}
}
void linear_q6_down_striped_t1(Q6DownStripeLease lease,float* output,Q6DownStripeBinding live){
    const auto& m=lease.weight;const auto& x=lease.input;const auto& b=lease.binding;
    if(!live.owner||!live.epoch||!live.stream||b.owner!=live.owner||b.epoch!=live.epoch||b.stream!=live.stream)
        throw std::invalid_argument("Gemma262 stale owner/epoch/stream");
    if(m.rows!=Rows||m.columns!=Columns||m.bytes<WeightBytes||!aligned(m.data,32)||
       x.columns!=Columns||x.tokens!=1||x.bytes<InputBytes||!aligned(x.data,alignof(SingleQ8Block))||
       !aligned(output,alignof(float)))
        throw std::invalid_argument("Gemma262 exact224 Down/Single40 T1 geometry");
    const Span spans[]{{m.data,m.bytes},{x.data,x.bytes},{output,size_t(Rows)*sizeof(float)}};
    for(const auto& span:spans)validate_span(span);
    for(int i=0;i<3;++i)for(int j=0;j<i;++j)
        if(overlap(spans[i],spans[j]))throw std::invalid_argument("Gemma262 borrowed spans alias");
    multiply<<<Rows/RowsPerBlock,Threads,0,live.stream>>>(m,x.data,output);
    primitive::check_cuda(cudaGetLastError(),"Gemma262 sameSingle40 Q6Down fourKstripes");
}
}
