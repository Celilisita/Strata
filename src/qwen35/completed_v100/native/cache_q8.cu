#include "strata/qwen35/completed_v100/native/cache_q8.hpp"
#include "strata/qwen35/completed_v100/native/ffn.hpp"
#include <cuda_fp16.h>
namespace strata::completed_v100::qwen35::native {
namespace {
template<bool Dynamic> __global__ void encode(const float *x,unsigned char *cache,
    size_t blocks,size_t first,const int *position=nullptr){
    if constexpr(Dynamic){const int p=*position;if(p<0||p>=262144)return;first=size_t(p)*blocks;}
    const size_t block=size_t(blockIdx.x)*4+threadIdx.x/32;
    if(block>=blocks)return;
    const int lane=threadIdx.x%32;const float value=x[block*32+lane];
    float maximum=fabsf(value);
    for(int d=16;d;d/=2)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,d));
    const float delta=maximum/127.0f;
    const int code=maximum?int(roundf(value*(127.0f/maximum))):0;
    auto *out=cache+(first+block)*34;
    if(!lane)*reinterpret_cast<__half *>(out)=__float2half_rn(delta);
    reinterpret_cast<signed char *>(out+2)[lane]=static_cast<signed char>(max(-127,min(127,code)));
}
}
void q8_cache_write(const float *x,void *cache,int dim,int heads,int tokens,int start,int capacity,cudaStream_t s){
    if(!x||!cache||dim<32||dim%32||heads<1||tokens<1||start<0||capacity<tokens||start>capacity-tokens)
        throw std::invalid_argument("invalid own Q8 cache extent");
    const size_t blocks=size_t(tokens)*heads*(dim/32),first=size_t(start)*heads*(dim/32);
    encode<false><<<unsigned((blocks+3)/4),128,0,s>>>(x,static_cast<unsigned char *>(cache),blocks,first);
    check_cuda(cudaGetLastError(),"own Q8 KV write");
}
void q8_cache_write_position(const float *x,void *cache,int dim,int heads,const int *position,cudaStream_t s){
    if(!x||!cache||!position||dim<32||dim%32||heads<1)throw std::invalid_argument("invalid own Q8 position");
    const size_t blocks=size_t(heads)*(dim/32);
    encode<true><<<unsigned((blocks+3)/4),128,0,s>>>(x,static_cast<unsigned char *>(cache),blocks,0,position);
    check_cuda(cudaGetLastError(),"own Q8 KV cursor write");
}
DecodeNode decode_q8_cache_node(void *p){
    return p==reinterpret_cast<void *>(encode<false>)||p==reinterpret_cast<void *>(encode<true>)?
        DecodeNode::CacheWrite:DecodeNode::None;
}
}
