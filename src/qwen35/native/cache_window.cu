#include "strata/qwen35/native/attention_window.hpp"
#include "strata/qwen35/native/ffn.hpp"
#include <cuda_fp16.h>
namespace strata::qwen35::native { namespace {
__global__ void encode(const float *x,unsigned char *cache,size_t blocks,int stride,const int *positions,int shift,int capacity){
    const size_t block=size_t(blockIdx.x)*4+threadIdx.x/32;if(block>=blocks)return;
    const int logical=positions[block/stride]-shift;if(logical<0||logical>=262144)return;
    const int lane=threadIdx.x%32;const float value=x[block*32+lane];float maximum=fabsf(value);
    for(int d=16;d;d/=2)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,d));
    const float delta=maximum/127.f;const int code=maximum?int(roundf(value*(127.f/maximum))):0;
    auto *out=cache+(size_t(logical%capacity)*stride+block%stride)*34;
    if(!lane)*reinterpret_cast<__half *>(out)=__float2half_rn(delta);
    reinterpret_cast<signed char *>(out+2)[lane]=static_cast<signed char>(max(-127,min(127,code)));
}
}
DecodeNode window_cache_node(void *function){return function==reinterpret_cast<void *>(encode)?DecodeNode::CacheWrite:DecodeNode::None;}
void q8_window_write(const float *x,void *cache,int dim,int heads,int tokens,const int *positions,int shift,int capacity,cudaStream_t stream){
    if(!x||!cache||!positions||dim!=256||heads<1||tokens<1||tokens>8192||(shift!=0&&shift!=1)||capacity<tokens)
        throw std::invalid_argument("invalid window Q8 write span");
    const int stride=heads*(dim/32);const size_t blocks=size_t(tokens)*stride;
    encode<<<unsigned((blocks+3)/4),128,0,stream>>>(x,static_cast<unsigned char *>(cache),blocks,stride,positions,shift,capacity);
    check_cuda(cudaGetLastError(),"sliding exact Q8 row write");
}
}
