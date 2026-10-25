#include "strata/qwen35/completed_v100/native/nvfp4_repack.hpp"
#include "strata/qwen35/completed_v100/native/ffn.hpp"
#include <cstdint>
namespace strata::completed_v100::qwen35::native {
namespace {
__device__ unsigned compact(unsigned x) {
    x&=0x0f0f0f0f;x=(x|(x>>4))&0x00ff00ff;
    return (x|(x>>8))&0xffff;
}
__device__ unsigned spread(unsigned x) {
    x=(x|(x<<8))&0x00ff00ff;return (x|(x<<4))&0x0f0f0f0f;
}
template<bool Inverse>__global__ void reorder(unsigned char* data,size_t blocks) {
    const size_t block=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(block>=blocks)return;
    auto* words=reinterpret_cast<unsigned*>(data+block*36+4);
    #pragma unroll
    for(int group=0;group<4;++group) {
        const unsigned a=words[group*2],b=words[group*2+1];
        if constexpr(Inverse) {
            words[group*2]=spread(a&0xffff)|(spread(b&0xffff)<<4);
            words[group*2+1]=spread(a>>16)|(spread(b>>16)<<4);
        }else {
            words[group*2]=compact(a)|(compact(b)<<16);
            words[group*2+1]=compact(a>>4)|(compact(b>>4)<<16);
        }
    }
}
}
void nvfp4_repack_gpu(void* payload,size_t blocks,bool inverse,cudaStream_t stream) {
    if(!payload||!blocks||blocks>size_t(1)<<31||reinterpret_cast<uintptr_t>(payload)%4)
        throw std::invalid_argument("NVFP4 lossless GPU permutation extent");
    if(inverse)reorder<true><<<unsigned((blocks+127)/128),128,0,stream>>>(static_cast<unsigned char*>(payload),blocks);
    else reorder<false><<<unsigned((blocks+127)/128),128,0,stream>>>(static_cast<unsigned char*>(payload),blocks);
    check_cuda(cudaGetLastError(),"NVFP4 reversible GPU nibble permutation");
}
}
