#include "strata/qwen35/completed_v100/native/mtp_position.hpp"
#include "strata/qwen35/completed_v100/native/ffn.hpp"
#include <cuda_fp16.h>
namespace strata::completed_v100::qwen35::native {
namespace {
__global__ void positions(int *out,const int *start,int offset,int tokens,int shift,
                           DecodeCursor *cursor,int heads,DecodeRounding rounding){
    if(!threadIdx.x){
        const int p=*start+offset;
        for(int t=0;t<tokens;++t)out[t]=p+t;
        if(cursor){cursor->invalid=0;cursor->index=0;cursor->position=p+tokens-1-shift;
            cursor->partitions=decode_partitions_unchecked(cursor->position+1,heads,rounding);}
    }
}
__global__ void cache(const float *x,unsigned char *out,size_t blocks,int stride,
                       const int *positions,int shift,int capacity){
    const size_t block=size_t(blockIdx.x)*4+threadIdx.x/32;if(block>=blocks)return;
    const int token=int(block/stride),p=positions[token]-shift;
    if(p<0||p>=capacity)return;
    const int lane=threadIdx.x%32;const float value=x[block*32+lane];float maximum=fabsf(value);
    for(int d=16;d;d/=2)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,d));
    const float delta=maximum/127.0f;const int code=maximum?int(roundf(value*(127.0f/maximum))):0;
    auto *target=out+(size_t(p)*stride+block%stride)*34;
    if(!lane)*reinterpret_cast<__half *>(target)=__float2half_rn(delta);
    reinterpret_cast<signed char *>(target+2)[lane]=static_cast<signed char>(max(-127,min(127,code)));
}
}
void mtp_positions(int *out,const int *start,int offset,int tokens,int shift,
    DecodeCursor *cursor,int heads,DecodeRounding rounding,cudaStream_t stream){
    if(!out||!start||offset<0||offset>2||tokens<1||tokens>3||(shift!=0&&shift!=1))
        throw std::invalid_argument("invalid bounded MTP graph position");
    positions<<<1,32,0,stream>>>(out,start,offset,tokens,shift,cursor,heads,rounding);
    check_cuda(cudaGetLastError(),"MTP graph actual position and KV partition");
}
void q8_cache_write_positions(const float *x,void *out,int dim,int heads,int tokens,
    const int *positions,int shift,int capacity,cudaStream_t stream){
    if(!x||!out||!positions||dim%32||heads<1||tokens<1||tokens>3||(shift!=0&&shift!=1)||capacity<3)
        throw std::invalid_argument("invalid bounded MTP graph Q8 span");
    const int stride=heads*(dim/32);const size_t blocks=size_t(tokens)*stride;
    cache<<<unsigned((blocks+3)/4),128,0,stream>>>(x,static_cast<unsigned char *>(out),blocks,stride,positions,shift,capacity);
    check_cuda(cudaGetLastError(),"MTP graph actual Q8 cache row write");
}
}
