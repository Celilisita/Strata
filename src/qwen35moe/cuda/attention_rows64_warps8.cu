#include "attention_rows64_warps8.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <math_constants.h>
#include <cstdlib>
#include <stdexcept>
namespace strata::qwen35moe {
namespace {
#include "attention_rows64_warps8_body.cuh"
static thread_local unsigned long long submissions=0;
AttentionRows64Warps8Resources resources(){
    const auto p=actual_sm75();
    AttentionRows64Warps8Resources r{64,256,0,0,0,0,0,shared_bytes<64>(),0,p.sharedMemPerBlockOptin};
    if(p.major!=7||p.minor!=5||p.multiProcessorCount!=72){r.reason_mask=64;return r;}
    cudaFuncAttributes a{};check(cudaFuncGetAttributes(&a,query_tile<64,256>));
    r.registers=a.numRegs;r.binary_version=a.binaryVersion;r.static_shared=a.sharedSizeBytes;r.local_bytes=a.localSizeBytes;
    if(r.static_shared+r.dynamic_shared>r.optin_limit)r.reason_mask|=1;
    if(r.local_bytes)r.reason_mask|=2;
    if(a.numRegs<1||a.numRegs>255||a.numRegs*256>p.regsPerBlock||a.maxThreadsPerBlock<256)r.reason_mask|=4;
    if(a.binaryVersion!=75)r.reason_mask|=16;
    if(!r.reason_mask){
        const auto error=cudaFuncSetAttribute(query_tile<64,256>,cudaFuncAttributeMaxDynamicSharedMemorySize,int(r.dynamic_shared));
        if(error!=cudaSuccess){(void)cudaGetLastError();r.reason_mask|=32;}
        if(!r.reason_mask){check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&r.active_blocks,query_tile<64,256>,256,r.dynamic_shared));if(!r.active_blocks)r.reason_mask|=8;}
    }
    return r;
}
void validate(const float* q,const uint16_t* k,const uint16_t* v,int first,int count,int capacity,float* out,void* stream){
    if(!q||!k||!v||!out||!stream||!attention_tc_fits(first,count,capacity))throw std::invalid_argument("unsupported Rows64 attention geometry");
    const void* ptrs[4]={q,k,v,out};const size_t bytes[4]={size_t(count)*4096*4,size_t(capacity)*1024,size_t(capacity)*1024,size_t(count)*4096*4};
    for(int i=0;i<4;++i){const auto a=reinterpret_cast<uintptr_t>(ptrs[i]);
        if((a&15)||bytes[i]>UINTPTR_MAX-a)throw std::invalid_argument("Rows64 pointer alignment/span overflow");
        for(int j=0;j<i;++j){const auto b=reinterpret_cast<uintptr_t>(ptrs[j]);if(a<b+bytes[j]&&b<a+bytes[i])throw std::invalid_argument("Rows64 pointer alias");}}
}
}
bool attention_rows64_warps8_selected(int first,int count,int capacity){
    static const bool enabled=attention_rows64_warps8_flag(std::getenv("STRATA_QWEN35_ATTN_QUERY_ROWS64_WARPS8"));
    return enabled&&attention_rows64_warps8_candidate(first,count,capacity);
}
AttentionRows64Warps8Resources attention_rows64_warps8_resources(){return resources();}
unsigned long long attention_rows64_warps8_submissions(){return submissions;}
bool attention_rows64_warps8_prefill(const float* q,uint16_t* k,uint16_t* v,int first,int count,int capacity,float* out,void* stream){
    validate(q,k,v,first,count,capacity,out,stream);
    if(!attention_rows64_warps8_candidate(first,count,capacity)||resources().reason_mask)return false;
    const int width=first+count,padded=((width+63)/64)*64;const auto s=static_cast<cudaStream_t>(stream);
    if(padded>width)zero_tail<<<((padded-width)*CacheStride+255)/256,256,0,s>>>(reinterpret_cast<half*>(k),reinterpret_cast<half*>(v),width,(padded-width)*CacheStride);
    query_tile<64,256><<<dim3(Heads,(count+63)/64),256,shared_bytes<64>(),s>>>(q,reinterpret_cast<const half*>(k),reinterpret_cast<const half*>(v),first,count,out);
    check(cudaGetLastError());++submissions;return true;
}
}
