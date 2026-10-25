#include "attention_rows64_early.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <math_constants.h>
#include <cstdlib>
#include <stdexcept>
namespace strata::qwen35moe {
namespace {
namespace early {
// Same frozen Rows64 arithmetic and128-thread geometry, with a distinct TU namespace.
#include "attention_rows64_body.cuh"
}
static thread_local unsigned long long submissions=0;
AttentionRows64Resources resources(){
    const auto p=early::actual_sm75();
    AttentionRows64Resources r{64,128,0,0,0,0,0,early::shared_bytes<64>(),0,p.sharedMemPerBlockOptin};
    if(p.major!=7||p.minor!=5||p.multiProcessorCount!=72){r.reason_mask=64;return r;}
    cudaFuncAttributes a{};early::check(cudaFuncGetAttributes(&a,early::query_tile<64,128>));
    r.registers=a.numRegs;r.binary_version=a.binaryVersion;r.static_shared=a.sharedSizeBytes;r.local_bytes=a.localSizeBytes;
    if(r.static_shared+r.dynamic_shared>r.optin_limit)r.reason_mask|=1;
    if(r.local_bytes)r.reason_mask|=2;
    if(a.numRegs<1||a.numRegs>255||a.numRegs*128>p.regsPerBlock||a.maxThreadsPerBlock<128)r.reason_mask|=4;
    if(a.binaryVersion!=75)r.reason_mask|=16;
    if(!r.reason_mask){
        const auto error=cudaFuncSetAttribute(early::query_tile<64,128>,cudaFuncAttributeMaxDynamicSharedMemorySize,int(r.dynamic_shared));
        if(error!=cudaSuccess){(void)cudaGetLastError();r.reason_mask|=32;}
        if(!r.reason_mask){early::check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&r.active_blocks,early::query_tile<64,128>,128,r.dynamic_shared));if(!r.active_blocks)r.reason_mask|=8;}
    }
    return r;
}
void validate(const float* q,const uint16_t* k,const uint16_t* v,int first,int count,int capacity,float* out,void* stream){
    if(!q||!k||!v||!out||!stream||!attention_tc_fits(first,count,capacity))throw std::invalid_argument("unsupported early Rows64 geometry");
    const void* ptrs[4]={q,k,v,out};const size_t bytes[4]={size_t(count)*4096*4,size_t(capacity)*1024,size_t(capacity)*1024,size_t(count)*4096*4};
    for(int i=0;i<4;++i){const auto a=reinterpret_cast<uintptr_t>(ptrs[i]);
        if((a&15)||bytes[i]>UINTPTR_MAX-a)throw std::invalid_argument("early Rows64 pointer alignment/span overflow");
        for(int j=0;j<i;++j){const auto b=reinterpret_cast<uintptr_t>(ptrs[j]);if(a<b+bytes[j]&&b<a+bytes[i])throw std::invalid_argument("early Rows64 pointer alias");}}
}
}
bool attention_rows64_early_selected(int first,int count,int capacity){
    static const bool enabled=attention_rows64_early_flag(std::getenv("STRATA_QWEN35_ATTN_ROWS64_EARLY"));
    return enabled&&attention_rows64_early_candidate(first,count,capacity);
}
AttentionRows64Resources attention_rows64_early_resources(){return resources();}
unsigned long long attention_rows64_early_submissions(){return submissions;}
bool attention_rows64_early_prefill(const float* q,uint16_t* k,uint16_t* v,int first,int count,int capacity,float* out,void* stream){
    validate(q,k,v,first,count,capacity,out,stream);
    if(!attention_rows64_early_candidate(first,count,capacity)||resources().reason_mask)return false;
    const auto s=static_cast<cudaStream_t>(stream);
    // All three bounded shapes end on64; no padding kernel or new storage is needed.
    early::query_tile<64,128><<<dim3(early::Heads,(count+63)/64),128,early::shared_bytes<64>(),s>>>(q,reinterpret_cast<const half*>(k),reinterpret_cast<const half*>(v),first,count,out);
    early::check(cudaGetLastError());++submissions;return true;
}
}
