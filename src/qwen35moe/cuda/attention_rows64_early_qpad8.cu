#include "attention_rows64_early_qpad8.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <math_constants.h>
#include <cstdlib>
#include <stdexcept>
namespace strata::qwen35moe {
namespace {
namespace early_qpad8 {
// Frozen Qpad8 body: only shared Q pitch264; same Rows64 math and128-thread geometry.
#include "attention_rows64_qpad8_body.cuh"
}
static thread_local unsigned long long submissions=0;
AttentionRows64Resources resources(){
    const auto p=early_qpad8::actual_sm75();
    AttentionRows64Resources r{64,128,0,0,0,0,0,early_qpad8::shared_bytes<64>(),0,p.sharedMemPerBlockOptin};
    if(p.major!=7||p.minor!=5||p.multiProcessorCount!=72){r.reason_mask=64;return r;}
    cudaFuncAttributes a{};early_qpad8::check(cudaFuncGetAttributes(&a,early_qpad8::query_tile_qpad8<64,128>));
    r.registers=a.numRegs;r.binary_version=a.binaryVersion;r.static_shared=a.sharedSizeBytes;r.local_bytes=a.localSizeBytes;
    if(r.static_shared+r.dynamic_shared>r.optin_limit)r.reason_mask|=1;
    if(r.local_bytes)r.reason_mask|=2;
    if(a.numRegs<1||a.numRegs>255||a.numRegs*128>p.regsPerBlock||a.maxThreadsPerBlock<128)r.reason_mask|=4;
    if(a.binaryVersion!=75)r.reason_mask|=16;
    if(!r.reason_mask){
        const auto error=cudaFuncSetAttribute(early_qpad8::query_tile_qpad8<64,128>,cudaFuncAttributeMaxDynamicSharedMemorySize,int(r.dynamic_shared));
        if(error!=cudaSuccess){(void)cudaGetLastError();r.reason_mask|=32;}
        if(!r.reason_mask){early_qpad8::check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&r.active_blocks,early_qpad8::query_tile_qpad8<64,128>,128,r.dynamic_shared));if(!r.active_blocks)r.reason_mask|=8;}
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
bool attention_rows64_early_qpad8_selected(int first,int count,int capacity){
    if(!attention_rows64_early_selected(first,count,capacity))return false;
    static const bool qpad=attention_rows64_qpad8_flag(std::getenv("STRATA_QWEN35_ATTN_ROWS64_QPAD8"));
    return attention_rows64_early_qpad8_candidate(true,qpad,first,count,capacity);
}
AttentionRows64Resources attention_rows64_early_qpad8_resources(){return resources();}
unsigned long long attention_rows64_early_qpad8_submissions(){return submissions;}
bool attention_rows64_early_qpad8_prefill(const float* q,uint16_t* k,uint16_t* v,int first,int count,int capacity,float* out,void* stream){
    validate(q,k,v,first,count,capacity,out,stream);
    if(!attention_rows64_early_candidate(first,count,capacity)||resources().reason_mask)return false;
    const auto s=static_cast<cudaStream_t>(stream);
    // All three bounded shapes end on64; no padding kernel or new storage is needed.
    early_qpad8::query_tile_qpad8<64,128><<<dim3(early_qpad8::Heads,(count+63)/64),128,early_qpad8::shared_bytes<64>(),s>>>(q,reinterpret_cast<const half*>(k),reinterpret_cast<const half*>(v),first,count,out);
    early_qpad8::check(cudaGetLastError());++submissions;return true;
}
}
