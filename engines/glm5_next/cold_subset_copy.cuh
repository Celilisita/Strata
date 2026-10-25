#pragma once
#include "cold_subset_copy.hpp"
#include <new>
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
namespace strata::glm5_next::cold_copy_detail {
struct Copies { ColdSubsetCopy rows[6]; };
struct Timing {int device=-1;cudaEvent_t events[3]{};};
__global__ void copy_rows(Copies copies) {
    const auto& row=copies.rows[blockIdx.y];
    const auto* source=static_cast<const uint4*>(row.source);
    auto* destination=static_cast<uint4*>(row.destination);
    for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<row.bytes/16;i+=uint64_t(gridDim.x)*blockDim.x)
        destination[i]=source[i];
}
// Do not widen Copies: old count6 callers/kernel still pass exactly six rows.
struct Copies12 { ColdSubsetCopy rows[12]; };
static_assert(sizeof(Copies)==6*sizeof(ColdSubsetCopy), "legacy copy packet layout differs");
static_assert(sizeof(Copies12)==12*sizeof(ColdSubsetCopy), "four-key copy packet layout differs");
__global__ void copy_rows_twelve(Copies12 copies) {
    const auto& row=copies.rows[blockIdx.y];
    const auto* source=static_cast<const uint4*>(row.source);
    auto* destination=static_cast<uint4*>(row.destination);
    for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<row.bytes/16;i+=uint64_t(gridDim.x)*blockDim.x)
        destination[i]=source[i];
}
}
#endif
static bool strata_glm53_probe_cold_timing(ggml_backend_t backend,void** state,uint32_t action,
    strata::glm5_next::ColdSubsetGpuTimes* info) {
    if(!info)return false;*info={};
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    (void)backend;(void)state;(void)action;info->cuda_error=801;return false;
#else
    const auto fail=[&](cudaError_t e){info->cuda_error=uint32_t(e);return false;};
    if(!backend||!ggml_backend_is_cuda(backend)||!state||action>5)return fail(cudaErrorInvalidValue);
    auto* ctx=static_cast<ggml_backend_cuda_context*>(backend->context);ggml_cuda_set_device(ctx->device);
    if(ctx->curr_stream_no!=0)return fail(cudaErrorInvalidValue);
    using Timing=strata::glm5_next::cold_copy_detail::Timing;
    if(action==0){if(*state)return fail(cudaErrorInvalidValue);auto* made=new(std::nothrow) Timing;
        if(!made)return fail(cudaErrorMemoryAllocation);made->device=ctx->device;
        for(auto& event:made->events){const auto e=cudaEventCreate(&event);if(e!=cudaSuccess){
            for(auto prior:made->events)if(prior)(void)cudaEventDestroy(prior);delete made;return fail(e);}}
        *state=made;return true;}
    auto* timing=static_cast<Timing*>(*state);if(!timing||timing->device!=ctx->device)return fail(cudaErrorInvalidValue);
    if(action<=3){const auto e=cudaEventRecord(timing->events[action-1],ctx->stream());return e==cudaSuccess?true:fail(e);}
    if(action==4){const auto ready=cudaEventQuery(timing->events[2]);if(ready!=cudaSuccess)return fail(ready);
        auto e=cudaEventElapsedTime(&info->copy_ms,timing->events[0],timing->events[1]);if(e!=cudaSuccess)return fail(e);
        e=cudaEventElapsedTime(&info->first_ms,timing->events[1],timing->events[2]);if(e!=cudaSuccess)return fail(e);
        e=cudaEventElapsedTime(&info->total_ms,timing->events[0],timing->events[2]);return e==cudaSuccess?true:fail(e);}
    cudaError_t result=cudaSuccess;for(auto event:timing->events){const auto e=cudaEventDestroy(event);if(e!=cudaSuccess)result=e;}
    delete timing;*state=nullptr;return result==cudaSuccess?true:fail(result);
#endif
}
static bool strata_glm53_probe_copy_expert_rows(ggml_backend_t backend,
    const strata::glm5_next::ColdSubsetCopy* rows,size_t count,strata::glm5_next::ColdSubsetCopyInfo* info) {
    if(!info)return false;*info={};
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    (void)backend;(void)rows;(void)count;info->cuda_error=801;return false;
#else
    using namespace strata::glm5_next;
    const auto fail=[&](cudaError_t e){info->cuda_error=uint32_t(e);return false;};
    if(!backend||!ggml_backend_is_cuda(backend)||(count!=0&&count!=6&&count!=12)||(count&&!rows))return fail(cudaErrorInvalidValue);
    auto* context=static_cast<ggml_backend_cuda_context*>(backend->context);
    ggml_cuda_set_device(context->device);
    if(context->curr_stream_no!=0)return fail(cudaErrorInvalidValue);
    if(!count){size_t free=0,total=0;const auto e=cudaMemGetInfo(&free,&total);
        info->free_bytes=free;info->total_bytes=total;return e==cudaSuccess?true:fail(e);}
    if(count==12){
        cold_copy_detail::Copies12 copies{};
        uint64_t copied_bytes=0;
        bool empty=true;for(size_t i=0;i<count;++i)empty&=!rows[i].source&&!rows[i].destination&&!rows[i].bytes;
        if(!empty){
            for(size_t i=0;i<count;++i){const auto& row=rows[i];
                const auto source=reinterpret_cast<uintptr_t>(row.source),destination=reinterpret_cast<uintptr_t>(row.destination);
                if(!source||!destination||!row.bytes||row.bytes>(32u<<20)||row.bytes%16||source%16||destination%16||
                    source>UINTPTR_MAX-row.bytes||destination>UINTPTR_MAX-row.bytes)return fail(cudaErrorInvalidValue);
                copies.rows[i]=row;copied_bytes+=row.bytes;
            }
            // Distinct destination ranges and no source/destination overlap:
            // the single launch cannot mutate another descriptor's input.
            for(size_t i=0;i<count;++i){const auto begin=reinterpret_cast<uintptr_t>(rows[i].destination),end=begin+rows[i].bytes;
                for(size_t j=0;j<count;++j){const auto source=reinterpret_cast<uintptr_t>(rows[j].source);
                    if(begin<source+rows[j].bytes&&source<end)return fail(cudaErrorInvalidValue);
                    if(j<i){const auto prior=reinterpret_cast<uintptr_t>(rows[j].destination);
                        if(begin<prior+rows[j].bytes&&prior<end)return fail(cudaErrorInvalidValue);}
                }
            }
        }
        info->copied_bytes=copied_bytes;
        cold_copy_detail::copy_rows_twelve<<<dim3(48,12),256,0,context->stream()>>>(copies);
        const auto e=cudaGetLastError();return e==cudaSuccess?true:fail(e);
    }
    cold_copy_detail::Copies copies{};
    bool empty=true;for(size_t i=0;i<count;++i)empty&=!rows[i].source&&!rows[i].destination&&!rows[i].bytes;
    // Diagnostic candidate controls pay the same kernel geometry with no reads/writes.
    // Any partial-empty list still follows the unchanged strict nonempty validator.
    if(!empty)for(size_t i=0;i<count;++i){const auto& row=rows[i];
        if(!row.source||!row.destination||!row.bytes||row.bytes>(32u<<20)||row.bytes%16||
            reinterpret_cast<uintptr_t>(row.source)%16||reinterpret_cast<uintptr_t>(row.destination)%16)
            return fail(cudaErrorInvalidValue);
        copies.rows[i]=row;info->copied_bytes+=row.bytes;
    }
    cold_copy_detail::copy_rows<<<dim3(48,6),256,0,context->stream()>>>(copies);
    const auto e=cudaGetLastError();return e==cudaSuccess?true:fail(e);
#endif
}
