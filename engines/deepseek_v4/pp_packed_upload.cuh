#pragma once
#include "pp_stream_graph.hpp"
#include "pp_stream_plan.hpp"
#include "packed_expert_store.hpp"
#include "pp_packed_prefetch.cuh"
#include "pp_stream_metrics.hpp"
#include "ggml-cuda/common.cuh"
#include <stdexcept>
namespace strata::deepseek_v4 {
inline void pp_packed_upload(const PpStreamSources& sources,const PpStreamGroup& group,
    const PackedExpertSpan& packed,void* destination,cudaStream_t copy,PpStreamMetrics& metrics,
    const PpPrefetchedSpan* prefetched=nullptr) {
    const size_t width=packed.expert_stride;
    if(!packed.data||width!=2*sources.gate->nb[2]+sources.down->nb[2]||packed.bytes!=width*group.experts||
       packed.gate_bytes!=sources.gate->nb[2]||packed.up_bytes!=sources.up->nb[2])
        throw std::runtime_error("PP packed expert payload/key projection differs");
    const auto* slots=sources.cache?static_cast<const int32_t*>(sources.cache->directory->data):nullptr;
    for(int first=0;first<group.experts;) {
        const int slot=slots?slots[group.first_expert+first]:-1;
        if(slot>=0) {
            int end=first+1;
            while(end<group.experts&&slots[group.first_expert+end]==slot+end-first)++end;
            size_t offset=0;
            for(int role=0;role<3;++role) {
                const auto* bank=sources.cache->bank[role];
                const size_t role_bytes=bank->nb[2];
                if(slot+end-first>bank->ne[2])throw std::runtime_error("PP packed cache lease outside bank");
                CUDA_CHECK(cudaMemcpy2DAsync(static_cast<char*>(destination)+size_t(first)*width+offset,width,
                    static_cast<const char*>(bank->data)+size_t(slot)*role_bytes,role_bytes,
                    role_bytes,end-first,cudaMemcpyDeviceToDevice,copy));
                metrics.cache_reused_bytes+=size_t(end-first)*role_bytes;++metrics.cache_copy_calls;offset+=role_bytes;
            }
            first=end;
        } else {
            const int expert=group.first_expert+first;
            const bool ready=prefetched&&prefetched->gate==sources.gate&&expert<prefetched->count&&
                (prefetched->missing&(uint64_t(1)<<expert));
            int end=first+1;
            while(end<group.experts&&(!slots||slots[group.first_expert+end]<0)){
                const int next=group.first_expert+end;
                const bool next_ready=prefetched&&prefetched->gate==sources.gate&&next<prefetched->count&&
                    (prefetched->missing&(uint64_t(1)<<next));
                if(next_ready!=ready)break;++end;
            }
            const size_t bytes=size_t(end-first)*width;
            const auto* input=ready?static_cast<const char*>(prefetched->data)+size_t(expert)*width:
                static_cast<const char*>(packed.data)+size_t(first)*width;
            CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(destination)+size_t(first)*width,
                input,bytes,ready?cudaMemcpyDeviceToDevice:cudaMemcpyHostToDevice,copy));
            if(ready)metrics.prefetch_consumed_bytes+=bytes;
            else{metrics.payload_bytes+=bytes;++metrics.dma_calls;}first=end;
        }
    }
}
}
