#pragma once
#include "pp_prefetch_graph.hpp"
#include "pp_stream_metrics.hpp"
#include "ggml-cuda/common.cuh"
namespace strata::deepseek_v4 {
struct PpPrefetchedSpan {const ggml_tensor* gate=nullptr;const void* data=nullptr;int count=0;uint64_t missing=0;};
template<class Memory>struct PpPackedPrefix {
    Memory device;
    PpPrefetchedSpan span;
    void clear(){device.clear();span={};}
    void load(const PpCacheView* view,PackedExpertQuery query,void* owner,cudaStream_t stream,PpStreamMetrics& metrics){
        if(!view||!query||!view->source[0]||!view->directory||!view->directory->data)return;
        const auto packed=query(view->source[0],0,32,owner);
        device.reserve(packed.bytes);span={view->source[0],device.p,32,0};
        const auto* slots=static_cast<const int32_t*>(view->directory->data);
        for(int first=0;first<32;){
            if(slots[first]>=0){++first;continue;}
            int end=first+1;while(end<32&&slots[end]<0)++end;
            const size_t offset=size_t(first)*packed.expert_stride,bytes=size_t(end-first)*packed.expert_stride;
            CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(device.p)+offset,static_cast<const char*>(packed.data)+offset,
                bytes,cudaMemcpyHostToDevice,stream));
            for(int expert=first;expert<end;++expert)span.missing|=uint64_t(1)<<expert;
            metrics.payload_bytes+=bytes;++metrics.dma_calls;metrics.prefetch_bytes+=bytes;first=end;
        }
        ++metrics.prefetch_launches;
    }
};
}
