#pragma once
#include "pp_transport.hpp"
#include "pp_stream_metrics.hpp"
#include "ggml-cuda/common.cuh"
#include <array>
#include <chrono>
#include <cstring>
#include <stdexcept>

namespace strata::deepseek_v4 {
// Tiny registration edges have a layer lifetime. They must not reuse the
// bulk slab whose completion event also follows preceding large DMA work.
class PpStreamEdges {
    static constexpr size_t capacity=6, maximum_bytes=65536;
    struct Edge { const char* source=nullptr; size_t bytes=0; };
    std::array<Edge,capacity> edges_{};
    void* memory_=nullptr;
    size_t count_=0;
public:
    ~PpStreamEdges() {if(memory_)cudaFreeHost(memory_);}
    void prepare(const std::array<const ggml_tensor*,3>& sources, PpTransportQuery query,
                 void* owner,PpStreamMetrics& metrics) {
        count_=0;
        if(!query)return;
        for(const auto* tensor:sources) {
            const auto* source=static_cast<const char*>(tensor->data);
            const size_t bytes=ggml_nbytes(tensor);
            size_t consumed=0;
            while(consumed<bytes) {
                const auto segment=query(source+consumed,bytes-consumed,owner);
                if(!segment.bytes)break; // This projection does not own the bank.
                if(segment.bytes>bytes-consumed)throw std::runtime_error("PP edge projection exceeds source bank");
                if(!segment.registered&&segment.bytes<=maximum_bytes) {
                    if(count_==capacity)throw std::runtime_error("PP edge transport exceeds bounded segment capacity");
                    if(!memory_)CUDA_CHECK(cudaHostAlloc(&memory_,capacity*maximum_bytes,cudaHostAllocDefault));
                    const auto start=std::chrono::steady_clock::now();
                    std::memcpy(static_cast<char*>(memory_)+count_*maximum_bytes,source+consumed,segment.bytes);
                    metrics.source_cpu_copy_ns+=uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now()-start).count());
                    edges_[count_++]={source+consumed,segment.bytes};
                }
                consumed+=segment.bytes;
            }
        }
    }
    const void* lookup(const char* source,size_t bytes) const {
        const auto address=reinterpret_cast<uintptr_t>(source);
        for(size_t i=0;i<count_;++i) {
            const auto first=reinterpret_cast<uintptr_t>(edges_[i].source);
            if(address>=first&&address-first<edges_[i].bytes&&bytes<=edges_[i].bytes-(address-first))
                return static_cast<const char*>(memory_)+i*maximum_bytes+(address-first);
        }
        return nullptr;
    }
};
}
