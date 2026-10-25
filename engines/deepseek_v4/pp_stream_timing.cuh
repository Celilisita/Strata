#pragma once
#include "pp_stream_metrics.hpp"
#include "ggml-cuda/common.cuh"
#include <array>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strata::deepseek_v4 {
// Opt-in bounded timing at the PP seam. Collection happens after the existing
// next-layer synchronization or the benchmark's phase synchronization.
class PpStreamTiming {
    struct Group { cudaEvent_t waiting=nullptr,ready=nullptr,done=nullptr; };
    std::array<Group,288> groups_{};
    cudaEvent_t start_=nullptr,end_=nullptr;
    size_t count_=0;
    bool pending_=false;
    static void create(cudaEvent_t& event) { if(!event)CUDA_CHECK(cudaEventCreate(&event)); }
    static uint64_t elapsed(cudaEvent_t a,cudaEvent_t b) {
        float ms=0;CUDA_CHECK(cudaEventElapsedTime(&ms,a,b));return uint64_t(double(ms)*1000000);
    }
public:
    bool enabled=false;
    PpStreamTiming() {
        const char* value=std::getenv("STRATA_DSV4_PP_TIMING");
        enabled=value&&std::strcmp(value,"1")==0;
    }
    ~PpStreamTiming() {
        for(auto& group:groups_)for(auto event:{group.waiting,group.ready,group.done})if(event)cudaEventDestroy(event);
        if(start_)cudaEventDestroy(start_);if(end_)cudaEventDestroy(end_);
    }
    void collect(PpStreamMetrics& metrics) {
        if(!enabled||!pending_)return;
        const auto status=cudaEventQuery(end_);
        if(status==cudaErrorNotReady)return;
        CUDA_CHECK(status);
        metrics.gpu_layer_ns+=elapsed(start_,end_);
        if(count_) {
            metrics.gpu_prelude_ns+=elapsed(start_,groups_[0].waiting);
            for(size_t i=1;i<count_;++i)
                metrics.gpu_intergroup_ns+=elapsed(groups_[i-1].done,groups_[i].waiting);
            metrics.gpu_tail_ns+=elapsed(groups_[count_-1].done,end_);
        }
        for(size_t i=0;i<count_;++i) {
            metrics.gpu_weight_wait_ns+=elapsed(groups_[i].waiting,groups_[i].ready);
            metrics.gpu_products_ns+=elapsed(groups_[i].ready,groups_[i].done);
        }
        ++metrics.timed_layers;pending_=false;
    }
    void begin(size_t count,cudaStream_t stream,PpStreamMetrics& metrics) {
        if(!enabled)return;
        collect(metrics);
        if(pending_||count>groups_.size())throw std::runtime_error("PP timing lifetime or group capacity exceeded");
        create(start_);create(end_);count_=count;
        for(size_t i=0;i<count;++i) {create(groups_[i].waiting);create(groups_[i].ready);create(groups_[i].done);}
        CUDA_CHECK(cudaEventRecord(start_,stream));
    }
    void waiting(size_t i,cudaStream_t stream) {if(enabled)CUDA_CHECK(cudaEventRecord(groups_[i].waiting,stream));}
    void ready(size_t i,cudaStream_t stream) {if(enabled)CUDA_CHECK(cudaEventRecord(groups_[i].ready,stream));}
    void done(size_t i,cudaStream_t stream) {if(enabled)CUDA_CHECK(cudaEventRecord(groups_[i].done,stream));}
    void end(cudaStream_t stream) {
        if(!enabled)return;
        CUDA_CHECK(cudaEventRecord(end_,stream));pending_=true;
    }
};
}
