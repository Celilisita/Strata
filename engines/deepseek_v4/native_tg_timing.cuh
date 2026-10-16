#pragma once
#include "native_tg_timing.hpp"
#include "ggml-cuda/common.cuh"
#include <array>
#include <cstdlib>
#include <cstring>
namespace strata::deepseek_v4 {
// Physical-cost diagnostic only. External event nodes retain timestamps when
// the stable all-hit CUDA graph replays. Disabled in normal execution.
struct NativeTgTiming {
    std::array<cudaEvent_t,43> start{},end{},join_start{},join_end{};
    std::array<bool,43> recorded{},join_recorded{};
    cudaEvent_t whole_start=nullptr,whole_end=nullptr;
    bool enabled=false;
    NativeTgTiming(){const char* flag=std::getenv("STRATA_DSV4_TG_TIMING");enabled=flag&&std::strcmp(flag,"1")==0;
        if(enabled){for(size_t i=0;i<43;++i){CUDA_CHECK(cudaEventCreate(&start[i]));CUDA_CHECK(cudaEventCreate(&end[i]));CUDA_CHECK(cudaEventCreate(&join_start[i]));CUDA_CHECK(cudaEventCreate(&join_end[i]));}
            CUDA_CHECK(cudaEventCreate(&whole_start));CUDA_CHECK(cudaEventCreate(&whole_end));}}
    ~NativeTgTiming(){for(size_t i=0;i<43;++i){if(start[i])cudaEventDestroy(start[i]);if(end[i])cudaEventDestroy(end[i]);if(join_start[i])cudaEventDestroy(join_start[i]);if(join_end[i])cudaEventDestroy(join_end[i]);}
        if(whole_start)cudaEventDestroy(whole_start);if(whole_end)cudaEventDestroy(whole_end);}
    static void record(cudaEvent_t event,cudaStream_t stream){cudaStreamCaptureStatus status;
        CUDA_CHECK(cudaStreamIsCapturing(stream,&status));
        CUDA_CHECK(cudaEventRecordWithFlags(event,stream,status==cudaStreamCaptureStatusActive?cudaEventRecordExternal:cudaEventRecordDefault));}
    void before(int layer,cudaStream_t stream){if(enabled){GGML_ASSERT(layer>=0&&layer<43);record(start[layer],stream);}}
    void after(int layer,cudaStream_t stream){if(enabled){record(end[layer],stream);recorded[layer]=true;}}
    void before_join(int layer,cudaStream_t stream){if(enabled)record(join_start[layer],stream);}
    void after_join(int layer,cudaStream_t stream){if(enabled){record(join_end[layer],stream);join_recorded[layer]=true;}}
    static uint64_t elapsed(cudaEvent_t a,cudaEvent_t b){float ms=0;CUDA_CHECK(cudaEventElapsedTime(&ms,a,b));return uint64_t(double(ms)*1000000);}
    bool begin(cudaStream_t stream){if(!enabled)return false;CUDA_CHECK(cudaEventRecord(whole_start,stream));return true;}
    bool read(cudaStream_t stream,NativeTgTimingResult* result){if(!enabled||!result)return false;
        CUDA_CHECK(cudaEventRecord(whole_end,stream));CUDA_CHECK(cudaEventSynchronize(whole_end));
        *result={};result->whole_gpu_ns=elapsed(whole_start,whole_end);
        for(size_t i=0;i<43;++i)if(recorded[i]){++result->layers;result->expert_compute_ns+=elapsed(start[i],end[i]);
            if(join_recorded[i])result->join_ns+=elapsed(join_start[i],join_end[i]);}
        result->expert_ns=result->expert_compute_ns+result->join_ns;
        return result->layers==43&&result->whole_gpu_ns>=result->expert_ns;}
};
}
