#pragma once
#include "../deepseek_v4/pp_stream_metrics.hpp"
#include "ggml-cuda/common.cuh"
#include <array>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
namespace strata::glm5_next {
inline bool pp_prefix_segment_ready_requested() {
    const char* value=std::getenv("STRATA_GLM53_PP_PREFIX_SEGMENT_READY");
    if(!value||!std::strcmp(value,"0"))return false;
    if(!std::strcmp(value,"1"))return true;
    throw std::runtime_error("PP prefix segment ready must be0 or1");
}
// Same PpPrefix owner and copy stream. These events authorize reads only;
// the original whole-bank ready/done still protect all allocation and mirrors.
class PpPrefixSegmentReady {
    std::array<cudaEvent_t,4> events_{};
    std::array<bool,4> recorded_{};
    int count_=0;
public:
    PpPrefixSegmentReady()=default;
    PpPrefixSegmentReady(const PpPrefixSegmentReady&)=delete;
    PpPrefixSegmentReady& operator=(const PpPrefixSegmentReady&)=delete;
    ~PpPrefixSegmentReady(){for(auto event:events_)if(event)cudaEventDestroy(event);}
    void reset(){count_=0;recorded_={};}
    bool active()const{return count_>32;}
    bool prepare(bool enabled,int count) {
        reset();
        if(!enabled||count<=32||count>160)return false;
        const int intermediate=(count-1)/32;
        for(int i=0;i<intermediate;++i)
            if(!events_[i])CUDA_CHECK(cudaEventCreateWithFlags(&events_[i],cudaEventDisableTiming));
        count_=count;return true;
    }
    void record(int end,cudaStream_t copy,deepseek_v4::PpStreamMetrics& metrics) {
        if(end==count_)return; // Last segment is covered by original whole ready_.
        if(end<=0||end>count_||end%32||end/32>4)throw std::runtime_error("PP prefix segment endpoint differs");
        const int index=end/32-1;
        if(!events_[index]||recorded_[index])throw std::runtime_error("PP prefix segment event owner differs");
        CUDA_CHECK(cudaEventRecord(events_[index],copy));recorded_[index]=true;
        ++metrics.prefix_segment_ready_events;
    }
    cudaEvent_t for_group(int first,int experts,int count,cudaEvent_t whole,
                         deepseek_v4::PpStreamMetrics& metrics) const {
        if(count_!=count||count_<=32||first<0||experts<1||experts>count_||first>count_-experts)return whole;
        const int index=(first+experts-1)/32;
        if(index>=4||!recorded_[index])return whole;
        ++metrics.prefix_segment_ready_waits;return events_[index];
    }
};
}
