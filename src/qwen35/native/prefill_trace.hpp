#pragma once
#include "strata/qwen35/native/ffn.hpp"
#include <array>
#include <chrono>
#include <cstdlib>
#include <ostream>
#include <stdexcept>

namespace strata::qwen35::native {
// Optional request metrics: one fixed slot per 8192-token chunk, never per
// layer/kernel. The service owns events; no model/KV/Graph ABI is changed.
class PrefillTrace {
    using Clock = std::chrono::steady_clock;
    struct Chunk {
        std::array<cudaEvent_t,3> events{};
        std::array<Clock::time_point,3> submitted{};
        int start=0,tokens=0;
        float main_ms=0,mtp_ms=0,total_ms=0;
    };
    std::array<Chunk,32> chunks_{};
    int count_=0;
    bool enabled_=false;
public:
    PrefillTrace()=default;
    PrefillTrace(const PrefillTrace&)=delete;
    PrefillTrace& operator=(const PrefillTrace&)=delete;
    ~PrefillTrace(){for(auto &c:chunks_)for(auto e:c.events)if(e)cudaEventDestroy(e);}
    void open(){
        const char *flag=std::getenv("STRATA_DENSE_PP_CHUNK_TIMING");
        if(!flag||flag[0]!='1')return;
        for(auto &c:chunks_)for(auto &e:c.events)
            check_cuda(cudaEventCreate(&e),"prefill chunk timing pool");
        enabled_=true;
    }
    bool enabled()const{return enabled_;}
    void reset(){count_=0;}
    int begin(int start,int tokens,cudaStream_t stream){
        if(!enabled_||count_>=int(chunks_.size())||start<0||tokens<1||tokens>8192)
            throw std::logic_error("prefill timing chunk extent");
        const int slot=count_++;auto &c=chunks_[slot];c.start=start;c.tokens=tokens;
        c.submitted[0]=Clock::now();
        check_cuda(cudaEventRecord(c.events[0],stream),"prefill chunk begin");return slot;
    }
    void mark(int slot,int part,cudaStream_t stream){
        if(slot<0||slot>=count_||part<1||part>2)throw std::logic_error("prefill timing marker");
        auto &c=chunks_[slot];
        check_cuda(cudaEventRecord(c.events[part],stream),"prefill chunk boundary");
        c.submitted[part]=Clock::now();
    }
    void resolve(){
        for(int i=0;i<count_;++i){auto &c=chunks_[i];
            // pick() has already joined this stream. Query, never synchronize:
            // events measure the observed stream interval, including stalls.
            check_cuda(cudaEventQuery(c.events[2]),"prefill timing existing completion");
            check_cuda(cudaEventElapsedTime(&c.main_ms,c.events[0],c.events[1]),"prefill main elapsed");
            check_cuda(cudaEventElapsedTime(&c.mtp_ms,c.events[1],c.events[2]),"prefill MTP elapsed");
            check_cuda(cudaEventElapsedTime(&c.total_ms,c.events[0],c.events[2]),"prefill chunk elapsed");
        }
    }
    void write(std::ostream &out)const{
        if(!enabled_)return;
        out<<" pp_chunk_count="<<count_<<" pp_chunks=";
        if(!count_)out<<'-';
        for(int i=0;i<count_;++i){const auto &c=chunks_[i];
            if(i)out<<';';
            const double host_main=std::chrono::duration<double,std::milli>(c.submitted[1]-c.submitted[0]).count();
            const double host_mtp=std::chrono::duration<double,std::milli>(c.submitted[2]-c.submitted[1]).count();
            out<<c.start<<':'<<c.tokens<<':'<<c.main_ms<<':'<<c.mtp_ms<<':'<<c.total_ms
               <<':'<<host_main<<':'<<host_mtp;
        }
    }
};
inline thread_local PrefillTrace *active_prefill_trace=nullptr;
class PrefillTraceScope {
    PrefillTrace *previous_;
public:
    explicit PrefillTraceScope(PrefillTrace &trace):previous_(active_prefill_trace){
        trace.reset();active_prefill_trace=trace.enabled()?&trace:nullptr;
    }
    ~PrefillTraceScope(){active_prefill_trace=previous_;}
};
} // namespace strata::qwen35::native
