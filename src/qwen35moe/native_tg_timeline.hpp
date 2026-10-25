#pragma once
#include "strata/qwen35moe/metrics.hpp"
#include "strata/qwen35moe/tg_wait_probe.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

namespace strata::qwen35moe::native_tg_timeline {
inline void need(bool value,const char* text) { if(!value)throw std::runtime_error(text); }
inline void ck(cudaError_t code) { if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code)); }
inline bool enabled() {
    static const bool value=[] {
        const char* raw=std::getenv("STRATA_NATIVE_TG_TIMELINE");
        if(!raw||!std::strcmp(raw,"0"))return false;
        if(!std::strcmp(raw,"1"))return true;
        throw std::invalid_argument("STRATA_NATIVE_TG_TIMELINE must be0 or1");
    }();
    return value;
}
class Probe {
    static constexpr size_t layers=40,reference=0,head_begin_index=121,head_ready_index=122;
    static constexpr uint64_t all_layers=(uint64_t(1)<<layers)-1;
    std::array<cudaEvent_t,123> events_{};
    TgWaitProbeStamp* host_=nullptr;
    cudaStream_t stream_=nullptr;
    std::array<bool,layers> recurrent_{};
    std::array<char,32768> text_{};
    uint32_t epoch_=0,token_=0,reference_epoch_=0,reference_token_=0;
    uint64_t begun_=0,dense_=0,moe_=0;
    bool active_=false,reference_token_now_=false,head_begun_=false,head_ready_=false,disabled_=false;
    size_t used_=0;
    void append(const char* text) {
        const size_t bytes=std::strlen(text);need(bytes<text_.size()-used_,"TG timeline bounded output exceeded");
        std::memcpy(text_.data()+used_,text,bytes+1);used_+=bytes;
    }
    template<class... Args> void append(const char* format,Args... args) {
        const int bytes=std::snprintf(text_.data()+used_,text_.size()-used_,format,args...);
        need(bytes>=0&&size_t(bytes)<text_.size()-used_,"TG timeline bounded output exceeded");
        used_+=size_t(bytes);
    }
    void record(size_t index,cudaStream_t stream) {
        need(stream==stream_,"TG timeline changed compute stream");ck(cudaEventRecord(events_[index],stream));
    }
    float offset(cudaEvent_t event,bool allow_negative=false) const {
        float value=0;ck(cudaEventElapsedTime(&value,events_[reference],event));
        need(std::isfinite(value)&&(allow_negative||value>=0),"TG timeline invalid GPU timestamp");return value;
    }
    void release() noexcept {
        // Single Runtime per process. Runtime destruction drains its streams before
        // this thread-local owner is destroyed; no asynchronous mapping is freed early.
        for(auto event:events_)if(event)(void)cudaEventDestroy(event);
        if(host_)(void)cudaFreeHost(host_);
    }
public:
    explicit Probe(cudaStream_t stream):stream_(stream) {
        need(stream,"TG timeline missing compute stream");
        try {
            for(auto& event:events_)ck(cudaEventCreate(&event));
            ck(cudaHostAlloc(reinterpret_cast<void**>(&host_),layers*sizeof(TgWaitProbeStamp),cudaHostAllocMapped));
            std::memset(host_,0,layers*sizeof(TgWaitProbeStamp));
            TgWaitProbeStamp* alias=nullptr;
            ck(cudaHostGetDevicePointer(reinterpret_cast<void**>(&alias),host_,0));
            configure_native_tg_wait_probe(alias,2,3,stream_);
        } catch(...) { release();throw; }
    }
    ~Probe() { release(); }
    Probe(const Probe&)=delete;
    Probe& operator=(const Probe&)=delete;
    void begin(cudaStream_t stream,uint32_t epoch,uint32_t token,size_t count) {
        need(stream==stream_&&count==layers&&!active_&&!reference_token_now_,"TG timeline Runtime/extent changed");
        epoch_=epoch;token_=token;begun_=dense_=moe_=0;head_begun_=head_ready_=false;
        if(epoch>3&&!disabled_) {configure_native_tg_wait_probe(nullptr,0,0,stream_);disabled_=true;}
        active_=!disabled_&&(epoch==2||epoch==3)&&token<=1920&&(token&255u)==128;
        reference_token_now_=!disabled_&&(epoch==2||epoch==3)&&token<=1919&&(token&255u)==127;
        if(active_)need(reference_epoch_==epoch&&reference_token_+1==token,"TG timeline missing prior-token reference");
    }
    void end() noexcept { active_=false;reference_token_now_=false; }
    void dense_begin(size_t layer,bool recurrent,cudaStream_t stream) {
        if(!active_)return;need(layer<layers&&!(begun_&(uint64_t(1)<<layer)),"TG timeline dense duplicate/extent");
        recurrent_[layer]=recurrent;record(1+3*layer,stream);begun_|=uint64_t(1)<<layer;
    }
    void dense_end(size_t layer,cudaStream_t stream) {
        if(!active_)return;need(layer<layers&&(begun_&(uint64_t(1)<<layer))&&!(dense_&(uint64_t(1)<<layer)),"TG timeline dense order");
        record(2+3*layer,stream);dense_|=uint64_t(1)<<layer;
    }
    void moe_end(size_t layer,cudaStream_t stream) {
        if(!active_)return;need(layer<layers&&(dense_&(uint64_t(1)<<layer))&&!(moe_&(uint64_t(1)<<layer)),"TG timeline MoE order");
        record(3+3*layer,stream);moe_|=uint64_t(1)<<layer;
    }
    void head_begin(cudaStream_t stream) {
        if(!active_)return;need(moe_==all_layers&&!head_begun_,"TG timeline head order");
        record(head_begin_index,stream);head_begun_=true;
    }
    void head_ready(cudaStream_t stream) {
        if(active_) {need(head_begun_&&!head_ready_,"TG timeline head-ready order");record(head_ready_index,stream);head_ready_=true;}
        if(reference_token_now_) {record(reference,stream);reference_epoch_=epoch_;reference_token_=token_;}
    }
    template<class Ready> void harvest(const Metrics& metrics,Ready ready) {
        if(!active_)return;
        need(begun_==all_layers&&dense_==all_layers&&moe_==all_layers&&head_ready_,"TG timeline incomplete sample");
        // Original head sync has completed all current compute and copy-ready
        // dependencies. No query retry, D2H, or new synchronization is used here.
        const float head_begin_ms=offset(events_[head_begin_index]),head_ready_ms=offset(events_[head_ready_index]);
        used_=0;
        append("NVFP4_TG_TIMELINE {\"schema\":\"nvfp4-tg-timeline-v1\",\"epoch\":%u,\"token_ordinal\":%u,\"reference_token\":%u,"
            "\"head_begin_ms\":%.9g,\"head_ready_ms\":%.9g,\"enqueue_ms\":%.9g,\"head_observe_ms\":%.9g,"
            "\"registered_bytes\":%llu,\"pageable_bytes\":%llu,\"copy_gap_is_upper_bound\":true,\"wait_units\":\"raw_cycles\",\"layers\":[",
            epoch_,token_,reference_token_,double(head_begin_ms),double(head_ready_ms),metrics.expert_enqueue_ms,metrics.head_observe_ms,
            static_cast<unsigned long long>(metrics.expert_h2d_registered_bytes),static_cast<unsigned long long>(metrics.expert_h2d_pageable_bytes));
        for(size_t layer=0;layer<layers;++layer) {
            const auto stamp=host_[layer];
            need(stamp.epoch==epoch_&&stamp.token==token_&&stamp.layer==layer&&!(stamp.mask&~255u),"TG timeline stale wait stamp");
            need(stamp.mask||stamp.await_cycles==0,"TG timeline hit wait must be zero");
            const float begin_ms=offset(events_[1+3*layer]),dense_ms=offset(events_[2+3*layer]),moe_ms=offset(events_[3+3*layer]);
            append("%s{\"layer\":%zu,\"recurrent\":%s,\"dense_begin_ms\":%.9g,\"dense_end_ms\":%.9g,\"moe_end_ms\":%.9g,",
                layer?",":"",layer,recurrent_[layer]?"true":"false",double(begin_ms),double(dense_ms),double(moe_ms));
            const cudaEvent_t copy=ready(layer);
            if(copy) {
                const float copy_ms=offset(copy,true);
                append("\"copy_ready_ms\":%.9g,\"potential_copy_ready_gap_ms\":%.9g,",double(copy_ms),double((std::max)(0.0f,copy_ms-dense_ms)));
            } else append("\"copy_ready_ms\":null,\"potential_copy_ready_gap_ms\":null,");
            append("\"await_cycles\":%llu,\"route_mask\":%u,\"stamp_epoch\":%u,\"stamp_token\":%u}",
                static_cast<unsigned long long>(stamp.await_cycles),stamp.mask,stamp.epoch,stamp.token);
        }
        append("]}");std::fprintf(stderr,"%s\n",text_.data());std::fflush(stderr);
    }
};
inline thread_local std::unique_ptr<Probe> owner;
struct Scope {
    Probe* probe=nullptr;
    explicit Scope(Probe* value):probe(value) {}
    Scope(const Scope&)=delete;
    ~Scope() {if(probe)probe->end();}
};
inline Scope begin(bool applicable,cudaStream_t stream,uint32_t epoch,uint32_t token,size_t count) {
    if(!applicable||!enabled())return Scope(nullptr);
    if(!owner)owner=std::make_unique<Probe>(stream);
    owner->begin(stream,epoch,token,count);return Scope(owner.get());
}
inline void dense_begin(size_t layer,bool recurrent,cudaStream_t stream) {if(owner)owner->dense_begin(layer,recurrent,stream);}
inline void dense_end(size_t layer,cudaStream_t stream) {if(owner)owner->dense_end(layer,stream);}
inline void moe_end(size_t layer,cudaStream_t stream) {if(owner)owner->moe_end(layer,stream);}
inline void head_begin(cudaStream_t stream) {if(owner)owner->head_begin(stream);}
inline void head_ready(cudaStream_t stream) {if(owner)owner->head_ready(stream);}
template<class Ready> void harvest(const Metrics& metrics,Ready ready) {if(owner)owner->harvest(metrics,ready);}
} // namespace strata::qwen35moe::native_tg_timeline
