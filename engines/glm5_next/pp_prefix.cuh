#pragma once
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include "pp_prefix_graph.hpp"
#include "pp_prefix_segment_ready.cuh"
#include "../deepseek_v4/pp_stream_graph.hpp"
#include "../deepseek_v4/pp_stream_plan.hpp"
#include "../deepseek_v4/pp_stream_edges.cuh"
#include <algorithm>
#include <array>
#include <vector>
#include <chrono>
namespace strata::glm5_next {
// One ephemeral original-byte prefix. Only the existing PP plan chooses expert
// work; the cache projection remains the sole residency authority.
template<class Memory>class PpPrefix {
    deepseek_v4::PpStreamEdges edges_;
    PpPrefixSegmentReady segment_ready_;
    cudaEvent_t ready_=nullptr,done_=nullptr,copy_start_=nullptr;
    bool ready_recorded_=false,done_recorded_=false,pending_=false,planar_=false,copy_timing_pending_=false;
    std::array<const ggml_tensor*,3> source_{};
    const ggml_tensor* directory_=nullptr;
    std::array<uint64_t,3> missing_{};
    size_t triplet_=0,stride_=0;int count_=0;
    bool contains(int id)const{return id>=0&&id<count_&&(missing_[id/64]&(1ull<<(id%64)));}
public:
    Memory device;
    PpPrefix(){CUDA_CHECK(cudaEventCreate(&ready_));CUDA_CHECK(cudaEventCreateWithFlags(&done_,cudaEventDisableTiming));CUDA_CHECK(cudaEventCreate(&copy_start_));}
    ~PpPrefix(){if(ready_)cudaEventDestroy(ready_);if(done_)cudaEventDestroy(done_);if(copy_start_)cudaEventDestroy(copy_start_);}
    void clear(){segment_ready_.reset();device.clear();pending_=false;source_={};directory_=nullptr;count_=0;missing_={};ready_recorded_=done_recorded_=copy_timing_pending_=false;}
    template<class Query,class IsPinned>
    void load(const deepseek_v4::PpCacheView* view,Query query,void* owner,cudaStream_t copy,
        deepseek_v4::PpStreamMetrics& metrics,IsPinned pinned) {
        const auto enqueue_start=std::chrono::steady_clock::now();
        const bool segment_requested=pp_prefix_segment_ready_requested();segment_ready_.reset();
        pending_=false;source_={};directory_=nullptr;count_=0;missing_={};planar_=pp_prefix_planar();
        const char* direct=std::getenv("STRATA_DSV4_PP_DIRECT_CACHE_GROUPS");
        if(!pp_prefix_requested()||(direct&&!std::strcmp(direct,"0"))||!view||!view->directory||!view->directory->data)return;
        const auto& source=view->source;
        if(!source[0]||!source[1]||!source[2]||source[0]->ne[2]!=288||
            source[0]->type!=source[1]->type||source[0]->nb[2]!=source[1]->nb[2])return;
        triplet_=source[0]->nb[2]+source[1]->nb[2]+source[2]->nb[2];stride_=planar_?triplet_:pp_prefix_payload_stride(source);
        constexpr size_t cap=1792ull<<20,remaining_scratch=2048ull<<20;
        if(!stride_)return;
        const int cap_count=(std::min(160,int(cap/stride_))/8)*8;
        const size_t cap_need=size_t(cap_count)*stride_;
        int count=cap_count;
        // Existing allocation already holds the largest legal aligned prefix.
        // Free memory cannot change count or make reserve grow in this case.
        if(device.bytes<cap&&(cap_count<32||device.bytes<cap_need)) {
            size_t free=0,total=0;CUDA_CHECK(cudaMemGetInfo(&free,&total));
            const size_t budget=std::min(cap,device.bytes+(free>remaining_scratch?free-remaining_scratch:0));
            count=(std::min(160,int(budget/stride_))/8)*8;
        }
        if(count<32)return;
        auto segment=[&](const char* data,size_t bytes) {
            auto span=query?query(data,bytes,owner):deepseek_v4::ExpertSourcePin::Segment{};
            if(!span.bytes)span={bytes,pinned(data,bytes)};
            if(span.bytes>bytes)throw std::runtime_error("GLM PP prefix source projection exceeds range");
            return span;
        };
        // Tiny unregistered alignment edges are mirrored separately. A bulk
        // unregistered span falls back to the normal bounded post-route path.
        for(const auto* tensor:source) {
            size_t done=0,bytes=size_t(count)*tensor->nb[2];
            while(done<bytes) {
                const auto span=segment(static_cast<const char*>(tensor->data)+done,bytes-done);
                if(!span.registered&&span.bytes>65536)return;
                done+=span.bytes;
            }
        }
        // Host mirrors must not be overwritten while their last DMA reads are
        // pending. GPU prefix storage separately waits for its final reader.
        if(ready_recorded_){CUDA_CHECK(cudaEventSynchronize(ready_));collect(metrics);}
        if(done_recorded_)CUDA_CHECK(cudaStreamWaitEvent(copy,done_,0));
        device.reserve(size_t(count)*stride_);
        std::array<ggml_tensor,3> bounded{*source[0],*source[1],*source[2]};
        for(auto& tensor:bounded)tensor.ne[2]=count;
        edges_.prepare({&bounded[0],&bounded[1],&bounded[2]},query,owner,metrics);
        const auto* slots=static_cast<const int32_t*>(view->directory->data);
        const size_t offsets[]{0,source[0]->nb[2],source[0]->nb[2]+source[1]->nb[2]};
        const bool segmented=segment_ready_.prepare(segment_requested&&planar_&&pp_prefix_true_fences(),count);
        CUDA_CHECK(cudaEventRecord(copy_start_,copy));
        auto upload=[&](int begin,int limit) {
        for(int first=begin;first<limit;) {
            if(slots[first]>=0){++first;continue;}
            int end=first+1;while(end<limit&&slots[end]<0)++end;
            for(int role=0;role<3;++role) {
                const size_t width=source[role]->nb[2];const char* base=static_cast<const char*>(source[role]->data);
                size_t consumed=size_t(first)*width,limit=size_t(end)*width;
                while(consumed<limit) {
                    const auto span=segment(base+consumed,limit-consumed);
                    const size_t row=consumed/width,column=consumed%width;
                    size_t bytes=std::min(span.bytes,width-column),rows=1;
                    if(planar_)bytes=span.bytes;
                    else if(!column&&span.bytes>=width){rows=span.bytes/width;bytes=rows*width;}
                    const void* input=base+consumed;
                    if(!span.registered){input=edges_.lookup(base+consumed,bytes);
                        if(!input)throw std::runtime_error("GLM PP prefix edge mirror unavailable");metrics.staged_bytes+=bytes;}
                    char* target=static_cast<char*>(device.p)+(planar_?size_t(count)*offsets[role]+consumed:row*stride_+offsets[role]+column);
                    if(rows==1||planar_)CUDA_CHECK(cudaMemcpyAsync(target,input,bytes,cudaMemcpyHostToDevice,copy));
                    else CUDA_CHECK(cudaMemcpy2DAsync(target,stride_,input,width,width,rows,cudaMemcpyHostToDevice,copy));
                    metrics.payload_bytes+=bytes;metrics.prefetch_bytes+=bytes;++metrics.dma_calls;consumed+=bytes;
                }
            }
            for(int id=first;id<end;++id)missing_[id/64]|=1ull<<(id%64);
            first=end;
        }
        };
        if(segmented)for(int first=0;first<count;first+=32) {
            const int end=std::min(count,first+32);upload(first,end);segment_ready_.record(end,copy,metrics);
        } else upload(0,count);
        CUDA_CHECK(cudaEventRecord(ready_,copy));ready_recorded_=true;
        count_=count;source_=source;directory_=view->directory;pending_=true;copy_timing_pending_=true;++metrics.prefetch_launches;
        if(segmented)++metrics.prefix_segment_ready_banks;
        metrics.prefetch_enqueue_ns+=uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-enqueue_start).count());
        const char* verify=std::getenv("STRATA_GLM53_PP_PREFIX_VERIFY");
        if(verify&&!std::strcmp(verify,"1")) {
            CUDA_CHECK(cudaEventSynchronize(ready_));std::vector<char> payload(triplet_);
            for(int id=0;id<count_;++id)if(contains(id)) {
                if(planar_)for(int role=0;role<3;++role)CUDA_CHECK(cudaMemcpy(payload.data()+offsets[role],role_payload(role,id),source[role]->nb[2],cudaMemcpyDeviceToHost));
                else CUDA_CHECK(cudaMemcpy(payload.data(),static_cast<const char*>(device.p)+size_t(id)*stride_,triplet_,cudaMemcpyDeviceToHost));
                for(int role=0;role<3;++role)if(std::memcmp(payload.data()+offsets[role],
                    static_cast<const char*>(source[role]->data)+size_t(id)*source[role]->nb[2],source[role]->nb[2]))
                    throw std::runtime_error("GLM PP prefix original payload bytes differ");
                metrics.prefetch_verified_bytes+=triplet_;
            }
        }
    }
    void collect(deepseek_v4::PpStreamMetrics& metrics) {
        if(!copy_timing_pending_)return;const auto status=cudaEventQuery(ready_);if(status==cudaErrorNotReady)return;CUDA_CHECK(status);
        float ms=0;CUDA_CHECK(cudaEventElapsedTime(&ms,copy_start_,ready_));metrics.prefetch_gpu_copy_ns+=uint64_t(double(ms)*1e6);copy_timing_pending_=false;
    }
    bool planar()const{return planar_;}
    void receipt_missing(const deepseek_v4::PpStreamSources& s,uint64_t* out)const {
        std::fill_n(out,3,uint64_t(0));if(boundary(s))std::copy(missing_.begin(),missing_.end(),out);
    }
    cudaEvent_t ready_event()const{return ready_;}
    const void* role_payload(int role,int first)const {
        size_t offset=0;for(int i=0;i<role;++i)offset+=size_t(count_)*source_[i]->nb[2];
        return static_cast<const char*>(device.p)+offset+size_t(first)*source_[role]->nb[2];
    }
    size_t payload_stride()const{return stride_;}
    int boundary(const deepseek_v4::PpStreamSources& source)const {
        return pending_&&source.gate==source_[0]&&source.up==source_[1]&&source.down==source_[2]&&
            source.cache&&source.cache->directory==directory_?count_:0;
    }
    bool covers_group(const deepseek_v4::PpStreamSources& source,const deepseek_v4::PpStreamGroup& group)const {
        if(!group.source_keys.empty())return false; // Tagged owned misses are never prefix intervals.
        if(!boundary(source)||group.cache_first_slot>=0)return false;
        for(int id=group.first_expert;id<group.first_expert+group.experts;++id)if(!contains(id))return false;
        return true;
    }
    bool segment_ready_active(const deepseek_v4::PpStreamSources& source)const {
        return planar_&&boundary(source)>0&&segment_ready_.active();
    }
    cudaEvent_t ready_for_group(const deepseek_v4::PpStreamSources& source,const deepseek_v4::PpStreamGroup& group,
                               deepseek_v4::PpStreamMetrics& metrics)const {
        if(!planar_||!covers_group(source,group))return ready_;
        return segment_ready_.for_group(group.first_expert,group.experts,count_,ready_,metrics);
    }
    const void* borrow(const deepseek_v4::PpStreamSources& source,const deepseek_v4::PpStreamGroup& group,
        deepseek_v4::PpStreamMetrics& metrics) {
        if(!covers_group(source,group))return nullptr;
        metrics.prefetch_consumed_bytes+=size_t(group.experts)*triplet_;
        return planar_?role_payload(0,group.first_expert):static_cast<const char*>(device.p)+size_t(group.first_expert)*stride_;
    }
    void consumed(cudaStream_t compute){CUDA_CHECK(cudaEventRecord(done_,compute));done_recorded_=true;}
    void end_layer(){pending_=false;}
};
}
