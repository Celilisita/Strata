#pragma once
#include "strata/qwen35moe/expert_prefetch.hpp"
#include "strata/qwen35moe/pp_consumed_victims.hpp"
#include "strata/prefill/native_grouped_mmq.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <limits>
#include <stdexcept>
namespace strata::qwen35moe {
// One actual-route layer. Caller drains prior compute readers before prepare;
// Runtime must drain compute then this owner before cache/lease reset or release.
class ExpertGuCopyPipeline {
public:
    struct Cost : HostCopyCost { uint64_t cold=0,warm=0,waves=0,waits=0;double enqueue_ms=0,copy_ms=0; };
    ExpertGuCopyPipeline(){
        try{ck(cudaStreamCreateWithFlags(&copy_,cudaStreamNonBlocking));
            for(auto& event:ready_)ck(cudaEventCreateWithFlags(&event,cudaEventDisableTiming));
            for(auto& event:timing_)ck(cudaEventCreate(&event));
        }catch(...){destroy();throw;}
    }
    ~ExpertGuCopyPipeline(){if(copy_)cudaStreamSynchronize(copy_);if(active_)invalidate();destroy();}
    ExpertGuCopyPipeline(const ExpertGuCopyPipeline&)=delete;
    ExpertGuCopyPipeline& operator=(const ExpertGuCopyPipeline&)=delete;
    bool prepare(PpPrefetchCache c,size_t layer,const PpPrefetchSource& source,
                 const HostReadOnlyLease& lease,const std::vector<int32_t>& ids){
        const bool consumed_first=pp_consumed_victims::enabled();
        need(!failed_&&!active_,"GU copy requires closed request");
        need(c.data&&c.keys&&c.slots&&c.ages&&c.slots->size()==c.ages->size()&&
             c.resident_prefix<=c.slots->size()&&c.stride&&c.slots->size()<=(std::numeric_limits<size_t>::max)()/c.stride&&
             c.data->bytes()==c.slots->size()*c.stride&&layer<c.keys->size()/256,"GU copy authority geometry");
        need(!ids.empty()&&ids.size()<=32768&&c.keys->size()%256==0,"GU copy route extent");
        need(source.bytes[0]==589824&&source.bytes[1]==589824&&
             (source.bytes[2]==589824||source.bytes[2]==720896)&&
             source.bytes[0]*2+source.bytes[2]<=c.stride,"GU copy actual payload domain");
        for(size_t p=0;p<3;++p)need(source.base[p]&&source.bytes[p]*256<=
            (std::numeric_limits<uintptr_t>::max)()-uintptr_t(source.base[p]),"GU copy source extent");
        std::array<Entry,256> prepared{};size_t misses=0;
        for(int id:ids){need(id>=0&&id<256,"GU copy route ID");auto& e=prepared[size_t(id)];
            if(e.key>=0)continue;e.key=int(layer*256+size_t(id));e.slot=(*c.keys)[size_t(e.key)];
            need(e.slot<0||(size_t(e.slot)<c.slots->size()&&(*c.slots)[size_t(e.slot)]==e.key),"GU copy warm inverse");
            e.cold=e.slot<0;if(e.cold)++misses;}
        if(c.slots->size()<c.resident_prefix+512)return false;
        std::vector<size_t> victims;
        for(size_t slot=c.resident_prefix;slot<c.slots->size();++slot){const int key=(*c.slots)[slot];
            need(key>=-1&&(key<0||(size_t(key)<c.keys->size()&&(*c.keys)[size_t(key)]==int(slot))),"GU copy victim inverse");
            if(key>=0&&(size_t(key)/256==layer||(layer&&size_t(key)/256==layer-1)))continue;
            victims.push_back(slot);}
        if(victims.size()<misses)return false;
        std::sort(victims.begin(),victims.end(),pp_consumed_victims::Order{
            *c.slots,*c.ages,layer,consumed_first});
        cache_=c;source_=source;lease_=&lease;source_epoch_=lease.epoch();entries_=prepared;
        active_=true;final_=false;cost_={};size_t next=0;
        for(auto& e:entries_)if(e.key>=0){if(e.cold){e.slot=int(victims[next++]);const int old=(*c.slots)[size_t(e.slot)];
                if(old>=0)(*c.keys)[size_t(old)]=-1;(*c.keys)[size_t(e.key)]=e.slot;
                (*c.slots)[size_t(e.slot)]=e.key;(*c.ages)[size_t(e.slot)]=0;++cost_.cold;
            }else ++cost_.warm;}
        return true;
    }
    // Address-only view: pending weights are NEVER consumed before ready().
    uint8_t* address(int id)const{need(active_&&!failed_&&id>=0&&id<256&&entries_[size_t(id)].key>=0,"GU copy address");
        return cache_.data->slot(size_t(entries_[size_t(id)].slot));}
    void ready(PpPrefetchCache current,const prefill::mmq::NativeMmqMatrix* matrices,int count,bool last,cudaStream_t compute){
        validate(current);need(matrices&&count>=1&&count<=64&&compute&&cost_.waves<8&&!final_,"GU copy wave extent");
        const auto start=std::chrono::steady_clock::now();const auto before=cost_.bytes;
        try{
            if(cost_.waves==0&&cost_.cold)ck(cudaEventRecord(timing_[0],copy_));
            for(int at=0;at<count;++at){Entry* entry=nullptr;
                for(auto& e:entries_)if(e.key>=0&&uintptr_t(cache_.data->slot(size_t(e.slot)))==matrices[at].gate){entry=&e;break;}
                need(entry&&!entry->seen&&matrices[at].up==matrices[at].gate+source_.bytes[0],"GU copy matrix identity/duplicate");
                entry->seen=true;if(entry->cold)transfer(*entry,copy_);
            }
            if(last){for(const auto& e:entries_)need(e.key<0||e.seen,"GU copy final coverage");
                if(cost_.cold)ck(cudaEventRecord(timing_[1],copy_));final_=true;}
            if(cost_.bytes!=before){auto event=ready_[size_t(cost_.waves)];ck(cudaEventRecord(event,copy_));
                ck(cudaStreamWaitEvent(compute,event,0));++cost_.waits;}
            else if(last&&cost_.cold){ck(cudaStreamWaitEvent(compute,timing_[1],0));++cost_.waits;}
            ++cost_.waves;cost_.enqueue_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        }catch(...){failed_=true;invalidate();throw;}
    }
    // Matched original control: all exact triplets on COMPUTE, then ordinary GU.
    void copy_all(PpPrefetchCache current,cudaStream_t compute){
        validate(current);need(compute&&!final_&&cost_.waves==0,"GU copy control state");
        const auto start=std::chrono::steady_clock::now();
        try{ck(cudaEventRecord(timing_[0],compute));for(auto& e:entries_)if(e.key>=0){e.seen=true;if(e.cold)transfer(e,compute);}
            ck(cudaEventRecord(timing_[1],compute));cost_.waves=1;final_=true;
            cost_.enqueue_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        }catch(...){failed_=true;invalidate();throw;}
    }
    Cost finish(PpPrefetchCache current){validate(current);need(final_,"GU copy incomplete product sequence");
        if(cost_.cold){ck(cudaEventQuery(timing_[1]));float ms=0;ck(cudaEventElapsedTime(&ms,timing_[0],timing_[1]));cost_.copy_ms=ms;}
        active_=false;return cost_;} // Caller has its existing compute completion.
    void drain(){if(copy_)ck(cudaStreamSynchronize(copy_));if(active_)invalidate();active_=false;
        need(!failed_,"failed GU copy requires Runtime recreation");}
    const Cost& cost()const{return cost_;}
private:
    struct Entry{int key=-1,slot=-1;bool cold=false,seen=false;};
    static void need(bool value,const char* text){if(!value)throw std::runtime_error(text);}
    static void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
    void validate(PpPrefetchCache current)const{
        need(current.data==cache_.data&&current.keys==cache_.keys&&current.slots==cache_.slots&&current.ages==cache_.ages&&
            current.epoch==cache_.epoch&&current.stride==cache_.stride&&current.resident_prefix==cache_.resident_prefix,"GU copy stale cache epoch");
        need(active_&&!failed_&&lease_&&lease_->epoch()==source_epoch_,"GU copy stale request/source");
        need(cache_.data&&cache_.keys&&cache_.slots&&cache_.stride&&cache_.data->bytes()==cache_.slots->size()*cache_.stride,"GU copy changed physical authority");
        for(const auto& e:entries_)if(e.key>=0)need(e.slot>=0&&size_t(e.slot)<cache_.slots->size()&&
            (*cache_.keys)[size_t(e.key)]==e.slot&&(*cache_.slots)[size_t(e.slot)]==e.key,"GU copy changed pending directory");}
    void transfer(const Entry& e,cudaStream_t stream){size_t offset=0;
        for(size_t p=0;p<3;++p){static_cast<HostCopyCost&>(cost_)+=lease_->copy_to_device(
            cache_.data->slot(size_t(e.slot))+offset,source_.base[p]+size_t(e.key%256)*source_.bytes[p],
            source_.bytes[p],stream,"GU pipeline actual expert H2D",e.key,int(p));offset+=source_.bytes[p];}}
    void invalidate()noexcept{for(const auto& e:entries_)if(e.key>=0&&e.cold&&e.slot>=0&&cache_.keys&&cache_.slots&&cache_.ages&&
        size_t(e.key)<cache_.keys->size()&&size_t(e.slot)<cache_.slots->size()&&size_t(e.slot)<cache_.ages->size()){
        if((*cache_.keys)[size_t(e.key)]==e.slot)(*cache_.keys)[size_t(e.key)]=-1;
        if((*cache_.slots)[size_t(e.slot)]==e.key){(*cache_.slots)[size_t(e.slot)]=-1;(*cache_.ages)[size_t(e.slot)]=0;}}}
    void destroy()noexcept{for(auto event:ready_)if(event)cudaEventDestroy(event);
        for(auto event:timing_)if(event)cudaEventDestroy(event);if(copy_)cudaStreamDestroy(copy_);}
    PpPrefetchCache cache_{};PpPrefetchSource source_{};const HostReadOnlyLease* lease_=nullptr;
    uint64_t source_epoch_=0;std::array<Entry,256> entries_{};Cost cost_{};
    bool active_=false,failed_=false,final_=false;
    cudaStream_t copy_=nullptr;std::array<cudaEvent_t,8> ready_{};std::array<cudaEvent_t,2> timing_{};
};
}
