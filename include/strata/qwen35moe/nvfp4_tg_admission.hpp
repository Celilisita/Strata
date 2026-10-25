#pragma once
#include "strata/qwen35moe/expert_prefetch.hpp"
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace strata::qwen35moe {
// A bounded pending transfer plan for the EXISTING cache directory. No weights,
// GPU arena, executor, stream or victim policy are owned here.
class Nvfp4TgAdmission {
public:
    struct Copy { int32_t key=-1,slot=-1; };
    struct Cost { HostCopyCost payload{};uint64_t directory_bytes=0,rows=0;double enqueue_ms=0; };
    explicit Nvfp4TgAdmission(size_t layers):layers_(layers){
        need(layers&&layers<=40,"TG admission layer bound");
        ck(cudaMallocHost(reinterpret_cast<void**>(&directory_),layers*256*sizeof(int32_t)));
    }
    // Runtime drains the borrowed copy owner before destroying this pinned source.
    ~Nvfp4TgAdmission(){if(directory_)cudaFreeHost(directory_);}
    Nvfp4TgAdmission(const Nvfp4TgAdmission&)=delete;
    Nvfp4TgAdmission& operator=(const Nvfp4TgAdmission&)=delete;
    void open(ExpertPrefetch& owner,PpPrefetchCache c,const HostReadOnlyLease& source,
              uint32_t decode_epoch,int32_t* gpu_directory,size_t directory_bytes){
        need(!owner_&&!failed_&&decode_epoch&&gpu_directory&&directory_bytes==layers_*256*sizeof(int32_t),"TG admission open state");
        geometry(c);owner_=&owner;cache_=c;source_=&source;source_epoch_=source.epoch();
        decode_epoch_=decode_epoch;gpu_directory_=gpu_directory;
        try{copy_=owner.borrow_decode_copy();}catch(...){owner_=nullptr;throw;}
        token_=0;count_=0;dirty_=submitted_=joined_=0;planning_=false;
    }
    bool opened()const{return owner_!=nullptr;}
    bool planning()const{return planning_;}
    bool pending()const{return dirty_!=submitted_;}
    cudaStream_t stream()const{need(owner_&&!failed_,"closed TG copy stream");return copy_;}
    size_t planned_copies()const{return count_;}
    uint64_t dirty_mask()const{return dirty_;}
    void begin_plan(PpPrefetchCache c,const HostReadOnlyLease& source,uint32_t epoch,uint32_t token){
        validate(c,source);need(epoch==decode_epoch_&&token&&token_!=(std::numeric_limits<uint32_t>::max)()&&token==token_+1&&!planning_,"TG plan epoch/token");
        // Existing head sync precedes this call. Reusing pinned rows/events before
        // every old dirty layer was joined and completed would corrupt a reader.
        need(submitted_==dirty_&&joined_==dirty_,"unconsumed prior TG admission plan");
        for(size_t l=0;l<layers_;++l)if(bit(dirty_,l))ck(cudaEventQuery(owner_->decode_ready(l)));
        token_=token;count_=0;dirty_=submitted_=joined_=0;planning_=true;
    }
    void reserve(int32_t key,int32_t slot,int32_t old_key){
        need(owner_&&!failed_&&planning_&&key>=0&&size_t(key)<layers_*256&&slot>=0&&
             size_t(slot)<cache_.slots->size()&&size_t(slot)>=cache_.resident_prefix&&count_<copies_.size(),"TG reservation geometry");
        need((*cache_.keys)[size_t(key)]<0&&(*cache_.slots)[size_t(slot)]==old_key,"TG reservation preimage");
        for(size_t i=0;i<count_;++i)need(copies_[i].slot!=slot&&copies_[i].key!=key,"TG current-token slot reused");
        copies_[count_++]={key,slot};dirty_|=uint64_t(1)<<(size_t(key)/256);
        if(old_key>=0){need(size_t(old_key)<layers_*256,"TG evicted key bound");dirty_|=uint64_t(1)<<(size_t(old_key)/256);}
    }
    void seal(PpPrefetchCache c,const HostReadOnlyLease& source){
        validate(c,source);need(planning_,"TG seal without reservation phase");
        for(size_t i=0;i<count_;++i){const auto r=copies_[i];need((*c.keys)[size_t(r.key)]==r.slot&&
            (*c.slots)[size_t(r.slot)]==r.key,"TG reservation not final");}
        std::memcpy(directory_,c.keys->data(),layers_*256*sizeof(int32_t));planning_=false;
    }
    template<class CopyPayload>
    Cost submit(size_t layer,PpPrefetchCache c,const HostReadOnlyLease& source,CopyPayload copy_payload){
        validate(c,source);need(!planning_&&layer<layers_,"TG submit state");Cost cost;
        if(!bit(dirty_,layer)||bit(submitted_,layer))return cost;
        try{
            owner_->begin_decode_layer(layer);
            for(size_t i=0;i<count_;++i)if(size_t(copies_[i].key)/256==layer){const auto r=copies_[i];
                need((*c.keys)[size_t(r.key)]==r.slot&&(*c.slots)[size_t(r.slot)]==r.key,"TG pending destination changed");
                cost.payload+=copy_payload(r,copy_);
            }
            // FINAL row includes evictions into later layers. Publish no transient directory.
            ck(cudaMemcpyAsync(gpu_directory_+layer*256,directory_+layer*256,256*sizeof(int32_t),cudaMemcpyHostToDevice,copy_));
            owner_->record_decode_layer(layer);submitted_|=uint64_t(1)<<layer;
            cost.directory_bytes=256*sizeof(int32_t);cost.rows=1;return cost;
        }catch(...){invalidate();throw;}
    }
    void wait(size_t layer,PpPrefetchCache c,const HostReadOnlyLease& source,cudaStream_t compute){
        validate(c,source);need(!planning_&&layer<layers_&&compute,"TG wait state");
        if(!bit(dirty_,layer)||bit(joined_,layer))return;
        need(bit(submitted_,layer),"TG wait before ready was recorded");
        try{ck(cudaStreamWaitEvent(compute,owner_->decode_ready(layer),0));joined_|=uint64_t(1)<<layer;}
        catch(...){invalidate();throw;}
    }
    // finish_decode has flushed all layers and synchronized compute after their
    // ready waits. No extra per-token synchronization is introduced here.
    void close(){
        need(owner_&&!failed_&&!planning_&&submitted_==dirty_&&joined_==dirty_,"TG close has unsent work");
        for(size_t l=0;l<layers_;++l)if(bit(dirty_,l))ck(cudaEventQuery(owner_->decode_ready(l)));
        owner_->release_decode_copy();owner_=nullptr;copy_=nullptr;count_=0;dirty_=submitted_=joined_=0;
    }
    void invalidate()noexcept{
        failed_=true;planning_=false;
        if(!cache_.keys||!cache_.slots||!cache_.ages)return;
        for(size_t i=0;i<count_;++i){const auto r=copies_[i];
            if(size_t(r.key)<cache_.keys->size()&&(*cache_.keys)[size_t(r.key)]==r.slot)(*cache_.keys)[size_t(r.key)]=-1;
            if(r.slot>=0&&size_t(r.slot)<cache_.slots->size()&&(*cache_.slots)[size_t(r.slot)]==r.key){
                (*cache_.slots)[size_t(r.slot)]=-1;(*cache_.ages)[size_t(r.slot)]=0;}}
    }
private:
    static void need(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
    static void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
    static bool bit(uint64_t mask,size_t layer){return bool(mask&(uint64_t(1)<<layer));}
    void geometry(PpPrefetchCache c)const{
        need(c.data&&c.keys&&c.slots&&c.ages&&c.keys->size()==layers_*256&&c.slots->size()==c.ages->size()&&
             c.slots->size()<=layers_*256&&c.resident_prefix%256==0&&c.resident_prefix/256<=layers_&&
             c.slots->size()>=c.resident_prefix+8*(layers_-c.resident_prefix/256)&&
             c.stride>=1769472&&c.stride%256==0&&c.stride<=(std::numeric_limits<size_t>::max)()/c.slots->size()&&
             c.data->bytes()==c.slots->size()*c.stride,"TG admission cache geometry");
    }
    void validate(PpPrefetchCache c,const HostReadOnlyLease& source)const{
        need(owner_&&!failed_&&source_==&source&&source.epoch()==source_epoch_&&
             c.data==cache_.data&&c.keys==cache_.keys&&c.slots==cache_.slots&&c.ages==cache_.ages&&
             c.stride==cache_.stride&&c.resident_prefix==cache_.resident_prefix&&c.epoch==cache_.epoch,
             "stale TG admission cache/source epoch");geometry(c);
    }
    size_t layers_=0,count_=0;
    std::array<Copy,320> copies_{};
    PpPrefetchCache cache_{};
    const HostReadOnlyLease* source_=nullptr;
    ExpertPrefetch* owner_=nullptr;
    cudaStream_t copy_=nullptr;
    int32_t *directory_=nullptr,*gpu_directory_=nullptr;
    uint64_t source_epoch_=0,dirty_=0,submitted_=0,joined_=0;
    uint32_t decode_epoch_=0,token_=0;
    bool planning_=false,failed_=false;
};
}
