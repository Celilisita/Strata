#pragma once
#include "strata/qwen35moe/expert_prefetch.hpp"
#include <limits>
#include <stdexcept>
namespace strata::qwen35moe {
struct ExpertCopyRunsResult { HostCopyCost copy;size_t loaded_unique=0,runs=0; };
// Disposable actual-route reservation. No directory, weights, arena or stream
// ownership; caller has completed the existing router's compute-stream drain.
inline ExpertCopyRunsResult copy_expert_runs(PpPrefetchCache c,size_t layer,
    const PpPrefetchSource& source,const std::vector<int32_t>& ids,
    const HostReadOnlyLease& lease,cudaStream_t stream){
    auto need=[](bool ok,const char* message){if(!ok)throw std::runtime_error(message);};
    need(c.data&&c.keys&&c.slots&&c.ages&&layer<128&&c.keys->size()%256==0&&c.keys->size()/256<=128&&c.slots->size()<=c.keys->size()&&
         layer<c.keys->size()/256&&c.slots->size()==c.ages->size()&&c.stride&&
         c.resident_prefix<=c.slots->size()&&c.slots->size()<=(std::numeric_limits<size_t>::max)()/c.stride&&
         c.data->bytes()==c.slots->size()*c.stride&&!ids.empty(),"invalid copy-runs authority");
    if(c.slots->size()-c.resident_prefix<256)return {};
    size_t bytes=0;
    for(size_t part=0;part<3;++part){
        const auto start=reinterpret_cast<uintptr_t>(source.base[part]);
        need(start&&source.bytes[part]&&source.bytes[part]<=(std::numeric_limits<size_t>::max)()/256&&
             start<=(std::numeric_limits<uintptr_t>::max)()-source.bytes[part]*256&&
             bytes<=c.stride&&source.bytes[part]<=c.stride-bytes,"invalid copy-runs source extent");
        bytes+=source.bytes[part];
    }
    std::array<bool,256> requested{};
    for(int id:ids){need(id>=0&&id<256,"invalid copy-runs actual ID");requested[size_t(id)]=true;}
    struct Entry {int id=-1,slot=-1,key=-1;};std::array<Entry,256> pending{};size_t count=0;
    for(size_t id=0;id<256;++id)if(requested[id]){
        const auto key=layer*256+id;const int slot=(*c.keys)[key];
        need(slot<0||(size_t(slot)<c.slots->size()&&(*c.slots)[size_t(slot)]==int(key)),"copy-runs warm inverse mismatch");
        if(slot<0)pending[count++]={int(id),-1,int(key)};
    }
    if(!count)return {};
    // Consume every free slot before eviction: do not collapse a960-slot
    // directory to the first256 slots. Then approximate LRU by64-slot chunk.
    // Current actual warm keys and every resident-prefix slot stay protected.
    struct Chunk {size_t first,last;uint64_t age;};std::vector<Chunk> chunks;
    size_t available=0;
    for(size_t first=0;first<c.slots->size();first+=ExpertStorage::chunk_slots){
        const size_t last=(std::min)(c.slots->size(),first+ExpertStorage::chunk_slots);
        bool occupied=false;uint64_t age=0;
        for(size_t slot=(std::max)(first,c.resident_prefix);slot<last;++slot){
            const int key=(*c.slots)[slot];
            need(key<0||(size_t(key)<c.keys->size()&&(*c.keys)[size_t(key)]==int(slot)),"copy-runs victim inverse mismatch");
            if(key>=0&&size_t(key)/256==layer&&requested[size_t(key)%256])continue;
            if(key<0){if(available<count)pending[available++].slot=int(slot);}
            else{occupied=true;age=(std::max)(age,(*c.ages)[slot]);}
        }
        if(occupied)chunks.push_back({first,last,age});
    }
    std::sort(chunks.begin(),chunks.end(),[](const Chunk& a,const Chunk& b){return a.age!=b.age?a.age<b.age:a.first<b.first;});
    for(const auto& chunk:chunks)for(size_t slot=(std::max)(chunk.first,c.resident_prefix);slot<chunk.last&&available<count;++slot){
        const int key=(*c.slots)[slot];
        if(key<0||(size_t(key)/256==layer&&requested[size_t(key)%256]))continue;
        pending[available++].slot=int(slot);
    }
    if(available<count)return {};
    const auto source_epoch=lease.epoch();ExpertCopyRunsResult result;
    try{
        for(size_t i=0;i<count;++i){const auto& e=pending[i];const int old=(*c.slots)[size_t(e.slot)];
            if(old>=0)(*c.keys)[size_t(old)]=-1;
            (*c.keys)[size_t(e.key)]=e.slot;(*c.slots)[size_t(e.slot)]=e.key;(*c.ages)[size_t(e.slot)]=0;
        }
        for(size_t at=0;at<count;){
            const auto& first=pending[at];size_t height=1;
            while(at+height<count&&pending[at+height].id==first.id+int(height)&&
                  pending[at+height].slot==first.slot+int(height)&&
                  size_t(pending[at+height].slot)/ExpertStorage::chunk_slots==size_t(first.slot)/ExpertStorage::chunk_slots)++height;
            auto* target=c.data->slot(size_t(first.slot));size_t offset=0;
            for(size_t part=0;part<3;++part){
                result.copy+=lease.copy_to_device_rows(target+offset,c.stride,
                    source.base[part]+size_t(first.id)*source.bytes[part],source.bytes[part],height,
                    stream,"actual expert run H2D",first.key,int(part));offset+=source.bytes[part];
            }
            ++result.runs;at+=height;
        }
        need(lease.epoch()==source_epoch,"copy-runs GGUF epoch changed");
        result.loaded_unique=count;return result;
    }catch(...){
        // A partial copy may already be queued. Never expose those reservations
        // after failure; Runtime poisons the batch and drains before reuse/free.
        for(size_t i=0;i<count;++i){const auto& e=pending[i];
            if((*c.keys)[size_t(e.key)]==e.slot)(*c.keys)[size_t(e.key)]=-1;
            if((*c.slots)[size_t(e.slot)]==e.key){(*c.slots)[size_t(e.slot)]=-1;(*c.ages)[size_t(e.slot)]=0;}
        }throw;
    }
}
}
