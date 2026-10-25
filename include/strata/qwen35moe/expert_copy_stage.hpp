#pragma once
#include "strata/qwen35moe/expert_copy_runs.hpp"
#include "strata/qwen35moe/expert_copy_stage_cuda.hpp"
#include <algorithm>
#include <array>
#include <limits>
namespace strata::qwen35moe {
struct ExpertCopyStageResult : ExpertCopyRunsResult {
    size_t scatter_launches=0,stage_peak_bytes=0;
    uint64_t scatter_bytes=0;
};
// Caller has completed the ordinary router drain. Borrow only DEAD dense phase
// scratch; every H2D/scatter/next overwrite/GU consumer uses this same stream.
inline ExpertCopyStageResult copy_expert_stage(PpPrefetchCache c,size_t layer,
    const PpPrefetchSource& source,const std::vector<int32_t>& ids,
    const HostReadOnlyLease& lease,void* scratch,size_t scratch_bytes,cudaStream_t stream){
    auto need=[](bool value,const char* what){if(!value)throw std::runtime_error(what);};
    const auto limit=(std::numeric_limits<uintptr_t>::max)();
    need(c.data&&c.keys&&c.slots&&c.ages&&c.keys->size()%256==0&&layer<c.keys->size()/256&&
        c.keys->size()/256<=128&&c.slots->size()==c.ages->size()&&c.resident_prefix<=c.slots->size()&&
        c.stride&&c.slots->size()<=limit/c.stride&&c.data->bytes()==c.slots->size()*c.stride&&
        !ids.empty()&&ids.size()<=32768&&stream,"invalid copy-stage authority");
    if(c.slots->size()-c.resident_prefix<256)return {};
    const size_t gu=source.bytes[0],down=source.bytes[2];
    need(gu==589824&&source.bytes[1]==gu&&(down==589824||down==720896)&&2*gu+down<=c.stride,
         "copy-stage actual Q4/Q5 triplet geometry");
    const size_t payload=2*gu+down;
    const auto stage=uintptr_t(scratch);
    if(!stage||stage%16||scratch_bytes>limit-stage||scratch_bytes<payload*4)return {};
    for(size_t part=0;part<3;++part){const auto base=uintptr_t(source.base[part]);
        need(base&&base<=limit-source.bytes[part]*256,"copy-stage GGUF extent");}
    std::array<bool,256> requested{};
    for(int id:ids){need(id>=0&&id<256,"copy-stage actual ID");requested[size_t(id)]=true;}
    struct Entry {int id=-1,key=-1,slot=-1;};std::array<Entry,256> pending{};size_t misses=0;
    for(size_t id=0;id<256;++id)if(requested[id]){const int key=int(layer*256+id),slot=(*c.keys)[size_t(key)];
        need(slot<0||(size_t(slot)<c.slots->size()&&(*c.slots)[size_t(slot)]==key),"copy-stage warm inverse");
        if(slot<0)pending[misses++]={int(id),key,-1};}
    if(!misses)return {};
    std::vector<size_t> victims;
    for(size_t slot=c.resident_prefix;slot<c.slots->size();++slot){const int key=(*c.slots)[slot];
        need(key<0||(size_t(key)<c.keys->size()&&(*c.keys)[size_t(key)]==int(slot)),"copy-stage victim inverse");
        if(key>=0&&size_t(key)/256==layer&&requested[size_t(key)%256])continue;
        victims.push_back(slot);
    }
    if(victims.size()<misses)return {};
    std::sort(victims.begin(),victims.end(),[&](size_t a,size_t b){
        const auto x=(*c.slots)[a]<0?0:(*c.ages)[a],y=(*c.slots)[b]<0?0:(*c.ages)[b];return x!=y?x<y:a<b;});
    // Validate every destination and staging disjointness BEFORE any mutation/H2D.
    for(size_t i=0;i<misses;++i){pending[i].slot=int(victims[i]);const auto dst=uintptr_t(c.data->slot(victims[i]));
        need(dst&&dst%16==0&&payload<=limit-dst&&!(dst<stage+scratch_bytes&&stage<dst+payload),"copy-stage cache/scratch alias");}
    const auto source_epoch=lease.epoch();ExpertCopyStageResult result;
    try{
        for(size_t i=0;i<misses;++i){const auto& e=pending[i];const int old=(*c.slots)[size_t(e.slot)];
            if(old>=0)(*c.keys)[size_t(old)]=-1;
            (*c.keys)[size_t(e.key)]=e.slot;(*c.slots)[size_t(e.slot)]=e.key;(*c.ages)[size_t(e.slot)]=0;
        }
        const size_t maximum=(std::min)(size_t(64),scratch_bytes/payload);
        for(size_t at=0;at<misses;){size_t count=1;
            while(at+count<misses&&count<maximum&&pending[at+count].id==pending[at].id+int(count))++count;
            const auto& first=pending[at];
            if(count<4){ // Sparse routes keep direct copies; never copy gaps for batching.
                auto* dst=c.data->slot(size_t(first.slot));size_t offset=0;
                for(size_t part=0;part<3;++part){result.copy+=lease.copy_to_device(dst+offset,
                    source.base[part]+size_t(first.id)*source.bytes[part],source.bytes[part],stream,
                    "direct sparse expert H2D",first.key,int(part));offset+=source.bytes[part];}
                count=1;
            }else{
                size_t offset=0;ExpertStageTargets destinations{};
                for(size_t i=0;i<count;++i)destinations.addresses[i]=uintptr_t(c.data->slot(size_t(pending[at+i].slot)));
                for(size_t part=0;part<3;++part){const size_t bytes=count*source.bytes[part];
                    result.copy+=lease.copy_to_device(static_cast<uint8_t*>(scratch)+offset,
                        source.base[part]+size_t(first.id)*source.bytes[part],bytes,stream,
                        "contiguous expert stage H2D",first.key,int(part));offset+=bytes;
                }
                result.copy.logical_copies+=3*(count-1); // Keep per-expert logical accounting.
                expert_stage_scatter(scratch,scratch_bytes,destinations,gu,down,unsigned(count),stream);
                ++result.scatter_launches;result.scatter_bytes+=count*payload;
                result.stage_peak_bytes=(std::max)(result.stage_peak_bytes,count*payload);
            }
            ++result.runs;at+=count;
        }
        need(lease.epoch()==source_epoch,"copy-stage GGUF epoch changed");
        result.loaded_unique=misses;return result;
    }catch(...){
        // Earlier GPU writes may be queued. Caller poisons the batch and drains;
        // none of the new reservations may survive as ready cache entries.
        for(size_t i=0;i<misses;++i){const auto& e=pending[i];
            if((*c.keys)[size_t(e.key)]==e.slot)(*c.keys)[size_t(e.key)]=-1;
            if((*c.slots)[size_t(e.slot)]==e.key){(*c.slots)[size_t(e.slot)]=-1;(*c.ages)[size_t(e.slot)]=0;}}
        throw;
    }
}
}
