#include "strata/qwen35moe/expert_prefetch.hpp"
#include "strata/qwen35moe/pp_consumed_victims.hpp"
#include "strata/qwen35moe/expert_copy_stage_cuda.hpp"
#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>

namespace strata::qwen35moe {
namespace {
void need(bool ok,const char* text){if(!ok)throw std::runtime_error(text);}
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
size_t stage_chunk(PpPrefetchCache c,size_t layer){
    const size_t none=size_t(-1),width=ExpertStorage::chunk_slots;
    if(c.slots->size()-c.resident_prefix<width+256)return none;
    size_t best=none;uint64_t oldest=(std::numeric_limits<uint64_t>::max)();
    for(size_t first=0;first+width<=c.slots->size();first+=width){
        if(first<c.resident_prefix)continue;bool safe=true;uint64_t age=0;
        for(size_t slot=first;slot<first+width;++slot){const int key=(*c.slots)[slot];
            // Prefer consumed older layers only: never sacrifice current/previous
            // or future-layer warm payloads merely to create an upload window.
            if(key< -1||(key>=0&&(layer<2||size_t(key)/256>=layer-1))){safe=false;break;}
            if(key>=0)age=(std::max)(age,(*c.ages)[slot]);
        }
        if(safe&&(best==none||age<oldest)){best=first;oldest=age;}
    }return best;
}
}
ExpertPrefetch::ExpertPrefetch(size_t layers):layers_(layers),events_(layers),recorded_(layers,false){
    need(layers&&layers<=128,"prefetch layer bound");
    try{
        ck(cudaStreamCreateWithFlags(&copy_,cudaStreamNonBlocking));
        for(auto& pair:events_)for(auto& event:pair)ck(cudaEventCreate(&event));
    }catch(...){
        for(auto& pair:events_)for(auto event:pair)if(event)cudaEventDestroy(event);
        if(copy_)cudaStreamDestroy(copy_);throw;
    }
}
ExpertPrefetch::~ExpertPrefetch(){
    // The Runtime resets this member before unregistering the immutable GGUF.
    if(copy_)cudaStreamSynchronize(copy_);
    release_stage();
    for(auto& pair:events_)for(auto event:pair)if(event)cudaEventDestroy(event);
    if(copy_)cudaStreamDestroy(copy_);
}
void ExpertPrefetch::begin_request(){
    need(!failed_&&!active_&&!decode_borrowed_,"prefetch requires drained request boundary");
    ++request_;cost_={};fallback_layer_=none;fallback_.fill(false);prefix_routes_.clear();actual_only_=false;
    std::fill(recorded_.begin(),recorded_.end(),false);
}
void ExpertPrefetch::validate(PpPrefetchCache c,const HostReadOnlyLease& lease)const{
    need(active_&&!failed_&&plan_request_==request_&&lease_==&lease&&lease.epoch()==source_epoch_,
         "stale prefetch request/GGUF epoch");
    need(c.data==cache_.data&&c.keys==cache_.keys&&c.slots==cache_.slots&&c.ages==cache_.ages&&
         c.epoch==cache_.epoch&&c.stride==cache_.stride&&c.resident_prefix==cache_.resident_prefix,
         "stale prefetch cache epoch/authority");
    if(stage_first_!=none){need(stage_first_+64<=c.slots->size(),"stale reserved stage extent");
        for(size_t i=stage_first_;i<stage_first_+64;++i)need((*c.slots)[i]==-2&&(*c.ages)[i]==0,"reserved stage directory changed");}
    for(const auto& e:entries_)if(e.key>=0)need(e.slot>=0&&size_t(e.slot)<c.slots->size()&&
        (*c.keys)[size_t(e.key)]==e.slot&&(*c.slots)[size_t(e.slot)]==e.key,"prefetch directory changed");
}
void ExpertPrefetch::close_layer(){
    if(!active_)return;
    need(joined_,"unjoined prefetch layer cannot advance");
    need(stage_first_==none,"unreleased stage cannot advance");
    const uint64_t bytes=sizes_[0]+sizes_[1]+sizes_[2];
    for(const auto& e:entries_)if(e.copied&&!e.used){++cost_.unused_unique;cost_.unused_bytes+=bytes;}
    active_=false;
}
bool ExpertPrefetch::enqueue(PpPrefetchCache c,size_t layer,const PpPrefetchSource& source,
                             const HostReadOnlyLease& lease,const std::vector<int32_t>* actual_ids,bool prefix_stage){
    const bool consumed_first=pp_consumed_victims::enabled();
    need(!failed_&&!decode_borrowed_&&layer<layers_,"prefetch invalid layer/failed runtime");
    need(!prefix_stage||actual_ids,"staged prefix requires actual routes");
    std::array<bool,256> requested{};
    if(actual_ids){
        need(!actual_ids->empty()&&actual_ids->size()<=32768,"prefix demand route extent");
        for(int id:*actual_ids){need(id>=0&&id<256,"prefix demand route ID");requested[size_t(id)]=true;}
    }else requested.fill(true);
    if(active_&&layer_==layer&&!actual_ids){validate(c,lease);return true;}
    if(active_){validate(c,lease);
        need(actual_ids ? (layer>=layer_||(layer==0&&layer_+1==layers_)) : layer==layer_+1,
             "prefetch layer order");close_layer();}
    // Prefix routing drained the compute stream, including the previous chunk's
    // consumers. Collect old timing before re-recording this layer's event pair.
    if(actual_ids&&recorded_[layer]){
        ck(cudaEventQuery(events_[layer][1]));float ms=0;
        ck(cudaEventElapsedTime(&ms,events_[layer][0],events_[layer][1]));
        cost_.copy_ms+=ms;recorded_[layer]=false;
    }
    need(c.data&&c.keys&&c.slots&&c.ages&&c.keys->size()==layers_*256&&
         c.slots->size()==c.ages->size()&&c.resident_prefix<=c.slots->size(),"prefetch cache view");
    need(c.stride&&c.slots->size()<=(std::numeric_limits<size_t>::max)()/c.stride&&
         c.data->bytes()==c.slots->size()*c.stride,"prefetch physical slot geometry");
    if(c.slots->size()<c.resident_prefix+512)return false;
    size_t payload=0;
    for(size_t p=0;p<3;++p){
        need(source.base[p]&&source.bytes[p]&&source.bytes[p]<=(std::numeric_limits<size_t>::max)()/256,
             "prefetch source extent");
        need(payload<=c.stride&&source.bytes[p]<=c.stride-payload,"prefetch slot stride");
        const auto begin=reinterpret_cast<uintptr_t>(source.base[p]);
        need(begin<=(std::numeric_limits<uintptr_t>::max)()-source.bytes[p]*256,"prefetch source address overflow");
        payload+=source.bytes[p];
    }
    const auto start=std::chrono::steady_clock::now();
    std::array<Entry,256> prepared{};std::vector<size_t> victims;size_t misses=0;
    const size_t requested_count=size_t(std::count(requested.begin(),requested.end(),true));
    for(size_t id=0;id<256;++id)if(requested[id]){
        const int32_t key=int32_t(layer*256+id),slot=(*c.keys)[size_t(key)];
        need(slot<0||(size_t(slot)<c.slots->size()&&(*c.slots)[size_t(slot)]==key),"prefetch warm key mismatch");
        prepared[id]={key,slot,slot<0,false};if(slot<0)++misses;
    }
    size_t reserved=none;bool has_run=false;unsigned run=0;
    if(prefix_stage&&misses&&source.bytes[0]==589824&&source.bytes[1]==589824&&
       (source.bytes[2]==589824||source.bytes[2]==720896)){
        for(const auto& e:prepared){run=e.copied?run+1:0;if(run>=4)has_run=true;}
        if(has_run)reserved=stage_chunk(c,layer);
    }
    auto collect=[&](size_t excluded){victims.clear();
        for(size_t slot=c.resident_prefix;slot<c.slots->size();++slot){const int key=(*c.slots)[slot];
            need(key>=-1&&(key<0||(size_t(key)<c.keys->size()&&(*c.keys)[size_t(key)]==int(slot))),"prefetch victim key mismatch");
            if(excluded!=none&&slot>=excluded&&slot<excluded+64)continue;
            if(key>=0&&(size_t(key)/256==layer||(layer&&size_t(key)/256==layer-1)))continue;
            victims.push_back(slot);
        }};
    collect(reserved);
    if(reserved!=none&&victims.size()<misses){reserved=none;collect(none);}
    if(victims.size()<misses)return false;
    std::sort(victims.begin(),victims.end(),pp_consumed_victims::Order{
        *c.slots,*c.ages,layer,consumed_first});
    cache_=c;layer_=layer;entries_=prepared;sizes_=source.bytes;lease_=&lease;
    actual_only_=actual_ids!=nullptr;prefix_routes_=actual_ids?*actual_ids:std::vector<int32_t>{};
    source_epoch_=lease.epoch();plan_request_=request_;active_=true;joined_=misses==0;
    size_t at=0;
    try{
        if(prefix_stage&&misses&&reserved==none)++cost_.stage_fallbacks;
        if(reserved!=none){stage_first_=reserved;
            for(size_t slot=reserved;slot<reserved+64;++slot){const int old=(*c.slots)[slot];
                if(old>=0){(*c.keys)[size_t(old)]=-1;++cost_.stage_evicted_keys;}
                (*c.slots)[slot]=-2;(*c.ages)[slot]=0;}
            ++cost_.stage_layers;cost_.stage_peak_slots=64;
        }
        // Publish reservations as pending; route() is the only ready barrier.
        for(auto& e:entries_)if(e.copied){
            e.slot=int32_t(victims[at++]);const int old=(*c.slots)[size_t(e.slot)];
            if(old>=0)(*c.keys)[size_t(old)]=-1;
            (*c.keys)[size_t(e.key)]=e.slot;(*c.slots)[size_t(e.slot)]=e.key;
            (*c.ages)[size_t(e.slot)]=0; // Speculation is not a demand-age hit.
        }
        if(misses){ck(cudaEventRecord(events_[layer][0],copy_));recorded_[layer]=true;}
        std::array<unsigned,256> cold{};size_t cold_count=0;
        for(unsigned id=0;id<256;++id)if(entries_[id].copied)cold[cold_count++]=id;
        for(size_t begin=0;begin<cold_count;){size_t count=1;
            if(stage_first_!=none)while(begin+count<cold_count&&count<64&&cold[begin+count]==cold[begin]+count)++count;
            if(count<4)count=1;
            const auto& first=entries_[cold[begin]];
            auto* dst=c.data->slot(count>1?stage_first_:size_t(first.slot));size_t offset=0;
            for(size_t part=0;part<3;++part){
                const auto* src=source.base[part]+size_t(cold[begin])*source.bytes[part];const size_t bytes=count*source.bytes[part];
                const auto copied=lease.copy_to_device(dst+offset,src,bytes,copy_,
                    count>1?"prefix contiguous stage H2D":"prefetch expert H2D",first.key,int(part));offset+=bytes;
                cost_.h2d_bytes+=copied.bytes;cost_.h2d_calls+=copied.calls;
                cost_.logical_copies+=copied.logical_copies;
                cost_.registered_bytes+=copied.registered_bytes;
                cost_.pageable_bytes+=copied.pageable_bytes;
            }
            if(count>1){ExpertStageTargets targets{};
                for(size_t i=0;i<count;++i)targets.addresses[i]=uintptr_t(c.data->slot(size_t(entries_[cold[begin+i]].slot)));
                expert_stage_scatter(dst,64*c.stride,targets,source.bytes[0],source.bytes[2],unsigned(count),copy_);
                cost_.logical_copies+=3*(count-1);++cost_.stage_scatter_launches;
                cost_.stage_bytes+=count*payload;
            }
            cost_.issued_unique+=count;begin+=count;
        }
        if(misses)ck(cudaEventRecord(events_[layer][1],copy_));
        ++cost_.layers;cost_.warm_present_unique+=requested_count-misses;
        cost_.enqueue_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        return true;
    }catch(...){failed_=true;invalidate();throw;}
}
void ExpertPrefetch::route(PpPrefetchCache c,size_t layer,const std::vector<int32_t>& ids,
                           const HostReadOnlyLease& lease,cudaStream_t compute,bool current_prefix){
    if(!active_)return;
    // Full router already drained prior consumers. A skipped prefix can leave
    // this same layer active from an older chunk; it is not a current prefix.
    if(actual_only_&&!current_prefix){validate(c,lease);close_layer();return;}
    // A skipped (e.g. fully warm) next layer releases the prior protection after
    // its existing router drain, before that next layer selects any cache victim.
    if(layer!=layer_){if(actual_only_){validate(c,lease);close_layer();}return;}
    validate(c,lease);need(!ids.empty(),"prefetch empty actual routes");
    for(int id:ids)need(id>=0&&id<256,"prefetch actual route ID");
    try{
        if(actual_only_)need(ids.size()>=prefix_routes_.size()&&
            std::equal(prefix_routes_.begin(),prefix_routes_.end(),ids.begin()),"prefix route sequence changed");
        if(!joined_){ck(cudaStreamWaitEvent(compute,events_[layer][1],0));joined_=true;++cost_.joins;}
        // Later cache copies use COMPUTE and follow the queued ready wait. They
        // may reuse this physical range only after COPY's final scatter finishes.
        release_stage();
        const uint64_t bytes=sizes_[0]+sizes_[1]+sizes_[2];
        for(int id:ids){auto& e=entries_[size_t(id)];if(e.key<0)continue;
            if(e.copied)++cost_.route_refs;
            if(!e.used){e.used=true;if(e.copied){++cost_.used_unique;cost_.used_bytes+=bytes;}else ++cost_.warm_used_unique;}
        }
        if(actual_only_)for(const auto& e:entries_)need(e.key<0||e.used,"unused prefix demand copy");
    }catch(...){failed_=true;invalidate();throw;}
}
bool ExpertPrefetch::protects(int32_t key)const{
    return active_&&key>=0&&(size_t(key)/256==layer_||(layer_&&size_t(key)/256==layer_-1));
}
bool ExpertPrefetch::copied(size_t layer,int id)const{
    if(!active_||layer!=layer_)return false;
    need(joined_&&id>=0&&id<256,"pending prefetch lookup before ready barrier");
    auto& e=entries_[size_t(id)];
    if(!e.copied)return false;
    if(!actual_only_)return true;
    const bool first=!e.looked;e.looked=true;return first; // Only first real cold reference is a miss.
}
void ExpertPrefetch::demand(size_t layer,int id){
    need(id>=0&&id<256,"prefetch fallback ID");
    if(layer!=fallback_layer_){fallback_layer_=layer;fallback_.fill(false);}
    if(!fallback_[size_t(id)]){fallback_[size_t(id)]=true;++cost_.demand_fallback_unique;}
}
void ExpertPrefetch::invalidate()noexcept{
    if(!cache_.keys||!cache_.slots)return;
    for(const auto& e:entries_)if(e.copied&&e.slot>=0&&size_t(e.slot)<cache_.slots->size()){
        if((*cache_.keys)[size_t(e.key)]==e.slot)(*cache_.keys)[size_t(e.key)]=-1;
        if((*cache_.slots)[size_t(e.slot)]==e.key){(*cache_.slots)[size_t(e.slot)]=-1;(*cache_.ages)[size_t(e.slot)]=0;}
    }
    release_stage();
}
void ExpertPrefetch::release_stage()noexcept{
    if(stage_first_==none||!cache_.slots)return;
    for(size_t slot=stage_first_;slot<stage_first_+64&&slot<cache_.slots->size();++slot)
        if((*cache_.slots)[slot]==-2){(*cache_.slots)[slot]=-1;(*cache_.ages)[slot]=0;}
    stage_first_=none;
}
void ExpertPrefetch::finish(){
    need(!failed_,"failed prefetch cannot finish");
    try{
        if(active_)validate(cache_,*lease_);
        close_layer();
        for(size_t i=0;i<layers_;++i)if(recorded_[i]){
            ck(cudaEventQuery(events_[i][1]));float ms=0;
            ck(cudaEventElapsedTime(&ms,events_[i][0],events_[i][1]));cost_.copy_ms+=ms;recorded_[i]=false;
        }
    }catch(...){failed_=true;invalidate();throw;}
}
void ExpertPrefetch::drain(){
    if(!active_&&!failed_&&std::find(recorded_.begin(),recorded_.end(),true)==recorded_.end())return;
    try{
        ck(cudaStreamSynchronize(copy_));++cost_.boundary_drains;
        if(active_&&!failed_){validate(cache_,*lease_);joined_=true;release_stage();close_layer();}
        need(!failed_,"failed prefetch requires runtime recreation");
    }catch(...){failed_=true;invalidate();throw;}
}
cudaStream_t ExpertPrefetch::borrow_decode_copy(){
    need(!failed_&&!active_&&!decode_borrowed_,"TG copy requires drained PP owner");
    for(size_t i=0;i<layers_;++i)if(recorded_[i])ck(cudaEventQuery(events_[i][1]));
    std::fill(recorded_.begin(),recorded_.end(),false);decode_borrowed_=true;return copy_;
}
void ExpertPrefetch::begin_decode_layer(size_t layer){
    need(decode_borrowed_&&!failed_&&layer<layers_,"invalid TG copy lease");
    recorded_[layer]=true; // A partial enqueue must still force drain() before freeing its source.
    ck(cudaEventRecord(events_[layer][0],copy_));
}
void ExpertPrefetch::record_decode_layer(size_t layer){
    need(decode_borrowed_&&!failed_&&layer<layers_,"invalid TG ready lease");
    ck(cudaEventRecord(events_[layer][1],copy_));recorded_[layer]=true;
}
cudaEvent_t ExpertPrefetch::decode_ready(size_t layer)const{
    need(decode_borrowed_&&!failed_&&layer<layers_&&recorded_[layer],"unrecorded TG ready event");
    return events_[layer][1];
}
void ExpertPrefetch::release_decode_copy(){
    need(decode_borrowed_&&!active_&&!failed_,"invalid TG copy release");
    for(size_t i=0;i<layers_;++i)if(recorded_[i])ck(cudaEventQuery(events_[i][1]));
    std::fill(recorded_.begin(),recorded_.end(),false);decode_borrowed_=false;
}
}
