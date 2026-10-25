#include "strata/runtime/capacity_storage.hpp"
#include "strata/gemma4/native/history.hpp"
#include "strata/qwen35/native/weights.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>
namespace strata::gemma4::native {
namespace {
void cold(cudaStream_t stream){
    if(!stream)throw std::invalid_argument("History cold stream");
    cudaStreamCaptureStatus status{};
    primitive::check_cuda(cudaStreamIsCapturing(stream,&status),"History cold capture check");
    if(status!=cudaStreamCaptureStatusNone)throw std::invalid_argument("History storage/reset outside Graph");
    primitive::check_cuda(cudaStreamSynchronize(stream),"History cold owner join");
}
}
struct HistoryCache::Allocations {
    strata::runtime::CapacityStorage gpu;void *host_key=nullptr,*host_value=nullptr;cudaStream_t stream=nullptr;
    ~Allocations(){if(stream)(void)cudaStreamSynchronize(stream);
        if(host_key)cudaFreeHost(host_key);if(host_value)cudaFreeHost(host_value);}
};
namespace history_detail {
void validate(HistoryView v,HistoryBinding live){
    if(!live.owner||!live.epoch||!live.stream||v.binding.owner!=live.owner||
       v.binding.epoch!=live.epoch||v.binding.stream!=live.stream||
       v.logical_capacity<1||v.logical_capacity>262144||
       v.pages!=(v.logical_capacity+15)/16||v.resident!=(std::min)(v.pages*16,HistoryResident)||
       v.slots!=v.resident/16||!v.host_key||!v.host_value||!v.key||!v.value||
       !v.bounds||!v.scores||!v.sorted_scores||!v.page_ids||!v.sorted_ids||!v.offsets||
       !v.page_slot||!v.slot_page||!v.marked||!v.miss_rank||!v.free_slot||
       !v.selected_page||!v.selected_slot||!v.miss||!v.selected_count||
       !v.valid_end||!v.invalid||!v.device_epoch||!v.counters||!v.sort_work||!v.sort_bytes)
        throw std::invalid_argument("History typed owner/epoch/stream/extent lease");
    const void* pools[]{v.host_key,v.host_value,v.key,v.value};
    const size_t sizes[]{size_t(v.pages)*HistoryPage*HistoryKvHeads*HistoryRowBytes,
        size_t(v.pages)*HistoryPage*HistoryKvHeads*HistoryRowBytes,
        size_t(v.resident)*HistoryKvHeads*HistoryRowBytes,size_t(v.resident)*HistoryKvHeads*HistoryRowBytes};
    for(int i=0;i<4;++i){const auto x=reinterpret_cast<std::uintptr_t>(pools[i]);
        if(x%16)throw std::invalid_argument("History aligned RAM/resident pools");
        for(int j=i+1;j<4;++j){const auto y=reinterpret_cast<std::uintptr_t>(pools[j]);
            if(x<=y?y-x<sizes[i]:x-y<sizes[j])
                throw std::invalid_argument("History distinct RAM/resident pool extents");}}
}
}
HistoryCache::HistoryCache(int logical,int maximum_resident,cudaStream_t stream){
    if(logical<1||logical>262144||maximum_resident!=HistoryResident)
        throw std::invalid_argument("History logical<=262144/maxresident32768");
    cold(stream);storage_=std::make_unique<Allocations>();storage_->stream=stream;
    auto& v=view_;v.logical_capacity=logical;v.pages=(logical+15)/16;
    v.resident=(std::min)(v.pages*16,maximum_resident);v.slots=v.resident/16;
    v.binding={this,1,stream};const size_t one_host=size_t(v.pages)*HistoryPage*HistoryKvHeads*HistoryRowBytes;
    primitive::check_cuda(cudaHostAlloc(&storage_->host_key,one_host,cudaHostAllocMapped),"History authoritative mapped K");
    primitive::check_cuda(cudaHostAlloc(&storage_->host_value,one_host,cudaHostAllocMapped),"History authoritative mapped V");
    primitive::check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&v.host_key),storage_->host_key,0),"History mapped K device alias");
    primitive::check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&v.host_value),storage_->host_value,0),"History mapped V device alias");
    auto allocate=[&](size_t bytes){return storage_->gpu.allocate(bytes);};
    auto ints=[&](size_t count){return static_cast<int*>(allocate(count*sizeof(int)));};
    const size_t pages=size_t(HistoryKvHeads)*v.pages,slots=size_t(HistoryKvHeads)*v.slots;
    v.key=static_cast<unsigned char*>(allocate(resident_bytes()/2));
    v.value=static_cast<unsigned char*>(allocate(resident_bytes()/2));
    v.bounds=static_cast<__half*>(allocate(pages*1024*sizeof(__half)));
    v.scores=static_cast<float*>(allocate(pages*sizeof(float)));
    v.sorted_scores=static_cast<float*>(allocate(pages*sizeof(float)));
    v.page_ids=ints(pages);v.sorted_ids=ints(pages);v.offsets=ints(HistoryKvHeads+1);
    v.page_slot=ints(pages);v.slot_page=ints(slots);v.marked=ints(slots);
    v.miss_rank=ints(slots);v.free_slot=ints(slots);
    v.selected_page=ints(slots*HistoryQueries);v.selected_slot=ints(slots*HistoryQueries);
    v.miss=ints(slots*HistoryQueries);v.selected_count=ints(HistoryKvHeads*HistoryQueries);
    v.valid_end=ints(1);v.invalid=static_cast<unsigned*>(allocate(sizeof(unsigned)));
    v.device_epoch=static_cast<std::uint64_t*>(allocate(sizeof(std::uint64_t)));
    v.counters=static_cast<HistoryCounters*>(allocate(sizeof(HistoryCounters)));
    std::vector<int> offsets(HistoryKvHeads+1);
    for(int h=0;h<=HistoryKvHeads;++h)offsets[h]=h*v.pages;
    primitive::check_cuda(cudaMemcpyAsync(v.offsets,offsets.data(),offsets.size()*sizeof(int),
        cudaMemcpyHostToDevice,stream),"History CUB segment offsets");
    v.sort_bytes=history_sort_workspace(v);v.sort_work=allocate(v.sort_bytes);
    history_reset(v,v.binding);
    primitive::check_cuda(cudaStreamSynchronize(stream),"History cold storage ready");
}
HistoryCache::~HistoryCache(){if(view_.binding.stream)(void)cudaStreamSynchronize(view_.binding.stream);}
const void* HistoryCache::host_key()const{return storage_->host_key;}
const void* HistoryCache::host_value()const{return storage_->host_value;}
size_t HistoryCache::host_bytes()const{return size_t(view_.pages)*HistoryPage*HistoryKvHeads*HistoryRowBytes*2;}
size_t HistoryCache::resident_bytes()const{return size_t(view_.resident)*HistoryKvHeads*HistoryRowBytes*2;}
size_t HistoryCache::device_bytes()const{return storage_->gpu.bytes();}
CacheView HistoryCache::initial_resident_view()const{
    return {view_.key,view_.value,(std::min)(view_.logical_capacity,view_.resident),HistoryKvHeads,HistoryDim,0};
}
void HistoryCache::reset(){
    cold(view_.binding.stream);
    if(view_.binding.epoch==(std::numeric_limits<std::uint64_t>::max)())
        throw std::overflow_error("History reset epoch overflow");
    ++view_.binding.epoch;history_reset(view_,view_.binding);
    primitive::check_cuda(cudaStreamSynchronize(view_.binding.stream),"History cold reset complete");
}
}
