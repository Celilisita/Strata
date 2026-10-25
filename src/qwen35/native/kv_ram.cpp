#include "strata/qwen35/native/kv_ram.hpp"
#include "strata/qwen35/native/weights.hpp"
#include <algorithm>
#include <cstdlib>
#include "fresh_pool_policy.hpp"
namespace strata::qwen35::native {
struct RamKvCache::Allocations {
    struct Fixed {
        DeviceStorage gpu;int heads,slots,resident,keep;
        explicit Fixed(const RamKvView& v):heads(v.heads),slots(v.slots),resident(v.resident),keep(v.keep_queries){}
        bool matches(const RamKvView& v)const{
            return heads==v.heads&&slots==v.slots&&resident==v.resident&&keep==v.keep_queries;
        }
    };
    DeviceStorage gpu;
    std::unique_ptr<Fixed> fixed;
    void *key=nullptr,*value=nullptr;bool device_rows=false;
    ~Allocations(){if(key)cudaFreeHost(key);if(value)cudaFreeHost(value);}
};
RamKvCache::RamKvCache(int capacity,int resident,int heads,cudaStream_t stream,int selected_tokens,int keep_queries)
    :RamKvCache(capacity,resident,heads,stream,selected_tokens,keep_queries,false){}
RamKvCache::RamKvCache(int capacity,int resident,int heads,cudaStream_t stream,int selected_tokens,int keep_queries,bool device_rows){
    if(capacity<1||capacity>262144||(resident!=32768&&resident!=40960)||heads<1)throw std::invalid_argument("invalid RAM KV lease");
    if(selected_tokens<1024||selected_tokens>resident||selected_tokens%256)throw std::invalid_argument("invalid history selection budget");
    if(keep_queries<0||(keep_queries+1)*selected_tokens>resident)throw std::invalid_argument("reuse horizon exceeds physical page budget");
    view_.keep_queries=keep_queries;
    view_.selected_tokens=selected_tokens;view_.selected_slots=selected_tokens/RamPage;
    view_.resident=resident;view_.heads=heads;view_.slots=resident/RamPage;grow(capacity,stream,false,device_rows);
}
RamKvCache::~RamKvCache()=default;
bool RamKvCache::device_rows()const{return storage_&&storage_->device_rows;}
size_t RamKvCache::host_bytes()const{return storage_&&storage_->key?size_t(view_.host_capacity)*view_.heads*272*2:0;}
void RamKvCache::reserve(int required,cudaStream_t stream){grow(required,stream,true,device_rows()&&required<=view_.resident);}
void RamKvCache::reserve_discarding(int required,cudaStream_t stream){grow(required,stream,false,false);}
void RamKvCache::reserve_discarding_device(int required,cudaStream_t stream){grow(required,stream,false,true);}
void RamKvCache::preserve_to_host(int required,cudaStream_t stream){
    if(required<1||required>262144)throw std::out_of_range("phase KV logical capacity");
    grow(std::max(required,view_.host_capacity),stream,true,false);
}
void RamKvCache::grow(int required,cudaStream_t stream,bool preserve,bool device){
    if(required<1||required>262144)throw std::out_of_range("RAM KV logical capacity");
    if(device&&required>view_.resident)throw std::invalid_argument("device logical rows exceed resident directory");
    if(required<=view_.host_capacity&&device==device_rows())return;
    if(preserve&&storage_&&!device_rows()&&device)throw std::logic_error("preserving RAM authority cannot shrink to GPU");
    cudaStreamCaptureStatus status{};check_cuda(cudaStreamIsCapturing(stream,&status),"RAM KV growth capture check");
    if(status!=cudaStreamCaptureStatusNone)throw std::logic_error("reserve RAM KV before graph capture");
    check_cuda(cudaStreamSynchronize(stream),"RAM KV cold growth lifetime");
    const int next=std::min(262144,((required+8191)/8192)*8192);
    const char* control=std::getenv("STRATA_DENSE_RAM_REUSE_FRESH_POOL");
    // Device-only growth keeps identity pages and the same fixed GPU pool.
    // Fresh mode changes discard rows; preserving GPU->RAM materializes them.
    const bool reuse=storage_&&storage_->fixed->matches(view_)&&
        ((preserve&&device_rows()&&view_.host_key==view_.key&&view_.host_value==view_.value)||(!preserve&&(control?control[0]!='0':fresh_pool_default(required))));
    auto owned=std::make_unique<Allocations>();owned->device_rows=device;
    if(!reuse)owned->fixed=std::make_unique<Allocations::Fixed>(view_);
    RamKvView n{};
    n.keep_queries=view_.keep_queries;
    n.selected_tokens=view_.selected_tokens;n.selected_slots=view_.selected_slots;
    n.heads=view_.heads;n.resident=view_.resident;n.slots=view_.slots;n.host_capacity=next;n.pages=next/RamPage;
    const size_t bytes=size_t(next)*n.heads*272;
    if(!device){
        check_cuda(cudaHostAlloc(&owned->key,bytes,cudaHostAllocMapped),"authoritative pinned RAM K");
        check_cuda(cudaHostAlloc(&owned->value,bytes,cudaHostAllocMapped),"authoritative pinned RAM V");
        check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void **>(&n.host_key),owned->key,0),"mapped RAM K pointer");
        check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void **>(&n.host_value),owned->value,0),"mapped RAM V pointer");
    }
    auto alloc=[&](size_t b){return owned->gpu.allocate(b);};
    auto ints=[&](size_t count){return static_cast<int *>(alloc(count*4));};
    auto fixed=[&](size_t bytes){return owned->fixed->gpu.allocate(bytes);};
    auto fixed_ints=[&](size_t count){return static_cast<int *>(fixed(count*4));};
    if(reuse){n.key=view_.key;n.value=view_.value;}
    else{n.key=static_cast<unsigned char *>(fixed(resident_bytes()/2));n.value=static_cast<unsigned char *>(fixed(resident_bytes()/2));}
    if(device){n.host_key=n.key;n.host_value=n.value;}
    const size_t pages=size_t(n.heads)*n.pages,slots=size_t(n.heads)*n.slots;
    n.bounds=static_cast<__half *>(alloc(pages*256*2*sizeof(__half)));
    n.scores=static_cast<float *>(alloc(pages*4));n.sorted_scores=static_cast<float *>(alloc(pages*4));
    n.page_ids=ints(pages);n.sorted_ids=ints(pages);n.offsets=ints(n.heads+1);
    n.page_slot=ints(pages);
    if(reuse){
        n.slot_page=view_.slot_page;n.selected_slot=view_.selected_slot;n.miss=view_.miss;
        n.marked=view_.marked;n.miss_rank=view_.miss_rank;n.free_slot=view_.free_slot;
        n.last_use=view_.last_use;n.epoch=view_.epoch;n.counters=view_.counters;
    }else{
        n.slot_page=fixed_ints(slots);n.selected_slot=fixed_ints(slots);n.miss=fixed_ints(slots);
        n.marked=fixed_ints(slots);n.miss_rank=fixed_ints(slots);n.free_slot=fixed_ints(slots);
        if(n.keep_queries){n.last_use=fixed_ints(slots);n.epoch=fixed_ints(n.heads);}
        n.counters=static_cast<RamKvCounters *>(fixed(sizeof(RamKvCounters)));
    }
    std::vector<int> offsets(n.heads+1);for(int h=0;h<=n.heads;++h)offsets[h]=h*n.pages;
    check_cuda(cudaMemcpyAsync(n.offsets,offsets.data(),offsets.size()*4,cudaMemcpyHostToDevice,stream),"RAM indexer segment offsets");
    n.sort_bytes=ram_sort_workspace(n);n.sort_work=alloc(n.sort_bytes);
    if(storage_&&preserve){
        const size_t old=size_t(view_.host_capacity)*n.heads*272;
        if(n.host_key!=view_.host_key)check_cuda(cudaMemcpyAsync(n.host_key,view_.host_key,old,cudaMemcpyDefault,stream),"preserve authoritative RAM K");
        if(n.host_value!=view_.host_value)check_cuda(cudaMemcpyAsync(n.host_value,view_.host_value,old,cudaMemcpyDefault,stream),"preserve authoritative RAM V");
        if(n.key!=view_.host_key)check_cuda(cudaMemcpyAsync(n.key,view_.host_key,std::min(size_t(view_.host_capacity)*n.heads*272,resident_bytes()/2),cudaMemcpyDefault,stream),"restore identity resident K");
        if(n.value!=view_.host_value)check_cuda(cudaMemcpyAsync(n.value,view_.host_value,std::min(size_t(view_.host_capacity)*n.heads*272,resident_bytes()/2),cudaMemcpyDefault,stream),"restore identity resident V");
        for(int h=0;h<n.heads;++h)check_cuda(cudaMemcpyAsync(n.bounds+size_t(h)*n.pages*512,
            view_.bounds+size_t(h)*view_.pages*512,size_t(view_.pages)*512*sizeof(__half),cudaMemcpyDeviceToDevice,stream),"preserve page summaries");
    }
    ram_reset(n,stream);check_cuda(cudaStreamSynchronize(stream),"RAM KV new owner ready");
    // Ownership transfer commits only after all new allocations and GPU reset
    // succeed. Exactly one RamKvCache continues to own the resident directory.
    if(reuse)owned->fixed=std::move(storage_->fixed);
    storage_=std::move(owned);view_=n;
}
void RamKvCache::reset(cudaStream_t stream){ram_reset(view_,stream);}
void RamKvCache::resume(int rows,cudaStream_t stream){
    if(rows<1||rows>view_.host_capacity)throw std::invalid_argument("invalid RAM KV resume extent");
    ram_reset(view_,stream);
    if(view_.key==view_.host_key)return;
    // Host rows stay authoritative across requests; only the resident rows a
    // decode pass may have remapped need their identity order back.
    const int copy=std::min(rows,view_.resident);
    const size_t row=size_t(view_.heads)*272;
    check_cuda(cudaMemcpyAsync(view_.key,view_.host_key,size_t(copy)*row,cudaMemcpyDefault,stream),"RAM KV resume K rows");
    check_cuda(cudaMemcpyAsync(view_.value,view_.host_value,size_t(copy)*row,cudaMemcpyDefault,stream),"RAM KV resume V rows");
    if(rows>view_.resident)ram_invalidate(view_,stream);
}
RamKvCounters RamKvCache::counters(cudaStream_t stream)const{
    RamKvCounters c;check_cuda(cudaMemcpyAsync(&c,view_.counters,sizeof(c),cudaMemcpyDeviceToHost,stream),"RAM KV bounded counters");
    check_cuda(cudaStreamSynchronize(stream),"RAM KV counter completion");return c;
}
}
