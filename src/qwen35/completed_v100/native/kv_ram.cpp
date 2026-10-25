#include "strata/qwen35/completed_v100/native/kv_ram.hpp"
#include "strata/qwen35/completed_v100/native/weights.hpp"
#include <algorithm>
namespace strata::completed_v100::qwen35::native {
struct RamKvCache::Allocations {
    DeviceStorage gpu;
    void *key=nullptr,*value=nullptr;
    ~Allocations(){if(key)cudaFreeHost(key);if(value)cudaFreeHost(value);}
};
RamKvCache::RamKvCache(int capacity,int resident,int heads,cudaStream_t stream,int selected_tokens,int keep_queries){
    if(capacity<1||capacity>262144||resident!=32768||heads<1)throw std::invalid_argument("invalid RAM KV lease");
    if(selected_tokens<1024||selected_tokens>resident||selected_tokens%256)throw std::invalid_argument("invalid history selection budget");
    if(keep_queries<0||(keep_queries+1)*selected_tokens>resident)throw std::invalid_argument("reuse horizon exceeds physical page budget");
    view_.keep_queries=keep_queries;
    view_.selected_tokens=selected_tokens;view_.selected_slots=selected_tokens/RamPage;
    view_.resident=resident;view_.heads=heads;view_.slots=resident/RamPage;reserve(capacity,stream);
}
RamKvCache::~RamKvCache()=default;
void RamKvCache::reserve(int required,cudaStream_t stream){
    if(required<1||required>262144)throw std::out_of_range("RAM KV logical capacity");
    if(required<=view_.host_capacity)return;
    cudaStreamCaptureStatus status{};check_cuda(cudaStreamIsCapturing(stream,&status),"RAM KV growth capture check");
    if(status!=cudaStreamCaptureStatusNone)throw std::logic_error("reserve RAM KV before graph capture");
    check_cuda(cudaStreamSynchronize(stream),"RAM KV cold growth lifetime");
    const int next=std::min(262144,((required+8191)/8192)*8192);
    auto owned=std::make_unique<Allocations>();RamKvView n{};
    n.keep_queries=view_.keep_queries;
    n.selected_tokens=view_.selected_tokens;n.selected_slots=view_.selected_slots;
    n.heads=view_.heads;n.resident=view_.resident;n.slots=view_.slots;n.host_capacity=next;n.pages=next/RamPage;
    const size_t bytes=size_t(next)*n.heads*272;
    check_cuda(cudaHostAlloc(&owned->key,bytes,cudaHostAllocMapped),"authoritative pinned RAM K");
    check_cuda(cudaHostAlloc(&owned->value,bytes,cudaHostAllocMapped),"authoritative pinned RAM V");
    check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void **>(&n.host_key),owned->key,0),"mapped RAM K pointer");
    check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void **>(&n.host_value),owned->value,0),"mapped RAM V pointer");
    auto alloc=[&](size_t b){return owned->gpu.allocate(b);};
    auto ints=[&](size_t count){return static_cast<int *>(alloc(count*4));};
    n.key=static_cast<unsigned char *>(alloc(resident_bytes()/2));n.value=static_cast<unsigned char *>(alloc(resident_bytes()/2));
    const size_t pages=size_t(n.heads)*n.pages,slots=size_t(n.heads)*n.slots;
    n.bounds=static_cast<__half *>(alloc(pages*256*2*sizeof(__half)));
    n.scores=static_cast<float *>(alloc(pages*4));n.sorted_scores=static_cast<float *>(alloc(pages*4));
    n.page_ids=ints(pages);n.sorted_ids=ints(pages);n.offsets=ints(n.heads+1);
    n.page_slot=ints(pages);n.slot_page=ints(slots);n.selected_slot=ints(slots);n.miss=ints(slots);
    n.marked=ints(slots);n.miss_rank=ints(slots);n.free_slot=ints(slots);
    if(n.keep_queries){n.last_use=ints(slots);n.epoch=ints(n.heads);}
    n.counters=static_cast<RamKvCounters *>(alloc(sizeof(RamKvCounters)));
    std::vector<int> offsets(n.heads+1);for(int h=0;h<=n.heads;++h)offsets[h]=h*n.pages;
    check_cuda(cudaMemcpyAsync(n.offsets,offsets.data(),offsets.size()*4,cudaMemcpyHostToDevice,stream),"RAM indexer segment offsets");
    n.sort_bytes=ram_sort_workspace(n);n.sort_work=alloc(n.sort_bytes);
    if(storage_){
        const size_t old=size_t(view_.host_capacity)*n.heads*272;
        check_cuda(cudaMemcpyAsync(n.host_key,view_.host_key,old,cudaMemcpyDefault,stream),"preserve authoritative RAM K");
        check_cuda(cudaMemcpyAsync(n.host_value,view_.host_value,old,cudaMemcpyDefault,stream),"preserve authoritative RAM V");
        check_cuda(cudaMemcpyAsync(n.key,view_.host_key,std::min(size_t(view_.host_capacity)*n.heads*272,resident_bytes()/2),cudaMemcpyDefault,stream),"restore identity resident K");
        check_cuda(cudaMemcpyAsync(n.value,view_.host_value,std::min(size_t(view_.host_capacity)*n.heads*272,resident_bytes()/2),cudaMemcpyDefault,stream),"restore identity resident V");
        for(int h=0;h<n.heads;++h)check_cuda(cudaMemcpyAsync(n.bounds+size_t(h)*n.pages*512,
            view_.bounds+size_t(h)*view_.pages*512,size_t(view_.pages)*512*sizeof(__half),cudaMemcpyDeviceToDevice,stream),"preserve page summaries");
    }
    ram_reset(n,stream);check_cuda(cudaStreamSynchronize(stream),"RAM KV new owner ready");
    storage_=std::move(owned);view_=n;
}
void RamKvCache::reset(cudaStream_t stream){ram_reset(view_,stream);}
RamKvCounters RamKvCache::counters(cudaStream_t stream)const{
    RamKvCounters c;check_cuda(cudaMemcpyAsync(&c,view_.counters,sizeof(c),cudaMemcpyDeviceToHost,stream),"RAM KV bounded counters");
    check_cuda(cudaStreamSynchronize(stream),"RAM KV counter completion");return c;
}
}
