#include "strata/qwen35/completed_v100/native/model.hpp"
#include <algorithm>
namespace strata::completed_v100::qwen35::native {
void DenseModel::reserve_cache(int required,cudaStream_t stream){
    if(required<1||required>limits_.context)throw std::out_of_range("own KV logical context exceeded");
    if(limits_.kv_ram_tokens){
        for(auto &c:caches_)if(c.ram){c.ram->reserve(required,stream);c.key=c.ram->view().key;c.value=c.ram->view().value;}return;
    }
    const int bound=limits_.kv_window_tokens?limits_.kv_window_tokens+limits_.max_tokens:limits_.context;
    if(required<=cache_capacity_||cache_capacity_==bound)return;
    cudaStreamCaptureStatus capture{};
    check_cuda(cudaStreamIsCapturing(stream,&capture),"own KV reservation capture check");
    if(capture!=cudaStreamCaptureStatusNone)throw std::runtime_error("reserve actual KV span before graph capture");
    check_cuda(cudaStreamSynchronize(stream),"own KV growth idle lease");
    const int next=(std::min)(bound,((required+8191)/8192)*8192);
    const size_t row=spec_.n_head_kv*(spec_.head_dim/32)*(limits_.kv_q8?34:18);
    // Replacement is cold and pairwise: no simultaneous second whole cache.
    for(auto &c:caches_)if(c.key){
        void *key=nullptr,*value=nullptr;
        try{
            key=storage_.allocate(size_t(next)*row);value=storage_.allocate(size_t(next)*row);
            check_cuda(cudaMemcpyAsync(key,c.key,size_t(cache_capacity_)*row,cudaMemcpyDeviceToDevice,stream),"grow own K rows");
            check_cuda(cudaMemcpyAsync(value,c.value,size_t(cache_capacity_)*row,cudaMemcpyDeviceToDevice,stream),"grow own V rows");
            check_cuda(cudaStreamSynchronize(stream),"own grown KV payload lease");
        }catch(...){cudaStreamSynchronize(stream);if(key)storage_.release(key);if(value)storage_.release(value);throw;}
        storage_.release(c.key);storage_.release(c.value);c.key=key;c.value=value;
    }
    cache_capacity_=next;
}
int DenseModel::cache_logical_capacity()const{
    if(!limits_.kv_ram_tokens)return cache_capacity_;
    for(const auto &c:caches_)if(c.ram)return c.ram->view().host_capacity;
    throw std::logic_error("missing RAM KV inventory");
}
size_t DenseModel::ram_bytes()const{size_t bytes=0;for(const auto &c:caches_)if(c.ram)bytes+=c.ram->host_bytes();return bytes;}
RamKvCounters DenseModel::ram_counters(cudaStream_t stream)const{
    RamKvCounters sum{};for(const auto &c:caches_)if(c.ram){const auto v=c.ram->counters(stream);
        sum.written_bytes+=v.written_bytes;sum.resident_write_bytes+=v.resident_write_bytes;sum.selected_pages+=v.selected_pages;
        sum.hits+=v.hits;sum.misses+=v.misses;sum.loaded_bytes+=v.loaded_bytes;sum.calls+=v.calls;}
    return sum;
}

}
