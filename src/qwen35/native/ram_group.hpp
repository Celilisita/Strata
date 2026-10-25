#pragma once
#include "strata/qwen35/native/kv_ram.hpp"
#include "strata/qwen35/native/weights.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <exception>
#include <thread>
#include <vector>
namespace strata::qwen35::native {
struct RamGroupReserve {size_t pending=0,guard_bytes=0;int workers=0;bool budget_fallback=false;};
inline RamGroupReserve &ram_group_last(){static thread_local RamGroupReserve value;return value;}
inline RamGroupReserve reserve_ram_group(const std::vector<RamKvCache*>& caches,int required,cudaStream_t stream,bool parallel,bool device_rows=false){
    ram_group_last()={};
    if(required<1||required>262144)throw std::out_of_range("RAM group logical capacity");
    std::vector<RamKvCache*> seen,pending;
    for(auto *cache:caches){
        if(!cache||std::find(seen.begin(),seen.end(),cache)!=seen.end())throw std::invalid_argument("RAM group unique cache ownership");
        seen.push_back(cache);if(required>cache->view().host_capacity||device_rows!=cache->device_rows())pending.push_back(cache);
    }
    RamGroupReserve result;result.pending=pending.size();
    if(pending.empty())return ram_group_last()=result;
    cudaStreamCaptureStatus capture{};check_cuda(cudaStreamIsCapturing(stream,&capture),"RAM group capture lease");
    if(capture!=cudaStreamCaptureStatusNone)throw std::logic_error("reserve RAM group before graph capture");
    if(device_rows)parallel=false; // No pinned allocation to parallelize in the GPU authority path.
    if(parallel&&pending.size()>1){
        // Admit the simultaneous replacement of every pending device owner,
        // even when its fixed pool will actually be reused. Old owners remain
        // charged. The existing per-allocation reserve checks stay active.
        for(auto *cache:pending){
            auto v=cache->view();v.host_capacity=std::min(262144,((required+8191)/8192)*8192);v.pages=v.host_capacity/RamPage;
            const size_t pages=size_t(v.heads)*v.pages,slots=size_t(v.heads)*v.slots;
            result.guard_bytes+=cache->resident_bytes()+slots*(6+int(v.keep_queries!=0))*4+
                (v.keep_queries?size_t(v.heads)*4:0)+sizeof(RamKvCounters)+pages*(1024+20)+
                size_t(v.heads+1)*4+ram_sort_workspace(v);
        }
        size_t free=0,total=0;check_cuda(cudaMemGetInfo(&free,&total),"RAM group peak reserve admission");
        const size_t reserve=DeviceStorage::reserve_bytes();
        if(free<reserve||result.guard_bytes>free-reserve){parallel=false;result.budget_fallback=true;}
    }
    if(!parallel||pending.size()==1){
        for(auto *cache:pending){if(device_rows)cache->reserve_discarding_device(required,stream);else cache->reserve_discarding(required,stream);}
        result.workers=1;return ram_group_last()=result;
    }
    int device=0;check_cuda(cudaGetDevice(&device),"RAM group current device");
    const int workers=int(std::min(size_t(4),pending.size()));
    std::atomic<size_t> next{0};std::array<std::exception_ptr,4> failures{};
    auto worker=[&](int index){
        try{
            check_cuda(cudaSetDevice(device),"RAM group worker context");
            for(;;){const size_t job=next.fetch_add(1,std::memory_order_relaxed);
                if(job>=pending.size())break;
                // Each cache still owns its stream drain, new directory reset,
                // and allocation commit. No cache appears in two workers.
                pending[job]->reserve_discarding(required,stream);
            }
        }catch(...){failures[index]=std::current_exception();}
    };
    std::array<std::thread,3> threads;std::exception_ptr launch_failure;
    try{for(int i=1;i<workers;++i)threads[i-1]=std::thread(worker,i);worker(0);}
    catch(...){launch_failure=std::current_exception();}
    for(auto &thread:threads)if(thread.joinable())thread.join();
    if(launch_failure)std::rethrow_exception(launch_failure);
    for(const auto &failure:failures)if(failure)std::rethrow_exception(failure);
    result.workers=workers;return ram_group_last()=result;
}
}
