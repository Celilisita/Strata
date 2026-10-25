#include "strata/runtime/capacity_storage.hpp"
#include <algorithm>
#include <cstdlib>
#include <limits>

namespace strata::runtime {
CapacityStorage::~CapacityStorage(){for(const auto &p:allocations_)cudaFree(p.first);}

// Retain the existing ABI/helper definition; allocation no longer queries or
// applies an external free-memory/reserve policy. There is no retry/fallback.
size_t CapacityStorage::reserve_bytes(){
    static const size_t reserve=[] {
        const char* text=std::getenv("STRATA_DENSE_RESERVE_MIB");
        if(!text)return size_t(4096)*1024*1024;
        size_t used=0;auto mib=std::stoull(text,&used);
        if(text[used]||*text=='-'||mib>1048576)throw std::invalid_argument("Dense reserve MiB");
        return size_t(mib)*1024*1024;
    }();
    return reserve;
}
void *CapacityStorage::allocate(size_t bytes){
    if(!bytes)throw std::invalid_argument("empty own device allocation");
    if(bytes>(std::numeric_limits<size_t>::max)()-bytes_)
        throw std::overflow_error("own Dense allocation byte ledger overflow");
    void *p=nullptr;const cudaError_t status=cudaMalloc(&p,bytes);
    if(status!=cudaSuccess)
        throw std::runtime_error("own Dense setup allocation: request_bytes="+std::to_string(bytes)+
            " owner_live_bytes="+std::to_string(bytes_)+" cuda_error="+cudaGetErrorString(status)+
            " cuda_code="+std::to_string(static_cast<int>(status)));
    try{allocations_.push_back({p,bytes});}catch(...){cudaFree(p);throw;}
    bytes_+=bytes;return p;
}
void CapacityStorage::release(void *p){
    auto found=std::find_if(allocations_.begin(),allocations_.end(),[&](const auto &a){return a.first==p;});
    if(found==allocations_.end())throw std::invalid_argument("foreign own allocation release");
    bytes_-=found->second;
    const auto status=cudaFree(p);
    if(status!=cudaSuccess)throw std::runtime_error(std::string("release idle own allocation: ")+cudaGetErrorString(status));allocations_.erase(found);
}
}
