#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
namespace strata::qwen35moe {
struct HostCopyCost {
    uint64_t bytes=0, calls=0, logical_copies=0;
    uint64_t registered_bytes=0, pageable_bytes=0;
    uint64_t rectangles=0;
    HostCopyCost& operator+=(const HostCopyCost& r) {
        bytes+=r.bytes; calls+=r.calls; logical_copies+=r.logical_copies;
        registered_bytes+=r.registered_bytes; pageable_bytes+=r.pageable_bytes;
        rectangles+=r.rectangles;
        return *this;
    }
};
// Read-only GGUF pages only: no packed copy, cache, GPU weights or ownership transfer.
class HostReadOnlyLease {
    struct Region { void* host; size_t bytes; void* alias; };
    std::vector<Region> spans_;
    size_t bytes_=0;
    uint64_t epoch_=0;
public:
    using Span=std::pair<uintptr_t,uintptr_t>;
    ~HostReadOnlyLease() { clear(); }
    HostReadOnlyLease()=default;
    HostReadOnlyLease(const HostReadOnlyLease&)=delete;
    HostReadOnlyLease& operator=(const HostReadOnlyLease&)=delete;
    void clear() noexcept {
        for(const auto& span:spans_) cudaHostUnregister(span.host);
        spans_.clear(); bytes_=0; ++epoch_;
    }
    void prepare(std::vector<Span> source,size_t bound,uint64_t(*available)(),bool gpu_aliases=false) {
        if(bytes_ || source.empty() || !bound || !available) throw std::invalid_argument("invalid GGUF page lease");
        std::sort(source.begin(),source.end());
        std::vector<Span> merged;
        for(const auto& span:source) {
            if(span.first>=span.second || span.first%4096 || span.second%4096)
                throw std::invalid_argument("GGUF lease page range");
            if(!merged.empty() && span.first<=merged.back().second)
                merged.back().second=(std::max)(merged.back().second,span.second);
            else merged.push_back(span);
        }
        size_t total=0;
        for(const auto& span:merged) {
            const size_t bytes=span.second-span.first;
            if(bytes>bound-total) throw std::invalid_argument("GGUF lease bound");
            total+=bytes;
        }
        if(total>bound || merged.size()>120) throw std::invalid_argument("GGUF lease bound");
        constexpr uint64_t reserve=768ull<<20;
        spans_.reserve(merged.size());
        ++epoch_;
        try {
            for(const auto& span:merged) {
                if(available()<reserve) throw std::runtime_error("GGUF page lease host reserve");
                auto* address=reinterpret_cast<void*>(span.first);const size_t bytes=span.second-span.first;
                const auto error=cudaHostRegister(address,bytes,cudaHostRegisterReadOnly);
                if(error!=cudaSuccess) throw std::runtime_error(std::string("GGUF page registration:")+cudaGetErrorString(error));
                spans_.push_back({address,bytes,nullptr});bytes_+=bytes;
                if(gpu_aliases) {
                    void* alias=nullptr;
                    const auto status=cudaHostGetDevicePointer(&alias,address,0);
                    if(status==cudaSuccess) spans_.back().alias=alias;
                    else if(status==cudaErrorInvalidValue || status==cudaErrorNotSupported)
                        (void)cudaGetLastError(); // Capability absent: caller keeps staged copies.
                    else throw std::runtime_error(std::string("GGUF registered alias:")+cudaGetErrorString(status));
                }
                if(available()<reserve) throw std::runtime_error("GGUF page lease host reserve after registration");
            }
        }catch(...) { clear(); throw; }
    }
    // Same CUDA12.8 runtime alias API already payload-qualified in the discarded
    // DMA proposal; no batch-copy driver API or DMA enable flag is involved.
    bool gpu_alias(const void* source,size_t bytes,const void*& alias) const {
        alias=nullptr;
        if(!source || !bytes) return false;
        const auto address=reinterpret_cast<uintptr_t>(source);
        for(const auto& span:spans_) {
            const auto begin=reinterpret_cast<uintptr_t>(span.host);
            if(address<begin) break;
            const size_t offset=size_t(address-begin);
            if(offset>span.bytes || bytes>span.bytes-offset || !span.alias) continue;
            const auto base=reinterpret_cast<uintptr_t>(span.alias);
            if(base+offset<base) return false;
            alias=reinterpret_cast<const void*>(base+offset);return true;
        }
        return false;
    }
    bool contains(const void* source,size_t bytes) const {
        if(!source || !bytes) return false;
        const auto address=reinterpret_cast<uintptr_t>(source);
        for(const auto& span:spans_) {
            const auto begin=reinterpret_cast<uintptr_t>(span.host);
            if(address<begin) break;
            if(address-begin<=span.bytes && bytes<=span.bytes-size_t(address-begin)) return true;
        }
        return false;
    }
    // A CUDA H2D source range must not mix registered and unregistered pages.
    // Emit the minimal contiguous segments at this lease's merged boundaries.
    // No allocation, synchronization, staging, new stream, or ownership change.
    HostCopyCost copy_to_device(void* destination,const void* source,size_t bytes,
                                cudaStream_t stream,const char* operation,
                                int64_t key=-1,int part=-1) const {
        const auto begin=reinterpret_cast<uintptr_t>(source);
        const auto target=reinterpret_cast<uintptr_t>(destination);
        const auto limit=(std::numeric_limits<uintptr_t>::max)();
        if(!source||!destination||!bytes||bytes>limit-begin||bytes>limit-target)
            throw std::invalid_argument("invalid host H2D copy extent");
        const auto end=begin+bytes;
        auto it=std::lower_bound(spans_.begin(),spans_.end(),begin,
            [](const Region& r,uintptr_t address) {
                return reinterpret_cast<uintptr_t>(r.host)+r.bytes<=address;
            });
        HostCopyCost cost; cost.logical_copies=1;
        for(auto cursor=begin;cursor<end;) {
            bool registered=false; uintptr_t stop=end;
            if(it!=spans_.end()) {
                const auto first=reinterpret_cast<uintptr_t>(it->host);
                const auto last=first+it->bytes;
                registered=cursor>=first;
                stop=(std::min)(end,registered?last:first);
            }
            const size_t count=size_t(stop-cursor),offset=size_t(cursor-begin);
            if(!count) throw std::runtime_error("non-progressing host H2D boundary");
            const auto status=cudaMemcpyAsync(reinterpret_cast<void*>(target+offset),
                reinterpret_cast<const void*>(cursor),count,cudaMemcpyHostToDevice,stream);
            if(status!=cudaSuccess)
                throw std::runtime_error(std::string(operation?operation:"host H2D")+
                    " key="+std::to_string(key)+" part="+std::to_string(part)+
                    " segment="+std::to_string(cost.calls)+" offset="+std::to_string(offset)+
                    " bytes="+std::to_string(count)+" src="+std::to_string(cursor)+
                    " registered="+std::to_string(int(registered))+" CUDA: "+cudaGetErrorString(status));
            cost.bytes+=count; ++cost.calls;
            (registered?cost.registered_bytes:cost.pageable_bytes)+=count;
            cursor=stop;
            if(registered&&it!=spans_.end()&&cursor==reinterpret_cast<uintptr_t>(it->host)+it->bytes)++it;
        }
        return cost;
    }
    // Source rows are consecutive expert payloads, destination rows are slots
    // in ONE existing allocation. Boundary rows retain the exact 1D splitter.
    HostCopyCost copy_to_device_rows(void* destination,size_t pitch,const void* source,
                                    size_t width,size_t rows,cudaStream_t stream,
                                    const char* operation,int64_t first_key,int part) const {
        const auto begin=reinterpret_cast<uintptr_t>(source);
        const auto target=reinterpret_cast<uintptr_t>(destination);
        const auto limit=(std::numeric_limits<uintptr_t>::max)();
        if(!source||!destination||!width||!rows||rows>64||pitch<width||first_key<0||
           first_key>(std::numeric_limits<int64_t>::max)()-int64_t(rows)||
           rows>limit/width||width*rows>limit-begin||width>limit-target||
           rows-1>(limit-target-width)/pitch)
            throw std::invalid_argument("invalid expert H2D rectangle extent");
        const auto end=begin+width*rows;HostCopyCost total;
        for(size_t row=0;row<rows;){
            const auto cursor=begin+row*width;
            const auto it=std::lower_bound(spans_.begin(),spans_.end(),cursor,
                [](const Region& r,uintptr_t address){return reinterpret_cast<uintptr_t>(r.host)+r.bytes<=address;});
            bool registered=false;auto stop=end;
            if(it!=spans_.end()){
                const auto first=reinterpret_cast<uintptr_t>(it->host);
                registered=cursor>=first;stop=(std::min)(end,registered?first+it->bytes:first);
            }
            const size_t whole=size_t(stop-cursor)/width;
            if(whole<2){
                total+=copy_to_device(reinterpret_cast<void*>(target+row*pitch),
                    reinterpret_cast<const void*>(cursor),width,stream,operation,first_key+int64_t(row),part);
                ++row;continue;
            }
            const auto status=cudaMemcpy2DAsync(reinterpret_cast<void*>(target+row*pitch),pitch,
                reinterpret_cast<const void*>(cursor),width,width,whole,cudaMemcpyHostToDevice,stream);
            if(status!=cudaSuccess)
                throw std::runtime_error(std::string(operation?operation:"expert H2D rectangle")+
                    " key="+std::to_string(first_key+int64_t(row))+" part="+std::to_string(part)+
                    " rows="+std::to_string(whole)+" width="+std::to_string(width)+
                    " registered="+std::to_string(int(registered))+" CUDA: "+cudaGetErrorString(status));
            total.bytes+=whole*width;++total.calls;++total.rectangles;total.logical_copies+=whole;
            (registered?total.registered_bytes:total.pageable_bytes)+=whole*width;row+=whole;
        }
        return total;
    }
    size_t bytes() const { return bytes_; }
    uint64_t epoch() const { return epoch_; }
};
}
