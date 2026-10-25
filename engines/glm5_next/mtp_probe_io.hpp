#pragma once
#include <string>
#include <vector>
// Diagnostic-only adapter to the existing hybrid memory state protocol.
#include "mtp_session.hpp"
#include "llama-io.h"
#include "llama-memory-hybrid-idx.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace strata::glm5_next {
inline uint64_t mtp_probe_state_hash(const std::vector<uint8_t>& bytes) {
    uint64_t hash=14695981039346656037ull;
    for(auto value:bytes){hash^=value;hash*=1099511628211ull;}return hash;
}
class MtpProbeWrite final:public llama_io_write_i {
public:
    MtpProbeWrite(size_t limit,std::vector<uint8_t>* destination=nullptr,
        const std::vector<uint8_t>* expected=nullptr):limit_(limit),destination_(destination),expected_(expected){}
    void write(const void* source,size_t size) override {
        bound(size);
        if(destination_&&size)std::memcpy(destination_->data()+used_,source,size);
        if(expected_&&size)compare(source,size,used_);
        used_+=size;
    }
    void write_tensor(ggml_tensor* tensor,size_t offset,size_t size) override {
        bound(size);tensor_bytes_+=size;
        if(destination_&&size)ggml_backend_tensor_get(tensor,destination_->data()+used_,offset,size);
        if(expected_) {
            scratch_.resize(std::min<size_t>(65536,size));
            for(size_t first=0;first<size;) {
                const auto count=std::min(scratch_.size(),size-first);
                ggml_backend_tensor_get(tensor,scratch_.data(),offset+first,count);
                compare(scratch_.data(),count,used_+first);first+=count;
            }
        }
        used_+=size;
    }
    size_t n_bytes() override{return used_;}
    size_t tensor_bytes() const{return tensor_bytes_;}
private:
    void bound(size_t size){if(size>limit_-used_)throw std::runtime_error("NextN stateIO exceeds bounded private state");}
    void compare(const void* source,size_t size,size_t at) {
        const auto* bytes=static_cast<const uint8_t*>(source);
        for(size_t i=0;i<size;++i)if(bytes[i]!=(*expected_)[at+i])
            throw std::runtime_error("NextN stateIO exact byte differs at "+std::to_string(at+i));
    }
    size_t limit_,used_=0,tensor_bytes_=0;std::vector<uint8_t>* destination_;const std::vector<uint8_t>* expected_;
    std::vector<uint8_t> scratch_;
};
class MtpProbeRead final:public llama_io_read_i {
public:
    explicit MtpProbeRead(const std::vector<uint8_t>& bytes):bytes_(bytes){}
    void read(void* target,size_t size) override {
        bound(size);if(size)std::memcpy(target,bytes_.data()+used_,size);used_+=size;
    }
    void read_tensor(ggml_tensor* tensor,size_t offset,size_t size) override {
        bound(size);if(size)ggml_backend_tensor_set(tensor,bytes_.data()+used_,offset,size);used_+=size;
    }
    size_t n_bytes() override{return used_;}
private:
    void bound(size_t size){if(size>bytes_.size()-used_)throw std::runtime_error("NextN stateIO input truncated");}
    const std::vector<uint8_t>& bytes_;size_t used_=0;
};
inline MtpProbeSnapshot save_mtp_probe_prefix(llama_memory_i& source,llama_pos next) {
    auto* memory=dynamic_cast<llama_memory_hybrid_idx*>(&source);
    if(!memory||next<1)throw std::runtime_error("NextN prefix snapshot requires private hybrid memory");
    const auto state=inspect_mtp_probe_memory(source,next);
    if(!state.layout_valid||!state.recurrent_empty||state.recurrent_storage_bytes||
       state.positions.size()!=size_t(next)||state.attn_min!=0||state.attn_max!=next-1)
        throw std::runtime_error("NextN prefix snapshot inventory differs");
    for(auto pos:memory->mem_idx_stale_get())if(pos!=llama_memory_hybrid_idx::POS_CLEAN)
        throw std::runtime_error("NextN prefix snapshot requires completed clean pooled state");
    MtpProbeSnapshot out;out.next=next;out.limit_bytes=state.memory_bytes+(uint64_t(1)<<20);
    if(out.limit_bytes>(uint64_t(32)<<20))throw std::runtime_error("NextN prefix snapshot exceeds32MiB host bound");
    MtpProbeWrite count(out.limit_bytes);memory->state_write(count,0,0);
    out.bytes.resize(count.n_bytes());MtpProbeWrite writer(out.bytes.size(),&out.bytes);
    memory->state_write(writer,0,0);
    if(writer.n_bytes()!=out.bytes.size())throw std::runtime_error("NextN prefix stateIO size changed");
    out.tensor_bytes=writer.tensor_bytes();out.fnv1a64=mtp_probe_state_hash(out.bytes);return out;
}
inline void restore_mtp_probe_prefix(llama_memory_i& source,const MtpProbeSnapshot& snapshot) {
    auto* memory=dynamic_cast<llama_memory_hybrid_idx*>(&source);
    if(!memory||snapshot.bytes.empty()||snapshot.bytes.size()>snapshot.limit_bytes||
       snapshot.limit_bytes>(uint64_t(32)<<20)||mtp_probe_state_hash(snapshot.bytes)!=snapshot.fnv1a64)
        throw std::runtime_error("NextN prefix snapshot integrity differs");
    memory->clear(true);MtpProbeRead reader(snapshot.bytes);memory->state_read(reader,0,0);
    if(reader.n_bytes()!=snapshot.bytes.size())throw std::runtime_error("NextN prefix restore left unread bytes");
    MtpProbeWrite verify(snapshot.bytes.size(),nullptr,&snapshot.bytes);memory->state_write(verify,0,0);
    if(verify.n_bytes()!=snapshot.bytes.size())throw std::runtime_error("NextN prefix roundtrip size differs");
    // state_read marks pools stale. Here every live key|gate|pooled byte has
    // been restored and verified; rebuild layout without recomputing the pools.
    memory->kpool_layout_update();memory->mem_idx_stale_clear();
}
}
