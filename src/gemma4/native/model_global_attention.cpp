#include "strata/gemma4/native/model.hpp"
#include "attention_global_t2.hpp"

namespace strata::gemma4::native {
void Model::select_global_shared_long(bool enabled){
    if(global_shared_long_==enabled)return;
    // This is an explicit cold transition, never part of forward/capture.
    // The epoch invalidates external controller leases even without reset.
    primitive::check_cuda(cudaStreamSynchronize(stream_),"Gemma shared attention policy join");
    release_graph();
    ++reset_epoch_;
    global_shared_long_=enabled;
}

bool Model::enqueue_shared_global_attention(CacheView cache,int tokens){
    if(!global_shared_long_||!global_shared_long_request_||!split_decode_||
       tokens!=2||capacity_!=32768||spec_.heads!=32||cache.capacity!=32768||
       cache.window!=0||cache.heads!=4||cache.dim!=512)return false;
    const auto arena=ffn_->attention_workspace();
    constexpr size_t borrowed_bytes=128u*1024*1024;
    if(!arena.data||arena.bytes<borrowed_bytes)
        throw std::invalid_argument("Gemma shared attention decode arena lease");
    // Q/K/V projections have finished consuming the encoded activation arena.
    // This partial+merge completes before Out reuses it, on the same stream.
    // Draft also borrows this prefix; ordered Draft->Main enqueue prevents any
    // overlap in lifetime. The original arena allocation and owner are kept.
    const GlobalT2Lease lease{this,stream_,arena.data,borrowed_bytes};
    shared_global_t2_attention(q_,attended_,cache,position_,spec_.heads,tokens,lease,stream_);
    ++global_shared_t2_enqueues_;
    return true;
}
}
