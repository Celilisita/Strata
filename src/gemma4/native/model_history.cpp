#include "strata/gemma4/native/model.hpp"
#include "strata/gemma4/native/pp_hh_only349_policy.hpp"
#include "strata/gemma4/native/global_gqa371_policy.hpp"
#include "attention_gqa_t2.hpp"
#include "strata/gemma4/native/attention_parallel_merge.hpp"
#include "strata/gemma4/native/attention_shared_kv_t2.hpp"
#include <stdexcept>
#include <limits>
namespace strata::gemma4::native {
size_t Model::kv_bytes()const{
    size_t bytes=kv_memory_?kv_memory_->bytes():0;
    for(const auto& history:history_owners_)if(history)bytes+=history->device_bytes();
    return bytes;
}
size_t Model::history_ram_bytes()const{
    size_t bytes=0;for(const auto& history:history_owners_)if(history)bytes+=history->host_bytes();
    return bytes;
}
size_t Model::history_resident_bytes()const{
    size_t bytes=0;for(const auto& history:history_owners_)if(history)bytes+=history->resident_bytes();
    return bytes;
}
const HistoryView* Model::global_history_view(size_t layer)const{
    if(layer>=spec_.layers.size())throw std::out_of_range("Gemma history layer");
    if(!global_history_enabled_||history_owners_.empty()||!history_owners_[layer])return nullptr;
    const auto& v=history_views_[layer];
    return capacity_>v.resident?&v:nullptr;
}
void Model::validate_global_attention_geometry()const{
    if(maximum_batch_!=8192||spec_.heads!=32||spec_.hidden!=5376||layers_.size()!=60)
        throw std::invalid_argument("Gemma323 fixed Model geometry");
    size_t global=0;
    for(const auto& layer:spec_.layers)if(!layer.sliding){
        if(layer.head_dim!=512||layer.kv_heads!=4)throw std::invalid_argument("Gemma323 globalD512/GQA8 inventory");++global;
    }
    if(global!=10)throw std::invalid_argument("Gemma323 ten actual global layers");
}
void Model::select_global_verify_merge(GlobalVerifyMerge mode){
    if(mode!=GlobalVerifyMerge::Parent174&&mode!=GlobalVerifyMerge::Parallel321&&mode!=GlobalVerifyMerge::SharedKv336&&mode!=tg371::Gqa371)
        throw std::invalid_argument("Gemma323 supported T2 merge selector");
    if(host_position_!=0||executable_)throw std::invalid_argument("Gemma323 T2 selector requires reset P0");
    if(mode!=GlobalVerifyMerge::Parent174)validate_global_attention_geometry();
    if(mode==tg371::Gqa371){
        const auto w=ffn_->attention_workspace();
        const size_t raw_bytes=size_t(maximum_batch_)*16*256*sizeof(float);
        const auto aligned=[](const void* p,size_t bytes,size_t alignment){const auto at=reinterpret_cast<std::uintptr_t>(p);
            return p&&bytes&&at%alignment==0&&bytes<=(std::numeric_limits<std::uintptr_t>::max)()-at;};
        if(raw_bytes!=gqa371::RawLimit||!aligned(k_,raw_bytes,16)||!aligned(v_,raw_bytes,16)||
           !aligned(w.data,w.bytes,16)||w.bytes<gqa371::layout(32768).arena_bytes||w.bytes>gqa371::ArenaLimit||
           !w.handle||w.blas_bytes!=gqa371::BlasBytes||!aligned(w.blas_data,w.blas_bytes,256))
            throw std::invalid_argument("Gemma371 fixed dead128MiB K/V and Ffnarena/BLAS lease");
        // The freshly-created Ffn handle is not stream-bound before firstPP.
        // Actual enqueue below uses core371's strict current-stream guard afterPP.
    }
    if(mode==global_verify_merge_)return;
    primitive::check_cuda(cudaStreamSynchronize(stream_),"Gemma323 T2 cold policy join");
    release_graph();++reset_epoch_;global_verify_merge_=mode;global_verify321_enqueues_=0;global_verify336_enqueues_=0;
}
void Model::select_pp_key_tile(PpKeyTile mode){
    if(mode!=PpKeyTile::Keys512&&!pp349::is_k1024(mode))
        throw std::invalid_argument("Gemma323 supported PP key tile");
    if(host_position_!=0||executable_)throw std::invalid_argument("Gemma323 PP selector requires reset P0");
    if(pp349::is_k1024(mode))validate_global_attention_geometry();
    if(mode==pp_key_tile_)return;
    primitive::check_cuda(cudaStreamSynchronize(stream_),"Gemma323 PP cold policy join");
    release_graph();++reset_epoch_;pp_key_tile_=mode;pp_k1024_enqueues_=0;pp_three_product333_enqueues_=0;
}
bool Model::enqueue_parallel_global_verify(CacheView c,int tokens){
    if(!global_verify321_request_||global_verify_merge_==GlobalVerifyMerge::Parent174||tokens!=2||c.window!=0||capacity_>32768)return false;
    if(c.capacity!=capacity_||c.dim!=512||c.heads!=4||spec_.heads!=32)
        throw std::invalid_argument("Gemma323 exact GPU global T2 cache");
    if(global_verify_merge_==tg371::Gqa371){
        if(!tg371::supported_request_capacity(capacity_))
            throw std::invalid_argument("Gemma371 exact8K/30K T2 request capacity");
        // Raw K/V have been appended to ownedQ8 cache. These full allocations
        // and the Ffn arena are dead until later Out/FFN/nextlayer work on stream_.
        const gqa371::Binding live{this,reset_epoch_,stream_};
        gqa371::attention(q_,attended_,c,position_,32,2,ffn_->attention_workspace(),
            {k_,gqa371::RawLimit,v_,gqa371::RawLimit,live},live,invalid_);
        // Exclusive selected-optimized-T2 backing slot; use typed leaf helpers.
        ++global_verify321_enqueues_;return true;
    }
    const merge321::Binding live{this,reset_epoch_,stream_};
    if(global_verify_merge_==GlobalVerifyMerge::SharedKv336){
        shared336::split_attention_shared_kv_t2(q_,attended_,c,position_,32,2,
            {ffn_->attention_workspace(),live},live);++global_verify336_enqueues_;
    }else{
        merge321::split_attention_parallel_merge(q_,attended_,c,position_,32,2,
            {ffn_->attention_workspace(),live},live,merge321::Mode::Parallel);++global_verify321_enqueues_;
    }
    return true;
}
}
