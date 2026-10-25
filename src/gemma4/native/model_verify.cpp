#include "strata/gemma4/native/model.hpp"
#include "strata/gemma4/native/vector_single.hpp"
#include "strata/qwen35/native/ops.hpp"
#include <cmath>
#include <cstdint>
#include <utility>
namespace strata::gemma4::native {
namespace {
bool aligned(const void* p){return p&&reinterpret_cast<std::uintptr_t>(p)%4==0;}
bool overlap(const void* a,size_t na,const void* b,size_t nb){
    if(!na||!nb)return false;const auto x=reinterpret_cast<std::uintptr_t>(a),y=reinterpret_cast<std::uintptr_t>(b);
    if(na>UINTPTR_MAX-x||nb>UINTPTR_MAX-y)throw std::invalid_argument("Gemma main verify span overflow");
    return x<y+nb&&y<x+na;
}
}
MainScratchLease Model::draft_scratch()const{
    const auto arena=ffn_->attention_workspace();
    return {this,stream_,arena.data,arena.bytes};
}
MainDraftLease Model::draft_lease()const{
    if(!capacity_||host_position_<1||cache_.size()!=layers_.size())throw std::invalid_argument("Gemma draft lease requires processed prefix");
    const auto arena=ffn_->attention_workspace();
    return {this,reset_epoch_,stream_,features_,position_,next_,invalid_,arena.data,arena.bytes};
}
MainCacheLease Model::draft_cache(size_t layer)const{
    if(!capacity_||layer>=cache_.size())throw std::invalid_argument("Gemma draft cache lease index/reset");
    const auto& c=cache_[layer];return {this,reset_epoch_,c.key,c.value,c.capacity,c.heads,c.dim,c.window};
}
const float* Model::enqueue_scaled_embedding(const int* token){
    if(!aligned(token)||!capacity_)throw std::invalid_argument("Gemma draft embedding device ID/reset");
    embed(embedding_,token,draft_embedding_,1,std::sqrt(float(spec_.hidden)),invalid_,stream_);
    return draft_embedding_;
}
void Model::validate_verify_policy(VerifyPolicy policy,int rows)const{
    if(host_position_>capacity_-rows||!aligned(policy.remaining)||policy.eog_count<0||policy.eog_count>16||
       (policy.eog_count&&!aligned(policy.eog_ids)))throw std::invalid_argument("Gemma main verify context/policy lease");
    const size_t eog_bytes=size_t(policy.eog_count)*4;const auto arena=ffn_->attention_workspace();
    const std::pair<const void*,size_t> protected_spans[]{
        {verify_inputs_,size_t(rows)*4},{verify_picks_,size_t(rows)*4},{features_,size_t(spec_.hidden)*4},
        {verify_features_,size_t(rows)*spec_.hidden*4},{verify_logits_,size_t(rows)*spec_.vocabulary*4},
        {verify_partials_,size_t(rows)*((spec_.vocabulary+255)/256)*8},{verify_packet_,sizeof(VerifyPacket)},
        {position_,4},{next_,4},{draft_embedding_,size_t(spec_.hidden)*4},{arena.data,arena.bytes}};
    for(const auto& span:protected_spans)
        if(overlap(policy.remaining,4,span.first,span.second)||overlap(policy.eog_ids,eog_bytes,span.first,span.second))
            throw std::invalid_argument("Gemma main verify policy aliases owned state/workspace");
    if(overlap(policy.remaining,4,policy.eog_ids,eog_bytes))throw std::invalid_argument("Gemma main verify policy self-alias");
}
void Model::enqueue_verify(const int* first,const int* second,int drafts,
    VerifyPolicy policy,const unsigned* draft_invalid){
    if(!capacity_||host_position_<1||(drafts!=1&&drafts!=2)||maximum_batch_<drafts+1||!aligned(first)||(drafts==2&&!aligned(second)))
        throw std::invalid_argument("Gemma main verify draft/batch/reset contract");
    const int rows=drafts+1;
    validate_verify_policy(policy,rows);
    // Host cursor is deliberately not advanced here. Captured graphs consume
    // authoritative GPU position/pending/remaining on every external replay.
    detail::begin_main_verify(verify_packet_,draft_invalid,rows,stream_);
    form_verify_inputs(next_,first,second,verify_inputs_,rows,spec_.vocabulary,invalid_,stream_);
    float* hidden=forward_hidden(verify_inputs_,rows);
    primitive::rms_norm(hidden,final_norm_,verify_features_,spec_.hidden,rows,spec_.epsilon,stream_);
    project_head(verify_features_,verify_logits_,rows);
    pick_verify_rows(verify_logits_,verify_partials_,verify_picks_,rows,spec_.vocabulary,spec_.logit_cap,invalid_,stream_);
    commit_verify_rows(verify_inputs_,verify_picks_,verify_features_,features_,next_,position_,
        &verify_packet_->response,rows,spec_.hidden,spec_.vocabulary,capacity_,policy,invalid_,stream_);
}
VerifyResponse Model::finish_verify(){
    VerifyPacket packet{};
    primitive::check_cuda(cudaMemcpyAsync(&packet,verify_packet_,sizeof(packet),cudaMemcpyDeviceToHost,stream_),"Gemma main verify response");
    primitive::check_cuda(cudaStreamSynchronize(stream_),"Gemma main verify round completion");
    const auto& r=packet.response;
    if(packet.invalid||r.reason==VerifyStop::Invalid)throw std::runtime_error("Gemma invalid main verify round");
    const bool rows=packet.rows>=1&&packet.rows<=3;
    const bool count=rows&&r.count>=0&&r.count<=packet.rows;
    const bool prefix=count&&r.matched>=0&&r.matched<packet.rows&&
        (r.count?r.accepted==r.count-1&&r.matched>=r.accepted:r.accepted==0&&r.matched==0);
    const bool cursor=count&&r.position>=r.count&&r.position<=capacity_&&r.position-r.count==host_position_;
    const bool stop=(r.reason==VerifyStop::Continue||r.reason==VerifyStop::Eog||r.reason==VerifyStop::Budget)&&
        r.finished==(r.reason!=VerifyStop::Continue)&&(r.count||r.finished);
    if(!prefix||!cursor||!stop)throw std::runtime_error("Gemma stale/malformed main verify response");
    for(int i=0;i<3;++i)if(i<r.count?(r.ids[i]<0||r.ids[i]>=spec_.vocabulary):r.ids[i]!=0)
        throw std::runtime_error("Gemma main verify response token range/padding");
    host_position_=r.position;return r;
}
}
