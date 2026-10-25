#include "strata/gemma4/native/sm120_model.hpp"
#include "strata/gemma4/native/sm120_budget.hpp"
#include "strata/qwen35/native/ops.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>
namespace strata::gemma4::native {
namespace {
bool aligned(const void* p){return p&&reinterpret_cast<std::uintptr_t>(p)%4==0;}
bool overlap(const void* a,std::size_t an,const void* b,std::size_t bn){
    const auto x=reinterpret_cast<std::uintptr_t>(a),y=reinterpret_cast<std::uintptr_t>(b);
    if(an>UINTPTR_MAX-x||bn>UINTPTR_MAX-y)throw std::invalid_argument("Gemma MTP span overflow");
    return an&&bn&&x<y+bn&&y<x+an;
}
}
void Sm120Model::prepare_mtp(int drafts){
    if((drafts!=1&&drafts!=2)||(batch_!=2048&&batch_!=8192)||host_position_||executable_||
       projections_->format()!=GemmaWeightFormat::Nvfp4||!projections_->tiny_compensated())
        throw std::invalid_argument("Gemma MTP cold native tiny/batch2048-or8192 policy");
    cudaStreamCaptureStatus capture{};
    primitive::check_cuda(cudaStreamIsCapturing(stream_,&capture),"Gemma MTP cold capture state");
    if(capture!=cudaStreamCaptureStatusNone)throw std::logic_error("Gemma MTP PREPARE precedes capture");
    const auto span=ffn_->mtp_decode_workspace_bytes();
    if(span<(std::size_t(128)<<20)||!ffn_->mtp_decode_workspace()||
       reinterpret_cast<std::uintptr_t>(ffn_->mtp_decode_workspace())%256)
        throw std::invalid_argument("Gemma MTP borrowed FFN decode span");
    // One optional allocation; it is absent when MTP0 is selected. The 256B
    // metadata tail keeps every row buffer and the state naturally aligned.
    if(!target_state_){
        admit_sm120_budget(detail::Sm120TargetStorageBytes,"MTP target rows");
        auto* block=static_cast<unsigned char*>(memory_.allocate(detail::Sm120TargetStorageBytes));
        target_features_=reinterpret_cast<float*>(block);block+=std::size_t(3)*5376*4;
        target_logits_=reinterpret_cast<float*>(block);block+=std::size_t(3)*262144*4;
        draft_embedding_=reinterpret_cast<float*>(block);block+=5376*4;
        target_state_=reinterpret_cast<detail::Sm120TargetState*>(block);
    }
    projections_->prepare({1,2,3});
    primitive::check_cuda(cudaMemsetAsync(target_state_,0,sizeof(*target_state_),stream_),"Gemma MTP cold state");
    primitive::check_cuda(cudaStreamSynchronize(stream_),"Gemma MTP cold storage lease");
    mtp_drafts_=drafts;target_rows_count_=0;++epoch_;
}
Sm120MtpScratch Sm120Model::mtp_scratch()const{
    if(!mtp_drafts_||!target_state_)throw std::invalid_argument("Gemma MTP scratch requires PREPARE");
    return {this,stream_,ffn_->mtp_decode_workspace(),ffn_->mtp_decode_workspace_bytes()};
}
Sm120MtpLease Sm120Model::mtp_lease()const{
    if(!capacity_||host_position_<1||caches_.size()!=layers_.size())
        throw std::invalid_argument("Gemma MTP lease requires processed prefix");
    const auto scratch=mtp_scratch();
    return {this,epoch_,stream_,features_,position_,input_,invalid_,scratch.decoded,scratch.decoded_bytes};
}
Sm120MtpCacheLease Sm120Model::mtp_cache(std::size_t layer)const{
    if(!mtp_drafts_||!capacity_||host_position_<1||layer>=caches_.size())
        throw std::invalid_argument("Gemma MTP cache lease/reset");
    const auto& c=caches_[layer];return {this,epoch_,c.key,c.value,c.capacity,c.heads,c.dim,c.window};
}
const float* Sm120Model::enqueue_scaled_embedding(const int* token){
    if(!mtp_drafts_||!capacity_||host_position_<1||!aligned(token))
        throw std::invalid_argument("Gemma MTP scaled embedding lease");
    embed(embedding_,token,draft_embedding_,1,std::sqrt(5376.f),invalid_,stream_);
    return draft_embedding_;
}
void Sm120Model::enqueue_target_rows(const int* inputs,int rows){
    if(!target_state_||!capacity_||host_position_<1||rows<1||rows>mtp_drafts_+1||
       !aligned(inputs)||host_position_>capacity_-rows)
        throw std::invalid_argument("Gemma MTP target row/context lease");
    // Retain all previous window entries even when two speculative suffix
    // inputs are later rejected. A cursor rollback cannot restore ring bytes.
    for(const auto& c:caches_)if(c.window&&c.capacity<std::min(capacity_,c.window+3))
        throw std::invalid_argument("Gemma MTP SWA ring must retain window+3");
    if(overlap(inputs,std::size_t(rows)*4,target_features_,std::size_t(3)*5376*4)||
       overlap(inputs,std::size_t(rows)*4,target_logits_,std::size_t(3)*262144*4))
        throw std::invalid_argument("Gemma MTP input aliases target output");
    detail::begin_sm120_target(target_state_,position_,rows,capacity_,invalid_,stream_);
    auto* hidden=forward_hidden(inputs,rows);
    primitive::rms_norm(hidden,final_norm_,target_features_,5376,rows,spec_.epsilon,stream_);
    project_head_rows(target_features_,target_logits_,rows);
    target_rows_count_=rows;
}
void Sm120Model::enqueue_commit_prefix(const int* count,const int* next,const unsigned* round_invalid){
    if(!target_state_||!target_rows_count_||!aligned(count)||!aligned(next)||!aligned(round_invalid))
        throw std::invalid_argument("Gemma MTP sampled prefix lease");
    const std::pair<const void*,std::size_t> writes[]{
        {features_,5376*4},{input_,4},{position_,4},{target_state_,sizeof(*target_state_)},{invalid_,4}};
    for(const auto& span:writes)if(overlap(count,4,span.first,span.second)||
       overlap(next,4,span.first,span.second)||overlap(round_invalid,4,span.first,span.second))
        throw std::invalid_argument("Gemma MTP decision aliases main commit state");
    detail::commit_sm120_target(target_state_,target_features_,features_,input_,position_,
        count,next,round_invalid,capacity_,invalid_,stream_);
}
void Sm120Model::finish_target_commit(int count,int position,std::uint64_t epoch){
    if(!mtp_drafts_||!capacity_||epoch!=epoch_||target_rows_count_<1||
       count<0||count>mtp_drafts_+1||position!=host_position_+count||position>capacity_)
        throw std::runtime_error("Gemma stale/malformed sampled target response");
    host_position_=position;
}
}
