#include "strata/qwen35/native/mtp.hpp"
#include "mtp_graph.hpp"
#include "prefix_copy.hpp"
#include <algorithm>

namespace strata::qwen35::native {
size_t NativeMtp::recurrent_bytes()const {
    const auto &s=model_.spec_;
    const size_t per_layer=(s.ssm_state*s.ssm_state*s.ssm_rank+
        3*s.ssm_state*(2*s.ssm_group+s.ssm_rank))*4;
    size_t count=0;for(const auto &state:model_.states_)count+=state.state!=nullptr;
    return count*per_layer;
}
void NativeMtp::use_prefix_states(bool enabled) {
    if(graphs_||retain_prefix_||!prefix_states_.empty())
        throw std::logic_error("MTP prefix state selection must precede capture");
    if(!enabled)return;
    const auto &s=model_.spec_;
    if(s.ssm_state!=128||!model_.q8_cache())throw std::invalid_argument("unsupported retained-prefix state geometry");
    const size_t state=s.ssm_state*s.ssm_state*s.ssm_rank;
    const size_t history=3*s.ssm_state*(2*s.ssm_group+s.ssm_rank);
    size_t free=0,total=0;check_cuda(cudaMemGetInfo(&free,&total),"retained prefix cold reserve");
    if(free<size_t(proposal_capacity_)*recurrent_bytes()+size_t(4)*1024*1024*1024)throw std::runtime_error("retained prefix lacks4GiB reserve");
    prefix_states_.resize(model_.states_.size());
    for(size_t i=0;i<prefix_states_.size();++i)if(model_.states_[i].state)
        for(int slot=0;slot<proposal_capacity_;++slot){
            prefix_states_[i].state[slot]=static_cast<float *>(storage_.allocate(state*4));
            prefix_states_[i].history[slot]=static_cast<float *>(storage_.allocate(history*4));
        }
    retain_prefix_=true;
}
void NativeMtp::restore_prefix(int consumed,cudaStream_t stream) {
    if(!retain_prefix_||consumed<1||consumed>proposal_capacity_)throw std::invalid_argument("invalid retained MTP prefix restore");
    const auto &s=model_.spec_;
    const size_t state=s.ssm_state*s.ssm_state*s.ssm_rank*4;
    const size_t history=3*s.ssm_state*(2*s.ssm_group+s.ssm_rank)*4;
    const auto count=std::count_if(model_.states_.begin(),model_.states_.end(),[](const auto &r){return r.state!=nullptr;});
    const bool fused=graph_building_&&gpu_draft_chain_&&proposal_capacity_==2&&s.ssm_state==128&&s.ssm_group==16&&s.ssm_rank==48&&count==PrefixCopyLayers&&prefix_copy_enabled();
    if(fused){PrefixCopyBatch batch{};size_t at=0;
        for(size_t i=0;i<prefix_states_.size();++i)if(model_.states_[i].state){const auto &r=model_.states_[i];const auto &p=prefix_states_[i];
            batch.rows[at++]={r.state,p.state[consumed-1],r.history[r.current],p.history[consumed-1]};}
        prefix_copy_restore(batch,stream);
    }else for(size_t i=0;i<prefix_states_.size();++i)if(model_.states_[i].state){
        auto &r=model_.states_[i];const auto &prefix=prefix_states_[i];
        check_cuda(cudaMemcpyAsync(r.state,prefix.state[consumed-1],state,cudaMemcpyDeviceToDevice,stream),"restore already verified GDN prefix");
        check_cuda(cudaMemcpyAsync(r.history[r.current],prefix.history[consumed-1],history,cudaMemcpyDeviceToDevice,stream),"restore already verified convolution prefix");
    }
    counters_.D2D_bytes+=recurrent_bytes();counters_.prefix_restored_rows+=consumed;
}
void NativeMtp::reject_prefix(int consumed,std::vector<int> &predicted,int position,cudaStream_t stream) {
    if(consumed<1||consumed>proposal_capacity_)throw std::invalid_argument("rejected prefix extent");
    ++counters_.rejected_prefix[consumed-1];++counters_.rejections;
    if(retain_prefix_){
        if(graph_enabled())graphs_->restore_prefix(consumed,stream);else restore_prefix(consumed,stream);
        // Later draft rows remain outside the committed logical KV extent.
        // The next target batch overwrites them before any attention read.
        predicted.resize(size_t(consumed));
    }else{
        if(graph_enabled())graphs_->rewind(stream);else rewind(stream);
        predicted=target(device_ids_,consumed,position,stream);counters_.replayed_rows+=consumed;
    }
}
}
