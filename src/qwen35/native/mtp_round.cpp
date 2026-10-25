#include "mtp_state_owner.hpp"
#include "strata/qwen35/native/mtp.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include "mtp_graph.hpp"
#include "rmsconcat_policy.hpp"
#include "history_snapshot_policy.hpp"
#include "state_snapshot_copy.hpp"
#include "strata/qwen35/native/mtp_sampling.hpp"
#include "strata/qwen35/native/token_ops.hpp"
namespace strata::qwen35::native {
void NativeMtp::snapshot(cudaStream_t stream){
    if(state_owner_target_capture(&model_)){
        const auto &spec=model_.spec_;const size_t history=3*spec.ssm_state*(2*spec.ssm_group+spec.ssm_rank)*4;
        for(size_t i=0;i<saved_.size();++i)if(saved_[i].state){saved_[i].parity=model_.states_[i].current;
            validate_history_snapshot_banks(model_.states_[i].history[0],model_.states_[i].history[1],history);}
        return;
    }
    if(!graph_building_)normalize_state_owner();
    const auto &s=model_.spec_;const size_t state=s.ssm_state*s.ssm_state*s.ssm_rank*4;
    const size_t history=3*s.ssm_state*(2*s.ssm_group+s.ssm_rank)*4;
    const bool elide=history_snapshot_elision_enabled(graph_building_,gpu_draft_chain_,retain_prefix_,proposal_capacity_);
    const bool fused=elide&&graph_building_&&s.ssm_state==128&&s.ssm_group==16&&s.ssm_rank==48&&
        std::count_if(saved_.begin(),saved_.end(),[](const auto &v){return v.state!=nullptr;})==StateSnapshotCopyLayers&&state_snapshot_copy_enabled();
    StateSnapshotCopyBatch batch{};size_t at=0;
    for(size_t i=0;i<saved_.size();++i)if(saved_[i].state){
        const auto &r=model_.states_[i];auto &save=saved_[i];save.parity=r.current;
        if(fused)batch.rows[at++]={save.state,r.state};
        else check_cuda(cudaMemcpyAsync(save.state,r.state,state,cudaMemcpyDeviceToDevice,stream),"draft transaction GDN state");
        if(elide)validate_history_snapshot_banks(r.history[0],r.history[1],history);
        else check_cuda(cudaMemcpyAsync(save.history,r.history[r.current],history,cudaMemcpyDeviceToDevice,stream),"draft transaction conv history");
        counters_.D2D_bytes+=state+(elide?0:history);
    }
    if(fused)state_snapshot_copy(batch,stream);
}
void NativeMtp::rewind(cudaStream_t stream){
    if(!graph_building_)normalize_state_owner();
    const auto &s=model_.spec_;const size_t state=s.ssm_state*s.ssm_state*s.ssm_rank*4;
    const size_t history=3*s.ssm_state*(2*s.ssm_group+s.ssm_rank)*4;
    const bool elide=history_snapshot_elision_enabled(graph_building_,gpu_draft_chain_,retain_prefix_,proposal_capacity_);
    for(size_t i=0;i<saved_.size();++i)if(saved_[i].state){
        auto &r=model_.states_[i];const auto &save=saved_[i];r.current=save.parity;
        check_cuda(cudaMemcpyAsync(r.state,save.state,state,cudaMemcpyDeviceToDevice,stream),"restore committed GDN state");
        // The single target batch wrote only the opposite bank. Roll back by parity, never stale saved.history.
        if(elide)validate_history_snapshot_banks(r.history[0],r.history[1],history);
        else check_cuda(cudaMemcpyAsync(r.history[r.current],save.history,history,cudaMemcpyDeviceToDevice,stream),"restore committed conv history");
        counters_.D2D_bytes+=state+(elide?0:history);
    }
}
void NativeMtp::enqueue_target(const int *ids,int count,int position,cudaStream_t stream,bool retain_prefix){
    const int h=int(model_.spec_.n_embd);const float *hidden=model_.embed(ids,position,count,stream);
    model_.mtp_prefix_=retain_prefix?prefix_states_.data():nullptr;
    try{for(int i=0;i<int(model_.spec_.n_layers);++i)hidden=model_.layer(i,hidden,position,count,stream);}
    catch(...){model_.mtp_prefix_=nullptr;throw;}
    model_.mtp_prefix_=nullptr;
    if(count>1&&model_.batch_head_enabled()){
        model_.head_batch(hidden,count,verify_rows_,target_ids_,target_flags_,stream);
        if(sampling_)for(int t=0;t<count;++t)
            sampling_->pick(model_.batch_logits_+size_t(t)*model_.spec_.n_vocab,unsigned(t),device_ids_,target_ids_+t,target_flags_+t,stream,true,t+1<count);
        return;
    }
    for(int t=0;t<count;++t){
        model_.head(hidden+size_t(t)*h,stream);
        check_cuda(cudaMemcpyAsync(verify_rows_+size_t(t)*h,model_.final_norm(),size_t(h)*4,cudaMemcpyDeviceToDevice,stream),"retain verified target seed row");
        check_cuda(cudaMemcpyAsync(target_ids_+t,model_.picked_token(),4,cudaMemcpyDeviceToDevice,stream),"retain verified target ID");
        check_cuda(cudaMemcpyAsync(target_flags_+t,model_.invalid_logits(),4,cudaMemcpyDeviceToDevice,stream),"retain verified finite flag");
        if(sampling_)sampling_->pick(model_.logits(),unsigned(t),device_ids_,target_ids_+t,target_flags_+t,stream,true,t+1<count);
    }
}
std::vector<int> NativeMtp::target(const int *ids,int count,int position,cudaStream_t stream,bool preserve,bool chain_response){
    if(!graph_building_&&!(graph_enabled()&&preserve&&count>=2&&count<=3))normalize_state_owner();
    if(graph_enabled()&&!graph_building_)graphs_->target(count,preserve,stream);
    else enqueue_target(ids,count,position,stream,preserve&&retain_prefix_);
    if(preserve&&retain_prefix_)counters_.prefix_written_bytes+=size_t(count-1)*recurrent_bytes();
    const size_t response_bytes=chain_response?56:32;
    check_cuda(cudaMemcpyAsync(host_->p,target_ids_,response_bytes,cudaMemcpyDeviceToHost,stream),"MTP verified bulk response");
    check_cuda(cudaStreamSynchronize(stream),"verified batch completion");
    std::vector<int> predictions(host_->p->ids,host_->p->ids+count);
    counters_.target_rows+=count;counters_.D2H_bytes+=response_bytes;
    counters_.D2D_bytes+=count>1&&model_.batch_head_enabled()?model_.spec_.n_embd*4+model_.spec_.n_vocab*4+8:size_t(count)*(model_.spec_.n_embd*4+8);
    ++counters_.completion_joins;
    for(int t=0;t<count;++t)if(host_->p->flags[t]||predictions[t]<0||predictions[t]>=int(model_.spec_.n_vocab))
        throw std::runtime_error("nonfinite speculative target logits");
    return predictions;
}
void NativeMtp::enqueue_commit(int consumed,int position,cudaStream_t stream){
    const int h=int(model_.spec_.n_embd);
    check_cuda(cudaMemcpyAsync(catch_rows_,seed_,size_t(h)*4,cudaMemcpyDeviceToDevice,stream),"draft committed lookbehind seed");
    if(consumed>1)check_cuda(cudaMemcpyAsync(catch_rows_+h,verify_rows_,size_t(consumed-1)*h*4,cudaMemcpyDeviceToDevice,stream),"draft committed actual seed rows");
    catch_up(device_ids_,catch_rows_,consumed,position,stream);
    check_cuda(cudaMemcpyAsync(seed_,verify_rows_+size_t(consumed-1)*h,size_t(h)*4,cudaMemcpyDeviceToDevice,stream),"commit next target draft seed");
    if(graph_building_)mtp_positions(model_.positions_,graph_batch_.start,consumed-1,1,0,nullptr,
        int(model_.spec_.n_head),attention_->decode_rounding(),stream);
    else set_positions(model_.positions_,position+consumed-1,1,stream);
}
std::vector<int> NativeMtp::round(int known,int position,int proposals,cudaStream_t stream){
    if(proposals<0||proposals>proposal_capacity_||position<1||position+proposals>=model_.logical_context())
        throw std::out_of_range("native MTP requested transaction span");
    reserve_cache(position+proposals+1,stream);
    if(known<0||known>=int(model_.spec_.n_vocab))throw std::out_of_range("native MTP known ID");
    if(gpu_draft_chain_)return round_gpu_chain(known,position,proposals,stream);
    if(graph_enabled())graphs_->begin(position,stream);
    const int h=int(model_.spec_.n_embd);std::array<int,4> inputs{known,-1,-1,-1};
    for(int t=0;t<proposals;++t){
        check_cuda(cudaMemcpyAsync(device_ids_+t,inputs.data()+t,4,cudaMemcpyHostToDevice,stream),"draft conditioning actual ID");
        counters_.H2D_bytes+=4;
        inputs[t+1]=draft(device_ids_+t,t?draft_seed_:seed_,position+t,stream,unsigned(t));
    }
    check_cuda(cudaMemcpyAsync(device_ids_,inputs.data(),size_t(proposals+1)*4,cudaMemcpyHostToDevice,stream),"actual speculative verification IDs");
    counters_.H2D_bytes+=size_t(proposals+1)*4;
    if(proposals&&!graph_enabled())snapshot(stream);
    auto predicted=target(device_ids_,proposals+1,position,stream,proposals!=0);
    int accepted=0;
    while(accepted<proposals&&inputs[accepted+1]==predicted[accepted])++accepted;
    if(accepted<proposals){
        reject_prefix(accepted+1,predicted,position,stream);
    }
    const int consumed=accepted+1;
    if(graph_enabled()){graphs_->commit(consumed,stream);counters_.D2D_bytes+=rmsconcat_copy_bytes(h,consumed);}
    else enqueue_commit(consumed,position,stream);
    counters_.D2D_bytes+=size_t(consumed+1)*h*4;
    ++counters_.accepted_width[accepted];
    ++counters_.rounds;counters_.proposed+=proposals;counters_.accepted+=accepted;
    return predicted;
}
void NativeMtp::check_rewind(int position,cudaStream_t stream){
    // Real resident state, not synthetic. Mutate with a real target token and
    // prove that rejection restores each authoritative state/history/parity.
    if(graph_enabled())graphs_->begin(position,stream);
    snapshot(stream);int known=-1;
    check_cuda(cudaMemcpyAsync(&known,model_.picked_token(),4,cudaMemcpyDeviceToHost,stream),"rewind canary real token");
    check_cuda(cudaStreamSynchronize(stream),"rewind canary real pick");
    check_cuda(cudaMemcpyAsync(device_ids_,&known,4,cudaMemcpyHostToDevice,stream),"rewind canary target input");
    target(device_ids_,1,position,stream);
    if(graph_enabled())graphs_->rewind(stream);else rewind(stream);
    size_t checks=0;const auto &s=model_.spec_;
    const size_t sizes[]={s.ssm_state*s.ssm_state*s.ssm_rank,3*s.ssm_state*(2*s.ssm_group+s.ssm_rank)};
    for(size_t i=0;i<saved_.size();++i)if(saved_[i].state){
        const auto &r=model_.states_[i];const auto &save=saved_[i];
        if(r.current!=save.parity)throw std::runtime_error("restored history ownership mismatch");
        const float *a[]={r.state,r.history[r.current]},*b[]={save.state,save.history};
        for(int field=0;field<2;++field){
            std::vector<float> actual(sizes[field]),expected(sizes[field]);
            check_cuda(cudaMemcpyAsync(actual.data(),a[field],sizes[field]*4,cudaMemcpyDeviceToHost,stream),"restored actual state");
            check_cuda(cudaMemcpyAsync(expected.data(),b[field],sizes[field]*4,cudaMemcpyDeviceToHost,stream),"saved expected state");
            check_cuda(cudaStreamSynchronize(stream),"restored state proof");
            for(float value:actual)if(!std::isfinite(value))throw std::runtime_error("nonfinite restored state");
            if(std::memcmp(actual.data(),expected.data(),sizes[field]*4))throw std::runtime_error("rejected state payload not restored");
            ++checks;
        }
    }
    mtp_rewind_diagnostic().checked_fields=checks;
    if(!mtp_rewind_diagnostic().quiet)std::printf("{\"native_MTP_rewind_fields\":%zu,\"history_parity_restored\":true,\"state_NRMS\":0,\"passed\":true}\n",checks);
}
}
