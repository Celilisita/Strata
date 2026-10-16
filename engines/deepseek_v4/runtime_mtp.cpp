#include "runtime.hpp"
#include "mtp_drafter.hpp"
#include "mtp_sidecar.hpp"
#include "device_cache.hpp"
#include "native_tg_graph.hpp"
#include "llama-context.h"
#include "llama-model.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
namespace strata::deepseek_v4 {
void mtp_invalidate_target_graph(Runtime& runtime) {
    runtime.context()->strata_invalidate_graph_for_fusion_switch();
}
void Runtime::mtp_begin(const std::vector<llama_token>& input) {
    if(!mtp_enabled()||input.size()<2||position_!=llama_pos(input.size()))throw std::runtime_error("MTP request initialization invalid");
    const auto& tail=context_->strata_nextn_tail();const int rows=int(tail.size()/16384);
    if(rows<2||rows>128)throw std::runtime_error("MTP target hidden tail absent");
    if(!std::all_of(tail.begin(),tail.end(),[](float f){return std::isfinite(f);}))throw std::runtime_error("MTP target hidden nonfinite");
    std::copy_n(tail.data()+(rows-1)*16384,16384,mtp_hidden_.data());
    // Every request re-enters TG, including reuse after the warmup PP phase.
    prepare_device_expert_cache(context_,1);
    const char* cached_only=std::getenv("STRATA_DSV4_MTP_CACHE_ONLY");
    const bool skip_misses=cached_only&&!std::strcmp(cached_only,"1");
    if(mtp_drafter_&&mtp_cached_only_!=skip_misses)mtp_drafter_.reset();
    if(!mtp_drafter_)mtp_drafter_=std::make_unique<MtpDrafter>(*mtp_weights_,device_expert_backend(context_),model_->tok_embd,model_->output,context_->strata_cpu_service(),1,context_);
    mtp_cached_only_=skip_misses;
    mtp_drafter_->reset();std::vector<int32_t> positions(rows-1);for(int i=0;i<rows-1;++i)positions[i]=position_-rows+i+1;
    mtp_drafter_->catch_up(tail.data(),input.data()+input.size()-rows+1,positions.data(),rows-1);
    if(mtp_weights_->dense_device_bytes()+mtp_drafter_->device_scratch_bytes()>(512ull<<20))throw std::runtime_error("MTP dense plus scratch residency exceeds0.5GiB");
    context_->strata_capture_nextn_tail(2);
}
llama_token Runtime::mtp_draft(llama_token next) {
    if(!mtp_drafter_)throw std::runtime_error("MTP drafter not initialized");
    const int32_t position=position_;return mtp_drafter_->forward(mtp_hidden_.data(),&next,&position).at(0);
}
MtpConfidence Runtime::mtp_confidence() const {
    return mtp_drafter_ ? mtp_drafter_->last_confidence() : MtpConfidence{};
}
MtpStageMetrics Runtime::mtp_stage_metrics() const {
    return mtp_drafter_ ? mtp_drafter_->stage_metrics() : MtpStageMetrics{};
}
uint64_t Runtime::cpu_worker_time_100ns() const {
    return context_->strata_cpu_service().worker_cpu_100ns();
}
void Runtime::mtp_set_async_input(bool enabled){
    if(!mtp_drafter_)throw std::runtime_error("MTP input phase precedes drafter initialization");
    mtp_drafter_->set_async_input(enabled);
}
void Runtime::mtp_accept_catchup(llama_token accepted_draft) {
    const auto& hidden=context_->strata_nextn_tail();if(hidden.size()!=32768)throw std::runtime_error("MTP verification hidden rows invalid");
    const int32_t position=position_-1;mtp_drafter_->catch_up(hidden.data(),&accepted_draft,&position,1);
    std::copy_n(hidden.data()+16384,16384,mtp_hidden_.data());
}
void Runtime::mtp_single_advance() {
    const auto& hidden=context_->strata_nextn_tail();
    if(hidden.size()!=16384)throw std::runtime_error("MTP single-position hidden row invalid");
    std::copy_n(hidden.data(),16384,mtp_hidden_.data());
}
void Runtime::rollback(llama_pos position) {
    if(position<0||position>=position_||position_-position>2||!llama_memory_seq_rm(llama_get_memory(context_),0,position,-1))throw std::runtime_error("MTP target rollback failed");
    const auto& hidden=context_->strata_nextn_tail();if(hidden.size()!=32768)throw std::runtime_error("MTP rollback hidden rows invalid");
    std::copy_n(hidden.data(),16384,mtp_hidden_.data());position_=position;
}
}
