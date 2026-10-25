#include "mtp_session.hpp"
#include "mtp_options.hpp"
#include "../deepseek_v4/runtime.hpp"
#include "../deepseek_v4/device_cache.hpp"
#include "llama-context.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
namespace strata::deepseek_v4 {
void Runtime::glm_mtp_request(bool enabled) {
    if(position_)throw std::runtime_error("GLM request mode must be selected after reset and before prefill");
    glm_collect_prefix_=enabled&&glm_mtp_;
    glm_prefix_hidden_.clear();glm_catchup_pending_=false;
}
void Runtime::glm_capture_hidden(size_t count,llama_pos start) {
    // Unmasked NextN capture exports every primary post-norm row, including
    // each external PP chunk. It does not alter the model's output head.
    context_->synchronize();glm_latest_hidden_.resize(count*4096);
    const float* values=context_->get_embeddings_nextn();
    if(!values)throw std::runtime_error("GLM primary hidden capture absent");
    std::copy_n(values,count*4096,glm_latest_hidden_.data());
    for(float value:glm_latest_hidden_)if(!std::isfinite(value))throw std::runtime_error("GLM primary hidden nonfinite");
    if(glm_collect_prefix_) {
        if(glm_prefix_hidden_.size()!=size_t(start)*4096)throw std::runtime_error("GLM full-prefix hidden positions are discontinuous");
        glm_prefix_hidden_.insert(glm_prefix_hidden_.end(),glm_latest_hidden_.begin(),glm_latest_hidden_.end());
    }
}
void Runtime::glm_mtp_begin(const std::vector<llama_token>& input) {
    if(!glm_mtp_||input.empty()||position_!=llama_pos(input.size())||glm_prefix_hidden_.size()!=input.size()*4096)
        throw std::runtime_error("GLM MTP full-prefix initialization invalid");
    context_->strata_retire_compute_buffers();
    if(!glm_session_)glm_session_=std::make_unique<strata::glm5_next::MtpSession>(*context_,options_.kv_type=="f16"?GGML_TYPE_F16:GGML_TYPE_Q8_0);
    glm_catchup_pending_=false;
    glm_session_->reset();glm_session_->warm(input,glm_prefix_hidden_);
    glm_session_->draft_profile().reset(); // Prefix warm never belongs to the draft profile.
    std::copy_n(glm_prefix_hidden_.data()+(input.size()-1)*4096,4096,glm_hidden_.data());
    glm_prefix_hidden_.clear();glm_prefix_hidden_.shrink_to_fit();glm_collect_prefix_=false;
    // The warm graph has retired; first draft enters TG after PP accounting.
}
llama_token Runtime::glm_mtp_draft(llama_token pending) {
    if(!glm_session_||glm_session_->position()!=position_-(glm_catchup_pending_?1:0))
        throw std::runtime_error("NextN/target position differs before draft");
    auto& profile=glm_session_->draft_profile();strata::glm5_next::MtpDraftCallScope total(profile);
    strata::glm5_next::MtpDraftPhaseScope phase(profile,strata::glm5_next::MtpDraftStage::Prepare);
    prepare_device_expert_cache(context_,1);
    phase.next(strata::glm5_next::MtpDraftStage::Advance);
    if(glm_catchup_pending_) {
        const llama_token pair[]{glm_catchup_token_,pending};
        const auto draft=glm_session_->advance_pair(pair,glm_catchup_hidden_.data(),position_-1);
        glm_catchup_pending_=false;
        return draft;
    }
    return glm_session_->advance(pending,glm_hidden_.data(),position_);
}
void Runtime::glm_mtp_catchup(llama_token accepted) {
    if(!glm_session_||glm_catchup_pending_||glm_latest_hidden_.size()!=8192||glm_session_->position()!=position_-1)
        throw std::runtime_error("NextN accepted catchup position/hidden differs");
    if(strata::glm5_next::defer_mtp_catchup_requested()) {
        std::copy_n(glm_latest_hidden_.data(),8192,glm_catchup_hidden_.data());
        glm_catchup_token_=accepted;glm_catchup_pending_=true;
    }else glm_session_->advance(accepted,glm_latest_hidden_.data(),position_-1);
    std::copy_n(glm_latest_hidden_.data()+4096,4096,glm_hidden_.data());
}
void Runtime::glm_mtp_reject(llama_pos after_pending) {
    if(!glm_session_||glm_catchup_pending_||position_-after_pending!=1||glm_session_->position()!=after_pending||glm_latest_hidden_.size()!=8192)
        throw std::runtime_error("GLM rejected draft rollback boundary differs");
    context_->synchronize();
    if(!llama_memory_seq_rm(llama_get_memory(context_),0,after_pending,-1))throw std::runtime_error("GLM KDA/MLA/indexer rollback failed");
    std::copy_n(glm_latest_hidden_.data(),4096,glm_hidden_.data());position_=after_pending;
}
void Runtime::glm_mtp_single_advance() {
    if(!glm_session_||glm_catchup_pending_||glm_latest_hidden_.size()!=4096||glm_session_->position()!=position_)
        throw std::runtime_error("GLM single final advance position differs");
    std::copy_n(glm_latest_hidden_.data(),4096,glm_hidden_.data());
}
strata::glm5_next::MtpDraftProfileSnapshot Runtime::glm_mtp_draft_profile() const {
    return glm_session_?glm_session_->draft_profile().snapshot():strata::glm5_next::MtpDraftProfileSnapshot{};
}
}
