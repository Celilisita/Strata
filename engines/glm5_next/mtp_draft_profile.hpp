#pragma once
// Request-owned host wall diagnostics only; no CUDA calls or alternate math.
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
namespace strata::glm5_next {
enum class MtpDraftStage : size_t {Total,Prepare,Advance,BatchMemory,GraphBuildAlloc,SetInputs,GraphCompute,Sync,Logits,Greedy,Count};
enum class MtpDraftCount : size_t {Draft,OneRow,TwoRow,Ubatches,RebuildAttempts,BuildCompleted,Reuse,Count};
inline constexpr std::array<const char*,size_t(MtpDraftStage::Count)> mtp_draft_stage_names{
    "draft_total","prepare_cache","session_advance","batch_memory","graph_build_alloc","set_inputs",
    "graph_compute_call","scheduler_sync","logits_download_finite","greedy"};
inline constexpr std::array<const char*,size_t(MtpDraftCount::Count)> mtp_draft_count_names{
    "draft_calls","one_row_advances","two_row_advances","compute_ubatches","graph_rebuild_attempts","graph_build_alloc_completed","graph_reuse_calls"};
struct MtpDraftHostSpan {uint64_t count=0,sum_ns=0,first_count=0,first_ns=0,max_ns=0;};
struct MtpDraftProfileSnapshot {
    bool enabled=false;uint64_t clock_order_errors=0,nesting_errors=0;
    std::array<MtpDraftHostSpan,size_t(MtpDraftStage::Count)> stages{};
    std::array<uint64_t,size_t(MtpDraftCount::Count)> counts{},first_counts{};
};
inline bool mtp_draft_profile_requested(){const char* v=std::getenv("STRATA_GLM53_MTP_DRAFT_PROFILE");
    if(!v||!std::strcmp(v,"0"))return false;if(!std::strcmp(v,"1"))return true;
    throw std::runtime_error("STRATA_GLM53_MTP_DRAFT_PROFILE must be 0 or 1");}
class MtpDraftProfile {
public:
    explicit MtpDraftProfile(bool enabled):enabled_(enabled){reset();}
    void reset(){value_={};value_.enabled=enabled_;active_=first_=false;}
    bool active()const{return active_;}
    static uint64_t now(){return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());}
    bool begin(){if(!enabled_)return false;if(active_){++value_.nesting_errors;return false;}
        first_=value_.counts[size_t(MtpDraftCount::Draft)]==0;active_=true;count(MtpDraftCount::Draft);return true;}
    void end(){active_=first_=false;}
    void count(MtpDraftCount kind){if(!active_)return;const auto i=size_t(kind);++value_.counts[i];if(first_)++value_.first_counts[i];}
    void record(MtpDraftStage kind,uint64_t begin,uint64_t end){if(!active_)return;if(end<begin){++value_.clock_order_errors;return;}
        auto& span=value_.stages[size_t(kind)];const auto elapsed=end-begin;++span.count;span.sum_ns+=elapsed;
        if(first_){++span.first_count;span.first_ns+=elapsed;}if(elapsed>span.max_ns)span.max_ns=elapsed;}
    MtpDraftProfileSnapshot snapshot()const{return value_;}
private:
    const bool enabled_;bool active_=false,first_=false;MtpDraftProfileSnapshot value_;
};
class MtpDraftCallScope {
public:
    explicit MtpDraftCallScope(MtpDraftProfile& owner):owner_(owner),active_(owner.begin()),begin_(active_?owner.now():0){}
    ~MtpDraftCallScope(){if(active_){owner_.record(MtpDraftStage::Total,begin_,owner_.now());owner_.end();}}
private:MtpDraftProfile& owner_;bool active_;uint64_t begin_;
};
class MtpDraftPhaseScope {
public:
    MtpDraftPhaseScope(MtpDraftProfile& owner,MtpDraftStage stage):owner_(owner),stage_(stage),active_(owner.active()),begin_(active_?owner.now():0){}
    ~MtpDraftPhaseScope(){finish();}
    void next(MtpDraftStage stage){if(active_){const auto at=owner_.now();owner_.record(stage_,begin_,at);begin_=at;}stage_=stage;}
    void finish(){if(active_){owner_.record(stage_,begin_,owner_.now());active_=false;}}
private:MtpDraftProfile& owner_;MtpDraftStage stage_;bool active_;uint64_t begin_;
};
}
