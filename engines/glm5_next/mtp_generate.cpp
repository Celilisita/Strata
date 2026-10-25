#include "mtp_generate.hpp"
#include "mtp_options.hpp"
#include "mtp_draft_profile.hpp"
#include "kernel_activity.hpp"
#include "mtp_match.hpp"
#include "../deepseek_v4/device_cache.hpp"
#include "llama-context.h"
#include <chrono>
#include <cmath>
#include <stdexcept>
namespace strata::glm5_next {
namespace {
using MtpClock=std::chrono::steady_clock;
double duration(MtpClock::time_point start){return std::chrono::duration<double,std::milli>(MtpClock::now()-start).count();}
void finite_logits(deepseek_v4::Runtime& runtime,int index) {
    const float* logits=llama_get_logits_ith(runtime.context(),index);
    if(!logits)throw std::runtime_error("MTP primary logits absent");
    bool any=false;
    for(int i=0;i<llama_vocab_n_tokens(runtime.vocab());++i)
        if(std::isnan(logits[i])||logits[i]==INFINITY)throw std::runtime_error("MTP primary logits nonfinite");
        else any=any||std::isfinite(logits[i]);
    if(!any)throw std::runtime_error("MTP primary logits have no finite candidate");
}
llama_token sample(llama_sampler* sampler,deepseek_v4::Runtime& runtime,int index,bool row1_prevalidated=false) {
    if(index!=1||!row1_prevalidated)finite_logits(runtime,index);
    // Only target sampling consumes sampler history/RNG. The greedy draft
    // has its own selection; sample-and-match preserves the target policy.
    return llama_sampler_sample(sampler,runtime.context(),index);
}
}
MtpGeneration generate_mtp(deepseek_v4::Runtime& runtime,const MtpTargetSample& target_sample,
    const std::vector<llama_token>& input,int limit,int cancel_after,const std::function<void(llama_token)>& emit,const std::function<void()>& on_warm,const std::function<bool()>& should_cancel) {
    if(!target_sample||input.empty()||limit<1||cancel_after<0||!emit||!on_warm)
        throw std::invalid_argument("invalid MTP generation request");
    if(!runtime.glm_mtp_enabled())throw std::runtime_error("GLM NextN payload is not loaded");
    MtpGeneration result;std::string error;
    uint64_t proposals=0,accepted=0,physical=0,discarded=0,catchups=0,produced=0;
    double warm_ms=0,draft_ms=0,verify_ms=0,catchup_ms=0,rollback_ms=0;
    auto cancellation=[&]{if(should_cancel&&should_cancel()){result.cancelled=true;return true;}return false;};
    if(cancellation()) {
        on_warm();result.metrics={{"enabled",true},{"cancelled_before_warm",true},{"proposals",0},{"accepted",0},
            {"physical_target_positions",0},{"actual_output_tokens",0},{"prefix_warm_ms",0.0}};return result;
    }
    auto start=MtpClock::now();runtime.glm_mtp_begin(input);warm_ms=duration(start);on_warm();
    llama_token pending=0;
    if(!cancellation())pending=target_sample(-1);
    auto output=[&](llama_token token) {
        if(cancellation())return false;
        emit(token);if(runtime.is_eog(token)){result.eos=true;return false;}
        ++produced;
        if(cancel_after&&produced==uint64_t(cancel_after)){result.cancelled=true;return false;}
        if(produced<uint64_t(limit))deepseek_v4::idle_device_expert_cache(runtime.context());
        return produced<uint64_t(limit);
    };
    // A pending sampled token has not yet entered primary KV. Every draft
    // step commits that real token exactly once to NextN's own state.
    if(!result.cancelled&&output(pending))while(produced<uint64_t(limit)&&!cancellation()) {
        start=MtpClock::now();const auto draft=runtime.glm_mtp_draft(pending);draft_ms+=duration(start);
        if(cancellation())break;
        const auto position=runtime.position();
        // Boundaries/EOS use one physical target position. No hidden forced
        // output, ignored EOS, or speculative position beyond the budget.
        if(runtime.is_eog(draft)||produced+1>=uint64_t(limit)||uint64_t(position)+2>runtime.options().n_ctx) {
            start=MtpClock::now();
            if(runtime.decode_tokens(&pending,1,position,true,error)!=0)throw std::runtime_error(error);
            ++physical;verify_ms+=duration(start);runtime.glm_mtp_single_advance();
            if(cancellation())break;
            pending=target_sample(-1);
            if(!output(pending))break;
        } else {
            const llama_token tokens[]{pending,draft};++proposals;
            start=MtpClock::now();
            kernel_activity_verify_begin(proposals);
            if(runtime.decode_tokens(tokens,2,position,true,error,true,2)!=0)throw std::runtime_error(error);
            kernel_activity_verify_end(proposals);
            physical+=2;verify_ms+=duration(start);
            if(cancellation())break;
            finite_logits(runtime,1);
            const auto match=mtp_sample_and_match(draft,pending,target_sample,output,[&]{
                start=MtpClock::now();runtime.glm_mtp_reject(position+1);rollback_ms+=duration(start);
            },[&](llama_token token){
                start=MtpClock::now();runtime.glm_mtp_catchup(token);catchup_ms+=duration(start);++catchups;
            },[&]{return !cancellation();});
            if(match.accepted)++accepted;else ++discarded;
            if(!match.continue_generation)break;
        }
    }
    result.metrics={{"enabled",true},{"proposals",proposals},{"accepted",accepted},{"physical_target_positions",physical},
        {"discarded_verification_positions",discarded},{"accepted_catchups",catchups},{"actual_output_tokens",produced},
        {"prefix_warm_ms",warm_ms},{"draft_ms",draft_ms},{"verify_ms",verify_ms},{"catchup_ms",catchup_ms},{"rollback_ms",rollback_ms},
        {"deferred_catchup",defer_mtp_catchup_requested()},
        {"unused_catchup_FFN_heads_deleted",defer_mtp_catchup_requested()?catchups:0},
        {"alignment","NextN(token_i,primary_postnorm_h_i-1);position0 uses zero hidden"},
        {"draft_policy","greedy"},{"target_policy","unchanged sampler, target sample-and-match"}};
    const auto profile=runtime.glm_mtp_draft_profile();
    if(profile.enabled){Json observed={{"schema","strata-mtp-draft-host-phases-v1"},{"enabled",true},{"clock","steady_clock_host_wall"},
        {"clock_order_errors",profile.clock_order_errors},{"nesting_errors",profile.nesting_errors},{"first_draft_counts",Json::object()},{"stages",Json::object()}};
        for(size_t i=0;i<profile.counts.size();++i){observed[mtp_draft_count_names[i]]=profile.counts[i];observed["first_draft_counts"][mtp_draft_count_names[i]]=profile.first_counts[i];}
        for(size_t i=0;i<profile.stages.size();++i){const auto& span=profile.stages[i];observed["stages"][mtp_draft_stage_names[i]]={{"count",span.count},{"sum_ns",span.sum_ns},
            {"first_count",span.first_count},{"first_ns",span.first_ns},{"max_ns",span.max_ns}};}
        observed["scope"]="Request reset; only glm_mtp_draft active, prefix warm excluded. First means all contribution in first actual draft; rest=sum-first/count-first_count. Host wall includes waits and scheduling, not GPU active time.";
        observed["overlap"]="draft_total includes prepare_cache+session_advance; session_advance includes inner compute phases and greedy. Compute stages are adjacent; batch_memory also covers loop/state tail. graph_build_alloc includes borrowed params/callback/reuse test even without rebuilding. graph_compute_call already synchronizes internally; scheduler_sync is the preserved extra call. max_ns is one recorded segment. Never add parent and child totals.";
        result.metrics["observed_host_draft_phases"]=std::move(observed);}
    return result;
}
MtpGeneration generate_mtp(deepseek_v4::Runtime& runtime,llama_sampler* sampler,
    const std::vector<llama_token>& input,int limit,int cancel_after,const std::function<void(llama_token)>& emit,
    const std::function<void()>& on_warm,const std::function<bool()>& should_cancel) {
    if(!sampler)throw std::invalid_argument("MTP target sampler absent");
    return generate_mtp(runtime,MtpTargetSample([&](int index){return sample(sampler,runtime,index,index==1);}),
                        input,limit,cancel_after,emit,on_warm,should_cancel);
}
}
