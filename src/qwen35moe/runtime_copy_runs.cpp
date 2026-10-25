#include "strata/qwen35moe/runtime.hpp"
#include "strata/qwen35moe/expert_copy_runs.hpp"
#include "strata/qwen35moe/expert_copy_stage.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include <chrono>
namespace strata::qwen35moe {
uint64_t Runtime::prepare_expert_copy_runs(size_t il,int count,const std::string& reference,
                                         const std::vector<int32_t>& ids,Metrics& metrics){
    if((!prefill_config_.expert_copy_runs&&!prefill_config_.expert_copy_stage)||decode_phase_||count<1024||!reference.empty()||
       checkpoint_sink_||paired_session_||sessions_||mtp_enabled_||batch_.expert_stage.data()||
       expert_cache_.retention.enabled()||prefill_config_.mapped_pp||
       (expert_prefetch_&&expert_prefetch_->active())||
       !prefill_config_.grouped_gu_mmq||!prefill_config_.grouped_down_mmq||!batch_.down_mmq)return 0;
    const auto& s=expert_sources_.at(il);
    if(int(s.gate_exps->type)!=12||int(s.up_exps->type)!=12||
       (int(s.down_exps->type)!=12&&int(s.down_exps->type)!=13)||
       spec_.n_expert!=256||spec_.n_embd!=2048||spec_.n_ff_exp!=512)return 0;
    const size_t gu=kernels::native_mmvq_weight_bytes(12,2048,512);
    const size_t down=kernels::native_mmvq_weight_bytes(int(s.down_exps->type),512,2048);
    const PpPrefetchSource source{{gguf_.tensor_data(*s.gate_exps),gguf_.tensor_data(*s.up_exps),
        gguf_.tensor_data(*s.down_exps)},{gu,gu,down}};
    const auto start=std::chrono::steady_clock::now();ExpertCopyRunsResult result;
    try{
        if(prefill_config_.expert_copy_stage){
            if(count>4096||prefill_config_.prefix_demand||verify_history_||verify_state_||!prefill_scan_)return 0;
            // Full router has drained this stream. Dense phase buffers are dead;
            // independent norm/mixer/routes survive until the ordinary GU call.
            auto& scratch=batch_.phase_arena.data()?batch_.phase_arena:batch_.scan_activated;
            const auto staged=copy_expert_stage(prefetch_cache(),il,source,ids,host_readonly_,
                scratch.data(),scratch.bytes(),stream_);
            result=staged;metrics.expert_launches+=staged.scatter_launches;
        }else result=copy_expert_runs(prefetch_cache(),il,source,ids,host_readonly_,stream_);
    }
    catch(...){batch_failed_=true;throw;}
    metrics.expert_h2d+=result.copy.bytes;metrics.expert_h2d_calls+=result.copy.calls;
    metrics.expert_h2d_logical_copies+=result.copy.logical_copies;
    metrics.expert_h2d_registered_bytes+=result.copy.registered_bytes;
    metrics.expert_h2d_pageable_bytes+=result.copy.pageable_bytes;
    metrics.expert_h2d_2d_calls+=result.copy.rectangles;
    metrics.expert_cache_misses+=result.loaded_unique;
    metrics.expert_enqueue_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    return result.loaded_unique;
}
}
