#pragma once
#include <cstring>
namespace strata::qwen35moe {
uint64_t Runtime::prepare_gu_copy(size_t il,int count,const std::string& reference,
                                 const std::vector<int32_t>& ids,Metrics& metrics){
    // Full router already drained prior consumers. Prefix enqueue closes even
    // earlier, before it can evict an older chunk's last-layer directory.
    finish_gu_copy(metrics);
    auto& b=batch_;const auto& c=expert_cache_;
    if(!prefill_config_.gu_copy_wave||decode_phase_||(count!=1280&&count!=4096)||
       !prefill_config_.prefix_demand||prefill_config_.prefix_stage||gu_copy_prefix_layer_!=il||
       !reference.empty()||checkpoint_sink_||paired_session_||sessions_||mtp_enabled_||
       verify_state_||verify_history_||visual_table()||dense_profile_.active()||
       b.expert_stage.data()||c.retention.enabled()||prefill_config_.expert_copy_runs||
       prefill_config_.expert_copy_stage||prefill_config_.expert_prefetch||prefill_config_.mapped_pp||
       prefill_config_.down_input_fused||!b.down_mmq||!prefill_config_.grouped_gu_mmq||
       !prefill_config_.grouped_down_mmq||spec_.n_embd!=2048||spec_.n_ff_exp!=512||spec_.n_expert!=256||
       c.slots<size_t(resident_layers_)*256+512||il>=expert_sources_.size()||ids.size()!=size_t(count)*8)return 0;
    for(const auto* name:{"STRATA_QWEN35_GU_GROUP_INT8","STRATA_QWEN35_GU_GROUP_INT8_I32"}){
        const char* raw=std::getenv(name);if(!raw||std::strcmp(raw,"1"))return 0;}
    for(const auto setting:{std::make_pair("STRATA_BENCH_GROUP_ROWS","auto"),
        std::make_pair("STRATA_BENCH_GROUP_PANEL_TOKENS","64"),std::make_pair("STRATA_NATIVE_MMQ_ACTIVE_TILES","1"),
        std::make_pair("STRATA_NATIVE_MMQ_BACKEND","group-fp8")}){
        const char* raw=std::getenv(setting.first);if(!raw||std::strcmp(raw,setting.second))return 0;}
    if(count==4096){const char* raw=std::getenv("STRATA_QWEN35_GU_GROUP_INT8_I64");if(!raw||std::strcmp(raw,"1"))return 0;}
    const auto& s=expert_sources_[il];
    if(int(s.gate_exps->type)!=12||int(s.up_exps->type)!=12||
       (int(s.down_exps->type)!=12&&int(s.down_exps->type)!=13)||
       (il<pinned_experts_.layers.size()&&pinned_experts_.layers[il].data))return 0;
    std::array<bool,256> requested{};size_t unique=0,cold=0;
    for(int id:ids){if(id<0||id>=256)throw std::runtime_error("GU pipeline route ID");
        if(!requested[size_t(id)]){requested[size_t(id)]=true;++unique;if(c.key_to_slot[il*256+size_t(id)]<0)++cold;}}
    if(unique<=64||!cold||cold==unique)return 0; // Preserve the proved ordinary hot/all-warm fallback.
    if(!gu_copy_)gu_copy_=std::make_unique<ExpertGuCopyPipeline>();
    const size_t gu=589824,down=int(s.down_exps->type)==13?720896:589824;
    try{gu_copy_pending_=gu_copy_->prepare(prefetch_cache(),il,
        {{gguf_.tensor_data(*s.gate_exps),gguf_.tensor_data(*s.up_exps),gguf_.tensor_data(*s.down_exps)},{gu,gu,down}},
        host_readonly_,ids);
    }catch(...){batch_failed_=true;throw;}
    return gu_copy_pending_?uint64_t(cold):0;
}
void Runtime::finish_gu_copy(Metrics&){
    if(!gu_copy_pending_)return;
    try{const auto cost=gu_copy_->finish(prefetch_cache());gu_copy_pending_=false;
        auto& sum=gu_copy_summary_;++sum.gu_copy_layers;sum.gu_copy_bytes+=cost.bytes;sum.gu_copy_calls+=cost.calls;
        sum.gu_copy_waves+=cost.waves;sum.gu_copy_waits+=cost.waits;sum.gu_copy_ms+=cost.copy_ms;
    }catch(...){batch_failed_=true;throw;}
}
}
