#include "strata/qwen35moe/runtime.hpp"
#include "strata/qwen35moe/nvfp4_expert_state.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/qwen35moe/nvfp4_prefix_rows.hpp"
namespace strata::qwen35moe {
PpPrefetchCache Runtime::prefetch_cache(){
    auto& c=expert_cache_;
    return {&c.data,&c.key_to_slot,&c.slot_to_key,&c.age,c.stride,size_t(resident_layers_)*256,c.epoch};
}
void Runtime::prefetch_begin(){
    if(gu_copy_pending_)throw std::runtime_error("unclosed GU copy request");
    gu_copy_summary_={};gu_copy_prefix_layer_=size_t(-1);
    if(!prefill_config_.expert_prefetch&&!prefill_config_.prefix_demand&&!prefill_config_.prefix_stage)return;
    if(!expert_prefetch_)expert_prefetch_=std::make_unique<ExpertPrefetch>(expert_sources_.size());
    expert_prefetch_->begin_request();
}
void Runtime::prefetch_layer(size_t il,int count,const std::string& reference,Metrics& metrics){
    if(!prefill_config_.expert_prefetch||decode_phase_||count<1024||!reference.empty()||
       checkpoint_sink_||paired_session_||sessions_||mtp_enabled_||batch_.expert_stage.data()||
       expert_cache_.retention.enabled()||prefill_config_.mapped_pp||!expert_prefetch_)return;
    const auto& s=expert_sources_.at(il);
    if(int(s.gate_exps->type)!=12||int(s.up_exps->type)!=12||
       (int(s.down_exps->type)!=12&&int(s.down_exps->type)!=13)||
       spec_.n_expert!=256||spec_.n_embd!=2048||spec_.n_ff_exp!=512)return;
    const size_t gu=kernels::native_mmvq_weight_bytes(12,2048,512);
    const size_t down=kernels::native_mmvq_weight_bytes(int(s.down_exps->type),512,2048);
    PpPrefetchSource source{{gguf_.tensor_data(*s.gate_exps),gguf_.tensor_data(*s.up_exps),
                              gguf_.tensor_data(*s.down_exps)},{gu,gu,down}};
    const auto before=expert_prefetch_->cost();
    try{expert_prefetch_->enqueue(prefetch_cache(),il,source,host_readonly_);}
    catch(...){batch_failed_=true;throw;}
    const auto after=expert_prefetch_->cost();
    metrics.expert_h2d+=after.h2d_bytes-before.h2d_bytes;
    metrics.expert_h2d_calls+=after.h2d_calls-before.h2d_calls;
    metrics.expert_h2d_logical_copies+=after.logical_copies-before.logical_copies;
    metrics.expert_h2d_registered_bytes+=after.registered_bytes-before.registered_bytes;
    metrics.expert_h2d_pageable_bytes+=after.pageable_bytes-before.pageable_bytes;
    metrics.expert_cache_misses+=after.issued_unique-before.issued_unique;
    metrics.expert_enqueue_ms+=after.enqueue_ms-before.enqueue_ms;
}
void Runtime::prefetch_routes(size_t il,const std::vector<int32_t>& ids){
    if(!expert_prefetch_||decode_phase_||(!prefill_config_.expert_prefetch&&!prefill_config_.prefix_demand&&!prefill_config_.prefix_stage))return;
    try{expert_prefetch_->route(prefetch_cache(),il,ids,host_readonly_,stream_,gu_copy_prefix_layer_==il);}
    catch(...){batch_failed_=true;throw;}
}
void Runtime::prefetch_finish(Metrics& metrics){
    finish_gu_copy(metrics);
    if(!expert_prefetch_||(!prefill_config_.expert_prefetch&&!prefill_config_.prefix_demand&&!prefill_config_.prefix_stage))return;
    try{expert_prefetch_->finish();metrics.pp_prefetch=expert_prefetch_->cost();
        auto& p=metrics.pp_prefetch;const auto& g=gu_copy_summary_;
        p.gu_copy_layers=g.gu_copy_layers;p.gu_copy_bytes=g.gu_copy_bytes;p.gu_copy_calls=g.gu_copy_calls;
        p.gu_copy_waves=g.gu_copy_waves;p.gu_copy_waits=g.gu_copy_waits;p.gu_copy_ms=g.gu_copy_ms;}
    catch(...){batch_failed_=true;throw;}
}
void Runtime::prefetch_drain(){
    if(nvfp4_&&nvfp4_->tg_admission&&nvfp4_->tg_admission->opened()&&decode_phase_)finish_decode();
    if(gu_copy_){gu_copy_->drain();gu_copy_pending_=false;}
    gu_copy_prefix_layer_=size_t(-1);
    if(!expert_prefetch_)return;
    expert_prefetch_->drain();++expert_cache_.epoch;
}
void Runtime::prefetch_prefix(size_t il,const std::vector<int32_t>& ids,Metrics& metrics){
    // Prefix router already drained the previous GU; close its lease before any cache mutation.
    finish_gu_copy(metrics);
    const int prefix_rows=nvfp4_?nvfp4_prefix_detail::rows(spec_.layers.at(il).recurrent):256;
    if(!expert_prefetch_||decode_phase_||(!prefill_config_.prefix_demand&&!prefill_config_.prefix_stage)||ids.size()!=size_t(prefix_rows)*8)
        throw std::invalid_argument("unprepared prefix demand copy");
    const auto& s=expert_sources_.at(il);
    const bool nv=bool(nvfp4_);
    if(nv&&(s.gate_exps->type!=40||s.up_exps->type!=40||s.down_exps->type!=40))
        throw std::invalid_argument("NVFP4 prefix inventory format changed");
    const auto layout=nv?Nvfp4ExpertInventory::layout():ExpertPayloadLayout{
        kernels::native_mmvq_weight_bytes(12,2048,512),
        kernels::native_mmvq_weight_bytes(int(s.down_exps->type),512,2048)};
    const size_t gu=layout.gu,down=layout.down;
    PpPrefetchSource source{{gguf_.tensor_data(*s.gate_exps),gguf_.tensor_data(*s.up_exps),
                            gguf_.tensor_data(*s.down_exps)},{gu,gu,down}};
    const auto before=expert_prefetch_->cost();
    try{const bool queued=expert_prefetch_->enqueue(prefetch_cache(),il,source,host_readonly_,&ids,prefill_config_.prefix_stage);
        gu_copy_prefix_layer_=queued?il:size_t(-1);}
    catch(...){batch_failed_=true;throw;}
    const auto after=expert_prefetch_->cost();
    metrics.expert_h2d+=after.h2d_bytes-before.h2d_bytes;
    metrics.expert_h2d_calls+=after.h2d_calls-before.h2d_calls;
    metrics.expert_h2d_logical_copies+=after.logical_copies-before.logical_copies;
    metrics.expert_h2d_registered_bytes+=after.registered_bytes-before.registered_bytes;
    metrics.expert_h2d_pageable_bytes+=after.pageable_bytes-before.pageable_bytes;
    metrics.expert_cache_misses+=after.issued_unique-before.issued_unique;
    metrics.expert_enqueue_ms+=after.enqueue_ms-before.enqueue_ms;
    metrics.expert_launches+=after.stage_scatter_launches-before.stage_scatter_launches;
}
}
