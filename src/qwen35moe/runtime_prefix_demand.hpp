#pragma once
#include <cstring>
#include <utility>
#include "strata/qwen35moe/nvfp4_prefix_rows.hpp"

namespace strata::qwen35moe {
namespace prefix_demand_detail {
// Move the real owner aside. Restoring a nonowning view never frees its storage;
// all queued kernels retain the same addresses, including during unwinding.
class RowView {
public:
    RowView(DeviceBuffer& member,int first,int rows):member_(member),owner_(std::move(member)) {
        try { member_.view(owner_,size_t(first)*2048*4,size_t(rows)*2048*4); }
        catch(...) { member_=std::move(owner_);throw; }
    }
    ~RowView() noexcept { member_=std::move(owner_); }
    RowView(const RowView&)=delete;
    RowView& operator=(const RowView&)=delete;
private:
    DeviceBuffer& member_;
    DeviceBuffer owner_;
};
inline int warm_skip() {
    static const int minimum=[] {
        const char* raw=std::getenv("STRATA_QWEN35_PP_PREFIX_WARM_SKIP");
        if(!raw||!std::strcmp(raw,"0"))return 0;
        if(!std::strcmp(raw,"224"))return 224;
        throw std::invalid_argument("PP_PREFIX_WARM_SKIP requires0 or224");
    }();
    return minimum;
}
inline bool setting(const char* name,const char* value) {
    const char* raw=std::getenv(name);return raw&&!std::strcmp(raw,value);
}
}

inline std::vector<int32_t> Runtime::batch_prefix_dense(size_t il,int count,int first,
                                                       const std::string& reference,Metrics& metrics) {
    // Only the already allocated, ordinary single-request PP schedule is split.
    // NV-only64/128 retain a tail>=1024; legacy/default prefix remains256.
    auto& b=batch_;
    if((!prefill_config_.prefix_demand&&!prefill_config_.prefix_stage)||decode_phase_||count<1280||count>4096||
       count>b.capacity||first<0||first>kv_capacity_||count>kv_capacity_-first||
       !reference.empty()||checkpoint_sink_||paired_session_||sessions_||mtp_enabled_||
       verify_state_||verify_history_||visual_table()||dense_profile_.active()||
       !prefill_scan_||!tile_attention_||!prefill_tc_||(!nvfp4_&&!b.down_mmq)||!expert_prefetch_||
       b.expert_stage.data()||expert_cache_.retention.enabled()||
       prefill_config_.expert_prefetch||prefill_config_.expert_copy_runs||
       prefill_config_.mapped_pp||prefill_config_.down_input_fused||
       (!nvfp4_&&(!prefill_config_.grouped_gu_mmq||!prefill_config_.grouped_down_mmq))||
       spec_.n_embd!=2048||spec_.n_ff_exp!=512||spec_.n_expert!=256||
       expert_cache_.slots<size_t(resident_layers_)*256+512||
       expert_cache_.key_to_slot.size()<(il+1)*256)return {};
    using prefix_demand_detail::setting;
    // Do not silently turn prefix256 into per-token attention or MMVQ dense.
    if(!setting("STRATA_QWEN35_BATCH_ATTN","1")||
       !setting("STRATA_BENCH_ATTN_MIN_ROWS","256")||
       !setting("STRATA_BENCH_DENSE_GROUP_F16","1")||
       !setting("STRATA_BENCH_DENSE_GROUP_MIN_ROWS","256")||
       !setting("STRATA_QWEN35_BULK_ROUTE","1"))return {};
    const auto& source=expert_sources_.at(il);
    const bool typed_nv=nvfp4_&&source.gate_exps->type==40&&source.up_exps->type==40&&source.down_exps->type==40;
    const bool legacy=!nvfp4_&&source.gate_exps->type==12&&source.up_exps->type==12&&
        (source.down_exps->type==12||source.down_exps->type==13)&&setting("STRATA_BENCH_Q5_TC","1");
    if(!typed_nv&&!legacy)return {};
    const int prefix_rows=typed_nv?nvfp4_prefix_detail::rows(spec_.layers.at(il).recurrent):256;
    if(prefix_rows<256&&!prefill_config_.short_attention_batch)return {};
    const int warm_minimum=prefix_demand_detail::warm_skip();
    size_t warm=0;
    for(size_t id=0;id<256;++id)warm+=expert_cache_.key_to_slot[il*256+id]>=0;
    if(warm==256||(warm_minimum&&first>0&&warm>=size_t(warm_minimum))) {
        // Full router drains prior GU, then prepare_gu_copy closes its owner.
        // No current prefix was issued: never inherit a prior chunk's marker.
        gu_copy_prefix_layer_=size_t(-1);
        return {}; // Exact full-route demand copies service any remaining cold IDs.
    }
    std::vector<int32_t> ids;
    try {
        {
            nvfp4_prefix_detail::DenseScope scope(prefix_rows);
            prefix_demand_detail::RowView hidden(b.hidden,0,prefix_rows),norm(b.norm,0,prefix_rows),mixer(b.mixer,0,prefix_rows);
            batch_dense_layer(il,prefix_rows,first,{});
            // Existing router D2H/drain also closes every previous layer reader.
            ids=batch_route(layers_[il],prefix_rows,{},metrics,false);
        }
        {
            prefix_demand_detail::RowView hidden(b.hidden,prefix_rows,count-prefix_rows),
                norm(b.norm,prefix_rows,count-prefix_rows),mixer(b.mixer,prefix_rows,count-prefix_rows);
            // Persistent GDN history/state and KV advance in the original order.
            // Prefix post-norm stays in its original b.norm prefix; only dead scratch is reused.
            batch_dense_layer(il,count-prefix_rows,first+prefix_rows,{});
        }
    } catch(...) { batch_failed_=true;throw; } // Never retry a partly advanced state.
    return ids;
}
} // namespace strata::qwen35moe
