#pragma once
#include "pp_miss_coalesce.hpp"
#include "pp_stream_graph.hpp"
namespace strata::deepseek_v4 {
template<class Prefix>
PpMissCoalesceResult pp_prepare_miss_coalesce(const PpStreamSources& source,
    PpStreamPlan& plan,const Prefix& prefix,bool packed,bool source_planar) {
    const size_t groups=plan.groups.size();
    const auto* gate=source.gate;const auto* up=source.up;const auto* down=source.down;
    if(packed||source_planar||!gate||!up||!down||gate->ne[2]!=288||up->ne[2]!=288||down->ne[2]!=288||
        gate->ne[0]!=4096||gate->ne[1]!=2048||!ggml_are_same_shape(gate,up)||gate->type!=up->type||
        gate->nb[2]!=up->nb[2]||down->ne[0]!=2048||down->ne[1]!=4096||
        !ggml_is_contiguous(gate)||!ggml_is_contiguous(up)||!ggml_is_contiguous(down))return {groups,groups};
    const auto* directory=source.cache?static_cast<const int32_t*>(source.cache->directory->data):nullptr;
    return pp_coalesce_miss_plan(plan,int(source.window_experts),directory,
        [&](const auto& group){return !prefix.covers_group(source,group);});
}
} // namespace strata::deepseek_v4
