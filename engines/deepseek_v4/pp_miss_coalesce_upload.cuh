#pragma once
#include "pp_stream_transfer.cuh"
#include "pp_stream_plan.hpp"
namespace strata::deepseek_v4 {
// Tagged groups have only immutable source misses. Copy each real contiguous
// source run into its local weight rows; never copy the holes between runs.
template<class Window,class IsPinned>
bool pp_upload_coalesced_misses(const PpStreamSources& s,const PpStreamGroup& g,Window& w,
    cudaStream_t copy,PpTransportQuery query,void* owner,PpStreamEdges& edges,
    PpStreamMetrics& metrics,IsPinned is_pinned) {
    if(g.source_keys.empty())return false;
    if(g.cache_first_slot>=0||w.borrowed||w.planar[0]||w.packed||g.experts>32||
        g.source_keys.size()!=size_t(g.experts)||g.source_runs.empty())
        throw std::runtime_error("PP coalesced upload ownership differs");
    const size_t required=size_t(s.window_experts)*(2*s.gate->nb[2]+s.down->nb[2]);
    if(!w.device.p||w.device.bytes<required)throw std::runtime_error("PP coalesced upload exceeds owned window");
    const char* setting=std::getenv("STRATA_DSV4_PP_SOURCE_RANGES");
    const auto source_query=setting&&!std::strcmp(setting,"0")?nullptr:query;
    const size_t half=s.gate->nb[2],down=s.down->nb[2];
    const size_t width[]{half,half,down},pitch[]{2*half,2*half,down};
    const size_t offset[]{0,half,size_t(s.window_experts)*2*half};
    const ggml_tensor* tensor[]{s.gate,s.up,s.down};
    for(const auto& run:g.source_runs)for(int role=0;role<3;++role) {
        const auto* source=static_cast<const char*>(tensor[role]->data)+size_t(run.original_first)*width[role];
        const size_t destination=offset[role]+size_t(run.local_first)*pitch[role];
        const auto owned=source_query?source_query(source,size_t(run.count)*width[role],owner):ExpertSourcePin::Segment{};
        // Authoritative registration spans and the original edge/slab fallback
        // remain inside pp_upload_rows, including partial expert edge pages.
        pp_upload_rows(source,width[role],pitch[role],destination,0,run.count,
            owned.bytes||is_pinned(source,size_t(run.count)*width[role]),w,copy,source_query,owner,&edges,metrics);
    }
    CUDA_CHECK(cudaEventRecord(w.ready,copy));w.ready_recorded=true;return true;
}
} // namespace strata::deepseek_v4
