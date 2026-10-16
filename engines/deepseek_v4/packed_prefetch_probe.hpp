#pragma once
#include "pp_prefetch_graph.hpp"
#include "pp_stream_probe_graph.hpp"
#include "pp_stream_metrics.hpp"
#include <array>
#include <memory>
#include <stdexcept>
namespace strata::deepseek_v4 {
// A bounded scheduling diagnostic: real layer0 prefix payload, then the same
// actual 8K MoE activation. It is not a full-model throughput measurement.
inline bool packed_prefetch_probe(PpProbeGraph& candidate,const std::array<ggml_tensor*,3>& sources,
    const float* activation,uint64_t& transferred,uint64_t& consumed){
    std::array<int32_t,256> slots;slots.fill(-1);ggml_tensor directory{};directory.data=slots.data();
    const PpCacheView view{{sources[0],sources[1],sources[2]},{},&directory};
    std::unique_ptr<ggml_context,decltype(&ggml_free)> meta(ggml_init({8192,nullptr,true}),ggml_free);
    if(!meta)throw std::runtime_error("prefetch probe metadata unavailable");
    auto* x=ggml_new_tensor_1d(meta.get(),GGML_TYPE_F32,4096);auto* norm=ggml_rms_norm(meta.get(),x,1e-6f);
    tag_pp_prefetch(norm,&view,0,8192);ggml_set_output(norm);
    auto* graph=ggml_new_graph_custom(meta.get(),8,false);ggml_build_forward_expand(graph,norm);
    std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(meta.get(),candidate.backend),ggml_backend_buffer_free);
    if(!buffer)throw std::runtime_error("prefetch probe activation allocation failed");
    ggml_backend_tensor_set(x,activation,0,4096*sizeof(float));
    const auto before=pp_stream_backend_metrics(candidate.backend);
    if(ggml_backend_graph_compute_async(candidate.backend,graph)!=GGML_STATUS_SUCCESS||
       ggml_backend_graph_compute_async(candidate.backend,candidate.graph)!=GGML_STATUS_SUCCESS)
        throw std::runtime_error("prefetch marker execution failed");
    ggml_backend_synchronize(candidate.backend);
    const auto after=pp_stream_backend_metrics(candidate.backend);
    transferred=after.prefetch_bytes-before.prefetch_bytes;consumed=after.prefetch_consumed_bytes-before.prefetch_consumed_bytes;
    return transferred==32*13369344ull&&consumed==transferred&&
        after.payload_bytes-before.payload_bytes==3422552064ull&&after.staged_bytes==before.staged_bytes;
}
}
