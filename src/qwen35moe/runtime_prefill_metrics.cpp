#include "strata/qwen35moe/runtime.hpp"
#include <stdexcept>

namespace strata::qwen35moe {
// Diagnostic-only CUDA intervals. All events are created before submission and
// read after the existing head synchronization: no extra GPU drain in PP.
void Runtime::prepare_prefill_events(int chunks) {
    prefill_event_offset_ = 0;
    dense_profile_.prepare(size_t(chunks) * layers_.size());
    if (prefill_events_.empty()) return;
    if (chunks < 1 || chunks > 192)
        throw std::invalid_argument("prefill diagnostic event capacity");
    const size_t wanted = size_t(chunks) * layers_.size();
    const size_t old = prefill_events_.size();
    if (wanted <= old) return;
    prefill_events_.resize(wanted);
    for (size_t i = old; i < wanted; ++i)
        for (auto& event : prefill_events_[i])
            cuda_check(cudaEventCreate(&event), "create long prefill event");
}

void Runtime::collect_prefill_events(int chunks, Metrics& metrics) {
    metrics.pp_dense_diagnostic_events = dense_profile_.event_count();
    for (size_t i = 0; i < size_t(chunks) * layers_.size(); ++i)
        dense_profile_.collect(i, spec_.layers[i % layers_.size()].recurrent ?
            metrics.pp_gdn_detail : metrics.pp_attention_detail);
    if (prefill_events_.empty()) return;
    const size_t count = size_t(chunks) * layers_.size();
    for (size_t i = 0; i < count; ++i) {
        const auto& group = prefill_events_[i];
        float ms = 0;
        cuda_check(cudaEventElapsedTime(&ms, group[0], group[1]), "prefill dense event");
        metrics.pp_dense_event_ms += ms;
        (spec_.layers[i % layers_.size()].recurrent ? metrics.pp_gdn_event_ms :
            metrics.pp_attention_event_ms) += ms;
        cuda_check(cudaEventElapsedTime(&ms, group[1], group[2]), "prefill route event");
        metrics.pp_route_event_ms += ms;
        cuda_check(cudaEventElapsedTime(&ms, group[2], group[3]), "prefill copy event");
        metrics.pp_copy_event_ms += ms;
        cuda_check(cudaEventElapsedTime(&ms, group[3], group[4]), "prefill MoE event");
        metrics.pp_moe_event_ms += ms;
        cuda_check(cudaEventElapsedTime(&ms, group[3], group[5]), "prefill GU event");
        metrics.pp_gu_event_ms += ms;
        cuda_check(cudaEventElapsedTime(&ms, group[5], group[6]), "prefill Down event");
        metrics.pp_down_event_ms += ms;
        cuda_check(cudaEventElapsedTime(&ms, group[6], group[4]), "prefill shared event");
        metrics.pp_shared_event_ms += ms;
    }
}
} // namespace strata::qwen35moe
