#include "strata/qwen35moe/runtime.hpp"
#include <stdexcept>
#include <cstdio>

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
    if (metrics.pp_gdn_detail.layers || metrics.pp_attention_detail.layers) {
        const auto& gdn = metrics.pp_gdn_detail;
        const auto& attention = metrics.pp_attention_detail;
        std::fprintf(stderr,
            "qwen35_pp_dense_detail chunks=%d batch_capacity=%d gdn_layers=%llu attention_layers=%llu event_objects=%llu "
            "gdn_input_ms=%.9g gdn_query_ms=%.9g gdn_key_ms=%.9g "
            "gdn_value_ms=%.9g gdn_preprocess_ms=%.9g gdn_short_attention_ms=%.9g "
            "gdn_core_ms=%.9g gdn_post_ms=%.9g gdn_output_quant_ms=%.9g "
            "gdn_output_projection_ms=%.9g gdn_residual_ms=%.9g "
            "attention_input_ms=%.9g attention_query_ms=%.9g attention_key_ms=%.9g "
            "attention_value_ms=%.9g attention_preprocess_ms=%.9g attention_short_attention_ms=%.9g "
            "attention_core_ms=%.9g attention_post_ms=%.9g attention_output_quant_ms=%.9g "
            "attention_output_projection_ms=%.9g attention_residual_ms=%.9g "
            "gdn_preprocess_submit_ms=%.9g gdn_core_submit_ms=%.9g\n",
            chunks, batch_.capacity, static_cast<unsigned long long>(gdn.layers),
            static_cast<unsigned long long>(attention.layers),
            static_cast<unsigned long long>(metrics.pp_dense_diagnostic_events),
            gdn.gpu_ms[size_t(DenseStage::Input)],
            gdn.gpu_ms[size_t(DenseStage::Query)],
            gdn.gpu_ms[size_t(DenseStage::Key)],
            gdn.gpu_ms[size_t(DenseStage::Value)],
            gdn.gpu_ms[size_t(DenseStage::Preprocess)],
            gdn.gpu_ms[size_t(DenseStage::ShortAttention)],
            gdn.gpu_ms[size_t(DenseStage::Core)],
            gdn.gpu_ms[size_t(DenseStage::Post)],
            gdn.gpu_ms[size_t(DenseStage::OutputQuant)],
            gdn.gpu_ms[size_t(DenseStage::OutputProjection)],
            gdn.gpu_ms[size_t(DenseStage::Residual)],
            attention.gpu_ms[size_t(DenseStage::Input)],
            attention.gpu_ms[size_t(DenseStage::Query)],
            attention.gpu_ms[size_t(DenseStage::Key)],
            attention.gpu_ms[size_t(DenseStage::Value)],
            attention.gpu_ms[size_t(DenseStage::Preprocess)],
            attention.gpu_ms[size_t(DenseStage::ShortAttention)],
            attention.gpu_ms[size_t(DenseStage::Core)],
            attention.gpu_ms[size_t(DenseStage::Post)],
            attention.gpu_ms[size_t(DenseStage::OutputQuant)],
            attention.gpu_ms[size_t(DenseStage::OutputProjection)],
            attention.gpu_ms[size_t(DenseStage::Residual)],
            gdn.submit_ms[size_t(DenseStage::Preprocess)],
            gdn.submit_ms[size_t(DenseStage::Core)]);
    }
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
    if (metrics.pp_gu_plan_calls || metrics.pp_gu_plan_invalid) {
        std::fprintf(stderr,
            "qwen35_pp_gu_plan chunks=%d batch_capacity=%d calls=%llu active_calls=%llu "
            "groups=%llu max_rows=%llu active_tiles=%llu rectangular_tiles=%llu scheduled_tiles=%llu "
            "entries=%llu tile_control_bytes=%llu invalid=%llu tile_J=64 cta_multiplier=8\n",
            chunks, batch_.capacity,
            static_cast<unsigned long long>(metrics.pp_gu_plan_calls),
            static_cast<unsigned long long>(metrics.pp_gu_plan_active_calls),
            static_cast<unsigned long long>(metrics.pp_gu_plan_groups),
            static_cast<unsigned long long>(metrics.pp_gu_plan_max_rows),
            static_cast<unsigned long long>(metrics.pp_gu_plan_active_tiles),
            static_cast<unsigned long long>(metrics.pp_gu_plan_rectangular_tiles),
            static_cast<unsigned long long>(metrics.pp_gu_plan_scheduled_tiles),
            static_cast<unsigned long long>(metrics.pp_gu_plan_entries),
            static_cast<unsigned long long>(metrics.pp_gu_plan_tile_control_bytes),
            static_cast<unsigned long long>(metrics.pp_gu_plan_invalid));
    }
}
} // namespace strata::qwen35moe
