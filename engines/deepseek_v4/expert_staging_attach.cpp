#include "expert_staging_attach.hpp"

#include "llama-context.h"
#include "llama-moe-cache.h"
#include "llama-model.h"

#include <cstring>
#include <cstdlib>
#include <stdexcept>

void llama_context::strata_set_expert_staging(bool enabled, std::shared_ptr<strata::deepseek_v4::ExpertSourcePin>* retained,
    std::shared_ptr<strata::deepseek_v4::PackedExpertStore>* packed) {
    if (enabled && strata_staging) return;
    synchronize();
    if (!enabled) {
        if (moe_cache) moe_cache->set_expert_upload_callback(nullptr, nullptr);
        strata_staging.reset();
        return;
    }
    ggml_backend_t selected = nullptr;
    for (const auto& owner : backends) {
        const auto device = ggml_backend_get_device(owner.get());
        const auto registration = ggml_backend_dev_backend_reg(device);
        if (std::strcmp(ggml_backend_reg_name(registration), "CUDA") != 0) continue;
        if (selected) throw std::runtime_error("pinned expert transport requires one selected CUDA backend");
        selected = owner.get();
    }
    if (!selected) throw std::runtime_error("pinned expert transport has no selected CUDA backend");
    strata_staging = std::make_unique<strata::deepseek_v4::ExpertStaging>(selected);
    const uint64_t source_budget = strata::deepseek_v4::expert_source_pin_budget();
    if (source_budget||strata::deepseek_v4::packed_experts_requested()) {
        std::vector<ggml_tensor*> sources;
        const char* mtp_priority=std::getenv("STRATA_DSV4_MTP_PIN_PRIORITY");
        const bool prioritize_mtp=mtp_priority&&std::strcmp(mtp_priority,"1")==0;
        if(prioritize_mtp&&(model.layers.size()!=44||!source_budget||strata::deepseek_v4::packed_experts_requested()))
            throw std::runtime_error("MTP pin priority requires the unified44-layer pinned source inventory");
        auto append_layer=[&](const auto& layer){
            for(auto* tensor:{layer.ffn_gate_exps,layer.ffn_up_exps,layer.ffn_down_exps})
                if(tensor)sources.push_back(tensor);
        };
        // The MTP layer has no PP routes. Reserve its source registration first
        // so idle TG admission can serve it from the same cache as trunk experts.
        if(prioritize_mtp)append_layer(model.layers[43]);
        for(size_t layer=0;layer<model.layers.size()-(prioritize_mtp?1:0);++layer)append_layer(model.layers[layer]);
        if(strata::deepseek_v4::packed_experts_requested())strata_staging->pack_sources(sources,packed);
        else strata_staging->register_sources(sources, source_budget, retained);
    }
    if (moe_cache)
        moe_cache->set_expert_upload_callback(strata::deepseek_v4::ExpertStaging::dispatch, strata_staging.get());
}

void llama_context::strata_upload_expert(ggml_backend_t backend, ggml_tensor* dst,
                                       const void* src, size_t offset, size_t size) {
    if (strata_staging) strata_staging->upload(backend, dst, src, offset, size);
    else ggml_backend_tensor_set_async(backend, dst, src, offset, size);
}

strata::deepseek_v4::ExpertStagingMetrics llama_context::strata_expert_staging_metrics() const {
    return strata_staging ? strata_staging->metrics() : strata::deepseek_v4::ExpertStagingMetrics{};
}

strata::deepseek_v4::SourcePinMetrics llama_context::strata_expert_source_pin_metrics() const {
    return strata_staging ? strata_staging->source_pin_metrics() : strata::deepseek_v4::SourcePinMetrics{};
}

namespace strata::deepseek_v4 {

void configure_expert_staging(llama_context* context, bool enabled, std::shared_ptr<ExpertSourcePin>* retained,
    std::shared_ptr<PackedExpertStore>* packed) {
    if (!context) throw std::runtime_error("cannot configure expert transport without context");
    context->strata_set_expert_staging(enabled, retained,packed);
}

ExpertStagingMetrics expert_staging_metrics(const llama_context* context) {
    return context ? context->strata_expert_staging_metrics() : ExpertStagingMetrics{};
}

SourcePinMetrics expert_source_pin_metrics(const llama_context* context) {
    return context ? context->strata_expert_source_pin_metrics() : SourcePinMetrics{};
}

} // namespace strata::deepseek_v4
