#include "cold_expert_attach.hpp"

#include "llama-context.h"
#include "llama-moe-cache.h"
#include <cstring>
#include <stdexcept>

void llama_context::strata_set_cold_experts(int threads, bool admit_free) {
    if (strata_cold || threads <= 0 || !moe_cache)
        throw std::runtime_error("cold experts require one initialization and an enabled canonical cache");
    synchronize();
    ggml_backend_t gpu = nullptr;
    for (const auto& backend : backends) {
        const auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get()));
        if (std::strcmp(ggml_backend_reg_name(reg), "CUDA") != 0) continue;
        if (gpu) throw std::runtime_error("cold experts require one selected CUDA backend");
        gpu = backend.get();
    }
    if (!gpu) throw std::runtime_error("cold experts selected CUDA backend unavailable");
    strata_cold = std::make_unique<strata::deepseek_v4::ColdExpertProvider>(
        model, gpu, threads, admit_free, cparams.cb_eval, cparams.cb_eval_user_data);
    moe_cache->set_cold_callback(strata::deepseek_v4::ColdExpertProvider::prepare, strata_cold.get());
    cparams.cb_eval = strata::deepseek_v4::ColdExpertProvider::evaluate;
    cparams.cb_eval_user_data = strata_cold.get();
}

strata::deepseek_v4::ColdExpertMetrics llama_context::strata_cold_metrics() const {
    return strata_cold ? strata_cold->metrics() : strata::deepseek_v4::ColdExpertMetrics{};
}

namespace strata::deepseek_v4 {
void configure_cold_experts(llama_context* context, int threads, bool admit_free) {
    if (!context) throw std::runtime_error("cannot configure cold experts without context");
    context->strata_set_cold_experts(threads, admit_free);
}
ColdExpertMetrics cold_expert_metrics(const llama_context* context) {
    return context ? context->strata_cold_metrics() : ColdExpertMetrics{};
}
}
