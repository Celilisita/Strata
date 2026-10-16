#include "cache_lease.hpp"
#include "llama-context.h"
#include "llama-graph.h"
#include "llama-moe-cache.h"
#include <stdexcept>

void llama_context::strata_set_all_hit_cache_lease(bool reject_layer_zero) {
    if (!moe_cache || strata_lease || strata_cold)
        throw std::runtime_error("all-hit lease requires one canonical cache and no CPU miss provider");
    synchronize();
    auto lease = std::make_unique<strata::deepseek_v4::AllHitCacheLease>(
        model, *moe_cache, sched.get(), reject_layer_zero);
    // Every subsequent graph must reference the GPU snapshot. Nothing can
    // admit/evict until context destruction; original cache ownership is retained.
    ggml_backend_sched_reset(sched.get());
    gf_res_prev_active = nullptr;
    for (auto& graph : gf_res_prev) if (graph) graph->reset();
    strata_lease = std::move(lease);
    moe_cache->strata_set_cache_hooks(strata::deepseek_v4::AllHitCacheLease::slot_map,
        strata::deepseek_v4::AllHitCacheLease::decorate, strata::deepseek_v4::AllHitCacheLease::guard,
        strata_lease.get());
}

namespace strata::deepseek_v4 {
void configure_all_hit_cache_lease(llama_context* context, bool reject_layer_zero) {
    if (!context) throw std::runtime_error("cannot lease a cache without context");
    context->strata_set_all_hit_cache_lease(reject_layer_zero);
}
void verify_all_hit_lease_layer_gate(llama_context* context) {
    if (!context) throw std::runtime_error("cannot verify a cache lease without context");
    context->strata_verify_all_hit_lease_layer_gate();
}
}

void llama_context::strata_verify_all_hit_lease_layer_gate() const {
    if (!strata_lease) throw std::runtime_error("layer gate has no active cache lease");
    strata_lease->verify_layer_gate();
}
