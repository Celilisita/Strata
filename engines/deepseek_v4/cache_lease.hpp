#pragma once
#include "ggml-backend.h"
#include <array>
#include <memory>

struct llama_model;
struct llama_context;
class llama_moe_cache;
class llm_graph_result;

namespace strata::deepseek_v4 {
class LeaseGraphArenas;
// Explicit benchmark-only lease. The canonical cache retains weight ownership;
// this immutable GPU mirror cannot admit/evict. Any missing route fails via
// standard output logits, before a token can be accepted. No production fallback.
class AllHitCacheLease {
public:
    AllHitCacheLease(const llama_model&, llama_moe_cache&, ggml_backend_sched_t, bool reject_layer_zero);
    ~AllHitCacheLease();
    void verify_layer_gate() const;
    bool graph_arenas_enabled() const noexcept;
    llm_graph_result* graph_result(uint32_t tokens, uint32_t outputs, int64_t max_nodes);
    AllHitCacheLease(const AllHitCacheLease&) = delete;
    AllHitCacheLease& operator=(const AllHitCacheLease&) = delete;
    static ggml_tensor* slot_map(int32_t, int64_t, int64_t, ggml_tensor*, void*);
    static void decorate(ggml_tensor*, int32_t, void*);
    static ggml_tensor* guard(ggml_context*, ggml_tensor*, void*);

private:
    void release() noexcept;
    ggml_context* context_ = nullptr;
    ggml_backend_t backend_ = nullptr;
    ggml_backend_sched_t scheduler_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    std::array<ggml_tensor*, 43> maps_{}, valid_{}, originals_{};
    ggml_tensor* status_ = nullptr;
    std::array<uint8_t, 8> graph_keys_{};
    int64_t graph_tokens_ = 0;
    std::unique_ptr<LeaseGraphArenas> graph_arenas_;
};
void configure_all_hit_cache_lease(llama_context*, bool reject_layer_zero = false);
void verify_all_hit_lease_layer_gate(llama_context*);
}
