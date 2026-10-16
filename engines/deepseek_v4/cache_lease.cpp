#include "cache_lease.hpp"
#include "cache_lease_protocol.hpp"
#include "cache_lease_graph_key.hpp"
#include "cache_graph_arenas.hpp"
#include "llama-model.h"
#include "llama-moe-cache.h"
#include "ggml-alloc.h"
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace strata::deepseek_v4 {
AllHitCacheLease::AllHitCacheLease(const llama_model& model, llama_moe_cache& cache,
                                 ggml_backend_sched_t scheduler, bool reject_layer_zero) : scheduler_(scheduler) {
    if (model.arch != LLM_ARCH_DEEPSEEK4 || model.layers.size() != 43 || !scheduler_)
        throw std::runtime_error("all-hit lease requires the validated 43-layer DeepSeek4 cache");
    graph_arenas_ = std::make_unique<LeaseGraphArenas>();
    backend_ = cache.backend(0);
    if (!backend_ || std::strcmp(ggml_backend_reg_name(
            ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_))), "CUDA") != 0)
        throw std::runtime_error("all-hit lease needs the selected canonical CUDA cache");
    try {
        context_ = ggml_init({87 * ggml_tensor_overhead(), nullptr, true});
        if (!context_) throw std::runtime_error("all-hit lease metadata allocation failed");
        for (int layer = 0; layer < 43; ++layer) {
            originals_[layer] = cache.get_slot_map(layer, 1, 6);
            const auto* original = originals_[layer];
            const auto* bank = cache.get_experts(model.layers[layer].ffn_up_exps);
            if (cache.backend(layer) != backend_ || !original || !bank || !bank->data ||
                original->type != GGML_TYPE_I32 || ggml_nelements(original) != 256 ||
                !ggml_is_contiguous(original) || !original->data ||
                !ggml_backend_buffer_is_host(original->buffer) || bank->ne[2] < 6)
                throw std::runtime_error("all-hit lease cache directory/bank contract differs");
            maps_[layer] = ggml_new_tensor_2d(context_, GGML_TYPE_I32, 1, 256);
            valid_[layer] = ggml_new_tensor_2d(context_, GGML_TYPE_I32, 1, 256);
            ggml_format_name(maps_[layer], "strata_lease.map-%d", layer);
            ggml_format_name(valid_[layer], "strata_lease.valid-%d", layer);
        }
        status_ = ggml_new_tensor_1d(context_, GGML_TYPE_I32, 1);
        ggml_set_name(status_, "strata_lease.missing");
        buffer_ = ggml_backend_alloc_ctx_tensors(context_, backend_);
        if (!buffer_) throw std::runtime_error("all-hit lease device allocation failed");
        ggml_backend_buffer_set_usage(buffer_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (int layer = 0; layer < 43; ++layer) {
            const auto* directory = static_cast<const int32_t*>(originals_[layer]->data);
            const auto slots = cache.get_experts(model.layers[layer].ffn_up_exps)->ne[2];
            std::array<int32_t, 256> safe{}, valid{};
            for (int expert = 0; expert < 256; ++expert) {
                const auto slot = directory[expert];
                if (slot < -1 || slot >= slots) throw std::runtime_error("all-hit lease slot is out of range");
                safe[expert] = slot < 0 ? 0 : slot;
                valid[expert] = slot >= 0 && !(reject_layer_zero && layer == 0);
            }
            ggml_backend_tensor_set(maps_[layer], safe.data(), 0, sizeof(safe));
            ggml_backend_tensor_set(valid_[layer], valid.data(), 0, sizeof(valid));
        }
        const int32_t empty = 0;
        ggml_backend_tensor_set(status_, &empty, 0, sizeof(empty));
        ggml_backend_synchronize(backend_);
    } catch (...) { release(); throw; }
}

AllHitCacheLease::~AllHitCacheLease() { release(); }
bool AllHitCacheLease::graph_arenas_enabled() const noexcept {
    return graph_arenas_ && graph_arenas_->enabled();
}
llm_graph_result* AllHitCacheLease::graph_result(uint32_t tokens, uint32_t outputs, int64_t max_nodes) {
    return graph_arenas_ ? graph_arenas_->select(tokens, outputs, max_nodes) : nullptr;
}
void AllHitCacheLease::verify_layer_gate() const {
    // A truncated diagnostic has no final logits guard. This bounded4-byte
    // diagnostic read is excluded from its timing and never enters PP/TG.
    int32_t status = -1;
    ggml_backend_tensor_get(status_, &status, 0, sizeof(status));
    if (status != 0) throw std::runtime_error("all-hit lease had an unsupported missing route in the layer gate");
}
void AllHitCacheLease::release() noexcept {
    if (backend_) ggml_backend_synchronize(backend_);
    graph_arenas_.reset();
    if (buffer_) { ggml_backend_buffer_free(buffer_); buffer_ = nullptr; }
    if (context_) { ggml_free(context_); context_ = nullptr; }
}

ggml_tensor* AllHitCacheLease::slot_map(int32_t layer, int64_t tokens, int64_t used,
                                     ggml_tensor* original, void* data) {
    auto& self = *static_cast<AllHitCacheLease*>(data);
    if (layer < 0 || layer >= 43 || tokens < 1 || tokens > 8 || used != 6 ||
        original != self.originals_[layer])
        throw std::runtime_error("frozen all-hit lease supports only canonical batches of1..8 tokens and6 routes");
    self.graph_tokens_ = tokens;
    return self.maps_[layer];
}

void AllHitCacheLease::decorate(ggml_tensor* tensor, int32_t layer, void* data) {
    auto& self = *static_cast<AllHitCacheLease*>(data);
    if (layer < 0 || layer >= 43 || tensor->op != GGML_OP_GET_ROWS ||
        tensor->src[0] != self.maps_[layer] || !tensor->src[1] ||
        ggml_nelements(tensor->src[1]) > 48 || !ggml_is_contiguous(tensor->src[1]))
        throw std::runtime_error("all-hit lease GPU lookup seam differs");
    tensor->src[2] = self.valid_[layer];
    tensor->src[3] = self.status_;
    tensor->op_params[lease_tag_index] = lease_lookup_tag;
}

ggml_tensor* AllHitCacheLease::guard(ggml_context* context, ggml_tensor* logits, void* data) {
    auto& self = *static_cast<AllHitCacheLease*>(data);
    if (!logits || logits->type != GGML_TYPE_F32 || !ggml_is_contiguous(logits) ||
        self.graph_tokens_ < 1 || self.graph_tokens_ > 8)
        throw std::runtime_error("all-hit lease standard logits seam differs");
    auto* result = ggml_clamp(context, logits, -INFINITY, INFINITY);
    result->src[1] = self.status_;
    result->op_params[lease_tag_index] = lease_guard_tag;
    set_lease_graph_key(result, &self.graph_keys_[self.graph_tokens_ - 1]);
    ggml_set_name(result, "strata_lease.logits_guard");
    ggml_backend_sched_set_tensor_backend(self.scheduler_, result, self.backend_);
    return result;
}
}
