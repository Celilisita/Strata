#pragma once

#include "llama.h"
#include "split_loader.hpp"
#include "llama-model.h"
#include "gguf.h"
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::glm5_next {

inline bool accepted_architecture(const char* value) {
    return value && (!std::strcmp(value, "deepseek4") || !std::strcmp(value, "glm5-next"));
}

inline bool validate(const llama_model& model, std::string& error) {
    if (model.arch != LLM_ARCH_GLM5_NEXT) return true;
    const auto& h = model.hparams;
    if (h.n_layer() != 45 || h.n_embd != 4096 || h.n_expert != 288 ||
        h.n_expert_used_max() != 8 || h.dsv4_hc_mult != 4 || h.indexer_kpool != 4 ||
        h.n_lora_kv != 512 || h.n_embd_head_kda != 128) {
        error = "GLM-5.3 Flash loaded dimensions differ from the verified model contract";
        return false;
    }
    for (uint32_t i = h.n_layer_dense_lead; i < h.n_layer(); ++i) {
        const auto& layer = model.layers.at(i);
        for (const auto* bank : {layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps}) {
            if (!bank || bank->ne[2] != 288 || !bank->data || !bank->buffer ||
                !ggml_is_contiguous(bank) || !ggml_backend_buffer_is_host(bank->buffer)) {
                error = "GLM routed experts must retain original contiguous host GGUF ownership";
                return false;
            }
        }
    }
    return true;
}

template<class Plan>
Plan warm_plan(const llama_model& model) {
    Plan plan;
    uint64_t bytes = 0;
    for (uint32_t i = 0; i < model.layers.size(); ++i) {
        const auto& layer = model.layers.at(i);
        for (const auto* tensor : {layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps}) {
            if (!tensor) continue; // the first three blocks are dense
            if (!tensor->data) throw std::runtime_error("GLM expert warm source is absent");
            const size_t count = ggml_nbytes(tensor);
            plan.read.push_back({static_cast<const uint8_t*>(tensor->data), count});
            plan.lock.push_back({static_cast<const uint8_t*>(tensor->data), count});
            bytes += count;
        }
    }
    if (bytes > (160ull << 30)) throw std::runtime_error("GLM expert warm exceeds its host budget");
    return plan;
}

} // namespace strata::glm5_next
