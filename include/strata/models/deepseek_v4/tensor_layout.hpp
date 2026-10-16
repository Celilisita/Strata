#pragma once

#include "strata/artifact/gguf_reader.hpp"
#include "strata/models/model_family.hpp"

#include <initializer_list>
#include <string>
#include <vector>

namespace strata::models::deepseek_v4 {

inline bool tensor_contract(const TensorInfo& tensor, uint32_t type,
                            std::initializer_list<uint64_t> shape) {
    return tensor.type == type && tensor.shape == std::vector<uint64_t>(shape);
}

inline bool required(const GgufModel& model, const std::string& name, uint32_t type,
                     std::initializer_list<uint64_t> shape, std::string& error) {
    size_t shard = 0;
    const TensorInfo* tensor = model.find(name, &shard);
    if (!tensor) {
        error = "missing DeepSeek tensor " + name;
        return false;
    }
    if (!tensor_contract(*tensor, type, shape)) {
        error = "DeepSeek tensor " + name + " has an unexpected type or shape";
        return false;
    }
    if (!model.in_bounds(*tensor, shard)) {
        error = "DeepSeek tensor " + name + " extends past its GGUF shard";
        return false;
    }
    return true;
}

// Text tower only. The vision tower and MTP sidecar have separate contracts;
// this check cannot be used as evidence that image input or drafting works.
inline bool check_text_tensors(const GgufModel& model, std::string& error) {
    ModelIdentity identity{};
    if (!identify(model.meta(), identity, error)) return false;
    if (identity.family != Family::deepseek4 || identity.layers != 43 ||
        identity.hidden != 4096 || identity.experts != 256 || identity.experts_used != 6) {
        error = "DeepSeek V4 text geometry differs from the implemented contract";
        return false;
    }
    auto want = [&](const std::string& name, uint32_t type, std::initializer_list<uint64_t> shape) {
        return required(model, name, type, shape, error);
    };
    if (!want("token_embd.weight", 30, {4096, 129280}) ||
        !want("output.weight", 30, {4096, 129280}) ||
        !want("output_norm.weight", 0, {4096}) ||
        !want("output_hc_fn.weight", 0, {16384, 4}) ||
        !want("output_hc_base.weight", 0, {4}) ||
        !want("output_hc_scale.weight", 0, {1})) return false;
    for (unsigned layer = 0; layer < 43; ++layer) {
        const std::string p = "blk." + std::to_string(layer) + ".";
        if (!want(p + "ffn_gate_exps.weight", 39, {4096, 2048, 256}) ||
            !want(p + "ffn_up_exps.weight", 39, {4096, 2048, 256}) ||
            !want(p + "ffn_down_exps.weight", 39, {2048, 4096, 256}) ||
            !want(p + "ffn_gate_inp.weight", 30, {4096, 256}) ||
            !want(p + "ffn_gate_shexp.weight", 8, {4096, 2048}) ||
            !want(p + "ffn_up_shexp.weight", 8, {4096, 2048}) ||
            !want(p + "ffn_down_shexp.weight", 8, {2048, 4096}) ||
            !want(p + "attn_norm.weight", 0, {4096}) ||
            !want(p + "ffn_norm.weight", 0, {4096}) ||
            !want(p + "attn_q_a.weight", 8, {4096, 1024}) ||
            !want(p + "attn_q_b.weight", 8, {1024, 32768}) ||
            !want(p + "attn_kv.weight", 8, {4096, 512}) ||
            !want(p + "attn_output_a.weight", 8, {4096, 8192}) ||
            !want(p + "attn_output_b.weight", 8, {8192, 4096}) ||
            !want(p + "attn_sinks.weight", 0, {64}) ||
            !want(p + "hc_attn_fn.weight", 0, {16384, 24}) ||
            !want(p + "hc_ffn_fn.weight", 0, {16384, 24}) ||
            !want(p + "hc_attn_base.weight", 0, {24}) ||
            !want(p + "hc_ffn_base.weight", 0, {24}) ||
            !want(p + "hc_attn_scale.weight", 0, {3}) ||
            !want(p + "hc_ffn_scale.weight", 0, {3})) return false;
        if (layer < 3) {
            if (!want(p + "ffn_gate_tid2eid.weight", 26, {6, 129280})) return false;
        } else if (!want(p + "exp_probs_b.bias", 0, {256})) return false;
        if (layer >= 2) {
            if (!want(p + "attn_compressor_norm.weight", 0, {512})) return false;
            if (layer % 2 == 0) {
                if (!want(p + "attn_compressor_ape.weight", 0, {1024, 4}) ||
                    !want(p + "attn_compressor_gate.weight", 30, {4096, 1024}) ||
                    !want(p + "attn_compressor_kv.weight", 30, {4096, 1024}) ||
                    !want(p + "indexer_compressor_norm.weight", 0, {128}) ||
                    !want(p + "indexer_compressor_ape.weight", 0, {256, 4}) ||
                    !want(p + "indexer_compressor_gate.weight", 30, {4096, 256}) ||
                    !want(p + "indexer_compressor_kv.weight", 30, {4096, 256}) ||
                    !want(p + "indexer.proj.weight", 30, {4096, 64}) ||
                    !want(p + "indexer.attn_q_b.weight", 8, {1024, 8192})) return false;
            } else if (!want(p + "attn_compressor_ape.weight", 0, {512, 128}) ||
                       !want(p + "attn_compressor_gate.weight", 30, {4096, 512}) ||
                       !want(p + "attn_compressor_kv.weight", 30, {4096, 512})) return false;
        }
    }
    error.clear();
    return true;
}

}  // namespace strata::models::deepseek_v4
