#pragma once

#include "strata/artifact/gguf_reader.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace strata::dense {

struct LayerTensors {
    bool recurrent = false;
    const TensorInfo *attn_norm = nullptr, *ffn_norm = nullptr;
    const TensorInfo *q = nullptr, *k = nullptr, *v = nullptr, *attn_out = nullptr;
    const TensorInfo *q_norm = nullptr, *k_norm = nullptr;
    const TensorInfo *qkv = nullptr, *gate = nullptr, *conv = nullptr;
    const TensorInfo *dt = nullptr, *a = nullptr, *alpha = nullptr, *beta = nullptr;
    const TensorInfo *ssm_norm = nullptr, *ssm_out = nullptr;
    const TensorInfo *ffn_gate = nullptr, *ffn_up = nullptr, *ffn_down = nullptr;
};

struct Spec {
    std::string architecture;
    uint64_t n_layers = 0, n_embd = 0, n_vocab = 0, n_ff = 0;
    uint64_t context_length = 0, n_head = 0, n_head_kv = 0, head_dim = 0;
    uint64_t rope_dim = 0, rope_sections[4] = {};
    double rope_base = 0, rms_epsilon = 0;
    uint64_t ssm_state = 0, ssm_group = 0, ssm_rank = 0, ssm_inner = 0, ssm_conv = 0;
    bool query_gated = false;
    const TensorInfo *embedding = nullptr, *output_norm = nullptr, *output = nullptr;
    std::vector<LayerTensors> layers;
};

// Tensor pointers borrow the GGUF directory; the file must outlive the spec,
// weight upload and every consumer that still reads mapped host payloads.
// Failure leaves result unchanged. Binding does not execute the model.
bool load_spec(const GgufFile &, Spec &result, std::string &error);
// Overflow-safe byte count, row-block geometry and file-span validation.
uint64_t tensor_bytes(const GgufFile &, const TensorInfo &);

} // namespace strata::dense
