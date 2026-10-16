#pragma once

#include "strata/artifact/gguf_reader.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::qwen35moe {

struct LayerTensors {
    bool recurrent = false;
    const TensorInfo * attn_norm = nullptr, * post_attn_norm = nullptr;
    const TensorInfo * q = nullptr, * k = nullptr, * v = nullptr, * attn_out = nullptr;
    const TensorInfo * q_norm = nullptr, * k_norm = nullptr;
    const TensorInfo * qkv = nullptr, * gate = nullptr, * conv = nullptr;
    const TensorInfo * dt = nullptr, * a = nullptr, * alpha = nullptr, * beta = nullptr;
    const TensorInfo * ssm_norm = nullptr, * ssm_out = nullptr;
    const TensorInfo * router = nullptr, * gate_exps = nullptr, * up_exps = nullptr, * down_exps = nullptr;
    const TensorInfo * shared_router = nullptr, * shared_gate = nullptr;
    const TensorInfo * shared_up = nullptr, * shared_down = nullptr;
};

struct MtpTensors {
    LayerTensors decoder;
    const TensorInfo * eh_proj = nullptr, * enorm = nullptr, * hnorm = nullptr;
    const TensorInfo * embedding = nullptr, * output_norm = nullptr, * output = nullptr;
};

struct Spec {
    uint64_t n_layers = 0, n_mtp_layers = 0, n_embd = 0, n_vocab = 0;
    uint64_t context_length = 0;
    int image_token = -1;
    uint64_t n_expert = 0, n_expert_used = 0;
    uint64_t n_ff_exp = 0, n_ff_shared = 0;
    uint64_t n_head = 0, n_head_kv = 0, head_dim = 0;
    uint64_t ssm_state = 0, ssm_group = 0, ssm_rank = 0, ssm_inner = 0, ssm_conv = 0;
    uint64_t rope_sections[4] = {};
    uint64_t rope_dim = 0;
    double rms_epsilon = 0, rope_base = 0;
    const TensorInfo * embedding = nullptr, * output_norm = nullptr, * output = nullptr;
    std::vector<LayerTensors> layers;
    std::vector<MtpTensors> mtp;
};

// Parse the native model geometry and bind every main-stack tensor by GGUF name.
// MTP binds separate decoder/conditioning tensors; main-stack execution stays distinct.
bool load_spec(const GgufFile & gguf, Spec & result, std::string & error);

} // namespace strata::qwen35moe
