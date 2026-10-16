#include "spec_bind.hpp"
#include <limits>
#include <stdexcept>

namespace strata::qwen35 {

uint64_t tensor_bytes(const GgufFile &gguf, const TensorInfo &tensor) {
    uint64_t elements = 1;
    for (uint64_t dimension : tensor.shape) {
        if (!dimension || elements > (std::numeric_limits<uint64_t>::max)() / dimension)
            throw std::runtime_error("invalid tensor dimensions: " + tensor.name);
        elements *= dimension;
    }
    int block = 0, bytes = 0;
    if (!block_geometry(tensor.type, block, bytes) || block <= 0 || bytes <= 0 ||
        elements % uint64_t(block) ||
        elements / uint64_t(block) > (std::numeric_limits<uint64_t>::max)() / uint64_t(bytes))
        throw std::runtime_error("unsupported tensor storage: " + tensor.name);
    const uint64_t size = elements / uint64_t(block) * uint64_t(bytes);
    if (gguf.data_start() > gguf.file_size() ||
        tensor.offset > gguf.file_size() - gguf.data_start() ||
        size > gguf.file_size() - gguf.data_start() - tensor.offset)
        throw std::runtime_error("tensor exceeds GGUF file: " + tensor.name);
    return size;
}

namespace detail {
const TensorInfo *bind_tensor(const GgufFile &gguf, const std::string &name,
                              std::initializer_list<uint64_t> shape) {
    const TensorInfo *tensor = gguf.find(name);
    if (!tensor) throw std::runtime_error("missing tensor: " + name);
    if (tensor->shape != std::vector<uint64_t>(shape))
        throw std::runtime_error("wrong tensor shape: " + name);
    tensor_bytes(gguf, *tensor);
    return tensor;
}

LayerTensors bind_layer(const GgufFile &gguf, const Spec &s, uint64_t index,
                       bool recurrent) {
    LayerTensors l;
    l.recurrent = recurrent;
    const uint64_t conv_width = 2 * s.ssm_group * s.ssm_state + s.ssm_inner;
    const uint64_t kv_width = s.n_head_kv * s.head_dim;
    const std::string prefix = "blk." + std::to_string(index) + ".";
    auto bind = [&](const char *suffix, std::initializer_list<uint64_t> shape) {
        return bind_tensor(gguf, prefix + suffix, shape);
    };
    l.attn_norm = bind("attn_norm.weight", {s.n_embd});
    l.post_attn_norm = bind("post_attention_norm.weight", {s.n_embd});
    if (recurrent) {
        l.qkv = bind("attn_qkv.weight", {s.n_embd, conv_width});
        l.gate = bind("attn_gate.weight", {s.n_embd, s.ssm_inner});
        l.conv = bind("ssm_conv1d.weight", {s.ssm_conv, conv_width});
        l.dt = bind("ssm_dt.bias", {s.ssm_rank});
        l.a = bind("ssm_a", {s.ssm_rank});
        l.alpha = bind("ssm_alpha.weight", {s.n_embd, s.ssm_rank});
        l.beta = bind("ssm_beta.weight", {s.n_embd, s.ssm_rank});
        l.ssm_norm = bind("ssm_norm.weight", {s.ssm_state});
        l.ssm_out = bind("ssm_out.weight", {s.ssm_inner, s.n_embd});
    } else {
        l.q = bind("attn_q.weight", {s.n_embd, 2 * s.n_head * s.head_dim});
        l.k = bind("attn_k.weight", {s.n_embd, kv_width});
        l.v = bind("attn_v.weight", {s.n_embd, kv_width});
        l.attn_out = bind("attn_output.weight", {s.n_head * s.head_dim, s.n_embd});
        l.q_norm = bind("attn_q_norm.weight", {s.head_dim});
        l.k_norm = bind("attn_k_norm.weight", {s.head_dim});
    }
    l.ffn_gate = bind("ffn_gate.weight", {s.n_embd, s.n_ff});
    l.ffn_up = bind("ffn_up.weight", {s.n_embd, s.n_ff});
    l.ffn_down = bind("ffn_down.weight", {s.n_ff, s.n_embd});
    return l;
}
} // namespace detail
} // namespace strata::qwen35
