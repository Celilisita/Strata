#include "spec_bind.hpp"
#include <stdexcept>
namespace strata::qwen35moe::detail {
const TensorInfo * bind_tensor(const GgufFile & gguf, const std::string & name,
                               std::initializer_list<uint64_t> shape) {
    const TensorInfo * tensor = gguf.find(name);
    if (!tensor) throw std::runtime_error("missing tensor: " + name);
    if (tensor->shape != std::vector<uint64_t>(shape)) {
        throw std::runtime_error("wrong shape for tensor: " + name);
    }
    int elements = 0, bytes = 0;
    if (!block_geometry(tensor->type, elements, bytes) || tensor->elements() % uint64_t(elements)) {
        throw std::runtime_error("unsupported tensor storage: " + name);
    }
    const uint64_t size = tensor->elements() / uint64_t(elements) * uint64_t(bytes);
    if (tensor->offset > gguf.file_size() - gguf.data_start() ||
        size > gguf.file_size() - gguf.data_start() - tensor->offset) {
        throw std::runtime_error("tensor exceeds GGUF file: " + name);
    }
    return tensor;
}


LayerTensors bind_layer(const GgufFile& gguf, const Spec& s, uint64_t il, bool recurrent) {
    LayerTensors l;
    l.recurrent = recurrent;
    const uint64_t conv_width = 2 * s.ssm_group * s.ssm_state + s.ssm_inner;
    const uint64_t q_width = 2 * s.n_head * s.head_dim;
    const uint64_t kv_width = s.n_head_kv * s.head_dim;
    const std::string p = "blk." + std::to_string(il) + ".";
    auto b = [&](const char* suffix, std::initializer_list<uint64_t> shape) {
        return bind_tensor(gguf, p + suffix, shape);
    };
    l.attn_norm = b("attn_norm.weight", {s.n_embd});
    l.post_attn_norm = b("post_attention_norm.weight", {s.n_embd});
    if (l.recurrent) {
        l.qkv = b("attn_qkv.weight", {s.n_embd, conv_width});
        l.gate = b("attn_gate.weight", {s.n_embd, s.ssm_inner});
        l.conv = b("ssm_conv1d.weight", {s.ssm_conv, conv_width});
        l.dt = b("ssm_dt.bias", {s.ssm_rank});
        l.a = b("ssm_a", {s.ssm_rank});
        l.alpha = b("ssm_alpha.weight", {s.n_embd, s.ssm_rank});
        l.beta = b("ssm_beta.weight", {s.n_embd, s.ssm_rank});
        l.ssm_norm = b("ssm_norm.weight", {s.ssm_state});
        l.ssm_out = b("ssm_out.weight", {s.ssm_inner, s.n_embd});
    } else {
        l.q = b("attn_q.weight", {s.n_embd, q_width});
        l.k = b("attn_k.weight", {s.n_embd, kv_width});
        l.v = b("attn_v.weight", {s.n_embd, kv_width});
        l.attn_out = b("attn_output.weight", {s.n_head * s.head_dim, s.n_embd});
        l.q_norm = b("attn_q_norm.weight", {s.head_dim});
        l.k_norm = b("attn_k_norm.weight", {s.head_dim});
    }
    l.router = b("ffn_gate_inp.weight", {s.n_embd, s.n_expert});
    l.gate_exps = b("ffn_gate_exps.weight", {s.n_embd, s.n_ff_exp, s.n_expert});
    l.up_exps = b("ffn_up_exps.weight", {s.n_embd, s.n_ff_exp, s.n_expert});
    l.down_exps = b("ffn_down_exps.weight", {s.n_ff_exp, s.n_embd, s.n_expert});
    l.shared_router = b("ffn_gate_inp_shexp.weight", {s.n_embd});
    l.shared_gate = b("ffn_gate_shexp.weight", {s.n_embd, s.n_ff_shared});
    l.shared_up = b("ffn_up_shexp.weight", {s.n_embd, s.n_ff_shared});
    l.shared_down = b("ffn_down_shexp.weight", {s.n_ff_shared, s.n_embd});
    return l;
}
} // namespace strata::qwen35moe::detail
