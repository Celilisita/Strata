#include "spec_bind.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace strata::dense {

uint64_t tensor_bytes(const GgufFile &file, const TensorInfo &tensor) {
    int block = 0, bytes = 0;
    if (tensor.shape.empty() || !block_geometry(tensor.type, block, bytes) ||
        !tensor.shape[0] || tensor.shape[0] % uint64_t(block))
        throw std::runtime_error("invalid tensor block geometry: " + tensor.name);
    uint64_t elements = 1;
    for (const auto dimension : tensor.shape) {
        if (!dimension || elements > (std::numeric_limits<uint64_t>::max)() / dimension)
            throw std::runtime_error("tensor extent overflow: " + tensor.name);
        elements *= dimension;
    }
    const uint64_t blocks = elements / uint64_t(block);
    if (blocks > (std::numeric_limits<uint64_t>::max)() / uint64_t(bytes))
        throw std::runtime_error("tensor byte count overflow: " + tensor.name);
    const uint64_t size = blocks * uint64_t(bytes);
    if (file.data_start() > file.file_size() ||
        tensor.offset > file.file_size() - file.data_start() ||
        size > file.file_size() - file.data_start() - tensor.offset)
        throw std::runtime_error("tensor payload exceeds GGUF file: " + tensor.name);
    return size;
}

namespace detail {
void validate_directory(const GgufFile &file) {
    uint64_t alignment = 32;
    if (const auto *value = file.get("general.alignment")) {
        if (value->type != MetaType::U32 || !value->u || value->u > 65536 ||
            (value->u & (value->u - 1)))
            throw std::runtime_error("invalid GGUF payload alignment");
        alignment = value->u;
    }
    std::set<std::string> names;
    std::vector<std::pair<uint64_t, uint64_t>> spans;
    spans.reserve(file.tensors().size());
    for (const auto &tensor : file.tensors()) {
        if (tensor.name.empty() || !names.insert(tensor.name).second)
            throw std::runtime_error("empty or duplicate tensor name: " + tensor.name);
        if (tensor.offset % alignment)
            throw std::runtime_error("unaligned tensor payload: " + tensor.name);
        spans.emplace_back(tensor.offset, tensor_bytes(file, tensor));
    }
    std::sort(spans.begin(), spans.end());
    for (size_t i = 1; i < spans.size(); ++i) {
        if (spans[i - 1].second > spans[i].first - spans[i - 1].first)
            throw std::runtime_error("overlapping GGUF tensor payloads");
    }
}

const TensorInfo *bind_tensor(const GgufFile &file, const std::string &name,
                             std::initializer_list<uint64_t> shape,
                             std::set<std::string> &used, bool parameter) {
    const auto *tensor = file.find(name);
    if (!tensor) throw std::runtime_error("missing tensor: " + name);
    if (tensor->shape != std::vector<uint64_t>(shape))
        throw std::runtime_error("wrong tensor shape: " + name);
    if (parameter ? tensor->type != 0 :
        (tensor->type != 0 && tensor->type != 1 && tensor->type != 8 &&
         tensor->type != 12 && tensor->type != 14))
        throw std::runtime_error("unsupported dense tensor format: " + name);
    tensor_bytes(file, *tensor);
    used.insert(name);
    return tensor;
}

LayerTensors bind_layer(const GgufFile &file, const Spec &s, uint64_t index,
                        bool recurrent, std::set<std::string> &used) {
    LayerTensors layer;
    layer.recurrent = recurrent;
    const std::string prefix = "blk." + std::to_string(index) + ".";
    auto matrix = [&](const char *suffix, std::initializer_list<uint64_t> shape) {
        return bind_tensor(file, prefix + suffix, shape, used);
    };
    auto parameter = [&](const char *suffix, std::initializer_list<uint64_t> shape) {
        return bind_tensor(file, prefix + suffix, shape, used, true);
    };
    layer.attn_norm = parameter("attn_norm.weight", {s.n_embd});
    layer.ffn_norm = parameter(s.architecture == "llama" ? "ffn_norm.weight" :
                               "post_attention_norm.weight", {s.n_embd});
    if (recurrent) {
        const uint64_t channels = 2 * s.ssm_group * s.ssm_state + s.ssm_inner;
        layer.qkv = matrix("attn_qkv.weight", {s.n_embd, channels});
        layer.gate = matrix("attn_gate.weight", {s.n_embd, s.ssm_inner});
        layer.conv = parameter("ssm_conv1d.weight", {s.ssm_conv, channels});
        layer.dt = parameter("ssm_dt.bias", {s.ssm_rank});
        layer.a = parameter("ssm_a", {s.ssm_rank});
        layer.alpha = matrix("ssm_alpha.weight", {s.n_embd, s.ssm_rank});
        layer.beta = matrix("ssm_beta.weight", {s.n_embd, s.ssm_rank});
        layer.ssm_norm = parameter("ssm_norm.weight", {s.ssm_state});
        layer.ssm_out = matrix("ssm_out.weight", {s.ssm_inner, s.n_embd});
    } else {
        layer.q = matrix("attn_q.weight", {s.n_embd, s.n_head * s.head_dim *
                                          (s.query_gated ? 2u : 1u)});
        layer.k = matrix("attn_k.weight", {s.n_embd, s.n_head_kv * s.head_dim});
        layer.v = matrix("attn_v.weight", {s.n_embd, s.n_head_kv * s.head_dim});
        layer.attn_out = matrix("attn_output.weight", {s.n_head * s.head_dim, s.n_embd});
        if (s.query_gated) {
            layer.q_norm = parameter("attn_q_norm.weight", {s.head_dim});
            layer.k_norm = parameter("attn_k_norm.weight", {s.head_dim});
        }
    }
    layer.ffn_gate = matrix("ffn_gate.weight", {s.n_embd, s.n_ff});
    layer.ffn_up = matrix("ffn_up.weight", {s.n_embd, s.n_ff});
    layer.ffn_down = matrix("ffn_down.weight", {s.n_ff, s.n_embd});
    return layer;
}
} // namespace detail
} // namespace strata::dense
