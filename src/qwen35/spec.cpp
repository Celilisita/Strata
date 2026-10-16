#include "strata/qwen35/spec.hpp"
#include "spec_bind.hpp"
#include <cmath>
#include <stdexcept>

namespace strata::qwen35 {
namespace {
uint64_t integer(const GgufFile &gguf, const char *suffix, uint64_t fallback = 0,
                 bool optional = false) {
    const std::string key = std::string("qwen35.") + suffix;
    const MetaValue *value = gguf.get(key);
    if (!value && optional) return fallback;
    if (!value || !value->is_num() || value->type == MetaType::F32 ||
        value->type == MetaType::F64 || value->type == MetaType::BOOL ||
        value->u > (uint64_t(1) << 32) || (!optional && !value->u))
        throw std::runtime_error("missing or invalid integer: " + key);
    return value->u;
}

double number(const GgufFile &gguf, const char *suffix) {
    const std::string key = std::string("qwen35.") + suffix;
    const MetaValue *value = gguf.get(key);
    if (!value || !value->is_num() || !std::isfinite(value->num()) || value->num() <= 0)
        throw std::runtime_error("missing or invalid number: " + key);
    return value->num();
}
} // namespace

bool load_spec(const GgufFile &gguf, Spec &result, std::string &error) {
    try {
        const auto *arch = gguf.get("general.architecture");
        if (!arch || arch->type != MetaType::STRING || arch->s != "qwen35")
            throw std::runtime_error("GGUF architecture is not Dense qwen35");
        if (integer(gguf, "expert_count", 0, true))
            throw std::runtime_error("Dense qwen35 cannot contain routed experts");
        for (const auto &tensor : gguf.tensors())
            if (tensor.name.find("_exps.") != std::string::npos)
                throw std::runtime_error("Dense qwen35 contains expert tensor: " + tensor.name);
        Spec s;
        const uint64_t blocks = integer(gguf, "block_count");
        s.n_mtp_layers = integer(gguf, "nextn_predict_layers", 0, true);
        if (s.n_mtp_layers >= blocks || s.n_mtp_layers > 8 || blocks > 72)
            throw std::runtime_error("unsupported Dense/MTP layer count");
        s.n_layers = blocks - s.n_mtp_layers;
        s.n_embd = integer(gguf, "embedding_length");
        s.n_ff = integer(gguf, "feed_forward_length");
        s.context_length = integer(gguf, "context_length");
        s.n_head = integer(gguf, "attention.head_count");
        s.n_head_kv = integer(gguf, "attention.head_count_kv");
        s.head_dim = integer(gguf, "attention.key_length");
        s.ssm_state = integer(gguf, "ssm.state_size");
        s.ssm_group = integer(gguf, "ssm.group_count");
        s.ssm_rank = integer(gguf, "ssm.time_step_rank");
        s.ssm_inner = integer(gguf, "ssm.inner_size");
        s.ssm_conv = integer(gguf, "ssm.conv_kernel");
        s.rope_dim = integer(gguf, "rope.dimension_count");
        s.rms_epsilon = number(gguf, "attention.layer_norm_rms_epsilon");
        s.rope_base = number(gguf, "rope.freq_base");
        if (s.n_layers > 64 || s.n_embd > 65536 || s.n_ff > 262144 ||
            s.n_head > 256 || s.n_head_kv > 256 || s.head_dim > 512 ||
            s.ssm_state > 512 || s.ssm_group > 256 || s.ssm_rank > 256 ||
            s.ssm_inner > 65536 || s.ssm_conv > 64 ||
            s.n_head % s.n_head_kv || s.ssm_rank % s.ssm_group ||
            s.ssm_inner != s.ssm_rank * s.ssm_state ||
            integer(gguf, "attention.value_length") != s.head_dim)
            throw std::runtime_error("unsupported Dense qwen35 geometry");
        const auto *sections = gguf.get("qwen35.rope.dimension_sections");
        if (!sections || sections->type != MetaType::ARRAY || sections->count != 4 ||
            sections->items.size() != 4)
            throw std::runtime_error("invalid Dense qwen35 RoPE sections");
        uint64_t sum = 0;
        for (int i = 0; i < 4; ++i) {
            const auto &value = sections->items[i];
            if (!value.is_num() || value.type == MetaType::F32 || value.type == MetaType::F64 ||
                value.u > s.head_dim)
                throw std::runtime_error("invalid RoPE section value");
            sum += (s.rope_sections[i] = value.u);
        }
        if (2 * sum != s.rope_dim || s.rope_dim > s.head_dim)
            throw std::runtime_error("RoPE dimensions disagree");
        const auto *embedding = gguf.find("token_embd.weight");
        if (!embedding || embedding->shape.size() != 2 || !embedding->shape[1])
            throw std::runtime_error("invalid token embedding");
        s.n_vocab = embedding->shape[1];
        if (s.n_vocab > (uint64_t(1) << 24)) throw std::runtime_error("invalid vocabulary size");
        s.embedding = detail::bind_tensor(gguf, "token_embd.weight", {s.n_embd, s.n_vocab});
        s.output_norm = detail::bind_tensor(gguf, "output_norm.weight", {s.n_embd});
        s.output = gguf.find("output.weight")
            ? detail::bind_tensor(gguf, "output.weight", {s.n_embd, s.n_vocab}) : s.embedding;
        const uint64_t interval = integer(gguf, "full_attention_interval", 4, true);
        if (!interval || interval > 64) throw std::runtime_error("invalid attention interval");
        const auto *mask = gguf.get("qwen35.attention.recurrent_layers");
        if (mask && (mask->type != MetaType::ARRAY || mask->count != blocks ||
                     mask->items.size() < s.n_layers))
            throw std::runtime_error("invalid recurrent layer mask");
        for (uint64_t index = 0; index < s.n_layers; ++index) {
            if (mask && (!mask->items[index].is_num() ||
                mask->items[index].type == MetaType::F32 ||
                mask->items[index].type == MetaType::F64 || mask->items[index].u > 1))
                throw std::runtime_error("invalid recurrent mask value");
            const bool recurrent = mask ? mask->items[index].u != 0 : (index + 1) % interval != 0;
            s.layers.push_back(detail::bind_layer(gguf, s, index, recurrent));
        }
        for (uint64_t index = s.n_layers; index < blocks; ++index) {
            MtpTensors m;
            m.decoder = detail::bind_layer(gguf, s, index, false);
            const std::string prefix = "blk." + std::to_string(index) + ".nextn.";
            m.eh_proj = detail::bind_tensor(gguf, prefix + "eh_proj.weight", {2 * s.n_embd, s.n_embd});
            m.enorm = detail::bind_tensor(gguf, prefix + "enorm.weight", {s.n_embd});
            m.hnorm = detail::bind_tensor(gguf, prefix + "hnorm.weight", {s.n_embd});
            auto optional = [&](const char *suffix, std::initializer_list<uint64_t> shape,
                                const TensorInfo *fallback) {
                return gguf.find(prefix + suffix)
                    ? detail::bind_tensor(gguf, prefix + suffix, shape) : fallback;
            };
            m.embedding = optional("embed_tokens.weight", {s.n_embd, s.n_vocab}, s.embedding);
            m.output = optional("shared_head_head.weight", {s.n_embd, s.n_vocab}, s.output);
            m.output_norm = optional("shared_head_norm.weight", {s.n_embd}, s.output_norm);
            s.mtp.push_back(m);
        }
        result = std::move(s);
        error.clear();
        return true;
    } catch (const std::exception &exception) {
        error = exception.what();
        return false;
    }
}
} // namespace strata::qwen35
