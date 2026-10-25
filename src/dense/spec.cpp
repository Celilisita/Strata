#include "strata/dense/spec.hpp"
#include "spec_bind.hpp"
#include <cmath>
#include <stdexcept>
#include <utility>

namespace strata::dense {
namespace {

bool integer_type(MetaType type) {
    return type == MetaType::U8 || type == MetaType::I8 || type == MetaType::U16 ||
           type == MetaType::I16 || type == MetaType::U32 || type == MetaType::I32 ||
           type == MetaType::U64 || type == MetaType::I64;
}

uint64_t integer(const GgufFile &file, const std::string &key, uint64_t fallback = 0,
                 bool optional = false) {
    const auto *value = file.get(key);
    if (!value && optional) return fallback;
    if (!value || !integer_type(value->type) || value->u > (uint64_t(1) << 32) ||
        (!optional && !value->u))
        throw std::runtime_error("missing or invalid dense integer: " + key);
    return value->u;
}

double number(const GgufFile &file, const std::string &key) {
    const auto *value = file.get(key);
    if (!value || !value->is_num() || value->type == MetaType::BOOL ||
        !std::isfinite(value->num()) || value->num() <= 0)
        throw std::runtime_error("missing or invalid dense number: " + key);
    return value->num();
}

void reject_math_overrides(const GgufFile &file, const Spec &s) {
    const std::string prefix = s.architecture + ".";
    const char *unit_scales[] = {"embedding_scale", "embedding_multiplier", "residual_scale",
        "residual_multiplier", "logit_scale", "logits_scale", "output_scale"};
    for (const auto *suffix : unit_scales) {
        const auto key = prefix + suffix;
        if (file.get(key) && number(file, key) != 1.0)
            throw std::runtime_error("unsupported dense model scaling: " + key);
    }
    if (const auto *value = file.get(prefix + "attention.scale")) {
        const double standard = 1.0 / std::sqrt(double(s.head_dim));
        if (!value->is_num() || value->type == MetaType::BOOL ||
            !std::isfinite(value->num()) || std::abs(value->num() - standard) > standard * 1e-6)
            throw std::runtime_error("unsupported dense attention scale");
    }
    if (integer(file, prefix + "attention.sliding_window", 0, true))
        throw std::runtime_error("dense sliding-window attention is not implemented");
    for (const auto &[key, value] : file.metadata()) {
        if (key.rfind(prefix + "rope.scaling.", 0) != 0) continue;
        if (key == prefix + "rope.scaling.type" && value.type == MetaType::STRING &&
            value.s == "none") continue;
        if (key == prefix + "rope.scaling.factor" && number(file, key) == 1.0) continue;
        throw std::runtime_error("unsupported dense RoPE scaling: " + key);
    }
}

void bind_qwen_metadata(const GgufFile &file, Spec &s) {
    const std::string p = s.architecture + ".";
    s.query_gated = true;
    s.ssm_state = integer(file, p + "ssm.state_size");
    s.ssm_group = integer(file, p + "ssm.group_count");
    s.ssm_rank = integer(file, p + "ssm.time_step_rank");
    s.ssm_inner = integer(file, p + "ssm.inner_size");
    s.ssm_conv = integer(file, p + "ssm.conv_kernel");
    if (s.ssm_state > 512 || s.ssm_group > 256 || s.ssm_rank > 256 ||
        s.ssm_inner > 65536 || s.ssm_conv != 4 || s.ssm_rank % s.ssm_group ||
        s.ssm_inner != s.ssm_rank * s.ssm_state)
        throw std::runtime_error("unsupported dense Qwen GDN geometry");
    const auto *sections = file.get(p + "rope.dimension_sections");
    if (!sections || sections->type != MetaType::ARRAY || sections->count != 4 ||
        sections->items.size() != 4)
        throw std::runtime_error("invalid dense Qwen RoPE sections");
    uint64_t sum = 0;
    for (int i = 0; i < 4; ++i) {
        const auto &value = sections->items[i];
        if (!integer_type(value.type) || value.u > s.head_dim)
            throw std::runtime_error("invalid dense Qwen RoPE section value");
        sum += (s.rope_sections[i] = value.u);
    }
    if (2 * sum != s.rope_dim)
        throw std::runtime_error("dense Qwen RoPE section dimensions disagree");
}
} // namespace

bool load_spec(const GgufFile &file, Spec &result, std::string &error) {
    try {
        const auto *architecture = file.get("general.architecture");
        if (!architecture || architecture->type != MetaType::STRING ||
            (architecture->s != "llama" && architecture->s != "qwen35"))
            throw std::runtime_error("dense runtime requires llama or qwen35 architecture");
        if (integer(file, "split.count", 1, true) != 1 ||
            integer(file, "split.no", 0, true) != 0)
            throw std::runtime_error("dense runtime requires one complete GGUF file");
        detail::validate_directory(file);
        Spec s;
        s.architecture = architecture->s;
        const std::string p = s.architecture + ".";
        if (integer(file, p + "expert_count", 0, true) ||
            integer(file, p + "nextn_predict_layers", 0, true))
            throw std::runtime_error("dense runtime does not execute experts or next-N layers");
        s.n_layers = integer(file, p + "block_count");
        s.n_embd = integer(file, p + "embedding_length");
        s.n_ff = integer(file, p + "feed_forward_length");
        s.context_length = integer(file, p + "context_length");
        s.n_head = integer(file, p + "attention.head_count");
        s.n_head_kv = integer(file, p + "attention.head_count_kv");
        if (s.n_layers > 64 || s.n_embd > 65536 || s.n_ff > 262144 ||
            s.n_head > 256 || s.n_head_kv > 256 || s.n_head % s.n_head_kv ||
            s.context_length > (uint64_t(1) << 24))
            throw std::runtime_error("unsupported dense model dimensions");
        s.head_dim = integer(file, p + "attention.key_length", s.n_embd / s.n_head, true);
        if (!s.head_dim || s.head_dim > 512 ||
            integer(file, p + "attention.value_length", s.head_dim, true) != s.head_dim)
            throw std::runtime_error("unsupported dense key/value head dimensions");
        s.rope_dim = integer(file, p + "rope.dimension_count", s.head_dim, true);
        s.rope_base = number(file, p + "rope.freq_base");
        s.rms_epsilon = number(file, p + "attention.layer_norm_rms_epsilon");
        if (!s.rope_dim || s.rope_dim > s.head_dim || s.rope_dim % 2)
            throw std::runtime_error("invalid dense RoPE dimensions");
        if (s.architecture == "qwen35") bind_qwen_metadata(file, s);
        else if (s.n_head * s.head_dim != s.n_embd)
            throw std::runtime_error("dense Llama query width must equal embedding width");
        reject_math_overrides(file, s);
        const auto *embedding = file.find("token_embd.weight");
        if (!embedding || embedding->shape.size() != 2 || !embedding->shape[1] ||
            embedding->shape[1] > (uint64_t(1) << 24))
            throw std::runtime_error("invalid dense token embedding");
        s.n_vocab = embedding->shape[1];
        if (integer(file, p + "vocab_size", s.n_vocab, true) != s.n_vocab)
            throw std::runtime_error("dense vocabulary metadata disagrees with embedding");
        if (const auto *tokens = file.get("tokenizer.ggml.tokens")) {
            if (tokens->type != MetaType::ARRAY || tokens->elem != MetaType::STRING ||
                tokens->count != s.n_vocab)
                throw std::runtime_error("dense tokenizer vocabulary disagrees with embedding");
        }
        std::set<std::string> used;
        s.embedding = detail::bind_tensor(file, "token_embd.weight", {s.n_embd, s.n_vocab}, used);
        s.output_norm = detail::bind_tensor(file, "output_norm.weight", {s.n_embd}, used, true);
        s.output = file.find("output.weight")
            ? detail::bind_tensor(file, "output.weight", {s.n_embd, s.n_vocab}, used) : s.embedding;
        const auto *mask = file.get(p + "attention.recurrent_layers");
        if (mask && (s.architecture != "qwen35" || mask->type != MetaType::ARRAY ||
                     mask->count != s.n_layers || mask->items.size() != s.n_layers))
            throw std::runtime_error("invalid dense recurrent-layer mask");
        const auto interval = s.query_gated ? integer(file, p + "full_attention_interval", 4, true) : 1;
        if (!interval || interval > 64) throw std::runtime_error("invalid dense full-attention interval");
        s.layers.reserve(s.n_layers);
        for (uint64_t i = 0; i < s.n_layers; ++i) {
            if (mask && ((!integer_type(mask->items[i].type) && mask->items[i].type != MetaType::BOOL) ||
                         mask->items[i].u > 1))
                throw std::runtime_error("invalid dense recurrent-layer mask value");
            const bool recurrent = s.query_gated && (mask ? mask->items[i].u != 0 : (i + 1) % interval != 0);
            s.layers.push_back(detail::bind_layer(file, s, i, recurrent, used));
        }
        for (const auto &tensor : file.tensors()) {
            if (!used.count(tensor.name))
                throw std::runtime_error("unsupported unbound dense tensor: " + tensor.name);
        }
        result = std::move(s);
        error.clear();
        return true;
    } catch (const std::exception &exception) {
        error = exception.what();
        return false;
    }
}
} // namespace strata::dense
