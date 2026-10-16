#include "strata/qwen35moe/spec.hpp"
#include "spec_bind.hpp"

#include <initializer_list>
#include <stdexcept>

namespace strata::qwen35moe {
namespace {

uint64_t required_int(const GgufFile & gguf, const char * suffix) {
    const std::string key = std::string("qwen35moe.") + suffix;
    const MetaValue * value = gguf.get(key);
    if (!value || !value->is_num() || value->type == MetaType::F32 || value->type == MetaType::F64 || !value->u) {
        throw std::runtime_error("missing or invalid GGUF metadata: " + key);
    }
    return value->u;
}

uint64_t optional_int(const GgufFile & gguf, const char * suffix, uint64_t fallback) {
    const std::string key = std::string("qwen35moe.") + suffix;
    const MetaValue * value = gguf.get(key);
    if (!value) return fallback;
    if (!value->is_num() || value->type == MetaType::F32 || value->type == MetaType::F64) {
        throw std::runtime_error("invalid GGUF metadata: " + key);
    }
    return value->u;
}


} // namespace

bool load_spec(const GgufFile & gguf, Spec & result, std::string & error) {
    try {
        const MetaValue * arch = gguf.get("general.architecture");
        if (!arch || arch->s != "qwen35moe") throw std::runtime_error("GGUF architecture is not qwen35moe");
        Spec s;
        s.context_length = required_int(gguf, "context_length");
        const uint64_t blocks = required_int(gguf, "block_count");
        s.n_mtp_layers = optional_int(gguf, "nextn_predict_layers", 0);
        if (s.n_mtp_layers >= blocks) throw std::runtime_error("invalid MTP layer count");
        s.n_layers = blocks - s.n_mtp_layers;
        s.n_embd = required_int(gguf, "embedding_length");
        s.n_expert = required_int(gguf, "expert_count");
        s.n_expert_used = required_int(gguf, "expert_used_count");
        s.n_ff_exp = required_int(gguf, "expert_feed_forward_length");
        s.n_ff_shared = required_int(gguf, "expert_shared_feed_forward_length");
        s.n_head = required_int(gguf, "attention.head_count");
        s.n_head_kv = required_int(gguf, "attention.head_count_kv");
        s.head_dim = required_int(gguf, "attention.key_length");
        if (required_int(gguf, "attention.value_length") != s.head_dim) {
            throw std::runtime_error("qwen35moe requires equal key and value head dimensions");
        }
        s.ssm_state = required_int(gguf, "ssm.state_size");
        s.ssm_group = required_int(gguf, "ssm.group_count");
        s.ssm_rank = required_int(gguf, "ssm.time_step_rank");
        s.ssm_inner = required_int(gguf, "ssm.inner_size");
        s.ssm_conv = required_int(gguf, "ssm.conv_kernel");
        s.rope_dim = required_int(gguf, "rope.dimension_count");
        const MetaValue * sections = gguf.get("qwen35moe.rope.dimension_sections");
        if (!sections || sections->type != MetaType::ARRAY || sections->count != 4 ||
            sections->items.size() != 4) throw std::runtime_error("invalid qwen35moe RoPE sections");
        for (int i = 0; i < 4; ++i) {
            if (!sections->items[i].is_num()) throw std::runtime_error("invalid qwen35moe RoPE section");
            s.rope_sections[i] = sections->items[i].u;
        }
        const MetaValue * rope_base = gguf.get("qwen35moe.rope.freq_base");
        if (!rope_base || rope_base->num() <= 0) throw std::runtime_error("invalid qwen35moe RoPE base");
        s.rope_base = rope_base->num();
        const MetaValue * epsilon = gguf.get("qwen35moe.attention.layer_norm_rms_epsilon");
        if (!epsilon || epsilon->num() <= 0) throw std::runtime_error("invalid qwen35moe RMS epsilon");
        s.rms_epsilon = epsilon->num();
        if (s.n_expert_used > s.n_expert || !s.n_head || !s.n_head_kv || s.n_head % s.n_head_kv ||
            !s.ssm_group || !s.ssm_rank || s.ssm_inner % s.ssm_rank ||
            s.ssm_inner / s.ssm_rank != s.ssm_state || s.ssm_rank % s.ssm_group ||
            s.n_layers > 64 || s.n_embd > 65536 || !s.ssm_conv ||
            2 * (s.rope_sections[0] + s.rope_sections[1] + s.rope_sections[2] + s.rope_sections[3]) != s.rope_dim ||
            s.rope_dim > s.head_dim) {
            throw std::runtime_error("unsupported qwen35moe model geometry");
        }
        const TensorInfo * embedding = gguf.find("token_embd.weight");
        if (!embedding || embedding->shape.size() != 2 || embedding->shape[0] != s.n_embd ||
            !embedding->shape[1]) throw std::runtime_error("invalid token embedding shape");
        const uint64_t vocab = embedding->shape[1];
        s.n_vocab = vocab;
        if (const auto* tokens = gguf.get("tokenizer.ggml.tokens"))
            for (size_t i = 0; i < tokens->items.size(); ++i)
                if (tokens->items[i].s == "<|image_pad|>") { s.image_token = int(i); break; }
        s.embedding = detail::bind_tensor(gguf, "token_embd.weight", {s.n_embd, vocab});
        s.output_norm = detail::bind_tensor(gguf, "output_norm.weight", {s.n_embd});
        const TensorInfo * output = gguf.find("output.weight");
        s.output = output ? detail::bind_tensor(gguf, "output.weight", {s.n_embd, vocab}) : s.embedding;
        const uint64_t interval = optional_int(gguf, "full_attention_interval", 4);
        if (!interval) throw std::runtime_error("invalid full-attention interval");
        const MetaValue * recurrent = gguf.get("qwen35moe.attention.recurrent_layers");
        if (recurrent && (recurrent->type != MetaType::ARRAY || recurrent->count != blocks ||
                          recurrent->items.size() < s.n_layers)) {
            throw std::runtime_error("invalid recurrent-layer mask");
        }
        s.layers.reserve(s.n_layers);
        for (uint64_t il = 0; il < s.n_layers; ++il) {
            const bool recr = recurrent ? recurrent->items[il].u != 0 : (il + 1) % interval != 0;
            s.layers.push_back(detail::bind_layer(gguf, s, il, recr));
        }
        for (uint64_t im = 0; im < s.n_mtp_layers; ++im) {
            MtpTensors m;
            const uint64_t il = s.n_layers + im;
            const std::string p = "blk." + std::to_string(il) + ".nextn.";
            m.decoder = detail::bind_layer(gguf, s, il, false);
            m.eh_proj = detail::bind_tensor(gguf, p + "eh_proj.weight", {2 * s.n_embd, s.n_embd});
            m.enorm = detail::bind_tensor(gguf, p + "enorm.weight", {s.n_embd});
            m.hnorm = detail::bind_tensor(gguf, p + "hnorm.weight", {s.n_embd});
            m.embedding = gguf.find(p + "embed_tokens.weight")
                ? detail::bind_tensor(gguf, p + "embed_tokens.weight", {s.n_embd, vocab}) : s.embedding;
            m.output = gguf.find(p + "shared_head_head.weight")
                ? detail::bind_tensor(gguf, p + "shared_head_head.weight", {s.n_embd, vocab}) : s.output;
            m.output_norm = gguf.find(p + "shared_head_norm.weight")
                ? detail::bind_tensor(gguf, p + "shared_head_norm.weight", {s.n_embd}) : s.output_norm;
            s.mtp.push_back(m);
        }
        result = std::move(s);
        error.clear();
        return true;
    } catch (const std::exception & e) {
        error = e.what();
        return false;
    }
}

} // namespace strata::qwen35moe
