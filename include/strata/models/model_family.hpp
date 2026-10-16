#pragma once

#include "strata/artifact/gguf_reader.hpp"

#include <cstdint>
#include <string>

namespace strata::models {

enum class Family { qwen4exp, deepseek4, qwen35_dense, qwen35_moe };

struct ModelIdentity {
    Family family;
    uint64_t layers = 0;
    uint64_t hidden = 0;
    uint64_t experts = 0;
    uint64_t experts_used = 0;
};

// This is the common GGUF entry point. Model-specific tensor contracts remain
// behind their own layer implementations; an unknown architecture never falls
// through to the Qwen executor.
inline bool identify(const GgufFile& gguf, ModelIdentity& out, std::string& error) {
    out = {};
    const MetaValue* architecture = gguf.get("general.architecture");
    if (!architecture) {
        error = "missing general.architecture";
        return false;
    }
    std::string prefix;
    if (architecture->s == "qwen4exp") {
        out.family = Family::qwen4exp;
        prefix = "qwen4exp.";
    } else if (architecture->s == "deepseek4") {
        out.family = Family::deepseek4;
        prefix = "deepseek4.";
    } else if (architecture->s == "qwen35") {
        out.family = Family::qwen35_dense;
        prefix = "qwen35.";
    } else if (architecture->s == "qwen35moe") {
        out.family = Family::qwen35_moe;
        prefix = "qwen35moe.";
    } else {
        error = "unsupported model architecture '" + architecture->s + "'";
        return false;
    }
    const char* names[] = {"block_count", "embedding_length", "expert_count", "expert_used_count"};
    uint64_t* values[] = {&out.layers, &out.hidden, &out.experts, &out.experts_used};
    for (unsigned i = 0; i < (out.family == Family::qwen35_dense ? 2u : 4u); ++i) {
        const std::string key = prefix + names[i];
        const MetaValue* value = gguf.get(key);
        if (!value || !value->u) {
            error = "missing or zero " + key;
            return false;
        }
        *values[i] = value->u;
    }
    if (out.experts_used > out.experts) {
        error = prefix + "expert_used_count exceeds expert_count";
        return false;
    }
    error.clear();
    return true;
}

inline const char* family_name(Family family) {
    switch (family) {
    case Family::qwen4exp: return "qwen4exp";
    case Family::deepseek4: return "deepseek4";
    case Family::qwen35_dense: return "qwen35_dense";
    case Family::qwen35_moe: return "qwen35_moe";
    }
    return "unsupported";
}

}  // namespace strata::models
