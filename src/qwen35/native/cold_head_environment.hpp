#pragma once
#include "strata/qwen35/native/cold_head_binding.hpp"
#include <array>
#include <cstdlib>
#include <string>

namespace strata::qwen35::native {
// Receipt-derived identity is an all-or-none cold-load contract. No inference
// or model hashing occurs here; the service config pins the verified assets.
inline ColdHeadBinding cold_head_binding_from_environment(const strata::GgufFile *fixture) {
    constexpr std::array<const char *, 6> names{
        "STRATA_NVFP4_HEAD_SOURCE_MODEL", "STRATA_NVFP4_HEAD_SOURCE_BYTES",
        "STRATA_NVFP4_HEAD_SOURCE_MODEL_SHA256", "STRATA_NVFP4_HEAD_SOURCE_HEAD_SHA256",
        "STRATA_NVFP4_HEAD_SOURCE_HEADER_SHA256", "STRATA_NVFP4_HEAD_SOURCE_NORM_SHA256"};
    std::array<std::string, 6> values;
    size_t present = 0;
    for (size_t i = 0; i < names.size(); ++i) {
        if (const char *value = std::getenv(names[i])) {
            ++present;
            if (!*value) throw std::invalid_argument(std::string("empty cold head identity: ") + names[i]);
            values[i] = value;
        }
    }
    ColdHeadBinding binding;
    binding.fixture = fixture;
    if (!present) return binding; // Frozen Qwen3.8 identity remains the fallback.
    if (present != names.size())
        throw std::invalid_argument("cold head source environment must provide all six identity fields");
    if (values[1].size() > 16 || values[1][0] == '0')
        throw std::invalid_argument("cold head source bytes must be canonical positive decimal");
    for (const char c : values[1])
        if (c < '0' || c > '9') throw std::invalid_argument("cold head source bytes must be decimal");
    binding.source_model = values[0]; binding.source_bytes = std::stoull(values[1]);
    binding.source_model_sha256 = values[2]; binding.source_head_sha256 = values[3];
    binding.source_header_sha256 = values[4]; binding.source_norm_sha256 = values[5];
    validate_cold_head_source_identity(binding);
    return binding;
}
} // namespace strata::qwen35::native
