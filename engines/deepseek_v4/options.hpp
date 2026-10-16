#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace strata::deepseek_v4 {

// Device names are exact GGML CUDA device names (for example, CUDA0).
// Expert tensors remain in their original GGUF format in CPU mmap buffers.
struct Options {
    std::string model;
    std::string mmproj;
    std::string device = "CUDA0";
    std::string prompt = "The capital of France is";
    uint32_t n_ctx = 512;
    uint32_t n_batch = 128;
    uint32_t n_ubatch = 8;
    // Q8_0 is the preferred long-context KV format; F16 remains available
    // for existing calibrated short-context runs.
    std::string kv_type = "f16";
    uint32_t kv_resident = 0;
    int32_t n_threads = 4;
    int32_t gpu_layers = -1;
    uint64_t moe_cache_mib = 2048;
    uint32_t pinned_experts = 0;
    int32_t cold_threads = 0;
    uint32_t cold_admit_free = 1;
    uint32_t cache_leaf_aliases = 0;
};

// Consume engine options and preserve driver-specific flags, in original order.
// Validation is separate so a driver's --help does not require a model path.
bool parse_options(int argc, char** argv, Options& options,
                   std::vector<std::string>& remaining, std::string& error);
bool validate_options(const Options& options, std::string& error);
const char* options_usage() noexcept;

}  // namespace strata::deepseek_v4
