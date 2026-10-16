#pragma once
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace strata::deepseek_v4 {
struct DriverOptions {
    std::string mode = "layer";
    std::filesystem::path output, reference, candidate, prompt_file;
    int predict = -1;
    int prompt_tokens = 0;
    int layer_tokens = 1;
    int warmup_requests = 0, measure_requests = 1;
    uint32_t cpu_route_mask = 63;
    bool cpu_route_mask_set = false, cpu_reference_device = false;
    bool help = false, layer_tokens_set = false, repeat_options_set = false;
    bool ignore_eos = false;
    bool layer_fusion_seam = false;
    bool cache_prime = false, all_hit_lease = false, lease_reject_layer_zero = false;
};

DriverOptions parse_driver(const std::vector<std::string>&);
std::string read_prompt_file(const std::filesystem::path&);
}
