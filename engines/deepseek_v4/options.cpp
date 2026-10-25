#include "options.hpp"

#include <charconv>
#include <limits>
#include <string_view>
#include <system_error>

namespace strata::deepseek_v4 {
namespace {

template <class T>
bool integer(std::string_view value, T& result, std::string& error, const char* flag) {
    T parsed{};
    const auto conversion = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (conversion.ec != std::errc{} || conversion.ptr != value.data() + value.size()) {
        error = std::string(flag) + ": invalid integer '" + std::string(value) + "'";
        return false;
    }
    result = parsed;
    return true;
}

bool known(std::string_view flag) {
    return flag == "--model" || flag == "-m" || flag == "--mmproj" ||
           flag == "--device" || flag == "--prompt" || flag == "-p" ||
           flag == "--ctx-size" || flag == "--n-ctx" || flag == "-c" ||
           flag == "--batch-size" || flag == "--n-batch" || flag == "-b" ||
           flag == "--ubatch-size" || flag == "--n-ubatch" || flag == "-ub" ||
           flag == "--kv-type" || flag == "--flash-attn" ||
           flag == "--kv-resident" ||
           flag == "--threads" || flag == "-t" || flag == "--gpu-layers" ||
           flag == "--moe-cache-mib" || flag == "--cache-mib" || flag == "--pinned-experts" ||
           flag == "--cold-threads" || flag == "--cold-admit-free" || flag == "--cache-leaf-aliases";
}

}  // namespace

bool parse_options(int argc, char** argv, Options& options,
                   std::vector<std::string>& remaining, std::string& error) {
    error.clear();
    remaining.clear();
    for (int index = 1; index < argc; ++index) {
        const std::string_view flag(argv[index]);
        if (!known(flag)) {
            remaining.emplace_back(flag);
            continue;
        }
        if (++index == argc) {
            error = std::string(flag) + " requires a value";
            return false;
        }
        const std::string_view value(argv[index]);
        const char* name = argv[index - 1];
        if (flag == "--model" || flag == "-m") options.model = value;
        else if (flag == "--mmproj") options.mmproj = value;
        else if (flag == "--device") options.device = value;
        else if (flag == "--prompt" || flag == "-p") options.prompt = value;
        else if (flag == "--ctx-size" || flag == "--n-ctx" || flag == "-c") {
            if (!integer(value, options.n_ctx, error, name)) return false;
        } else if (flag == "--batch-size" || flag == "--n-batch" || flag == "-b") {
            if (!integer(value, options.n_batch, error, name)) return false;
        } else if (flag == "--ubatch-size" || flag == "--n-ubatch" || flag == "-ub") {
            if (!integer(value, options.n_ubatch, error, name)) return false;
        } else if (flag == "--kv-type") {
            options.kv_type = value;
        } else if (flag == "--flash-attn") {
            if (value != "on" && value != "off" && value != "auto") {
                error = "--flash-attn must be on, off or auto";
                return false;
            }
            options.flash_attn = value;
        } else if (flag == "--kv-resident") {
            if (!integer(value, options.kv_resident, error, name)) return false;
        } else if (flag == "--threads" || flag == "-t") {
            if (!integer(value, options.n_threads, error, name)) return false;
        } else if (flag == "--gpu-layers") {
            if (!integer(value, options.gpu_layers, error, name)) return false;
        } else if (flag == "--pinned-experts") {
            if (!integer(value, options.pinned_experts, error, name)) return false;
        } else if (flag == "--cold-threads") {
            if (!integer(value, options.cold_threads, error, name)) return false;
        } else if (flag == "--cold-admit-free") {
            if (!integer(value, options.cold_admit_free, error, name)) return false;
        } else if (flag == "--cache-leaf-aliases") {
            if (!integer(value, options.cache_leaf_aliases, error, name)) return false;
        } else if (!integer(value, options.moe_cache_mib, error, name)) {
            return false;
        }
    }
    return true;
}

bool validate_options(const Options& options, std::string& error) {
    error.clear();
    if (options.model.empty()) error = "--model is required";
    else if (options.device.size() <= 4 || options.device.compare(0, 4, "CUDA") != 0) {
        error = "--device must be an exact CUDA device name, such as CUDA0";
    } else if (options.n_ctx == 0 || options.n_ctx > uint32_t(std::numeric_limits<int32_t>::max())) {
        error = "context size must be positive and fit llama_pos";
    } else if (options.n_batch == 0 || options.n_batch > uint32_t(std::numeric_limits<int32_t>::max())) {
        error = "batch size must be positive and fit int32_t";
    } else if (options.n_ubatch == 0 || options.n_ubatch > options.n_batch) {
        error = "ubatch size must be positive and no larger than batch size";
    } else if (options.kv_type != "f16" && options.kv_type != "q8_0") {
        error = "--kv-type must be f16 or q8_0";
    } else if (!options.flash_attn.empty() && options.flash_attn != "on" &&
               options.flash_attn != "off" && options.flash_attn != "auto") {
        error = "--flash-attn must be on, off or auto";
    } else if (options.kv_resident > options.n_ctx) {
        error = "--kv-resident cannot exceed --ctx-size";
    } else if (options.n_threads <= 0) error = "thread count must be positive";
    else if (options.pinned_experts > 1) error = "--pinned-experts must be 0 or 1";
    else if (options.cold_admit_free > 1) error = "--cold-admit-free must be 0 or 1";
    else if (options.cache_leaf_aliases > 1) error = "--cache-leaf-aliases must be 0 or 1";
    else if (options.cache_leaf_aliases && !options.moe_cache_mib)
        error = "cache leaf aliases require the canonical expert cache";
    else if (options.cold_threads < 0 || options.cold_threads > 32)
        error = "--cold-threads must be in 0..32 (0 disables the cold CPU provider)";
    else if (options.cold_threads && !options.moe_cache_mib)
        error = "cold CPU experts require the canonical hot expert cache";
    else if (options.gpu_layers < -1) error = "gpu layers must be -1 or nonnegative";
    else if (options.moe_cache_mib > std::numeric_limits<size_t>::max() / (1024ull * 1024ull)) {
        error = "MoE cache size overflows this process's address space";
    }
    return error.empty();
}

const char* options_usage() noexcept {
    return "--model GGUF --device CUDA0 [--mmproj GGUF] [--prompt TEXT]\n"
           "  --ctx-size N --batch-size N --ubatch-size N --threads N --kv-type f16|q8_0\n"
           "  --flash-attn on|off|auto (omitted: architecture-specific historical default)\n"
           "  --kv-resident N (DSV4 raw KV ring in VRAM; compressed history stays accessible)\n"
           "  --gpu-layers N (-1: all dense layers) --moe-cache-mib N (0: disabled)\n"
           "  --cache-mib is an alias for --moe-cache-mib; --pinned-experts 0|1 (default: 0)\n"
           "  --cold-threads 0..32 (default: 0; canonical CPU TG misses with GPU hot experts)\n"
           "  --cold-admit-free 0|1 (default: 1; 0 forces CPU misses for diagnostic comparison)\n"
           "  --cache-leaf-aliases 0|1 (default: 0; expose persistent expert cache as graph leaf weights)\n";
}

}  // namespace strata::deepseek_v4
