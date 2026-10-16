#include "benchmark_args.hpp"
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace strata::deepseek_v4 {
DriverOptions parse_driver(const std::vector<std::string>& arguments) {
    DriverOptions result;
    for (std::size_t i = 0; i < arguments.size(); ++i) {
        const std::string& flag = arguments[i];
        if (flag == "--help" || flag == "-h") { result.help = true; continue; }
        if (i + 1 == arguments.size()) throw std::runtime_error(flag + " requires a value");
        const std::string& value = arguments[++i];
        if (flag == "--mode") result.mode = value;
        else if (flag == "--output-dir") result.output = value;
        else if (flag == "--reference-dir") result.reference = value;
        else if (flag == "--prompt-file") result.prompt_file = value;
        else if (flag == "--ignore-eos") {
            if (value != "0" && value != "1") throw std::runtime_error("--ignore-eos must be 0 or 1");
            result.ignore_eos = value == "1";
        }
        else if (flag == "--prompt-tokens") {
            std::size_t used = 0;
            result.prompt_tokens = std::stoi(value, &used);
            if (used != value.size() || result.prompt_tokens < 1 || result.prompt_tokens > 16384)
                throw std::runtime_error("--prompt-tokens must be in 1..16384");
        }
        else if (flag == "--cache-prime" || flag == "--all-hit-cache-lease" || flag == "--lease-reject-layer-zero") {
            if (value != "0" && value != "1") throw std::runtime_error(flag + " must be 0 or 1");
            if (flag == "--cache-prime") result.cache_prime = value == "1";
            else if (flag == "--all-hit-cache-lease") result.all_hit_lease = value == "1";
            else result.lease_reject_layer_zero = value == "1";
        }
        else if (flag == "--layer-fusion-seam") {
            if (value != "0" && value != "1") throw std::runtime_error("--layer-fusion-seam must be 0 or 1");
            result.layer_fusion_seam = value == "1";
        }
        else if (flag == "--cpu-reference-device") {
            if (value != "0" && value != "1") throw std::runtime_error("--cpu-reference-device must be 0 or 1");
            result.cpu_reference_device = value == "1";
        }
        else if (flag == "--candidate-dir") result.candidate = value;
        else if (flag == "--cpu-route-mask") {
            std::size_t used = 0;
            const int mask = std::stoi(value, &used);
            if (used != value.size() || mask < 1 || mask > 63)
                throw std::runtime_error("--cpu-route-mask must be in 1..63");
            result.cpu_route_mask = uint32_t(mask);
            result.cpu_route_mask_set = true;
        }
        else if (flag == "--warmup-requests" || flag == "--measure-requests") {
            std::size_t used = 0;
            const int count = std::stoi(value, &used);
            const bool warmup = flag == "--warmup-requests";
            if (used != value.size() || count < (warmup ? 0 : 1) || count > 3)
                throw std::runtime_error(flag + (warmup ? " must be in 0..3" : " must be in 1..3"));
            (warmup ? result.warmup_requests : result.measure_requests) = count;
            result.repeat_options_set = true;
        }
        else if (flag == "--layer-tokens") {
            std::size_t used = 0;
            result.layer_tokens = std::stoi(value, &used);
            result.layer_tokens_set = true;
            if (used != value.size() || result.layer_tokens < 1 || result.layer_tokens > 64)
                throw std::runtime_error("--layer-tokens must be in 1..64");
        }
        else if (flag == "--n-predict") {
            std::size_t used = 0;
            result.predict = std::stoi(value, &used);
            if (used != value.size() || result.predict < 1 || result.predict > 4096)
                throw std::runtime_error("--n-predict must be in 1..4096");
        } else throw std::runtime_error("unknown benchmark argument: " + flag);
    }
    if (result.help) return result;
    if (result.mode != "layer" && result.mode != "pp" && result.mode != "tg" &&
        result.mode != "compare" && result.mode != "cpu-probe")
        throw std::runtime_error("--mode must be layer, pp, tg, compare, or cpu-probe");
    if (result.output.empty()) throw std::runtime_error("--output-dir is required");
    if (result.repeat_options_set && result.mode != "pp" && result.mode != "tg")
        throw std::runtime_error("--warmup-requests and --measure-requests apply only to PP/TG");
    if ((result.prompt_tokens || result.ignore_eos) && result.mode != "pp" && result.mode != "tg")
        throw std::runtime_error("--prompt-tokens and --ignore-eos apply only to PP/TG");
    if (result.cpu_reference_device && result.mode != "cpu-probe")
        throw std::runtime_error("--cpu-reference-device applies only to the CPU expert differential probe");
    if (result.cpu_route_mask_set && result.mode != "cpu-probe")
        throw std::runtime_error("--cpu-route-mask applies only to the CPU expert probe");
    const char* profiling = std::getenv("STRATA_DSV4_PROFILE");
    if (result.measure_requests > 1 && profiling && std::strcmp(profiling, "1") == 0)
        throw std::runtime_error("STRATA_DSV4_PROFILE=1 requires --measure-requests 1; capture-end-stop retains only the first measured request");
    if (result.mode == "compare" && (result.reference.empty() || result.candidate.empty()))
        throw std::runtime_error("compare requires --reference-dir and --candidate-dir");
    if (result.mode == "cpu-probe" && (result.reference.empty() || result.predict != -1))
        throw std::runtime_error("cpu-probe requires --reference-dir and does not accept --n-predict");
    if (result.mode == "layer" && result.predict != -1)
        throw std::runtime_error("--n-predict does not apply to the truncated layer gate");
    if (result.mode != "layer" && result.layer_tokens_set)
        throw std::runtime_error("--layer-tokens applies only to the truncated layer gate");
    if (result.mode != "layer" && result.layer_fusion_seam)
        throw std::runtime_error("--layer-fusion-seam applies only to the truncated layer gate");
    if ((result.cache_prime || result.all_hit_lease) &&
        result.mode != "layer" && result.mode != "pp" && result.mode != "tg")
        throw std::runtime_error("cache priming/lease applies only to real layer/PP/TG gates");
    if (result.all_hit_lease && !result.cache_prime)
        throw std::runtime_error("the experimental all-hit lease requires --cache-prime 1");
    if (result.lease_reject_layer_zero && (!result.all_hit_lease || result.mode == "layer"))
        throw std::runtime_error("forced lease rejection requires a full PP/TG lease gate");
    if (!result.prompt_file.empty() && result.mode != "pp" && result.mode != "tg" && result.mode != "layer")
        throw std::runtime_error("--prompt-file applies only to real text layer/PP/TG gates");
    if (result.all_hit_lease && result.predict > 8)
        throw std::runtime_error("the frozen all-hit lease remains bounded to at most 8 output tokens");
    if (result.predict == -1) result.predict = result.mode == "tg" ? 8 : 1;
    if (result.mode == "pp" && result.predict != 1)
        throw std::runtime_error("the bounded pp gate samples exactly one output token");
    return result;
}

std::string read_prompt_file(const std::filesystem::path& path) {
    const auto bytes = std::filesystem::file_size(path);
    if (!bytes || bytes > (1u << 20))
        throw std::runtime_error("prompt file must contain 1..1048576 bytes");
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open prompt file");
    std::string text(size_t(bytes), '\0');
    stream.read(text.data(), std::streamsize(bytes));
    if (!stream || stream.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("prompt file changed or could not be fully read");
    return text;
}
}
