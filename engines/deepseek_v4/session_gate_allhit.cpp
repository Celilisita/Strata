#include "session_gate.hpp"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <stdexcept>

namespace strata::deepseek_v4 {
namespace {
void environment(const char* name, const std::string& value) {
#ifdef _WIN32
    if (_putenv_s(name, value.c_str())) throw std::runtime_error("cannot set gate environment");
#else
    if (setenv(name, value.c_str(), 1)) throw std::runtime_error("cannot set gate environment");
#endif
}
}
void session_set_source_pin(uint64_t mib) {
    if (mib > 131072) throw std::runtime_error("source pin gate budget exceeds131072MiB");
    environment("STRATA_DSV4_SOURCE_PIN_MIB", std::to_string(mib));
}
void session_set_private_source(bool enabled) { environment("STRATA_DSV4_SOURCE_PRIVATE_MAPPING", enabled ? "1" : "0"); }
void session_set_native(bool stream_pp, bool hybrid_tg) {
    environment("STRATA_DSV4_STREAM_PP", stream_pp ? "1" : "0");
    environment("STRATA_DSV4_NATIVE_TG", hybrid_tg ? "1" : "0");
}
AllHitResult session_all_hit(Runtime& runtime, bool candidate) {
    AllHitResult result;
    std::vector<double> measured;
    for (int step = 0; step < 32 && measured.size() < 8; ++step) {
        runtime.reset();
        llama_synchronize(runtime.context());
        const auto before = runtime.expert_transport_metrics();
        const auto cache_before = device_expert_cache_metrics(runtime.context());
        const auto start = std::chrono::steady_clock::now();
        std::string error;
        if (!runtime.evaluate({0}, error)) throw std::runtime_error(error);
        const auto token = runtime.greedy(true);
        llama_synchronize(runtime.context());
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        verify_device_expert_cache(runtime.context());
        const auto after = runtime.expert_transport_metrics();
        const auto cache_after = device_expert_cache_metrics(runtime.context());
        const uint64_t bytes = after.bytes - before.bytes + cache_after.transport.bytes - cache_before.transport.bytes;
        const auto services = cache_after.host_services - cache_before.host_services;
        const auto lookups = cache_after.gpu[0] - cache_before.gpu[0];
        const auto hits = cache_after.gpu[1] - cache_before.gpu[1];
        const bool all_hit = bytes == 0 && services == 0 && (!candidate || (lookups == 43 && hits == 43));
        if (step < 8 || !all_hit) { ++result.warmups; continue; }
        if (result.output >= 0 && token != result.output) result.repeated_ids_equal = false;
        result.output = token;
        result.expert_bytes += bytes;
        result.host_services += services;
        result.gpu_lookups += lookups;
        result.gpu_allhit += hits;
        measured.push_back(ms);
    }
    if (measured.size() != 8) throw std::runtime_error("pure allhit gate did not converge within32single-token requests");
    std::sort(measured.begin(), measured.end());
    result.median_ms = (measured[3] + measured[4]) / 2;
    return result;
}
void session_write_all_hit(std::ostream& out, const AllHitResult& result) {
    out << "{\"median_ms\":" << result.median_ms << ",\"warmups\":" << result.warmups
        << ",\"measured\":8,\"output_id\":" << result.output << ",\"expert_bytes\":" << result.expert_bytes
        << ",\"host_services\":" << result.host_services << ",\"gpu_lookups\":" << result.gpu_lookups
        << ",\"gpu_allhit\":" << result.gpu_allhit << ",\"repeated_ids_equal_diagnostic\":" << (result.repeated_ids_equal ? "true" : "false")
        << ",\"output_id_equality_required\":false,\"checkpoint_io_in_timing\":false}";
}
}
