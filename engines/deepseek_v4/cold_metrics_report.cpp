#include "cold_metrics_report.hpp"
#include <cmath>
#include <iterator>
#include <limits>
#include <ostream>
#include <stdexcept>

namespace strata::deepseek_v4 {
namespace {
constexpr uint64_t ColdExpertMetrics::* counters[] = {
    &ColdExpertMetrics::jobs, &ColdExpertMetrics::cold_experts, &ColdExpertMetrics::avoided_weight_bytes,
    &ColdExpertMetrics::contribution_h2d_bytes, &ColdExpertMetrics::activation_d2h_bytes,
    &ColdExpertMetrics::route_control_d2h_bytes, &ColdExpertMetrics::host_continuations,
    &ColdExpertMetrics::joins, &ColdExpertMetrics::placeholder_gpu_experts,
};
constexpr const char* names[] = {
    "jobs", "cold_experts", "avoided_weight_bytes", "contribution_h2d_bytes", "activation_d2h_bytes",
    "route_control_d2h_bytes", "host_continuations", "joins", "placeholder_gpu_experts",
};
}

ColdExpertMetrics cold_metrics_delta(const ColdExpertMetrics& after, const ColdExpertMetrics& before) {
    ColdExpertMetrics result;
    for (auto member : counters) {
        if (after.*member < before.*member) throw std::runtime_error("cold expert counter decreased");
        result.*member = after.*member - before.*member;
    }
    result.cpu_service_ms = after.cpu_service_ms - before.cpu_service_ms;
    result.join_wait_ms = after.join_wait_ms - before.join_wait_ms;
    if (!std::isfinite(result.cpu_service_ms) || !std::isfinite(result.join_wait_ms) ||
        result.cpu_service_ms < 0 || result.join_wait_ms < 0)
        throw std::runtime_error("cold expert time counter decreased or is nonfinite");
    return result;
}

ColdExpertMetrics cold_metrics_sum(const ColdExpertMetrics& total, const ColdExpertMetrics& delta) {
    ColdExpertMetrics result;
    for (auto member : counters) {
        if (delta.*member > (std::numeric_limits<uint64_t>::max)() - total.*member)
            throw std::runtime_error("cold expert measured counter sum overflow");
        result.*member = total.*member + delta.*member;
    }
    result.cpu_service_ms = total.cpu_service_ms + delta.cpu_service_ms;
    result.join_wait_ms = total.join_wait_ms + delta.join_wait_ms;
    if (!std::isfinite(result.cpu_service_ms) || !std::isfinite(result.join_wait_ms))
        throw std::runtime_error("cold expert measured time sum is nonfinite");
    return result;
}

void write_cold_metrics(std::ostream& output, const ColdExpertMetrics& metrics, bool available) {
    if (!available) { output << "null"; return; }
    output << '{';
    for (std::size_t i = 0; i < std::size(counters); ++i) {
        if (i) output << ',';
        output << '"' << names[i] << "\":" << metrics.*counters[i];
    }
    output << ",\"cpu_service_ms\":" << metrics.cpu_service_ms
           << ",\"join_wait_ms\":" << metrics.join_wait_ms
           << ",\"expert_compute_launches\":null,\"all_hit_device_only\":false"
           << ",\"scope\":\"selected CPU miss provider; original cache route IDs still return to host; cold GPU placeholder lanes execute and are overwritten before original weighting/reduction; avoids logical miss weight payload only; CPU service time may overlap GPU and is not additive to makespan\"}";
}
}
