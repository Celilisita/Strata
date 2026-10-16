#include "cpu_probe_report.hpp"

#include "benchmark_report.hpp"
#include "runtime.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace strata::deepseek_v4 {

int run_cpu_probe_case(const CpuProbeOptions& options) {
    std::filesystem::create_directories(options.output_dir);
    if (!std::filesystem::is_empty(options.output_dir))
        throw std::runtime_error("CPU probe output directory must be empty");
    CpuProbeResult result;
    const bool completed = run_cpu_probe(options, result);
    const bool passed = completed && result.expert_stages_exact;
    std::filesystem::create_directories(options.output_dir);
    std::ofstream identity(options.output_dir / "input-identity.txt", std::ios::binary);
    identity << result.input_identity;
    if (!identity) throw std::runtime_error("cannot write CPU probe input identity");
    std::ofstream output(options.output_dir / "cpu-probe-report.json");
    output << std::setprecision(12)
           << "{\n  \"schema\":\"strata-deepseek-v4-cpu-probe-v1\",\n"
           << "  \"canonical_commit\":" << json_string(Runtime::source_commit())
           << ",\n  \"model\":" << json_string(options.model)
           << ",\n  \"reference\":" << json_string(options.reference_dir.string())
           << ",\n  \"allow_device_reference\":" << (options.allow_device_reference ? "true" : "false")
           << ",\n  \"completed\":" << (completed ? "true" : "false")
           << ",\n  \"expert_stages_exact\":" << (result.expert_stages_exact ? "true" : "false")
           << ",\n  \"reference_host_storage_recorded\":"
           << (result.reference_host_storage_recorded ? "true" : "false")
           << ",\n  \"error\":" << json_string(result.error)
           << ",\n  \"first_difference\":" << json_string(result.first_difference)
           << ",\n  \"router_first_difference\":" << json_string(result.router_first_difference)
           << ",\n  \"weighted_first_difference\":" << json_string(result.weighted_first_difference)
           << ",\n  \"compared_expert_stages\":" << result.compared_expert_stages
           << ",\n  \"threads\":" << result.n_threads
           << ",\n  \"threadpool_resident\":" << (result.threadpool_resident ? "true" : "false")
           << ",\n  \"effective_expert_payload_bytes\":" << result.effective_expert_payload_bytes
           << ",\n  \"raw_experts_only\":" << (result.raw_experts_only ? "true" : "false")
           << ",\n  \"contribution_output_bytes\":" << result.contribution_output_bytes
           << ",\n  \"original_router_ranks\":[";
    for (std::size_t i = 0; i < result.original_router_ranks.size(); ++i) {
        if (i) output << ',';
        output << result.original_router_ranks[i];
    }
    output << ']'
           << ",\n  \"payload_basis\":\"logical distinct original MXFP4 planes; not measured memory or disk traffic\""
           << ",\n  \"load_wall_ms\":" << result.load_wall_ms
           << ",\n  \"diagnostic_compute_wall_ms\":" << result.compute_wall_ms
           << ",\n  \"service_curve_available\":" << (result.service_curve_available ? "true" : "false")
           << ",\n  \"service_compute_wall_ms\":";
    if (result.service_curve_available) output << result.service_compute_wall_ms;
    else output << "null";
    output
           << ",\n  \"service_compute_wall_ms_raw\":[";
    for (std::size_t i = 0; i < result.service_compute_wall_ms_raw.size(); ++i) {
        if (i) output << ',';
        output << result.service_compute_wall_ms_raw[i];
    }
    output << "],\n  \"service_scope\":"
           << json_string(result.raw_experts_only ?
               "canonical selected CPU Gate/Up/clamped-SwiGLU/unweighted-Down graph; input reload and file I/O excluded" :
               "canonical CPU graph including routing/weighting/reduction; input reload and file I/O excluded")
           << ",\n  \"correctness_scope\":\"CPU Gate/Up/clamped-SwiGLU/unweighted-Down only; routing and weighted seams separately reported\""
           << ",\n  \"full_model_provider_validated\":false\n}\n";
    if (!output) throw std::runtime_error("cannot write CPU probe report");
    std::cout << "{\"status\":" << json_string(passed ? "passed" : "failed")
              << ",\"mode\":\"cpu-probe\",\"expert_stages_exact\":"
              << (result.expert_stages_exact ? "true" : "false")
              << ",\"first_difference\":" << json_string(result.first_difference)
              << ",\"error\":" << json_string(result.error) << "}\n";
    return passed ? 0 : 1;
}

} // namespace strata::deepseek_v4
