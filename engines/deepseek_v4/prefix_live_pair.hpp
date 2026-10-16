#pragma once
#include "long_prefix_compare.hpp"
#include "device_cache_probe_report.hpp"
#include <chrono>
#include <fstream>
#include <stdexcept>
namespace strata::deepseek_v4 {
struct PrefixMetricsStart {
    ExpertStagingMetrics transport;
    DeviceCacheMetrics cache;
};
inline DeviceCacheDmaCosts prefix_cost_delta(DeviceCacheDmaCosts now, const DeviceCacheDmaCosts& before) {
    now.submit_ns -= before.submit_ns; now.commit_wait_ns -= before.commit_wait_ns;
    now.enqueue_miss_waits -= before.enqueue_miss_waits; now.enqueue_miss_wait_ns -= before.enqueue_miss_wait_ns;
    now.enqueue_miss_deadlines -= before.enqueue_miss_deadlines;
    now.frequency_d2h_bytes -= before.frequency_d2h_bytes; now.frequency_read_ns -= before.frequency_read_ns;
    now.enqueue_budget_records -= before.enqueue_budget_records; now.enqueue_budget_queries -= before.enqueue_budget_queries;
    now.enqueue_budget_waits -= before.enqueue_budget_waits; now.enqueue_budget_wait_ns -= before.enqueue_budget_wait_ns;
    now.graph_preuploads -= before.graph_preuploads; now.graph_preupload_ns -= before.graph_preupload_ns;
    return now; // API/resource worst spans describe the retained context.
}
inline void prefix_live_metrics(Runtime& runtime, const PrefixMetricsStart& before, double wall_ms, std::ostream& report) {
    const auto perf = runtime.performance();
    const auto transport = runtime.expert_transport_metrics();
    const auto cache = device_expert_cache_metrics(runtime.context());
    report << ",\"metrics\":{\"pp_tokens\":" << perf.n_p_eval << ",\"tg_tokens\":" << perf.n_eval
        << ",\"pp_ms\":" << perf.t_p_eval_ms << ",\"tg_ms\":" << perf.t_eval_ms
        << ",\"pp_tps\":" << (perf.t_p_eval_ms > 0 ? perf.n_p_eval*1000.0/perf.t_p_eval_ms : 0)
        << ",\"tg_tps\":" << (perf.t_eval_ms > 0 ? perf.n_eval*1000.0/perf.t_eval_ms : 0) << ",\"request_wall_ms\":" << wall_ms
        << ",\"expert_h2d_bytes\":" << transport.bytes-before.transport.bytes+cache.transport.bytes-before.cache.transport.bytes
        << ",\"expert_h2d_calls\":" << transport.calls-before.transport.calls+cache.transport.calls-before.cache.transport.calls
        << ",\"expert_cpu_copy_ns\":" << transport.host_copy_wall_ns-before.transport.host_copy_wall_ns+cache.transport.host_copy_wall_ns-before.cache.transport.host_copy_wall_ns
        << ",\"directory_h2d_bytes\":" << cache.directory_h2d_bytes-before.cache.directory_h2d_bytes
        << ",\"host_miss_services\":" << cache.host_services-before.cache.host_services
        << ",\"history_d2h_bytes\":" << cache.dma_costs.frequency_d2h_bytes-before.cache.dma_costs.frequency_d2h_bytes
        << ",\"pin_bytes\":" << runtime.source_pin_metrics().registered_bytes
        << ",\"device_counters\":[";
    for (size_t i = 0; i < cache.gpu.size(); ++i) { if (i) report << ','; report << cache.gpu[i]-before.cache.gpu[i]; }
    report << "],\"api_worst_spans_context_cumulative\":true,\"api_costs\":";
    write_cache_api_costs(report, prefix_cost_delta(cache.dma_costs, before.cache.dma_costs)); report << '}';
}
inline int prefix_live_pair(Runtime& runtime, PrefixReference& reference,
        const std::filesystem::path& output, std::ostream& report, std::string& phase, int count) {
    if (count < 1 || count > 33) throw std::runtime_error("numerical oracle is a bounded diagnostic, not a long generation gate");
    reference.logits.clear(); reference.logits.reserve(count);
    bool failed = false;
    for (int mode = 0; mode < 2; ++mode) {
        if (mode) {
            // Preserve backend/allocator identity for the numerical oracle.
            // Canonical CPU directory planning retains the same cache and
            // frequency authority; placement cannot change model mathematics.
            enable_device_expert_cache(runtime.context(), false);
        }
        phase = std::string("diagnostic_") +
            (mode ? "canonical_cpu_directory_against_live_candidate" : "live_device_frequency_reference");
        const PrefixMetricsStart before{runtime.expert_transport_metrics(), device_expert_cache_metrics(runtime.context())};
        const auto start = std::chrono::steady_clock::now();
        PrefixDifference difference;
        std::string failure;
        try { difference = prefix_case(runtime, reference, output, phase, false, count, mode == 0, nullptr, mode == 0); }
        catch (const std::exception& caught) { failure = caught.what(); }
        const double wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-start).count();
        if (mode) report << ',';
        report << "{\"name\":" << session_json(phase) << ",\"difference\":";
        prefix_write_difference(report, difference); prefix_live_metrics(runtime, before, wall_ms, report);
        if (!failure.empty()) report << ",\"failure\":" << session_json(failure);
        report << '}'; report.flush();
        if (!failure.empty()) {
            report << "],\"requested_outputs\":" << count << ",\"paired_correctness_exact\":false,\"performance_target_met\":false}\n";
            return 2;
        }
        failed |= difference.step >= 0 || difference.token_step >= 0;
        if (!mode && reference.logits.size() != size_t(count)) throw std::runtime_error("live reference incomplete");
    }
    report << "],\"input_tokens\":8192,\"generated_outputs\":" << count << ",\"compared_logits_per_output\":129280,\"live_reference_binary_is_candidate_binary\":true"
        << ",\"canonical_source_uses_same_retained_pin_budget\":true,\"canonical_source_retains_same_context_cache_frequency\":true"
        << ",\"diagnostic_checkpoint_work_in_request_wall\":true,\"paired_correctness_exact\":"
        << (failed ? "false" : "true") << ",\"equality_is_diagnostic_only\":true,\"fully_tuned_bare_llama_comparison\":false,\"performance_target_met\":false}\n";
    return 0;
}
}
