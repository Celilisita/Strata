#pragma once
#include "long_prefix_dispatch.hpp"
#include <fstream>
#include <memory>
namespace strata::deepseek_v4 {
inline int prefix_dependency_cases(Runtime& runtime, std::unique_ptr<PrefixDispatchSnapshots>& snapshots,
        const Options& options, PrefixReference& reference, const std::filesystem::path& output,
        std::ostream& report, std::string& phase) {
    phase = "uninstrumented_bounded_staging";
    const auto control = prefix_case(runtime, reference, output, phase, false, 206, true);
    report << "{\"name\":" << session_json(phase) << ",\"difference_from_frozen_ids\":";
    prefix_write_difference(report, control); report << '}'; report.flush();
    session_set_source_pin(131072); session_set_device_cache(true); session_set_frequency(true);
    const char* names[]{"no_gpu_checkpoints", "boundary_checkpoints", "ffn37_checkpoints", "attention37_checkpoints", "full_checkpoints"};
    const PrefixSnapshotScope scopes[]{PrefixSnapshotScope::Boundary, PrefixSnapshotScope::Boundary,
        PrefixSnapshotScope::Ffn37, PrefixSnapshotScope::Attention37, PrefixSnapshotScope::Full};
    bool failed = control.token_step >= 0;
    std::string error;
    for (int case_index = 0; case_index < 5; ++case_index) {
        phase = names[case_index];
        if (!runtime.reconfigure_context(options, error)) throw std::runtime_error(error);
        if (case_index) {
            if (!snapshots) snapshots = std::make_unique<PrefixDispatchSnapshots>(runtime);
            snapshots->set_scope(scopes[case_index]);
        }
        const auto difference = prefix_case(runtime, reference, output, phase, false, 1, false, case_index ? snapshots.get() : nullptr);
        if (case_index) snapshots->save(output / (phase + "-boundaries.f32"));
        report << ",{\"name\":" << session_json(phase) << ",\"compared_full_f32_logits\":129280,\"difference\":";
        prefix_write_difference(report, difference); report << '}'; report.flush();
        failed |= difference.step >= 0;
        { std::ofstream progress(output / "progress.json"); progress << "{\"completed_case\":" << session_json(phase) << '}'; }
    }
    report << "],\"diagnostic_only\":true,\"snapshot_read_at\":\"prefill_end_before_tg\",\"performance_target_met\":false}\n";
    return failed ? 2 : 0;
}
}
