#include "long_prefix_compare.hpp"
#include "long_prefix_layers.hpp"
#include "long_prefix_dispatch.hpp"
#include "prefix_dependency_cases.hpp"
#include "attention_repeat_gate.hpp"
#include "prefix_live_pair.hpp"
#include "prefix_free_pair.hpp"
#include "native_tg_graph.hpp"
#include "pp_stream_graph.hpp"
#include "benchmark_fault_guard.hpp"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
using namespace strata::deepseek_v4;
int main(int argc, char** argv) {
    benchmark_fault_guard();
    std::filesystem::path output;
    std::string phase = "initialization";
    try {
        Options options;
        std::vector<std::string> extra;
        std::string error;
        if (!parse_options(argc, argv, options, extra, error)) throw std::runtime_error(error);
        std::filesystem::path fixture;
        bool candidate_only = false, bounded_baseline = false, prefill_only = false, dispatch_snapshots = false, skip_layer_replay = false;
        int prefix_limit = 2048;
        bool dependency_brackets = false, attention_repeat = false, live_pair = false, free_pair = false, free_candidate = false;
        for (size_t i = 0; i < extra.size(); ++i) {
            if (i + 1 == extra.size()) throw std::runtime_error("prefix option lacks value");
            if (extra[i] == "--output-dir") output = extra[++i];
            else if (extra[i] == "--reference-fixture") fixture = extra[++i];
            else if (extra[i] == "--candidate-only" && (extra[i+1] == "0" || extra[i+1] == "1")) candidate_only = extra[++i] == "1";
            else if (extra[i] == "--bounded-baseline" && (extra[i+1] == "0" || extra[i+1] == "1")) bounded_baseline = extra[++i] == "1";
            else if (extra[i] == "--prefill-only" && (extra[i+1] == "0" || extra[i+1] == "1")) prefill_only = extra[++i] == "1";
            else if (extra[i] == "--dispatch-snapshots" && (extra[i+1] == "0" || extra[i+1] == "1")) dispatch_snapshots = extra[++i] == "1";
            else if (extra[i] == "--skip-layer-replay" && (extra[i+1] == "0" || extra[i+1] == "1")) skip_layer_replay = extra[++i] == "1";
            else if (extra[i] == "--dependency-brackets" && (extra[i+1] == "0" || extra[i+1] == "1")) dependency_brackets = extra[++i] == "1";
            else if (extra[i] == "--attention-repeat" && (extra[i+1] == "0" || extra[i+1] == "1")) attention_repeat = extra[++i] == "1";
            else if (extra[i] == "--live-pair" && (extra[i+1] == "0" || extra[i+1] == "1")) live_pair = extra[++i] == "1";
            else if (extra[i] == "--free-pair" && (extra[i+1] == "0" || extra[i+1] == "1")) free_pair = extra[++i] == "1";
            else if (extra[i] == "--free-candidate" && (extra[i+1] == "0" || extra[i+1] == "1")) free_candidate = extra[++i] == "1";
            else if (extra[i] == "--prefix-limit") prefix_limit = std::stoi(extra[++i]);
            else throw std::runtime_error("unknown prefix option");
        }
        free_pair |= free_candidate;
        if (prefix_limit < 1 || prefix_limit > 2048) throw std::runtime_error("prefix limit outside1..2048");
        if (!free_pair && prefix_limit > 33) throw std::runtime_error("numerical diagnostics are bounded to33 outputs; use --free-pair1 for the normal8192/2048 workload");
        if (free_pair && (prefix_limit != 2048 || candidate_only || bounded_baseline || dependency_brackets || attention_repeat || dispatch_snapshots || prefill_only || live_pair))
            throw std::runtime_error("free pair requires the complete2048-output workload and excludes numerical diagnostics");
        if (output.empty() || fixture.empty() || !options.pinned_experts || options.n_ctx < 10240 ||
            options.n_batch < 8192 || options.n_ubatch != 8192) throw std::runtime_error("invalid prefix configuration");
        std::filesystem::create_directories(output);
        if (!std::filesystem::is_empty(output)) throw std::runtime_error("prefix output exists");
        auto reference = prefix_read_reference(fixture);
        configure_device_cache_process();
        if (candidate_only && bounded_baseline) throw std::runtime_error("prefix modes are exclusive");
        if (prefill_only && !bounded_baseline) throw std::runtime_error("prefill attribution requires bounded baseline");
        if (dispatch_snapshots && !bounded_baseline) throw std::runtime_error("dispatch snapshots require bounded baseline");
        if (live_pair && (candidate_only || bounded_baseline || dependency_brackets || attention_repeat || dispatch_snapshots || prefill_only))
            throw std::runtime_error("live pair modes are exclusive");
        session_set_private_source(!packed_experts_requested()); session_set_source_pin(candidate_only || live_pair || free_pair ? 131072 : 0);
        session_set_device_cache(candidate_only || live_pair || free_candidate);
        session_set_frequency(candidate_only || live_pair || free_candidate);
        const bool native_pp_candidate = pp_stream_enabled(), native_tg_candidate = native_tg_enabled();
        // Baseline context must be built with its full normal resident budget.
        // Candidate-only flags are applied before constructing its new context.
        if (free_pair && !free_candidate) session_set_native(false, false);
        if (free_candidate && (!native_pp_candidate || !native_tg_candidate))
            throw std::runtime_error("free candidate requires native PP and TG");
        std::unique_ptr<PrefixDispatchSnapshots> snapshots; // Outlives runtime and its captured graphs.
        std::unique_ptr<AttentionRepeatReceipt> attention_receipt;
        const auto start = std::chrono::steady_clock::now();
        auto runtime = Runtime::load(options, error);
        if (!runtime) throw std::runtime_error(error);
        const double load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        std::ofstream report(output / "report.json");
        report << std::setprecision(16) << "{\"model_loads\":1,\"load_ms\":" << load_ms << ",\"cases\":[";
        if (free_pair) return prefix_free_pair(*runtime, options, reference, output, report, phase,
            native_pp_candidate, native_tg_candidate, free_candidate);
        if (live_pair) {
            return prefix_live_pair(*runtime, reference, output, report, phase, prefix_limit);
        }
        if (attention_repeat) {
            if (!bounded_baseline || dependency_brackets || dispatch_snapshots || prefill_only) throw std::runtime_error("invalid attention repeat mode");
            return prefix_attention_repeat_cases(*runtime, attention_receipt, options, reference, output, report, phase);
        }
        if (dependency_brackets) {
            if (!bounded_baseline || dispatch_snapshots || prefill_only) throw std::runtime_error("invalid dependency bracket mode");
            return prefix_dependency_cases(*runtime, snapshots, options, reference, output, report, phase);
        }
        if (bounded_baseline) {
            if (dispatch_snapshots) snapshots = std::make_unique<PrefixDispatchSnapshots>(*runtime);
            phase = prefill_only ? "bounded_staging_prefill" : "bounded_staging_until_step205";
            const auto control = prefix_case(*runtime, reference, output, phase, false, prefill_only ? 1 : prefix_limit, true, snapshots.get());
            const auto layer_control = snapshots ? snapshots->read() : std::vector<float>{};
            report << "{\"name\":" << session_json(phase) << ",\"difference_from_frozen_ids\":";
            prefix_write_difference(report, control); report << '}'; report.flush();
            phase = "device_frequency_against_bounded_staging";
            session_set_source_pin(131072); session_set_device_cache(true); session_set_frequency(true);
            if (!runtime->reconfigure_context(options, error)) throw std::runtime_error(error);
            const auto difference = prefix_case(*runtime, reference, output, phase, false, int(reference.logits.size()), false, snapshots.get());
            report << ",{\"name\":" << session_json(phase)
                   << ",\"current_path\":\"device_frequency\",\"golden_path\":\"bounded_staging\",\"difference\":";
            prefix_write_difference(report, difference); report << "}],\"performance_target_met\":false}\n"; report.flush();
            if (snapshots) prefix_write_dispatch_difference(layer_control, snapshots->read(), output, *snapshots);
            else if (difference.step >= 0 && !skip_layer_replay) {
                phase = "layer_attribution";
                prefix_compare_layers(*runtime, options, reference, output, difference.step);
            }
            return difference.step >= 0 || control.token_step >= 0 ? 2 : 0;
        }
        if (candidate_only) {
            // At most1GiB of CPU logits, retained only for this diagnostic comparison.
            phase = "candidate_until_first_token_difference";
            const auto candidate = prefix_case(*runtime, reference, output, phase, false, prefix_limit, true);
            report << "{\"name\":" << session_json(phase) << ",\"difference\":";
            prefix_write_difference(report, candidate); report << '}'; report.flush();
            const int count = candidate.token_step >= 0 ? candidate.token_step + 1 : prefix_limit;
            phase = "source_control_against_captured_candidate";
            session_set_device_cache(false); session_set_frequency(false);
            if (!runtime->reconfigure_context(options, error)) throw std::runtime_error(error);
            const auto difference = prefix_case(*runtime, reference, output, phase, false, count);
            report << ",{\"name\":" << session_json(phase)
                   << ",\"current_path\":\"canonical_source_only\",\"golden_path\":\"captured_device_frequency\",\"difference\":";
            prefix_write_difference(report, difference);
            report << "}],\"performance_target_met\":false}\n";
            return candidate.token_step >= 0 || difference.step >= 0 ? 2 : 0;
        }
        const char* names[]{"control", "source_only", "device_cache", "device_frequency"};
        bool failed = false;
        for (int mode = 0; mode < 4; ++mode) {
            phase = names[mode];
            { std::ofstream progress(output / "progress.json"); progress << "{\"phase\":" << session_json(phase) << '}'; }
            const auto setup = std::chrono::steady_clock::now();
            if (mode) {
                session_set_source_pin(131072);
                session_set_device_cache(mode >= 2); session_set_frequency(mode == 3);
                if (!runtime->reconfigure_context(options, error)) throw std::runtime_error(error);
            }
            const double setup_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - setup).count();
            const auto difference = prefix_case(*runtime, reference, output, phase, mode == 0, 33);
            if (mode) report << ',';
            const auto perf = runtime->performance();
            const auto pin = runtime->source_pin_metrics();
            report << "{\"name\":" << session_json(phase) << ",\"setup_ms\":" << setup_ms
                   << ",\"pin_bytes\":" << pin.registered_bytes << ",\"registration_ns_cumulative\":" << pin.registration_wall_ns
                   << ",\"pp_ms\":" << perf.t_p_eval_ms << ",\"tg_ms\":" << perf.t_eval_ms << ",\"difference\":";
            prefix_write_difference(report, difference); report << '}'; report.flush();
            failed |= difference.step >= 0 || difference.token_step >= 0;
        }
        report << "],\"prefix_exact\":" << (failed ? "false" : "true") << ",\"performance_target_met\":false}\n";
        if (!report) throw std::runtime_error("prefix report write failed");
        return failed ? 2 : 0;
    } catch (const std::exception& failure) {
        if (!output.empty()) {
            std::ofstream saved(output / "failure.json");
            saved << "{\"phase\":" << session_json(phase) << ",\"error\":" << session_json(failure.what()) << '}';
        }
        std::cerr << failure.what() << '\n'; return 1;
    }
}
