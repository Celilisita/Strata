#include "session_gate.hpp"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace strata::deepseek_v4 {
bool session_long_gate(Runtime& runtime, const Options& options, const std::filesystem::path& prompt,
    const std::filesystem::path& output, uint64_t source_pin_mib, bool frequency) {
    if (options.n_ctx < 10240 || options.n_batch < 8192 || !source_pin_mib)
        throw std::runtime_error("long gate requires sufficient context/batch and the validated source provider");
    if (std::filesystem::file_size(prompt) > 1024 * 1024) throw std::runtime_error("long fixture exceeds1MiB");
    std::ifstream text(prompt, std::ios::binary);
    std::ostringstream content;
    content << text.rdbuf();
    if (!text || content.str().empty()) throw std::runtime_error("cannot read long fixture");
    auto input = runtime.tokenize(content.str());
    if (input.size() < 8192) throw std::runtime_error("long fixture must supply8192actual input tokens");
    // Same deterministic prefix plus final256-token selection as the frozen control.
    if (input.size() > 8192) std::copy(input.end() - 256, input.end(), input.begin() + 7936);
    input.resize(8192);
    std::ofstream report(output / "long-report.json");
    report << std::setprecision(16) << "{\"baseline_provider\":\"canonical_pinned_cpu_directory\",\"fully_tuned_bare_llama_comparison\":false,\"performance_target_met\":false,\"input_ids\":[";
    for (size_t i = 0; i < input.size(); ++i) { if (i) report << ','; report << input[i]; }
    report << "],\"requests\":[";
    std::vector<SessionRequest> control, candidate;
    for (int mode = 0; mode < 2; ++mode) {
        {
            std::ofstream progress(output / "stage-progress.json");
            progress << "{\"phase\":\"long_" << (mode ? "candidate" : "control") << "\",\"context_setup_started\":true}\n";
        }
        session_set_device_cache(mode != 0);
        session_set_frequency(mode != 0 && frequency);
        session_set_source_pin(source_pin_mib);
        const auto start = std::chrono::steady_clock::now();
        std::string error;
        if (!runtime.reconfigure_context(options, error)) throw std::runtime_error(error);
        const double setup_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        const auto pin = runtime.source_pin_metrics();
        for (int request = 0; request < 2; ++request) {
            const auto result = session_request_count(runtime, input, output / "long-request-failure.json", 2048);
            if (mode || request) report << ',';
            report << "{\"candidate\":" << (mode ? "true" : "false") << ",\"measured\":" << (request ? "true" : "false")
                   << ",\"context_setup_ms\":" << setup_ms << ",\"source_registered_bytes\":" << pin.registered_bytes
                   << ",\"source_registration_ns_cumulative\":" << pin.registration_wall_ns << ",\"data\":";
            session_write_request(report, result);
            report << '}';
            report.flush();
            if (result.output.size() != 2048 || result.pp_tokens != 8192 || result.tg_tokens != 2047)
                throw std::runtime_error("long request did not complete the fixed workload");
            std::ofstream generated(output / (std::string("long-") + (mode ? "candidate-" : "control-") + std::to_string(request) + ".txt"), std::ios::binary);
            generated << runtime.detokenize(result.output);
            if (!generated) throw std::runtime_error("long generated text write failed");
            if (mode) candidate.push_back(result); else control.push_back(result);
            std::ofstream progress(output / "long-progress.json");
            progress << "{\"phase\":\"" << (mode ? "candidate" : "control") << "\",\"completed_request\":" << request
                     << ",\"pp_tps\":" << result.pp_tokens * 1000.0 / result.pp_ms
                     << ",\"tg_tps\":" << result.tg_tokens * 1000.0 / result.tg_ms << ",\"generated\":2048}\n";
        }
    }
    const double pp = control[1].pp_ms / candidate[1].pp_ms, tg = control[1].tg_ms / candidate[1].tg_ms;
    report << "],\"finite_fixed_workload_passed\":true,\"quality_assessed\":false,\"output_id_equality_required\":false,\"independent_autoregressive_generation\":true,\"pp_ratio\":" << pp << ",\"tg_ratio\":" << tg
           << ",\"pinned_directory_comparison_2x\":" << (pp >= 2 && tg >= 2 ? "true" : "false") << '}';
    if (!report) throw std::runtime_error("long report write failed");
    return pp >= 2 && tg >= 2;
}
}
