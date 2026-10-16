#include "session_gate.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

using namespace strata::deepseek_v4;
int main(int argc, char** argv) {
    std::filesystem::path output, prompt, long_prompt;
    std::ofstream report;
    std::unique_ptr<Runtime> runtime;
    Options diagnosis_options;
    std::vector<llama_token> diagnosis_input;
    bool run_layers = true, short_started = false, diagnose_failure = false, candidate_frequency = false;
    uint64_t source_pin_mib = 0;
    bool run_allhit = false;
    std::string phase = "initialization";
    try {
        Options options;
        std::vector<std::string> extra;
        std::string error;
        if (!parse_options(argc, argv, options, extra, error)) throw std::runtime_error(error);
        for (size_t i = 0; i < extra.size(); ++i) {
            if (i + 1 == extra.size()) throw std::runtime_error("suite option lacks its value");
            if (extra[i] == "--output-dir") output = extra[++i];
            else if (extra[i] == "--prompt-file") prompt = extra[++i];
            else if (extra[i] == "--long-prompt-file") long_prompt = extra[++i];
            else if (extra[i] == "--source-pin-mib") {
                const auto value = extra[++i];
                if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
                    throw std::runtime_error("invalid source pin budget");
                source_pin_mib = std::stoull(value);
                if (source_pin_mib > 131072) throw std::runtime_error("source pin budget exceeds131072MiB");
            }
            else if (extra[i] == "--allhit" && (extra[i+1] == "0" || extra[i+1] == "1")) run_allhit = extra[++i] == "1";
            else if (extra[i] == "--layers" && (extra[i+1] == "0" || extra[i+1] == "1")) run_layers = extra[++i] == "1";
            else if (extra[i] == "--diagnose" && (extra[i+1] == "0" || extra[i+1] == "1")) diagnose_failure = extra[++i] == "1";
            else if (extra[i] == "--frequency" && (extra[i+1] == "0" || extra[i+1] == "1")) candidate_frequency = extra[++i] == "1";
            else throw std::runtime_error("unknown suite option: " + extra[i]);
        }
        if (output.empty() || prompt.empty() || !device_expert_cache_requested())
            throw std::runtime_error("suite requires output, prompt and the opt-in device cache");
        std::filesystem::create_directories(output);
        if (!std::filesystem::is_empty(output)) throw std::runtime_error("suite output must be empty");
        if (std::filesystem::file_size(prompt) > 1024 * 1024) throw std::runtime_error("short suite input exceeds1 MiB");
        std::ifstream text(prompt, std::ios::binary);
        std::ostringstream content;
        content << text.rdbuf();
        if (!text || content.str().empty()) throw std::runtime_error("cannot read suite prompt");
        options.prompt = content.str();
        const Options short_options = options;
        diagnosis_options = short_options;
        if (source_pin_mib && !options.pinned_experts) throw std::runtime_error("source pin requires bounded expert provider");
        if (!long_prompt.empty() && (!source_pin_mib || !run_allhit))
            throw std::runtime_error("long gate requires source pin and pure allhit gates");
        session_set_private_source(source_pin_mib != 0);
        session_set_source_pin(run_layers ? 0 : source_pin_mib);
        configure_device_cache_process();
        session_set_device_cache(run_layers);
        session_set_frequency(run_layers && candidate_frequency);
        if (run_layers) options.moe_cache_mib = 96;
        SessionGateCallback observer;
        const auto started = std::chrono::steady_clock::now();
        runtime = Runtime::load(options, error, run_layers ? SessionGateCallback::evaluate : nullptr,
                                run_layers ? &observer : nullptr);
        if (!runtime) throw std::runtime_error(error);
        const double load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        const auto input = runtime->tokenize(options.prompt);
        diagnosis_input = input;
        if (input.size() != 302) throw std::runtime_error("suite prompt must tokenize to the frozen302-token fixture");
        report.open(output / "report.json");
        report << std::setprecision(16) << "{\"model_loads\":1,\"load_ms\":" << load_ms
               << ",\"input_ids\":[";
        for (size_t i = 0; i < input.size(); ++i) { if (i) report << ','; report << input[i]; }
        report << "],\"layers\":[";
        phase = "layer_gates";
        const auto layers = run_layers ? session_layer_gates(*runtime, observer, input, output) : std::vector<LayerGateResult>{};
        for (size_t i = 0; i < layers.size(); ++i) {
            if (i) report << ',';
            session_write_layer(report, layers[i]);
        }
        report << "],\"requests\":[";
        std::vector<SessionRequest> control, candidate;
        std::vector<AllHitResult> allhit;
        std::vector<double> setup_ms;
        std::vector<SourcePinMetrics> pins;
        short_started = true;
        for (int mode = 0; mode < 2; ++mode) {
            phase = mode ? "candidate_short" : "control_short";
            session_set_device_cache(mode != 0);
            session_set_frequency(mode != 0 && candidate_frequency);
            {
                std::ofstream progress(output / "stage-progress.json");
                progress << "{\"phase\":" << session_json(phase) << ",\"context_setup_started\":true}\n";
            }
            session_set_source_pin(source_pin_mib);
            const auto setup_start = std::chrono::steady_clock::now();
            // Release one context before allocating the next; retain all model weights.
            if ((run_layers || mode) && !runtime->reconfigure_context(short_options, error)) throw std::runtime_error(error);
            setup_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - setup_start).count());
            pins.push_back(runtime->source_pin_metrics());
            for (int request = 0; request < 2; ++request) {
                const auto result = session_request(*runtime, input, output / "request-failure.json");
                if (mode == 0) control.push_back(result);
                if (mode) candidate.push_back(result);
                if (mode || request) report << ',';
                report << "{\"candidate\":" << (mode ? "true" : "false")
                       << ",\"measured\":" << (request ? "true" : "false") << ",\"data\":";
                session_write_request(report, result);
                report << '}';
                report.flush();
            }
            if (run_allhit) {
                phase = mode ? "candidate_pure_allhit" : "control_pure_allhit";
                allhit.push_back(session_all_hit(*runtime, mode != 0));
            }
            if (mode == 0) {
                std::ofstream saved(output / "control-receipt.json");
                saved << "{\"model_loads\":1,\"load_ms\":" << load_ms << ",\"input_ids\":[";
                for (size_t i = 0; i < input.size(); ++i) { if (i) saved << ','; saved << input[i]; }
                saved << "],\"requests\":[";
                for (size_t i = 0; i < control.size(); ++i) {
                    if (i) saved << ',';
                    session_write_request(saved, control[i]);
                }
                saved << "],\"status\":\"passed\"}\n";
            }
        }
        const double pp_ratio = (candidate[1].pp_tokens / candidate[1].pp_ms) / (control[1].pp_tokens / control[1].pp_ms);
        const double tg_ratio = (candidate[1].tg_tokens / candidate[1].tg_ms) / (control[1].tg_tokens / control[1].tg_ms);
        const double allhit_ratio = run_allhit ? allhit[0].median_ms / allhit[1].median_ms : 1;
        const bool performance_passed = pp_ratio >= 0.95 && tg_ratio >= 0.95 && allhit_ratio >= 0.95;
        report << "],\"finite_fixed_workload_passed\":true,\"output_id_equality_required\":false,\"quality_assessed\":false,\"output_ids_equal_diagnostic\":"
               << (control[1].output == candidate[1].output ? "true" : "false") << ",\"pp_ratio\":" << pp_ratio
               << ",\"tg_ratio\":" << tg_ratio << ",\"performance_gate_passed\":" << (performance_passed ? "true" : "false")
               << ",\"status\":" << session_json(performance_passed ? "passed" : "cost_regression")
               << ",\"performance_target_met\":false,\"source_pin_mib\":" << source_pin_mib
               << ",\"context_setup_ms\":[" << setup_ms[0] << ',' << setup_ms[1] << "],\"source_registration\":[";
        for (size_t i = 0; i < pins.size(); ++i) {
            if (i) report << ',';
            const auto& p = pins[i];
            report << "{\"bytes\":" << p.registered_bytes << ",\"attempts\":" << p.attempts
                   << ",\"failures\":" << p.failures << ",\"registration_ns\":" << p.registration_wall_ns << '}';
        }
        report << "],\"pure_allhit_ratio\":" << allhit_ratio << ",\"pure_allhit\":[";
        for (size_t i = 0; i < allhit.size(); ++i) { if (i) report << ','; session_write_all_hit(report, allhit[i]); }
        report << "]}\n";
        report.close();
        if (performance_passed && !long_prompt.empty()) {
            phase = "long_gate";
            session_long_gate(*runtime, short_options, long_prompt, output, source_pin_mib, candidate_frequency);
        }
        std::cout << "session suite passed; one model load; inspect structured report\n";
        return performance_passed ? 0 : 2;
    } catch (const std::exception& failure) {
        // Failure receipt is separate from any partially written progress report.
        if (!output.empty()) {
            std::ofstream failed(output / "failure.json");
            failed << "{\"status\":\"failed\",\"phase\":" << session_json(phase)
                   << ",\"error\":" << session_json(failure.what()) << "}\n";
        }
        if (diagnose_failure && short_started && runtime) session_diagnose(*runtime, diagnosis_options, diagnosis_input, output);
        std::cerr << failure.what() << '\n';
        return 1;
    }
}
