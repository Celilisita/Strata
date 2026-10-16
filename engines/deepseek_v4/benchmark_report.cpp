#include "benchmark_report.hpp"
#include "cold_metrics_report.hpp"
#ifdef STRATA_DEEPSEEK_CUDA_PROFILER
#include "graph_diag.hpp"
#endif

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace strata::deepseek_v4 {
namespace {

void token_file(const std::filesystem::path& path, const std::vector<int32_t>& values) {
    std::ofstream output(path);
    for (int32_t value : values) output << value << '\n';
    if (!output) throw std::runtime_error("cannot write token IDs");
}

std::vector<int32_t> read_tokens(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot read token IDs");
    std::vector<int32_t> result;
    int32_t value;
    while (input >> value) {
        if (result.size() >= 16384 || value < 0) throw std::runtime_error("invalid or oversized token IDs");
        result.push_back(value);
    }
    if (!input.eof()) throw std::runtime_error("malformed token IDs");
    return result;
}

std::string read_text(const std::filesystem::path& path) {
    if (std::filesystem::file_size(path) > (4u << 20)) throw std::runtime_error("oversized identity file");
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot read identity file");
    std::ostringstream value;
    value << input.rdbuf();
    return value.str();
}

void text_file(const std::filesystem::path& path, const std::string& text) {
    std::ofstream output(path, std::ios::binary);
    output << text;
    if (!output) throw std::runtime_error("cannot write benchmark artifact");
}

void tokens_json(std::ostream& output, const std::vector<int32_t>& values) {
    output << '[';
    for (std::size_t i = 0; i < values.size(); ++i) { if (i) output << ','; output << values[i]; }
    output << ']';
}

double rate(int32_t count, double milliseconds) {
    return count > 0 && milliseconds > 0 ? 1000.0 * count / milliseconds : -1;
}

void optional_number(std::ostream& output, double value) {
    if (value < 0) output << "null"; else output << value;
}

std::string environment_identity(const char* name) {
    const char* value = std::getenv(name);
    if (value && std::strlen(value) > 128)
        throw std::runtime_error(std::string(name) + " identity exceeds 128 bytes");
    return value ? json_string(value) : "null";
}

void request_tokens_file(const std::filesystem::path& path, const std::vector<RequestReport>& requests) {
    std::ofstream output(path, std::ios::binary);
    output << "index\tmeasured\tstate\tstatus\tinput_ids\toutput_ids\n";
    auto ids = [&](const std::vector<int32_t>& values) {
        for (std::size_t i = 0; i < values.size(); ++i) { if (i) output << ','; output << values[i]; }
    };
    for (const auto& item : requests) {
        output << item.index << '\t' << (item.measured ? 1 : 0) << '\t' << item.state << '\t' << item.status << '\t';
        ids(item.input_ids); output << '\t'; ids(item.output_ids); output << '\n';
    }
    if (!output) throw std::runtime_error("cannot write per-request token evidence");
}

void request_json(std::ostream& output, const RequestReport& item, bool pinned, bool cold) {
    output << "{\"index\":" << item.index << ",\"measured\":" << (item.measured ? "true" : "false")
           << ",\"state\":" << json_string(item.state) << ",\"status\":" << json_string(item.status)
           << ",\"error\":" << json_string(item.error) << ",\"input_ids\":";
    tokens_json(output, item.input_ids);
    output << ",\"output_ids\":"; tokens_json(output, item.output_ids);
    output << ",\"output_text\":" << json_string(item.output_text)
           << ",\"timing\":{\"request_ms\":" << item.request_ms << ",\"ttft_ms\":";
    optional_number(output, item.ttft_ms);
    output << ",\"prompt_eval_ms\":" << item.prompt_eval_ms << ",\"token_eval_ms\":" << item.token_eval_ms
           << ",\"prompt_eval_tokens\":" << item.prompt_eval_tokens << ",\"token_eval_tokens\":" << item.token_eval_tokens
           << ",\"pp_tokens_per_second\":";
    optional_number(output, rate(item.prompt_eval_tokens, item.prompt_eval_ms));
    output << ",\"tg_tokens_per_second\":";
    optional_number(output, rate(item.token_eval_tokens, item.token_eval_ms));
    output << ",\"graph_reuses\":" << item.graph_reuses << "},\"expert_transport_delta\":{\"weight_h2d_bytes\":";
    if (pinned) output << item.expert_transport_delta.bytes; else output << "null";
    output << ",\"submitted_chunks\":";
    if (pinned) output << item.expert_transport_delta.calls; else output << "null";
    output << ",\"slab_reuse_waits\":";
    if (pinned) output << item.expert_transport_delta.slab_waits; else output << "null";
    output << ",\"host_copy_wall_ns\":";
    if (pinned) output << item.expert_transport_delta.host_copy_wall_ns; else output << "null";
    output << ",\"slab_wait_wall_ns\":";
    if (pinned) output << item.expert_transport_delta.slab_wait_wall_ns; else output << "null";
    output << ",\"direct_h2d_bytes\":";
    if (pinned) output << item.expert_transport_delta.direct_bytes; else output << "null";
    output << ",\"direct_h2d_calls\":";
    if (pinned) output << item.expert_transport_delta.direct_calls; else output << "null";
    output << "},\"expert_transport_by_phase\":{\"pp\":";
    auto transport = [&](const ExpertStagingMetrics& value) {
        if (!pinned || !item.transport_phases_complete) { output << "null"; return; }
        output << "{\"weight_h2d_bytes\":" << value.bytes
               << ",\"submitted_chunks\":" << value.calls
               << ",\"slab_reuse_waits\":" << value.slab_waits
               << ",\"host_copy_wall_ns\":" << value.host_copy_wall_ns
               << ",\"slab_wait_wall_ns\":" << value.slab_wait_wall_ns
               << ",\"direct_h2d_bytes\":" << value.direct_bytes
               << ",\"direct_h2d_calls\":" << value.direct_calls << '}';
    };
    transport(item.prefill_transport);
    output << ",\"tg\":"; transport(item.decode_transport);
    output << "},\"cold_experts_delta\":";
    write_cold_metrics(output, item.cold_delta, cold);
    output << '}';
}

} // namespace

std::string json_string(const std::string& value) {
    std::ostringstream output;
    output << '"';
    for (unsigned char character : value) {
        switch (character) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (character < 32) output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                                           << (int) character << std::dec;
                else output << character;
        }
    }
    output << '"';
    return output.str();
}

void write_report(const std::filesystem::path& directory, const BenchmarkReport& report,
                  const NativeMetrics* metrics) {
    const std::string offload_identity = environment_identity("GGML_OP_OFFLOAD_MIN_BATCH");
    const std::string mmq_identity = environment_identity("GGML_CUDA_MMQ_PREC");
    const std::string pp_bank_identity = environment_identity("STRATA_DSV4_PP_CACHE_BANKS");
    const std::string source_pin_identity = environment_identity("STRATA_DSV4_SOURCE_PIN_MIB");
    const std::string tg_capture_identity = environment_identity("STRATA_DSV4_PROFILE_TG_TOKENS");
    std::filesystem::create_directories(directory);
#ifdef STRATA_DEEPSEEK_CUDA_PROFILER
    write_graph_diagnostic(directory);
#endif
    token_file(directory / "input-ids.txt", report.input_ids);
    token_file(directory / "output-ids.txt", report.output_ids);
    text_file(directory / "workload.txt", report.workload_identity +
              "GGML_OP_OFFLOAD_MIN_BATCH=" + offload_identity + "\n" +
              "GGML_CUDA_MMQ_PREC=" + mmq_identity + "\n");
    text_file(directory / "output.txt", report.output_text);
    text_file(directory / "status.txt", report.status + "\n" + report.mode + "\n");
    if (!report.requests.empty()) request_tokens_file(directory / "request-tokens.tsv", report.requests);
    if (!report.priming_requests.empty()) request_tokens_file(directory / "cache-prime-tokens.tsv", report.priming_requests);
    std::ofstream output(directory / "report.json");
    output << std::setprecision(12) << "{\n  \"schema\":\"strata-deepseek-v4-benchmark-v1\",\n"
           << "  \"engine\":\"Strata in-process canonical DeepSeek4\",\n"
           << "  \"mode\":" << json_string(report.mode) << ",\n  \"status\":" << json_string(report.status)
           << ",\n  \"error\":" << json_string(report.error)
           << ",\n  \"model\":" << json_string(report.model)
           << ",\n  \"device\":" << json_string(report.device)
           << ",\n  \"canonical_commit\":" << json_string(report.canonical_commit)
           << ",\n  \"architecture\":" << json_string(report.architecture)
           << ",\n  \"cache_mib\":" << report.cache_mib
           << ",\n  \"threadpool_resident\":" << (report.threadpool_resident ? "true" : "false")
           << ",\n  \"cold_threads\":" << report.cold_threads
           << ",\n  \"cold_admit_free\":" << (report.cold_admit_free ? "true" : "false")
           << ",\n  \"cache_leaf_aliases\":" << (report.cache_leaf_aliases ? "true" : "false")
           << ",\n  \"layer_fusion_seam\":" << (report.layer_fusion_seam ? "true" : "false")
           << ",\n  \"cache_prime\":" << (report.cache_prime ? "true" : "false")
           << ",\n  \"all_hit_cache_lease\":{\"active\":" << (report.all_hit_lease ? "true" : "false")
           << ",\"production_ready\":false,\"scope\":\"experimental frozen canonical-cache snapshot; no admission/eviction or cold fallback; missing routes poison standard logits and fail before accepting a token; applies to batches1..8 only; device-only contract requires trace validation\"}"
           << ",\n  \"cold_experts\":";
    write_cold_metrics(output, report.cold, report.cold_threads > 0);
    output
           << ",\n  \"expert_transport\":{\"pinned\":" << (report.pinned_experts ? "true" : "false")
           << ",\"host_slab_bytes\":" << (report.pinned_experts ? uint64_t(32) << 20 : 0)
           << ",\"weight_h2d_bytes\":";
    if (report.pinned_experts) output << report.expert_transport.bytes; else output << "null";
    output << ",\"submitted_chunks\":";
    if (report.pinned_experts) output << report.expert_transport.calls; else output << "null";
    output << ",\"slab_reuse_waits\":";
    if (report.pinned_experts) output << report.expert_transport.slab_waits; else output << "null";
    output << ",\"host_copy_wall_ns\":";
    if (report.pinned_experts) output << report.expert_transport.host_copy_wall_ns; else output << "null";
    output << ",\"slab_wait_wall_ns\":";
    if (report.pinned_experts) output << report.expert_transport.slab_wait_wall_ns; else output << "null";
    output << ",\"direct_h2d_bytes\":";
    if (report.pinned_experts) output << report.expert_transport.direct_bytes; else output << "null";
    output << ",\"direct_h2d_calls\":";
    if (report.pinned_experts) output << report.expert_transport.direct_calls; else output << "null";
    output << ",\"scope\":\"submitted expert payload including canonical range padding; excludes slot-map/control and activation traffic; slab_reuse_waits counts issued event waits; wall ns measure upload memcpy and slab reuse event waits only, exclude reset/release/destructor, may overlap GPU activity and are not additive critical-path stalls\"}"
           << ",\n  \"expert_source_pin\":{\"budget_bytes\":" << report.source_pin.budget_bytes
           << ",\"registered_bytes\":" << report.source_pin.registered_bytes
           << ",\"attempts\":" << report.source_pin.attempts
           << ",\"failures\":" << report.source_pin.failures
           << ",\"last_error\":" << report.source_pin.last_error
           << ",\"registration_wall_ns\":" << report.source_pin.registration_wall_ns
           << ",\"registration_included_in_load_wall_ms\":true}"
           << ",\n  \"request_policy\":{\"warmup_requests\":" << report.warmup_requests
           << ",\"measure_requests\":" << report.measure_requests
           << ",\"reset\":\"synchronize; clear raw/compressed KV and rolling state/perf; retain model, resident pool, graph state and expert cache; synchronize\""
           << ",\"host_page_state\":\"uncontrolled at the first request; prior identical requests may warm touched pages\""
           << ",\"state\":" << json_string(report.diagnostics ? "single truncated layer" :
                report.requests.empty() ? "no request started" :
                report.warmup_requests ? "same prompt after " + std::to_string(report.warmup_requests) +
                                        " warmup requests; expert cache retained" :
                report.measure_requests > 1 ? "first fresh context followed by identical prompt repeats" :
                                             "fresh context; expert cache initially empty")
           << ",\"performance_scope\":\"repeated identical prompt with KV reset; retained-cache hit rate is not a cold-cache gain\"}"
           << ",\n  \"environment\":{\"GGML_OP_OFFLOAD_MIN_BATCH\":" << offload_identity
           << ",\"GGML_CUDA_MMQ_PREC\":" << mmq_identity
           << ",\"STRATA_DSV4_SOURCE_PIN_MIB\":" << source_pin_identity
           << ",\"STRATA_DSV4_EXPERT_FREQUENCY\":" << environment_identity("STRATA_DSV4_EXPERT_FREQUENCY")
           << ",\"STRATA_DSV4_LAST_LOGIT_ONLY\":" << environment_identity("STRATA_DSV4_LAST_LOGIT_ONLY")
           << ",\"STRATA_DSV4_PP_CACHE_BANKS\":" << pp_bank_identity
           << ",\"STRATA_DSV4_PROFILE_TG_TOKENS\":" << tg_capture_identity << "}"
           << ",\n  \"diagnostics\":" << (report.diagnostics ? "true" : "false")
           << ",\n  \"layer_zero_complete\":" << (report.layer_complete ? "true" : "false")
           << ",\n  \"original_gguf_quantization_preserved\":true,\n"
           << "  \"mxfp4_abi\":{\"elements_per_block\":32,\"bytes_per_block\":17,\"validated\":true},\n"
           << "  \"quantization_identity_note\":\"Actual expert GGUF dtype and shard hashes require the model inventory; no weight conversion occurs in this engine.\",\n"
           << "  \"input_ids\":";
    tokens_json(output, report.input_ids);
    output << ",\n  \"output_ids\":";
    tokens_json(output, report.output_ids);
    output << ",\n  \"output_text\":" << json_string(report.output_text)
           << ",\n  \"timing\":{\"load_wall_ms\":" << report.load_wall_ms
           << ",\"request_ms\":" << report.request_ms << ",\"ttft_ms\":";
    optional_number(output, report.ttft_ms);
    output << ",\"prompt_eval_ms\":" << report.prompt_eval_ms << ",\"token_eval_ms\":" << report.token_eval_ms
           << ",\"prompt_eval_tokens\":" << report.prompt_eval_tokens
           << ",\"token_eval_tokens\":" << report.token_eval_tokens << ",\"pp_tokens_per_second\":";
    optional_number(output, rate(report.prompt_eval_tokens, report.prompt_eval_ms));
    output << ",\"tg_tokens_per_second\":";
    optional_number(output, rate(report.token_eval_tokens, report.token_eval_ms));
    output << ",\"graph_reuses\":" << report.graph_reuses
           << ",\"summary_basis\":" << json_string(report.diagnostics ? "single truncated layer" :
                "median of successful measured request times; token counts per request; graph_reuses summed; warmup excluded")
           << "},\n  \"expert_transport_basis\":" << json_string(report.diagnostics ?
                "one truncated layer request; cache priming excluded" : "sum of measured per-request deltas; warmup and cache priming excluded")
           << ",\n  \"requests\":[";
    for (std::size_t i = 0; i < report.requests.size(); ++i) {
        if (i) output << ',';
        request_json(output, report.requests[i], report.pinned_experts, report.cold_threads > 0);
    }
    output << "],\n  \"cache_priming_requests\":[";
    for (std::size_t i = 0; i < report.priming_requests.size(); ++i) {
        if (i) output << ',';
        request_json(output, report.priming_requests[i], report.pinned_experts, report.cold_threads > 0);
    }
    output << "],\n  \"native_metrics\":{";
    const bool available = metrics && metrics->available;
    auto metric = [&](const char* name, uint64_t value, bool comma) {
        if (comma) output << ',';
        output << json_string(name) << ':';
        if (available) output << value; else output << "null";
    };
    metric("h2d_bytes", available ? metrics->h2d_bytes : 0, false);
    metric("d2h_bytes", available ? metrics->d2h_bytes : 0, true);
    metric("expert_launches", available ? metrics->expert_launches : 0, true);
    metric("host_continuations", available ? metrics->host_continuations : 0, true);
    metric("synchronizations", available ? metrics->synchronizations : 0, true);
    output << ",\"availability\":" << json_string(available ? "authoritative backend snapshot" : "unavailable: no backend metrics snapshot connected")
           << "},\n  \"memory\":{\"peak_host_bytes\":null,\"peak_device_bytes\":null},\n  \"checkpoints\":[";
    for (std::size_t i = 0; i < report.checkpoints.size(); ++i) {
        const auto& value = report.checkpoints[i];
        if (i) output << ',';
        output << "{\"name\":" << json_string(value.name) << ",\"dtype\":" << json_string(value.dtype)
               << ",\"file\":" << json_string(value.file) << ",\"logical_bytes\":" << value.bytes << ",\"dimensions\":[";
        for (std::size_t dim = 0; dim < 4; ++dim) { if (dim) output << ','; output << value.dimensions[dim]; }
        output << "]}";
    }
    output << "]\n}\n";
    if (!output) throw std::runtime_error("cannot write benchmark report");
}

bool compare_runs(const std::filesystem::path& reference, const std::filesystem::path& candidate,
                  const std::filesystem::path& output_directory) {
    bool equal = false;
    std::string error, first_difference;
    CheckpointComparison checkpoints;
    bool has_checkpoints = false;
    bool has_request_sequences = false;
    try {
        const auto identity = read_text(reference / "workload.txt");
        if (identity != read_text(candidate / "workload.txt"))
            throw std::runtime_error("workload, source, model, or placement differs");
        if (identity.find("\ncache_prime=1\n") != std::string::npos &&
            read_text(reference / "cache-prime-tokens.tsv") != read_text(candidate / "cache-prime-tokens.tsv"))
            throw std::runtime_error("cache priming token evidence differs");
        const auto status = read_text(reference / "status.txt");
        if (status != read_text(candidate / "status.txt") || status.rfind("passed\n", 0) != 0)
            throw std::runtime_error("both runs must have passed the same gate");
        if (read_tokens(reference / "input-ids.txt") != read_tokens(candidate / "input-ids.txt"))
            first_difference = "input token IDs";
        else if (read_tokens(reference / "output-ids.txt") != read_tokens(candidate / "output-ids.txt"))
            first_difference = "output token IDs";
        else {
            has_request_sequences = identity.find("\nrequest_reset=") != std::string::npos ||
                                    std::filesystem::exists(reference / "request-tokens.tsv") ||
                                    std::filesystem::exists(candidate / "request-tokens.tsv");
            if (has_request_sequences) {
                if (!std::filesystem::exists(reference / "request-tokens.tsv") ||
                    !std::filesystem::exists(candidate / "request-tokens.tsv"))
                    throw std::runtime_error("per-request token evidence is missing from one run");
                if (read_text(reference / "request-tokens.tsv") != read_text(candidate / "request-tokens.tsv")) {
                    first_difference = "per-request input/output token IDs, state, or completion status";
                    throw std::runtime_error("per-request evidence differs");
                }
            }
            has_checkpoints = status == "passed\nlayer\n";
            if (has_checkpoints) {
                checkpoints = compare_checkpoints(reference, candidate);
                error = checkpoints.error;
                first_difference = checkpoints.first_difference;
                equal = checkpoints.equal;
            } else equal = true;
        }
    } catch (const std::exception& failure) { error = failure.what(); }
    std::filesystem::create_directories(output_directory);
    std::ofstream output(output_directory / "comparison.json");
    output << std::setprecision(12) << "{\"schema\":\"strata-deepseek-v4-comparison-v1\",\"equal\":"
           << (equal ? "true" : "false") << ",\"reference\":" << json_string(reference.string())
           << ",\"candidate\":" << json_string(candidate.string()) << ",\"error\":" << json_string(error)
           << ",\"first_difference\":" << json_string(first_difference)
           << ",\"compared_checkpoints\":" << checkpoints.compared_tensors
           << ",\"first_difference_byte\":" << checkpoints.first_difference_byte
           << ",\"max_absolute_error\":" << checkpoints.max_absolute_error
           << ",\"max_relative_error\":" << checkpoints.max_relative_error
           << ",\"has_nonfinite\":" << (checkpoints.has_nonfinite ? "true" : "false")
           << ",\"comparison_basis\":" << json_string(has_checkpoints ? "exact scalar activation bytes" :
                has_request_sequences ? "exact input/output token IDs for every warmup/measured request" : "exact input/output token IDs")
           << "}\n";
    if (!output) throw std::runtime_error("cannot write comparison report");
    return equal;
}

} // namespace strata::deepseek_v4
