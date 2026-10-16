#include "benchmark_report.hpp"
#include "benchmark_args.hpp"
#include "benchmark_workload.hpp"
#include "cpu_probe_report.hpp"
#include "cold_metrics_report.hpp"
#include "profile_scope.hpp"
#include "benchmark_profile.hpp"
#include "runtime.hpp"
#include "cache_lease.hpp"

#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace strata::deepseek_v4 {
namespace {

using Clock = std::chrono::steady_clock;


double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

std::string workload(const Options& options, const DriverOptions& driver) {
    std::ostringstream output;
    output << "strata-deepseek-v4-workload-v1\n"
           << "mode=" << driver.mode << "\nmodel=" << options.model
           << "\ncanonical=" << Runtime::source_commit() << "\ndevice=" << options.device
           << "\nctx=" << options.n_ctx << "\nbatch=" << options.n_batch << "\nubatch=" << options.n_ubatch
           << "\nthreads=" << options.n_threads << "\ngpu_layers=" << options.gpu_layers
           << "\nkv=f16/f16\nflash_attention=enabled\nthreadpool=resident\nexpert_source=original CPU GGUF mmap\n"
           << "sample=greedy\npredict=" << driver.predict
           << "\nignore_eos=" << driver.ignore_eos << "\nprompt_tokens_target=" << driver.prompt_tokens
           << "\nprompt_selection=target prefix plus final min(256,target/2) tokens; no padding"
           << "\nlayer_tokens_requested=" << (driver.mode == "layer" ? driver.layer_tokens : 0)
           << "\nlayer_fusion_seam=" << driver.layer_fusion_seam
           << "\ncache_prime=" << driver.cache_prime << "\ncache_prime_predict=" << (driver.cache_prime ? 8 : 0)
           << "\nprompt_bytes=" << options.prompt.size()
           << "\nprompt=" << options.prompt << "\n";
    if (driver.mode == "pp" || driver.mode == "tg")
        output << "warmup_requests=" << driver.warmup_requests
               << "\nmeasure_requests=" << driver.measure_requests
               << "\nrequest_reset=synchronize;clear raw/compressed KV and perf;retain expert cache;synchronize\n";
    // Cache capacity and upload transport may differ; model mathematics must not.
    return output.str();
}

void ensure_empty_output(const std::filesystem::path& output) {
    std::filesystem::create_directories(output);
    if (!std::filesystem::is_empty(output))
        throw std::runtime_error("--output-dir must be empty to prevent stale or overwritten evidence");
}

void check_mxfp4_contract() {
    if (ggml_blck_size(GGML_TYPE_MXFP4) != 32 || ggml_type_size(GGML_TYPE_MXFP4) != 17)
        throw std::runtime_error("canonical MXFP4 ABI differs from 32 elements / 17 bytes");
}

void copy_performance(Runtime& runtime, RequestReport& report) {
    const auto performance = runtime.performance();
    report.prompt_eval_ms = performance.t_p_eval_ms;
    report.token_eval_ms = performance.t_eval_ms;
    report.prompt_eval_tokens = performance.n_p_eval;
    report.token_eval_tokens = performance.n_eval;
    report.graph_reuses = performance.n_reused;
}

ExpertStagingMetrics transport_delta(const ExpertStagingMetrics& after,
                                     const ExpertStagingMetrics& before) {
    if (after.bytes < before.bytes || after.calls < before.calls || after.slab_waits < before.slab_waits ||
        after.host_copy_wall_ns < before.host_copy_wall_ns || after.slab_wait_wall_ns < before.slab_wait_wall_ns ||
        after.direct_bytes < before.direct_bytes || after.direct_calls < before.direct_calls)
        throw std::runtime_error("expert transport counters decreased within one request");
    return {after.bytes - before.bytes, after.calls - before.calls, after.slab_waits - before.slab_waits,
            after.host_copy_wall_ns - before.host_copy_wall_ns, after.slab_wait_wall_ns - before.slab_wait_wall_ns,
            after.direct_bytes - before.direct_bytes, after.direct_calls - before.direct_calls};
}

ExpertStagingMetrics transport_sum(const ExpertStagingMetrics& total, const ExpertStagingMetrics& delta) {
    auto add = [](uint64_t a, uint64_t b) {
        if (b > (std::numeric_limits<uint64_t>::max)() - a)
            throw std::runtime_error("measured expert transport total exceeds its fixed counter range");
        return a + b;
    };
    return {add(total.bytes, delta.bytes), add(total.calls, delta.calls), add(total.slab_waits, delta.slab_waits),
            add(total.host_copy_wall_ns, delta.host_copy_wall_ns), add(total.slab_wait_wall_ns, delta.slab_wait_wall_ns),
            add(total.direct_bytes, delta.direct_bytes), add(total.direct_calls, delta.direct_calls)};
}

RequestReport request(Runtime& runtime, const std::vector<llama_token>& input,
                      int predict, uint32_t index, bool measured, const DriverOptions& driver) {
    RequestReport report;
    report.index = index;
    report.measured = measured;
    report.state = index == 0 ? "fresh context; host page state uncontrolled" :
                               "identical prompt repeated; expert cache and model retained";
    report.input_ids.assign(input.begin(), input.end());
    // DeepSeek4 reset clears raw/compressed KV and rolling state, not the MoE
    // cache. Drain previous work before clearing, and exclude clearing from timing.
    llama_synchronize(runtime.context());
    runtime.reset(true);
    llama_synchronize(runtime.context());
    const auto before = runtime.expert_transport_metrics();
    const auto cold_before = runtime.cold_metrics();
    std::unique_ptr<ProfileScope> scope;
    TgProfileWindow tg_profile(measured, predict);
    if (measured) scope = std::make_unique<ProfileScope>("STRATA_DSV4_REQUEST", !tg_profile.enabled());
    const auto start = Clock::now();
    try {
        std::string error;
        benchmark_progress(driver.output, "prefill_started", index, runtime, elapsed_ms(start), input.size(), 0);
        if (!runtime.evaluate(input, error)) throw std::runtime_error(error);
        benchmark_progress(driver.output, "prefill_completed", index, runtime, elapsed_ms(start), input.size(), 0);
        for (int i = 0; i < predict; ++i) {
            const llama_token token = runtime.greedy(driver.ignore_eos);
            if (i == 0) {
                report.ttft_ms = elapsed_ms(start);
                report.prefill_transport = transport_delta(runtime.expert_transport_metrics(), before);
                report.transport_phases_complete = true;
            }
            tg_profile.after_greedy(i);
            report.output_ids.push_back(token);
            report.output_text += runtime.piece(token, true);
            if ((i + 1) % 256 == 0)
                benchmark_progress(driver.output, "generation", index, runtime, elapsed_ms(start), input.size(), i + 1);
            if (runtime.is_eog(token) || i + 1 == predict) break;
            if (!runtime.evaluate(std::vector<llama_token>{token}, error)) throw std::runtime_error(error);
        }
        report.status = "passed";
    } catch (const std::exception& failure) {
        report.status = "failed";
        report.error = failure.what();
    }
    llama_synchronize(runtime.context());
    report.request_ms = elapsed_ms(start);
    benchmark_progress(driver.output, report.status.c_str(), index, runtime, report.request_ms,
                       input.size(), int(report.output_ids.size()));
    copy_performance(runtime, report);
    report.expert_transport_delta = transport_delta(runtime.expert_transport_metrics(), before);
    report.decode_transport = transport_delta(report.expert_transport_delta, report.prefill_transport);
    report.cold_delta = cold_metrics_delta(runtime.cold_metrics(), cold_before);
    return report;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const auto middle = values.size() / 2;
    return values.size() % 2 ? values[middle] : (values[middle - 1] + values[middle]) / 2;
}

void summarize_requests(BenchmarkReport& report) {
    std::vector<double> wall, ttft, pp, tg;
    for (const auto& item : report.requests) {
        if (!item.measured) continue;
        report.expert_transport = transport_sum(report.expert_transport, item.expert_transport_delta);
        report.cold = cold_metrics_sum(report.cold, item.cold_delta);
        if (item.status != "passed") continue;
        wall.push_back(item.request_ms); ttft.push_back(item.ttft_ms);
        pp.push_back(item.prompt_eval_ms); tg.push_back(item.token_eval_ms);
        report.prompt_eval_tokens = item.prompt_eval_tokens;
        report.token_eval_tokens = item.token_eval_tokens;
        report.graph_reuses += item.graph_reuses;
    }
    if (!wall.empty()) {
        report.request_ms = median(wall); report.ttft_ms = median(ttft);
        report.prompt_eval_ms = median(pp); report.token_eval_ms = median(tg);
    }
}

int run(const Options& options, const DriverOptions& driver) {
    if (driver.mode == "layer" && ((uint32_t) driver.layer_tokens > options.n_ubatch ||
                                   (uint32_t) driver.layer_tokens > options.n_batch))
        throw std::runtime_error("--layer-tokens must fit one configured ubatch and batch to observe the complete input");
    if (driver.all_hit_lease && (!options.moe_cache_mib || options.cold_threads))
        throw std::runtime_error("experimental all-hit lease needs one cache and disables the CPU miss provider");
    ensure_empty_output(driver.output);
    check_mxfp4_contract();
    BenchmarkReport report;
    report.mode = driver.mode;
    report.status = "failed";
    report.model = options.model;
    report.device = options.device;
    report.cache_mib = options.moe_cache_mib;
    report.canonical_commit = Runtime::source_commit();
    report.workload_identity = workload(options, driver);
    report.diagnostics = driver.mode == "layer";
    report.warmup_requests = driver.warmup_requests;
    report.measure_requests = driver.measure_requests;
    report.cold_threads = options.cold_threads;
    report.cold_admit_free = options.cold_admit_free != 0;
    report.cache_leaf_aliases = options.cache_leaf_aliases != 0;
    report.layer_fusion_seam = driver.layer_fusion_seam;
    report.cache_prime = driver.cache_prime;
    std::unique_ptr<CheckpointCollector> collector;
    if (report.diagnostics) collector = std::make_unique<CheckpointCollector>(driver.output, driver.layer_fusion_seam);
    if (collector) collector->pause(driver.cache_prime);
    std::string error;
    const auto loading = Clock::now();
    auto runtime = Runtime::load(options, error, collector ? CheckpointCollector::callback : nullptr,
                                 collector.get());
    report.load_wall_ms = elapsed_ms(loading);
    if (!runtime) {
        report.error = error;
        write_report(driver.output, report);
        std::cout << "{\"status\":\"failed\",\"error\":" << json_string(error) << "}\n";
        return 1;
    }
    report.architecture = runtime->metadata("general.architecture");
    report.threadpool_resident = runtime->threadpool_resident();
    report.pinned_experts = options.pinned_experts != 0;
    report.source_pin = runtime->source_pin_metrics();
    ExpertStagingMetrics layer_transport_before;
    ColdExpertMetrics layer_cold_before;
    try {
        auto input = benchmark_input(*runtime, options, driver);
        if (driver.all_hit_lease && (input.size() > 8 || driver.layer_tokens > 8))
            throw std::runtime_error("experimental all-hit lease supports only1..8 input tokens");
        if (driver.cache_prime) {
            if (input.size() > 64 || input.size() + 8 > options.n_ctx)
                throw std::runtime_error("cache priming exceeds the bounded canary/context");
            report.priming_requests.push_back(request(*runtime, input, 8, 0, false, driver));
            if (report.priming_requests.back().status != "passed")
                throw std::runtime_error(report.priming_requests.back().error);
            if (driver.all_hit_lease) {
                configure_all_hit_cache_lease(runtime->context(), driver.lease_reject_layer_zero);
                report.all_hit_lease = true;
            }
            if (collector) collector->pause(false);
        }
        if (report.diagnostics) {
            if (driver.cache_prime) { llama_synchronize(runtime->context()); runtime->reset(true); llama_synchronize(runtime->context()); }
            layer_transport_before = runtime->expert_transport_metrics();
            layer_cold_before = runtime->cold_metrics();
            if (input.size() < (std::size_t) driver.layer_tokens)
                throw std::runtime_error("layer prompt supplies " + std::to_string(input.size()) +
                                         " tokens; --layer-tokens requested " + std::to_string(driver.layer_tokens));
            input.resize((std::size_t) driver.layer_tokens);
        }
        if (input.size() + (std::size_t) driver.predict > options.n_ctx)
            throw std::runtime_error("canary exceeds configured context");
        report.input_ids.assign(input.begin(), input.end());
        if (report.diagnostics) {
            llama_perf_context_reset(runtime->context());
            ProfileScope request_scope("STRATA_DSV4_REQUEST");
            const auto start = Clock::now();
            const int status = runtime->decode_tokens(input.data(), input.size(), 0, true, error);
            report.request_ms = elapsed_ms(start);
            report.layer_complete = collector->complete();
            collector->write_manifest();
            report.checkpoints = collector->checkpoints();
            if (collector->failed()) throw std::runtime_error(collector->error());
            if (!report.layer_complete) throw std::runtime_error("layer-zero l_last checkpoint was not observed: " + error);
            if (status != 2) throw std::runtime_error("layer observer must abort the entire graph (decode status 2); got " + std::to_string(status));
            if (driver.all_hit_lease) verify_all_hit_lease_layer_gate(runtime->context());
            // A truncated layer has no final logits or generated output token.
        } else {
            const int total = driver.warmup_requests + driver.measure_requests;
            std::vector<int32_t> expected;
            for (int index = 0; index < total; ++index) {
                auto item = request(*runtime, input, driver.predict, uint32_t(index + driver.cache_prime),
                                    index >= driver.warmup_requests, driver);
                if (item.measured && index == driver.warmup_requests) {
                    report.output_ids = item.output_ids;
                    report.output_text = item.output_text;
                }
                if (index == 0) expected = item.output_ids;
                else if (item.status == "passed" && item.output_ids != expected) {
                    item.status = "failed";
                    item.error = "output token IDs changed across identical requests in one Runtime";
                } else if (index > 0 && item.status == "passed" &&
                           (item.prompt_eval_tokens != report.requests.front().prompt_eval_tokens ||
                            item.token_eval_tokens != report.requests.front().token_eval_tokens)) {
                    item.status = "failed";
                    item.error = "canonical performance token counts changed across identical requests";
                }
                report.requests.push_back(std::move(item));
                if (report.requests.back().status != "passed")
                    throw std::runtime_error(report.requests.back().error);
            }
        }
        report.status = "passed";
    } catch (const std::exception& failure) {
        report.error = failure.what();
        if (collector) {
            collector->write_manifest();
            report.checkpoints = collector->checkpoints();
        }
    }
    if (report.diagnostics) {
        report.expert_transport = transport_delta(runtime->expert_transport_metrics(), layer_transport_before);
        report.cold = cold_metrics_delta(runtime->cold_metrics(), layer_cold_before);
    }
    else summarize_requests(report);
    write_report(driver.output, report);
    std::cout << "{\"status\":" << json_string(report.status) << ",\"mode\":" << json_string(driver.mode)
              << ",\"output_dir\":" << json_string(driver.output.string())
              << ",\"error\":" << json_string(report.error) << "}\n";
    return report.status == "passed" ? 0 : 1;
}

} // namespace
} // namespace strata::deepseek_v4

int main(int argc, char** argv) {
    using namespace strata::deepseek_v4;
    try {
        Options options;
        std::vector<std::string> remaining;
        std::string error;
        if (!parse_options(argc, argv, options, remaining, error)) throw std::runtime_error(error);
        const auto driver = parse_driver(remaining);
        if (driver.help) {
            std::cout << "Strata native DeepSeek4 bounded correctness/performance gates\n"
                      << options_usage()
                      << "  --mode layer|pp|tg --output-dir EMPTY_DIRECTORY [--n-predict 1..4096]\n"
                      << "  --mode pp|tg [--warmup-requests 0..3] [--measure-requests 1..3] (defaults: 0/1)\n"
                      << "  --prompt-file UTF8_FILE (at most 1MiB; replaces --prompt for text gates)\n"
                      << "  --prompt-tokens 1..16384 (exact input count; keep prefix and final min(256,N/2) tokens)\n"
                      << "  --ignore-eos 0|1 (default: 0; mask end-of-generation tokens for fixed-length TG)\n"
                      << "  --mode layer [--layer-tokens 1..64] (default: 1)\n"
                      << "  --mode layer [--layer-fusion-seam 0|1] (default: 0; 1 observes GLU onward without Gate/Up splits)\n"
                      << "  --cache-prime 0|1 (default: 0; one separate full PP/TG cache priming request)\n"
                      << "  --all-hit-cache-lease 0|1 (default: 0; experimental frozen GPU cache, missing routes fail)\n"
                      << "  --lease-reject-layer-zero 0|1 (default: 0; full-request negative lease gate)\n"
                      << "  --mode compare --reference-dir DIR --candidate-dir DIR --output-dir DIR\n"
                      << "  --mode cpu-probe --model GGUF --reference-dir CPU_LAYER_DIR --output-dir DIR --threads N\n"
                      << "  --mode cpu-probe [--cpu-reference-device 0|1] (default: 0; explicit CPU vs GPU input/output differential)\n"
                      << "  --mode cpu-probe [--cpu-route-mask 1..63] (raw experts; bits select original router ranks)\n"
                      << "Layer requires at least the requested prompt tokens and stops after layer-zero l_last.\n"
                      << "Use --layer-tokens 32 --ubatch-size 32 to expose PP MMQ errors; batch must also fit 32.\n"
                      << "PP/TG use at most 16384 prompt tokens; diagnostics are disabled.\n"
                      << "Repeated requests clear KV/perf, retain expert cache; timings summarize measured requests only.\n"
                      << "STRATA_DSV4_PROFILE=1 requires one measured request; one warmup remains allowed.\n";
            return 0;
        }
        if (!driver.prompt_file.empty()) options.prompt = read_prompt_file(driver.prompt_file);
        if (driver.mode == "compare") {
            ensure_empty_output(driver.output);
            const bool equal = compare_runs(driver.reference, driver.candidate, driver.output);
            std::cout << "{\"equal\":" << (equal ? "true" : "false")
                      << ",\"output_dir\":" << json_string(driver.output.string()) << "}\n";
            return equal ? 0 : 1;
        }
        if (driver.mode == "cpu-probe") {
            check_mxfp4_contract();
            return run_cpu_probe_case({options.model, driver.reference, driver.output, options.n_threads,
                                       driver.cpu_route_mask, driver.cpu_route_mask_set, driver.cpu_reference_device});
        }
        return run(options, driver);
    } catch (const std::exception& error) {
        std::cerr << "{\"status\":\"failed\",\"error\":" << json_string(error.what()) << "}\n";
        return 1;
    }
}
