#include "session_gate.hpp"
#include "device_cache_probe_report.hpp"
#include <chrono>
#include <fstream>
#include <stdexcept>

namespace strata::deepseek_v4 {
namespace {
double layer(Runtime& runtime, SessionGateCallback& observer, const std::filesystem::path& path,
             llama_token token, bool candidate) {
    enable_device_expert_cache(runtime.context(), candidate);
    llama_synchronize(runtime.context());
    runtime.reset();
    CheckpointCollector collector(path);
    observer.collector = &collector;
    const auto start = std::chrono::steady_clock::now();
    std::string error;
    const int status = runtime.decode_tokens(&token, 1, 0, false, error);
    llama_synchronize(runtime.context());
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    observer.collector = nullptr;
    verify_device_expert_cache(runtime.context());
    collector.write_manifest();
    if (status != 2 || !collector.complete() || collector.failed())
        throw std::runtime_error("real layer gate did not stop at its canonical boundary: " + error + collector.error());
    return ms;
}
}
std::vector<LayerGateResult> session_layer_gates(Runtime& runtime, SessionGateCallback& observer,
        const std::vector<llama_token>& input, const std::filesystem::path& output) {
    if (input.size() < 100) throw std::runtime_error("layer suite requires the representative prompt");
    std::vector<LayerGateResult> results;
    const size_t positions[]{0, 0, 37, 87};
    for (size_t test = 0; test < 4; ++test) {
        LayerGateResult result;
        result.token = input[positions[test]];
        const auto candidate = output / ("layer-" + std::to_string(test) + "-candidate");
        const auto control = output / ("layer-" + std::to_string(test) + "-control");
        const auto before = device_expert_cache_metrics(runtime.context());
        result.candidate_ms = layer(runtime, observer, candidate, result.token, true);
        const auto after = device_expert_cache_metrics(runtime.context());
        result.all_hit = after.gpu[1] - before.gpu[1];
        result.misses = after.gpu[3] - before.gpu[3];
        result.hit_routes = after.gpu[2] - before.gpu[2];
        result.bytes = after.transport.bytes - before.transport.bytes;
        result.service_ns = after.service_wall_ns - before.service_wall_ns;
        result.memcpy_ns = after.transport.host_copy_wall_ns - before.transport.host_copy_wall_ns;
        result.slab_wait_ns = after.transport.slab_wait_wall_ns - before.transport.slab_wait_wall_ns;
        result.submit_ns = after.dma_costs.submit_ns - before.dma_costs.submit_ns;
        result.commit_wait_ns = after.dma_costs.commit_wait_ns - before.dma_costs.commit_wait_ns;
        // Compare after admission, using identical banks and the same canonical GPU provider.
        result.control_ms = layer(runtime, observer, control, result.token, false);
        const auto compared = compare_checkpoints(control, candidate);
        if (!compared.equal) throw std::runtime_error("real layer mismatch: " + compared.first_difference + compared.error);
        result.compared = compared.compared_tensors;
        results.push_back(result);
        std::ofstream progress(output / "layer-progress.json");
        progress << "{\"layers\":[";
        for (size_t i = 0; i < results.size(); ++i) {
            if (i) progress << ',';
            session_write_layer(progress, results[i]);
        }
        progress << "]}\n";
        if (!progress) throw std::runtime_error("cannot save completed layer results");
    }
    if (results[0].misses != 1 || results[0].hit_routes != 0 || results[1].all_hit != 1 ||
        results[1].bytes != 0 || results[1].misses != 0 || results[2].misses + results[3].misses == 0)
        throw std::runtime_error("suite did not exercise cold fill, zero-copy hit and changed-route eviction");
    return results;
}
}
