#include "pp_stream_probe_graph.hpp"
#include "pp_stream_metrics.hpp"
#include "mxfp4_cpu_probe.hpp"
#include "native_tg_probe.hpp"
#include "benchmark_fault_guard.hpp"
#include "expert_source_pin.hpp"
#include "pp_cache_probe.hpp"
#include "pp_transport_probe.hpp"
#include "checkpoint.hpp"
#include "llama.h"
#include "llama-model.h"
#include "ggml-cpu.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>
namespace {
using namespace strata::deepseek_v4;
void environment(const char* name, const char* value) {
#ifdef _WIN32
    if (_putenv_s(name, value)) throw std::runtime_error("PP probe environment failed");
#else
    if (setenv(name, value, 1)) throw std::runtime_error("PP probe environment failed");
#endif
}
void ck(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
uint64_t hash(const void* data, size_t bytes) {
    uint64_t h = 14695981039346656037ull;
    for (size_t i = 0; i < bytes; ++i) { h ^= static_cast<const uint8_t*>(data)[i]; h *= 1099511628211ull; }
    return h;
}
struct Sources {
    std::array<ggml_tensor*, 3> tensors{};
    std::array<ggml_tensor*, 3> originals{};
    std::array<uint64_t, 3> hashes{};
    ggml_context* meta = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    void* pinned = nullptr;
    size_t bytes = 0;
    explicit Sources(llama_model& model) {
        const char* names[] = {"blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight", "blk.0.ffn_down_exps.weight"};
        auto& original = originals;
        for (int i = 0; i < 3; ++i) {
            const auto it = std::find_if(model.tensors_by_name.begin(), model.tensors_by_name.end(),
                [&](const auto& t) { return t.first == names[i]; });
            if (it == model.tensors_by_name.end() || it->second->type != GGML_TYPE_MXFP4 ||
                !it->second->data || !ggml_is_contiguous(it->second))
                throw std::runtime_error("PP probe real MXFP4 bank unavailable");
            original[i] = it->second; bytes += ggml_nbytes(original[i]);
        }
        if (bytes != 3422552064ull) throw std::runtime_error("PP probe real layer payload differs");
        ck(cudaHostAlloc(&pinned, bytes, cudaHostAllocDefault));
        buffer = ggml_backend_cpu_buffer_from_ptr(pinned, bytes);
        if (buffer) ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        meta = ggml_init({16 * ggml_tensor_overhead(), nullptr, true});
        if (!meta || !buffer) throw std::runtime_error("PP probe source metadata failed");
        size_t offset = 0;
        for (int i = 0; i < 3; ++i) {
            tensors[i] = ggml_new_tensor_3d(meta, GGML_TYPE_MXFP4, original[i]->ne[0], original[i]->ne[1], 256);
            tensors[i]->data = static_cast<char*>(pinned) + offset; tensors[i]->buffer = buffer;
            const auto n = ggml_nbytes(original[i]);
            std::memcpy(tensors[i]->data, original[i]->data, n);
            // Exact source transport is checked before any numerical tolerance.
            if (std::memcmp(tensors[i]->data, original[i]->data, n))
                throw std::runtime_error("PP source payload copy differs");
            hashes[i] = hash(tensors[i]->data, n); offset += n;
        }
    }
    ~Sources() { if (buffer) ggml_backend_buffer_free(buffer); if (meta) ggml_free(meta); if (pinned) cudaFreeHost(pinned); }
};
double compute(PpProbeGraph& graph, const std::array<ggml_tensor*, 3>& sources, bool upload) {
    const auto begin = std::chrono::steady_clock::now();
    if (upload) graph.upload(sources);
    if (ggml_backend_graph_compute_async(graph.backend, graph.graph) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("PP operator compute failed");
    ggml_backend_synchronize(graph.backend);
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
}
int run(int argc, char** argv) {
    std::string path, reference, report_path;
    int tokens = 8192, window = 32;
    bool pp_only = false, tg_only = false;
    bool skew = false;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc) throw std::runtime_error("PP probe option needs a value");
        const std::string flag = argv[i], value = argv[i + 1];
        if (flag == "--model") path = value;
        else if (flag == "--reference") reference = value;
        else if (flag == "--report") report_path = value;
        else if (flag == "--tokens") tokens = std::stoi(value);
        else if (flag == "--window") window = std::stoi(value);
        else if (flag == "--pp-only" && (value == "0" || value == "1")) pp_only = value == "1";
        else if (flag == "--tg-only" && (value == "0" || value == "1")) tg_only = value == "1";
        else if (flag == "--route-pattern" && (value == "balanced" || value == "skew")) skew = value == "skew";
        else throw std::runtime_error("unknown PP probe option");
    }
    if (path.empty() || reference.empty() || report_path.empty() || tokens < 64 || tokens > 8192 ||
        window < 1 || window > 64 || std::filesystem::exists(report_path))
        throw std::runtime_error("PP probe input/output outside bounded slice");
    // CUDA reads graph-disable once per process. Keep it absent so the native
    // TG gate actually tests captured execution; PP sync ops reject capture.
#ifdef _WIN32
    if (_putenv_s("GGML_CUDA_DISABLE_GRAPHS", "")) throw std::runtime_error("cannot enable CUDA graphs for native gate");
#else
    if (unsetenv("GGML_CUDA_DISABLE_GRAPHS")) throw std::runtime_error("cannot enable CUDA graphs for native gate");
#endif
    environment("STRATA_DSV4_SOURCE_PRIVATE_MAPPING", "1");
    const auto activation = read_checkpoint_data(reference, "ffn_norm-0");
    if (activation.metadata.type != GGML_TYPE_F32 || activation.metadata.dimensions[0] != 4096 ||
        activation.metadata.dimensions[1] != 5 || activation.data.size() != 4096 * 5 * sizeof(float))
        throw std::runtime_error("PP probe needs actual layer-zero PP5 activation");
    llama_backend_init();
    ggml_backend_dev_t devices[] = {nullptr};
    auto parameters = llama_model_default_params();
    parameters.devices = devices; parameters.n_gpu_layers = 0; parameters.split_mode = LLAMA_SPLIT_MODE_NONE;
    parameters.load_mode = LLAMA_LOAD_MODE_MMAP; parameters.lazy_mode = LLAMA_LAZY_MODE_ON;
    parameters.use_extra_bufts = false; parameters.load_mtp = false;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(path.c_str(), parameters), llama_model_free);
    if (!model || model->arch != LLM_ARCH_DEEPSEEK4) throw std::runtime_error("real D4 model metadata unavailable");
    auto* dev = ggml_backend_dev_by_name("CUDA0");
    if (!dev) throw std::runtime_error("selected CUDA0 unavailable");
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_dev_init(dev, nullptr), ggml_backend_free);
    if (!backend) throw std::runtime_error("selected CUDA backend unavailable");
    Sources sources(*model);
    std::array<float, 4096> actual_norm;
    std::memcpy(actual_norm.data(), activation.data.data(), sizeof(actual_norm));
    std::ostringstream cpu_report;
    if (pp_only && tg_only) throw std::runtime_error("PP-only and TG-only are exclusive");
    if (pp_only || tg_only) cpu_report << "null";
    const bool cpu_passed = pp_only || tg_only || mxfp4_cpu_operator_probe(*model, sources.tensors, actual_norm.data(), cpu_report);
    std::ostringstream native_tg_report;
    if (pp_only) native_tg_report << "null";
    const bool native_tg_passed = pp_only || native_tg_operator_probe(path,backend.get(),sources.tensors,sources.originals,actual_norm.data(),
        model->hparams.swiglu_clamp_exp[0],native_tg_report);
    if(tg_only) {
        std::ofstream report(report_path);
        report<<"{\"status\":\""<<(native_tg_passed?"passed":"validation_failure")<<"\",\"native_tg_service\":"
              <<native_tg_report.str()<<",\"end_to_end_target_met\":false}\n";
        if(!report)throw std::runtime_error("native TG report write failed");
        return native_tg_passed?0:2;
    }
    // Exercise the real mmap registration edges, not only cudaHostAlloc copies.
    const char* pin_fraction=std::getenv("STRATA_DSV4_PROBE_PP_PIN_PERCENT");
    const int pin_percent=pin_fraction?std::stoi(pin_fraction):100;
    if(pin_percent<1||pin_percent>100)throw std::runtime_error("PP probe pin percent outside1..100");
    const uint64_t pin_budget=sources.bytes*uint64_t(pin_percent)/100;
    ExpertSourcePin source_pin(backend.get(), std::vector<ggml_tensor*>(sources.originals.begin(), sources.originals.end()), pin_budget);
    if (source_pin.metrics().registered_bytes < pin_budget - 6 * 4096)
        throw std::runtime_error("PP original bank registration did not cover its complete pages");
    PpTransportProbeBinding transport_binding(backend.get(), source_pin, sources.originals);
    std::vector<float> x(size_t(tokens) * 4096), weights(tokens * 6, 0.25f);
    std::vector<int32_t> ids(tokens * 6);
    for (int t = 0; t < tokens; ++t) {
        std::memcpy(x.data() + t * 4096, activation.data.data() + (t % 5) * 4096 * sizeof(float), 4096 * sizeof(float));
        for (int r = 0; r < 6; ++r) ids[t * 6 + r] = skew && t < tokens/2 ? r*43 : (t * 17 + r * 43) % 256;
    }
    PpProbeGraph canonical(backend.get(), sources.tensors, tokens, model->hparams.swiglu_clamp_exp[0], false, window);
    PpProbeGraph native(backend.get(), sources.originals, tokens, model->hparams.swiglu_clamp_exp[0], true, window);
    canonical.inputs(x, ids, weights); native.inputs(x, ids, weights);
    // Warm the kernel/pool, then alternate complete transfer+compute samples.
    compute(canonical, sources.tensors, true); compute(native, sources.tensors, false);
    const auto primary_before=pp_stream_backend_metrics(backend.get());
    std::array<double, 3> reference_ms{}, native_ms{};
    for (int i = 0; i < 3; ++i) {
        reference_ms[i] = compute(canonical, sources.tensors, true);
        native_ms[i] = compute(native, sources.tensors, false);
    }
    const auto primary_after=pp_stream_backend_metrics(backend.get());
    const auto a = canonical.output(), b = native.output();
    double max_abs = 0, square_error = 0, square_ref = 0;
    uint64_t nonfinite = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) { ++nonfinite; continue; }
        const double e = double(a[i]) - b[i];
        max_abs = std::max(max_abs, std::abs(e)); square_error += e * e; square_ref += double(a[i]) * a[i];
    }
    const double rms = std::sqrt(square_error / a.size());
    const double relative_rms = std::sqrt(square_error / std::max(square_ref, 1e-30));
    std::ostringstream range_report, edge_report;
    const bool ranges_passed = pp_transport_operator_probe(native, range_report);
    const bool edges_passed = pp_transport_operator_probe(native, edge_report, true);
    std::ostringstream cache_report;
    const bool cache_passed=pp_cache_operator_probe(path,backend.get(),sources.originals,x,ids,weights,a,
        model->hparams.swiglu_clamp_exp[0],window,cache_report,&canonical);
    // Same quantized inputs/math. The grouping/reduction changes F32 ordering;
    // 1e-4 relative RMS leaves ample room above six-term float addition error.
    const bool passed = cpu_passed && native_tg_passed && cache_passed && ranges_passed && edges_passed && !nonfinite && relative_rms <= 1e-4 && max_abs <= 0.01;
    const auto m = pp_stream_backend_metrics(backend.get());
    auto ra = reference_ms, rb = native_ms;
    std::sort(ra.begin(), ra.end()); std::sort(rb.begin(), rb.end());
    std::ofstream report(report_path);
    report << std::setprecision(12) << "{\"status\":\"" << (passed ? "passed" : "validation_failure")
        << "\",\"scope\":\"one real MXFP4 layer; repeated actual PP5 norm; balanced valid routes; operator only\""
        << ",\"tokens\":" << tokens << ",\"window_experts\":" << window << ",\"expert_payload_bytes\":" << sources.bytes
        << ",\"route_pattern\":\"" << (skew?"skew":"balanced") << "\""
        << ",\"source_payload_exact\":true,\"source_fnv64\":[" << sources.hashes[0] << ',' << sources.hashes[1] << ',' << sources.hashes[2]
        << "],\"reference_transfer_compute_ms\":[" << reference_ms[0] << ',' << reference_ms[1] << ',' << reference_ms[2]
        << "],\"native_transfer_compute_ms\":[" << native_ms[0] << ',' << native_ms[1] << ',' << native_ms[2]
        << "],\"median_speedup\":" << ra[1] / rb[1] << ",\"max_abs\":" << max_abs << ",\"rms\":" << rms
        << ",\"relative_rms\":" << relative_rms << ",\"nonfinite\":" << nonfinite
        << ",\"native_layers\":" << m.layers << ",\"native_groups\":" << m.groups
        << ",\"native_payload_bytes\":" << m.payload_bytes << ",\"native_dma_calls\":" << m.dma_calls
        << ",\"native_grouped_products\":" << m.grouped_products << ",\"native_staged_bytes\":" << m.staged_bytes
        << ",\"native_weight_window_bytes\":" << m.weight_window_bytes
        << ",\"native_activation_scratch_bytes\":" << m.activation_scratch_bytes
        << ",\"timed_layers\":" << m.timed_layers << ",\"gpu_layer_ns\":" << m.gpu_layer_ns
        << ",\"gpu_weight_wait_ns\":" << m.gpu_weight_wait_ns << ",\"gpu_products_ns\":" << m.gpu_products_ns
        << ",\"gpu_prelude_ns\":" << m.gpu_prelude_ns << ",\"gpu_intergroup_ns\":" << m.gpu_intergroup_ns
        << ",\"gpu_tail_ns\":" << m.gpu_tail_ns
        << ",\"primary_gpu_layer_ns\":" << primary_after.gpu_layer_ns-primary_before.gpu_layer_ns
        << ",\"primary_gpu_prelude_ns\":" << primary_after.gpu_prelude_ns-primary_before.gpu_prelude_ns
        << ",\"primary_gpu_intergroup_ns\":" << primary_after.gpu_intergroup_ns-primary_before.gpu_intergroup_ns
        << ",\"primary_gpu_weight_wait_ns\":" << primary_after.gpu_weight_wait_ns-primary_before.gpu_weight_wait_ns
        << ",\"primary_gpu_products_ns\":" << primary_after.gpu_products_ns-primary_before.gpu_products_ns
        << ",\"primary_gpu_tail_ns\":" << primary_after.gpu_tail_ns-primary_before.gpu_tail_ns
        << ",\"device_alloc_ns\":" << m.device_alloc_ns << ",\"device_alloc_calls\":" << m.device_alloc_calls
        << ",\"staging_wait_ns\":" << m.staging_wait_ns
        << ",\"native_graph_activation_bytes\":" << ggml_backend_buffer_get_size(native.buffer)
        << ",\"reference_weight_bytes\":" << ggml_backend_buffer_get_size(canonical.wbuffer)
        << ",\"reference_activation_bytes\":" << ggml_backend_buffer_get_size(canonical.buffer)
        << ",\"native_cpu_service\":" << cpu_report.str()
        << ",\"native_tg_service\":" << native_tg_report.str()
        << ",\"pp_cache_reuse\":" << cache_report.str()
        << ",\"pp_source_ranges\":" << range_report.str()
        << ",\"pp_fixed_edges\":" << edge_report.str()
        << ",\"end_to_end_target_met\":false}\n";
    if (!report) throw std::runtime_error("PP probe report write failed");
    std::cout << "{\"status\":\"" << (passed ? "passed" : "validation_failure")
              << "\",\"speedup\":" << ra[1] / rb[1] << ",\"relative_rms\":" << relative_rms << "}\n";
    return passed ? 0 : 2;
}
}
int main(int argc, char** argv) {
    benchmark_fault_guard();
    try { return run(argc, argv); }
    catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
