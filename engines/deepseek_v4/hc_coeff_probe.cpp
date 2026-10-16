#include "hc_coeff.hpp"
#include "checkpoint.hpp"
#include "matrix_timing.hpp"
#include "ggml-alloc.h"
#include "llama.h"
#include "llama-model.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
using namespace strata::deepseek_v4;
struct Graph {
    ggml_backend_t backend;
    ggml_context* context = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_cgraph* graph = nullptr;
    std::array<ggml_tensor*, 2> outputs{};
    ggml_tensor* packed = nullptr;
    Graph(ggml_backend_t b, const CheckpointData& input, const std::vector<float>& scale_data,
          const std::vector<float>& base_data, float eps, bool fused) : backend(b) {
        context = ggml_init({1ull << 18, nullptr, true});
        if (!context) throw std::runtime_error("mHC probe metadata allocation failed");
        graph = ggml_new_graph_custom(context, 32, false);
        auto* mixes = ggml_new_tensor_2d(context, GGML_TYPE_F32, 24, input.metadata.dimensions[1]);
        auto* scale = ggml_new_tensor_1d(context, GGML_TYPE_F32, 3);
        auto* base = ggml_new_tensor_1d(context, GGML_TYPE_F32, 24);
        if (fused) {
            packed = build_hc_coeff(context, mixes, scale, base, eps);
            ggml_set_output(packed);
            ggml_build_forward_expand(graph, packed);
        } else {
            for (int role = 0; role < 2; ++role) {
                auto* input_view = ggml_view_2d(context, mixes, 4, mixes->ne[1], mixes->nb[1], 4 * role * sizeof(float));
                auto* scale_view = ggml_view_1d(context, scale, 1, role * sizeof(float));
                auto* base_view = ggml_view_1d(context, base, 4, 4 * role * sizeof(float));
                auto* value = ggml_add(context, ggml_mul(context, input_view, scale_view), base_view);
                value = ggml_sigmoid(context, value);
                outputs[role] = ggml_scale_bias(context, value, role ? 2.0f : 1.0f, role ? 0.0f : eps);
                ggml_set_output(outputs[role]);
                ggml_build_forward_expand(graph, outputs[role]);
            }
        }
        buffer = ggml_backend_alloc_ctx_tensors(context, backend);
        if (!buffer) { ggml_free(context); context = nullptr; throw std::runtime_error("mHC probe GPU allocation failed"); }
        ggml_backend_tensor_set(mixes, input.data.data(), 0, input.data.size());
        ggml_backend_tensor_set(scale, scale_data.data(), 0, scale_data.size() * sizeof(float));
        ggml_backend_tensor_set(base, base_data.data(), 0, base_data.size() * sizeof(float));
    }
    ~Graph() {
        ggml_backend_synchronize(backend);
        if (buffer) ggml_backend_buffer_free(buffer);
        if (context) ggml_free(context);
    }
    std::vector<float> read(int tokens) {
        std::vector<float> result(8 * tokens);
        if (packed) ggml_backend_tensor_get(packed, result.data(), 0, result.size() * sizeof(float));
        else for (int role = 0; role < 2; ++role)
            ggml_backend_tensor_get(outputs[role], result.data() + role * 4 * tokens, 0, 4 * tokens * sizeof(float));
        for (const float value : result) if (!std::isfinite(value)) throw std::runtime_error("mHC coefficient nonfinite");
        return result;
    }
};
std::vector<float> weight(const llama_model& model, const std::string& name, std::size_t count) {
    const auto found = std::find_if(model.tensors_by_name.begin(), model.tensors_by_name.end(),
                                  [&](const auto& entry) { return entry.first == name; });
    if (found == model.tensors_by_name.end()) throw std::runtime_error("actual mHC weight missing");
    const auto* tensor = found->second;
    if (tensor->type != GGML_TYPE_F32 || ggml_nelements(tensor) != int64_t(count) || !tensor->data ||
        !tensor->buffer || !ggml_backend_buffer_is_host(tensor->buffer) || !ggml_is_contiguous(tensor))
        throw std::runtime_error("actual mHC weight storage differs");
    std::vector<float> result(count); std::memcpy(result.data(), tensor->data, count * sizeof(float));
    return result;
}
double median(std::array<double, 3> samples) { std::sort(samples.begin(), samples.end()); return samples[1]; }
void samples(std::ostream& report, const char* label, const std::array<double, 3>& values) {
    report << std::quoted(label) << ":[" << values[0] << ',' << values[1] << ',' << values[2] << ']';
}
void binary(const std::filesystem::path& path, const std::vector<float>& data) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    if (!out) throw std::runtime_error("mHC probe output write failed");
}
void run_case(const llama_model& model, ggml_backend_t backend, const std::filesystem::path& reference,
              const std::filesystem::path& out, int occurrence, std::ostream& report) {
    const auto input = read_checkpoint_occurrence(reference, "hc_mixes-0", occurrence);
    const int tokens = int(input.metadata.dimensions[1]);
    if ((tokens != 1 && tokens != 5) || input.metadata.type != GGML_TYPE_F32 ||
        input.metadata.dimensions != std::array<int64_t, 4>{24, tokens, 1, 1} || input.data.size() != 24 * tokens * sizeof(float))
        throw std::runtime_error("mHC probe requires actual layer0 TG1/PP5 mixes");
    const std::string role = occurrence == 0 ? "attn" : "ffn";
    const auto scale = weight(model, "blk.0.hc_" + role + "_scale.weight", 3);
    const auto base = weight(model, "blk.0.hc_" + role + "_base.weight", 24);
    Graph control(backend, input, scale, base, model.hparams.dsv4_hc_eps, false);
    Graph candidate(backend, input, scale, base, model.hparams.dsv4_hc_eps, true);
    const auto control_samples = measure_matrix_graph(backend, control.graph);
    const auto expected = control.read(tokens);
    const auto candidate_samples = measure_matrix_graph(backend, candidate.graph);
    const auto actual = candidate.read(tokens);
    if (std::memcmp(expected.data(), actual.data(), expected.size() * sizeof(float)))
        throw std::runtime_error("first actual mHC coefficient divergence");
    for (int r = 0; r < 2; ++r) {
        const auto captured = read_checkpoint_occurrence(reference, r ? "hc_post-0" : "hc_pre-0", occurrence);
        if (captured.data.size() != 4 * tokens * sizeof(float) ||
            std::memcmp(captured.data.data(), expected.data() + r * 4 * tokens, captured.data.size()))
            throw std::runtime_error("mHC isolated canonical control differs from captured model");
    }
    binary(out / (role + "-control.bin"), expected); binary(out / (role + "-fused.bin"), actual);
    if (occurrence) report << ',';
    report << "{\"subblock\":" << std::quoted(role) << ",\"tokens\":" << tokens
           << ",\"full_output_bytes\":" << expected.size() * sizeof(float) << ",\"bit_exact\":true"
           << ",\"control_matches_captured_model\":true,\"control_gpu_ms\":" << median(control_samples)
           << ",\"fused_gpu_ms\":" << median(candidate_samples)
           << ",\"speedup\":" << median(control_samples) / median(candidate_samples) << ',';
    samples(report, "control_samples_ms", control_samples); report << ',';
    samples(report, "fused_samples_ms", candidate_samples);
    report << ",\"expected_control_kernels\":8,\"expected_fused_kernels\":1}";
}
void run_two_case(const llama_model& model, ggml_backend_t backend,
                  const std::filesystem::path& reference, int occurrence, std::ostream& report) {
    auto input = read_checkpoint_occurrence(reference, "hc_mixes-0", occurrence);
    if (input.metadata.type != GGML_TYPE_F32 || input.metadata.dimensions != std::array<int64_t, 4>{24, 5, 1, 1})
        throw std::runtime_error("Actual mHC 5-position seed missing");
    input.metadata.dimensions[1] = 2;
    input.data.resize(48 * sizeof(float));
    const std::string role = occurrence == 0 ? "attn" : "ffn";
    const auto scale = weight(model, "blk.0.hc_" + role + "_scale.weight", 3);
    const auto base = weight(model, "blk.0.hc_" + role + "_base.weight", 24);
    Graph control(backend, input, scale, base, model.hparams.dsv4_hc_eps, false);
    Graph candidate(backend, input, scale, base, model.hparams.dsv4_hc_eps, true);
    const auto before = measure_matrix_graph(backend, control.graph);
    const auto expected = control.read(2);
    const auto after = measure_matrix_graph(backend, candidate.graph);
    const auto actual = candidate.read(2);
    if (std::memcmp(expected.data(), actual.data(), expected.size() * sizeof(float)))
        throw std::runtime_error("Two-position real-weight mHC coefficient differs");
    if (occurrence) report << ',';
    report << "{\"subblock\":" << std::quoted(role) << ",\"positions\":2,\"bit_exact\":true"
           << ",\"control_gpu_ms\":" << median(before) << ",\"fused_gpu_ms\":" << median(after) << '}';
}
int run(int argc, char** argv) {
    std::string model_path, reference, directory;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 == argc) throw std::runtime_error("mHC probe requires flag/value pairs");
        const std::string flag = argv[i], value = argv[i + 1];
        if (flag == "--model") model_path = value;
        else if (flag == "--reference-dir") reference = value;
        else if (flag == "--output-dir") directory = value;
        else throw std::runtime_error("unknown mHC probe option");
    }
    if (model_path.empty() || reference.empty() || directory.empty()) throw std::runtime_error("mHC probe paths required");
    std::filesystem::create_directories(directory);
    if (!std::filesystem::is_empty(directory)) throw std::runtime_error("mHC output must be empty");
    llama_backend_init();
    ggml_backend_dev_t devices[] = {nullptr};
    auto params = llama_model_default_params();
    params.devices = devices; params.n_gpu_layers = 0; params.split_mode = LLAMA_SPLIT_MODE_NONE;
    params.load_mode = LLAMA_LOAD_MODE_MMAP; params.lazy_mode = LLAMA_LAZY_MODE_ON;
    params.use_extra_bufts = false; params.load_mtp = false;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(model_path.c_str(), params), llama_model_free);
    if (!model || model->arch != LLM_ARCH_DEEPSEEK4) throw std::runtime_error("actual mHC model unavailable");
    auto* device = ggml_backend_dev_by_name("CUDA0");
    if (!device) throw std::runtime_error("mHC CUDA0 unavailable");
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_dev_init(device, nullptr), ggml_backend_free);
    if (!backend) throw std::runtime_error("mHC CUDA backend unavailable");
    std::ofstream report(std::filesystem::path(directory) / "report.json");
    report << std::setprecision(10) << "{\"status\":\"passed\",\"scope\":\"actual layer0 attn/FFN coefficient transformation; GPU events3warm+3x32; no model integration or whole-model gain claim\",\"cases\":[";
    for (int occurrence = 0; occurrence < 2; ++occurrence) run_case(*model, backend.get(), reference, directory, occurrence, report);
    report << "],\"two_position_cases\":[";
    if (read_checkpoint_occurrence(reference, "hc_mixes-0", 0).metadata.dimensions[1] == 5)
        for (int occurrence = 0; occurrence < 2; ++occurrence) run_two_case(*model, backend.get(), reference, occurrence, report);
    report << "]}\n";
    if (!report) throw std::runtime_error("mHC report write failed");
    std::cout << "{\"status\":\"passed\",\"actual_subblocks\":2}\n";
    return 0;
}
}
int main(int argc, char** argv) {
    try { return run(argc, argv); }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
