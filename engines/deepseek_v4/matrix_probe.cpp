#include "checkpoint.hpp"
#include "matrix_timing.hpp"
#include "ggml-alloc.h"
#include "llama.h"
#include "llama-model.h"
#include <algorithm>
#include <array>
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
enum Input { norm, experts_down, shared_down, qr, wo_b };
struct Case { const char* name; const char* tensor; ggml_type type; Input input; bool routed; bool fused = false; };
constexpr Case cases[] = {
    {"expert_gate", "ffn_gate_exps", GGML_TYPE_MXFP4, norm, true},
    {"expert_up", "ffn_up_exps", GGML_TYPE_MXFP4, norm, true},
    {"expert_down", "ffn_down_exps", GGML_TYPE_MXFP4, experts_down, true},
    {"shared_gate", "ffn_gate_shexp", GGML_TYPE_Q8_0, norm, false},
    {"shared_up", "ffn_up_shexp", GGML_TYPE_Q8_0, norm, false},
    {"shared_down", "ffn_down_shexp", GGML_TYPE_Q8_0, shared_down, false},
    {"query_b", "attn_q_b", GGML_TYPE_Q8_0, qr, false},
    {"output_b", "attn_output_b", GGML_TYPE_Q8_0, wo_b, false},
    {"expert_gate_up_clamp", "ffn_up_exps", GGML_TYPE_MXFP4, norm, true, true},
    {"shared_gate_up_clamp", "ffn_up_shexp", GGML_TYPE_Q8_0, norm, false, true},
};
uint64_t hash(const std::vector<uint8_t>& bytes) {
    uint64_t value = 14695981039346656037ull;
    for (auto byte : bytes) { value ^= byte; value *= 1099511628211ull; }
    return value;
}
std::string hex(uint64_t value) { std::ostringstream s; s << std::hex << value; return s.str(); }
void environment(const char* name, const char* value) {
#ifdef _WIN32
    if (_putenv_s(name, value)) throw std::runtime_error("matrix probe environment failed");
#else
    if (setenv(name, value, 1)) throw std::runtime_error("matrix probe environment failed");
#endif
}
struct Inputs {
    CheckpointData activation, glu, ids, query, output;
    int tokens = 0;
    std::vector<int32_t> physical_ids, original_ids, unique_ids;
    explicit Inputs(const std::filesystem::path& directory,int selected_tokens=0) {
        activation = read_checkpoint_data(directory, "ffn_norm-0");
        glu = read_checkpoint_data(directory, "ffn_moe_swiglu_limited-0");
        ids = read_checkpoint_data(directory, "ffn_moe_topk-0");
        query = read_checkpoint_data(directory, "qr_norm-0");
        output = read_checkpoint_data(directory, "attn_wo_a-0");
        tokens = int(activation.metadata.dimensions[1]);
        if ((tokens != 1 && tokens != 5) || activation.metadata.type != GGML_TYPE_F32 ||
            activation.metadata.dimensions != std::array<int64_t, 4>{4096, tokens, 1, 1} ||
            glu.metadata.type != GGML_TYPE_F32 || glu.metadata.dimensions != std::array<int64_t, 4>{2048, 6, tokens, 1} ||
            ids.metadata.type != GGML_TYPE_I32 || ids.metadata.dimensions != std::array<int64_t, 4>{6, tokens, 1, 1} ||
            query.metadata.type != GGML_TYPE_F32 || query.metadata.dimensions != std::array<int64_t, 4>{1024, tokens, 1, 1} ||
            output.metadata.type != GGML_TYPE_F32 || output.metadata.dimensions != std::array<int64_t, 4>{1024, tokens, 8, 1})
            throw std::runtime_error("matrix probe reference must be real layer-zero PP5 or TG1");
        if(selected_tokens){
            if(selected_tokens!=2||tokens!=5)throw std::runtime_error("Dense pair replay requires two actual PP5 positions");
            std::vector<uint8_t> paired(1024*2*8*sizeof(float));
            for(int group=0;group<8;++group)std::memcpy(paired.data()+group*1024*2*sizeof(float),output.data.data()+group*1024*5*sizeof(float),1024*2*sizeof(float));
            output.data=std::move(paired);activation.data.resize(4096*2*sizeof(float));glu.data.resize(2048*6*2*sizeof(float));
            ids.data.resize(6*2*sizeof(int32_t));query.data.resize(1024*2*sizeof(float));tokens=2;
        }
        original_ids.resize(6 * tokens); physical_ids.resize(6 * tokens);
        std::memcpy(original_ids.data(), ids.data.data(), ids.data.size());
        std::array<int32_t, 256> slots; slots.fill(-1);
        for (int token = 0; token < tokens; ++token) {
            std::array<bool, 256> seen{};
            for (int rank = 0; rank < 6; ++rank) {
                const auto index = token * 6 + rank;
                const int id = original_ids[index];
                if (id < 0 || id >= 256 || seen[id]) throw std::runtime_error("matrix probe route IDs invalid");
                seen[id] = true;
                if (slots[id] < 0) { slots[id] = int(unique_ids.size()); unique_ids.push_back(id); }
                physical_ids[index] = slots[id];
            }
        }
    }
    std::vector<uint8_t> data(Input kind) const {
        if (kind == norm) return activation.data;
        if (kind == experts_down) return glu.data;
        if (kind == qr) return query.data;
        if (kind == shared_down) {
            // Same actual activation slice for both geometries. This is rank0
            // of routed GLU, not an assertion about the shared expert's GLU.
            std::vector<uint8_t> result(2048 * tokens * sizeof(float));
            for (int token = 0; token < tokens; ++token)
                std::memcpy(result.data() + token * 2048 * sizeof(float),
                            glu.data.data() + token * 6 * 2048 * sizeof(float), 2048 * sizeof(float));
            return result;
        }
        std::vector<uint8_t> result(8192 * tokens * sizeof(float));
        for (int token = 0; token < tokens; ++token)
            for (int group = 0; group < 8; ++group)
                std::memcpy(result.data() + (token * 8192 + group * 1024) * sizeof(float),
                            output.data.data() + (group * tokens + token) * 1024 * sizeof(float), 1024 * sizeof(float));
        return result;
    }
};
struct Graph {
    ggml_backend_t backend;
    ggml_context * weights = nullptr, * context = nullptr;
    ggml_backend_buffer_t weight_buffer = nullptr, data_buffer = nullptr;
    explicit Graph(ggml_backend_t b) : backend(b) {}
    ~Graph() {
        ggml_backend_synchronize(backend);
        if (data_buffer) ggml_backend_buffer_free(data_buffer);
        if (weight_buffer) ggml_backend_buffer_free(weight_buffer);
        if (context) ggml_free(context);
        if (weights) ggml_free(weights);
    }
};
ggml_tensor* model_tensor(llama_model& model, const Case& item) {
    const std::string name = std::string("blk.0.") + item.tensor + ".weight";
    auto found = std::find_if(model.tensors_by_name.begin(), model.tensors_by_name.end(),
                              [&](const auto& t) { return t.first == name; });
    if (found == model.tensors_by_name.end()) throw std::runtime_error("actual GGUF matrix missing: " + name);
    auto* tensor = found->second;
    if (tensor->type != item.type || !tensor->data || !tensor->buffer ||
        !ggml_backend_buffer_is_host(tensor->buffer) || !ggml_is_contiguous(tensor) ||
        tensor->ne[2] != (item.routed ? 256 : 1) || tensor->ne[3] != 1)
        throw std::runtime_error("actual GGUF matrix format/storage differs");
    return tensor;
}
void run_case(llama_model& model, ggml_backend_t backend, const Inputs& input, const Case& item,
              const std::filesystem::path& directory, std::ostream& report, int index, bool l2_cold) {
    auto* original = model_tensor(model, item);
    const int64_t channels = item.routed ? int64_t(input.unique_ids.size()) : 1;
    const auto activation = input.data(item.input);
    if (activation.size() != std::size_t(original->ne[0] * input.tokens * (item.input == experts_down ? 6 : 1) * 4))
        throw std::runtime_error("matrix probe activation shape differs");
    const auto weight_bytes = std::size_t(original->nb[2] * channels);
    if (weight_bytes > (256ull << 20)) throw std::runtime_error("matrix probe weight slice exceeds bound");
    std::vector<uint8_t> payload(weight_bytes);
    for (int64_t slot = 0; slot < channels; ++slot) {
        const int expert = item.routed ? input.unique_ids[slot] : 0;
        std::memcpy(payload.data() + slot * original->nb[2],
                    static_cast<const uint8_t*>(original->data) + expert * original->nb[2], original->nb[2]);
    }
    Graph owner(backend);
    owner.weights = ggml_init({4 * ggml_tensor_overhead(), nullptr, true});
    owner.context = ggml_init({1ull << 20, nullptr, true});
    if (!owner.weights || !owner.context) throw std::runtime_error("matrix probe context allocation failed");
    auto* weight = ggml_new_tensor_3d(owner.weights, item.type, original->ne[0], original->ne[1], channels);
    auto* act = item.routed ? ggml_new_tensor_3d(owner.context, GGML_TYPE_F32, original->ne[0],
                   item.input == experts_down ? 6 : 1, input.tokens) :
                   ggml_new_tensor_2d(owner.context, GGML_TYPE_F32, original->ne[0], input.tokens);
    auto* ids = item.routed ? ggml_new_tensor_2d(owner.context, GGML_TYPE_I32, 6, input.tokens) : nullptr;
    auto* output = ids ? ggml_mul_mat_id(owner.context, weight, act, ids) : ggml_mul_mat(owner.context, weight, act);
    model.prec_policy.apply(output);
    std::vector<uint8_t> gate_payload;
    ggml_tensor* gate_weight = nullptr;
    if (item.fused) {
        const Case gate_case{"gate", item.routed ? "ffn_gate_exps" : "ffn_gate_shexp", item.type, norm, item.routed};
        auto* source = model_tensor(model, gate_case);
        if (source->ne[0] != original->ne[0] || source->ne[1] != original->ne[1] || source->nb[2] != original->nb[2])
            throw std::runtime_error("matrix probe fused gate shape differs");
        gate_payload.resize(weight_bytes);
        for (int64_t slot = 0; slot < channels; ++slot)
            std::memcpy(gate_payload.data() + slot * source->nb[2],
                static_cast<const uint8_t*>(source->data) + (item.routed ? input.unique_ids[slot] : 0) * source->nb[2], source->nb[2]);
        gate_weight = ggml_new_tensor_3d(owner.weights, item.type, source->ne[0], source->ne[1], channels);
        auto* gate = ids ? ggml_mul_mat_id(owner.context, gate_weight, act, ids) : ggml_mul_mat(owner.context, gate_weight, act);
        model.prec_policy.apply(gate);
        const float limit = item.routed ? model.hparams.swiglu_clamp_exp[0] : model.hparams.swiglu_clamp_shexp[0];
        if (limit != 10.0f) throw std::runtime_error("matrix probe actual clamp differs");
        output = ggml_swiglu_clamp(owner.context, gate, output, limit);
    }
    ggml_set_output(output);
    auto* graph = ggml_new_graph_custom(owner.context, 8, false);
    ggml_build_forward_expand(graph, output);
    owner.weight_buffer = ggml_backend_alloc_ctx_tensors(owner.weights, backend);
    owner.data_buffer = ggml_backend_alloc_ctx_tensors(owner.context, backend);
    if (!owner.weight_buffer || !owner.data_buffer) throw std::runtime_error("matrix probe device allocation failed");
    ggml_backend_buffer_set_usage(owner.weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_set(weight, payload.data(), 0, payload.size());
    if (gate_weight) ggml_backend_tensor_set(gate_weight, gate_payload.data(), 0, gate_payload.size());
    ggml_backend_tensor_set(act, activation.data(), 0, activation.size());
    if (ids) ggml_backend_tensor_set(ids, input.physical_ids.data(), 0, input.physical_ids.size() * sizeof(int32_t));
    const auto samples = measure_matrix_graph(backend, graph, l2_cold);
    const auto bytes = ggml_nbytes(output);
    if (bytes > (2ull << 20)) throw std::runtime_error("matrix probe output exceeds bound");
    std::vector<uint8_t> result(bytes);
    ggml_backend_tensor_get(output, result.data(), 0, bytes);
    for (std::size_t at = 0; at < result.size(); at += sizeof(float)) {
        float value; std::memcpy(&value, result.data() + at, sizeof(value));
        if (!std::isfinite(value)) throw std::runtime_error("matrix probe non-finite output");
    }
    const std::string filename = std::string(item.name) + ".bin";
    std::ofstream binary(directory / filename, std::ios::binary);
    binary.write(reinterpret_cast<const char*>(result.data()), std::streamsize(result.size()));
    if (!binary) throw std::runtime_error("matrix probe output write failed");
    auto sorted = samples; std::sort(sorted.begin(), sorted.end());
    if (index) report << ",\n";
    report << "{\"name\":" << std::quoted(item.name) << ",\"type\":" << int(item.type)
           << ",\"columns\":" << original->ne[0] << ",\"rows\":" << original->ne[1]
           << ",\"tokens\":" << input.tokens << ",\"routed\":" << (item.routed ? "true" : "false")
           << ",\"fused_graph\":" << (item.fused ? "true" : "false")
           << ",\"activation_origin\":" << std::quoted(item.input == shared_down ? "routed_GLU_rank0_shape_replay" : "actual_layer_checkpoint")
           << ",\"weight_bytes\":" << payload.size() << ",\"weight_fnv64\":" << std::quoted(hex(hash(payload)))
           << ",\"gate_weight_fnv64\":" << std::quoted(hex(hash(gate_payload)))
           << ",\"input_fnv64\":" << std::quoted(hex(hash(activation)))
           << ",\"output_fnv64\":" << std::quoted(hex(hash(result))) << ",\"output_file\":" << std::quoted(filename)
           << ",\"output_bytes\":" << bytes << ",\"gpu_service_ms\":" << sorted[1]
           << ",\"l2_cold\":" << (l2_cold ? "true" : "false")
           << ",\"gpu_service_samples_ms\":[" << samples[0] << ',' << samples[1] << ',' << samples[2] << "]}";
}
int run(int argc, char** argv) {
    std::string model_path, reference, directory, wide, load = "0", cold = "0", routed = "0";int selected_tokens=0;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 == argc) throw std::runtime_error("matrix probe requires flag/value pairs");
        const std::string flag = argv[i], value = argv[i + 1];
        if (flag == "--model") model_path = value;
        else if (flag == "--reference-dir") reference = value;
        else if (flag == "--output-dir") directory = value;
        else if (flag == "--wide") wide = value;
        else if (flag == "--fast-load") load = value;
        else if (flag == "--l2-cold") cold = value;
        else if (flag == "--only-routed") routed = value;
        else if (flag == "--tokens") selected_tokens=std::stoi(value);
        else throw std::runtime_error("unknown matrix probe option");
    }
    if (model_path.empty() || reference.empty() || directory.empty() || (wide != "0" && wide != "1") ||
        (load != "0" && load != "1") || (cold != "0" && cold != "1") || (routed != "0" && routed != "1"))
        throw std::runtime_error("matrix probe requires model/reference/output/wide0|1");
    std::filesystem::create_directories(directory);
    if (!std::filesystem::is_empty(directory)) throw std::runtime_error("matrix probe output must be empty");
    environment("STRATA_DSV4_MMVQ_WIDE", wide.c_str());
    environment("STRATA_DSV4_MXFP4_LOAD", load.c_str());
    environment("GGML_CUDA_DISABLE_GRAPHS", "1");
    const Inputs input(reference,selected_tokens);
    llama_backend_init();
    ggml_backend_dev_t devices[] = {nullptr};
    auto parameters = llama_model_default_params();
    parameters.devices = devices; parameters.n_gpu_layers = 0; parameters.split_mode = LLAMA_SPLIT_MODE_NONE;
    parameters.load_mode = LLAMA_LOAD_MODE_MMAP; parameters.lazy_mode = LLAMA_LAZY_MODE_ON;
    parameters.use_extra_bufts = false; parameters.load_mtp = false;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(model_path.c_str(), parameters), llama_model_free);
    if (!model || model->arch != LLM_ARCH_DEEPSEEK4) throw std::runtime_error("matrix probe actual model unavailable");
    auto* device = ggml_backend_dev_by_name("CUDA0");
    if (!device) throw std::runtime_error("matrix probe CUDA0 unavailable");
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
        ggml_backend_dev_init(device, nullptr), ggml_backend_free);
    if (!backend) throw std::runtime_error("matrix probe CUDA backend initialization failed");
    std::ofstream report(std::filesystem::path(directory) / "report.json");
    const char* scope = cold == "1" ?
        "canonical quantizer+matrix; CUDA events; three warmups, three16-replay windows; four L2 capacities read before each replay, eviction excluded; no full-model speedup claim" :
        "canonical quantizer+matrix; CUDA events; three warmups, three32-replay windows; isolated weights may remain in L2; no HBM or full-model speedup claim";
    report << std::setprecision(10) << "{\"status\":\"passed\",\"wide\":" << wide
           << ",\"fast_load\":" << load << ",\"l2_cold\":" << cold
           << ",\"timing_scope\":" << std::quoted(scope) << ",\"cases\":[\n";
    int count = 0;
    for (const auto& item : cases) {
        if (routed == "1" && (!item.routed || item.fused)) continue;
        run_case(*model, backend.get(), input, item, directory, report, count++, cold == "1");
    }
    report << "]}\n";
    if (!report) throw std::runtime_error("matrix probe report write failed");
    std::cout << "{\"status\":\"passed\",\"cases\":" << count << ",\"tokens\":" << input.tokens << "}\n";
    return 0;
}
}
int main(int argc, char** argv) {
    try { return run(argc, argv); }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
