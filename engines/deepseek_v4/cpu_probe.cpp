#include "cpu_probe.hpp"
#include "canonical_expert_graph.hpp"
#include "expert_route_slice.hpp"

#include "ggml-cpu.h"
#include "llama.h"
#include "llama-model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <type_traits>

namespace strata::deepseek_v4 {
namespace {

using Clock = std::chrono::steady_clock;
const char* expert_names[] = {
    "ffn_moe_gate-0", "ffn_moe_up-0", "ffn_moe_swiglu_limited-0", "ffn_moe_down-0"
};

double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

uint64_t checksum(const std::vector<uint8_t>& bytes) {
    uint64_t value = 14695981039346656037ull;
    for (uint8_t byte : bytes) { value ^= byte; value *= 1099511628211ull; }
    return value;
}

std::string filename(std::size_t index) {
    std::ostringstream name;
    name << "checkpoint-" << std::setfill('0') << std::setw(3) << index << ".bin";
    return name.str();
}

bool read_line(std::istream& input, std::string& line) {
    if (!std::getline(input, line)) return false;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    return true;
}

std::vector<Checkpoint> read_manifest(const std::filesystem::path& directory) {
    std::ifstream input(directory / "checkpoints.tsv");
    std::string line;
    if (!read_line(input, line) || line != "strata-deepseek-v4-checkpoints-v1")
        throw std::runtime_error("CPU probe requires a v1 reference checkpoint manifest");
    std::vector<Checkpoint> result;
    uint64_t total = 0;
    while (read_line(input, line)) {
        std::istringstream row(line);
        Checkpoint value;
        std::size_t index = 0;
        row >> index >> value.name >> value.dtype >> value.type;
        for (auto& dim : value.dimensions) row >> dim;
        for (auto& stride : value.source_strides) row >> stride;
        row >> value.bytes >> std::hex >> value.checksum >> std::dec >> value.file;
        std::string extra;
        if (!row || row >> extra || index != result.size() || value.file != filename(index) ||
            value.bytes > checkpoint_tensor_cap || result.size() >= 48)
            throw std::runtime_error("CPU probe reference manifest is malformed or oversized");
        uint64_t logical = value.type == GGML_TYPE_F32 || value.type == GGML_TYPE_I32 ? 4 :
                           value.type == GGML_TYPE_F16 || value.type == GGML_TYPE_BF16 ? 2 : 0;
        if (!logical) throw std::runtime_error("CPU probe reference has an unsupported scalar dtype");
        for (int64_t dim : value.dimensions) {
            if (dim < 1 || logical > checkpoint_tensor_cap / uint64_t(dim))
                throw std::runtime_error("CPU probe reference dimensions exceed the checkpoint bound");
            logical *= uint64_t(dim);
        }
        if (logical != value.bytes || total > checkpoint_total_cap - value.bytes)
            throw std::runtime_error("CPU probe reference logical size is invalid");
        total += value.bytes;
        result.push_back(std::move(value));
    }
    if (result.empty()) throw std::runtime_error("CPU probe reference manifest is empty");
    return result;
}

const Checkpoint& find(const std::vector<Checkpoint>& values, const std::string& name) {
    const auto count = std::count_if(values.begin(), values.end(), [&](const Checkpoint& value) {
        return value.name == name;
    });
    if (count != 1)
        throw std::runtime_error("CPU probe requires exactly one checkpoint named " + name +
                                 "; found " + std::to_string(count));
    const auto entry = std::find_if(values.begin(), values.end(), [&](const Checkpoint& value) {
        return value.name == name;
    });
    return *entry;
}

std::vector<uint8_t> read_binary(const std::filesystem::path& directory, const Checkpoint& value) {
    if (std::filesystem::file_size(directory / value.file) != value.bytes)
        throw std::runtime_error("CPU probe checkpoint file size differs from its manifest");
    std::ifstream input(directory / value.file, std::ios::binary);
    std::vector<uint8_t> data(std::size_t(value.bytes));
    input.read(reinterpret_cast<char*>(data.data()), std::streamsize(data.size()));
    if (!input || checksum(data) != value.checksum)
        throw std::runtime_error("CPU probe checkpoint checksum mismatch");
    if (value.type == GGML_TYPE_F32) {
        for (std::size_t offset = 0; offset < data.size(); offset += sizeof(float)) {
            float scalar;
            std::memcpy(&scalar, data.data() + offset, sizeof(scalar));
            if (!std::isfinite(scalar)) throw std::runtime_error("CPU probe checkpoint has nonfinite F32 data");
        }
    }
    return data;
}

std::string difference(const Checkpoint& left, const Checkpoint& right,
                       const std::filesystem::path& reference, const std::filesystem::path& output) {
    if (left.type != right.type || left.dimensions != right.dimensions || left.bytes != right.bytes)
        return left.name + ": metadata";
    const auto a = read_binary(reference, left), b = read_binary(output, right);
    const auto mismatch = std::mismatch(a.begin(), a.end(), b.begin());
    if (mismatch.first == a.end()) return {};
    return left.name + ": byte " + std::to_string(std::distance(a.begin(), mismatch.first));
}

bool recorded_host_storage(const std::filesystem::path& directory, bool allow_device_reference) {
    std::ifstream input(directory / "checkpoint-backends.tsv");
    if (!input) return false;
    std::string line;
    if (!read_line(input, line) || line != "strata-deepseek-v4-checkpoint-backends-v1" ||
        !read_line(input, line) || line != "index\tname\tbuffer_type\tis_host\tbuffer_usage")
        throw std::runtime_error("CPU probe reference storage metadata is malformed");
    std::set<std::string> host_names;
    bool all_host = true;
    while (read_line(input, line)) {
        std::istringstream row(line);
        std::size_t index;
        std::string name, buffer, extra;
        int host, usage;
        if (!(row >> index >> name >> buffer >> host >> usage) || row >> extra || (host != 0 && host != 1))
            throw std::runtime_error("CPU probe reference storage row is malformed");
        for (const char* expected : expert_names) {
            if (name == expected) {
                if (!host && !allow_device_reference)
                    throw std::runtime_error("CPU probe expert reference is not in host storage: " + name);
                all_host = all_host && host;
                if (!host_names.insert(name).second) throw std::runtime_error("duplicate CPU probe storage row");
            }
        }
    }
    if (host_names.size() != std::size(expert_names))
        throw std::runtime_error("CPU probe reference lacks expert storage metadata");
    return all_host;
}

ggml_tensor* replay_input(ggml_context* context, const Checkpoint& value) {
    ggml_tensor* tensor = ggml_new_tensor(context, ggml_type(value.type), 4, value.dimensions.data());
    ggml_set_name(tensor, value.name.c_str());
    ggml_set_input(tensor);
    return tensor;
}

void check_bank(const ggml_tensor* bank, int64_t input, int64_t output, int64_t experts) {
    if (!bank || bank->type != GGML_TYPE_MXFP4 || bank->ne[0] != input || bank->ne[1] != output ||
        bank->ne[2] != experts || bank->ne[3] != 1 || !ggml_is_contiguous(bank) ||
        !bank->buffer || !ggml_backend_buffer_is_host(bank->buffer))
        throw std::runtime_error("CPU probe requires original contiguous host MXFP4 expert banks");
}

void free_cpu_backend(ggml_backend_t backend) {
    ggml_backend_cpu_set_threadpool(backend, nullptr);
    ggml_backend_free(backend);
}

} // namespace

bool run_cpu_probe(const CpuProbeOptions& options, CpuProbeResult& result) noexcept {
    result = {};
    result.n_threads = options.n_threads;
    result.raw_experts_only = options.raw_experts_only;
    try {
        if (options.model.empty() || options.reference_dir.empty() || options.output_dir.empty() ||
            options.n_threads < 1 || options.n_threads > 256)
            throw std::runtime_error("CPU probe requires model/reference/output and threads in 1..256");
        std::filesystem::create_directories(options.output_dir);
        if (!std::filesystem::is_empty(options.output_dir))
            throw std::runtime_error("CPU probe output directory must be empty");
        const auto reference = read_manifest(options.reference_dir);
        result.reference_host_storage_recorded = recorded_host_storage(options.reference_dir, options.allow_device_reference);
        const auto& activation = find(reference, "ffn_norm-0");
        const auto& logits = find(reference, "ffn_moe_logits-0");
        const auto& original_topk = find(reference, "ffn_moe_topk-0");
        const auto original_topk_data = read_binary(options.reference_dir, original_topk);
        result.original_router_ranks = expert_route_ranks(options.route_mask, uint32_t(original_topk.dimensions[0]));
        if (!options.raw_experts_only && result.original_router_ranks.size() != std::size_t(original_topk.dimensions[0]))
            throw std::runtime_error("partial cold routes require the raw expert provider graph");
        const auto selected_routes = slice_expert_routes(original_topk, original_topk_data,
                                                         result.original_router_ranks, 0);
        const auto& topk = selected_routes.metadata;
        if (activation.type != GGML_TYPE_F32 || logits.type != GGML_TYPE_F32 || topk.type != GGML_TYPE_I32 ||
            activation.dimensions[2] != 1 || activation.dimensions[3] != 1 ||
            logits.dimensions[2] != 1 || logits.dimensions[3] != 1 ||
            topk.dimensions[2] != 1 || topk.dimensions[3] != 1 ||
            activation.dimensions[1] < 1 || activation.dimensions[1] > 32 ||
            logits.dimensions[1] != activation.dimensions[1] || topk.dimensions[1] != activation.dimensions[1])
            throw std::runtime_error("CPU probe inputs must be matching bounded F32/F32/I32 matrices");
        const auto activation_data = read_binary(options.reference_dir, activation);
        const auto logits_data = read_binary(options.reference_dir, logits);
        const auto& topk_data = selected_routes.data;
        std::ostringstream identity;
        identity << "canonical-cpu-moe-replay-v1\nmodel=" << options.model;
        for (const auto* input : {&activation, &logits, &topk})
            identity << '\n' << input->name << "=" << std::hex << input->checksum << std::dec;
        identity << "\nraw_experts_only=" << options.raw_experts_only << "\nroute_mask=" << options.route_mask;
        result.input_identity = identity.str();

        llama_backend_init();
        if (!llama_supports_mmap()) throw std::runtime_error("CPU probe requires canonical mmap support");
        ggml_backend_dev_t devices[] = {nullptr};
        auto model_params = llama_model_default_params();
        model_params.devices = devices;
        model_params.n_gpu_layers = 0;
        model_params.split_mode = LLAMA_SPLIT_MODE_NONE;
        model_params.load_mode = LLAMA_LOAD_MODE_MMAP;
        model_params.lazy_mode = LLAMA_LAZY_MODE_ON;
        model_params.use_extra_bufts = false;
        model_params.load_mtp = false;
        const auto loading = Clock::now();
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(options.model.c_str(), model_params), llama_model_free);
        result.load_wall_ms = milliseconds(loading);
        if (!model || model->arch != LLM_ARCH_DEEPSEEK4 || model->layers.empty())
            throw std::runtime_error("CPU probe requires a loaded canonical DeepSeek4 model");
        const auto& hp = model->hparams;
        if (activation.dimensions[0] != hp.n_embd || logits.dimensions[0] != hp.n_expert ||
            original_topk.dimensions[0] != hp.n_expert_used(0) || !hp.dsv4_hash_layer_count ||
            hp.expert_gating_func != LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS)
            throw std::runtime_error("CPU probe inputs do not match layer-zero hash routing metadata");
        const auto& layer = model->layers[0];
        check_bank(layer.ffn_up_exps, hp.n_embd, layer.ffn_up_exps ? layer.ffn_up_exps->ne[1] : 0, hp.n_expert);
        check_bank(layer.ffn_gate_exps, hp.n_embd, layer.ffn_up_exps->ne[1], hp.n_expert);
        check_bank(layer.ffn_down_exps, layer.ffn_up_exps->ne[1], hp.n_embd, hp.n_expert);
        std::set<int32_t> distinct_experts;
        for (int64_t token = 0; token < topk.dimensions[1]; ++token) {
            std::set<int32_t> token_experts;
            for (int64_t rank = 0; rank < topk.dimensions[0]; ++rank) {
                int32_t id;
                std::memcpy(&id, topk_data.data() + sizeof(id) * (rank + token * topk.dimensions[0]), sizeof(id));
                if (id < 0 || id >= hp.n_expert || !token_experts.insert(id).second)
                    throw std::runtime_error("CPU probe captured expert ids are invalid or duplicated");
                distinct_experts.insert(id);
            }
        }
        result.effective_expert_payload_bytes = distinct_experts.size() *
            (layer.ffn_up_exps->nb[2] + layer.ffn_gate_exps->nb[2] + layer.ffn_down_exps->nb[2]);
        result.contribution_output_bytes = uint64_t(hp.n_embd) * uint64_t(topk.dimensions[0]) *
                                           uint64_t(topk.dimensions[1]) * sizeof(float);
        auto pool_params = ggml_threadpool_params_default(options.n_threads);
        using PoolType = std::remove_pointer_t<ggml_threadpool_t>;
        std::unique_ptr<PoolType, decltype(&ggml_threadpool_free)> threadpool(
            ggml_threadpool_new(&pool_params), ggml_threadpool_free);
        if (!threadpool) throw std::runtime_error("CPU probe resident threadpool initialization failed");
        using CpuType = std::remove_pointer_t<ggml_backend_t>;
        std::unique_ptr<CpuType, decltype(&free_cpu_backend)> cpu(ggml_backend_cpu_init(), free_cpu_backend);
        if (!cpu) throw std::runtime_error("CPU probe backend initialization failed");
        ggml_backend_cpu_set_n_threads(cpu.get(), options.n_threads);
        ggml_backend_cpu_set_threadpool(cpu.get(), threadpool.get());
        result.threadpool_resident = true;
        ggml_backend_t backends[] = {cpu.get()};
        using SchedType = std::remove_pointer_t<ggml_backend_sched_t>;
        std::unique_ptr<SchedType, decltype(&ggml_backend_sched_free)> scheduler(
            ggml_backend_sched_new(backends, nullptr, 1, 256, false, false), ggml_backend_sched_free);
        if (!scheduler) throw std::runtime_error("CPU probe scheduler initialization failed");
        llm_graph_result graph_result(256);
        llama_adapter_loras loras;
        llm_graph_params params{};
        params.arch = model->arch;
        params.hparams = hp;
        params.hparams.n_expert_used_arr[0] = uint32_t(topk.dimensions[0]);
        params.cparams.n_ctx = 512;
        params.cparams.n_batch = params.cparams.n_ubatch = uint32_t(activation.dimensions[1]);
        params.cparams.n_threads = params.cparams.n_threads_batch = options.n_threads;
        params.cparams.warmup = false;
        params.cparams.op_offload = false;
        params.ubatch.n_tokens = uint32_t(activation.dimensions[1]);
        params.sched = scheduler.get();
        params.backend_cpu = cpu.get();
        params.loras = &loras;
        params.moe_cache = nullptr;
        params.prec_policy = &model->prec_policy;
        params.n_outputs = 1;
        params.cb = [](const llama_ubatch&, ggml_tensor* tensor, const char* name, int layer_id) {
            ggml_format_name(tensor, "%s-%d", name, layer_id);
        };
        params.res = &graph_result;
        llm_graph_context graph_context(params);
        ggml_tensor* cur = replay_input(graph_context.ctx0, activation);
        ggml_tensor* router = replay_input(graph_context.ctx0, logits);
        ggml_tensor* selected = replay_input(graph_context.ctx0, topk);
        ggml_tensor* output = options.raw_experts_only ?
            build_canonical_expert_down(graph_context, *model, cur, selected, 0) :
            graph_context.build_moe_ffn(cur, layer.ffn_gate_inp,
            layer.ffn_up_exps, layer.ffn_gate_exps, layer.ffn_down_exps, nullptr,
            hp.n_expert, hp.n_expert_used(0), LLM_FFN_SILU, hp.expert_weights_norm,
            hp.expert_weights_scale, llama_expert_gating_func_type(hp.expert_gating_func), 0,
            router, nullptr, nullptr, nullptr, nullptr, selected);
        ggml_build_forward_expand(graph_context.gf, output);
        CheckpointCollector collector(options.output_dir);
        ggml_backend_sched_set_eval_callback(scheduler.get(), CheckpointCollector::callback, &collector);
        if (!ggml_backend_sched_alloc_graph(scheduler.get(), graph_context.gf))
            throw std::runtime_error("CPU probe graph allocation failed");
        ggml_backend_tensor_set(cur, activation_data.data(), 0, activation_data.size());
        if (!options.raw_experts_only) ggml_backend_tensor_set(router, logits_data.data(), 0, logits_data.size());
        ggml_backend_tensor_set(selected, topk_data.data(), 0, topk_data.size());
        const auto computing = Clock::now();
        const auto status = ggml_backend_sched_graph_compute(scheduler.get(), graph_context.gf);
        ggml_backend_sched_synchronize(scheduler.get());
        result.compute_wall_ms = milliseconds(computing);
        collector.write_manifest();
        result.checkpoints = collector.checkpoints();
        if (collector.failed()) throw std::runtime_error(collector.error());
        if (status != GGML_STATUS_SUCCESS) throw std::runtime_error("CPU probe canonical graph compute failed");
        for (const char* name : expert_names) {
            const auto& actual = find(result.checkpoints, name);
            if (!actual.buffer_is_host) throw std::runtime_error("CPU probe expert output is not host storage");
            const auto& source = find(reference, name);
            const auto expected = slice_expert_routes(source, read_binary(options.reference_dir, source),
                                                      result.original_router_ranks, 1);
            std::string delta;
            if (actual.type != expected.metadata.type || actual.dimensions != expected.metadata.dimensions ||
                actual.bytes != expected.metadata.bytes) delta = actual.name + ": metadata";
            else {
                const auto data = read_binary(options.output_dir, actual);
                const auto mismatch = std::mismatch(expected.data.begin(), expected.data.end(), data.begin());
                if (mismatch.first != expected.data.end())
                    delta = actual.name + ": byte " + std::to_string(std::distance(expected.data.begin(), mismatch.first));
            }
            ++result.compared_expert_stages;
            if (!delta.empty() && result.first_difference.empty()) result.first_difference = delta;
        }
        for (const auto& actual : result.checkpoints) {
            const bool routing = actual.name == "ffn_moe_probs-0" || actual.name.find("ffn_moe_weights") == 0;
            const bool weighted = actual.name == "ffn_moe_weighted-0" || actual.name == "ffn_moe_out-0";
            if (!routing && !weighted) continue;
            const auto delta = difference(find(reference, actual.name), actual, options.reference_dir, options.output_dir);
            auto& first = routing ? result.router_first_difference : result.weighted_first_difference;
            if (!delta.empty() && first.empty()) first = delta;
        }
        result.expert_stages_exact = result.first_difference.empty() && result.compared_expert_stages == 4;
        if (result.expert_stages_exact) {
            // Reuse this canonical graph unchanged; reset all captured inputs
            // before each measured pass in case allocator aliases changed them.
            ggml_backend_sched_set_eval_callback(scheduler.get(), nullptr, nullptr);
            for (int replay = 0; replay < 3; ++replay) {
                ggml_backend_tensor_set(cur, activation_data.data(), 0, activation_data.size());
                if (!options.raw_experts_only) ggml_backend_tensor_set(router, logits_data.data(), 0, logits_data.size());
                ggml_backend_tensor_set(selected, topk_data.data(), 0, topk_data.size());
                ggml_backend_sched_synchronize(scheduler.get());
                const auto start = Clock::now();
                const auto replay_status = ggml_backend_sched_graph_compute(scheduler.get(), graph_context.gf);
                ggml_backend_sched_synchronize(scheduler.get());
                const double elapsed = milliseconds(start);
                if (replay_status != GGML_STATUS_SUCCESS)
                    throw std::runtime_error("CPU probe canonical service replay failed");
                result.service_compute_wall_ms_raw.push_back(elapsed);
            }
            auto sorted = result.service_compute_wall_ms_raw;
            std::sort(sorted.begin(), sorted.end());
            result.service_compute_wall_ms = sorted[1];
            result.service_curve_available = true;
        }
        result.completed = true;
        return true;
    } catch (const std::exception& error) { result.error = error.what(); }
    catch (...) { result.error = "unknown CPU probe failure"; }
    return false;
}

} // namespace strata::deepseek_v4
