#include "qkv_pair.hpp"
#include "checkpoint.hpp"
#include "matrix_timing.hpp"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "llama.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace strata::deepseek_v4;
std::vector<uint8_t> read_weight(const char * first, const char * name, int rows) {
    std::string path(first);
    const auto segment = path.find("00001-of-00004");
    if (segment == std::string::npos) throw std::runtime_error("Expected first of four GGUF shards");
    for (int shard = 1; shard <= 4; ++shard) {
        path[segment + 4] = char('0' + shard);
        std::unique_ptr<gguf_context, decltype(&gguf_free)> file(gguf_init_from_file(path.c_str(), {true, nullptr}), gguf_free);
        if (!file) throw std::runtime_error("Actual GGUF header missing");
        const auto id = gguf_find_tensor(file.get(), name);
        if (id < 0) continue;
        const auto ne = gguf_get_tensor_ne(file.get(), id);
        if (gguf_get_tensor_type(file.get(), id) != GGML_TYPE_Q8_0 || ne[0] != 4096 || ne[1] != rows ||
            gguf_get_tensor_size(file.get(), id) != 4096ull * rows * 34 / 32)
            throw std::runtime_error(std::string("Actual Q/KV weight format differs: ") + name);
        std::vector<uint8_t> bytes(gguf_get_tensor_size(file.get(), id));
        std::ifstream source(path, std::ios::binary);
        source.seekg(gguf_get_data_offset(file.get()) + gguf_get_tensor_offset(file.get(), id));
        source.read(reinterpret_cast<char *>(bytes.data()), bytes.size());
        if (!source) throw std::runtime_error("Q/KV weight payload truncated");
        return bytes;
    }
    throw std::runtime_error(std::string("Actual Q/KV weight absent: ") + name);
}
struct Graph {
    ggml_backend_t backend;
    ggml_context * meta = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * q_output = nullptr, * kv_output = nullptr, * packed = nullptr;
    Graph(ggml_backend_t b, const std::vector<float>& x, const std::vector<uint8_t>& q_data,
          const std::vector<uint8_t>& kv_data, bool fused) : backend(b) {
        meta = ggml_init({1ull << 20, nullptr, true});
        if (!meta) throw std::runtime_error("Q/KV pair graph metadata absent");
        auto * input = ggml_new_tensor_2d(meta, GGML_TYPE_F32, 4096, 2);
        auto * q = ggml_new_tensor_2d(meta, GGML_TYPE_Q8_0, 4096, 1024);
        auto * kv = ggml_new_tensor_2d(meta, GGML_TYPE_Q8_0, 4096, 512);
        graph = ggml_new_graph_custom(meta, 32, false);
        if (fused) {
            packed = build_qkv_pair(meta, input, q, kv);
            ggml_set_output(packed);
            ggml_build_forward_expand(graph, packed);
        } else {
            q_output = ggml_mul_mat(meta, q, input);
            kv_output = ggml_mul_mat(meta, kv, input);
            ggml_set_output(q_output); ggml_set_output(kv_output);
            ggml_build_forward_expand(graph, q_output);
            ggml_build_forward_expand(graph, kv_output);
        }
        buffer = ggml_backend_alloc_ctx_tensors(meta, backend);
        if (!buffer) throw std::runtime_error("Q/KV pair GPU allocation absent");
        ggml_backend_tensor_set(input, x.data(), 0, x.size() * sizeof(float));
        ggml_backend_tensor_set(q, q_data.data(), 0, q_data.size());
        ggml_backend_tensor_set(kv, kv_data.data(), 0, kv_data.size());
    }
    ~Graph() { ggml_backend_synchronize(backend); if (buffer) ggml_backend_buffer_free(buffer); if (meta) ggml_free(meta); }
    std::vector<float> read() {
        std::vector<float> values(1536 * 2);
        if (packed) ggml_backend_tensor_get(packed, values.data(), 0, values.size() * sizeof(float));
        else {
            std::vector<float> q(1024 * 2), kv(512 * 2);
            ggml_backend_tensor_get(q_output, q.data(), 0, q.size() * sizeof(float));
            ggml_backend_tensor_get(kv_output, kv.data(), 0, kv.size() * sizeof(float));
            for (int token = 0; token < 2; ++token) {
                std::copy_n(q.data() + token * 1024, 1024, values.data() + token * 1536);
                std::copy_n(kv.data() + token * 512, 512, values.data() + token * 1536 + 1024);
            }
        }
        return values;
    }
};
double median(std::array<double, 3> values) { std::sort(values.begin(), values.end()); return values[1]; }
}
int main(int argc, char ** argv) try {
    if (argc != 4) throw std::runtime_error("Expected first GGUF shard, actual activation fixture, Y-temp report");
    llama_backend_init();
    auto * device = ggml_backend_dev_by_name("CUDA0");
    if (!device) throw std::runtime_error("CUDA0 unavailable");
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_dev_init(device, nullptr), ggml_backend_free);
    const auto checkpoint = read_checkpoint_data(argv[2], "hc_attn_pre-0");
    if (checkpoint.metadata.type != GGML_TYPE_F32 || checkpoint.metadata.dimensions[0] != 4096 ||
        checkpoint.data.size() < 2 * 4096 * sizeof(float)) throw std::runtime_error("Actual Q/KV activation missing");
    std::vector<float> x(8192);
    std::memcpy(x.data(), checkpoint.data.data(), x.size() * sizeof(float));
    const auto q = read_weight(argv[1], "blk.0.attn_q_a.weight", 1024);
    const auto kv = read_weight(argv[1], "blk.0.attn_kv.weight", 512);
    Graph reference(backend.get(), x, q, kv, false), candidate(backend.get(), x, q, kv, true);
    const auto before = measure_matrix_graph(backend.get(), reference.graph);
    const auto a = reference.read();
    const auto after = measure_matrix_graph(backend.get(), candidate.graph);
    const auto b = candidate.read();
    double squared = 0, total = 0, maximum = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) throw std::runtime_error("Q/KV pair nonfinite");
        const double difference = double(a[i]) - b[i];
        squared += difference * difference;
        total += double(a[i]) * a[i];
        maximum = std::max(maximum, std::abs(difference));
    }
    const double relative = std::sqrt(squared / std::max(total, 1e-30));
    std::ofstream report(argv[3]);
    report << std::setprecision(12) << "{\"whole_model_loaded\":false,\"actual_qkv_weights\":true,"
           << "\"actual_two_position_activation\":true,\"reference_gpu_ms\":" << median(before)
           << ",\"candidate_gpu_ms\":" << median(after) << ",\"relative_rms\":" << relative
           << ",\"max_abs\":" << maximum << ",\"numerical_passed\":" << (relative <= 1e-3 ? "true" : "false")
           << "}\n";
    if (!report || relative > 1e-3) throw std::runtime_error("Real-weight Q/KV pair numerical gate failed");
    std::cout << "passed\n";
    return 0;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
