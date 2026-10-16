#include "device_cache_backend.hpp"
#include "device_cache.hpp"
#include "device_cache_protocol.hpp"
#include "device_cache_probe_report.hpp"
#include "expert_frequency.hpp"
#include "ggml-alloc.h"
#include "llama.h"
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <thread>
#include <string>
#include <vector>
#include <stdexcept>

using namespace strata::deepseek_v4;
int main(int argc, char** argv) {
    if (argc != 3 || (std::string(argv[2]) != "0" && std::string(argv[2]) != "1")) return 1;
    constexpr int fixture_layers = 43;
    constexpr int fixture_slots = fixture_layers * 6 + 1;
    const bool prepared = std::string(argv[2]) == "1";
    const bool frequency_enabled = expert_frequency_enabled();
    configure_device_cache_process();
    llama_backend_init();
    auto* device = ggml_backend_dev_by_name("CUDA0");
    if (!device) return 2;
    auto* backend = ggml_backend_dev_init(device, nullptr);
    if (!backend) return 3;
    auto* context = ggml_init({20480 * ggml_tensor_overhead() + ggml_graph_overhead_custom(16384, false), nullptr, true});
    auto* directory = ggml_new_tensor_1d(context, GGML_TYPE_I32, device_cache_max_layers * 256);
    auto* status = ggml_new_tensor_1d(context, GGML_TYPE_I32, 1);
    auto* counters = ggml_new_tensor_1d(context, GGML_TYPE_I64, 5);
    auto* frequency = frequency_enabled ? ggml_new_tensor_1d(context, GGML_TYPE_I32, device_frequency_words) : nullptr;
    const std::array<ggml_tensor*, 3> banks{
        ggml_new_tensor_3d(context, GGML_TYPE_MXFP4, 4096, 2048, fixture_slots),
        ggml_new_tensor_3d(context, GGML_TYPE_MXFP4, 4096, 2048, fixture_slots),
        ggml_new_tensor_3d(context, GGML_TYPE_MXFP4, 2048, 4096, fixture_slots)};
    auto* activation = ggml_new_tensor_2d(context, GGML_TYPE_F32, 4096, 1);
    std::array<ggml_tensor*, fixture_layers> ids{}, lookup{};
    auto* graph = ggml_new_graph_custom(context, 16384, false);
    ggml_tensor* input = activation;
    for (int layer = 0; layer < fixture_layers; ++layer) {
        auto* map = ggml_view_2d(context, directory, 1, 256, sizeof(int32_t), layer * 256 * sizeof(int32_t));
        ids[layer] = ggml_new_tensor_1d(context, GGML_TYPE_I32, 6);
        lookup[layer] = ggml_get_rows(context, map, ids[layer]);
        lookup[layer]->src[2] = status;
        lookup[layer]->src[3] = counters;
        lookup[layer]->src[4] = frequency;
        lookup[layer]->op_params[2] = layer;
        lookup[layer]->op_params[device_cache_tag_index] = device_cache_lookup_tag;
        // Match the real early lookup plus later stream work. Scheduler
        // optimization creates canonical Q/K/V fork/join streams below.
        ggml_build_forward_expand(graph, lookup[layer]);
        for (int queued = 0; queued < 160; ++queued) input = ggml_rms_norm(context, input, 1e-6f);
        if (layer == 1) {
            for (int queued = 0; queued < 5000; ++queued) input = ggml_rms_norm(context, input, 1e-6f);
            auto* norm = ggml_rms_norm(context, input, 1e-6f);
            std::array<ggml_tensor*, 3> branch{};
            for (auto& output : branch) {
                auto* weight = ggml_new_tensor_2d(context, GGML_TYPE_MXFP4, 4096, 4096);
                output = ggml_mul_mat(context, weight, norm);
            }
            input = ggml_add(context, ggml_add(context, branch[0], branch[1]), branch[2]);
            // D4's head geometry; first-use Attention setup occurs after the
            // second miss lookup has been submitted, as in the whole graph.
            auto* repeated = ggml_repeat_4d(context, input, 4096, 8, 1, 1);
            auto* query = ggml_permute(context, ggml_reshape_4d(context, repeated, 512, 64, 1, 1), 0, 2, 1, 3);
            auto* key = ggml_new_tensor_4d(context, GGML_TYPE_F16, 512, 512, 1, 1);
            auto* value = ggml_new_tensor_4d(context, GGML_TYPE_F16, 512, 512, 1, 1);
            auto* mask = ggml_new_tensor_4d(context, GGML_TYPE_F16, 512, 1, 1, 1);
            auto* attention = ggml_flash_attn_ext(context, query, key, value, mask, 1.0f / std::sqrt(512.0f), 0, 0);
            auto* heads = ggml_reshape_2d(context, ggml_cont(context, attention), 4096, 8);
            input = ggml_reshape_2d(context, ggml_sum_rows(context,
                ggml_cont(context, ggml_permute(context, heads, 1, 0, 2, 3))), 4096, 1);
        }
        auto* slots = ggml_reshape_2d(context, lookup[layer], 6, 1);
        auto* up = ggml_mul_mat_id(context, banks[0], input, slots);
        auto* gate = ggml_mul_mat_id(context, banks[1], input, slots);
        auto* glu = ggml_swiglu_clamp(context, gate, up, 10.0f);
        auto* down = ggml_mul_mat_id(context, banks[2], glu, slots);
        // Test fixture reduction only; both expert chains use canonical math.
        input = ggml_reshape_2d(context, ggml_sum_rows(context,
            ggml_cont(context, ggml_permute(context, down, 1, 0, 2, 3))), 4096, 1);
        ggml_build_forward_expand(graph, input);
    }
    auto* guarded = ggml_clamp(context, input, -INFINITY, INFINITY);
    guarded->src[1] = status;
    guarded->op_params[device_cache_tag_index] = device_cache_guard_tag;
    ggml_build_forward_expand(graph, guarded);
    auto* buffer = ggml_backend_alloc_ctx_tensors(context, backend);
    if (!buffer) return 4;
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (auto* tensor = ggml_get_first_tensor(context); tensor; tensor = ggml_get_next_tensor(context, tensor))
        if (!tensor->view_src && tensor->op == GGML_OP_NONE) ggml_backend_tensor_memset(tensor, 0, 0, ggml_nbytes(tensor));
    const std::vector<float> initial(4096, 1.0f);
    ggml_backend_tensor_set(activation, initial.data(), 0, initial.size() * sizeof(float));
    ExpertFrequency frequency_reference(43, 256);
    std::vector<uint32_t> score_state(device_frequency_words);
    if (frequency_enabled) {
        score_state[0] = 65535;
        score_state[device_frequency_scores - 1] = 22222;
        score_state[device_frequency_scores] = 65533;
        frequency_reference.import_state(score_state.data(), score_state.size());
        ggml_backend_tensor_set(frequency, score_state.data(), 0, score_state.size()*sizeof(uint32_t));
    }
    auto* cpu_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    auto* cpu_backend = cpu_device ? ggml_backend_dev_init(cpu_device, nullptr) : nullptr;
    if (!cpu_backend) return 5;
    ggml_backend_t backends[]{backend, cpu_backend};
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        if (!ggml_backend_dev_supports_op(device, ggml_graph_node(graph, i))) {
            std::cerr << "CUDA fixture unsupported: " << ggml_op_name(ggml_graph_node(graph, i)->op) << '\n';
            return 5;
        }
    }
    auto* scheduler = ggml_backend_sched_new(backends, nullptr, 2, 16384, false, true);
    if (!scheduler || !ggml_backend_sched_alloc_graph(scheduler, graph)) return 5;
    std::ofstream report(argv[1]);
    report << "{\"model_loaded\":false,\"layers\":43,\"gpu_frequency\":" << (frequency_enabled ? "true" : "false")
           << ",\"enqueue_pressure_nodes\":11880,\"head_size\":512,\"query_heads\":64,\"kv_heads\":1,\"kv_columns\":512,\"all_streams_prepared\":"
           << (prepared ? "true" : "false") << ",\"cases\":[";
    bool passed = true;
    try {
        DeviceCacheDma dma(backend, nullptr, prepared);
        const uintptr_t packet_pointer = dma.device_packet();
        const uint64_t timeout = dma.timeout_ns();
        for (auto* tensor : lookup) {
            std::memcpy(tensor->op_params, &packet_pointer, sizeof(packet_pointer));
            std::memcpy(tensor->op_params + 4, &timeout, sizeof(timeout));
        }
        std::array<int32_t, device_cache_max_layers * 256> table;
        table.fill(-1);
        dma.commit(directory->data, table);
        const size_t expert_bytes = banks[0]->nb[2];
        std::vector<uint8_t> expected_bank(size_t(ggml_nbytes(banks[0])), 0);
        for (int test = 0; test < 3; ++test) {
            std::array<std::array<int32_t, 6>, fixture_layers> selected{}, expected_slots{};
            for (int layer = 0; layer < fixture_layers; ++layer) {
                selected[layer] = test != 0 ? (layer == 0 ? std::array<int32_t,6>{12,13,14,15,16,17} :
                    std::array<int32_t,6>{18,19,20,21,22,23}) : (layer == 0 ? std::array<int32_t,6>{0,0,1,2,3,4} :
                    std::array<int32_t,6>{6,7,8,9,10,11});
                for (int rank = 0; rank < 6; ++rank) expected_slots[layer][rank] =
                    test != 0 ? (layer == 0 && rank == 0 ? fixture_slots - 1 : layer * 6 + rank - 1) :
                                (layer == 0 && rank == 1 ? 0 : layer * 6 + rank - (layer == 0 && rank > 1));
            }
            std::vector<std::vector<uint32_t>> expected_scores(fixture_layers,std::vector<uint32_t>(device_frequency_words));
            if (frequency_enabled) for (int layer = 0; layer < fixture_layers; ++layer) {
                frequency_reference.observe(layer, selected[layer].data(), 6);
                frequency_reference.export_state(expected_scores[layer].data(), device_frequency_words);
            }
            for (int layer = 0; layer < fixture_layers; ++layer) ggml_backend_tensor_set(ids[layer], selected[layer].data(), 0, 24);
            const int32_t empty = 0;
            const std::array<uint64_t, 5> zero{};
            ggml_backend_tensor_set(status, &empty, 0, sizeof(empty));
            ggml_backend_tensor_set(counters, zero.data(), 0, sizeof(zero));
            const auto before_costs = dma.costs();
            std::atomic<bool> ready{false}, stop{false};
            uint64_t service_ns = 0, services = 0;
            std::string worker_error;
            uint32_t seen = dma.packet()->sequence;
            std::thread worker([&] {
                try {
                    dma.bind_thread();
                    ready.store(true, std::memory_order_release);
                    while (!stop.load(std::memory_order_acquire)) {
                        const uint32_t sequence = dma.packet()->sequence;
                        if (sequence == seen) { std::this_thread::yield(); continue; }
                        std::atomic_thread_fence(std::memory_order_acquire);
                        const int layer = int(dma.packet()->layer);
                        if (layer < 0 || layer >= fixture_layers || std::memcmp(dma.packet()->ids, selected[layer].data(), 24))
                            throw std::runtime_error("Second-packet identity differs");
                        const auto start = std::chrono::steady_clock::now();
                        if (frequency_enabled) {
                            std::vector<uint32_t> observed(device_frequency_words);
                            dma.read_miss_frequency(observed);
                            if (observed != expected_scores[layer]) throw std::runtime_error("GPU frequency miss snapshot differs");
                        }
                        const std::vector<uint8_t> payload(expert_bytes, uint8_t(0xa5 + test + layer));
                        for (int slot : expected_slots[layer]) {
                            for (int bank_layer = 0; bank_layer < fixture_layers; ++bank_layer)
                                for (int id = 0; id < 256; ++id)
                                    if (table[bank_layer * 256 + id] == slot) table[bank_layer * 256 + id] = -1;
                            std::memset(expected_bank.data() + slot * expert_bytes, payload[0], expert_bytes);
                        }
                        for (auto* role : banks) for (int rank = 0; rank < 6; ++rank)
                            dma.upload(role, payload.data(), expected_slots[layer][rank] * expert_bytes, expert_bytes);
                        for (int rank = 0; rank < 6; ++rank) table[layer * 256 + selected[layer][rank]] = expected_slots[layer][rank];
                        dma.commit(directory->data, table);
                        service_ns += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - start).count());
                        ++services;
                        seen = sequence;
                        std::atomic_thread_fence(std::memory_order_release);
                        dma.packet()->response = sequence;
                        std::atomic_thread_fence(std::memory_order_seq_cst);
                    }
                } catch (const std::exception& error) { worker_error = error.what(); ready.store(true); }
            });
            while (!ready.load(std::memory_order_acquire)) std::this_thread::yield();
            const auto start = std::chrono::steady_clock::now();
            const int graph_status = int(ggml_backend_sched_graph_compute(scheduler, graph));
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            stop.store(true, std::memory_order_release);
            worker.join(); // drains late submission/commit before inspection.
            int32_t error = -1;
            std::array<uint64_t, 5> metrics{};
            ggml_backend_tensor_get(status, &error, 0, sizeof(error));
            ggml_backend_tensor_get(counters, metrics.data(), 0, sizeof(metrics));
            bool bytes_equal = true;
            for (auto* role : banks) {
                std::vector<uint8_t> readback(expected_bank.size());
                ggml_backend_tensor_get(role, readback.data(), 0, readback.size());
                bytes_equal &= readback == expected_bank;
            }
            std::vector<float> output(4096);
            ggml_backend_tensor_get(guarded, output.data(), 0, output.size() * sizeof(float));
            const bool poisoned = std::isnan(output[0]);
            const bool expect_timeout = false;
            const auto after_costs = dma.costs();
            const auto enqueue_waits = after_costs.enqueue_miss_waits - before_costs.enqueue_miss_waits;
            const auto history_bytes = after_costs.frequency_d2h_bytes - before_costs.frequency_d2h_bytes;
            const auto preuploads = after_costs.graph_preuploads - before_costs.graph_preuploads;
            bool scores_equal = true;
            if (frequency_enabled) {
                ggml_backend_tensor_get(frequency, score_state.data(), 0, score_state.size()*sizeof(uint32_t));
                scores_equal = score_state == expected_scores.back();
            }
            const bool ok = !graph_status && worker_error.empty() && bytes_equal && scores_equal &&
                (expect_timeout ? error && metrics[4] && poisoned : !error && !metrics[4] && !poisoned) &&
                (test == 2 ? services == 0 && metrics[1] == fixture_layers && enqueue_waits == 0 && history_bytes == 0 && preuploads == 0 : services == fixture_layers) &&
                (test != 1 || preuploads > 0);
            passed &= ok;
            if (test) report << ',';
            report << "{\"expected_timeout\":" << (expect_timeout ? "true" : "false")
                << ",\"case\":\"" << (test == 0 ? "cold_eager" : test == 1 ? "cold_first_graph_launch" : "all_hit_replay") << "\""
                << ",\"passed\":" << (ok ? "true" : "false") << ",\"host_services\":" << services
                << ",\"timeouts\":" << metrics[4] << ",\"sequence\":" << dma.packet()->sequence
                << ",\"response\":" << dma.packet()->response << ",\"service_ns\":" << service_ns
                << ",\"payload_byte_exact\":" << (bytes_equal ? "true" : "false")
                << ",\"logits_poisoned\":" << (poisoned ? "true" : "false") << ",\"wall_ms\":" << ms
                << ",\"enqueue_miss_waits_delta\":" << enqueue_waits
                << ",\"frequency_state_exact\":" << (scores_equal ? "true" : "false")
                << ",\"history_d2h_bytes_delta\":" << history_bytes
                << ",\"graph_preuploads_delta\":" << preuploads
                << ",\"enqueue_miss_wait_ns_delta\":" << after_costs.enqueue_miss_wait_ns - before_costs.enqueue_miss_wait_ns
                << ",\"api_costs\":";
            write_cache_api_costs(report, dma.costs());
            report << '}';
            if (metrics[4]) break;
        }
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; passed = false; }
    report << "],\"passed\":" << (passed ? "true" : "false") << "}\n";
    ggml_backend_sched_free(scheduler);
    ggml_backend_buffer_free(buffer);
    ggml_free(context);
    ggml_backend_free(cpu_backend);
    ggml_backend_free(backend);
    return passed ? 0 : 1;
}
