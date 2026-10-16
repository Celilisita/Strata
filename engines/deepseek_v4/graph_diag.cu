#include "graph_diag_cuda.hpp"
#include "graph_diag.hpp"
#include "cache_lease_protocol.hpp"
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace strata::deepseek_v4 {
namespace {
std::atomic<bool> active{false};
std::mutex mutex;
GraphDiagnostic diagnostic;
#ifdef USE_CUDA_GRAPH
uint32_t changed_fields(const ggml_cuda_graph::node_properties& before,
                        const ggml_cuda_graph::node_properties& after) {
    uint32_t flags = 0;
    const auto& a = before.node;
    const auto& b = after.node;
    if (a.data != b.data) flags |= graph_data;
    if (std::memcmp(a.ne, b.ne, sizeof(a.ne))) flags |= graph_shape;
    if (std::memcmp(a.nb, b.nb, sizeof(a.nb))) flags |= graph_stride;
    if (a.op != b.op || a.type != b.type) flags |= graph_operation;
    if (std::memcmp(a.op_params, b.op_params, sizeof(a.op_params))) flags |= graph_parameters;
    if (std::memcmp(a.src, b.src, sizeof(a.src))) flags |= graph_cpu_sources;
    if (std::memcmp(before.node_src_data_ptrs, after.node_src_data_ptrs, sizeof(before.node_src_data_ptrs))) flags |= graph_device_sources;
    if (std::memcmp(before.node_src_ne, after.node_src_ne, sizeof(before.node_src_ne))) flags |= graph_source_shape;
    if (std::memcmp(before.node_src_nb, after.node_src_nb, sizeof(before.node_src_nb))) flags |= graph_source_stride;
    if (a.view_src != b.view_src || a.view_offs != b.view_offs) flags |= graph_view;
    if (a.buffer != b.buffer) flags |= graph_buffer;
    if (a.extra != b.extra) flags |= graph_extra;
    if (a.flags != b.flags) flags |= graph_flags;
    if (std::memcmp(a.name, b.name, sizeof(a.name))) flags |= graph_other;
    return flags;
}
#endif
}
void graph_diag_start() {
    const char* option = std::getenv("STRATA_DSV4_GRAPH_DIAG");
    const bool enabled = option && std::strcmp(option, "1") == 0;
    if (!enabled) return;
    std::lock_guard<std::mutex> lock(mutex);
    diagnostic = {}; diagnostic.enabled = true;
    active.store(true, std::memory_order_release);
}
void graph_diag_end() noexcept { active.store(false, std::memory_order_release); }
GraphDiagnostic graph_diag_snapshot() {
    std::lock_guard<std::mutex> lock(mutex);
    return diagnostic;
}
void graph_diag_compare(const ggml_cgraph* graph, const ggml_cuda_graph* cached) {
#ifdef USE_CUDA_GRAPH
    if (!active.load(std::memory_order_acquire)) return;
    GraphChangeRecord record;
    record.old_uid = cached->uid; record.new_uid = graph->uid;
    record.old_nodes = uint32_t(cached->node_props.size()); record.new_nodes = uint32_t(graph->n_nodes);
    const int bound = graph->n_nodes >= 0 && graph->n_nodes <= 65536 ? graph->n_nodes : 0;
    if (bound != graph->n_nodes) { record.first_node = -2; record.all_fields = graph_other; }
    for (int i = 0; i < bound; ++i) {
        const auto* node = graph->nodes[i];
        if (lease_lookup(node) && node->src[1]) record.tokens = uint32_t(ggml_nelements(node->src[1]) / 6);
        if (uint32_t(i) >= record.old_nodes) continue;
        ggml_cuda_graph::node_properties current{};
        std::memcpy(&current.node, node, sizeof(ggml_tensor));
        for (int source = 0; source < GGML_MAX_SRC; ++source) {
            if (!node->src[source]) continue;
            current.node_src_data_ptrs[source] = node->src[source]->data;
            std::memcpy(current.node_src_ne[source], node->src[source]->ne, sizeof(node->src[source]->ne));
            std::memcpy(current.node_src_nb[source], node->src[source]->nb, sizeof(node->src[source]->nb));
        }
        const auto& previous = cached->node_props[i];
        if (!std::memcmp(&previous, &current, sizeof(current))) continue;
        ++record.changed_nodes;
        uint32_t fields = changed_fields(previous, current);
        if (!fields) fields = graph_other; // Covers any struct bytes beyond the named fields.
        record.all_fields |= fields;
        if (record.first_node >= 0) continue;
        record.first_node = i; record.operation = int(node->op); record.first_fields = fields;
        std::memcpy(record.first_name, node->name, sizeof(record.first_name) - 1);
        record.tensor_bytes = uint32_t(sizeof(ggml_tensor));
        record.name_offset = uint32_t(offsetof(ggml_tensor, name));
        record.reserved_offset = uint32_t(offsetof(ggml_tensor, padding));
        const auto* old_bytes = reinterpret_cast<const uint8_t*>(&previous.node);
        const auto* new_bytes = reinterpret_cast<const uint8_t*>(&current.node);
        for (uint32_t byte = 0; byte < sizeof(ggml_tensor); ++byte) {
            if (old_bytes[byte] == new_bytes[byte]) continue;
            if (record.first_byte == UINT32_MAX) record.first_byte = byte;
            record.last_byte = byte; ++record.different_bytes;
        }
    }
    std::lock_guard<std::mutex> lock(mutex);
    if (diagnostic.count < diagnostic.records.size()) diagnostic.records[diagnostic.count++] = record;
    else ++diagnostic.dropped;
#else
    (void) graph; (void) cached;
#endif
}
}
