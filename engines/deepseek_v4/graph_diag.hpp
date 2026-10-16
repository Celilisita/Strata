#pragma once
#include <array>
#include <cstdint>
#include <filesystem>

namespace strata::deepseek_v4 {
enum GraphChange : uint32_t {
    graph_data = 1u, graph_shape = 2u, graph_stride = 4u, graph_operation = 8u,
    graph_parameters = 16u, graph_cpu_sources = 32u, graph_device_sources = 64u,
    graph_source_shape = 128u, graph_source_stride = 256u, graph_view = 512u,
    graph_buffer = 1024u, graph_extra = 2048u, graph_flags = 4096u, graph_other = 8192u
};
struct GraphChangeRecord {
    uint64_t old_uid = 0, new_uid = 0;
    uint32_t old_nodes = 0, new_nodes = 0, changed_nodes = 0, tokens = 0;
    int32_t first_node = -1, operation = -1;
    uint32_t first_fields = 0, all_fields = 0;
    uint32_t first_byte = UINT32_MAX, last_byte = 0, different_bytes = 0;
    uint32_t tensor_bytes = 0, name_offset = 0, reserved_offset = 0;
    char first_name[64]{};
};
struct GraphDiagnostic {
    bool enabled = false;
    uint32_t count = 0, dropped = 0;
    std::array<GraphChangeRecord, 32> records{};
};
void graph_diag_start();
void graph_diag_end() noexcept;
GraphDiagnostic graph_diag_snapshot();
void write_graph_diagnostic(const std::filesystem::path&);
}
