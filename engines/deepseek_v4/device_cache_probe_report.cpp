#include "device_cache_probe_report.hpp"
namespace strata::deepseek_v4 {
void write_cache_api_costs(std::ostream& out, const DeviceCacheDmaCosts& costs) {
    constexpr const char* labels[]{"cudaMemcpyAsync_payload", "cudaEventRecord_slab", "cudaMemcpyAsync_directory",
        "cudaEventRecord_directory", "cudaEventSynchronize_commit", "cudaEventSynchronize_slab",
        "cudaMemcpyAsync_frequency_D2H", "cudaEventRecord_frequency", "cudaEventSynchronize_frequency",
        "cudaEventRecord_enqueue_budget", "cudaEventQuery_enqueue_budget",
        "cudaEventRecord_native_payload_ready", "cudaEventSynchronize_native_payload_ready"};
    out << "{\"phase_labels\":[\"idle\",\"direct_enqueue\",\"capture\",\"instantiate\",\"update\",\"launch\",\"enqueued\",\"end_capture\",\"prepare\",\"cuMemCreate\",\"cuMemMap\",\"stream_sync\",\"event_sync\",\"tensor_set\",\"enqueue_miss_progress\"],\"api\":[";
    for (size_t i = 0; i < costs.api.size(); ++i) {
        const auto& s = costs.api[i];
        if (i) out << ',';
        out << "{\"name\":\"" << labels[i] << "\",\"max_ns\":" << s.max_ns << ",\"begin_ns\":" << s.begin_ns
            << ",\"end_ns\":" << s.end_ns << ",\"thread\":" << s.thread << ",\"copy_stream\":" << s.copy_stream
            << ",\"copy_stream_capture\":false,\"owner_phase_before\":" << s.phase_before
            << ",\"owner_phase_after\":" << s.phase_after << ",\"owner_stream_before\":" << s.owner_stream_before
            << ",\"owner_stream_after\":" << s.owner_stream_after
            << ",\"owner_op_before\":" << s.owner_op_before << ",\"owner_op_after\":" << s.owner_op_after
            << ",\"owner_node_before\":" << s.owner_node_before << ",\"owner_node_after\":" << s.owner_node_after << '}';
    }
    out << "],\"resource\":[";
    constexpr const char* resource_labels[]{"cuMemCreate", "cuMemMap", "cudaStreamSynchronize_main",
        "cudaEventSynchronize_main", "tensor_set_H2D_sync"};
    for (size_t i = 0; i < cache_resource_count; ++i) {
        if (i) out << ',';
        out << "{\"name\":\"" << resource_labels[i] << "\",\"max_ns\":" << costs.resource_max_ns[i]
            << ",\"begin_ns\":" << costs.resource_begin_ns[i] << ",\"end_ns\":" << costs.resource_end_ns[i]
            << ",\"owner_stream\":" << costs.resource_stream[i] << '}';
    }
    out << "],\"enqueue_miss_waits\":" << costs.enqueue_miss_waits
        << ",\"enqueue_miss_wait_ns\":" << costs.enqueue_miss_wait_ns
        << ",\"enqueue_miss_deadlines\":" << costs.enqueue_miss_deadlines
        << ",\"frequency_d2h_bytes\":" << costs.frequency_d2h_bytes
        << ",\"frequency_read_ns\":" << costs.frequency_read_ns
        << ",\"enqueue_budget_records\":" << costs.enqueue_budget_records
        << ",\"enqueue_budget_queries\":" << costs.enqueue_budget_queries
        << ",\"enqueue_budget_waits\":" << costs.enqueue_budget_waits
        << ",\"enqueue_budget_wait_ns\":" << costs.enqueue_budget_wait_ns
        << ",\"graph_preuploads\":" << costs.graph_preuploads
        << ",\"graph_preupload_ns\":" << costs.graph_preupload_ns
        << ",\"enqueue_budget_node_quantum\":64,\"enqueue_budget_event_capacity\":4}";
}
}
