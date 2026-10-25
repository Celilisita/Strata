#pragma once
#include <cstdint>
namespace strata::glm5_next {
// Provider-lifetime preparation gauges plus current serial-owner active state.
struct CpuGuQ8PreparationMetrics {
    bool runtime_requested=false,runtime_active=false;
    uint64_t prepared_typed_layers=0,prepared_typed_graphs=0,prepared_stock_routed_graphs=0,prepared_routed_graphs=0;
    uint64_t routed_scheduler_cpu_buffer_bytes=0,routed_activation_capacity_bytes=0;
    double single_graph_prepare_ms=0,routed_graph_prepare_ms=0;
};
} // namespace strata::glm5_next
