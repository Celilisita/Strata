#pragma once
#include "llama-graph.h"
#include <array>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>

namespace strata::deepseek_v4 {
// CPU metadata arenas only. The canonical scheduler, allocator, graph-reuse
// predicate and CUDA property checks still decide every build/bind/replay.
// Keeping one arena per validated shape prevents PP and TG overwriting each
// other's tensor metadata addresses. No additional GPU workspace or weights.
class LeaseGraphArenas {
public:
    LeaseGraphArenas() {
        const char* option = std::getenv("STRATA_DSV4_LEASE_GRAPH_ARENAS");
        enabled_ = option && std::strcmp(option, "1") == 0;
    }
    bool enabled() const noexcept { return enabled_; }
    llm_graph_result* select(uint32_t tokens, uint32_t outputs, int64_t max_nodes) {
        if (!enabled_ || tokens < 1 || tokens > graphs_.size() || outputs != 1) return nullptr;
        if (max_nodes < 1 || max_nodes > 65536)
            throw std::runtime_error("lease graph arena exceeds its bounded metadata budget");
        auto& graph = graphs_[tokens - 1];
        if (!graph) graph = std::make_unique<llm_graph_result>(max_nodes);
        if (graph->get_max_nodes() != max_nodes)
            throw std::runtime_error("lease graph arena node capacity changed");
        return graph.get();
    }
private:
    bool enabled_ = false;
    std::array<std::unique_ptr<llm_graph_result>, 8> graphs_;
};
}
