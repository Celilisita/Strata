#pragma once

#include "checkpoint.hpp"
#include <cstdint>
#include <vector>

namespace strata::deepseek_v4 {

// A request-local view of original router ranks, never an expert inventory.
struct ExpertRouteSlice {
    Checkpoint metadata;
    std::vector<uint8_t> data;
};

std::vector<uint32_t> expert_route_ranks(uint32_t mask, uint32_t count);
ExpertRouteSlice slice_expert_routes(const Checkpoint& source,
                                    const std::vector<uint8_t>& data,
                                    const std::vector<uint32_t>& ranks,
                                    uint32_t route_axis);

} // namespace strata::deepseek_v4
