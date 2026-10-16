#include "expert_route_slice.hpp"

#include "ggml.h"
#include <cstring>
#include <stdexcept>

namespace strata::deepseek_v4 {

std::vector<uint32_t> expert_route_ranks(uint32_t mask, uint32_t count) {
    if (!count || count > 6 || !mask || mask >= (uint32_t(1) << count))
        throw std::runtime_error("cold expert route mask exceeds the original six router ranks");
    std::vector<uint32_t> ranks;
    for (uint32_t rank = 0; rank < count; ++rank)
        if (mask & (uint32_t(1) << rank)) ranks.push_back(rank);
    return ranks;
}

ExpertRouteSlice slice_expert_routes(const Checkpoint& source,
                                    const std::vector<uint8_t>& data,
                                    const std::vector<uint32_t>& ranks,
                                    uint32_t route_axis) {
    if (route_axis > 1 || ranks.empty() || source.dimensions[3] != 1 ||
        (route_axis == 0 && (source.type != GGML_TYPE_I32 || source.dimensions[2] != 1)) ||
        (route_axis == 1 && source.type != GGML_TYPE_F32) || data.size() != source.bytes)
        throw std::runtime_error("cold expert route slice requires compact I32 ids or F32 expert outputs");
    uint64_t block_bytes = 4;
    for (uint32_t axis = 0; axis < route_axis; ++axis)
        block_bytes *= uint64_t(source.dimensions[axis]);
    const auto original_count = uint64_t(source.dimensions[route_axis]);
    if (!original_count || block_bytes > checkpoint_tensor_cap / original_count)
        throw std::runtime_error("cold expert route slice exceeds the checkpoint bound");
    uint64_t outer_count = 1;
    for (uint32_t axis = route_axis + 1; axis < 4; ++axis)
        outer_count *= uint64_t(source.dimensions[axis]);
    if (outer_count > checkpoint_tensor_cap / (block_bytes * original_count) ||
        block_bytes * original_count * outer_count != source.bytes)
        throw std::runtime_error("cold expert route slice has invalid logical dimensions");
    for (std::size_t i = 0; i < ranks.size(); ++i)
        if (ranks[i] >= original_count || (i && ranks[i] <= ranks[i - 1]))
            throw std::runtime_error("cold expert ranks must be unique and in original router order");
    ExpertRouteSlice result{source, {}};
    result.metadata.dimensions[route_axis] = int64_t(ranks.size());
    result.metadata.bytes = block_bytes * ranks.size() * outer_count;
    result.data.resize(std::size_t(result.metadata.bytes));
    for (uint64_t outer = 0; outer < outer_count; ++outer)
        for (std::size_t rank = 0; rank < ranks.size(); ++rank)
            std::memcpy(result.data.data() + (outer * ranks.size() + rank) * block_bytes,
                        data.data() + (outer * original_count + ranks[rank]) * block_bytes,
                        std::size_t(block_bytes));
    result.metadata.source_strides[0] = 4;
    for (uint32_t axis = 1; axis < 4; ++axis)
        result.metadata.source_strides[axis] = result.metadata.source_strides[axis - 1] *
                                               uint64_t(result.metadata.dimensions[axis - 1]);
    uint64_t hash = 14695981039346656037ull;
    for (uint8_t byte : result.data) { hash ^= byte; hash *= 1099511628211ull; }
    result.metadata.checksum = hash;
    return result;
}

} // namespace strata::deepseek_v4
