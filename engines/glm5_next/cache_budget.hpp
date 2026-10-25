#pragma once
#include "mtp_options.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strata::glm5_next {

// Preserve one inventory and cache authority across PP/TG. A heterogeneous
// dynamic quant may have a small, expensive group that proportional allocation
// starves. Reserve a whole layer's unique experts per actual layout group so
// larger PP batches can execute on GPU without changing their model math.
template<class Groups, class Group, class AllocSize>
size_t cache_budget(const Groups& groups, const Group& selected,
                    size_t device_budget, size_t device_host_bytes,
                    size_t original_budget, AllocSize allocation_size, int minimum_slots = 8) {
    const char* option = std::getenv("STRATA_GLM53_PP_BANK_RESERVE");
    const bool full_layer = option && !std::strcmp(option,"1");
    if(option && std::strcmp(option,"0") && std::strcmp(option,"1"))
        throw std::runtime_error("GLM PP bank reserve must be0 or1");
    const char* native = std::getenv("STRATA_GLM53_STREAM_PP");
    if(!full_layer && !(native && !std::strcmp(native,"1"))) return original_budget;
    minimum_slots=minimum_cache_routes(minimum_slots);
    if(minimum_slots<1)throw std::runtime_error("GLM cache route minimum is invalid");
    const auto slots=[&](const auto& group){return full_layer?int(group.ref.at(0)->ne[2]):minimum_slots;};
    size_t required = 0;
    for (const auto& group : groups) {
        if (group.id != selected.id) continue;
        const auto bytes = allocation_size(group, slots(group));
        if (bytes > device_budget - std::min(required, device_budget))
            throw std::runtime_error("GLM PP bank reserve exceeds the explicit expert cache budget");
        required += bytes;
    }
    if (!device_host_bytes || required > device_budget)
        throw std::runtime_error("invalid GLM expert cache group budget");
    const size_t minimum = allocation_size(selected, slots(selected));
    const size_t remainder = size_t(double(device_budget - required) * selected.host_bytes / device_host_bytes);
    return minimum + remainder;
}

} // namespace strata::glm5_next
