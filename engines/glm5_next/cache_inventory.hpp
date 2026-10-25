#pragma once
#include "../deepseek_v4/cache_cost_audit.hpp"
#include "../deepseek_v4/expert_frequency.hpp"
#include "ggml-backend.h"
#include <memory>
#include <vector>

namespace strata::glm5_next {

template<class Groups>
void unify_frequency(Groups& groups, int layers, int experts) {
    if (groups.empty() || !groups.front().lru.frequency) return;
    auto frequency = std::make_shared<deepseek_v4::ExpertFrequency>(layers, experts);
    for (auto& group : groups) group.lru.frequency = frequency;
}

template<class Groups>
deepseek_v4::ExpertFrequency* shared_frequency(const Groups& groups) {
    if (groups.empty()) return nullptr;
    auto* frequency = groups.front().lru.frequency.get();
    for (const auto& group : groups)
        if (group.lru.frequency.get() != frequency) return nullptr;
    return frequency;
}

template<class Impl>
deepseek_v4::CacheCostSnapshot cache_snapshot(const Impl& source) {
    deepseek_v4::CacheCostSnapshot result;
    for (const auto& group : source.groups) {
        if(group.lru.audit) {
            const auto audit = group.lru.audit->snapshot();
            result.plans += audit.plans; result.planned_fills += audit.planned_fills;
            result.planned_evictions += audit.planned_evictions; result.repeated_fills += audit.repeated_fills;
            result.refills_within43_miss_plans += audit.refills_within43_miss_plans;
        }
        result.permanent_slots += group.lru.n_slots;
        result.transient_slots += group.native.slots;
        result.model_expert_keys += group.layers.size() * group.lru.n_expert;
        for (auto* bank : group.banks) result.weight_tensor_bytes += ggml_nbytes(bank);
        // Layer IDs retain the model's absolute numbering, including dense
        // blocks; group-local cardinality cannot bound those keys.
        std::vector<bool> seen(source.layers.size() * group.lru.n_expert, false);
        for (int slot = 0; slot < group.lru.n_slots; ++slot) {
            const int key = group.lru.key_of[slot];
            if (key < 0) continue;
            ++result.resident_keys;
            if (size_t(key) >= seen.size()) { ++result.invalid_directory_links; continue; }
            if (seen[key]) ++result.invalid_directory_links;
            seen[key] = true;
            const auto* map = group.lru.slot_map[key / group.lru.n_expert];
            if (!map || map[key % group.lru.n_expert] != slot) ++result.invalid_directory_links;
        }
    }
    for (const auto& device : source.devices) {
        if (device.buf) result.backend_alloc_bytes += ggml_backend_buffer_get_size(device.buf.get());
        for (const auto& bank : device.phase_banks)
            result.backend_alloc_bytes += ggml_backend_buffer_get_size(bank.get());
    }
    return result;
}

} // namespace strata::glm5_next
