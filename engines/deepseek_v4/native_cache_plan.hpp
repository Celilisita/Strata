#pragma once
#include "expert_fill_upload.hpp"
#include <array>
#include <algorithm>
#include <stdexcept>
namespace strata::deepseek_v4 {
struct NativeCacheFill {
    size_t bytes = 0, permanent_fills = 0, transient_fills = 0, reused_transient_routes = 0;
};
// Owned by the same cache group as its banks/directory/LRU. These six entries
// identify the group's reserved transport rows, never resident directory keys.
struct NativeCacheLease {
    int slots = 0;
    uint32_t epoch = 0;
    std::array<int32_t, 6> keys{-1, -1, -1, -1, -1, -1};
    int find(int key) const {
        for (int slot = 0; slot < slots; ++slot) if (keys[slot] == key) return slot;
        return -1;
    }
    int claim(int key, const std::array<int32_t, 6>& protected_keys) {
        if (const int existing = find(key); existing >= 0) return existing;
        for (int slot = 0; slot < slots; ++slot)
            if (std::find(protected_keys.begin(), protected_keys.end(), keys[slot]) == protected_keys.end()) {
                keys[slot] = key;
                return slot;
            }
        throw std::runtime_error("native cache transport lease has no unprotected row");
    }
};
// The real cache LRU plans admission; this helper places refused GPU work in
// its own group's transport rows and uploads each selected expert only once.
template<class Group, class Layer, class Fills, class Upload>
NativeCacheFill native_cache_fill(Group& g, const Layer& layer, int il, const int32_t* ids,
    uint32_t gpu_mask, uint32_t epoch, int32_t* result, Fills& fills, Upload upload, uint64_t* planning_ns) {
    if (!ids || !result || gpu_mask > 63 || g.native.slots != 6 || !epoch || epoch <= g.native.epoch)
        throw std::runtime_error("native cache lease epoch/admission request invalid");
    NativeCacheFill cost;
    const auto* resident = g.lru.slot_map[il];
    std::array<int32_t, 6> protected_keys{};
    uint32_t permanent_mask = 0;
    for (int rank = 0; rank < 6; ++rank) {
        if (ids[rank] < 0 || ids[rank] >= g.lru.n_expert)
            throw std::runtime_error("native cache expert outside model bank");
        for (int before = 0; before < rank; ++before)
            if (ids[before] == ids[rank]) throw std::runtime_error("native cache duplicate original expert route");
        protected_keys[rank] = il * g.lru.n_expert + ids[rank];
        result[rank] = resident[ids[rank]];
        if (!(gpu_mask & (1u << rank))) {
            if (result[rank] >= 0 || g.native.find(protected_keys[rank]) >= 0)
                throw std::runtime_error("native CPU provider selected an already GPU-available expert");
            result[rank] = -1; continue;
        }
        if (result[rank] >= 0) continue;
        const int transport = g.native.find(protected_keys[rank]);
        if (transport >= 0) {
            result[rank] = g.lru.n_slots + transport;
            ++cost.reused_transient_routes;
        } else permanent_mask |= 1u << rank;
    }
    Fills bypass;
    size_t hits = 0;
    if (!timed_cache_plan([&] {
        return g.lru.plan(il, ids, 6, fills, hits, true, false, permanent_mask, true, &bypass);
    }, planning_ns)) throw std::runtime_error("native cache resident capacity invalid");
    cost.permanent_fills = fills.size();
    for (auto item : bypass) {
        const int key = il * g.lru.n_expert + item.expert;
        item.slot = g.lru.n_slots + g.native.claim(key, protected_keys);
        fills.push_back(item);
        ++cost.transient_fills;
    }
    for (int rank = 0; rank < 6; ++rank) {
        if (!(gpu_mask & (1u << rank))) continue;
        if (resident[ids[rank]] >= 0) result[rank] = resident[ids[rank]];
        else {
            const int transport = g.native.find(protected_keys[rank]);
            if (transport < 0) throw std::runtime_error("native cache lost a selected transport lease");
            result[rank] = g.lru.n_slots + transport;
        }
        if (result[rank] < 0 || result[rank] >= g.n_slots)
            throw std::runtime_error("native cache returned an invalid GPU bank row");
    }
    cost.bytes = upload_cache_fills(layer.experts, g.banks, fills, upload);
    g.native.epoch = epoch;
    if (g.lru.audit) {
        g.lru.audit->transport(cost.transient_fills, cost.reused_transient_routes);
    }
    return cost;
}
}
