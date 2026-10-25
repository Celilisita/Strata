#pragma once
#include "strata/qwen35moe/host_readonly_lease.hpp"
#include <array>
#include <cstdint>
#include <stdexcept>

namespace strata::qwen35moe {
struct MappedPpWeights { const void* gate = nullptr; const void* up = nullptr; const void* down = nullptr; };
struct MappedPpCost {
    uint64_t entries = 0, unique = 0, logical_weight_bytes = 0, expected_tile_bytes = 0;
};
// Disposable address decisions for ONE PP layer, not a residency directory.
// Registered GGUF remains owned by the runtime until its stream is drained.
class MappedPpLayer {
    struct Entry { MappedPpWeights weights; uint32_t routes = 0; uint16_t total_routes = 0; uint8_t state = 0; };
    const HostReadOnlyLease& lease_;
    const uint64_t epoch_;
    const std::array<const void*, 3> sources_;
    const size_t gu_bytes_, down_bytes_;
    const int tile_rows_;
    uint64_t entries_ = 0;
    std::array<Entry, 256> table_{};
    static const void* offset(const void* base, size_t bytes, int id) {
        const auto begin = reinterpret_cast<uintptr_t>(base), add = size_t(id) * bytes;
        if (begin + add < begin) throw std::invalid_argument("mapped PP source overflow");
        return reinterpret_cast<const void*>(begin + add);
    }
public:
    MappedPpLayer(const HostReadOnlyLease& lease, const void* gate, const void* up, const void* down,
                  size_t gu_bytes, size_t down_bytes, int tile_rows,
                  const int32_t* route_ids, size_t route_entries)
        : lease_(lease), epoch_(lease.epoch()), sources_{gate, up, down},
          gu_bytes_(gu_bytes), down_bytes_(down_bytes), tile_rows_(tile_rows) {
        if (!gate || !up || !down || !gu_bytes || !down_bytes ||
            gu_bytes > (8u << 20) || down_bytes > (8u << 20) || (tile_rows != 64 && tile_rows != 128) ||
            !route_ids || !route_entries || route_entries > 16384)
            throw std::invalid_argument("invalid bounded mapped PP layer");
        // Complete actual routes BEFORE supplying any address. A prefix count
        // cannot establish whether a later row would reread mapped weights.
        for (size_t i = 0; i < route_entries; ++i) {
            const int id = route_ids[i];
            if (id < 0 || id >= 256) throw std::invalid_argument("mapped PP route histogram ID");
            ++table_[size_t(id)].total_routes;
        }
    }
    MappedPpLayer(const MappedPpLayer&) = delete;
    MappedPpLayer& operator=(const MappedPpLayer&) = delete;
    static bool single_tile_routes(uint32_t routes, int tile_rows) {
        return (tile_rows == 64 || tile_rows == 128) && routes > 0 && routes <= uint32_t(tile_rows);
    }
    // A real resident always goes to expert_view, preserving its age/hit path.
    // Failed registration/range decisions are cached too: at most3 lookups per
    // cold unique expert. No device cache key, age, slot or epoch is written.
    bool view(int id, bool genuine_device_resident, MappedPpWeights& out) {
        out = {};
        if (id < 0 || id >= 256) throw std::invalid_argument("mapped PP expert ID");
        auto& entry = table_[size_t(id)];
        if (!entry.total_routes) throw std::invalid_argument("mapped PP ID absent from prepared routes");
        if (genuine_device_resident) return false;
        if (lease_.epoch() != epoch_) throw std::runtime_error("mapped PP GGUF lease expired");
        if (!single_tile_routes(entry.total_routes, tile_rows_)) return false;
        if (!entry.state) {
            entry.state = 2;
            MappedPpWeights alias;
            if (lease_.gpu_alias(offset(sources_[0], gu_bytes_, id), gu_bytes_, alias.gate) &&
                lease_.gpu_alias(offset(sources_[1], gu_bytes_, id), gu_bytes_, alias.up) &&
                lease_.gpu_alias(offset(sources_[2], down_bytes_, id), down_bytes_, alias.down)) {
                entry.weights = alias;
                entry.state = 1;
            }
        }
        if (entry.state != 1) return false;
        if (entries_ == 16384 || entry.routes == entry.total_routes)
            throw std::invalid_argument("mapped PP exceeds prepared route histogram");
        ++entries_; ++entry.routes;
        out = entry.weights;
        return true;
    }
    MappedPpCost cost() const {
        if (lease_.epoch() != epoch_) throw std::runtime_error("mapped PP GGUF lease expired");
        MappedPpCost result;
        result.entries = entries_;
        const uint64_t weight_bytes = 2 * uint64_t(gu_bytes_) + down_bytes_;
        for (const auto& entry : table_) if (entry.routes) {
            ++result.unique;
            result.logical_weight_bytes += weight_bytes;
            result.expected_tile_bytes += weight_bytes * ((entry.routes + tile_rows_ - 1) / tile_rows_);
        }
        return result; // Logical tile demand; neither value is measured PCIe traffic.
    }
};
} // namespace strata::qwen35moe
