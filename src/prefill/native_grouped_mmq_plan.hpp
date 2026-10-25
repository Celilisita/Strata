#pragma once
#include "strata/prefill/native_grouped_mmq.hpp"
#include "strata/prefill/native_mmq_tile_policy.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <vector>
namespace strata::prefill::mmq {
inline int native_mmq_group_rows(int type, int backend, int entries = 0) {
    const auto mode = native_mmq_tile_mode(std::getenv("STRATA_BENCH_GROUP_ROWS"), "group rows");
    const int rows = native_mmq_group_tile_rows(mode, type, backend, entries);
    if (rows == 64) return 64;
    const char* panel = std::getenv("STRATA_BENCH_GROUP_PANEL_TOKENS");
    if (panel && std::strcmp(panel, "0") && std::strcmp(panel, "64"))
        throw std::invalid_argument("group128 requires panel tokens0 or64");
    return rows;
}
struct NativeMmqPlan {
    std::vector<NativeMmqMatrix> matrices;
    std::vector<int32_t> input_ids, output_ids;
    int max_rows = 0, active_count = 0;
    std::vector<NativeMmqTile> tiles;
};
inline NativeMmqPlan native_mmq_plan(const unsigned long long* gate, const unsigned long long* up,
    const int32_t* source_ids, int entries, int source_rows, bool active = false, int tile_rows = 64) {
    if (!gate || !source_ids || entries < 1 || entries > 65536 ||
        source_rows < 1 || source_rows > 65536 || (tile_rows != 64 && tile_rows != 128))
        throw std::invalid_argument("invalid bounded grouped MMQ plan");
    NativeMmqPlan p;
    if (active) p.tiles.reserve((entries + tile_rows - 1) / tile_rows + 256);
    std::unordered_map<uint64_t, int> directory;
    directory.reserve(256);
    std::vector<int> group_ids(entries), counts;
    for (int i = 0; i < entries; ++i) {
        if (!gate[i] || (up && !up[i]) || source_ids[i] < 0 || source_ids[i] >= source_rows)
            throw std::invalid_argument("invalid grouped MMQ address/input row");
        auto [item, inserted] = directory.emplace(gate[i], int(p.matrices.size()));
        if (inserted) {
            if (p.matrices.size() == 256) throw std::invalid_argument("grouped MMQ exceeds 256 matrices");
            p.matrices.push_back({gate[i], up ? up[i] : 0, 0, 0}); counts.push_back(0);
        } else if (p.matrices[item->second].up != (up ? up[i] : 0))
            throw std::invalid_argument("inconsistent grouped MMQ matrix identity");
        group_ids[i] = item->second; ++counts[item->second];
    }
    int begin = 0;
    for (size_t i = 0; i < counts.size(); ++i) {
        p.matrices[i].begin = begin; begin += counts[i]; p.matrices[i].end = begin;
        p.max_rows = (std::max)(p.max_rows, counts[i]);
        const int tiles = (counts[i] + tile_rows - 1) / tile_rows;
        p.active_count += tiles;
        if (active) for (int j = 0; j < tiles; ++j) p.tiles.push_back({int(i), j});
        counts[i] = p.matrices[i].begin;
    }
    p.input_ids.resize(entries); p.output_ids.resize(entries);
    for (int i = 0; i < entries; ++i) {
        const int row = counts[group_ids[i]]++;
        p.input_ids[row] = source_ids[i]; p.output_ids[row] = i;
    }
    return p;
}
} // namespace strata::prefill::mmq
