#pragma once
#include "strata/qwen35moe/metrics.hpp"
#include "strata/prefill/native_grouped_mmq.hpp"
#include <algorithm>
namespace strata::qwen35moe {
// Pure host bookkeeping of the successfully returned plan. Invalid diagnostics never alter execution.
inline void record_gu_plan_cost(Metrics& m, const prefill::mmq::NativeGroupedMmqWork& w, bool active) noexcept {
    if (w.entries < 1 || w.entries > 65536 || w.groups < 1 || w.groups > 256 ||
        w.max_rows < 1 || w.max_rows > w.entries || w.active_tiles < w.groups ||
        w.rectangular_tiles != ((w.max_rows + 63) / 64) * w.groups ||
        w.active_tiles > w.rectangular_tiles ||
        w.scheduled_tiles != (active ? w.active_tiles : w.rectangular_tiles)) {
        ++m.pp_gu_plan_invalid;
        return;
    }
    ++m.pp_gu_plan_calls;
    m.pp_gu_plan_active_calls += uint64_t(active);
    m.pp_gu_plan_groups += uint64_t(w.groups);
    m.pp_gu_plan_max_rows = (std::max)(m.pp_gu_plan_max_rows, uint64_t(w.max_rows));
    m.pp_gu_plan_entries += uint64_t(w.entries);
    m.pp_gu_plan_active_tiles += uint64_t(w.active_tiles);
    m.pp_gu_plan_rectangular_tiles += uint64_t(w.rectangular_tiles);
    m.pp_gu_plan_scheduled_tiles += uint64_t(w.scheduled_tiles);
    m.pp_gu_plan_tile_control_bytes += active ? size_t(w.active_tiles) * sizeof(prefill::mmq::NativeMmqTile) : 0;
}
} // namespace strata::qwen35moe
