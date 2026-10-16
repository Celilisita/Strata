#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
namespace strata::deepseek_v4 {
// Four-thread single-layer medians exist for counts 1/3/6. Counts 2/4/5
// interpolate those points; measured PCIe bandwidth seeds the DMA estimate.
// These are scheduling estimates, not end-to-end claims. Completion samples
// refine them. Admission favours hotter experts; all six routes stay protected.
struct NativeTgPolicy {
    std::array<double, 7> cpu_ns{0, 279300, 612600, 945900, 1138300, 1330700, 1523100};
    double dma_ns_per_expert = 13369344.0 / 28.6;
    uint32_t choose(const std::array<int32_t, 6>& ids, const int32_t* directory,
                    const uint32_t* scores, int layer) const {
        std::array<int, 6> missing{};
        int count = 0;
        for (int rank = 0; rank < 6; ++rank)
            if (directory[ids[rank]] < 0) missing[count++] = rank;
        if (!count) return 0;
        std::sort(missing.begin(), missing.begin() + count, [&](int a, int b) {
            const auto sa = scores ? scores[layer * 256 + ids[a]] : 0;
            const auto sb = scores ? scores[layer * 256 + ids[b]] : 0;
            return sa == sb ? a < b : sa < sb;
        });
        int best_count = 0;
        double best = count * dma_ns_per_expert;
        for (int n = 1; n <= count; ++n) {
            const double cost = std::max(cpu_ns[n], (count - n) * dma_ns_per_expert);
            if (cost < best) { best = cost; best_count = n; }
        }
        uint32_t mask = 0;
        for (int n = 0; n < best_count; ++n) mask |= 1u << missing[n];
        return mask;
    }
};
}
