#pragma once
#include <stdexcept>

namespace strata::deepseek_v4 {
struct CacheGrowthResult {
    int added_slots = 0;
    int attempts = 0;
    int rejected = 0;
};

// Resize only at the quiescent PP-to-TG transition. A rejected trial returns
// to the last safe bank size; the final bank retains all of its live keys.
template<class Resize, class IsSafe>
CacheGrowthResult converge_cache_growth(int maximum, Resize resize, IsSafe is_safe) {
    if (maximum < 0) throw std::runtime_error("negative cache growth bound");
    CacheGrowthResult result;
    int current = 0, low = 0, high = maximum;
    while (low <= high && result.attempts < 8) {
        const int trial = result.attempts == 0 ? high : low + (high - low + 1)/2;
        if (trial != current) { resize(trial); current = trial; }
        ++result.attempts;
        if (is_safe()) {
            result.added_slots = trial;
            low = trial + 1;
        } else {
            if (current != result.added_slots) { resize(result.added_slots); current = result.added_slots; }
            ++result.rejected;
            high = trial - 1;
        }
    }
    if (current != result.added_slots) resize(result.added_slots);
    return result;
}
}
