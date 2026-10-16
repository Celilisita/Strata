#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::deepseek_v4 {

bool expert_frequency_enabled();

// Replacement scores only. Canonical keys, slots, weights and ownership stay
// in the original cache. Current routes are protected until their graph reads.
class ExpertFrequency {
public:
    ExpertFrequency(int32_t layers, int32_t experts);
    void observe(int32_t layer, const int32_t* ids, std::size_t count);
    void export_state(uint32_t*, std::size_t words) const;
    void import_state(const uint32_t*, std::size_t words);
    uint32_t score(int32_t key) const;
    int32_t victim(int32_t head, const std::vector<int32_t>& next,
                   const std::vector<int32_t>& keys, int32_t layer,
                   const std::vector<uint32_t>& selected, uint32_t generation) const;
private:
    int32_t experts_;
    uint32_t samples_ = 0;
    std::vector<uint16_t> scores_;
};

} // namespace strata::deepseek_v4
