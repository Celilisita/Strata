#include "expert_frequency.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace strata::deepseek_v4 {

bool expert_frequency_enabled() {
    const char* value = std::getenv("STRATA_DSV4_EXPERT_FREQUENCY");
    if (!value || std::strcmp(value, "0") == 0) return false;
    if (std::strcmp(value, "1") == 0) return true;
    throw std::runtime_error("expert frequency must be0 or1");
}

ExpertFrequency::ExpertFrequency(int32_t layers, int32_t experts) : experts_(experts) {
    if (layers <= 0 || layers > 256 || experts <= 0 || experts > 1024)
        throw std::runtime_error("expert frequency metadata exceeds its fixed bound");
    scores_.assign(static_cast<std::size_t>(layers) * experts, 0);
}
uint32_t ExpertFrequency::score(int32_t key) const {
    if (key < 0 || size_t(key) >= scores_.size()) throw std::runtime_error("frequency score key differs");
    return scores_[key];
}

void ExpertFrequency::observe(int32_t layer, const int32_t* ids, std::size_t count) {
    if (!ids || layer < 0 || static_cast<std::size_t>(layer) >= scores_.size() / experts_)
        throw std::runtime_error("expert frequency route layer differs");
    for (std::size_t i = 0; i < count; ++i) {
        if (ids[i] < 0 || ids[i] >= experts_)
            throw std::runtime_error("expert frequency route id differs");
        // Age at a fixed number of original route occurrences, including
        // multiplicity. A phase change can replace formerly frequent experts.
        if (++samples_ == 65536) {
            for (auto& score : scores_) score /= 2;
            samples_ = 0;
        }
        auto& score = scores_[static_cast<std::size_t>(layer) * experts_ + ids[i]];
        if (score < (std::numeric_limits<uint16_t>::max)()) ++score;
    }
}

void ExpertFrequency::export_state(uint32_t* state, std::size_t words) const {
    if (!state || words != scores_.size() + 1) throw std::runtime_error("frequency export dimensions differ");
    std::copy(scores_.begin(), scores_.end(), state);
    state[scores_.size()] = samples_;
}
void ExpertFrequency::import_state(const uint32_t* state, std::size_t words) {
    if (!state || words != scores_.size() + 1 || state[scores_.size()] >= 65536)
        throw std::runtime_error("frequency import dimensions or epoch differ");
    for (std::size_t i = 0; i < scores_.size(); ++i)
        if (state[i] > 65535) throw std::runtime_error("frequency imported score differs");
    std::copy(state, state + scores_.size(), scores_.begin());
    samples_ = state[scores_.size()];
}
int32_t ExpertFrequency::victim(int32_t head, const std::vector<int32_t>& next,
                               const std::vector<int32_t>& keys, int32_t layer,
                               const std::vector<uint32_t>& selected, uint32_t generation) const {
    if (next.size() != keys.size() || selected.size() != static_cast<std::size_t>(experts_))
        throw std::runtime_error("expert frequency canonical cache dimensions differ");
    int32_t best = -1;
    uint32_t minimum = 65536;
    std::size_t visited = 0;
    for (int32_t slot = head; slot >= 0; slot = next[slot]) {
        if (static_cast<std::size_t>(slot) >= keys.size() || ++visited > keys.size())
            throw std::runtime_error("expert frequency canonical LRU chain differs");
        const int32_t key = keys[slot];
        if (key < 0) return slot;
        if (static_cast<std::size_t>(key) >= scores_.size())
            throw std::runtime_error("expert frequency canonical cache key differs");
        if (key / experts_ == layer && selected[key % experts_] == generation) continue;
        const uint32_t score = scores_[key];
        // Equal scores keep the original least-recently-used tie order.
        if (score < minimum) { minimum = score; best = slot; }
    }
    if (best < 0) throw std::runtime_error("expert frequency found no unselected victim");
    return best;
}

} // namespace strata::deepseek_v4
