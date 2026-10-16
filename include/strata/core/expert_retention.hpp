#pragma once
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace strata::core {
// Host policy only: directory, slots, weights and CUDA ordering stay caller-owned.
// Scores are normalized per routed batch, so a short tail does not lose priority
// solely because it has fewer tokens. Previous layers retain their last sample.
class ExpertRetention {
public:
    void enable(bool value) { enabled_ = value; }
    bool enabled() const { return enabled_; }
    void enable_planning(bool value) { planning_ = value; }
    void verify_planning(bool value) { verify_ = value; }
    struct Cost { uint64_t scanned_slots = 0, plans = 0, planned_pops = 0; };
    const Cost& cost() const { return cost_; }
    class LayerLease {
    public:
        LayerLease(ExpertRetention& policy, size_t layer, size_t layers,
                   size_t experts, const std::vector<int32_t>& ids, bool eligible = true)
            : policy_(policy), previous_(policy.active_) {
            if (!policy.enabled_ || !eligible) return;
            if (!experts || layer >= layers || ids.empty() || policy.active_ != none)
                throw std::invalid_argument("invalid expert retention layer lease");
            std::vector<uint32_t> counts(experts, 0);
            for (int id : ids) {
                if (id < 0 || size_t(id) >= experts)
                    throw std::invalid_argument("invalid retention expert ID");
                ++counts[size_t(id)];
            }
            if (policy.scores_.empty()) policy.scores_.resize(layers * experts, 0);
            if (policy.scores_.size() != layers * experts)
                throw std::invalid_argument("retention inventory changed");
            policy.experts_ = experts;
            for (size_t id = 0; id < experts; ++id)
                policy.scores_[layer * experts + id] =
                    uint64_t(counts[id]) * (uint64_t(1) << 24) / ids.size();
            policy.active_ = layer;
            policy.order_.clear(); policy.order_ready_ = false;
            activated_ = true;
        }
        ~LayerLease() {
            if (activated_) { policy_.order_.clear(); policy_.order_ready_ = false; }
            policy_.active_ = previous_;
        }
        LayerLease(const LayerLease&) = delete;
        LayerLease& operator=(const LayerLease&) = delete;
    private:
        ExpertRetention& policy_;
        size_t previous_;
        bool activated_ = false;
    };
    size_t victim(const std::vector<int32_t>& keys, const std::vector<uint64_t>& ages,
                  size_t first) const {
        if (!enabled_ || active_ == none || keys.size() != ages.size() || first >= keys.size())
            throw std::invalid_argument("inactive expert retention victim selection");
        if (planning_) {
            const size_t selected = planned_victim(keys, ages, first);
            if (verify_ && selected != scan_victim(keys, ages, first))
                throw std::runtime_error("planned eviction differs from preserved retention");
            return selected;
        }
        return scan_victim(keys, ages, first);
    }
    bool leased() const { return enabled_ && active_ != none; }
private:
    size_t scan_victim(const std::vector<int32_t>& keys, const std::vector<uint64_t>& ages,
                       size_t first) const {
        cost_.scanned_slots += keys.size() - first;
        size_t selected = none;
        for (size_t slot = first; slot < keys.size(); ++slot) {
            const int key = keys[slot];
            if (key < 0 || size_t(key) >= scores_.size())
                throw std::invalid_argument("invalid retention residency key");
            // Addresses of this layer remain live until grouped compute is
            // submitted. Later-layer replacements are ordered after that compute.
            if (size_t(key) / experts_ == active_) continue;
            if (selected == none || scores_[size_t(key)] < scores_[size_t(keys[selected])] ||
                (scores_[size_t(key)] == scores_[size_t(keys[selected])] && ages[slot] < ages[selected]))
                selected = slot;
        }
        if (selected == none) throw std::runtime_error("no safe retention victim");
        return selected;
    }
    struct Candidate { size_t slot; int32_t key; uint64_t age, score; };
    static bool later(const Candidate& a, const Candidate& b) {
        if (a.score != b.score) return a.score > b.score;
        if (a.age != b.age) return a.age > b.age;
        return a.slot > b.slot; // Preserve the scan's first-slot tie break.
    }
    size_t planned_victim(const std::vector<int32_t>& keys,
                          const std::vector<uint64_t>& ages, size_t first) const {
        // During a PP layer lease only this layer's ages change and each miss
        // replaces its victim with a protected current-layer key. Thus all
        // remaining eligible priorities stay constant. This is a disposable
        // ordering plan, never another residency directory or weight owner.
        if (!order_ready_) {
            const size_t begin = active_ * experts_, end = begin + experts_;
            order_.reserve(keys.size() - first);
            for (size_t slot = first; slot < keys.size(); ++slot) {
                const int key = keys[slot];
                if (key < 0 || size_t(key) >= scores_.size())
                    throw std::invalid_argument("invalid retention residency key");
                if (size_t(key) >= begin && size_t(key) < end) continue;
                order_.push_back({slot, key, ages[slot], scores_[size_t(key)]});
            }
            cost_.scanned_slots += keys.size() - first; ++cost_.plans;
            std::make_heap(order_.begin(), order_.end(), later);
            order_ready_ = true;
        }
        if (order_.empty()) throw std::runtime_error("no safe planned retention victim");
        const auto selected = order_.front();
        if (keys[selected.slot] != selected.key || ages[selected.slot] != selected.age)
            throw std::runtime_error("eligible residency changed during retention lease");
        std::pop_heap(order_.begin(), order_.end(), later); order_.pop_back();
        ++cost_.planned_pops;
        return selected.slot;
    }
    static constexpr size_t none = size_t(-1);
    bool enabled_ = false;
    bool planning_ = false, verify_ = false;
    mutable bool order_ready_ = false;
    mutable std::vector<Candidate> order_;
    mutable Cost cost_;
    size_t experts_ = 0, active_ = none;
    std::vector<uint64_t> scores_;
};
} // namespace strata::core
