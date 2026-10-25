#pragma once
#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
namespace strata::qwen35moe {
// Physical capacity only. Residency/key/age/epoch stay in the existing cache.
class ExpertSlotLayout {
public:
    static constexpr size_t chunk = 64;
    bool compact() const { return high_slots_ != 0; }
    size_t high_slots() const { return high_slots_; }
    size_t inventory() const { return inventory_; }
    size_t low_stride() const { return low_stride_; }
    static size_t product(size_t a, size_t b) {
        if (b && a > (std::numeric_limits<size_t>::max)()/b)
            throw std::overflow_error("expert layout byte overflow");
        return a*b;
    }
    static size_t sum(size_t a,size_t b) {
        if (a>(std::numeric_limits<size_t>::max)()-b)
            throw std::overflow_error("expert layout byte sum overflow");
        return a+b;
    }
    bool select(size_t high, size_t low, size_t maximum, size_t inventory, size_t budget) {
        if (compact() || !high || high%chunk || inventory%chunk || high > inventory || inventory-high < chunk ||
            !low || low%256 || maximum%256 || maximum<=low)
            throw std::invalid_argument("invalid compact expert classes");
        (void)sum(product(high,maximum),product(inventory-high,low));
        const size_t base = product(high, maximum), minimum = product(chunk, low);
        if (budget < base || budget-base < minimum) return false;
        high_slots_=high; low_stride_=low; high_stride_=maximum; inventory_=inventory;
        return true;
    }
    size_t bytes(size_t slots, size_t legacy) const {
        if (!compact()) return product(slots, legacy);
        if (slots < high_slots_ || slots > inventory_)
            throw std::invalid_argument("compact prefix outside inventory");
        return sum(product(high_slots_,high_stride_),product(slots-high_slots_,low_stride_));
    }
    size_t capacity(size_t slot, size_t legacy) const {
        if (compact() && slot >= inventory_) throw std::out_of_range("compact slot capacity");
        return compact() ? (slot<high_slots_ ? high_stride_:low_stride_) : legacy;
    }
    size_t slots_for_bytes(size_t budget, size_t legacy, size_t inventory) const {
        if (!legacy) throw std::invalid_argument("zero legacy expert stride");
        if (!compact()) return (std::min)(budget/legacy,inventory);
        if (inventory!=inventory_ || legacy!=high_stride_)
            throw std::invalid_argument("compact plan identity changed");
        const size_t base=product(high_slots_,high_stride_);
        if (budget < base || (budget-base)/low_stride_ < chunk)
            throw std::runtime_error("compact phase cannot retain high prefix and a low slab");
        return high_slots_+(std::min)((budget-base)/low_stride_,inventory_-high_slots_)/chunk*chunk;
    }
    std::pair<size_t,size_t> range(size_t payload, size_t slots, size_t legacy) const {
        if (!compact()) {
            if (payload>legacy) throw std::invalid_argument("expert payload exceeds legacy slot");
            return {0,slots};
        }
        (void)bytes(slots,legacy);
        if (payload==high_stride_) return {0,high_slots_};
        if (payload==low_stride_) return {high_slots_,slots};
        throw std::invalid_argument("expert payload has no exact capacity class");
    }
private:
    size_t high_slots_=0,low_stride_=0,high_stride_=0,inventory_=0;
};
} // namespace strata::qwen35moe
