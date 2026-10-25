#pragma once
#include "strata/qwen35moe/device_buffer.hpp"
#include "strata/qwen35moe/expert_slot_layout.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

namespace strata::qwen35moe {
// Physical backing for the single expert directory. Fixed-size allocations
// allow a cold tail to be released without recreating surviving GPU weights.
class ExpertStorage {
public:
    static constexpr size_t chunk_slots = 64;
    void resize(size_t slots, size_t stride, size_t inventory, cudaStream_t stream);
    uint8_t* slot(size_t index) const;
    const uint64_t* device_addresses() const { return addresses_.as<uint64_t>(); }
    size_t bytes() const { return slots_ ? layout_.bytes(slots_, stride_) : 0; }
    size_t capacity(size_t slot) const {
        if (slot >= slots_) throw std::out_of_range("expert slot capacity");
        return layout_.capacity(slot, stride_);
    }
    bool compact() const { return layout_.compact(); }
    size_t high_slots() const { return layout_.high_slots(); }
    const ExpertSlotLayout& layout() const { return layout_; }
    bool select_compact(size_t high, size_t low, size_t maximum, size_t inventory, size_t budget) {
        if (slots_) throw std::logic_error("cannot change live expert layout");
        return layout_.select(high, low, maximum, inventory, budget);
    }
    size_t owned_bytes() const { return bytes() + addresses_.bytes(); }
private:
    size_t slots_ = 0, stride_ = 0;
    ExpertSlotLayout layout_;
    std::vector<DeviceBuffer> chunks_;
    DeviceBuffer addresses_;
};
} // namespace strata::qwen35moe
