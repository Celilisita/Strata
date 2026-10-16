#pragma once
#include "strata/qwen35moe/device_buffer.hpp"
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
    size_t bytes() const { return slots_ * stride_; }
    size_t owned_bytes() const { return bytes() + addresses_.bytes(); }
private:
    size_t slots_ = 0, stride_ = 0;
    std::vector<DeviceBuffer> chunks_;
    DeviceBuffer addresses_;
};
} // namespace strata::qwen35moe
