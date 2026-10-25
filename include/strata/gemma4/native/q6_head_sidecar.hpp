#include "strata/runtime/capacity_storage.hpp"
#pragma once
#include "strata/gemma4/native/q6_aligned.hpp"
#include <memory>

namespace strata::gemma4::native {
// Owns an optional exact-byte224-stride copy; raw tied embedding/head remains
// owned by Model. This class never selects activation precision or frees raw.
class Q6HeadSidecar {
    std::unique_ptr<strata::runtime::CapacityStorage> storage_;
    AlignedQ6View matrix_{};
    const void* raw_ = nullptr;
    cudaStream_t stream_ = nullptr;
    bool enabled_ = false;
    double cold_pack_wall_s_ = 0;
public:
    Q6HeadSidecar() = default;
    Q6HeadSidecar(const Q6HeadSidecar&) = delete;
    Q6HeadSidecar& operator=(const Q6HeadSidecar&) = delete;
    // Cold only: ON allocates+GPU-packs+joins; OFF joins then really releases
    // native bytes. Owner resets to idle and destroys all external graphs first.
    void select(bool enabled, primitive::Matrix raw, cudaStream_t);
    void project(SingleQ8View, float* output, cudaStream_t) const;
    bool enabled() const { return enabled_; }
    size_t bytes() const { return storage_ ? storage_->bytes() : 0; }
    // Actual pack wall in the last select call; zero for reuse or OFF.
    double cold_pack_wall_s() const { return cold_pack_wall_s_; }
};
}
