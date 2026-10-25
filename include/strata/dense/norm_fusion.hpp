#pragma once

#include "ggml-backend.h"
#include <cstddef>
#include <cstdint>

namespace strata::dense {

// Immutable F32 multiplier for GDN's additive-epsilon L2 normalization.
// A borrowed backend must outlive this object and all request graphs using
// scale() must be destroyed before it. Construction accounts the backend's
// exact allocation, including alignment, against the supplied free budget.
class NormFusion {
public:
    NormFusion(ggml_backend_t backend, int64_t dimension, size_t available_bytes);
    ~NormFusion();
    NormFusion(const NormFusion &) = delete;
    NormFusion &operator=(const NormFusion &) = delete;
    NormFusion(NormFusion &&) = delete;
    NormFusion &operator=(NormFusion &&) = delete;

    ggml_tensor *scale() const noexcept { return scale_; }
    size_t bytes() const noexcept { return bytes_; }
    double cold_ms() const noexcept { return cold_ms_; }

private:
    ggml_backend_t backend_ = nullptr;
    ggml_context *context_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    ggml_tensor *scale_ = nullptr;
    size_t bytes_ = 0;
    double cold_ms_ = 0;
    void release() noexcept;
};

} // namespace strata::dense
