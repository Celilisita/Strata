#include "strata/dense/norm_fusion.hpp"
#include "ggml-alloc.h"
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace strata::dense {

NormFusion::NormFusion(ggml_backend_t backend, int64_t dimension,
                       size_t available_bytes) : backend_(backend) {
    const auto begin = std::chrono::steady_clock::now();
    try {
        if (!backend || dimension <= 0 || dimension > 512)
            throw std::invalid_argument("norm fusion requires a backend and dimension in [1,512]");
        const auto buft = ggml_backend_get_default_buffer_type(backend);
        if (!buft || ggml_backend_buft_is_host(buft))
            throw std::invalid_argument("norm fusion requires a resident device buffer");
        context_ = ggml_init({ggml_tensor_overhead() * 4, nullptr, true});
        if (!context_) throw std::runtime_error("norm fusion metadata allocation failed");
        scale_ = ggml_new_tensor_1d(context_, GGML_TYPE_F32, dimension);
        ggml_set_name(scale_, "dense.gdn_l2_scale");
        // This metadata has no allocated data or views, so the dry size is
        // the exact allocation requested below, including backend alignment.
        const size_t required = ggml_backend_alloc_ctx_tensors_from_buft_size(context_, buft);
        if (!required || required > available_bytes)
            throw std::runtime_error("norm fusion persistent constant exceeds available budget");
        buffer_ = ggml_backend_alloc_ctx_tensors_from_buft(context_, buft);
        if (!buffer_) throw std::runtime_error("norm fusion device allocation failed");
        bytes_ = ggml_backend_buffer_get_size(buffer_);
        if (bytes_ != required || bytes_ > available_bytes)
            throw std::runtime_error("norm fusion allocation differs from budgeted dry size");
        ggml_backend_buffer_set_usage(buffer_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        const std::vector<float> values(size_t(dimension), 1.0f / std::sqrt(float(dimension)));
        // Blocking upload preserves values' lifetime and makes the constant
        // ready before a request can capture a graph referencing it.
        ggml_backend_tensor_set(scale_, values.data(), 0, values.size() * sizeof(float));
        cold_ms_ = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin).count();
    } catch (...) {
        release();
        throw;
    }
}

NormFusion::~NormFusion() { release(); }

void NormFusion::release() noexcept {
    if (backend_ && buffer_) ggml_backend_synchronize(backend_);
    if (buffer_) ggml_backend_buffer_free(buffer_);
    if (context_) ggml_free(context_);
    buffer_ = nullptr; context_ = nullptr; scale_ = nullptr; bytes_ = 0;
}

} // namespace strata::dense
