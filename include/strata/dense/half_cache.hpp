#pragma once

#include "strata/dense/spec.hpp"
#include "ggml-backend.h"
#include <cstddef>
#include <unordered_map>

namespace strata::dense {

// Optional immutable FP16 representations of quantized block projections.
// Original GGUF blocks remain the authoritative quantized inventory for TG.
// Construction converts each selected payload once and joins its uploads.
// Selection is deterministic within the supplied budget: FFN gate/up first,
// then FFN down, then other projections, ordered by numeric layer and name.
// Zero budget/no fitting matrix yields an empty cache and quantized fallback.
// The borrowed backend must outlive this cache; destroy request graphs and
// synchronize their work before destroying the cache. A returned tensor is a
// borrowed view whose metadata and resident buffer are owned by this object.
class HalfCache {
public:
    HalfCache(const GgufFile &, const Spec &, ggml_backend_t, size_t available_bytes, bool ffn_only = false);
    ~HalfCache();
    HalfCache(const HalfCache &) = delete;
    HalfCache &operator=(const HalfCache &) = delete;
    HalfCache(HalfCache &&) = delete;
    HalfCache &operator=(HalfCache &&) = delete;

    ggml_tensor *tensor(const TensorInfo *) const noexcept;
    size_t bytes() const noexcept { return bytes_; }
    double cold_ms() const noexcept { return cold_ms_; }
    size_t tensor_count() const noexcept { return tensors_.size(); }
    size_t candidate_count() const noexcept { return candidate_count_; }
    size_t budget_bytes() const noexcept { return budget_bytes_; }
    const char *selection_status() const noexcept { return selection_status_; }

private:
    ggml_context *context_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    std::unordered_map<const TensorInfo *, ggml_tensor *> tensors_;
    size_t bytes_ = 0;
    size_t candidate_count_ = 0, budget_bytes_ = 0;
    double cold_ms_ = 0;
    const char *selection_status_ = "no_candidates";
    void release() noexcept;
};

} // namespace strata::dense
