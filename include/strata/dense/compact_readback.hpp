#pragma once
#include "ggml-backend.h"
#include <cstdint>

namespace strata::dense {

// Declare after request/transient storage and before the first asynchronous
// input upload. Its destructor must run while every borrowed source is alive.
class BackendPending {
public:
    explicit BackendPending(ggml_backend_t backend) noexcept : backend_(backend) {}
    ~BackendPending() noexcept;
    BackendPending(const BackendPending &) = delete;
    BackendPending &operator=(const BackendPending &) = delete;
    void synchronize();
    // Only when an existing synchronous call has already drained the backend.
    void dismiss() noexcept { pending_ = false; }
private:
    ggml_backend_t backend_;
    bool pending_ = true;
};

struct CompactGreedyResult {
    int32_t token;
    float finite_sum;
};

// One model-owned 8-byte pinned destination, not per-append host allocation.
// This class performs no graph construction, token validation, or state commit.
class CompactReadback {
public:
    explicit CompactReadback(ggml_backend_t backend);
    ~CompactReadback() noexcept;
    CompactReadback(const CompactReadback &) = delete;
    CompactReadback &operator=(const CompactReadback &) = delete;
    bool available() const noexcept { return buffer_ != nullptr; }
    const char *status() const noexcept { return status_; }
    void enqueue(const ggml_tensor *greedy_id, const ggml_tensor *finite_sum);
    // Caller must first synchronize the same backend (normally via its guard).
    CompactGreedyResult read();
    void drain();
private:
    ggml_backend_t backend_;
    ggml_backend_buffer_t buffer_ = nullptr;
    unsigned char *data_ = nullptr;
    bool queued_ = false;
    const char *status_ = "uninitialized";
};

} // namespace strata::dense
