#include "strata/dense/compact_readback.hpp"
#include <cstring>
#include <stdexcept>

namespace strata::dense {
namespace {
constexpr size_t packet_bytes = sizeof(int32_t) + sizeof(float);
static_assert(packet_bytes == 8, "compact readback packet must contain two 32-bit scalars");
}

BackendPending::~BackendPending() noexcept {
    if (pending_ && backend_) ggml_backend_synchronize(backend_);
}

void BackendPending::synchronize() {
    if (pending_ && backend_) ggml_backend_synchronize(backend_);
    pending_ = false;
}

CompactReadback::CompactReadback(ggml_backend_t backend) : backend_(backend) {
    if (!backend_) { status_ = "no_backend"; return; }
    auto *device = ggml_backend_get_device(backend_);
    ggml_backend_dev_props properties{};
    ggml_backend_dev_get_props(device, &properties);
    if (!properties.caps.async) { status_ = "backend_not_async"; return; }
    if (!properties.caps.host_buffer) { status_ = "pinned_host_unavailable"; return; }
    auto *type = ggml_backend_dev_host_buffer_type(device);
    if (!type) { status_ = "no_device_host_buffer_type"; return; }
    auto *candidate = ggml_backend_buft_alloc_buffer(type, packet_bytes);
    if (!candidate) { status_ = "host_buffer_allocation_failed"; return; }
    // The CUDA allocator can silently return a plain CPU allocation. Generic
    // is_host() accepts both and therefore does not establish pinned storage.
    if (ggml_backend_buffer_get_type(candidate) != type) {
        ggml_backend_buffer_free(candidate);
        status_ = "host_buffer_fell_back_to_unpinned";
        return;
    }
    data_ = static_cast<unsigned char *>(ggml_backend_buffer_get_base(candidate));
    if (!data_ || ggml_backend_buffer_get_size(candidate) < packet_bytes) {
        ggml_backend_buffer_free(candidate);
        data_ = nullptr;
        status_ = "invalid_host_packet";
        return;
    }
    buffer_ = candidate;
    status_ = "verified_device_pinned_host_buffer";
}

CompactReadback::~CompactReadback() noexcept {
    // A caller exception after enqueue must never free a live DMA destination.
    if (queued_) ggml_backend_synchronize(backend_);
    if (buffer_) ggml_backend_buffer_free(buffer_);
}

void CompactReadback::enqueue(const ggml_tensor *greedy_id, const ggml_tensor *finite_sum) {
    if (!available() || queued_)
        throw std::logic_error("compact pinned readback unavailable or already pending");
    if (!greedy_id || !finite_sum || greedy_id->type != GGML_TYPE_I32 ||
        finite_sum->type != GGML_TYPE_F32 || ggml_nbytes(greedy_id) != sizeof(int32_t) ||
        ggml_nbytes(finite_sum) != sizeof(float))
        throw std::invalid_argument("compact readback requires I32/F32 scalar tensors");
    queued_ = true;
    ggml_backend_tensor_get_async(backend_, greedy_id, data_, 0, sizeof(int32_t));
    ggml_backend_tensor_get_async(backend_, finite_sum, data_ + sizeof(int32_t), 0, sizeof(float));
}

CompactGreedyResult CompactReadback::read() {
    if (!queued_) throw std::logic_error("compact readback has no queued packet");
    CompactGreedyResult result;
    std::memcpy(&result.token, data_, sizeof(result.token));
    std::memcpy(&result.finite_sum, data_ + sizeof(result.token), sizeof(result.finite_sum));
    queued_ = false;
    return result;
}

void CompactReadback::drain() {
    if (queued_) ggml_backend_synchronize(backend_);
    queued_ = false;
}

} // namespace strata::dense
