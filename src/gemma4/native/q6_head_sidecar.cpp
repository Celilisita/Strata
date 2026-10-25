#include "strata/runtime/capacity_storage.hpp"
#include "strata/gemma4/native/q6_head_sidecar.hpp"
#include <chrono>
#include <stdexcept>
#include <utility>

namespace strata::gemma4::native {
void Q6HeadSidecar::select(bool enabled, primitive::Matrix raw, cudaStream_t stream) {
    cold_pack_wall_s_ = 0;
    if (!enabled) {
        if (storage_) {
            if (stream != stream_)
                throw std::invalid_argument("Gemma aligned head release owner stream changed");
            enabled_ = false;
            primitive::check_cuda(cudaStreamSynchronize(stream_), "Gemma aligned head cold release drain");
            storage_->release(const_cast<void*>(matrix_.data));
            storage_.reset();
            matrix_ = {};
            raw_ = nullptr;
            stream_ = nullptr;
        }
        enabled_ = false;
        return;
    }
    if (!stream || !raw.data || raw.quant != primitive::Quant::Q6K ||
        raw.layout != primitive::BlockLayout::Gguf || raw.rows != 262144 || raw.columns != 5376)
        throw std::invalid_argument("Gemma aligned head requires tied raw Q6K262144x5376");
    if (storage_) {
        if (raw_ != raw.data || stream_ != stream)
            throw std::invalid_argument("Gemma aligned head immutable owner/stream changed");
        enabled_ = true;
        return;
    }
    constexpr size_t required = 1233125376;
    if (aligned_q6_bytes(raw) != required)
        throw std::invalid_argument("Gemma aligned head byte budget changed");
    const auto begin = std::chrono::steady_clock::now();
    auto owner = std::make_unique<strata::runtime::CapacityStorage>();
    void* destination = owner->allocate(required); // Existing allocator reserve remains enforced.
    const auto candidate = pack_aligned_q6(raw, destination, required, stream);
    primitive::check_cuda(cudaStreamSynchronize(stream), "Gemma aligned head cold pack completion");
    // Publish only after a complete pack. Local owner releases allocation if
    // cold setup fails; no partially initialized view enters graph capture.
    matrix_ = candidate;
    raw_ = raw.data;
    stream_ = stream;
    storage_ = std::move(owner);
    cold_pack_wall_s_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    enabled_ = true;
}
void Q6HeadSidecar::project(SingleQ8View input, float* output, cudaStream_t stream) const {
    if (!enabled_ || !storage_ || stream != stream_)
        throw std::invalid_argument("Gemma aligned head disabled/stale typed projection");
    linear_aligned_q6(matrix_, input, output, stream);
}
}
