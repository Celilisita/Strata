#include "strata/qwen35moe/expert_storage.hpp"
#include <algorithm>
#include <stdexcept>

namespace strata::qwen35moe {
void cuda_check(cudaError_t, const char*);

uint8_t* ExpertStorage::slot(size_t index) const {
    if (index >= slots_) throw std::out_of_range("expert storage slot");
    return chunks_[index / chunk_slots].as<uint8_t>() + (index % chunk_slots) * layout_.capacity(index, stride_);
}

void ExpertStorage::resize(size_t slots, size_t stride, size_t inventory, cudaStream_t stream) {
    if (!slots || !stride || slots > inventory || (stride_ && stride_ != stride))
        throw std::invalid_argument("invalid expert storage resize");
    if (layout_.compact() && (slots < layout_.high_slots()+chunk_slots || slots%chunk_slots ||
        inventory!=layout_.inventory() || stride!=layout_.capacity(0,stride)))
        throw std::invalid_argument("compact resize must preserve high prefix and low slab");
    if (layout_.compact()) (void)layout_.bytes(slots,stride);
    if (slots == slots_) return;
    const size_t count = (slots + chunk_slots - 1) / chunk_slots;
    // Partial chunks occur only at a fixed-slot cap. Resizing a partial chunk
    // would change live addresses; automatic capacity is rounded to whole chunks.
    if (slots_ && slots_ % chunk_slots && slots > slots_)
        throw std::invalid_argument("cannot grow partial expert storage chunk");
    if (slots < slots_ && slots % chunk_slots)
        throw std::invalid_argument("cannot shrink into a live expert storage chunk");
    std::vector<DeviceBuffer> additions;
    for (size_t i = chunks_.size(); i < count; ++i)
        additions.emplace_back((std::min)(chunk_slots, slots - i * chunk_slots) * layout_.capacity(i*chunk_slots, stride));
    if (!addresses_.data()) addresses_.reset(inventory * sizeof(uint64_t));
    // Reserve before commit: DeviceBuffer moves preserve backing addresses.
    chunks_.reserve(count);
    // Strong allocation guarantee: an allocation failure leaves all old chunks.
    for (auto& buffer : additions) chunks_.push_back(std::move(buffer));
    chunks_.resize(count);
    slots_ = slots; stride_ = stride;
    std::vector<uint64_t> addresses(slots);
    for (size_t i = 0; i < slots; ++i)
        addresses[i] = reinterpret_cast<uint64_t>(slot(i));
    cuda_check(cudaMemcpyAsync(addresses_.data(), addresses.data(), slots * sizeof(uint64_t),
        cudaMemcpyHostToDevice, stream), "mirror expert storage addresses");
    cuda_check(cudaStreamSynchronize(stream), "publish expert storage addresses");
}
} // namespace strata::qwen35moe
