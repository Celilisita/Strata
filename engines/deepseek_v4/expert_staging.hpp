#pragma once

#include "ggml-backend.h"
#include "expert_source_pin.hpp"
#include "packed_expert_store.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace strata::deepseek_v4 {

struct ExpertStagingMetrics {
    uint64_t bytes = 0;
    uint64_t calls = 0;       // Actual canonical asynchronous upload calls.
    uint64_t slab_waits = 0;  // Event waits before reusing a previously submitted slab.
    // Host wall time around upload memcpy and slab reuse waits only. These
    // may overlap GPU activity; release/destructor waits are excluded.
    uint64_t host_copy_wall_ns = 0;
    uint64_t host_copy_bytes = 0;
    uint64_t slab_wait_wall_ns = 0;
    uint64_t direct_bytes = 0, direct_calls = 0;
};

// Bounded transport storage only: no expert identities, weights cache or graph.
// Calls must be serialized, and the selected backend must outlive this object.
class ExpertStaging {
public:
    static constexpr std::size_t slab_bytes = 16ull * 1024 * 1024;
    static constexpr std::size_t slab_count = 2;

    // Allocates both pinned slabs and both backend events eagerly. A pageable
    // allocation fallback or unsupported backend fails construction.
    explicit ExpertStaging(ggml_backend_t backend);
    ~ExpertStaging() noexcept;
    ExpertStaging(const ExpertStaging&) = delete;
    ExpertStaging& operator=(const ExpertStaging&) = delete;
    ExpertStaging(ExpertStaging&&) = delete;
    ExpertStaging& operator=(ExpertStaging&&) = delete;

    void upload(ggml_backend_t backend, ggml_tensor* destination,
                const void* source, std::size_t destination_offset, std::size_t bytes);

    // False means no helper was supplied. Upload failures are not converted
    // into a fallback after any bytes may have been submitted.
    static bool dispatch(ggml_backend_t backend, ggml_tensor* destination,
                         const void* source, std::size_t destination_offset,
                         std::size_t bytes, void* user_data);

    ExpertStagingMetrics metrics() const noexcept { return metrics_; }
    void reset_metrics() noexcept { metrics_ = {}; }
    ExpertSourcePin::Segment transport_segment(const void* source, size_t bytes) const {
        return source_pin_ ? source_pin_->segment(source, bytes) : ExpertSourcePin::Segment{bytes, false};
    }
    void register_sources(const std::vector<ggml_tensor*>& sources, uint64_t budget,
                          std::shared_ptr<ExpertSourcePin>* retained = nullptr);
    void pack_sources(const std::vector<ggml_tensor*>&, std::shared_ptr<PackedExpertStore>* retained = nullptr);
    PackedExpertStore* packed_store() const noexcept { return packed_.get(); }
    SourcePinMetrics source_pin_metrics() const noexcept {
        return packed_ ? packed_->metrics() : source_pin_ ? source_pin_->metrics() : SourcePinMetrics{};
    }

private:
    struct Slab {
        ggml_backend_buffer_t buffer = nullptr;
        ggml_backend_event_t event = nullptr;
        void* data = nullptr;
        bool busy = false;
    };

    void release() noexcept;

    ggml_backend_t backend_ = nullptr;
    std::array<Slab, slab_count> slabs_{};
    std::size_t next_slab_ = 0;
    ExpertStagingMetrics metrics_{};
    std::shared_ptr<ExpertSourcePin> source_pin_;
    std::shared_ptr<PackedExpertStore> packed_;
    bool pp_transport_bound_ = false;
    bool pp_packed_bound_ = false;
};

} // namespace strata::deepseek_v4
