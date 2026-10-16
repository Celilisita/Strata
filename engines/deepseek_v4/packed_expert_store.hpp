#pragma once
#include "expert_source_pin.hpp"
#include "mxfp4_cpu_service.hpp"
#include <array>
#include <vector>

namespace strata::deepseek_v4 {
struct PackedExpertSpan {
    const void* data = nullptr;
    size_t bytes = 0, expert_stride = 0, gate_bytes = 0, up_bytes = 0;
};
using PackedExpertQuery = PackedExpertSpan(*)(const ggml_tensor*, int, int, void*);
// Immutable source inventory shared by PP, CPU experts and cache admission.
// Residency and leases continue to belong to the existing cache authority.
class PackedExpertStore {
public:
    PackedExpertStore(ggml_backend_t, const std::vector<ggml_tensor*>&);
    ~PackedExpertStore();
    PackedExpertStore(const PackedExpertStore&) = delete;
    PackedExpertStore& operator=(const PackedExpertStore&) = delete;
    bool matches(ggml_backend_t, const std::vector<ggml_tensor*>&) const;
    PackedExpertSpan span(const ggml_tensor* gate, int first, int count) const;
    struct Projection { const void* data; size_t bytes; };
    Projection project(const void* original, size_t bytes) const;
    Mxfp4CpuBank cpu_bank(const ggml_tensor* gate, float clamp) const;
    SourcePinMetrics metrics() const noexcept { return metrics_; }
private:
    struct Layer {
        std::array<ggml_tensor*,3> original{};
        std::array<ggml_tensor,3> view{};
        ggml_backend_buffer_t buffer = nullptr;
        char* data = nullptr;
        size_t stride = 0;
    };
    struct Range { uintptr_t begin, end; size_t layer; int role; };
    const Layer& find(const ggml_tensor*) const;
    void release() noexcept;
    ggml_backend_dev_t device_ = nullptr;
    std::vector<Layer> layers_;
    std::vector<Range> ranges_;
    SourcePinMetrics metrics_{};
};
bool packed_experts_requested();
}
