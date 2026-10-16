#pragma once

#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::deepseek_v4 {

struct SourcePinMetrics {
    uint64_t budget_bytes = 0, registered_bytes = 0;
    uint64_t attempts = 0, failures = 0, registration_wall_ns = 0;
    uint32_t last_error = 0;
};

// Transport registrations borrow immutable model pages; they own no weights or
// expert residency. Destroy this object before freeing the model or backend.
class ExpertSourcePin {
public:
    ExpertSourcePin(ggml_backend_t backend, const std::vector<ggml_tensor*>& sources,
                    uint64_t budget_bytes);
    ~ExpertSourcePin() noexcept;
    ExpertSourcePin(const ExpertSourcePin&) = delete;
    ExpertSourcePin& operator=(const ExpertSourcePin&) = delete;

    struct Segment { std::size_t bytes; bool registered; };
    Segment segment(const void* source, std::size_t bytes) const;
    void submitted(ggml_backend_t);
    bool matches(ggml_backend_t, const std::vector<ggml_tensor*>&, uint64_t budget) const;
    SourcePinMetrics metrics() const noexcept { return metrics_; }

private:
    struct Range { uintptr_t begin, end; };
    void release() noexcept;

    ggml_backend_dev_t device_ = nullptr;
    ggml_backend_event_t event_ = nullptr;
    bool busy_ = false;
    std::vector<Range> ranges_;
    std::vector<Range> sources_;
    SourcePinMetrics metrics_{};
};

uint64_t expert_source_pin_budget();
bool expert_source_private_mapping_requested();

} // namespace strata::deepseek_v4
