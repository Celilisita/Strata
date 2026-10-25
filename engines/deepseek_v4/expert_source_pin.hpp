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
    uint64_t alias_bytes = 0, alias_failures = 0;
    uint32_t alias_last_error = 0;
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
    // Returns only a complete span in this owner's mapped registration.
    uintptr_t borrow_device_alias(const void* source, std::size_t bytes) const;
    bool matches(ggml_backend_t, const std::vector<ggml_tensor*>&, uint64_t budget) const;
    SourcePinMetrics metrics() const noexcept { return metrics_; }

private:
    struct Range { uintptr_t begin, end, alias = 0; };
    void release() noexcept;

    ggml_backend_dev_t device_ = nullptr;
    ggml_backend_event_t event_ = nullptr;
    bool busy_ = false;
    bool aliases_enabled_ = false;
    int alias_device_ = -1;
    std::vector<Range> ranges_;
    std::vector<Range> sources_;
    SourcePinMetrics metrics_{};
};

uint64_t expert_source_pin_budget();
bool expert_source_private_mapping_requested();
bool expert_source_alias_requested();

} // namespace strata::deepseek_v4
