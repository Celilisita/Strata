#pragma once
#include "cpu_iq3_exact_lut_view.h"
#include "../deepseek_v4/cpu_expert_service.hpp"
#include "ggml-backend.h"
#include <array>
#include <stdexcept>

namespace strata::glm5_next {
// Diagnostic control and private worker counters only:no weight allocation.
class CpuIq3LutDiagnostic {
public:
    struct Counts {
        std::array<uint64_t,2> ny1_rows{}, ny2_pairs{}, lookup_misses{}, fallback_rows{}, chunk_calls{}, lut_blocks{};
    };
    CpuIq3LutDiagnostic(const ggml_tensor * gate, const ggml_tensor * up) {
        capabilities_ = strata_iq3_exact_lut_capabilities();
        if (capabilities_ != 7 || !strata_iq3_exact_lut_grid_identity() || gate == up)
            throw std::runtime_error("exactLUT requires actualAVX2/FMA/F16C/OS support and exactconstantmap");
        view_.banks[0] = gate; view_.banks[1] = up; view_.counters = counters_.data();
        for (const auto * bank : view_.banks) {
            if (!bank || !bank->data || !bank->buffer || bank->type != GGML_TYPE_IQ3_S ||
                bank->ne[0] != 4096 || bank->ne[1] != 2048 || bank->ne[2] != 288 || bank->ne[3] != 1 ||
                bank->nb[1] != 1760 || bank->nb[2] != 3604480 || !ggml_is_contiguous(bank) ||
                !ggml_backend_buffer_is_host(bank->buffer) ||
                ggml_backend_buffer_get_usage(bank->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS)
                throw std::runtime_error("exactLUT borrows originalcontiguous IQ3_S GU banks only");
        }
    }
    CpuIq3LutDiagnostic(const CpuIq3LutDiagnostic&) = delete;
    CpuIq3LutDiagnostic& operator=(const CpuIq3LutDiagnostic&) = delete;
    const strata_cpu_iq3_lut_view * view() const noexcept { return &view_; }
    const ggml_tensor * gate() const noexcept { return view_.banks[0]; }
    const ggml_tensor * up() const noexcept { return view_.banks[1]; }
    unsigned capabilities() const noexcept { return capabilities_; }
    // Conservative16worker kernelstack/spill allowance; no owneralloc or payload.
    size_t scratch_bytes() const noexcept { return 16*32768; }
    void reset_counts() noexcept { counters_ = {}; }
    Counts counts() const noexcept {
        Counts result;
        for (const auto& worker : counters_) for (size_t role = 0; role < 2; ++role) {
            result.ny1_rows[role] += worker.ny1_rows[role];
            result.ny2_pairs[role] += worker.ny2_pairs[role];
            result.lookup_misses[role] += worker.lookup_misses[role];
            result.fallback_rows[role] += worker.fallback_rows[role];
            result.chunk_calls[role] += worker.chunk_calls[role];
            result.lut_blocks[role] += worker.lut_blocks[role];
        }
        return result;
    }
private:
    alignas(128) std::array<strata_cpu_iq3_lut_worker_counts,16> counters_{};
    strata_cpu_iq3_lut_view view_{};
    unsigned capabilities_ = 0;
};
class CpuIq3LutScope {
public:
    CpuIq3LutScope(deepseek_v4::CpuExpertService& service, const CpuIq3LutDiagnostic& diagnostic) : service_(service) {
        if (!service_.set_diagnostic_exact_lut(&diagnostic)) throw std::runtime_error("exactLUT poolscope couldnotbind");
    }
    ~CpuIq3LutScope() { service_.set_diagnostic_exact_lut(nullptr); }
    CpuIq3LutScope(const CpuIq3LutScope&) = delete;
    CpuIq3LutScope& operator=(const CpuIq3LutScope&) = delete;
private:
    deepseek_v4::CpuExpertService& service_;
};
} // namespace strata::glm5_next
