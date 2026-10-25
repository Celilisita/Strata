#pragma once
#include "cpu_iq3_r4_view.h"
#include "../deepseek_v4/cpu_expert_service.hpp"
#include "ggml-backend.h"
#include <algorithm>
#include <array>
#include <stdexcept>
#include <vector>

namespace strata::glm5_next {

// Bounded benchmark-owned derived view; not another expert inventory/cache.
class CpuIq3R4Layout {
public:
    struct Span { const void * data; size_t bytes; };
    struct Counts {
        std::array<uint64_t,2> ny1_rows{}, ny2_pairs{}, lookup_misses{}, fallback_rows{};
    };
    CpuIq3R4Layout(const ggml_tensor * gate, const ggml_tensor * up, const std::vector<int32_t>& keys) {
        capabilities_ = strata_iq3_r4_capabilities();
        if ((capabilities_&7) != 7 || keys.empty() || keys.size() > 12 || gate == up ||
            !std::is_sorted(keys.begin(),keys.end()) || std::adjacent_find(keys.begin(),keys.end()) != keys.end() ||
            keys.front() < 0 || keys.back() >= 288)
            throw std::runtime_error("R4 fixture requires up to twelve unique keys and the existing VNNI256 backend");
        view_.banks[0] = gate; view_.banks[1] = up; view_.counters = counters_.data();
        original_.reserve(2*keys.size()); derived_.reserve(2);
        for (int role = 0; role < 2; ++role) {
            const auto * bank = view_.banks[role];
            if (!bank || !bank->data || !bank->buffer || bank->type != GGML_TYPE_IQ3_S ||
                bank->ne[0] != 4096 || bank->ne[1] != 2048 || bank->ne[2] != 288 || bank->ne[3] != 1 ||
                bank->nb[1] != 1760 || bank->nb[2] != expert_bytes || !ggml_is_contiguous(bank) ||
                !ggml_backend_buffer_is_host(bank->buffer) ||
                ggml_backend_buffer_get_usage(bank->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS)
                throw std::runtime_error("R4 view requires original contiguous IQ3_S Gate/Up banks");
            auto& storage = storage_[role];
            storage.resize(keys.size()*expert_bytes);
            derived_.push_back({storage.data(),storage.size()});
            bytes_ += storage.size();
            for (size_t index = 0; index < keys.size(); ++index) {
                auto& entry = view_.experts[role][keys[index]];
                entry.original = static_cast<const char *>(bank->data)+size_t(keys[index])*bank->nb[2];
                entry.packed = storage.data()+index*expert_bytes;
                entry.bytes = expert_bytes; entry.role = role;
                if (!strata_iq3_r4_pack(entry.original,2048,16,storage.data()+index*expert_bytes,expert_bytes) ||
                    !strata_iq3_r4_identity(entry.original,2048,16,entry.packed,expert_bytes))
                    throw std::runtime_error("R4 byte permutation identity failed");
                original_.push_back({entry.original,expert_bytes});
            }
        }
    }
    CpuIq3R4Layout(const CpuIq3R4Layout&) = delete;
    CpuIq3R4Layout& operator=(const CpuIq3R4Layout&) = delete;
    const strata_cpu_iq3_r4_view * view() const noexcept { return &view_; }
    const ggml_tensor * gate() const noexcept { return view_.banks[0]; }
    const ggml_tensor * up() const noexcept { return view_.banks[1]; }
    unsigned capabilities() const noexcept { return capabilities_; }
    size_t payload_bytes() const noexcept { return bytes_; }
    const std::vector<Span>& original_spans() const noexcept { return original_; }
    const std::vector<Span>& derived_spans() const noexcept { return derived_; }
    strata_iq3_r4_rows group(const ggml_tensor * bank, int key, size_t first_row) const {
        const auto * entry = strata_cpu_iq3_r4_find(&view_,bank,key,-1);
        const auto group = strata_cpu_iq3_r4_group(entry,first_row);
        if (!group.data) throw std::runtime_error("R4 group is not an original registered key/row");
        return group;
    }
    // Quiescent only; each existing pool worker writes one private lane.
    void reset_counts() noexcept { counters_ = {}; }
    Counts counts() const noexcept {
        Counts result;
        for (const auto& worker : counters_) for (size_t role = 0; role < 2; ++role) {
            result.ny1_rows[role] += worker.ny1_rows[role];
            result.ny2_pairs[role] += worker.ny2_pairs[role];
            result.lookup_misses[role] += worker.lookup_misses[role];
            result.fallback_rows[role] += worker.fallback_rows[role];
        }
        return result;
    }
private:
    static constexpr size_t expert_bytes = 3604480;
    std::array<std::vector<uint8_t>,2> storage_;
    alignas(128) std::array<strata_cpu_iq3_r4_worker_counts,16> counters_{};
    strata_cpu_iq3_r4_view view_{};
    std::vector<Span> original_, derived_;
    size_t bytes_ = 0;
    unsigned capabilities_ = 0;
};

class CpuIq3R4Scope {
public:
    CpuIq3R4Scope(deepseek_v4::CpuExpertService& service, const CpuIq3R4Layout& layout) : service_(service) {
        if (!service_.set_diagnostic_iq3_r4(&layout)) throw std::runtime_error("R4 CPU scope could not bind");
    }
    ~CpuIq3R4Scope() { service_.set_diagnostic_iq3_r4(nullptr); }
    CpuIq3R4Scope(const CpuIq3R4Scope&) = delete;
    CpuIq3R4Scope& operator=(const CpuIq3R4Scope&) = delete;
private:
    deepseek_v4::CpuExpertService& service_;
};

} // namespace strata::glm5_next
