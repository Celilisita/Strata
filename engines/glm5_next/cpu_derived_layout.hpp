#pragma once
#include "cpu_derived_view.h"
#include "../deepseek_v4/cpu_expert_service.hpp"
#include "ggml-backend.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace strata::glm5_next {

class CpuDerivedLayout {
public:
    struct Span { const void * data; size_t bytes; };
    struct Counts {
        std::array<uint64_t,2> single_rows{}, pair_rows{}, lookup_misses{}, fallback_rows{};
    };
    CpuDerivedLayout(const ggml_tensor * gate, const ggml_tensor * up, const std::vector<int32_t>& keys) {
        if (!strata_iq3_derived_compiled_avx2() || keys.size() != 12 ||
            !std::is_sorted(keys.begin(),keys.end()) || std::adjacent_find(keys.begin(),keys.end()) != keys.end() ||
            keys.front() < 0 || keys.back() >= 288 || gate == up)
            throw std::runtime_error("derived CPU fixture requires twelve unique original keys and AVX2");
        view_.banks[0] = gate; view_.banks[1] = up; view_.counters = counters_.data();
        size_t expert_bytes = 0;
        if (!strata_iq3_derived_payload_bytes(expert_blocks,&expert_bytes))
            throw std::runtime_error("derived CPU payload size overflow");
        bytes_ = 2*keys.size()*expert_bytes;
        original_.reserve(2*keys.size()); derived_.reserve(6);
        for (int role = 0; role < 2; ++role) {
            const auto * bank = view_.banks[role];
            if (!bank || !bank->data || !bank->buffer || bank->type != GGML_TYPE_IQ3_S ||
                bank->ne[0] != 4096 || bank->ne[1] != 2048 || bank->ne[2] != 288 || bank->ne[3] != 1 ||
                bank->nb[1] != 1760 || bank->nb[2] != 3604480 || !ggml_is_contiguous(bank) ||
                !ggml_backend_buffer_is_host(bank->buffer) ||
                ggml_backend_buffer_get_usage(bank->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS)
                throw std::runtime_error("derived CPU view is not an original contiguous IQ3_S Gate/Up bank");
            auto& owner = storage_[role];
            const size_t blocks = keys.size()*expert_blocks;
            owner.levels.resize(blocks*256); owner.d.resize(blocks); owner.scales.resize(blocks*4);
            derived_.push_back({owner.levels.data(),owner.levels.size()});
            derived_.push_back({owner.d.data(),owner.d.size()*sizeof(uint16_t)});
            derived_.push_back({owner.scales.data(),owner.scales.size()});
            for (size_t index = 0; index < keys.size(); ++index) {
                const int32_t key = keys[index];
                auto& entry = view_.experts[role][key];
                entry.original = static_cast<const char *>(bank->data) + size_t(key)*bank->nb[2];
                entry.payload = {owner.levels.data()+index*expert_blocks*256,owner.d.data()+index*expert_blocks,
                    owner.scales.data()+index*expert_blocks*4,expert_blocks};
                entry.rows = 2048; entry.row_blocks = 16; entry.role = role;
                if (!strata_iq3_derived_predecode(entry.original,expert_blocks,
                        owner.levels.data()+index*expert_blocks*256,owner.d.data()+index*expert_blocks,
                        owner.scales.data()+index*expert_blocks*4) ||
                    !strata_iq3_derived_identity(entry.original,entry.payload))
                    throw std::runtime_error("derived CPU payload differs from the original IQ3_S storage");
                original_.push_back({entry.original,bank->nb[2]});
            }
        }
    }
    CpuDerivedLayout(const CpuDerivedLayout&) = delete;
    CpuDerivedLayout& operator=(const CpuDerivedLayout&) = delete;
    const strata_cpu_derived_view * view() const noexcept { return &view_; }
    const ggml_tensor * gate() const noexcept { return view_.banks[0]; }
    const ggml_tensor * up() const noexcept { return view_.banks[1]; }
    size_t payload_bytes() const noexcept { return bytes_; }
    const std::vector<Span>& original_spans() const noexcept { return original_; }
    const std::vector<Span>& derived_spans() const noexcept { return derived_; }
    strata_iq3_derived_row row(const ggml_tensor * bank, int key, size_t index) const {
        const auto * expert = strata_cpu_derived_find(&view_,bank,key,-1);
        const auto result = strata_cpu_derived_row(expert,index);
        if (!result.blocks) throw std::runtime_error("derived CPU row does not belong to this original key");
        return result;
    }
    // Quiescent only: readers are the existing pool workers, one private lane each.
    void reset_counts() noexcept { counters_ = {}; }
    Counts counts() const noexcept {
        Counts result;
        for (const auto& worker : counters_) for (size_t role = 0; role < 2; ++role) {
            result.single_rows[role] += worker.single_rows[role];
            result.pair_rows[role] += worker.pair_rows[role];
            result.lookup_misses[role] += worker.lookup_misses[role];
            result.fallback_rows[role] += worker.fallback_rows[role];
        }
        return result;
    }
private:
    static constexpr size_t expert_blocks = 32768;
    struct Storage { std::vector<int8_t> levels; std::vector<uint16_t> d; std::vector<uint8_t> scales; };
    std::array<Storage,2> storage_;
    alignas(128) std::array<strata_cpu_derived_worker_counts,16> counters_{};
    strata_cpu_derived_view view_{};
    std::vector<Span> original_, derived_;
    size_t bytes_ = 0;
};

class CpuDerivedScope {
public:
    CpuDerivedScope(deepseek_v4::CpuExpertService& service, const CpuDerivedLayout& layout) : service_(service) {
        if (!service_.set_diagnostic_derived(&layout)) throw std::runtime_error("derived CPU pool scope could not bind");
    }
    ~CpuDerivedScope() { service_.set_diagnostic_derived(nullptr); }
    CpuDerivedScope(const CpuDerivedScope&) = delete;
    CpuDerivedScope& operator=(const CpuDerivedScope&) = delete;
private:
    deepseek_v4::CpuExpertService& service_;
};

} // namespace strata::glm5_next
