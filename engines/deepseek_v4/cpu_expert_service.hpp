#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include "cpu_partial_down.hpp"

struct llama_model;
namespace strata::glm5_next { class CpuDerivedLayout; class CpuIq3R4Layout; class CpuIq3LutDiagnostic; }

namespace strata::deepseek_v4 {

// One canonical CPU compute resource, borrowing original mmap expert banks.
// Calls are serial; the hybrid provider owns its persistent submission worker.
class CpuExpertService {
public:
    CpuExpertService(const llama_model& model, int threads);
    ~CpuExpertService();
    CpuExpertService(const CpuExpertService&) = delete;
    CpuExpertService& operator=(const CpuExpertService&) = delete;
    void prepare();
    void prepare_routed_rows();
    void compute_routed_rows(int layer,const float* two_activations,const int32_t* ids,
                             const uint8_t* positions,std::size_t routes,float* output);
    // Explicit diagnostic B4 contract; separate preparation never widens B2.
    // Exact1..32 shapes on the same existing native16 pool, four input rows.
    void prepare_routed_rows_four();
    void compute_routed_rows_four(int layer,const float* four_activations,const int32_t* ids,
                                  const uint8_t* positions,std::size_t routes,float* output);
    struct RoutedRowsCapacity {
        std::size_t prepared_graphs=0, scheduler_cpu_buffer_bytes=0, activation_capacity_bytes=0;
        std::size_t gu_q8_prepared_graphs=0,gu_q8_prepared_layers=0;
    };
    // Cold setup inventory only; excludes graph/context metadata and process RSS.
    RoutedRowsCapacity routed_rows_capacity() const;
    // Cost diagnostics only: prepare each exact layer/route shape before timing.
    // One existing CPU16 pool, at most64 routes and8 supplied activation rows.
    void prepare_routed_rows_diagnostic(int layer,std::size_t routes);
    void compute_routed_rows_diagnostic(int layer,const float* activations,std::size_t positions_count,
                                        const int32_t* ids,const uint8_t* positions,std::size_t routes,float* output);
    // Default-off bounded input-preparation variant, same canonical graph/pool.
    // Prepare before timing; selection and snapshot require completed prior jobs.
    void prepare_gu_q8_reuse_diagnostic(int layer,std::size_t routes);
    // Constructor-only exact1 option; unsupported B1/B4/GU formats stay stock.
    bool gu_q8_reuse_runtime_requested() const noexcept;
    bool gu_q8_reuse_runtime_active() const noexcept;
    // Same serial-owner quiescence required; cannot enable absent capability.
    bool set_gu_q8_reuse_runtime_active(bool enabled) noexcept;
    bool set_diagnostic_gu_q8_reuse(bool enabled) noexcept;
    struct GuQ8ReuseSnapshot {
        std::size_t routes=0,input_bytes=0,packed_bytes=0;
        uint64_t worker_calls=0,quantized_blocks=0,copied_blocks=0;
        std::vector<uint8_t> packed;
    };
    GuQ8ReuseSnapshot gu_q8_reuse_snapshot(int layer,std::size_t routes) const;
    // Diagnostic-only canonical Gate/Up plus SwiGLU, on the existing CPU16 pool.
    // Prepare exact shapes first; two activation rows, 1..16 actual routes,
    // and one contiguous F32[2048] output row per supplied route occurrence.
    void prepare_gate_up_diagnostic(int layer,std::size_t routes);
    void compute_gate_up_diagnostic(int layer,const float* two_activations,const int32_t* ids,
                                    const uint8_t* positions,std::size_t routes,float* output);
    // Private prototype: preflight before partialdecision; compute only after it.
    // One checked L24 GU10/GLU -> gather6 -> canonical Down6 graph on this pool.
    void prepare_partial_down_diagnostic(int layer,std::size_t gu_routes,std::size_t down_routes);
    bool partial_down_preflight(const PartialDownJob&) const noexcept;
    PartialDownReceipt compute_partial_down_diagnostic(const PartialDownJob&);
    PartialDownCapacity partial_down_capacity() const;
    // Diagnostic fixture only; caller drains the existing pool before either transition.
    bool set_diagnostic_derived(const strata::glm5_next::CpuDerivedLayout*) noexcept;
    bool set_diagnostic_iq3_r4(const strata::glm5_next::CpuIq3R4Layout*) noexcept;
    bool set_diagnostic_exact_lut(const strata::glm5_next::CpuIq3LutDiagnostic*) noexcept;
    void compute(int layer, const float* activation, const int32_t* ids,
                 std::size_t count, float* output);
private:
    void compute_routed_rows_impl(int layer,const float* activations,std::size_t positions_count,
                                  const int32_t* ids,const uint8_t* positions,std::size_t routes,float* output);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace strata::deepseek_v4
