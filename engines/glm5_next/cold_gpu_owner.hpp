#pragma once
// Phase-owned metadata only. Weight storage and registration remain borrowed.
#include "ggml-backend.h"
#include "partial_down_protocol.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
struct llama_model;
class llama_moe_cache;
namespace strata::deepseek_v4 { class ExpertStaging; struct DeviceMissPacket; }
namespace strata::glm5_next {
class ColdGpuOwner {
public:
    ColdGpuOwner(const llama_model&,llama_moe_cache&,deepseek_v4::ExpertStaging&,
                 ggml_context*,uint64_t owner_epoch);
    ~ColdGpuOwner();
    static bool requested();
    static uint32_t requested_max_rows();
    // Read at construction and each existing quiescent refresh boundary.
    static uint32_t topology_requested();
    static bool conditional4_requested();
    bool conditional4_requested_value() const;
    bool conditional4_active() const;
    uint32_t topology_requested_value() const;
    uint32_t requested_max_rows_value() const;
    bool requested_value() const;
    void initialize();
    // Caller owns GPU/CPU/background quiescence, including mapped readers.
    void release_rows();
    void refresh(bool requested,bool tg);
    uintptr_t device_layer(int layer) const;
    void prepare_partial_down_diagnostic(ggml_backend_t,PartialDownPhase*,uintptr_t,uint64_t generation);
    PartialDownGraphBinding partial_down_binding(int layer) const;
    PartialDownPhase* partial_down_phase(int layer) const;
    uint64_t partial_down_generation() const;
    uint32_t validate_ticket(int layer,int routes,uint32_t missing,const int32_t* directory,
                             const deepseek_v4::DeviceMissPacket&);
    std::array<uint64_t,8> metrics() const;
    std::array<uint64_t,2> selected_job_counts() const;
    std::array<uint64_t,4> topology_metrics() const;
    std::array<uint64_t,3> repeat_priority_metrics() const;
    std::array<uint64_t,5> conditional4_metrics() const;
    uint64_t reserved_rows() const;
    size_t metadata_bytes() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
