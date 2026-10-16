#pragma once
#include "mtp_sidecar.hpp"
#include "mxfp4_cpu_service.hpp"
#include "mtp_stage_metrics.hpp"
#include <memory>
#include <array>
#include <vector>
struct llama_context;class llama_moe_cache;
namespace strata::deepseek_v4 {
struct MtpHashPrefetchInputs {std::array<ggml_tensor*,3> tables{},directories{};bool retain_pending_token=true;};
// One MTP block; borrows target embedding/head and the Strata expert service.
class MtpDrafter {
public:
    MtpDrafter(MtpSidecar&,ggml_backend_t,ggml_tensor* embedding,ggml_tensor* head,Mxfp4CpuService&,int tokens=1,llama_context* target=nullptr,llama_moe_cache* unified_cache=nullptr,const MtpHashPrefetchInputs* hash_fixture=nullptr);
    ~MtpDrafter();
    std::vector<int32_t> forward(const float* hidden,const int32_t* tokens,const int32_t* positions);
    void set_async_input(bool enabled);
    MtpStageMetrics stage_metrics() const;
    MtpConfidence last_confidence() const;
    // Catch up accepted target rows without executing the MTP expert/head path.
    void catch_up(const float* hidden,const int32_t* tokens,const int32_t* positions,int count);
    void reset();
    int history_rows() const;
    std::vector<int32_t> diagnostic_routes() const;
    size_t device_scratch_bytes() const;
    int32_t diagnostic_pending_token() const;
private:
    struct Impl;std::unique_ptr<Impl> impl_;
};
}
