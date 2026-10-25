#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
namespace strata::glm5_next {
inline constexpr size_t verify_activity_windows=8,verify_activity_layers=39;
inline constexpr size_t verify_activity_capacity=262144;
inline constexpr uint64_t verify_activity_first=17,verify_activity_last=24;
enum class VerifyKernel : uint32_t {Other,Lookup,Copy2,Copy4,Join};
struct VerifyKernelRecord {
    uint64_t start=0,end=0,node=0;
    uint32_t device=0,context=0,stream=0,graph=0;
    VerifyKernel kind=VerifyKernel::Other;uint32_t padding=0;
};
static_assert(sizeof(VerifyKernelRecord)==48,"bounded activity record changed");
struct VerifyGpuBlock {
    int layer=-1;uint32_t copy_capacity=0,device=0,context=0,stream=0,graph=0,join_variant=0;
    uint64_t kernel_count=0,lookup_ns=0,lookup_to_copy_gap_ns=0,copy_ns=0;
    uint64_t branch_ns=0,branch_kernel_sum_ns=0,branch_kernel_union_ns=0;
    uint64_t join_ns=0,total_ns=0;
    bool passed=false;
};
struct VerifyGpuWindow {
    uint64_t ordinal=0,begin=0,end=0,kernels=0,copies=0,paired=0;
    uint64_t outside_stream_kernels=0,kernel_sum_ns=0,kernel_union_ns=0;
    uint64_t compute_stream_kernel_union_ns=0,outside_stream_kernel_union_ns=0;
    uint64_t order_errors=0,identity_errors=0;
    std::array<VerifyGpuBlock,verify_activity_layers> blocks{};
    bool passed=false;
};
struct CausalActivityReport {
    bool requested=false,passed=false,insufficient=false;
    uint32_t record_version=10,cupti_api_version=0;
    uint64_t windows_started=0,windows_completed=0,records=0,records_outside_windows=0;
    uint64_t overflow=0,timestamp_errors=0,marker_errors=0,api_errors=0,dropped=0,buffer_errors=0;
    uint64_t callback_buffers_peak=0;
    std::array<VerifyGpuWindow,verify_activity_windows> windows{};
};
// No-op unless one serial request owns the default-off causal collector.
// Begin/end enclose the existing synchronized B2 primary decode, not sampling.
void kernel_activity_verify_begin(uint64_t ordinal);
void kernel_activity_verify_end(uint64_t ordinal);
}
