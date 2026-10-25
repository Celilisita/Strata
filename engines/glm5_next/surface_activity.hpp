#pragma once
#include "verify_activity.hpp"
#include <array>
#include <vector>
namespace strata::glm5_next {
inline constexpr size_t surface_activity_capacity=512,surface_name_bytes=192;
struct SurfaceKernelRecord {
    VerifyKernelRecord kernel{};
    std::array<char,surface_name_bytes> name{};
    bool name_truncated=false;
};
struct SurfaceCall {
    uint64_t begin=0,end=0,kernels=0,lookups=0,copies=0,joins=0;
    uint64_t graph_nodes=0,zero_graph_nodes=0,overlap_pairs=0,overlap_ns=0;
    uint64_t graph_lookups=0,graph_copies=0,graph_joins=0;
};
struct SurfaceActivityReport {
    bool requested=false,passed=false;
    uint32_t api_version=0;
    uint64_t overflow=0,timestamp_errors=0,api_errors=0,marker_errors=0,dropped=0,buffer_errors=0;
    uint64_t missing_lookup=0,missing_join=0,nonadjacent_lookup=0,early_other_overlap=0;
    uint64_t zero_node=0,graph_mismatch=0,stream_mismatch=0;
    std::array<SurfaceCall,3> calls{};
    std::vector<SurfaceKernelRecord> records;
};
class KernelActivitySurface {
    bool active_=false;
public:
    explicit KernelActivitySurface(bool enabled=false);
    ~KernelActivitySurface();
    KernelActivitySurface(const KernelActivitySurface&)=delete;
    KernelActivitySurface& operator=(const KernelActivitySurface&)=delete;
    void begin_call(unsigned one_based);
    void end_call(unsigned one_based);
    SurfaceActivityReport finish();
};
}
