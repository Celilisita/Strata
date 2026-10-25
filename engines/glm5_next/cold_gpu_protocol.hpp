#pragma once
// One phase-owned device metadata entry per layer. Host mutates only quiescent.
#include <cstddef>
#include <cstdint>
#include <type_traits>
#ifdef __CUDACC__
#define STRATA_COLD_GPU_HD __host__ __device__
#else
#define STRATA_COLD_GPU_HD
#endif
namespace strata::glm5_next {
inline constexpr int cold_gpu_pointer_word=12,cold_gpu_magic_word=14;
inline constexpr int32_t cold_gpu_op_magic=0x43475031;
inline constexpr int32_t cold_gpu_op_magic4=0x43475032;
STRATA_COLD_GPU_HD inline constexpr uint32_t cold_gpu_magic_rows(int32_t magic){
    return magic==cold_gpu_op_magic?2u:magic==cold_gpu_op_magic4?4u:0u;
}
struct alignas(64) ColdGpuLayer {
    uint64_t owner_epoch=0;
    uintptr_t alias[3][288]{};
    uintptr_t bank_base[3]{};
    uint64_t stride[3]{};
    uint32_t first_row=0,bank_rows=0;
    int32_t layer=-1;
    uint32_t reserved_rows=2; // Host immutable while any graph reader is live.
    // GPU scratch, never a delta. Cleared by each lookup, including all-hit.
    int32_t selected_keys[4]{-1,-1,-1,-1};
    uint32_t active_mask=0,original_missing_mask=0,selected_count=0;
    uint32_t topology_fallback=0; // Quiescent policy mode0/1/2/3; original padding.
    // GPU aggregate counters; read/reset only at an existing quiescent boundary.
    uint64_t lookup_calls=0,allhit_lookups=0,ineligible_lookups=0,eligible_lookups=0;
    uint64_t copy_dispatches=0,copied_bytes=0,offloaded_occurrences=0,metadata_errors=0;
    uint32_t conditional4_policy=0; // Original trailing padding: 0legacy,1conditional,2diagnostic force2.
};
static_assert(sizeof(uintptr_t)==8&&std::is_standard_layout<ColdGpuLayer>::value,"cold GPU metadata requires native64 pointers");
static_assert(sizeof(ColdGpuLayer)==7104&&alignof(ColdGpuLayer)==64,"cold GPU metadata ABI changed");
static_assert(offsetof(ColdGpuLayer,topology_fallback)==7012&&offsetof(ColdGpuLayer,lookup_calls)==7016,"cold topology moved old counters");
static_assert(offsetof(ColdGpuLayer,conditional4_policy)==7080,"cold4 policy moved old ABI");
// Actual R51 seed group0: IQ3_S/IQ3_S/IQ4_XS, layers3..10 and13..43.
// Host additionally proves exact group/type/layout before attaching this hook.
STRATA_COLD_GPU_HD inline constexpr bool cold_gpu_layer_valid(int layer){return layer>=3&&layer<44&&layer!=11&&layer!=12;}
STRATA_COLD_GPU_HD inline constexpr uint64_t cold_gpu_stride(int role){return role==2?4456448ull:3604480ull;}
STRATA_COLD_GPU_HD inline bool cold_gpu_layout_valid(const ColdGpuLayer& meta,uint64_t epoch,int layer,uint32_t expected_rows=0){
    if(!epoch||meta.owner_epoch!=epoch||meta.layer!=layer||!cold_gpu_layer_valid(layer)||
        (meta.reserved_rows!=2&&meta.reserved_rows!=4)||(expected_rows&&meta.reserved_rows!=expected_rows)||
        meta.bank_rows<meta.reserved_rows||meta.first_row!=meta.bank_rows-meta.reserved_rows||
        meta.topology_fallback>3||(meta.topology_fallback&&meta.reserved_rows!=2)||
        meta.conditional4_policy>2||(meta.conditional4_policy&&(meta.reserved_rows!=4||meta.topology_fallback)))return false;
    for(int role=0;role<3;++role){const uint64_t extent=uint64_t(meta.bank_rows)*meta.stride[role];
        if(meta.stride[role]!=cold_gpu_stride(role)||!meta.bank_base[role]||meta.bank_base[role]%16||
            meta.bank_base[role]>UINTPTR_MAX-extent)return false;}
    return true;
}
STRATA_COLD_GPU_HD inline bool cold_gpu_alias_valid(const ColdGpuLayer& meta,int key){
    if(key<1||key>286)return false;
    for(int role=0;role<3;++role)if(!meta.alias[role][key]||meta.alias[role][key]%16||
        meta.alias[role][key]>UINTPTR_MAX-meta.stride[role])return false;
    return true;
}
}
#undef STRATA_COLD_GPU_HD
