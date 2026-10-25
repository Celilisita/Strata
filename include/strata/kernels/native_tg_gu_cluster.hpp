#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
namespace strata::kernels {
// One canonical activation row, eight original routed ranks and one shared expert.
// Active routed entries require coherent Gate/Up/down pointers from the same lease.
struct NativeTgGuCluster {
    const unsigned long long* addresses;
    const int32_t* input_rows;
    const unsigned long long* down_addresses;
    const void* input_q8;
    const void* shared_gate;
    const void* shared_up;
    float* expert_hidden;
    float* shared_hidden;
};
struct NativeTgGuClusterResources {
    int device,major,minor,sms,binary,registers,threads,max_threads,active_blocks,reason_mask;
    size_t shared,local;
};
// Cached only for one fixed device/context per host thread, as owned by Runtime.
// No GPU allocation, synchronization or per-token resource query.
extern "C" int strata_tg_gu_cluster_resources(int* values);
inline NativeTgGuClusterResources native_tg_gu_cluster_resources(){
    int v[12]{};if(strata_tg_gu_cluster_resources(v)!=0)
        throw std::runtime_error("TG GU cluster resource query failed");
    return {v[0],v[1],v[2],v[3],v[4],v[5],128,v[6],v[7],v[8],size_t(v[9]),size_t(v[10])};
}
// False before any device mutation when resources are unqualified. Other errors throw.
// Existing Q4_K/nativeQ8 math; null-down ranks remain completely unwritten.
bool native_tg_gu_cluster_launch(const unsigned long long*,const int32_t*,const unsigned long long*,
    const void*,const void*,const void*,float*,float*,void*);
inline bool native_tg_gu_cluster(const NativeTgGuCluster& a,void* stream){
    return native_tg_gu_cluster_launch(a.addresses,a.input_rows,a.down_addresses,a.input_q8,
        a.shared_gate,a.shared_up,a.expert_hidden,a.shared_hidden,stream);
}
// Host submissions include graph captures; graph replays do not increment this.
unsigned long long native_tg_gu_cluster_submissions();
}
