#pragma once
#include <cstddef>
namespace strata::kernels {
struct NativeGdnClusterResources {
    int device=-1,major=0,minor=0,sms=0,binary=0,regs=0,local=0,shared=0,blocks=0,threads=128;
    bool eligible=false;
};
// Probe-only observation. Production caches its first per-thread/device result.
NativeGdnClusterResources native_gdn_cluster_resources() noexcept;
// Fixed one-token geometry: Q16x128,K16x128,V32x128. History[c,3] oldest-first,
// conv[c,4]. All F32, caller-owned, aligned/disjoint spans and explicit stream.
// beta32 is in-place. raw8192 is retained; activated8192 contains normalized Q/K
// followed by SiLU V. Gate32=threshold20_softplus(alpha+dt)*A, no extra exp.
// Implements the existing non-warp L2 mode max(sum squares,epsilon^2).
// No allocation/wait. Malformed spans/epsilon/stream throw before any write;
// false means unsupported resources. A submitted CUDA error throws, no replay.
// Native processes retain one current device/context; cached eligibility is
// thread-local for that lifetime. DeviceReset is followed by process exit.
bool native_gdn_cluster_step(float* history,const float* input,const float* conv,
    float* raw,float* activated,float* beta,const float* alpha,const float* dt,
    const float* a,float* gate,float epsilon,void* stream);
}
