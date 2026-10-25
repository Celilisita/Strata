#pragma once
#include "strata/kernels/gdn.hpp"
namespace strata::kernels {
struct NativeGdnCoalescedResources {
    int regs=0, local=0, shared=0, blocks=0, binary=0;
    int device=-1, major=0, minor=0, sms=0;
    bool eligible=false;
};
// Current-device host queries only; no allocation, stream operation or pointer cache.
NativeGdnCoalescedResources native_gdn_coalesced_resources() noexcept;
// Exact S128/HK16/HV32 F32 state layout and native q/k scaling contract.
// Caller owns valid current-device spans and a nonnull ordered stream. False
// means unsupported shape/span/device/resources, before any kernel/state write.
// A queued-kernel error throws; callers must never replay native after that.
bool native_gdn_coalesced_step(float* state,const float* q,const float* k,const float* v,
    const float* gate,const float* beta,float* output,const GdnShapes& shape,void* stream);
}
