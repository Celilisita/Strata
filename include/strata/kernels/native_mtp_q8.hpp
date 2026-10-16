#pragma once
#include <cstddef>
#include <cstdint>
namespace strata::kernels {
struct NativeMtpQ8Ref {
    unsigned long long blob;
    int32_t dst, token;
};
static_assert(sizeof(NativeMtpQ8Ref) == 16);
size_t native_mtp_q8_scratch_bytes(int hits, int n_ff);
void native_mtp_q8_refs(const int32_t* ids, int hits, int k, const uint8_t* experts,
                        size_t blob_bytes, NativeMtpQ8Ref* refs, void* stream);
void native_mtp_q8(const NativeMtpQ8Ref* refs, int hits, int n_in, int n_ff,
                    const void* x_q8_1, void* scratch, float* out, void* stream);
}
