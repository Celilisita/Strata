#pragma once
#include <stdexcept>
namespace strata::kernels {
// Host-only strict selector, shared with local CPU checks. Null/unset means off.
inline bool native_dense_q4_row2_enabled(const char* value) {
    if (!value) return false;
    if ((value[0] != '0' && value[0] != '1') || value[1] != '\0')
        throw std::invalid_argument("TG_DENSE_Q4_ROW2 requires 0 or 1");
    return value[0] == '1';
}
inline bool native_dense_q4_row2_eligible(bool decode, int type, int n_in, int n_out) noexcept {
    return decode && type == 12 && n_in == 2048 &&
        (n_out == 4096 || n_out == 8192);
}
// Original native Q4_K/Q8_1 payloads; no allocation, repack, wait, or CUDA query.
// Non-null 4-byte aligned, non-overlapping device spans and explicit stream.
// Primitive accepts only K2048, N512/4096/8192 and ncols1; Runtime owns TG selection.
void native_q4_k_dense_row2(const void* weights, const void* input_q8, float* output,
    int n_in, int n_out, int ncols, void* stream);
// Probe-only resource observation. Counter counts host submissions, including
// capture; graph replays do not increment it and it is not a token counter.
extern "C" int strata_tg_dense_q4_row2_resources(int* values);
extern "C" unsigned long long strata_tg_dense_q4_row2_submissions();
} // namespace strata::kernels
