#pragma once
#include "strata/kernels/cpu/native_expert.hpp"
#include "ggml.h"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/nvfp4_avx2.hpp"
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <limits>

namespace strata::kernels::cpu {
// CPU cold experts read the same type40 bytes and globals as grouped GPU PP.
// Q8_0 activations use the existing pool quantizer and scratch ownership.
inline bool nvfp4_cpu_format(int gu, int down, int64_t hidden, int64_t ff,
                              NativeFmt& f, std::string& error) {
    if (gu != 40 || down != 40 || hidden < 64 || ff < 64 || hidden % 64 || ff % 64) {
        error = "NVFP4 CPU experts require three type40 matrices and whole block64 rows";
        return false;
    }
    NativeFmt candidate;
    candidate.gu_type = candidate.d_type = 40;
    candidate.gu_act = candidate.d_act = 8; // GGML_TYPE_Q8_0
    candidate.n_embd = hidden;
    candidate.n_ff = ff;
    candidate.gu_row = size_t(hidden / 64) * 36;
    candidate.d_row = size_t(ff / 64) * 36;
    candidate.up_off = candidate.gu_row * size_t(ff);
    candidate.down_off = 2 * candidate.up_off;
    candidate.bytes = candidate.down_off + candidate.d_row * size_t(hidden);
    candidate.act_bytes = size_t(hidden / 32) * 34;
    candidate.h_bytes = size_t(ff / 32) * 34;
    if (candidate.act_bytes > kNativeActBytes || candidate.h_bytes > kNativeHBytes) {
        error = "NVFP4 CPU activation exceeds the existing pool scratch";
        return false;
    }
    f = candidate;
    return true;
}

inline float nvfp4_cpu_scale(unsigned code) {
    // 127 is NaN. Reject signed/invalid scale bytes instead of decoding as zero.
    if (code > 126) return std::numeric_limits<float>::quiet_NaN();
    if (code < 8) return float(code) * (1.f / 512.f);
    return std::ldexp(1.f + float(code & 7) * .125f, int(code >> 3) - 7);
}

inline float nvfp4_cpu_dot(const uint8_t* weight, const void* input, int columns) {
    constexpr int magnitudes[16] = {0, 1, 2, 3, 4, 6, 8, 12,
                                    0, -1, -2, -3, -4, -6, -8, -12};
    const auto* activation = static_cast<const uint8_t*>(input);
    float sum = 0.f;
    for (int block = 0; block < columns / 64; ++block) {
        const uint8_t* w = weight + size_t(block) * 36;
        for (int group = 0; group < 4; ++group) {
            const uint8_t* a = activation + size_t(2 * block + group / 2) * 34;
            ggml_fp16_t delta;
            std::memcpy(&delta, a, sizeof(delta));
            const auto* q = reinterpret_cast<const int8_t*>(a + 2) + (group % 2) * 16;
            const uint8_t* packed = w + 4 + group * 8;
            int product = 0;
            for (int j = 0; j < 8; ++j) {
                product += magnitudes[packed[j] & 15] * int(q[j]);
                product += magnitudes[packed[j] >> 4] * int(q[j + 8]);
            }
            sum += (ggml_fp16_to_fp32(delta) * nvfp4_cpu_scale(w[group]) * .5f)
                   * float(product);
        }
    }
    return sum;
}

inline float nvfp4_cpu_fast_dot(const uint8_t* weight, const void* input, int columns) {
    static const bool enabled = cpu_avx2_ok() && std::getenv("STRATA_NVFP4_CPU_SCALAR") == nullptr;
    return enabled ? nvfp4_avx2_dot(weight, input, columns) : nvfp4_cpu_dot(weight, input, columns);
}

inline void nvfp4_cpu_gu(const NativeFmt& f, const uint8_t* gate, const uint8_t* up,
                         const void* const* x, int tokens, float* const* hidden,
                         int first, int last, float gate_scale, float up_scale) {
    for (int row = first; row < last; ++row) {
        const auto* g = gate + size_t(row) * f.gu_row;
        const auto* u = up + size_t(row) * f.gu_row;
        for (int token = 0; token < tokens; ++token) {
            const float gv = nvfp4_cpu_fast_dot(g, x[token], int(f.n_embd)) * gate_scale;
            const float uv = nvfp4_cpu_fast_dot(u, x[token], int(f.n_embd)) * up_scale;
            hidden[token][row] = (gv / (1.f + std::exp(-gv))) * uv;
        }
    }
}

inline void nvfp4_cpu_down(const NativeFmt& f, const uint8_t* down,
                           const void* const* hidden, int tokens, float* const* out,
                           int first, int last, float global) {
    for (int row = first; row < last; ++row)
        for (int token = 0; token < tokens; ++token)
            out[token][row] = nvfp4_cpu_fast_dot(down + size_t(row) * f.d_row,
                                           hidden[token], int(f.n_ff)) * global;
}
}
