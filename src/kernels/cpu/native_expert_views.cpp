#include "strata/kernels/cpu/native_expert.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include <cmath>
#include "nvfp4_cpu.hpp"
namespace strata::kernels::cpu {
void native_gu_rows_view(const NativeFmt& f, const uint8_t* gate, const uint8_t* up,
                         const void* const* act, int nt, float* const* ff, int r0, int r1,
                         float gate_scale, float up_scale) {
    if (f.gu_type == 40) {
        nvfp4_cpu_gu(f, gate, up, act, nt, ff, r0, r1, gate_scale, up_scale);
        return;
    }
    const ggml_vec_dot_t dot = ggml_get_type_traits_cpu(static_cast<ggml_type>(f.gu_type))->vec_dot;
    const int n = (int) f.n_embd;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* gr = gate + (size_t) r * f.gu_row;
        const uint8_t* ur = up + (size_t) r * f.gu_row;
        for (int t = 0; t < nt; ++t) {
            float g = 0.f, u = 0.f;
            dot(n, &g, 0, gr, 0, act[t], 0, 1);
            dot(n, &u, 0, ur, 0, act[t], 0, 1);
            ff[t][r] = (g / (1.f + std::exp(-g))) * u;
        }
    }
}
void native_down_rows_view(const NativeFmt& f, const uint8_t* down,
                           const void* const* hq, int nt, float* const* out, int r0, int r1,
                           float down_scale) {
    if (f.d_type == 40) {
        nvfp4_cpu_down(f, down, hq, nt, out, r0, r1, down_scale);
        return;
    }
    const ggml_vec_dot_t dot = ggml_get_type_traits_cpu(static_cast<ggml_type>(f.d_type))->vec_dot;
    const int n = (int) f.n_ff;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* dr = down + (size_t) r * f.d_row;
        for (int t = 0; t < nt; ++t) {
            float s = 0.f;
            dot(n, &s, 0, dr, 0, hq[t], 0, 1);
            out[t][r] = s;
        }
    }
}
}
