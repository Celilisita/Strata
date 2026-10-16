// Canonical Q6_K trait; included in the native quant namespace.
struct Q6KTraits {
    using Block = Q6KBlock;
    static constexpr int DIV = 256, T = 32, KBY = 8, BPI = WARPS * WARP / 32;
    __device__ static int kqs(int tid) { return tid % 32; }
    struct W { int vl, vh; float d; const int8_t* scales; int bq8_offset; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
        r.bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
        const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
        const int vh_shift = 2 * ((iqs % 16) / 8);
        r.vl = load_int_b2(w->ql, iqs);
        r.vh = load_int_b2(w->qh, 8 * (iqs / 16) + iqs % 8) >> vh_shift;
        r.scales = w->scales + scale_offset;
        r.d = w->d;
        return r;
    }
    struct Input { int u[2]; float d8[2]; };
    __device__ static Input load_input(const W& r, const Q81Block* __restrict__ x, int iqs) {
        Input in;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            in.u[i] = reinterpret_cast<const int*>(x[r.bq8_offset + 2 * i].qs)[iqs % 8];
            in.d8[i] = __low2float(x[r.bq8_offset + 2 * i].ds);
        }
        return in;
    }
    __device__ static float apply_preloaded(const W& r, const Input& in) {
        return q6_q8_dot_impl(r.vl, r.vh, in.u, r.scales, r.d, in.d8);
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        return apply_preloaded(r, load_input(r, x, iqs));
    }
};
