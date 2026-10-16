// Canonical integer MMA product, metadata/scale expression and fragment mapping.
// Dense accumulation retains the native MMVQ compiler's multiply-add contraction;
// existing grouped single-K accumulation keeps its explicit rounded add.
template<class F, bool Dense, class View>
__device__ __forceinline__ void native_tc_product(const typename F::Block* w,
    const Q81Block* inputs, int n_in, int blocks, int out_base, int native_tid,
    int kb, const View& view, float* values) {
    const int lane = int(threadIdx.x) % 32;
    const int part = (lane >> 3) & 1, token_base = ((lane >> 2) & 1) * 8;
    const int arow = (lane & 3) + (lane >= 16 ? 4 : 0), btoken = arow + token_base;
    const int iqs = F::kqs(native_tid % F::T);
    typename F::W a;
    const Q81Block* bx = nullptr;
    if constexpr (!Dense) {
        a = F::load(w + size_t(out_base + arow) * blocks + kb, iqs);
        bx = view.input_row(btoken) >= 0
            ? inputs + size_t(view.input_row(btoken)) * (n_in / 32) + kb * 8 : nullptr;
    }
    float dots[8] = {0,0,0,0,0,0,0,0};
    int bsum = 0;
    #pragma unroll
    for (int step = 0; step < F::K / 4; ++step) {
        uint32_t a0, a1, b0 = 0, b1 = 0;
        if constexpr (Dense) {
            const size_t group = size_t(kb) * F::T + native_tid % F::T;
            const int offset = part * F::K + step * 4;
            const auto* aw = reinterpret_cast<const uint32_t*>(view.packed.weight_codes +
                (group * view.packed.weight_rows + out_base + arow) * (2 * F::K) + offset);
            a0 = aw[0]; a1 = aw[1];
            const int token = view.input_row(btoken);
            if (token >= 0) {
                const auto* input = reinterpret_cast<const uint32_t*>(view.packed.input_codes +
                    (group * view.packed.columns + token) * (2 * F::K) + offset);
                b0 = input[0]; b1 = input[1];
            }
        } else {
            int aq[4], bq[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                aq[k] = F::weight_code(a, part, step * 4 + k);
                bq[k] = bx ? F::input_code(a, bx, iqs, part, step * 4 + k) : 0;
                bsum += bq[k];
            }
            a0 = native_half_pair(aq[0], aq[1]); a1 = native_half_pair(aq[2], aq[3]);
            b0 = native_half_pair(bq[0], bq[1]); b1 = native_half_pair(bq[2], bq[3]);
        }
        native_mma_884(a0, a1, b0, b1, dots);
    }
    unsigned ds = 0;
    const int row0 = (lane & 1) + (lane >= 16 ? 4 : 0);
    NativeTcMeta m0, m1;
    if constexpr (Dense) {
        const size_t group = size_t(kb) * F::T + native_tid % F::T;
        m0 = view.packed.weight_meta[group * view.packed.weight_rows + out_base + row0];
        m1 = view.packed.weight_meta[group * view.packed.weight_rows + out_base + row0 + 2];
    } else {
        ds = bx ? __half_as_ushort(F::input_scale(a, bx, part)) : 0;
        m0 = F::metadata(F::load(w + size_t(out_base + row0) * blocks + kb, iqs));
        m1 = F::metadata(F::load(w + size_t(out_base + row0 + 2) * blocks + kb, iqs));
    }
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int col = (j & 4) + (lane & 2) + (j & 1);
        const int source = (lane & 12) + (col & 3) + (col >= 4 ? 16 : 0);
        const float paired = __shfl_xor_sync(0xffffffff, dots[j], 8);
        int sum[2]{};
        half2 input_ds{};
        if constexpr (!Dense) {
            sum[0] = __shfl_sync(0xffffffff, bsum, source);
            sum[1] = __shfl_sync(0xffffffff, bsum, source ^ 8);
            const unsigned d0 = __shfl_sync(0xffffffff, ds, source);
            const unsigned d1 = __shfl_sync(0xffffffff, ds, source ^ 8);
            input_ds = __halves2half2(__ushort_as_half(uint16_t(d0)), __ushort_as_half(uint16_t(d1)));
        }
        if (!part) {
            const int dot[2] = {int(dots[j]), int(paired)};
            if constexpr (Dense) {
                const int token = view.input_row(col + token_base);
                NativeTcInputMeta<F::K> m{};
                if (token >= 0) m = static_cast<const NativeTcInputMeta<F::K>*>(view.packed.input_meta)[
                    (size_t(kb) * F::T + native_tid % F::T) * view.packed.columns + token];
                input_ds = *reinterpret_cast<const half2*>(&m.ds);
                if constexpr (F::K == 8) { sum[0] = m.sums[0]; sum[1] = m.sums[1]; }
            }
            const float contribution = F::finish(j & 2 ? m1 : m0, input_ds, dot, sum);
            if constexpr (Dense) values[j] += contribution;
            else values[j] = __fadd_rn(values[j], contribution);
        }
    }
}
