// Included after canonical native quant traits. Backend data-layout adaptation
// only: model GU/Down, activation, route weighting and merge stay with callers.
struct TcQ4 : Q4KTraits {
    __device__ static int code_index(int iqs, int part, int k) {
        return (2 * ((iqs / 2) / 4) + part) * 32 +
            4 * ((iqs / 2) % 4) + k % 4 + 16 * (k / 4);
    }
    __device__ static int weight_code(const W& w, int part, int k) {
        return (unsigned(w.v[k / 4]) >> (4 * part + 8 * (k % 4))) & 15;
    }
    __device__ static int input_code(const W& w, const Q81Block* x,
                                     int iqs, int part, int k) {
        return x[w.bq8_offset + part].qs[4 * ((iqs / 2) % 4) + k % 4 + 16 * (k / 4)];
    }
    __device__ static NativeTcMeta metadata(const W& w) {
        return {*reinterpret_cast<const uint32_t*>(w.aux), *reinterpret_cast<const uint32_t*>(&w.dm)};
    }
    __device__ static half input_scale(const W& w, const Q81Block* x, int part) {
        return __low2half(x[w.bq8_offset + part].ds);
    }
    __device__ static float finish(const NativeTcMeta& m, const half2& ds,
                                   const int* dots, const int* sums) {
        const float2 d = __half22float2(ds); const float d8[2] = {d.x, d.y};
        const auto* sc = reinterpret_cast<const uint8_t*>(&m.aux);
        const auto& dm = *reinterpret_cast<const half2*>(&m.dm);
        return q4_q8_finalize(dots, sums, sc, sc + 2, dm, d8);
    }
    static constexpr int K = 8;
};
struct TcQ6 : Q6KTraits {
    __device__ static int code_index(int iqs, int part, int k) {
        return (4 * (iqs / 16) + (iqs % 16) / 8 + 2 * part) * 32 +
            4 * (iqs % 8) + k;
    }
    __device__ static int weight_code(const W& w, int part, int k) {
        const unsigned lo = (unsigned(w.vl) >> (4 * part + 8 * k)) & 15;
        const unsigned hi = ((unsigned(w.vh) >> (4 * part + 8 * k)) & 3) << 4;
        return int(lo | hi) - 32;
    }
    __device__ static int input_code(const W& w, const Q81Block* x,
                                     int iqs, int part, int k) {
        return x[w.bq8_offset + 2 * part].qs[4 * (iqs % 8) + k];
    }
    __device__ static NativeTcMeta metadata(const W& w) {
        return {uint32_t(uint8_t(w.scales[0])) | (uint32_t(uint8_t(w.scales[4])) << 8),
                uint32_t(__half_as_ushort(__float2half_rn(w.d)))};
    }
    __device__ static half input_scale(const W& w, const Q81Block* x, int part) {
        return __low2half(x[w.bq8_offset + 2 * part].ds);
    }
    __device__ static float finish(const NativeTcMeta& m, const half2& ds,
                                   const int* dots, const int*) {
        const float2 d = __half22float2(ds); const float d8[2] = {d.x, d.y};
        const int8_t sc[8] = {int8_t(m.aux), 0, 0, 0, int8_t(m.aux >> 8), 0, 0, 0};
        const half dh = __ushort_as_half(uint16_t(m.dm));
        return q6_q8_finalize(dots, sc, __half2float(dh), d8);
    }
    static constexpr int K = 4;
};

template<class F>
__launch_bounds__(256, 3)
__global__ void native_tc_grouped_kernel(const NativeTcTile* tiles,
    const Q81Block* inputs, float* output, int n_in, int n_out) {
#if __CUDA_ARCH__ >= 700
    const auto& tile = tiles[blockIdx.y];
    const auto* w = reinterpret_cast<const typename F::Block*>(uintptr_t(tile.address));
    const int out_base = int(blockIdx.x) * 8, blocks = n_in / 256;
    const int warp = int(threadIdx.x) / 32, lane = int(threadIdx.x) % 32;
    const int part = (lane >> 3) & 1, token_base = ((lane >> 2) & 1) * 8;
    const int arow = (lane & 3) + (lane >= 16 ? 4 : 0);
    const int btoken = arow + token_base;
    __shared__ float lanes[32][129];
    for (int slice = 0; slice < 4; ++slice) {
        const int canonical_lane = warp * 4 + slice;
        float values[8] = {0,0,0,0,0,0,0,0};
        for (int native_warp = 0; native_warp < 4; ++native_warp) {
            const int native_tid = native_warp * 32 + canonical_lane;
            const int kb = native_tid / F::T, chunk = native_tid % F::T;
            if (kb >= blocks) continue;
            const int iqs = F::kqs(chunk);
            const auto a = F::load(w + size_t(out_base + arow) * blocks + kb, iqs);
            const auto* bx = tile.input_rows[btoken] >= 0
                ? inputs + size_t(tile.input_rows[btoken]) * (n_in / 32) + kb * 8 : nullptr;
            float dots[8] = {0,0,0,0,0,0,0,0};
            int bsum = 0;
#pragma unroll
            for (int step = 0; step < F::K / 4; ++step) {
                int aq[4], bq[4];
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    aq[k] = F::weight_code(a, part, step * 4 + k);
                    bq[k] = bx ? F::input_code(a, bx, iqs, part, step * 4 + k) : 0;
                    bsum += bq[k];
                }
                native_mma_884(native_half_pair(aq[0], aq[1]), native_half_pair(aq[2], aq[3]),
                    native_half_pair(bq[0], bq[1]), native_half_pair(bq[2], bq[3]), dots);
            }
            const unsigned ds = bx ? __half_as_ushort(F::input_scale(a, bx, part)) : 0;
            const int row0 = (lane & 1) + (lane >= 16 ? 4 : 0);
            const auto m0 = F::metadata(F::load(w + size_t(out_base + row0) * blocks + kb, iqs));
            const auto m1 = F::metadata(F::load(w + size_t(out_base + row0 + 2) * blocks + kb, iqs));
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int col = (j & 4) + (lane & 2) + (j & 1);
                const int source = (lane & 12) + (col & 3) + (col >= 4 ? 16 : 0);
                const float paired = __shfl_xor_sync(0xffffffff, dots[j], 8);
                const int sum[2] = {__shfl_sync(0xffffffff, bsum, source),
                                    __shfl_sync(0xffffffff, bsum, source ^ 8)};
                const unsigned d0 = __shfl_sync(0xffffffff, ds, source);
                const unsigned d1 = __shfl_sync(0xffffffff, ds, source ^ 8);
                if (!part) {
                    const int dot[2] = {int(dots[j]), int(paired)};
                    const half2 input_ds = __halves2half2(__ushort_as_half(uint16_t(d0)), __ushort_as_half(uint16_t(d1)));
                    const float contribution = F::finish(j & 2 ? m1 : m0, input_ds, dot, sum);
                    values[j] = __fadd_rn(values[j], contribution);
                }
            }
        }
        if (!part) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int row = (lane & 1) + (j & 2) + (lane >= 16 ? 4 : 0);
                const int token = (j & 4) + (lane & 2) + (j & 1) + token_base;
                lanes[canonical_lane][row * 16 + token] = values[j];
            }
        }
    }
    __syncthreads();
    for (int cell = warp; cell < 128; cell += 8) {
        const int token = cell % 16, row = cell / 16;
        const float value = warp_sum(lanes[lane][cell]);
        if (lane == 0 && tile.output_rows[token] >= 0)
            output[size_t(tile.output_rows[token]) * n_out + out_base + row] = value;
    }
#endif
}

#include "native_tc_pack.cuh"
#include "native_tc_pack_transpose.cuh"
#include "native_tc_view.cuh"
#include "native_tc_product.cuh"
#include "native_tc_projection_kernel.cuh"
