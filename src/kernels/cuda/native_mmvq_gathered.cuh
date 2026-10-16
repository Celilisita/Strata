// Included after the canonical quant traits. Gathered columns retain the
// one-column K iteration and warp reduction, sharing only weight loads.
template<typename F, int Rows = 1>
__launch_bounds__(WARPS * WARP, Rows == 2 ? 8 : 1)
__global__ void native_mmvq_gathered_kernel(const strata::kernels::NativeTcTile* tiles,
    const Q81Block* __restrict__ inputs, float* __restrict__ output, int n_in, int n_out) {
    const auto& tile = tiles[blockIdx.y];
    const int row = Rows * int(blockIdx.x), first = int(blockIdx.z) * 8;
    const auto* weights = reinterpret_cast<const typename F::Block*>(
        static_cast<uintptr_t>(tile.address));
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int blocks_per_row = n_in / F::DIV;
    float values[8][Rows] = {};
    for (int kbx = tid / F::T; kbx < blocks_per_row; kbx += F::BPI) {
        const int kqs = F::kqs(tid);
        if constexpr (Rows == 1) {
            const auto wv = F::load(weights + size_t(row) * blocks_per_row + kbx, kqs);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int input_row = tile.input_rows[first + j];
                if (input_row >= 0) {
                    const auto* x = inputs + size_t(input_row) * (n_in / Q8K);
                    values[j][0] += F::apply(wv, x + kbx * F::KBY, kqs);
                }
            }
        } else {
            typename F::W wv[Rows];
#pragma unroll
            for (int i = 0; i < Rows; ++i)
                if (row + i < n_out)
                    wv[i] = F::load(weights + size_t(row + i) * blocks_per_row + kbx, kqs);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int input_row = tile.input_rows[first + j];
                if (input_row >= 0) {
                    const auto* x = inputs + size_t(input_row) * (n_in / Q8K) + kbx * F::KBY;
                    const auto in = MultiInputReuse<F>::load(wv[0], x, kqs);
#pragma unroll
                    for (int i = 0; i < Rows; ++i)
                        if (row + i < n_out)
                            values[j][i] += MultiInputReuse<F>::apply(wv[i], in);
                }
            }
        }
    }
    __shared__ float partial[WARPS - 1][8][Rows][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int i = 0; i < Rows; ++i)
                partial[threadIdx.y - 1][j][i][threadIdx.x] = values[j][i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
#pragma unroll
        for (int i = 0; i < Rows; ++i) {
#pragma unroll
            for (int warp = 0; warp < WARPS - 1; ++warp)
                values[j][i] += partial[warp][j][i][threadIdx.x];
            values[j][i] = warp_sum(values[j][i]);
            const int output_row = tile.output_rows[first + j];
            if (threadIdx.x == 0 && output_row >= 0 && row + i < n_out)
                output[size_t(output_row) * n_out + row + i] = values[j][i];
        }
    }
}
