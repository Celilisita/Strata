// Admitted batched schedule over the one mainline quant trait implementation.
template<typename F, int NCOLS, int NW, int ROWS, bool Bulk = false, bool ReuseInput = false>
__launch_bounds__(NW * WARP, 1)
__global__ void native_mmvq_bulk_kernel(const typename F::Block* __restrict__ w,
                                         const Q81Block* __restrict__ x,
                                         float* __restrict__ y, int n_in, int n_out) {
    constexpr int BPI = F::BPI * NW / WARPS;           // blocks per iteration scale with the warp count
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / F::DIV;
    const int x_stride = n_in / Q8K;                   // Q8_1 blocks per activation column
    // Each grid.y cell is the same canonical column tile, with disjoint output.
    if constexpr (Bulk) {
        x += std::size_t(blockIdx.y) * NCOLS * x_stride;
        y += std::size_t(blockIdx.y) * NCOLS * n_out;
    }
    float tmp[NCOLS][ROWS] = {};
    for (int kbx = tid / F::T; kbx < blocks_per_row; kbx += BPI) {
        const int kby = kbx * F::KBY;
        const int kqs = F::kqs(tid);
        if constexpr (ReuseInput) {
            // Explicit row-tiling grid guarantees row0 is valid. Preload one
            // canonical activation view per column for all independent rows.
            typename F::W rows[ROWS];
#pragma unroll
            for (int i = 0; i < ROWS; ++i)
                if (i == 0 || row0 + i < n_out)
                    rows[i] = F::load(w + std::size_t(row0 + i) * blocks_per_row + kbx, kqs);
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
                const auto in = MultiInputReuse<F>::load(rows[0], x + std::size_t(j) * x_stride + kby, kqs);
#pragma unroll
                for (int i = 0; i < ROWS; ++i)
                    if (row0 + i < n_out)
                        tmp[j][i] += MultiInputReuse<F>::apply(rows[i], in);
            }
        } else {
    #pragma unroll
            for (int i = 0; i < ROWS; ++i) {
                if (row0 + i < n_out) {
                    const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                    const typename F::W wv = F::load(w + block, kqs);      // once per (row, block)
    #pragma unroll
                    for (int j = 0; j < NCOLS; ++j)                         // then per column
                        tmp[j][i] += F::apply(wv, x + std::size_t(j) * x_stride + kby, kqs);
                }
            }
        }
    }
    __shared__ float partial[NW - 1 > 0 ? NW - 1 : 1][NCOLS][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < NCOLS; ++j)
#pragma unroll
            for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][j][i][threadIdx.x] = tmp[j][i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
#pragma unroll
            for (int l = 0; l < NW - 1; ++l) tmp[j][i] += partial[l][j][i][threadIdx.x];
            tmp[j][i] = warp_sum(tmp[j][i]);
            if (threadIdx.x == i && row0 + i < n_out) y[std::size_t(j) * n_out + row0 + i] = tmp[j][i];
        }
    }
}
