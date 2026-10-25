// Included after unchanged Q4KTraits inside native_mmvq.cu's anonymous namespace.
// Two contiguous output rows per CTA; each row owns two warps and all eight K blocks.
__launch_bounds__(4 * WARP, 1)
__global__ void native_q4_k_dense_row2_kernel(const Q4KBlock* __restrict__ weights,
    const Q81Block* __restrict__ input, float* __restrict__ output) {
    using F = Q4KTraits;
    static_assert(F::DIV == 256 && F::T == 16 && F::KBY == 8 && WARP == 32);
    const int local_row = int(threadIdx.y) / 2;
    const int row = 2 * int(blockIdx.x) + local_row;
    const int lane = int(threadIdx.x), local_warp = int(threadIdx.y) % 2;
    const int tid = local_warp * WARP + lane;
    float value = 0.0f;
    for (int kbx = tid / F::T; kbx < 2048 / F::DIV; kbx += 2 * WARP / F::T) {
        const int kqs = F::kqs(tid);
        const auto wv = F::load(weights + size_t(row) * (2048 / F::DIV) + kbx, kqs);
        value += F::apply(wv, input + kbx * F::KBY, kqs);
    }
    __shared__ float partial[2][WARP];
    if (local_warp) partial[local_row][lane] = value;
    __syncthreads();
    if (local_warp) return;
    value += partial[local_row][lane];
    value = warp_sum(value);
    if (lane == 0) output[row] = value;
}
