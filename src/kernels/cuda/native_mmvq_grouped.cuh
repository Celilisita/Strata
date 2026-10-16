// Included inside native_mmvq.cu's anonymous namespace after its quantized
// weight traits. This shares the exact Q4_K/Q6_K dot math with the one-column
// implementation without duplicating it in a Qwen-specific kernel.
template<typename F>
__launch_bounds__(WARPS * WARP, 1)
__global__ void native_mmvq_grouped_kernel(const unsigned long long* __restrict__ addresses,
                                           const int32_t* __restrict__ input_rows,
                                           const Q81Block* __restrict__ inputs,
                                           float* __restrict__ output, int n_in, int n_out, int entries) {
    const int entry = int(blockIdx.y) + int(blockIdx.z) * 16384, row = int(blockIdx.x);
    if (entry >= entries) return;
    const auto* weights = reinterpret_cast<const typename F::Block*>(
        static_cast<uintptr_t>(addresses[entry]));
    if (!weights) return; // A nullable row is assigned to another provider.
    const auto* x = inputs + size_t(input_rows[entry]) * (n_in / Q8K);
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int blocks_per_row = n_in / F::DIV;
    float value = 0.0f;
    for (int kbx = tid / F::T; kbx < blocks_per_row; kbx += F::BPI) {
        const int kqs = F::kqs(tid);
        const auto wv = F::load(weights + size_t(row) * blocks_per_row + kbx, kqs);
        value += F::apply(wv, x + kbx * F::KBY, kqs);
    }
    __shared__ float partial[WARPS - 1][WARP];
    if (threadIdx.y > 0) partial[threadIdx.y - 1][threadIdx.x] = value;
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int warp = 0; warp < WARPS - 1; ++warp)
        value += partial[warp][threadIdx.x];
    value = warp_sum(value);
    if (threadIdx.x == 0) output[size_t(entry) * n_out + row] = value;
}
