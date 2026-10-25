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


// One complete 512-wide Down row per warp. Four rows share an entry/CTA.
// Quantized dot traits are unchanged; no cache/route/epoch authority moves here.
template<typename F>
__launch_bounds__(4 * WARP, 1)
__global__ void native_mmvq_grouped_down_row4_kernel(
    const unsigned long long* __restrict__ addresses,
    const int32_t* __restrict__ input_rows, const Q81Block* __restrict__ inputs,
    float* __restrict__ output, int n_out) {
    const int entry = int(blockIdx.y);
    const int row = 4 * int(blockIdx.x) + int(threadIdx.y);
    const auto* weights = reinterpret_cast<const typename F::Block*>(
        static_cast<uintptr_t>(addresses[entry]));
    if (!weights) return; // The existing miss provider owns every output of this entry.
    const auto* x = inputs + size_t(input_rows[entry]) * (512 / Q8K);
    const int lane = int(threadIdx.x);
    float value = 0.0f;
    // Q4/Q5: two blocks occupy 32 lanes. Q6: each lane visits both blocks.
    for (int kbx = lane / F::T; kbx < 512 / F::DIV; kbx += WARP / F::T) {
        const int kqs = F::kqs(lane);
        const auto wv = F::load(weights + size_t(row) * (512 / F::DIV) + kbx, kqs);
        value += F::apply(wv, x + kbx * F::KBY, kqs);
    }
    value = warp_sum(value);
    if (lane == 0) output[size_t(entry) * n_out + row] = value;
}

// Two complete 2048-wide Q4 Gate/Up output rows per CTA, two warps per row.
// Exact same quant traits; only K ownership and cross-warp reduction change.
__launch_bounds__(4 * WARP, 1)
__global__ void native_mmvq_grouped_gu_row2_kernel(
    const unsigned long long* __restrict__ addresses,
    const int32_t* __restrict__ input_rows, const Q81Block* __restrict__ inputs,
    float* __restrict__ output) {
    using F = Q4KTraits;
    static_assert(F::DIV == 256 && F::T == 16 && F::KBY == 8 && WARP == 32);
    const int entry = int(blockIdx.y), local_row = int(threadIdx.y) / 2;
    const int row = 2 * int(blockIdx.x) + local_row;
    const auto* weights = reinterpret_cast<const F::Block*>(static_cast<uintptr_t>(addresses[entry]));
    if (!weights) return; // All four warps agree; no other provider's output is touched.
    const auto* x = inputs + size_t(input_rows[entry]) * (2048 / Q8K);
    const int lane = int(threadIdx.x), local_warp = int(threadIdx.y) % 2;
    const int tid = local_warp * WARP + lane;
    float value = 0.0f;
    for (int kbx = tid / F::T; kbx < 2048 / F::DIV; kbx += 2 * WARP / F::T) {
        const int kqs = F::kqs(tid);
        const auto wv = F::load(weights + size_t(row) * (2048 / F::DIV) + kbx, kqs);
        value += F::apply(wv, x + kbx * F::KBY, kqs);
    }
    __shared__ float partial[2][WARP];
    if (local_warp) partial[local_row][lane] = value;
    __syncthreads();
    if (local_warp) return;
    value += partial[local_row][lane];
    value = warp_sum(value);
    if (lane == 0) output[size_t(entry) * 512 + row] = value;
}

#include "native_mmvq_dense_q4_row2.cuh"

#include "native_mmvq_gu_cluster.cuh"

#include "native_mmvq_quant_down_pair.cuh"
