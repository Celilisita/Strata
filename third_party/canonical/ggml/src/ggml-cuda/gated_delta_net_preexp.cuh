#pragma once

// Included after common.cuh; deliberately private to gated_delta_net.cu.
// This is the same expf as the original KDA kernel, with no clamp/approximation.
template <bool PreExp>
static __device__ __forceinline__ float ggml_cuda_gdn_decay(float gate) {
    if constexpr (PreExp) {
        return gate;
    } else {
        return expf(gate);
    }
}

static __global__ void ggml_cuda_gdn_preexp_f32(const float * gate, float * decay, int64_t count) {
    // ggml_cuda_kernel_launch may use PDL: do not read the preceding gate producer
    // until its dependency completes. No early completion signal is published.
    ggml_cuda_pdl_sync();
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        decay[i] = expf(gate[i]);
    }
}

static void ggml_cuda_gdn_launch_preexp(
        const float * gate, float * decay, int64_t count, cudaStream_t stream) {
    constexpr int threads = 256;
    const dim3 blocks(static_cast<unsigned>((count + threads - 1) / threads));
    const ggml_cuda_kernel_launch_params params(blocks, dim3(threads), 0, stream);
    ggml_cuda_kernel_launch(ggml_cuda_gdn_preexp_f32, params, gate, decay, count);
}
