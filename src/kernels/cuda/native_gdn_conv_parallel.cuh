// Independent time windows keep the original tap/FMA/SiLU expressions. The
// first window reads old history; later windows read immutable earlier input.
// Only the second same-stream kernel may overwrite history after every reader.
constexpr int ConvWindow = 32;
__global__ void conv_silu_parallel(const float* __restrict__ history,
    const float* __restrict__ input, const float* __restrict__ weights,
    float* __restrict__ output, int channels, int tokens) {
    const int c = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    const int first = int(blockIdx.y) * ConvWindow;
    if (c >= channels || first >= tokens) return;
    float values[4];
#pragma unroll
    for (int tap = 0; tap < 3; ++tap)
        values[tap] = first ? input[size_t(first - 3 + tap) * channels + c] : history[c * 3 + tap];
    const float w[4] = {weights[c * 4], weights[c * 4 + 1], weights[c * 4 + 2], weights[c * 4 + 3]};
    for (int t = first; t < tokens && t < first + ConvWindow; ++t) {
        values[3] = input[size_t(t) * channels + c];
        float sum = 0.0f;
#pragma unroll
        for (int tap = 0; tap < 4; ++tap) sum += values[tap] * w[tap];
        sum = __fadd_rn(sum, 0.0f);
        output[size_t(t) * channels + c] = sum / (1.0f + expf(-sum));
#pragma unroll
        for (int tap = 0; tap < 3; ++tap) values[tap] = values[tap + 1];
    }
}
__global__ void conv_history_update(float* history, const float* input, int channels, int tokens) {
    const int c = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (c >= channels) return;
    float final_values[3];
#pragma unroll
    for (int tap = 0; tap < 3; ++tap) {
        const int source = tokens - 3 + tap;
        final_values[tap] = source < 0 ? history[c * 3 + source + 3] : input[size_t(source) * channels + c];
    }
#pragma unroll
    for (int tap = 0; tap < 3; ++tap) history[c * 3 + tap] = final_values[tap];
}
