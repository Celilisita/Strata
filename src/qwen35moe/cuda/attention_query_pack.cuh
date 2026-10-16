// Reuse the dead Q half of [Q256, gate256] projection slots. Gate stays intact.
namespace {
__global__ void pack_attention_query(const float* q, half2* slots, int count) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(count) * kHeads * 128) return;
    const size_t head = i / 128, d = i % 128;
    const half2 scaled = __floats2half2_rn(q[head * kDim + 2 * d] * 0.0625f,
        q[head * kDim + 2 * d + 1] * 0.0625f);
    slots[head * 2 * kDim + d] = __hmul2(scaled, __float2half2_rn(0.25f));
}
}
void attention_pack_query(const float* q, float* slots, int count, void* stream) {
    if (!q || !slots || q == slots || !stream || count < 1 || count > 8192)
        throw std::invalid_argument("invalid packed attention query slots");
    configure_attention_shared<true>();
    pack_attention_query<<<unsigned((size_t(count) * kHeads * 128 + 255) / 256), 256, 0,
        static_cast<cudaStream_t>(stream)>>>(q, reinterpret_cast<half2*>(slots), count);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
