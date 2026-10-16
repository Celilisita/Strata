// Adopted resident-Q8 route schedule; dot math uses the shared SmallTraits.
__global__ void mtp_q8_refs_kernel(const int32_t* ids, int hits, int k,
                                   const uint8_t* experts, size_t bytes, NativeMtpQ8Ref* refs) {
    const int i = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (i < hits) refs[i] = {reinterpret_cast<unsigned long long>(experts + size_t(ids[i]) * bytes), i, i / k};
}

template<int ROLES>
__global__ void mtp_q8_rows_kernel(const NativeMtpQ8Ref* refs, size_t role_offset,
                                   const Q81Block* inputs, float* output,
                                   int n_in, int n_out, int hits) {
    using F = SmallTraits<Q80Block, 8>;
    const int hit = int(blockIdx.y);
    const int row = int(blockIdx.x) * 8 + int(threadIdx.x) / 32;
    if (row >= n_out * ROLES) return;
    const int role = row / n_out, r = row % n_out, lane = int(threadIdx.x) % 32;
    const auto* blob = reinterpret_cast<const uint8_t*>(refs[hit].blob);
    const size_t offset = ROLES == 1 ? role_offset : size_t(role) * role_offset;
    const auto* weights = reinterpret_cast<const Q80Block*>(blob + offset) + size_t(r) * (n_in / 32);
    const auto* x = inputs + size_t(ROLES == 1 ? hit : refs[hit].token) * (n_in / 32);
    float sum = 0.0f;
    for (int q = lane; q < (n_in / F::DIV) * F::T; q += WARP) {
        const int block = q / F::T, component = F::kqs(q);
        sum += F::apply(F::load(weights + block, component), x + block * F::KBY, component);
    }
    sum = warp_sum(sum);
    if (lane == 0) {
        const size_t destination = ROLES == 1 ? size_t(refs[hit].dst) : size_t(role) * hits + hit;
        output[destination * n_out + r] = sum;
    }
}
