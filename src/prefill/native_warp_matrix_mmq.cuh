// One warp owns a matrix row. Preserve all four native K chains, merge them
// in order, then use the same XOR reduction without a CTA shared join.
template<class F, bool Active>
__launch_bounds__(128, 4)
__global__ void canonical_warp_matrix_mmq(const NativeMmqMatrix* matrices,
    const NativeMmqTile* tiles, const int32_t* output_ids,
    const block_q8_1_mmq* input, float* output, int width, int rows, int outputs, int parts) {
    const int lane = int(threadIdx.x)%32, warp = int(threadIdx.x)/32;
    const auto tile = Active ? tiles[blockIdx.y/8] : NativeMmqTile{int(blockIdx.z)/parts,int(blockIdx.y)/8};
    const auto m = matrices[tile.matrix];
    const int part = int(blockIdx.z)%parts;
    const int token0 = m.begin+tile.tile*64+int(blockIdx.y%8)*8;
    if (token0 >= m.end) return;
    const auto* weights = reinterpret_cast<const typename F::Block*>(uintptr_t(part ? m.up : m.gate));
    const int row = int(blockIdx.x)*4+warp, blocks = width/256;
    float sum[8] = {};
    for (int chain = 0; chain < 4; ++chain) {
        const int tid = chain*32+lane, iqs = F::kqs(tid);
        float contribution[8] = {};
        for (int kb = tid/F::T; kb < blocks; kb += F::BPI) {
            const auto w = F::load(weights+size_t(row)*blocks+kb,iqs);
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                if (token0+c < m.end)
                    contribution[c] += F::apply_preloaded(w,F::input(w,input,rows,token0+c,kb,iqs));
            }
        }
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            if (!chain) sum[c] = contribution[c];
            else sum[c] += contribution[c];
        }
    }
#pragma unroll
    for (int c = 0; c < 8; ++c) {
        if (token0+c >= m.end) continue;
#pragma unroll
        for (int offset = 16; offset; offset >>= 1)
            sum[c] += __shfl_xor_sync(0xffffffff,sum[c],offset);
        if (!lane)
            output[size_t(part)*rows*outputs+size_t(output_ids[token0+c])*outputs+row] = sum[c];
    }
}
