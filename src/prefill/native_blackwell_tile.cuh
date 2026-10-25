// M16 rows encode eight output rows and two separate native subproducts.
// Zero padding isolates each 4/8-term integer sum before native F32 scaling.
template<class F, bool Fp4, bool Grouped, bool Active, bool Int8 = false>
__launch_bounds__(256, 2)
__global__ void blackwell_matrix_mmq(const typename F::Block* dense,
    const NativeMmqMatrix* matrices, const NativeMmqTile* tiles, const int32_t* output_ids,
    const block_q8_1_mmq* input, float* output, int width, int rows, int outputs, int stride, int parts, NativeMmqEncodedInput encoded) {
    const int lane = int(threadIdx.x)%32, warp = int(threadIdx.x)/32;
    const int group = lane/4, column = 2*(lane%4);
    int end = rows, token0 = int(blockIdx.y)*64, part = 0;
    const typename F::Block* weights = dense;
    if constexpr (Grouped) {
        const auto t = Active ? tiles[blockIdx.y] : NativeMmqTile{int(blockIdx.z)/parts,int(blockIdx.y)};
        const auto m = matrices[t.matrix];
        part = int(blockIdx.z)%parts; end = m.end;
        token0 = m.begin+t.tile*64;
        weights = reinterpret_cast<const typename F::Block*>(uintptr_t(part ? m.up : m.gate));
    }
    if (token0 >= end) return;
    const int row0 = int(blockIdx.x)*8, blocks = width/256;
    // Load each contiguous quantized weight panel and activation panel once.
    // Their representation and integer codes stay unchanged in shared memory.
    extern __shared__ unsigned char staging[];
    auto* staged_weights = reinterpret_cast<typename F::Block*>(staging);
    const int row_words = blocks*int(sizeof(typename F::Block))/4;
    const int weight_stride = 4*(row_words+1);
    const int weight_words = 8*row_words;
    const auto* source_weights = reinterpret_cast<const unsigned*>(weights+size_t(row0)*blocks);
    for (int i = int(threadIdx.x); i < weight_words; i += int(blockDim.x))
        reinterpret_cast<unsigned*>(staged_weights)[(i/row_words)*(row_words+1)+i%row_words] = source_weights[i];
    __syncthreads();
    constexpr int AD = Int8 ? 1 : Fp4 ? F::WeightDigits4 : F::WeightDigits8, BD = Int8 ? 1 : Fp4 ? 3 : 2;
    constexpr int Words = 2*F::K/(Fp4 ? 8 : 4);
    auto* encoded_a = reinterpret_cast<unsigned*>(staging+8*weight_stride);
    const auto* encoded_b = static_cast<const unsigned*>(encoded.codes);
    const auto* input_meta = static_cast<const BlackwellInputMeta*>(encoded.metadata);
    encode_blackwell_weights<F,Fp4,Int8>(staged_weights,weight_stride,blocks,encoded_a);
    __syncthreads();
    __shared__ float partial[32][65];
    // Reuse one quantized/encoded weight panel for eight adjacent N8 panels.
    // The shared reduction buffer is consumed before the next panel overwrites it.
    const int first_token = token0;
#pragma unroll 1
    for (int panel = 0; panel < 8 && first_token+panel*8 < end; ++panel) {
        token0 = first_token+panel*8;
        const int source_b = token0+group < end ? (Grouped ? __ldg(encoded.source_ids+token0+group) : token0+group) : 0;
        const int source0 = token0+column < end ? (Grouped ? __ldg(encoded.source_ids+token0+column) : token0+column) : 0;
        const int source1 = token0+column+1 < end ? (Grouped ? __ldg(encoded.source_ids+token0+column+1) : token0+column+1) : 0;
        for (int slice = 0; slice < 4; ++slice) {
            const int native_lane = warp*4+slice;
            float sum[2] = {};
            for (int chain = 0; chain < 4; ++chain) {
                const int tid = chain*32+native_lane, iqs = F::kqs(tid);
                float chain_sum[2] = {};
                for (int kb = tid/F::T; kb < blocks; kb += F::BPI) {
                    const auto w = F::load(reinterpret_cast<const typename F::Block*>(staging+group*weight_stride)+kb,iqs);
                    float dots[4] = {};
                    const int code_group = kb*F::T+tid%F::T;
                    blackwell_encoded_product<F,Fp4,Int8>(encoded_a+(code_group*8+group)*AD*Words,
                        encoded_b+(size_t(code_group)*encoded.source_rows+source_b)*BD*Words,lane,dots);
    #pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        typename F::Input x{};
                        int sums[2] = {};
                        if (token0+column+j < end) {
                            const int source = j ? source1 : source0;
                            const int4 m = __ldg(reinterpret_cast<const int4*>(input_meta+
                                size_t(code_group)*encoded.source_rows+source));
                            x.d8[0] = __int_as_float(m.x); x.d8[1] = __int_as_float(m.y);
                            sums[0] = m.z; sums[1] = m.w;
                        }
                        const float product[2] = {dots[j],dots[j+2]};
                        chain_sum[j] += F::finish(w,x,product,sums);
                    }
                }
    #pragma unroll
                for (int j = 0; j < 2; ++j) sum[j] += chain_sum[j];
            }
    #pragma unroll
            for (int j = 0; j < 2; ++j) partial[native_lane][group*8+column+j] = sum[j];
        }
        __syncthreads();
        for (int cell = warp; cell < 64; cell += 8) {
            float value = partial[lane][cell];
    #pragma unroll
            for (int offset = 16; offset; offset >>= 1)
                value += __shfl_xor_sync(0xffffffff,value,offset);
            const int token = token0+cell%8;
            if (!lane && token < end) {
                const int dst = Grouped ? output_ids[token] : token;
                output[size_t(part)*rows*outputs+size_t(dst)*stride+row0+cell/8] = value;
            }
        }
        __syncthreads();
    }
}
