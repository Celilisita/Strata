// Canonical Q8 signed codes and Q4/Q5 integer weights: one exact K32 IMMA.
// Reuses the existing code/meta arena, stream, 512-K panel and dense row owner.
__device__ __forceinline__ unsigned dense_int8_word(const Q81Block& x, int word) {
    unsigned result = 0;
#pragma unroll
    for (int p = 0; p < 4; ++p) result |= unsigned(uint8_t(x.qs[word*4+p])) << (p*8);
    return result;
}
__global__ void encode_dense_group_int8(const Q81Block* source, unsigned* codes,
    GroupInputMeta* metadata, int width, int rows) {
    const int cell = int(blockIdx.x)*int(blockDim.x)+int(threadIdx.x);
    if (cell >= rows*(width/32)) return;
    const int group = cell/rows, token = cell%rows;
    const auto& x = source[size_t(token)*(width/32)+group];
    int sum = 0;
#pragma unroll
    for (int k = 0; k < 32; ++k) sum += int(x.qs[k]);
    metadata[cell] = {__half2float(__low2half(x.ds)), sum};
#pragma unroll
    for (int word = 0; word < 8; ++word) codes[size_t(cell)*8+word] = dense_int8_word(x,word);
}
template<class F, int Tile>
__launch_bounds__(256,2)
__global__ void dense_group_int8(const typename F::Block* weights, float* output,
    int width, int rows, int outputs, NativeMmqEncodedInput encoded) {
    static_assert(F::K == 32 && Tile == 128);
    constexpr int Panel = 512, Groups = 16, Words = 8, Stride = 12, Subs = Tile/64;
    constexpr int RawStride = (Panel/256)*int(sizeof(typename F::Block))+4;
    constexpr int Raw = 16*RawStride, Meta = 16*Groups*int(sizeof(uint2));
    constexpr int Staging = Raw > Meta ? Raw : Meta;
    __shared__ __align__(16) unsigned char staging[Staging+16*Groups*Stride*4];
    auto* aq = reinterpret_cast<unsigned*>(staging+Staging);
    auto* wm = reinterpret_cast<uint2*>(staging);
    const auto* bq = static_cast<const unsigned*>(encoded.codes);
    const auto* bm = static_cast<const GroupInputMeta*>(encoded.metadata);
    const int lane = int(threadIdx.x)%32, warp = int(threadIdx.x)/32;
    const int row0 = int(blockIdx.x)*16, first = int(blockIdx.y)*Tile+warp*8;
    const int row = lane/4, col = 2*(lane%4), word = lane%4;
    int source_b[Subs], source0[Subs], source1[Subs];
#pragma unroll
    for (int sub = 0; sub < Subs; ++sub) {
        const int token = first+sub*64;
        source_b[sub] = token+row < rows ? token+row : 0;
        source0[sub] = token+col < rows ? token+col : 0;
        source1[sub] = token+col+1 < rows ? token+col+1 : 0;
    }
    float total[Subs][4] = {};
    for (int panel = 0; panel < width/Panel; ++panel) {
        constexpr int RowWords = (Panel/256)*int(sizeof(typename F::Block))/4;
        for (int i = int(threadIdx.x); i < 16*RowWords; i += 256) {
            const auto* original = reinterpret_cast<const unsigned*>(weights+
                size_t(row0+i/RowWords)*(width/256)+panel*2);
            reinterpret_cast<unsigned*>(staging)[(i/RowWords)*(RowWords+1)+i%RowWords] = original[i%RowWords];
        }
        __syncthreads();
        const int cell = int(threadIdx.x), g = cell/16, r = cell%16;
        const auto& w = reinterpret_cast<const typename F::Block*>(staging+r*RawStride)[g/8];
        const uint2 saved = F::meta(w,g%8);
#pragma unroll
        for (int v = 0; v < Words; ++v) {
            unsigned packed = 0;
#pragma unroll
            for (int p = 0; p < 4; ++p)
                packed |= unsigned(uint8_t(F::code(w,g%8,v*4+p))) << (p*8);
            aq[cell*Stride+v] = packed;
        }
        __syncthreads();
        wm[cell] = saved;
        __syncthreads();
        for (int group = 0; group < Groups; ++group) {
            const int input_group = panel*Groups+group;
            const auto w0 = wm[group*16+row], w1 = wm[group*16+row+8];
            unsigned a[4] = {aq[(group*16+row)*Stride+word],
                aq[(group*16+row+8)*Stride+word], aq[(group*16+row)*Stride+word+4],
                aq[(group*16+row+8)*Stride+word+4]};
#pragma unroll
            for (int sub = 0; sub < Subs; ++sub) if (first+sub*64 < rows) {
                const size_t base = (size_t(input_group)*rows+source_b[sub])*Words+word;
                unsigned b[2] = {bq[base], bq[base+4]};
                float dot[4] = {};
                blackwell_int8_mma(a,b,dot);
                const auto x0 = bm[size_t(input_group)*rows+source0[sub]];
                const auto x1 = bm[size_t(input_group)*rows+source1[sub]];
                total[sub][0] = __fadd_rn(total[sub][0],F::finish(w0,x0.d8,x0.sum,dot[0]));
                total[sub][1] = __fadd_rn(total[sub][1],F::finish(w0,x1.d8,x1.sum,dot[1]));
                total[sub][2] = __fadd_rn(total[sub][2],F::finish(w1,x0.d8,x0.sum,dot[2]));
                total[sub][3] = __fadd_rn(total[sub][3],F::finish(w1,x1.d8,x1.sum,dot[3]));
            }
        }
        __syncthreads(); // Uniform protection before the shared panel is reused.
    }
#pragma unroll
    for (int sub = 0; sub < Subs; ++sub)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int j = 0; j < 2; ++j) if (first+sub*64+col+j < rows)
                output[size_t(first+sub*64+col+j)*outputs+row0+row+h*8] = total[sub][h*2+j];
}
template<class F, int Tile>
void launch_dense_group_int8(const void* weights, const void* input, void* codes, void* meta,
    float* output, int width, int rows, int outputs, void* stream) {
    const int cells = rows*(width/32);
    encode_dense_group_int8<<<(cells+127)/128,128,0,static_cast<cudaStream_t>(stream)>>>(
        static_cast<const Q81Block*>(input),static_cast<unsigned*>(codes),
        static_cast<GroupInputMeta*>(meta),width,rows);
    NativeMmqEncodedInput encoded{codes,meta,nullptr,rows,Tile};
    dense_group_int8<F,Tile><<<dim3(outputs/16,(rows+Tile-1)/Tile),256,0,static_cast<cudaStream_t>(stream)>>>(
        static_cast<const typename F::Block*>(weights),output,width,rows,outputs,encoded);
}
