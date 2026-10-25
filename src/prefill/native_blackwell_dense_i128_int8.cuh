// Dense-only I128/K256. Same canonical Q8 and ordered K32 finish; dynamic shared opt-in.
template<class F>
__launch_bounds__(256,1)
__global__ void dense_i128_group_int8(const typename F::Block* weights, float* output,
    int width, int rows, int outputs, NativeMmqEncodedInput encoded) {
    static_assert(F::K == 32);
    constexpr int Features=128;
    constexpr int Groups = 8, Words = 8, Stride = 12, Subs = 2, Msubs = Features/16;
    constexpr int RawStride = int(sizeof(typename F::Block))+4;
    constexpr int Raw = Features*RawStride, Meta = Features*Groups*int(sizeof(uint2));
    constexpr int Staging = Raw > Meta ? Raw : Meta;
    extern __shared__ __align__(16) unsigned char staging[];
    auto* aq = reinterpret_cast<unsigned*>(staging+Staging);
    auto* wm = reinterpret_cast<uint2*>(staging);
    const auto* bq = static_cast<const unsigned*>(encoded.codes);
    const auto* bm = static_cast<const GroupInputMeta*>(encoded.metadata);
    const int lane = int(threadIdx.x)%32, warp = int(threadIdx.x)/32;
    const int row0 = int(blockIdx.x)*Features, first = int(blockIdx.y)*128+warp*8;
    const int row = lane/4, col = 2*(lane%4), word = lane%4;
    int source_b[Subs], source0[Subs], source1[Subs];
#pragma unroll
    for (int sub = 0; sub < Subs; ++sub) {
        const int token = first+sub*64;
        source_b[sub] = token+row < rows ? token+row : 0;
        source0[sub] = token+col < rows ? token+col : 0;
        source1[sub] = token+col+1 < rows ? token+col+1 : 0;
    }
    float total[Subs][Msubs][4] = {};
    for (int panel = 0; panel < width/256; ++panel) {
        constexpr int RowWords = int(sizeof(typename F::Block))/4;
        for (int i = int(threadIdx.x); i < Features*RowWords; i += 256) {
            const auto* original = reinterpret_cast<const unsigned*>(weights+
                size_t(row0+i/RowWords)*(width/256)+panel);
            reinterpret_cast<unsigned*>(staging)[(i/RowWords)*(RowWords+1)+i%RowWords] = original[i%RowWords];
        }
        __syncthreads();
        uint2 saved[Features*Groups/256];
#pragma unroll
        for (int item = 0; item < Features*Groups/256; ++item) {
            const int cell = int(threadIdx.x)+item*256, g = cell/Features, r = cell%Features;
            const auto& w = *reinterpret_cast<const typename F::Block*>(staging+r*RawStride);
            saved[item] = F::meta(w,g);
#pragma unroll
            for (int v = 0; v < Words; ++v) {
                unsigned packed = 0;
#pragma unroll
                for (int p = 0; p < 4; ++p)
                    packed |= unsigned(uint8_t(F::code(w,g,v*4+p))) << (p*8);
                aq[cell*Stride+v] = packed;
            }
        }
        __syncthreads();
#pragma unroll
        for (int item = 0; item < Features*Groups/256; ++item)
            wm[int(threadIdx.x)+item*256] = saved[item];
        __syncthreads();
        for (int group = 0; group < Groups; ++group) {
            const int input_group = panel*Groups+group;
#pragma unroll
            for (int sub = 0; sub < Subs; ++sub) if (first+sub*64 < rows) {
                const size_t base = (size_t(input_group)*rows+source_b[sub])*Words+word;
                const unsigned b[2] = {bq[base],bq[base+4]};
                const auto x0 = bm[size_t(input_group)*rows+source0[sub]];
                const auto x1 = bm[size_t(input_group)*rows+source1[sub]];
#pragma unroll
                for (int m = 0; m < Msubs; ++m) {
                    const int r = m*16+row;
                    const auto w0 = wm[group*Features+r], w1 = wm[group*Features+r+8];
                    const unsigned a[4] = {aq[(group*Features+r)*Stride+word],
                        aq[(group*Features+r+8)*Stride+word],aq[(group*Features+r)*Stride+word+4],
                        aq[(group*Features+r+8)*Stride+word+4]};
                    float dot[4] = {};
                    blackwell_int8_mma(a,b,dot);
                    total[sub][m][0] = __fadd_rn(total[sub][m][0],F::finish(w0,x0.d8,x0.sum,dot[0]));
                    total[sub][m][1] = __fadd_rn(total[sub][m][1],F::finish(w0,x1.d8,x1.sum,dot[1]));
                    total[sub][m][2] = __fadd_rn(total[sub][m][2],F::finish(w1,x0.d8,x0.sum,dot[2]));
                    total[sub][m][3] = __fadd_rn(total[sub][m][3],F::finish(w1,x1.d8,x1.sum,dot[3]));
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int sub = 0; sub < Subs; ++sub)
#pragma unroll
        for (int m = 0; m < Msubs; ++m)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int j = 0; j < 2; ++j) if (first+sub*64+col+j < rows)
                    output[size_t(first+sub*64+col+j)*outputs+row0+m*16+row+h*8] = total[sub][m][h*2+j];
}
