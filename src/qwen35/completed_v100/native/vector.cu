#include "strata/qwen35/completed_v100/native/ffn.hpp"
#include "math.cuh"
#include "vector_fragment.cuh"
#include <cuda_fp16.h>
#include <cstdint>

namespace strata::completed_v100::qwen35::native {
namespace {
using detail::ActivationBlock;
using detail::accumulate_fragment;
__global__ void quantize(const float *input, ActivationBlock *output, int columns) {
    const int group = int(blockIdx.x) * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (group >= columns / 32) return;
    const float value = input[group * 32 + lane];
    float maximum = fabsf(value);
    for (int offset = 16; offset; offset /= 2) {
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
    }
    const float delta = __fmul_rn(maximum, 1.0f / 127.0f);
    const int code = maximum ? int(roundf(__fdividef(value, delta))) : 0;
    output[group].codes[lane] = static_cast<signed char>(code);
    if (!lane) output[group].delta = __float2half_rn(delta);
}
template<Quant Type, bool Pair, int Parts, bool PackedMetadata,bool Tiled=false>
__global__ void multiply(Matrix a, Matrix b, const ActivationBlock *input, float *first,
                         float *second, float *activation) {
    // Four logical K stripes define scalar rounding, independently of the
    // physical CTA size. Two-warp projections keep two stripes per worker.
    constexpr int LogicalThreads=128,Extra=Parts==2?1:0,Workers=Type==Quant::Q4K?16:32;
    __shared__ float partial[(Pair?2:1)*LogicalThreads];
    const int row=int(blockIdx.x),tid=threadIdx.x,lane=tid%32;
    if(row>=a.rows)return;
    constexpr int Bytes=Type==Quant::Q4K?144:210;
    float one[Extra+1]{},two[Extra+1]{};
    #pragma unroll
    for(int stripe=0;stripe<=Extra;++stripe){
        const int logical=tid+stripe*64;
        for(int block=logical/Workers;block<a.columns/256;block+=LogicalThreads/Workers){
            const int blocks=a.columns/256;
            const size_t offset=Tiled?((size_t(row/64)*blocks+block)*64+row%64)*16:(size_t(row)*blocks+block)*Bytes;
            const size_t code_offset=size_t(a.rows)*blocks*16+(size_t(row)*blocks+block)*128;
            const auto *ac=static_cast<const unsigned char *>(a.data),*bc=static_cast<const unsigned char *>(b.data);
            one[stripe]=accumulate_fragment<Type,PackedMetadata,Tiled>(ac+offset,input+block*8,one[stripe],Tiled?ac+code_offset:nullptr);
            if constexpr(Pair)two[stripe]=accumulate_fragment<Type,PackedMetadata,Tiled>(bc+offset,input+block*8,two[stripe],Tiled?bc+code_offset:nullptr);
        }
        partial[logical]=one[stripe];
        if constexpr(Pair)partial[LogicalThreads+logical]=two[stripe];
    }
    __syncthreads();
    if(tid<32){
        float x=partial[lane],y=0;
        if constexpr(Pair)y=partial[LogicalThreads+lane];
        #pragma unroll
        for(int stripe=1;stripe<4;++stripe){
            x=__fadd_rn(x,partial[stripe*32+lane]);
            if constexpr(Pair)y=__fadd_rn(y,partial[LogicalThreads+stripe*32+lane]);
        }
        for(int offset=16;offset;offset/=2){
            x=__fadd_rn(x,__shfl_xor_sync(0xffffffff,x,offset));
            if constexpr(Pair)y=__fadd_rn(y,__shfl_xor_sync(0xffffffff,y,offset));
        }
        if(!lane){first[row]=x;if constexpr(Pair){second[row]=y;activation[row]=swiglu_value(x,y);}}
    }
}
template<bool Pair>
void launch(Matrix a, Matrix b, const void *input, float *first, float *second,
            float *activation, cudaStream_t stream) {
    const int blocks = a.rows;
    if (a.quant == Quant::Q4K) {
        constexpr int Parts = Pair ? 4 : 2;
        if(a.layout==BlockLayout::Q4Tiles){
            matrix_bytes(a,false,false,true);matrix_bytes(b,false,false,true);
            multiply<Quant::Q4K,Pair,Parts,true,true><<<blocks,Parts*32,0,stream>>>(
                a,b,static_cast<const ActivationBlock *>(input),first,second,activation);
        }else if (a.layout == BlockLayout::Q4Pairs)
            multiply<Quant::Q4K, Pair, Parts, true><<<blocks, Parts * 32, 0, stream>>>(
                a, b, static_cast<const ActivationBlock *>(input), first, second, activation);
        else
            multiply<Quant::Q4K, Pair, Parts, false><<<blocks, Parts * 32, 0, stream>>>(
                a, b, static_cast<const ActivationBlock *>(input), first, second, activation);
    } else
        multiply<Quant::Q6K, Pair, 4, false><<<blocks, 128, 0, stream>>>(
            a, b, static_cast<const ActivationBlock *>(input), first, second, activation);
    check_cuda(cudaGetLastError(), "native integer vector projection");
}
}
void vector_quantize(const float *input, void *output, int columns, cudaStream_t stream) {
    quantize<<<(columns / 32 + 3) / 4, 128, 0, stream>>>(input, static_cast<ActivationBlock *>(output), columns);
    check_cuda(cudaGetLastError(), "native vector activation quantization");
}
void vector_pair(Matrix gate, Matrix up, const void *input, float *first, float *second,
                 float *activation, cudaStream_t stream) {
    if (gate.quant != up.quant || gate.layout != up.layout || !activation)
        throw std::invalid_argument("invalid native vector pair");
    launch<true>(gate, up, input, first, second, activation, stream);
}
void vector_linear(Matrix weight, const void *input, float *output, cudaStream_t stream) {
    launch<false>(weight, weight, input, output, nullptr, nullptr, stream);
}
} // namespace strata::completed_v100::qwen35::native
