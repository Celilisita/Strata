#include "strata/qwen35/completed_v100/native/decode_graph_patch.hpp"
#include "strata/qwen35/completed_v100/native/attention_ops.hpp"
#include "strata/qwen35/completed_v100/native/cache_q8.hpp"
#include "strata/qwen35/completed_v100/native/ffn.hpp"
#include <cuda_fp16.h>
#include <cmath>
#include "strata/kernels/mrope.hpp"

namespace strata::completed_v100::qwen35::native {
namespace {
template<bool Visual=false>
__global__ void rotary(const float *x, const int *pos, float *out, int dim, int rot, int heads,
                       size_t pairs, float frequency_step, const int* table=nullptr) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= pairs) return;
    const int pair = int(index % (dim / 2));
    const size_t row = index / (dim / 2), offset = row * dim;
    if (pair >= rot / 2) {
        const int d = rot + (pair - rot / 2) * 2;
        out[offset + d] = x[offset + d]; out[offset + d + 1] = x[offset + d + 1];
        return;
    }
    const float frequency = __powf(frequency_step, float(pair));
    const int position = Visual ? strata::kernels::mrope_pos(table,pos[row / heads],pair) : pos[row / heads];
    const float angle = __fmul_rn(float(position), frequency);
    float s, c; __sincosf(angle, &s, &c);
    const float a = x[offset + pair], b = x[offset + pair + rot / 2];
    out[offset + pair] = __fmaf_rn(a, c, -__fmul_rn(b, s));
    out[offset + pair + rot / 2] = __fmaf_rn(a, s, __fmul_rn(b, c));
}
template<int Width> __global__ void butterflies(const float *x, float *out, size_t rows) {
    constexpr int PerLane = Width / 32;
    const size_t row = size_t(blockIdx.x) * 4 + threadIdx.x / 32;
    if (row >= rows) return;
    const int lane = threadIdx.x % 32;
    float values[PerLane];
    #pragma unroll
    for (int j = 0; j < PerLane; ++j) values[j] = x[row * Width + j * 32 + lane];
    #pragma unroll
    for (int stride = 1; stride < 32; stride *= 2) {
        #pragma unroll
        for (int j = 0; j < PerLane; ++j) {
            const float other = __shfl_xor_sync(0xffffffff, values[j], stride);
            values[j] = lane & stride ? __fsub_rn(other, values[j]) : __fadd_rn(values[j], other);
        }
    }
    #pragma unroll
    for (int stride = 1; stride < PerLane; stride *= 2) {
        #pragma unroll
        for (int j = 0; j < PerLane; j += 2 * stride) {
            #pragma unroll
            for (int k = 0; k < stride; ++k) {
                const float a = values[j+k], b = values[j+k+stride];
                values[j+k] = __fadd_rn(a,b); values[j+k+stride] = __fsub_rn(a,b);
            }
        }
    }
    constexpr float Scale = Width == 64 ? 0.125f : 0.0625f;
    #pragma unroll
    for (int j = 0; j < PerLane; ++j) out[row * Width + j * 32 + lane] = __fmul_rn(values[j], Scale);
}
template<bool Dynamic=false>
__global__ void cache_quantize(const float *x,unsigned char *cache,size_t blocks,size_t start_blocks,const int *position=nullptr) {
    if constexpr(Dynamic){const int at=*position;if(at<0||at>=262144)return;start_blocks=size_t(at)*blocks;}
    const size_t block = size_t(blockIdx.x) * 4 + threadIdx.x / 32;
    if (block >= blocks) return;
    const int lane = threadIdx.x % 32;
    const float value = x[block * 32 + lane];
    float maximum = value; int selected = lane;
    for (int d = 16; d; d /= 2) {
        const float other = __shfl_xor_sync(0xffffffff, maximum, d);
        const int other_index = __shfl_xor_sync(0xffffffff, selected, d);
        if (fabsf(other) > fabsf(maximum) || (fabsf(other) == fabsf(maximum) && other_index < selected)) {
            maximum = other; selected = other_index;
        }
    }
    const float delta = __fmul_rn(maximum, -0.125f), inv = delta == 0 ? 0 : __fdividef(1.0f, delta);
    const int code = min(15, max(0, __float2int_rz(__fmaf_rn(value, inv, 8.5f))));
    auto *out = cache + (start_blocks + block) * 18;
    const int paired = __shfl_sync(0xffffffff, code, lane + 16 < 32 ? lane + 16 : lane);
    if (!lane) *reinterpret_cast<__half *>(out) = __float2half_rn(delta);
    if (lane < 16) out[2 + lane] = static_cast<unsigned char>(code | (paired << 4));
}
__global__ void gate_values(const float *q, const float *a, float *raw, float *sig, float *out, int dim, size_t n) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float value = q[(i / dim) * (2 * dim) + dim + i % dim];
    const float sigmoid = __fdividef(1.0f, __fadd_rn(1.0f, __expf(-value)));
    raw[i] = value; sig[i] = sigmoid; out[i] = __fmul_rn(a[i], sigmoid);
}
}
void rotary_text(const float *x, const int *pos, float *out, int dim, int rot, int heads, int tokens, float base, cudaStream_t s) {
    if (!x || !pos || !out || dim < 32 || dim % 2 || rot < 2 || rot % 2 || rot > dim || heads < 1 || tokens < 1 ||
        !(base > 1) || !std::isfinite(base)) throw std::invalid_argument("invalid own text rotary geometry");
    const size_t count = size_t(tokens) * heads * (dim / 2);
    const float step = std::pow(base, -2.0f / float(rot));
    rotary<false><<<unsigned((count + 255)/256),256,0,s>>>(x,pos,out,dim,rot,heads,count,step);
    check_cuda(cudaGetLastError(), "own text rotary");
}
void rotary_visual(const float* x,const int* pos,float* out,int dim,int rot,int heads,int tokens,float base,cudaStream_t s,const int* table) {
    if (!x||!pos||!out||!table||dim!=256||rot!=64||heads<1||tokens<1||!(base>1)||!std::isfinite(base))
        throw std::invalid_argument("invalid Dense visual rotary geometry");
    const size_t count=size_t(tokens)*heads*(dim/2);
    const float step=std::pow(base,-2.0f/float(rot));
    rotary<true><<<unsigned((count+255)/256),256,0,s>>>(x,pos,out,dim,rot,heads,count,step,table);
    check_cuda(cudaGetLastError(),"Dense visual rotary");
}
void hadamard(const float *x, float *out, int width, size_t rows, cudaStream_t s) {
    if (!x || !out || !rows || (width != 64 && width != 256)) throw std::invalid_argument("invalid own orthogonal rotation");
    if (width == 64) butterflies<64><<<unsigned((rows+3)/4),128,0,s>>>(x,out,rows);
    else butterflies<256><<<unsigned((rows+3)/4),128,0,s>>>(x,out,rows);
    check_cuda(cudaGetLastError(), "own orthogonal rotation");
}
void q4_cache_write(const float *x, void *cache, int dim, int heads, int tokens, int start, int capacity, cudaStream_t s) {
    if (!x || !cache || dim < 32 || dim % 32 || heads < 1 || tokens < 1 || start < 0 || capacity < tokens || start > capacity-tokens)
        throw std::invalid_argument("invalid own Q4 cache write extent");
    const size_t blocks = size_t(tokens) * heads * (dim/32), first = size_t(start) * heads * (dim/32);
    cache_quantize<false><<<unsigned((blocks+3)/4),128,0,s>>>(x,static_cast<unsigned char *>(cache),blocks,first);
    check_cuda(cudaGetLastError(), "own Q4 cache write");
}
void attention_gate(const float *q, const float *a, float *raw, float *sig, float *out, int dim, int heads, int tokens, cudaStream_t s) {
    if (!q || !a || !raw || !sig || !out || dim < 1 || heads < 1 || tokens < 1) throw std::invalid_argument("invalid own attention gate");
    const size_t count = size_t(tokens) * heads * dim;
    gate_values<<<unsigned((count+255)/256),256,0,s>>>(q,a,raw,sig,out,dim,count);
    check_cuda(cudaGetLastError(), "own attention sigmoid gate");
}
DecodeNode decode_cache_node(void *function){return (function==reinterpret_cast<void *>(cache_quantize<false>)||function==reinterpret_cast<void *>(cache_quantize<true>))?DecodeNode::CacheWrite:decode_q8_cache_node(function);}
void update_decode_cache(cudaGraphExec_t exec,cudaGraphNode_t node,const cudaKernelNodeParams &original,int position){
    auto args=borrowed_graph_arguments<5>(original);const size_t blocks=*static_cast<const size_t *>(args[2]);
    if(position<0||position>=262144||!blocks||blocks>2048)throw std::invalid_argument("invalid graph cache row geometry");
    size_t first=size_t(position)*blocks;args[3]=&first;auto p=original;p.kernelParams=args.data();
    check_cuda(cudaGraphExecKernelNodeSetParams(exec,node,&p),"update own graph KV row offset");
}

void q4_cache_write_position(const float *x,void *cache,int dim,int heads,const int *position,cudaStream_t stream){
    if(!x||!cache||!position||dim<32||dim%32||heads<1)throw std::invalid_argument("invalid graph KV cursor geometry");
    const size_t blocks=size_t(heads)*(dim/32);
    cache_quantize<true><<<unsigned((blocks+3)/4),128,0,stream>>>(x,static_cast<unsigned char *>(cache),blocks,0,position);
    check_cuda(cudaGetLastError(),"own graph KV position write");
}

}
