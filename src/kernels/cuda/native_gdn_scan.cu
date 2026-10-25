// Arithmetic adapted from gated_delta_net.cu at pinned llama.cpp
// 3cf03257f219afbe7334045ff7c6a06ac68c627d. Only state addressing differs:
// each warp owns one Strata (head,column) and retains its four rows in registers.
// Compile with --use_fast_math to match the pinned CUDA implementation.
//
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "strata/kernels/native_gdn.hpp"
#include <cuda_runtime.h>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strata::kernels {
namespace {

constexpr int S = 128;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_xor_sync(0xffffffff, value, offset, 32);
    return value;
}

#include "native_gdn_step.cuh"
#include "native_gdn_pair2.cuh"
#include "native_gdn_chunk4.cuh"

bool valid_span(const void* pointer, size_t bytes) {
    const auto address = reinterpret_cast<uintptr_t>(pointer);
    return pointer && address % sizeof(float) == 0 && bytes <= UINTPTR_MAX - address;
}
bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    const auto ap = reinterpret_cast<uintptr_t>(a), bp = reinterpret_cast<uintptr_t>(b);
    return ap < bp + bn && bp < ap + an;
}
}

static void validate_gdn(bool scan, int tokens, int input_stride, float* state, const float* q, const float* k, const float* v,
                     const float* gate, const float* beta, float* output,
                     const GdnShapes& shape, void* stream, float* prefix = nullptr, int columns = 1) {
    const bool valid_columns = columns == 1 || columns == 2 || columns == 4 || columns == 8;
    // Bound signed geometry before computing the strided input width. h_k<=h_v
    // follows from positive, divisible head counts and keeps that width in int.
    if (!valid_columns || (columns != 1 && (!scan || prefix)) ||
        tokens < 1 || tokens > 8192 || !stream || shape.S != S ||
        shape.h_k <= 0 || shape.h_v <= 0 || shape.h_v > 65535 ||
        shape.h_v % shape.h_k != 0 ||
        input_stride < (2 * shape.h_k + shape.h_v) * S)
        throw std::invalid_argument("native GDN requires a stream, S=128 and positive divisible head counts <=65535");
    const size_t state_bytes = size_t(S) * S * size_t(shape.h_v) * sizeof(float);
    const size_t qk_bytes = (size_t(tokens - 1) * input_stride + size_t(S) * shape.h_k) * sizeof(float);
    const size_t output_bytes = size_t(tokens) * S * shape.h_v * sizeof(float);
    const size_t v_bytes = (size_t(tokens - 1) * input_stride + size_t(S) * shape.h_v) * sizeof(float);
    const size_t head_bytes = size_t(tokens) * shape.h_v * sizeof(float);
    if (!valid_span(state, state_bytes) || !valid_span(output, output_bytes) ||
        overlap(state, state_bytes, output, output_bytes))
        throw std::invalid_argument("native GDN requires aligned, disjoint state and output spans");
    const void* inputs[] = {q, k, v, gate, beta};
    const size_t bytes[] = {qk_bytes, qk_bytes, v_bytes, head_bytes, head_bytes};
    for (int i = 0; i < 5; ++i) {
        if (!valid_span(inputs[i], bytes[i]) || overlap(state, state_bytes, inputs[i], bytes[i]) ||
            overlap(output, output_bytes, inputs[i], bytes[i]))
            throw std::invalid_argument("native GDN requires aligned input spans disjoint from state and output");
    }
    if (prefix) {
        const size_t checkpoint_bytes = size_t(tokens - 1) * state_bytes;
        if (!scan || tokens < 2 || !valid_span(prefix, checkpoint_bytes) ||
            overlap(prefix, checkpoint_bytes, state, state_bytes) ||
            overlap(prefix, checkpoint_bytes, output, output_bytes))
            throw std::invalid_argument("GDN checkpoint requires disjoint multi-token scan");
        for (int i = 0; i < 5; ++i)
            if (overlap(prefix, checkpoint_bytes, inputs[i], bytes[i]))
                throw std::invalid_argument("GDN checkpoint aliases input");
    }
}

static void launch_gdn(bool scan, int tokens, int input_stride, float* state, const float* q, const float* k, const float* v,
                     const float* gate, const float* beta, float* output,
                     const GdnShapes& shape, void* stream, float* prefix = nullptr, int columns = 1, bool input_alpha = false) {
    validate_gdn(scan,tokens,input_stride,state,q,k,v,gate,beta,output,shape,stream,prefix,columns);
    const float scale = 1.0f / sqrtf(float(S));
    const dim3 grid(unsigned(shape.h_v), 1, S / (4 * columns)), threads(32, 4);
    if(input_alpha) {
        if(!scan || prefix || columns!=8)throw std::invalid_argument("GDN alpha scan requires ordinary Columns8");
        step<true,false,8,true><<<grid,threads,0,static_cast<cudaStream_t>(stream)>>>(
            state,q,k,v,gate,beta,output,int(shape.h_k),int(shape.h_v),scale,tokens,input_stride,nullptr);
    } else if (prefix)
        step<true, true><<<grid, threads, 0, static_cast<cudaStream_t>(stream)>>>(
            state, q, k, v, gate, beta, output, int(shape.h_k), int(shape.h_v), scale, tokens, input_stride, prefix);
    else if (scan && columns == 8)
        step<true, false, 8><<<grid, threads, 0, static_cast<cudaStream_t>(stream)>>>(
            state, q, k, v, gate, beta, output, int(shape.h_k), int(shape.h_v), scale, tokens, input_stride, nullptr);
    else if (scan && columns == 4)
        step<true, false, 4><<<grid, threads, 0, static_cast<cudaStream_t>(stream)>>>(
            state, q, k, v, gate, beta, output, int(shape.h_k), int(shape.h_v), scale, tokens, input_stride, nullptr);
    else if (scan && columns == 2)
        step<true, false, 2><<<grid, threads, 0, static_cast<cudaStream_t>(stream)>>>(
            state, q, k, v, gate, beta, output, int(shape.h_k), int(shape.h_v), scale, tokens, input_stride, nullptr);
    else if (scan)
        step<true><<<grid, threads, 0, static_cast<cudaStream_t>(stream)>>>(
            state, q, k, v, gate, beta, output, int(shape.h_k), int(shape.h_v), scale, tokens, input_stride, nullptr);
    else
        step<false><<<grid, threads, 0, static_cast<cudaStream_t>(stream)>>>(
            state, q, k, v, gate, beta, output, int(shape.h_k), int(shape.h_v), scale, 1, input_stride, nullptr);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void native_gdn_scan(float* state, const float* q, const float* k, const float* v,
                     const float* gate, const float* beta, float* output,
                     const GdnShapes& shape, int tokens, int input_stride, void* stream, float* prefix, int columns) {
    launch_gdn(true, tokens, input_stride, state, q, k, v, gate, beta, output, shape, stream, prefix, columns);
}
void native_gdn_scan_alpha(float* state,const float* q,const float* k,const float* v,
    const float* alpha,const float* beta,float* output,const GdnShapes& shape,int tokens,int stride,void* stream) {
    launch_gdn(true,tokens,stride,state,q,k,v,alpha,beta,output,shape,stream,nullptr,8,true);
}
bool native_gdn_pair2_enabled() {
    static const bool enabled=[] {
        const char* value=std::getenv("STRATA_QWEN35_GDN_PAIR2");
        if(!value || !std::strcmp(value,"0")) return false;
        if(!std::strcmp(value,"1")) return true;
        throw std::invalid_argument("GDN_PAIR2 requires0 or1");
    }();
    return enabled;
}
size_t native_gdn_pair2_workspace(const GdnShapes& shape,int tokens) {
    if(tokens<1 || tokens>8192 || shape.S!=128 || shape.h_k<1 || shape.h_v<1 ||
       shape.h_v>65535 || shape.h_v%shape.h_k)
        throw std::invalid_argument("unsupported GDN pair2 workspace geometry");
    return size_t((tokens+1)/2)*size_t(shape.h_v)*sizeof(GdnPair2);
}
void native_gdn_scan_pair2(float* state,const float* q,const float* k,const float* v,
    const float* gate,const float* beta,float* output,const GdnShapes& shape,int tokens,
    int stride,void* scratch,size_t scratch_bytes,void* stream) {
    validate_gdn(true,tokens,stride,state,q,k,v,gate,beta,output,shape,stream,nullptr,8);
    const size_t needed=native_gdn_pair2_workspace(shape,tokens);
    if(!valid_span(scratch,needed) || uintptr_t(scratch)%16 || scratch_bytes<needed)
        throw std::invalid_argument("GDN pair2 requires aligned metadata workspace");
    const size_t state_bytes=size_t(128)*128*size_t(shape.h_v)*4;
    const size_t qk_bytes=(size_t(tokens-1)*stride+size_t(128)*shape.h_k)*4;
    const size_t v_bytes=(size_t(tokens-1)*stride+size_t(128)*shape.h_v)*4;
    const size_t head_bytes=size_t(tokens)*size_t(shape.h_v)*4;
    const size_t out_bytes=size_t(tokens)*128*size_t(shape.h_v)*4;
    const void* pointers[]={state,q,k,v,gate,beta,output};
    const size_t bytes[]={state_bytes,qk_bytes,qk_bytes,v_bytes,head_bytes,head_bytes,out_bytes};
    for(int i=0;i<7;++i) if(overlap(scratch,needed,pointers[i],bytes[i]))
        throw std::invalid_argument("GDN pair2 metadata aliases live tensors");
    auto s=static_cast<cudaStream_t>(stream);auto* metadata=static_cast<GdnPair2*>(scratch);
    const int work=((tokens+1)/2)*int(shape.h_v);
    pair2_metadata<<<(work+3)/4,128,0,s>>>(q,k,gate,beta,metadata,int(shape.h_k),int(shape.h_v),tokens,stride);
    auto error=cudaGetLastError();
    if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    pair2_scan<<<dim3(unsigned(shape.h_v),1,4),dim3(32,4),0,s>>>(
        state,q,k,v,metadata,output,int(shape.h_k),int(shape.h_v),tokens,stride);
    error=cudaGetLastError();
    if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
bool native_gdn_chunk4_enabled() {
    static const bool enabled=[] {
        const char* value=std::getenv("STRATA_QWEN35_GDN_CHUNK4");
        if(!value || !std::strcmp(value,"0")) return false;
        if(!std::strcmp(value,"1")) return true;
        throw std::invalid_argument("GDN_CHUNK4 requires0 or1");
    }();
    return enabled;
}
size_t native_gdn_chunk4_workspace(const GdnShapes& shape,int tokens) {
    if(tokens<1 || tokens>8192 || shape.S!=128 || shape.h_k<1 || shape.h_v<1 ||
       shape.h_v>65535 || shape.h_v%shape.h_k)
        throw std::invalid_argument("unsupported GDN chunk4 workspace geometry");
    return size_t((tokens+3)/4)*size_t(shape.h_v)*sizeof(GdnChunk4);
}
void native_gdn_scan_chunk4(float* state,const float* q,const float* k,const float* v,
    const float* gate,const float* beta,float* output,const GdnShapes& shape,int tokens,
    int stride,void* scratch,size_t scratch_bytes,void* stream) {
    validate_gdn(true,tokens,stride,state,q,k,v,gate,beta,output,shape,stream,nullptr,8);
    const size_t needed=native_gdn_chunk4_workspace(shape,tokens);
    if(!valid_span(scratch,needed) || uintptr_t(scratch)%16 || scratch_bytes<needed)
        throw std::invalid_argument("GDN chunk4 requires aligned metadata workspace");
    const size_t state_bytes=size_t(128)*128*size_t(shape.h_v)*4;
    const size_t qk_bytes=(size_t(tokens-1)*stride+size_t(128)*shape.h_k)*4;
    const size_t v_bytes=(size_t(tokens-1)*stride+size_t(128)*shape.h_v)*4;
    const size_t head_bytes=size_t(tokens)*size_t(shape.h_v)*4;
    const size_t out_bytes=size_t(tokens)*128*size_t(shape.h_v)*4;
    const void* pointers[]={state,q,k,v,gate,beta,output};
    const size_t bytes[]={state_bytes,qk_bytes,qk_bytes,v_bytes,head_bytes,head_bytes,out_bytes};
    for(int i=0;i<7;++i) if(overlap(scratch,needed,pointers[i],bytes[i]))
        throw std::invalid_argument("GDN chunk4 metadata aliases live tensors");
    auto s=static_cast<cudaStream_t>(stream);auto* metadata=static_cast<GdnChunk4*>(scratch);
    const int work=((tokens+3)/4)*int(shape.h_v);
    chunk4_metadata<<<(work+3)/4,128,0,s>>>(q,k,gate,beta,metadata,int(shape.h_k),int(shape.h_v),tokens,stride);
    auto error=cudaGetLastError();
    if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    chunk4_scan<<<dim3(unsigned(shape.h_v),1,8),dim3(32,4),0,s>>>(
        state,q,k,v,metadata,output,int(shape.h_k),int(shape.h_v),tokens,stride);
    error=cudaGetLastError();
    if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
} // namespace strata::kernels
