#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace strata::qwen35::native {
// Lossless execution view of GGUF NVFP4 (type40). Values are adjacent E2M1
// nibbles; E4M3 block scales use the documented cuBLAS 128x4 tiled order.
// Global factors are separate from block scales and never baked into weights.
struct Nvfp4View {
    const void *values=nullptr;
    const void *scales=nullptr;
    int rows=0,columns=0;
    float weight_scale=1, input_scale=1;
};
inline size_t nvfp4_scale_bytes(int rows,int columns){
    return size_t((rows+127)/128)*128*size_t((columns+63)/64)*4;
}
inline size_t nvfp4_value_bytes(int rows,int columns){return size_t(rows)*columns/2;}
inline size_t nvfp4_gguf_bytes(int rows,int columns){return size_t(rows)*size_t(columns/64)*36;}
// Borrowed output storage, zero-initialized once by its owner. Transient input
// quantization may reuse it only on one ordered stream, after prior GEMM reads.
struct Nvfp4Input {void *values=nullptr,*scales=nullptr;int capacity_rows=0,capacity_columns=0;};
void nvfp4_repack(const void *gguf,void *values,void *scales,int rows,int columns,
                  unsigned *invalid,cudaStream_t);
void nvfp4_quantize(const float *,Nvfp4Input,int tokens,int columns,float global,cudaStream_t);
// Paired projection output is [T,2R], with Gate then Up within each token row.
void nvfp4_swiglu_quantize(const float *pair,Nvfp4Input,int tokens,int features,float global,cudaStream_t);
void nvfp4_decode(Nvfp4View,float *,cudaStream_t);
void nvfp4_decode_input(Nvfp4Input,float *,int tokens,int columns,float global,cudaStream_t);

// One execution owner, stable workspace, cached shape plans, borrowed operands.
// This provider contains no model scheduler, KV policy, or speculative state.
class Nvfp4Gemm {
public:
    Nvfp4Gemm(cudaStream_t,size_t workspace_bytes=32u*1024*1024,int max_rows=34816);
    ~Nvfp4Gemm();
    Nvfp4Gemm(const Nvfp4Gemm &)=delete;
    Nvfp4Gemm &operator=(const Nvfp4Gemm &)=delete;
    void prepare(Nvfp4View,Nvfp4Input,int tokens);
    void project(Nvfp4View,Nvfp4Input,float *,int tokens,cudaStream_t);
    // Measured whole-8192 FFN Down only: independent FP32 C and final D.
    void project_add(Nvfp4View,Nvfp4Input,const float *residual,float *,int tokens,cudaStream_t);
    size_t bytes()const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    void project_impl(Nvfp4View,Nvfp4Input,const float *residual,float *,int tokens,cudaStream_t);
};
}
