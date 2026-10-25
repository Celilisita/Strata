#pragma once
#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include "strata/qwen35/completed_v100/native/quant.hpp"

namespace strata::completed_v100::qwen35::native {
struct GemmBatch {
    int m,n,k,lda,ldb,ldc,batches;
    long long stride_a,stride_b,stride_c;
    bool transpose_a=false,transpose_b=false;
    float alpha=1.0f,beta=0.0f;
};
struct GemmWorkspace {void *data=nullptr;size_t bytes=0;cudaStream_t stream=nullptr;};
// Borrowed service: one execution owner and one ordered stream. The provider
// outlives every FFN/projection invocation and its queued CUDA work.
class DenseGemm {
public:
    virtual ~DenseGemm() = default;
    // Row-major weights[R,K], input[T,K], output[T,R]; half/half -> float.
    virtual void project(const void *weights,const void *input,float *output,
                         int rows,int columns,int tokens,cudaStream_t)=0;
    virtual size_t workspace_bytes() const=0;
    virtual unsigned long long calls() const=0;
    // Optional adapter lease: same stream, owner outlives adapter and queued work.
    virtual GemmWorkspace borrowed_workspace()const{return {};}
    // Optional concatenated Gate/Up output [T,2R]; same decoded weight inventory.
    virtual bool project_pair(const void *,const void *,float *,int,int,int,cudaStream_t){return false;}
    // Optional interleaved [G0,U0,G1,U1,...] weights and fused half output.
    // Raw Gate/Up pointers are a cold diagnostic lease, null in normal work.
    virtual bool can_fuse_swiglu(int,int,int)const{return false;}
    virtual void project_swiglu(const void *,const void *,void *,float *,float *,float *,
                               int,int,int,cudaStream_t){throw std::invalid_argument("provider has no fused SwiGLU");}
    virtual bool can_fuse_quant_swiglu(Matrix,Matrix,int)const{return false;}
    virtual void project_quant_swiglu(Matrix,Matrix,const void *,void *,void *,float *,float *,float *,
                                     int,cudaStream_t){throw std::invalid_argument("provider has no direct quantized SwiGLU");}
    // Idle-owner experimental selector; previously queued work is already joined.
    virtual void select_pair(bool enabled){if(enabled)throw std::invalid_argument("provider has no paired mode");}
    virtual void batched_fp16(const void *,const void *,float *,GemmBatch,cudaStream_t)=0;
};
std::unique_ptr<DenseGemm> make_cublas_gemm(cudaStream_t,size_t workspace_bytes=32u*1024*1024);
std::unique_ptr<DenseGemm> make_cublas_pair_gemm(cudaStream_t,size_t workspace_bytes=32u*1024*1024);
std::unique_ptr<DenseGemm> make_cutlass_swiglu_gemm(DenseGemm &,cudaStream_t,int tile=0);
// Idle benchmark selection; both paths keep the same paired cuBLAS fallback.
void select_cutlass_swiglu(DenseGemm &,bool enabled);
struct CutlassTileInfo {int tile=0,threads=0,registers=0;size_t shared_bytes=0;};
// Cold benchmark metadata; no production logging or hot-path sampling.
CutlassTileInfo cutlass_swiglu_info(DenseGemm &);
std::unique_ptr<DenseGemm> make_cutlass_quant_gemm(DenseGemm &,cudaStream_t,int staging=0);
CutlassTileInfo cutlass_quant_info(DenseGemm &);
void swiglu_half(const float *,const float *,void *,size_t,cudaStream_t);
void swiglu_pair_half(const float *,void *,int,int,cudaStream_t);
void split_swiglu_pair(const float *,float *,float *,float *,void *,int,int,cudaStream_t);
}
