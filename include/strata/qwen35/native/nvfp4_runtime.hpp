#pragma once
#include "strata/qwen35/native/nvfp4.hpp"
#include "strata/qwen35/native/dense_gemm.hpp"
#include "strata/qwen35/native/ffn.hpp"
#include <vector>
#include <initializer_list>

namespace strata {class GgufFile;struct TensorInfo;}
namespace strata::qwen35::native {
class DeviceStorage;
// One losslessly repacked resident inventory, borrowed by PP and decode.
Matrix load_nvfp4_matrix(const strata::GgufFile &,const strata::TensorInfo &,
    DeviceStorage &,Nvfp4View &,cudaStream_t);
std::unique_ptr<DenseGemm> make_nvfp4_runtime(DenseGemm &,const std::vector<Nvfp4View> &,
    int max_tokens,int max_columns,cudaStream_t);
std::unique_ptr<DenseGemm> make_nvfp4_runtime(DenseGemm &,const std::vector<Nvfp4View> &,
    int max_tokens,int max_columns,cudaStream_t,Matrix tiny_only_head);
struct TinyHeadCounters {uint64_t scalar_submissions=0,batch_submissions=0,fallback_rows=0;size_t weight_bytes=0;};
// Normalized input; only the explicitly registered shared head is admitted.
void nvfp4_project_head(DenseGemm &,Matrix,const float *,float *,int,cudaStream_t);
// PREPARE only, after releasing graphs. Off means row-wise T1 on the same head.
void nvfp4_configure_head_request(DenseGemm &,bool batch_T3);
TinyHeadCounters nvfp4_head_counters(DenseGemm &);
struct Nvfp4Projection {Matrix weight;float* output;cudaEvent_t completion=nullptr;};
bool nvfp4_shared_projections_enabled();
// Input encoding is shared only within this ordered call; no persistent cache.
void nvfp4_project_set(DenseGemm &,const float *,std::initializer_list<Nvfp4Projection>,int,cudaStream_t);
void nvfp4_project(DenseGemm &,Matrix,const float *,float *,int,cudaStream_t);
struct DownAddCounters {uint64_t fused=0,fallback=0;};
// PREPARE-only request choice; clears eager/capture submission counters.
void nvfp4_configure_down_add_request(DenseGemm &,bool enabled);
DownAddCounters nvfp4_down_add_counters(DenseGemm &);
void nvfp4_ffn(DenseGemm &,Matrix,Matrix,Matrix,const float *,const Buffers &,int,cudaStream_t);
// Returns whether FP32 intermediate diagnostics were materialized.
bool nvfp4_ffn_fused(DenseGemm &,Matrix,Matrix,Matrix,const float *,const Buffers &,int,cudaStream_t,bool diagnostic);
}
