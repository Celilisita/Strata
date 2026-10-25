#pragma once
#include "pp_stream_graph.hpp"
#include "ggml-cuda/common.cuh"
#include "pp_stream_metrics.hpp"
#include "pp_plan_receipt.hpp"
#include "pp_transport.hpp"
#include <cstdint>

namespace strata::deepseek_v4 {
void pp_stream_run(ggml_backend_cuda_context&, ggml_tensor*);
void pp_stream_release(ggml_backend_cuda_context*);
PpStreamMetrics pp_stream_metrics(ggml_backend_cuda_context*);
bool pp_stream_get_metrics(ggml_backend_t, PpStreamMetrics*);
bool pp_stream_enter_tg(ggml_backend_t);
bool pp_plan_receipt_begin(ggml_backend_t,uint64_t,bool);
bool pp_plan_receipt_read(ggml_backend_t,const PpPlanReceiptPod**);
bool pp_stream_bind_transport(ggml_backend_t, PpTransportQuery, void*);
bool pp_bind_packed(ggml_backend_t, PackedExpertQuery, void*);
void pp_prefetch_node(ggml_backend_cuda_context&,const ggml_tensor*);
void pp_stream_product(ggml_backend_cuda_context&, const void* w, const void* xq,
                       const int32_t* bounds, const int32_t* identity, float* dst,
                       int experts, int rows, int quant_rows, int max_rows, int cols, int out_cols,
                       size_t expert_bytes, bool fp4, cudaStream_t, int destination_rows = 0,
                       const int32_t* ragged_prefix = nullptr, const int32_t* host_bounds = nullptr,
                       ggml_type type = GGML_TYPE_MXFP4, const int32_t* host_tile_counts = nullptr);
void pp_stream_swiglu(const float* gu, float* h, int rows, float clamp, cudaStream_t);
void pp_stream_scatter(const float* down, const int32_t* pairs, const float* weights,
                       float* dst, int rows, cudaStream_t, int topk = 6);
void pp_stream_iota(int32_t*, int n, cudaStream_t);
}
