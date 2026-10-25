#pragma once
#include <cuda_runtime.h>

namespace strata::completed_v100::qwen35::native {
struct ReductionSchedule { int partitions = 1; int granularity = 64; };
// FP16 operands, FP32 partials/output. Contiguous bounded K slices, then
// ascending FP32 reduction. Caller owns enough partial storage for all slices.
void dense_partitioned(const void *, const void *, float *, float *,
                       int rows, int columns, int tokens, int partitions, int granularity, cudaStream_t);
// Wider token tile; same FP16 operands, per-output K order and FP32 reduction.
void dense_wide_pair(const void *,const void *,float *,float *,int,int,int,cudaStream_t);
void dense_wide_linear(const void *,const void *,float *,int,int,int,cudaStream_t);
void dense_wide_split(const void *,const void *,float *,float *,int,int,int,int,cudaStream_t);
void configure_dense_wide();
void dense_balanced_pair(const void *,const void *,float *,float *,int,int,int,cudaStream_t);
void dense_balanced_linear(const void *,const void *,float *,int,int,int,cudaStream_t);
void dense_balanced_split(const void *,const void *,float *,float *,int,int,int,int,cudaStream_t);
void configure_dense_balanced();
void dense_raster_pair(const void *,const void *,float *,float *,int,int,int,cudaStream_t);
void dense_raster_linear(const void *,const void *,float *,int,int,int,cudaStream_t);
void dense_raster_split(const void *,const void *,float *,float *,int,int,int,int,cudaStream_t);
void configure_dense_raster();
void dense_joint_pair(const void *,const void *,float *,float *,int,int,int,cudaStream_t);
void dense_joint_linear(const void *,const void *,float *,int,int,int,cudaStream_t);
void dense_joint_split(const void *,const void *,float *,float *,int,int,int,int,cudaStream_t);
void configure_dense_joint();
void dense_bank_pair(const void *,const void *,float *,float *,int,int,int,cudaStream_t);
void dense_bank_linear(const void *,const void *,float *,int,int,int,cudaStream_t);
void dense_bank_split(const void *,const void *,float *,float *,int,int,int,int,cudaStream_t);
void configure_dense_bank();
void dense_folded_pair(const void *,const void *,float *,float *,float *,int,int,int,bool,cudaStream_t);
void configure_dense_folded();
void dense_folded_compact_pair(const void *,const void *,float *,float *,float *,int,int,int,bool,cudaStream_t);
void configure_dense_folded_compact();
void dense_folded_gray_pair(const void *,const void *,float *,float *,float *,int,int,int,bool,cudaStream_t);
void configure_dense_folded_gray();
void dense_folded_widewarp_pair(const void *,const void *,float *,float *,float *,int,int,int,bool,cudaStream_t);
void configure_dense_folded_widewarp();
void dense_folded_ring_pair(const void *,const void *,float *,float *,float *,int,int,int,bool,cudaStream_t);
void configure_dense_folded_ring();
void configure_dense_operand_lookahead();
void dense_operand_lookahead_pair(const void *,const void *,float *,float *,int,int,int,cudaStream_t);
void configure_dense_warp64();
void dense_warp64_pair(const void *,const void *,float *,float *,int,int,int,cudaStream_t);
void configure_dense_token256();
void dense_token256_pair(const void *,const void *,float *,float *,int,int,int,cudaStream_t);
void dense_full_k_pair(const void *,const void *,float *,float *,int,int,int,cudaStream_t);
void dense_full_k_linear(const void *,const void *,float *,int,int,int,cudaStream_t);
void dense_full_k_split(const void *,const void *,float *,float *,int,int,int,int,cudaStream_t);
void configure_dense_full_k();
void dense_shared_pair(const void *,const void *,float *,float *,int,int,int,cudaStream_t);
void dense_shared_linear(const void *,const void *,float *,int,int,int,cudaStream_t);
void dense_shared_split(const void *,const void *,float *,float *,int,int,int,int,cudaStream_t);
void configure_dense_shared();
void dense_pipeline_pair(const void *,const void *,float *,float *,int,int,int,cudaStream_t);
void dense_pipeline_linear(const void *,const void *,float *,int,int,int,cudaStream_t);
void dense_pipeline_split(const void *,const void *,float *,float *,int,int,int,int,cudaStream_t);
void configure_dense_pipeline();
}
