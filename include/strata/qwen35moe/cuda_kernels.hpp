#pragma once

#include <cstdint>
#include "strata/core/device_miss.hpp"

namespace strata::qwen35moe {

// One-token 256-expert softmax, top-8 selection, then selected-weight
// renormalization. The caller owns CUDA memory and the ordered stream.
void route_top8_continue(const float* logits, int32_t* ids, float* weights,
                          const core::NativeMissArgs* args, void* stream, int tokens = 1);
void moe_combine_continue(float* parts, const float* weights, const float* shared,
                           const float* residual, float* output,
                           const core::NativeMissArgs* args, void* stream, int tokens = 1);

void route_top8_batch(const float* logits, int32_t* ids, float* weights, int tokens, void* stream);
void route_top8(const float * logits, int32_t * ids, float * weights, void * stream);

// F32 router rows, including the shared gate row, against one F32 token.
void route_project(const float * rows, const float * x, float * logits,
                   int width, int outputs, void * stream, int tokens = 1);

// Build the grouped one-token expert tables from a fully resident cache
// directory. The directory and expert storage must remain stable until the
// ordered stream finishes this layer.
void gather_resident_experts(const int32_t * ids, const int32_t * slots,
                             const void * cache, uint64_t stride, uint64_t gu_bytes,
                             unsigned long long * gu_addresses, int32_t * gu_rows,
                             unsigned long long * down_addresses, int32_t * down_rows,
                             void * stream, const uint64_t* slot_addresses = nullptr);

// One-token GDN output: RMSNorm(output, gamma) * SiLU(z), 128 channels per
// value head.
void gdn_post_silu(const float * output, const float * z, const float * gamma,
                   float * destination, int heads, float epsilon, void * stream);

// Qwen3.5 uses ggml_l2_norm: x / sqrt(max(sum(x*x), epsilon*epsilon)).
void gdn_l2_norm(float * values, int rows, int width, float epsilon, void * stream,
                  int tokens = 1, int stride = 0, bool canonical_warp = false);

// SiLU(gate) * up before the canonical quantized Down projection.
void swiglu(const float * gate, const float * up, float * hidden, int width, void * stream,
              const unsigned long long* row_addresses = nullptr, int row_width = 0);

// Apply sigmoid(shared gate logit) to the shared expert's output.
void gate_shared(float * values, const float * logit, int width, void * stream);

// Qwen3.5 full-attention decode: Q16x256, KV2x256, one query, scale 1/16.
// Q/output are F32 [16,256]; K/V are F16 [capacity,2,256]. The ordered
// device step uses the Strata QSA step layout. This short path supports
// context lengths 1..1024 and does not allocate or synchronize.
void flash_attn_short_step(const float * q, const uint16_t * k, const uint16_t * v,
                           const int32_t * step, int64_t capacity, int max_context, int width,
                           float * output, int32_t * status, const uint16_t * mask,
                           void * stream);

// Reconstruct GGML Q8_1 activation blocks as F16 for Tensor Core projection.
void q8_1_to_f16(const void * blocks, uint16_t * output, int elements, void * stream,
                   uint16_t * residual = nullptr);

void gate_shared_batch(float * shared, const float * logits, int count, void * stream);
void moe_combine_batch(const float * parts, const float * weights, const float * shared,
                       const float * residual, float * output, int count, void * stream);

} // namespace strata::qwen35moe
