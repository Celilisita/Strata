#pragma once
#include "quant.hpp"
#include <cuda_runtime_api.h>

namespace strata::completed_v100::qwen35::native {
// F32 and rounded half metadata share integer codes; transient, token-major.
constexpr size_t SmallPrefillActivationBytes = 40;
void small_prefill_quantize(const float *input, void *output, int columns, int tokens,
                            cudaStream_t stream);
void small_prefill_linear(Matrix weight, const void *input, float *output, int tokens,
                          cudaStream_t stream, int processors);
// Setup-only resource policy; no capability query in projection calls.
void configure_small_prefill_tensor();
void small_prefill_tensor_linear(Matrix weight, const void *input, float *output, int tokens,
                                 cudaStream_t stream, int processors);
void configure_small_prefill_direct();
void small_prefill_direct_linear(Matrix weight, const void *input, float *output, int tokens,
                                 cudaStream_t stream, int processors);
void small_prefill_direct_blocked_linear(Matrix weight, const void *input, float *output, int tokens,
                                         cudaStream_t stream, int processors);
// Lossless setup-prepared Q6 byte-code plane; rejected by raw providers.
void small_prefill_direct_codes_linear(Matrix weight, const void *input, float *output, int tokens,
                                      cudaStream_t stream, int processors);
// SM70 diagnostic PP32 SplitK3 with prepared half input and hot reduction.
size_t small_prefill_direct_codes_split_bytes(Matrix prepared, int tokens);
void small_prefill_direct_codes_split_linear(Matrix prepared, const void *input, float *output, int tokens,
                                             void *scratch, size_t scratch_bytes, cudaStream_t stream, int processors);
// Isolated SM70 scaled-half plus compressed exact integer residual experiment.
size_t prepared_q6_scaled_bytes(Matrix raw);
Matrix prepare_q6_scaled(Matrix raw, void *output, cudaStream_t stream);
void configure_small_prefill_scaled();
void small_prefill_scaled_linear(Matrix prepared, const void *input, float *output, int tokens,
                                 cudaStream_t stream, int processors);
size_t small_prefill_half_input_bytes(int columns, int tokens);
void small_prefill_half_quantize(const float *input, void *output, int columns, int tokens,
                                 cudaStream_t stream);
void small_prefill_scaled_half_linear(Matrix prepared, const void *input, float *output, int tokens,
                                      cudaStream_t stream, int processors);
void small_prefill_vector_linear(Matrix weight, const void *input, float *output, int tokens,
                                 cudaStream_t stream, int processors);
// Blocked provider input storage includes token padding to32.
size_t small_prefill_vector_bytes(int columns, int tokens);
void small_prefill_vector_quantize(const float *input, void *output, int columns, int tokens,
                                   cudaStream_t stream);
}
