#pragma once
#include "strata/qwen35/native/model.hpp"
#include "strata/qwen35/native/dense_gemm.hpp"
#include "strata/qwen35/native/gdn_vectorized.hpp"
#include "strata/qwen35/native/attention_gather.hpp"
#include "strata/qwen35/native/attention_keys64.hpp"
#include "strata/qwen35/native/attention_verify_grouped.hpp"
#include "strata/qwen35/native/attention_verify_padded.hpp"
#include "ffn_shape_policy.hpp"
#include <cstdio>

namespace strata::qwen35::native {
inline DenseLimits service_limits(int context, bool ram) {
    DenseLimits limits;
    limits.context = context;
    int device = 0;
    check_cuda(cudaGetDevice(&device), "Dense profile device");
    const auto caps = capabilities(device);
    const bool volta = caps.major == 7 && caps.minor == 0;
    if (!volta && caps.major < 8)
        throw std::invalid_argument("Dense service requires SM70 or the SM80+ staged provider");
    // Volta's prepared PP32, joint and folded layouts keep their original admission.
    // Other devices use the existing raw staged inventory with native tiny/vector
    // operations and the architecture-selected cuBLAS/CUTLASS long-PP service.
    limits.native32 = volta; limits.joint_ffn = volta;
    limits.joint_projections = volta; limits.folded_ffn = volta;
    std::fprintf(stderr, "strata_dense_profile sm=%d.%d native32=%d long_pp=%s\n",
                 caps.major, caps.minor, int(volta), volta ? "volta-folded" : "staged-cublas-cutlass");
    limits.kv_q8 = true; limits.kv_initial_tokens = (std::min)(8192, context);
    if (ram) {
        limits.kv_ram_tokens = 32768;
        limits.kv_select_tokens = 8192;
        limits.kv_keep_queries = 3;
    }
    return limits;
}
inline void service_verify(const float* q, const void* k, const void* v, float* out,
                           void* scratch, size_t bytes, PrefillAttentionShape s,
                           int partitions, cudaStream_t stream) {
    const auto kernel = s.tokens == 2 || s.capacity >= 16384
        ? verify_attention_grouped_q8 : verify_attention_padded_q8;
    kernel(q, k, v, out, scratch, bytes, s, partitions, stream);
}
inline void apply_service_profile(DenseModel& model, DenseGemm& shape, int prompt, int proposals) {
    model.attention_work().use_verify_split(true);
    model.attention_work().use_verify_grid(prompt >= 8192);
    model.use_batch_head(true); model.enable_tiny_batches();
    if (proposals == 1) model.enable_tiny_row_ffn(); else model.enable_tiny_meta_ffn();
    model.use_pp_gemm(&shape); model.use_pp_gdn_rows(4);
    model.use_pp_gdn_kernel(gdn_recurrence_vectorized, gdn_vectorized_workspace_bytes);
    configure_prefill_gather(48);
    if (prompt >= 1280) configure_prefill_keys64(48);
    model.attention_work().use_pp_attention_kernel(prompt >= 1280 ? prefill_attention_keys64 : prefill_attention_gather, 48);
    model.attention_work().use_verify_attention_kernel(service_verify);
}
} // namespace strata::qwen35::native
