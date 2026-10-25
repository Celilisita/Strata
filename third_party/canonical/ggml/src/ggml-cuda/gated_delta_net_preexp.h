#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include <cstddef>
#include <cstdint>

// Private CUDA GDN contract. No model/env policy or allocation belongs here.
// Param 0 remains K. Only param 1 == 1 requests the optional pre-exp path.
constexpr int ggml_cuda_gdn_preexp_param = 1;
constexpr int32_t ggml_cuda_gdn_preexp_request = 1;
constexpr int64_t ggml_cuda_gdn_preexp_max_tokens = 8192;

inline void ggml_cuda_gdn_set_preexp_requested(ggml_tensor * op, bool requested) {
    if (op != nullptr && op->op == GGML_OP_GATED_DELTA_NET) {
        op->op_params[ggml_cuda_gdn_preexp_param] = requested ? ggml_cuda_gdn_preexp_request : 0;
    }
}

inline bool ggml_cuda_gdn_preexp_requested(const ggml_tensor * op) {
    return op != nullptr && op->op == GGML_OP_GATED_DELTA_NET &&
           op->op_params[ggml_cuda_gdn_preexp_param] == ggml_cuda_gdn_preexp_request;
}

inline bool ggml_cuda_gdn_preexp_shape(const ggml_tensor * tensor,
        int64_t n0, int64_t n1, int64_t n2, int64_t n3) {
    return tensor != nullptr && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor) &&
           tensor->ne[0] == n0 && tensor->ne[1] == n1 && tensor->ne[2] == n2 && tensor->ne[3] == n3;
}

// Geometry eligibility is independent of the request bit, for builder/fixture use.
// Unknown strides/shapes use the unchanged original operator, including non-KDA.
inline bool ggml_cuda_gdn_preexp_eligible(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_GATED_DELTA_NET || op->type != GGML_TYPE_F32 ||
        op->op_params[0] != 3 || op->src[0] == nullptr || op->src[2] == nullptr) {
        return false;
    }
    const int64_t tokens = op->src[2]->ne[2];
    const int64_t qheads = op->src[0]->ne[1];
    if (tokens < 1 || tokens > ggml_cuda_gdn_preexp_max_tokens ||
        qheads < 1 || qheads > 64 || 64 % qheads != 0) {
        return false;
    }
    return ggml_cuda_gdn_preexp_shape(op->src[0], 128, qheads, tokens, 1) &&
           ggml_cuda_gdn_preexp_shape(op->src[1], 128, qheads, tokens, 1) &&
           ggml_cuda_gdn_preexp_shape(op->src[2], 128, 64, tokens, 1) &&
           ggml_cuda_gdn_preexp_shape(op->src[3], 128, 64, tokens, 1) &&
           ggml_cuda_gdn_preexp_shape(op->src[4], 1, 64, tokens, 1) &&
           ggml_cuda_gdn_preexp_shape(op->src[5], 128, 128, 64, 1) &&
           ggml_cuda_gdn_preexp_shape(op, 128 * 64, tokens + 3 * 128, 1, 1);
}

// Logical scratch requested from the existing context pool (its retained capacity
// and allocator rounding may be larger). At T=8192: 256 MiB; at T=2: 64 KiB.
inline size_t ggml_cuda_gdn_preexp_scratch_bytes(const ggml_tensor * op) {
    return ggml_cuda_gdn_preexp_eligible(op) ?
        static_cast<size_t>(op->src[2]->ne[2]) * 128 * 64 * sizeof(float) : 0;
}

// Host eager/capture dispatch receipts only: graph replay does not call the
// operator implementation. These are not token/kernel counts or GPU timings.
constexpr size_t ggml_cuda_gdn_preexp_receipt_domains = 5;
enum ggml_cuda_gdn_preexp_receipt_domain : size_t {
    GGML_CUDA_GDN_PREEXP_T1 = 0,
    GGML_CUDA_GDN_PREEXP_T2 = 1,
    GGML_CUDA_GDN_PREEXP_T128 = 2,
    GGML_CUDA_GDN_PREEXP_T8192 = 3,
    GGML_CUDA_GDN_PREEXP_OTHER = 4,
};
struct ggml_cuda_gdn_preexp_domain_receipt {
    uint64_t requested;
    uint64_t selected;
    uint64_t fallback;
};
struct ggml_cuda_gdn_preexp_receipt {
    ggml_cuda_gdn_preexp_domain_receipt domains[ggml_cuda_gdn_preexp_receipt_domains];
};
// Process starts at zero; no reset API. Read at operator/request quiescence.
// Individual relaxed loads are safe, but a concurrent snapshot is not a
// transaction over requested/selected/fallback. No CUDA APIs or synchronization.
GGML_BACKEND_API void ggml_cuda_gdn_preexp_get_receipt(ggml_cuda_gdn_preexp_receipt * receipt);
