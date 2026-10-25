#include "sm120_cached_global.hpp"
#include "pp_tc_ops.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace strata::gemma4::native {
namespace {
std::size_t aligned(std::size_t n) { return (n + 255) & ~std::size_t(255); }
void valid_maximum(int maximum) {
    if (maximum < 4 || maximum > 2048)
        throw std::invalid_argument("Gemma cached global PP batch4..2048");
}
void blas(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("Gemma cached global GemmBatchedEx");
}
struct Span { const void* data; std::size_t bytes; };
void disjoint(const Span* all, int count) {
    for (int i = 0; i < count; ++i) {
        const auto x = reinterpret_cast<std::uintptr_t>(all[i].data);
        if (!x || !all[i].bytes || all[i].bytes > std::numeric_limits<std::uintptr_t>::max() - x)
            throw std::invalid_argument("Gemma cached global live span");
        for (int j = 0; j < i; ++j) {
            const auto y = reinterpret_cast<std::uintptr_t>(all[j].data);
            if (x < y + all[j].bytes && y < x + all[i].bytes)
                throw std::invalid_argument("Gemma cached global input/output/arena alias");
        }
    }
}
}
bool cached_global_requested() {
    static const bool enabled = [] {
        const char* raw = std::getenv("STRATA_GEMMA_PP_CACHED_GLOBAL");
        if (!raw || std::strcmp(raw, "0") == 0) return false;
        if (std::strcmp(raw, "1") != 0)
            throw std::invalid_argument("STRATA_GEMMA_PP_CACHED_GLOBAL requires0or1");
        const char* segmented = std::getenv("STRATA_GEMMA_PP_SEGMENTED_FLASH");
        if (segmented && std::strcmp(segmented, "0") != 0)
            throw std::invalid_argument("Gemma cached global forbids segmented flash");
        return true;
    }();
    return enabled;
}
std::size_t cached_global_workspace_bytes(int maximum, int capacity) {
    valid_maximum(maximum);
    const int keys = cached_global_detail::staged_key_bound(capacity);
    return aligned(pp_tc_workspace_bytes(maximum)) +
        2 * aligned(std::size_t(keys) * 4 * 512 * sizeof(half)) + 256;
}
namespace cached_global_detail {
int staged_key_bound(int upper) {
    if (upper < 1 || upper > 262144)
        throw std::invalid_argument("Gemma cached global physical extent");
    return (upper + 255) & ~255;
}
Buffers bind(AttentionWorkspace workspace, int maximum, int capacity) {
    const auto bytes = cached_global_workspace_bytes(maximum, capacity);
    if (!workspace.data || reinterpret_cast<std::uintptr_t>(workspace.data) % 256 || workspace.bytes < bytes)
        throw std::invalid_argument("Gemma cached global stable owned arena");
    Buffers b{};
    b.keys = staged_key_bound(capacity);
    auto* p = static_cast<unsigned char*>(workspace.data) + aligned(pp_tc_workspace_bytes(maximum));
    const auto size = aligned(std::size_t(b.keys) * 4 * 512 * sizeof(half));
    b.key = reinterpret_cast<half*>(p); p += size;
    b.value = reinterpret_cast<half*>(p); p += size;
    b.invalid = reinterpret_cast<unsigned*>(p);
    return b;
}
// Only these pointers differ from parent TC. Q/S/P/output retain its exact layout.
void pointers(Buffers buffers, pp_tc_detail::Buffers tile, int tokens,
    int query, int key, cudaStream_t stream);
}
void cached_global_attention(const float* q, float* out, CacheView cache,
    const int* position, int past, int tokens, int maximum,
    AttentionWorkspace workspace, cudaStream_t stream) {
    static_assert(pp_tc_detail::Query == 512 && pp_tc_detail::Keys == 512,
        "Cached global must inherit the actual Query512/Keys512 parent geometry");
    if (!stream || !q || !out || !position || cache.dim != 512 || cache.heads != 4 || cache.window ||
        tokens < 4 || tokens > maximum || maximum > 2048 || past < 0 || cache.capacity < tokens ||
        past > cache.capacity - tokens || !workspace.handle || !workspace.blas_data || !workspace.blas_bytes)
        throw std::invalid_argument("Gemma cached global geometry/stream/owner");
    const auto qbytes = std::size_t(tokens) * 32 * 512 * sizeof(float);
    const auto kvbytes = std::size_t(cache.capacity) * 4 * (512 / 32) * 34;
    const Span spans[] = {{q, qbytes}, {out, qbytes}, {cache.key, kvbytes}, {cache.value, kvbytes},
        {position, 4}, {workspace.data, workspace.bytes}, {workspace.blas_data, workspace.blas_bytes}};
    disjoint(spans, 7);
    const auto b = cached_global_detail::bind(workspace, maximum, cache.capacity);
    const auto tile = pp_tc_detail::bind(workspace.data, workspace.bytes, maximum);
    blas(cublasSetStream(workspace.handle, stream));
    blas(cublasSetWorkspace(workspace.handle, workspace.blas_data, workspace.blas_bytes));
    blas(cublasSetPointerMode(workspace.handle, CUBLAS_POINTER_MODE_HOST));
    blas(cublasSetMathMode(workspace.handle, static_cast<cublasMath_t>(
        CUBLAS_TENSOR_OP_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)));
    const float one = 1, zero = 0;
    try {
        pp_tc_detail::prepare(tile, q, out, tokens, 512, stream);
        cached_global_detail::stage(b, cache, position, tokens, past + tokens, stream);
        for (int query = 0; query < tokens; query += pp_tc_detail::Query) {
            const int queries = std::min(pp_tc_detail::Query, tokens - query);
            const int last = past + query + queries - 1;
            for (int key = 0; key <= last; key += pp_tc_detail::Keys) {
                // Match parent key-tail sizes, including masked future keys within its tile.
                const int keys = std::min(pp_tc_detail::Keys, past + tokens - key);
                cached_global_detail::pointers(b, tile, tokens, query, key, stream);
                blas(cublasGemmBatchedEx(workspace.handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    keys, queries, 512, &one, tile.kt, CUDA_R_16F, 512, tile.qt, CUDA_R_16F, 512,
                    &zero, tile.st, CUDA_R_32F, pp_tc_detail::Keys, 32,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                pp_tc_detail::normalize(tile, cache, position, query, key, keys, queries, stream);
                blas(cublasGemmBatchedEx(workspace.handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    512, queries, keys, &one, tile.vt, CUDA_R_16F, 512, tile.pt, CUDA_R_16F,
                    pp_tc_detail::Keys, &zero, tile.ot, CUDA_R_32F, 512, 32,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                pp_tc_detail::accumulate(tile, out, query, queries, 512, stream);
            }
        }
        pp_tc_detail::finish(tile, out, tokens, 512, stream);
        cached_global_detail::propagate(b, out, tokens, stream);
    } catch (...) {
        (void)cublasSetMathMode(workspace.handle, CUBLAS_DEFAULT_MATH);
        throw;
    }
    blas(cublasSetMathMode(workspace.handle, CUBLAS_DEFAULT_MATH));
    primitive::check_cuda(cudaGetLastError(), "Gemma cached global query-outer result");
}
}
