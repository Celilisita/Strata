#pragma once
#include "ggml-backend.h"
#include <stdbool.h>
#include <stdint.h>

// Private six-argument leaf; never cast to ggml_vec_dot_t.
#if defined(_MSC_VER)
#define STRATA_IQ4_PAIR_ALIGN __declspec(align(128))
#else
#define STRATA_IQ4_PAIR_ALIGN __attribute__((aligned(128)))
#endif
struct STRATA_IQ4_PAIR_ALIGN strata_iq4_down_pair_stats {
    uint64_t nodes, pair_enabled_nodes, paired_chunks, paired_weight_rows;
    uint64_t paired_output_rows, scalar_chunks, scalar_output_rows, nonfinite_outputs;
    double error_squared, reference_squared, max_abs_error, reference_max_abs;
    uint64_t padding[4]; // 128 bytes; each worker owns a distinct cache line pair.
};
#undef STRATA_IQ4_PAIR_ALIGN
struct strata_iq4_down_pair_snapshot {
    struct strata_iq4_down_pair_stats totals;
    int threads;
    bool verification_enabled;
};
#ifdef __cplusplus
extern "C" {
#endif
GGML_BACKEND_API int strata_glm_cpu_iq4_down_pair_available(void);
GGML_BACKEND_API void strata_glm_cpu_iq4_down_pair(int n, float * out0, float * out1,
    const void * original, const void * q8a, const void * q8b);
// Fixture only: one owner, one synchronous CPU graph, no concurrent CPU capture.
// Begin after prior work retires; end after that graph's final existing barrier.
// No capture is active by default, and timed samples must leave it inactive.
GGML_BACKEND_API bool strata_glm_cpu_iq4_down_pair_capture_begin(
    const struct ggml_tensor * original_down, bool verify);
GGML_BACKEND_API bool strata_glm_cpu_iq4_down_pair_capture_end(
    struct strata_iq4_down_pair_snapshot * output);
#ifdef __cplusplus
}
#endif
