#pragma once
#include "cpu_iq3_exact_lut.h"
#include "ggml.h"

struct strata_cpu_iq3_lut_worker_counts {
    uint64_t ny1_rows[2], ny2_pairs[2], lookup_misses[2], fallback_rows[2], chunk_calls[2], lut_blocks[2];
    uint64_t padding[4];
};
struct strata_cpu_iq3_lut_view {
    const struct ggml_tensor * banks[2];
    struct strata_cpu_iq3_lut_worker_counts * counters;
};
#ifdef __cplusplus
extern "C" {
#endif
struct ggml_threadpool;
// Quiescent diagnostic caller; existingpoolkickoff publishes borrowed control.
bool strata_glm_cpu_iq3_lut_bind(struct ggml_threadpool *, const struct strata_cpu_iq3_lut_view *);
#ifdef __cplusplus
}
#endif
