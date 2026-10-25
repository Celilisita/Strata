#pragma once
#include "cpu_iq3_derived.h"
#include "ggml.h"

enum { STRATA_CPU_DERIVED_BANKS = 2, STRATA_CPU_DERIVED_KEYS = 288, STRATA_CPU_DERIVED_THREADS = 16 };

struct strata_cpu_derived_expert {
    const void * original;
    struct strata_iq3_derived_row payload;
    size_t rows;
    size_t row_blocks;
    int role;
};

struct strata_cpu_derived_worker_counts {
    uint64_t single_rows[2], pair_rows[2], lookup_misses[2], fallback_rows[2];
    uint64_t padding[8];
};

struct strata_cpu_derived_view {
    const struct ggml_tensor * banks[2];
    struct strata_cpu_derived_expert experts[2][288];
    struct strata_cpu_derived_worker_counts * counters;
};

static inline const struct strata_cpu_derived_expert * strata_cpu_derived_find(
        const struct strata_cpu_derived_view * view, const struct ggml_tensor * bank, int key, int ith) {
    if (!view || !bank || bank->type != GGML_TYPE_IQ3_S || key < 0 || key >= 288) return NULL;
    for (int role = 0; role < 2; ++role) {
        if (view->banks[role] != bank) continue;
        const struct strata_cpu_derived_expert * result = &view->experts[role][key];
        if (result->original == (const char *) bank->data + (size_t) key*bank->nb[2] &&
            result->rows == 2048 && result->row_blocks == 16 && result->payload.blocks == 32768) return result;
        if (view->counters && ith >= 0 && ith < 16) ++view->counters[ith].lookup_misses[role];
        return NULL;
    }
    return NULL;
}

static inline struct strata_iq3_derived_row strata_cpu_derived_row(
        const struct strata_cpu_derived_expert * expert, size_t row) {
    struct strata_iq3_derived_row result = {NULL, NULL, NULL, 0};
    if (!expert || row >= expert->rows) return result;
    result.levels = expert->payload.levels + row*expert->row_blocks*256;
    result.d_bits = expert->payload.d_bits + row*expert->row_blocks;
    result.packed_scales = expert->payload.packed_scales + row*expert->row_blocks*4;
    result.blocks = expert->row_blocks;
    return result;
}

#ifdef __cplusplus
extern "C" {
#endif
struct ggml_threadpool;
// Diagnostic caller must drain the existing pool before bind or unbind.
bool strata_glm_cpu_derived_bind(struct ggml_threadpool *, const struct strata_cpu_derived_view *);
#ifdef __cplusplus
}
#endif
