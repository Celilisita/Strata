#pragma once
#include "cpu_iq3_r4.h"
#include "ggml.h"

struct strata_cpu_iq3_r4_expert {
    const void * original;
    const uint8_t * packed;
    size_t bytes;
    int role;
};
struct strata_cpu_iq3_r4_worker_counts {
    uint64_t ny1_rows[2], ny2_pairs[2], lookup_misses[2], fallback_rows[2];
    uint64_t padding[8];
};
struct strata_cpu_iq3_r4_view {
    const struct ggml_tensor * banks[2];
    struct strata_cpu_iq3_r4_expert experts[2][288];
    struct strata_cpu_iq3_r4_worker_counts * counters;
};
static inline const struct strata_cpu_iq3_r4_expert * strata_cpu_iq3_r4_find(
        const struct strata_cpu_iq3_r4_view * view, const struct ggml_tensor * bank, int key, int ith) {
    if (!view || !bank || bank->type != GGML_TYPE_IQ3_S || key < 0 || key >= 288) return NULL;
    for (int role = 0; role < 2; ++role) {
        if (view->banks[role] != bank) continue;
        const struct strata_cpu_iq3_r4_expert * entry = &view->experts[role][key];
        if (entry->original == (const char *)bank->data+(size_t)key*bank->nb[2] &&
            entry->packed && entry->bytes == 3604480 && entry->role == role) return entry;
        if (view->counters && ith >= 0 && ith < 16) ++view->counters[ith].lookup_misses[role];
        return NULL;
    }
    return NULL;
}
static inline struct strata_iq3_r4_rows strata_cpu_iq3_r4_group(
        const struct strata_cpu_iq3_r4_expert * entry, size_t first_row) {
    struct strata_iq3_r4_rows group = {NULL,0};
    if (!entry || !entry->packed || first_row >= 2048 || first_row%4) return group;
    group.data = entry->packed+(first_row/4)*16*STRATA_IQ3_R4_BLOCK_BYTES;
    group.blocks = 16;
    return group;
}

#ifdef __cplusplus
extern "C" {
#endif
struct ggml_threadpool;
// Fixture-only. Caller drains the original pool before both transitions.
bool strata_glm_cpu_iq3_r4_bind(struct ggml_threadpool *, const struct strata_cpu_iq3_r4_view *);
#ifdef __cplusplus
}
#endif
