#pragma once
#include "ggml.h"
#include <stdbool.h>

// Recover only the unretained IQ4_XS Down Ny2 schedule from R16. The old
// IQ3_S two-column branch now returns through the existing paired-dot path.
// The caller snapshots the opt-in; this helper does not read environment or
// own weights, scratch, work queues, or a second pool.
static inline bool strata_glm_cpu_r16_down_schedule(
        const struct ggml_tensor * bank, const struct ggml_tensor * ids,
        bool requested, int threads) {
    return requested && threads == 16 && bank->type == GGML_TYPE_IQ4_XS &&
        bank->ne[0] == 2048 && bank->ne[1] == 4096 && bank->ne[2] == 288 &&
        bank->ne[3] == 1 && ggml_is_contiguous(bank) && ids->ne[0] == 1 &&
        ids->ne[1] >= 2 && ids->ne[1] <= 16 && ids->ne[2] == 1 && ids->ne[3] == 1;
}

static inline int64_t strata_glm_cpu_r16_down_block(int64_t columns, bool enabled) {
    // Ny1 retains the original16-row scalar loop and its original chunk plan.
    return enabled && columns == 2 ? 8 : 16;
}
