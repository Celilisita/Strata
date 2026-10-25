#pragma once
// Included in ggml-cpu.c after its atomic compatibility definitions.
#include "cpu_iq4_down_pair.h"
struct strata_iq4_down_pair_capture {
    const struct ggml_tensor * bank;
    bool verify;
    struct strata_iq4_down_pair_stats worker[16];
};
#ifdef STRATA_GLM53_CPU_PAIR_DOT
static atomic_flag strata_iq4_down_pair_owned = ATOMIC_FLAG_INIT;
static atomic_bool strata_iq4_down_pair_active = false;
static struct strata_iq4_down_pair_capture strata_iq4_down_pair_data;
bool strata_glm_cpu_iq4_down_pair_capture_begin(const struct ggml_tensor * bank, bool verify) {
    if (!bank || bank->type != GGML_TYPE_IQ4_XS || bank->ne[0] != 2048 ||
        bank->ne[1] != 4096 || bank->ne[2] != 288 || bank->ne[3] != 1 ||
        !ggml_is_contiguous(bank)) return false;
    if (atomic_flag_test_and_set(&strata_iq4_down_pair_owned)) return false;
    memset(&strata_iq4_down_pair_data,0,sizeof(strata_iq4_down_pair_data));
    strata_iq4_down_pair_data.bank = bank;
    strata_iq4_down_pair_data.verify = verify;
    atomic_store_explicit(&strata_iq4_down_pair_active,true,memory_order_release);
    return true;
}
static struct strata_iq4_down_pair_capture * strata_iq4_down_pair_capture_current(void) {
    return atomic_load_explicit(&strata_iq4_down_pair_active,memory_order_acquire) ?
        &strata_iq4_down_pair_data : NULL;
}
bool strata_glm_cpu_iq4_down_pair_capture_end(struct strata_iq4_down_pair_snapshot * out) {
    if (!out || !atomic_load_explicit(&strata_iq4_down_pair_active,memory_order_acquire)) return false;
    // Synchronous graph return has crossed the existing final worker barrier.
    atomic_store_explicit(&strata_iq4_down_pair_active,false,memory_order_release);
    memset(out,0,sizeof(*out));
    out->threads = 16; out->verification_enabled = strata_iq4_down_pair_data.verify;
    for (int i = 0; i < 16; ++i) {
        const struct strata_iq4_down_pair_stats * s = &strata_iq4_down_pair_data.worker[i];
#define STRATA_IQ4_SUM(field) out->totals.field += s->field
        STRATA_IQ4_SUM(nodes); STRATA_IQ4_SUM(pair_enabled_nodes);
        STRATA_IQ4_SUM(paired_chunks); STRATA_IQ4_SUM(paired_weight_rows);
        STRATA_IQ4_SUM(paired_output_rows); STRATA_IQ4_SUM(scalar_chunks);
        STRATA_IQ4_SUM(scalar_output_rows); STRATA_IQ4_SUM(nonfinite_outputs);
        STRATA_IQ4_SUM(error_squared); STRATA_IQ4_SUM(reference_squared);
#undef STRATA_IQ4_SUM
        if (s->max_abs_error > out->totals.max_abs_error) out->totals.max_abs_error = s->max_abs_error;
        if (s->reference_max_abs > out->totals.reference_max_abs) out->totals.reference_max_abs = s->reference_max_abs;
    }
    atomic_flag_clear(&strata_iq4_down_pair_owned);
    return true;
}
#else
bool strata_glm_cpu_iq4_down_pair_capture_begin(const struct ggml_tensor * bank, bool verify) {
    GGML_UNUSED(bank); GGML_UNUSED(verify); return false;
}
bool strata_glm_cpu_iq4_down_pair_capture_end(struct strata_iq4_down_pair_snapshot * out) {
    GGML_UNUSED(out); return false;
}
#endif
