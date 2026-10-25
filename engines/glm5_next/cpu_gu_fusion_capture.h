#pragma once
// Included after ggml-cpu.c native atomics; matches the existing warm capture.
#include "cpu_gu_fusion.h"
struct strata_cpu_gu_fusion_capture {
    const struct ggml_tensor * gate, * up;
    struct strata_cpu_gu_fusion_stats worker[16];
    bool phase_enabled;
    struct strata_cpu_gu_fusion_phase_worker phase[16];
};
#ifdef STRATA_GLM53_CPU_PAIR_DOT
static atomic_flag strata_gu_fusion_owned = ATOMIC_FLAG_INIT;
static atomic_bool strata_gu_fusion_active = false;
static struct strata_cpu_gu_fusion_capture strata_gu_fusion_data;
static bool strata_gu_fusion_capture_begin_impl(
        const struct ggml_tensor * gate, const struct ggml_tensor * up, bool phase_enabled) {
    if (!gate || !up || gate == up || gate->type != GGML_TYPE_IQ3_S || up->type != GGML_TYPE_IQ3_S ||
        gate->ne[0] != 4096 || gate->ne[1] != 2048 || gate->ne[2] != 288 || gate->ne[3] != 1 ||
        !ggml_are_same_shape(gate,up) || !ggml_is_contiguous(gate) || !ggml_is_contiguous(up)) return false;
    if (atomic_flag_test_and_set(&strata_gu_fusion_owned)) return false;
    memset(&strata_gu_fusion_data,0,sizeof(strata_gu_fusion_data));
    strata_gu_fusion_data.gate=gate; strata_gu_fusion_data.up=up;
    strata_gu_fusion_data.phase_enabled=phase_enabled;
    atomic_store_explicit(&strata_gu_fusion_active,true,memory_order_release);
    return true;
}
bool strata_glm_cpu_gu_fusion_capture_begin(const struct ggml_tensor * gate, const struct ggml_tensor * up) {
    return strata_gu_fusion_capture_begin_impl(gate,up,false);
}
bool strata_glm_cpu_gu_fusion_phase_capture_begin(const struct ggml_tensor * gate, const struct ggml_tensor * up) {
    return strata_gu_fusion_capture_begin_impl(gate,up,true);
}
static struct strata_cpu_gu_fusion_capture * strata_gu_fusion_capture_current(void) {
    return atomic_load_explicit(&strata_gu_fusion_active,memory_order_acquire) ? &strata_gu_fusion_data : NULL;
}
bool strata_glm_cpu_gu_fusion_capture_end(struct strata_cpu_gu_fusion_snapshot * result) {
    if (!result || !atomic_load_explicit(&strata_gu_fusion_active,memory_order_acquire)) return false;
    atomic_store_explicit(&strata_gu_fusion_active,false,memory_order_release);
    memset(result,0,sizeof(*result)); result->threads=16;
    result->phase_enabled=strata_gu_fusion_data.phase_enabled;
    memcpy(result->phase,strata_gu_fusion_data.phase,sizeof(result->phase));
    // Synchronous graph return has retired writes through its original barrier.
    for (int i=0;i<16;++i) {
        const struct strata_cpu_gu_fusion_stats * value=&strata_gu_fusion_data.worker[i];
#define STRATA_GU_SUM(field) result->totals.field += value->field
        STRATA_GU_SUM(fused_nodes); STRATA_GU_SUM(quantized_blocks); STRATA_GU_SUM(mapped_routes);
        STRATA_GU_SUM(chunks); STRATA_GU_SUM(gate_rows); STRATA_GU_SUM(up_rows);
        STRATA_GU_SUM(glu_elements); STRATA_GU_SUM(ny1_chunks); STRATA_GU_SUM(ny2_chunks);
        STRATA_GU_SUM(lut_nodes); STRATA_GU_SUM(lut_classes); STRATA_GU_SUM(lut_table_builds);
        STRATA_GU_SUM(lut_worker_build_calls); STRATA_GU_SUM(lut_chunk_calls); STRATA_GU_SUM(lut_barrier_arrivals); STRATA_GU_SUM(lut_workspace_bytes);
#undef STRATA_GU_SUM
    }
    atomic_flag_clear(&strata_gu_fusion_owned); return true;
}
#else
bool strata_glm_cpu_gu_fusion_capture_begin(const struct ggml_tensor * gate, const struct ggml_tensor * up) {
    GGML_UNUSED(gate); GGML_UNUSED(up); return false;
}
bool strata_glm_cpu_gu_fusion_phase_capture_begin(const struct ggml_tensor * gate, const struct ggml_tensor * up) {
    GGML_UNUSED(gate); GGML_UNUSED(up); return false;
}
bool strata_glm_cpu_gu_fusion_capture_end(struct strata_cpu_gu_fusion_snapshot * result) {
    GGML_UNUSED(result); return false;
}
#endif
