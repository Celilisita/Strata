#pragma once
#include "ggml-backend.h"
#include <stdbool.h>
#include <stdint.h>
#if defined(_MSC_VER)
#define STRATA_GU_ALIGN __declspec(align(128))
#else
#define STRATA_GU_ALIGN __attribute__((aligned(128)))
#endif
struct STRATA_GU_ALIGN strata_cpu_gu_fusion_stats {
    uint64_t fused_nodes, quantized_blocks, mapped_routes, chunks;
    uint64_t gate_rows, up_rows, glu_elements, ny1_chunks, ny2_chunks;
    uint64_t lut_nodes, lut_classes, lut_table_builds, lut_worker_build_calls;
    uint64_t lut_chunk_calls, lut_barrier_arrivals;
    uint64_t lut_workspace_bytes;
};
// Diagnostic-only monotonic microseconds from ggml_time_us(). Equal stamps are
// legal at this resolution; zero means no phase capture/point not reached.
struct strata_cpu_gu_fusion_phase_block {
    uint64_t build_enter_us, build_return_us, publish_return_us;
    uint64_t consume_return_us, retire_return_us; // Block15 retire stays zero.
};
struct STRATA_GU_ALIGN strata_cpu_gu_fusion_phase_worker {
    uint64_t helper_enter_us, class_enter_us, class_return_us, helper_return_us;
    struct strata_cpu_gu_fusion_phase_block block[16];
};
#undef STRATA_GU_ALIGN
struct strata_cpu_gu_fusion_snapshot {
    struct strata_cpu_gu_fusion_stats totals;
    int threads;
    bool phase_enabled;
    struct strata_cpu_gu_fusion_phase_worker phase[16];
};
#ifdef __cplusplus
extern "C" {
#endif
// One warm-only synchronous graph capture. Begin/end require prior retirement;
// timed calls leave capture inactive. Worker slots publish at the old barrier.
GGML_BACKEND_API bool strata_glm_cpu_gu_fusion_capture_begin(
    const struct ggml_tensor * gate_bank, const struct ggml_tensor * up_bank);
// Same ownership and synchronous-retirement contract; phase clocks are opt-in.
GGML_BACKEND_API bool strata_glm_cpu_gu_fusion_phase_capture_begin(
    const struct ggml_tensor * gate_bank, const struct ggml_tensor * up_bank);
GGML_BACKEND_API bool strata_glm_cpu_gu_fusion_capture_end(
    struct strata_cpu_gu_fusion_snapshot * result);
#ifdef __cplusplus
}
#endif
