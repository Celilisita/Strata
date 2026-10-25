#pragma once
// Fixture-only, one synchronous CPU graph per capture. No new barriers or pool.
#include "ggml-backend.h"
#include "cpu_barrier_poll.h"
#include <stdint.h>
#if defined(_MSC_VER)
#define STRATA_CPU_PHASE_ALIGN __declspec(align(64))
#else
#define STRATA_CPU_PHASE_ALIGN __attribute__((aligned(64)))
#endif
struct STRATA_CPU_PHASE_ALIGN StrataCpuPhaseThread { uint64_t stamp_us[8];uint64_t row_dot_calls[2],row_dot_tail_rows,vnni_single_rows,vnni_pair_rows;
    uint64_t pair4_calls,pair4_canonical_rows,pair4_tail_rows; };
struct StrataCpuPhaseRole {
    int type, threads, final_barrier, pair_path, wide_path, graph_snapshot;
    int static_down_requested, static_down_experts, dynamic_down_experts;
    int barrier_poll; // Actual immutable pool state, only sampled during capture.
    int row_dot_option,vnni_dot_option; // Supported caller snapshot; actual calls are per worker.
    int pair4_option,pair4_path; // Independent option, actual shape eligibility.
    int64_t input_columns, input_rows;
    struct StrataCpuPhaseThread worker[16];
};
struct StrataCpuPhaseProfile {
    int version, dropped, wide_available;
    struct StrataCpuPhaseRole role[3]; // Gate, Up, Down; original bank identity.
};
#ifdef __cplusplus
extern "C" {
#endif
// begin/end must surround one completed, synchronous graph on the caller's
// existing CPU backend. Only one capture may be active process-wide.
// configure is called once before creating workers. Without it, hooks are off.
GGML_BACKEND_API bool strata_glm_cpu_phase_configure(void);
GGML_BACKEND_API bool strata_glm_cpu_phase_begin(const struct ggml_tensor * gate,
    const struct ggml_tensor * up, const struct ggml_tensor * down);
GGML_BACKEND_API void strata_glm_cpu_phase_end(struct StrataCpuPhaseProfile * output);
GGML_BACKEND_API int strata_glm_cpu_phase_available(void);
#ifdef __cplusplus
}
#endif

#ifdef STRATA_CPU_PHASE_IMPLEMENTATION
// Included after ggml-cpu.c's native MSVC/C11 atomic compatibility definitions.
static atomic_int strata_cpu_phase_active = 0;
static atomic_int strata_cpu_phase_dropped = 0;
static atomic_flag strata_cpu_phase_owned = ATOMIC_FLAG_INIT;
static bool strata_cpu_phase_configured = false;
static bool strata_cpu_phase_enabled = false; // Immutable before workers exist.
static const struct ggml_tensor * strata_cpu_phase_banks[3];
static struct StrataCpuPhaseProfile strata_cpu_phase_data;

int strata_glm_cpu_phase_available(void) { return 1; }
bool strata_glm_cpu_phase_configure(void) {
    if (strata_cpu_phase_configured) return true;
    const char * flag = getenv("STRATA_GLM53_CPU_PHASE_PROFILE");
    if (flag && strcmp(flag, "0") && strcmp(flag, "1")) return false;
    strata_cpu_phase_enabled = flag && !strcmp(flag, "1");
    strata_cpu_phase_configured = true;
    return true;
}
bool strata_glm_cpu_phase_begin(const struct ggml_tensor * gate,
        const struct ggml_tensor * up, const struct ggml_tensor * down) {
    if (!strata_cpu_phase_enabled || !gate || !up || !down ||
            gate == up || gate == down || up == down) return false;
    if (atomic_flag_test_and_set(&strata_cpu_phase_owned)) return false;
    memset(&strata_cpu_phase_data, 0, sizeof(strata_cpu_phase_data));
    strata_cpu_phase_data.version = 1;
#ifdef STRATA_GLM53_CPU_PAIR_DOT
    strata_cpu_phase_data.wide_available = ggml_vec_dot_iq3_s_q8_K_wide_available();
#endif
    strata_cpu_phase_banks[0] = gate; strata_cpu_phase_banks[1] = up; strata_cpu_phase_banks[2] = down;
    atomic_store_explicit(&strata_cpu_phase_dropped, 0, memory_order_relaxed);
    atomic_store_explicit(&strata_cpu_phase_active, 1, memory_order_release);
    return true;
}
void strata_glm_cpu_phase_end(struct StrataCpuPhaseProfile * output) {
    // All writes precede the existing final graph barrier. CPU caller return
    // crosses that barrier; it need not join the persistent worker threads.
    if (!atomic_load_explicit(&strata_cpu_phase_active, memory_order_acquire)) {
        if (output) memset(output, 0, sizeof(*output));
        return;
    }
    atomic_store_explicit(&strata_cpu_phase_active, 0, memory_order_release);
    strata_cpu_phase_data.dropped = atomic_load_explicit(&strata_cpu_phase_dropped, memory_order_relaxed);
    if (output) memcpy(output, &strata_cpu_phase_data, sizeof(*output));
    atomic_flag_clear(&strata_cpu_phase_owned);
}
static inline int strata_cpu_phase_role(const struct ggml_tensor * bank) {
    if (!strata_cpu_phase_enabled) return -1;
    if (!atomic_load_explicit(&strata_cpu_phase_active, memory_order_acquire)) return -1;
    for (int role = 0; role < 3; ++role) if (bank == strata_cpu_phase_banks[role]) return role;
    return -1;
}
static inline void strata_cpu_phase_stamp(int role, int ith, int slot) {
    if (role < 0) return;
    if (ith < 0 || ith >= 16 || slot < 0 || slot >= 8) {
        atomic_fetch_add_explicit(&strata_cpu_phase_dropped, 1, memory_order_relaxed); return;
    }
    uint64_t * stamp = &strata_cpu_phase_data.role[role].worker[ith].stamp_us[slot];
    if (*stamp) atomic_fetch_add_explicit(&strata_cpu_phase_dropped, 1, memory_order_relaxed);
    *stamp = (uint64_t) ggml_time_us();
}
static inline void strata_cpu_phase_final_arrival(int role, int ith) {
    if (role >= 0 && ith == 0) strata_cpu_phase_data.role[role].final_barrier = 1;
    strata_cpu_phase_stamp(role, ith, 6);
}
static inline void strata_cpu_phase_paths(int role, int ith, bool pair, bool wide, bool snapshot, bool static_down, bool row_dot, bool vnni_dot,
        const struct ggml_threadpool * pool) {
    if (role >= 0 && ith == 0) {
        strata_cpu_phase_data.role[role].pair_path = pair;
        strata_cpu_phase_data.role[role].wide_path = wide;
        strata_cpu_phase_data.role[role].graph_snapshot = snapshot;
        strata_cpu_phase_data.role[role].static_down_requested = static_down;
        strata_cpu_phase_data.role[role].barrier_poll = strata_glm_cpu_barrier_poll_state(pool);
        strata_cpu_phase_data.role[role].row_dot_option = row_dot;
        strata_cpu_phase_data.role[role].vnni_dot_option = vnni_dot;
    }
}
static inline void strata_cpu_phase_row_dot(int role,int ith,int columns,uint64_t calls,uint64_t tails) {
    if(role<0)return;
    if(ith<0||ith>=16||columns<1||columns>2){atomic_fetch_add_explicit(&strata_cpu_phase_dropped,1,memory_order_relaxed);return;}
    strata_cpu_phase_data.role[role].worker[ith].row_dot_calls[columns-1]+=calls;
    strata_cpu_phase_data.role[role].worker[ith].row_dot_tail_rows+=tails;
}
static inline void strata_cpu_phase_vnni_dot(int role,int ith,int columns,uint64_t rows) {
    if(role<0)return;
    if(ith<0||ith>=16||columns<1||columns>2){atomic_fetch_add_explicit(&strata_cpu_phase_dropped,1,memory_order_relaxed);return;}
    if(columns==1)strata_cpu_phase_data.role[role].worker[ith].vnni_single_rows+=rows;
    else strata_cpu_phase_data.role[role].worker[ith].vnni_pair_rows+=rows;
}
static inline void strata_cpu_phase_pair4_path(int role,int ith,bool option,bool path) {
    if(role>=0&&ith==0){strata_cpu_phase_data.role[role].pair4_option=option;strata_cpu_phase_data.role[role].pair4_path=path;}
}
static inline void strata_cpu_phase_pair4_rows(int role,int ith,uint64_t pairs,uint64_t canonical,uint64_t tails) {
    if(role<0)return;
    if(ith<0||ith>=16){atomic_fetch_add_explicit(&strata_cpu_phase_dropped,1,memory_order_relaxed);return;}
    struct StrataCpuPhaseThread* worker=&strata_cpu_phase_data.role[role].worker[ith];
    worker->pair4_calls+=pairs;worker->pair4_canonical_rows+=canonical;worker->pair4_tail_rows+=tails;
}
static inline void strata_cpu_phase_down_chunk(int role, int ith, bool is_static) {
    if (role != 2 || ith != 0) return;
    if (is_static) ++strata_cpu_phase_data.role[role].static_down_experts;
    else ++strata_cpu_phase_data.role[role].dynamic_down_experts;
}
static inline int strata_cpu_phase_enter(const struct ggml_tensor * bank,
        const struct ggml_tensor * input, int ith, int nth) {
    const int role = strata_cpu_phase_role(bank);
    strata_cpu_phase_stamp(role, ith, 0);
    if (role >= 0 && ith == 0) {
        strata_cpu_phase_data.role[role].type = bank->type;
        strata_cpu_phase_data.role[role].threads = nth;
        strata_cpu_phase_data.role[role].input_columns = input->ne[0];
        strata_cpu_phase_data.role[role].input_rows = ggml_nelements(input) / input->ne[0];
    }
    return role;
}
#endif
#undef STRATA_CPU_PHASE_ALIGN
