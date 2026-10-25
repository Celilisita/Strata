#ifndef STRATA_CPU_BARRIER_POLL_H
#define STRATA_CPU_BARRIER_POLL_H
// Internal Strata diagnostics, not part of ggml's installed/public API.
#include "ggml-backend.h"
struct ggml_threadpool;
#ifdef __cplusplus
extern "C" {
#endif
// Actual immutable state of this pool: 1 on, 0 off, -1 unsupported build,
// -2 null pool. Changing the environment does not change an existing pool.
GGML_BACKEND_API int strata_glm_cpu_barrier_poll_state(const struct ggml_threadpool * pool);
// Publication fixture only: call from every active worker of the same bound
// native graph/pool. This delegates to the original barrier; it creates no pool.
GGML_BACKEND_API void strata_glm_cpu_barrier_wait_diagnostic(struct ggml_threadpool * pool);
#ifdef __cplusplus
}
#endif
#endif

#if defined(STRATA_CPU_BARRIER_POLL_IMPLEMENTATION) && !defined(STRATA_CPU_BARRIER_POLL_IMPLEMENTED)
#define STRATA_CPU_BARRIER_POLL_IMPLEMENTED
// Include implementation only after ggml_threadpool and cpu_relax are defined.
// Never cast a C11 _Atomic object into a plain/volatile access on other builds.
#if defined(STRATA_GLM53_CPU_PAIR_DOT) && defined(_WIN32) && defined(_MSC_VER) && \
    !defined(__clang__) && !defined(__INTEL_COMPILER) && !defined(__INTEL_LLVM_COMPILER) && \
    defined(_M_AMD64) && !defined(_M_ARM64EC) && !defined(_M_ARM64) && \
    !defined(GGML_USE_OPENMP) && !defined(GGML_TSAN_ENABLED)
#define STRATA_CPU_BARRIER_POLL_SUPPORTED 1
#else
#define STRATA_CPU_BARRIER_POLL_SUPPORTED 0
#endif

#ifdef STRATA_GLM53_CPU_PAIR_DOT
static void strata_cpu_barrier_poll_prepare(struct ggml_threadpool * pool) {
    pool->strata_barrier_poll = false;
#if STRATA_CPU_BARRIER_POLL_SUPPORTED
    GGML_ASSERT(sizeof(LONG) == 4 && sizeof(atomic_int) == 4);
    GGML_ASSERT(((uintptr_t) &pool->n_barrier_passed & 3u) == 0);
    const char * flag = getenv("STRATA_GLM53_CPU_BARRIER_POLL");
    // Only an exact opt-in enables the candidate. No per-barrier getenv.
    pool->strata_barrier_poll = flag && flag[0] == '1' && flag[1] == '\0';
#endif
}
#endif

#if STRATA_CPU_BARRIER_POLL_SUPPORTED
// Stable C symbol for a bounded map-address disassembly check. The measured
// candidate includes this one call/return per non-last barrier waiter.
__declspec(noinline) void strata_cpu_barrier_poll_wait(const volatile LONG * generation, LONG expected) {
    // ReadNoFence is a naturally aligned32-bit read, not a locked RMW.
    // Arrival/publication atomics and the original exit fence remain outside.
    while (ReadNoFence(generation) == expected) {
        ggml_thread_cpu_relax();
    }
}
#endif

int strata_glm_cpu_barrier_poll_state(const struct ggml_threadpool * pool) {
    if (!pool) return -2;
#if STRATA_CPU_BARRIER_POLL_SUPPORTED
    return pool->strata_barrier_poll ? 1 : 0;
#else
    return -1;
#endif
}

void strata_glm_cpu_barrier_wait_diagnostic(struct ggml_threadpool * pool) {
    GGML_ASSERT(pool);
    ggml_barrier(pool);
}
#endif
