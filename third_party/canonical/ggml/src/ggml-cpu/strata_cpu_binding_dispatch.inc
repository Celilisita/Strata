bool strata_glm_cpu_derived_bind(struct ggml_threadpool * pool, const struct strata_cpu_derived_view * view) {
#ifdef STRATA_GLM53_CPU_PAIR_DOT
    if (!pool || (view && (pool->n_threads != 16 || pool->strata_derived_view || pool->strata_iq3_r4_view || pool->strata_exact_lut_view))) return false;
    pool->strata_derived_view = view;
    return true;
#else
    return pool && !view;
#endif
}

bool strata_glm_cpu_iq3_r4_bind(struct ggml_threadpool * pool, const struct strata_cpu_iq3_r4_view * view) {
#ifdef STRATA_GLM53_CPU_PAIR_DOT
    if (!pool || (view && (pool->n_threads != 16 || pool->strata_iq3_r4_view || pool->strata_derived_view || pool->strata_exact_lut_view))) return false;
    pool->strata_iq3_r4_view = view;
    return true;
#else
    return pool && !view;
#endif
}

bool strata_glm_cpu_iq3_lut_bind(struct ggml_threadpool * pool, const struct strata_cpu_iq3_lut_view * view) {
#ifdef STRATA_GLM53_CPU_PAIR_DOT
    if (!pool || (view && (pool->n_threads != 16 || pool->strata_exact_lut_view ||
            pool->strata_iq3_r4_view || pool->strata_derived_view))) return false;
    pool->strata_exact_lut_view = view;
    return true;
#else
    return pool && !view;
#endif
}

#ifdef STRATA_GLM53_CPU_PAIR_DOT
static void strata_cpu_dot_options_prepare(struct ggml_threadpool * threadpool) {
    threadpool->strata_dot_options = strata_glm_cpu_dot_options_from_env();
    if(threadpool->strata_dot_options.gu_fusion && ggml_tiled_force_enabled())
        threadpool->strata_dot_options.gu_fusion=false;
    threadpool->strata_gu_fusion_capture=threadpool->strata_dot_options.gu_fusion?strata_gu_fusion_capture_current():NULL;
    threadpool->strata_dot_options.gu_dual_lut &= threadpool->strata_gu_dual_lut_capability && threadpool->strata_dot_options.gu_fusion;
    threadpool->strata_iq4_down_pair_capture = strata_iq4_down_pair_capture_current();
}
#endif

