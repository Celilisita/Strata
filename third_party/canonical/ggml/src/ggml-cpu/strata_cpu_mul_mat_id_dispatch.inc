static void ggml_compute_forward_mul_mat_id_impl(
        const struct ggml_compute_params * params, struct ggml_tensor * dst,
        struct ggml_tensor * fused_up, struct ggml_tensor * fused_glu, int lut_next_node) {
#ifndef STRATA_GLM53_CPU_PAIR_DOT
    GGML_UNUSED(fused_up); GGML_UNUSED(fused_glu); GGML_UNUSED(lut_next_node);
#endif

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const struct ggml_tensor * ids = dst->src[2];

    GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;
#ifdef STRATA_GLM53_CPU_PHASE_PROFILE
    const int strata_phase = strata_cpu_phase_enter(src0, src1, ith, nth);
#else
    const int strata_phase = -1;
#endif

    const enum ggml_type type = src0->type;

    const bool src1_cont = ggml_is_contiguous(src1);
    const struct strata_cpu_derived_view * strata_derived_view = NULL;
    const struct strata_cpu_iq3_r4_view * strata_r4_view = NULL;
    const struct strata_cpu_iq3_lut_view * strata_lut_view = NULL;
#ifdef STRATA_GLM53_CPU_PAIR_DOT
    strata_derived_view = params->threadpool->strata_derived_view;
    strata_r4_view = params->threadpool->strata_iq3_r4_view;
    strata_lut_view = params->threadpool->strata_exact_lut_view;
#endif
    bool strata_pair=false,strata_wide=false,strata_static_down=false,strata_row_dot=false,strata_vnni=false;
    bool strata_pair4=false,strata_pair4_shape=false,strata_iq4_packed=false;
    bool strata_down_pair=false,strata_schedule=false;
    struct strata_iq4_down_pair_capture * strata_down_capture=NULL;
#ifdef STRATA_GLM53_CPU_PAIR_DOT
    strata_pair=strata_glm_cpu_pair_dot(src0,ids,&params->threadpool->strata_dot_options);
    strata_wide=strata_glm_cpu_iq3_wide(src0,ids,&params->threadpool->strata_dot_options);
    strata_static_down=strata_glm_cpu_static_down(src0,ids,&params->threadpool->strata_dot_options,nth);
    strata_row_dot=strata_glm_cpu_row_dot_shape(src0,ids,&params->threadpool->strata_dot_options,nth);
    strata_vnni=strata_glm_cpu_vnni_dot_shape(src0,ids,&params->threadpool->strata_dot_options,nth);
    strata_pair4_shape=strata_glm_cpu_pair4_shape(src0,ids,nth);
    strata_pair4=strata_pair4_shape&&strata_glm_cpu_pair4_enabled(&params->threadpool->strata_dot_options);
    strata_iq4_packed=strata_glm_cpu_iq4_packed_shape(src0,ids,&params->threadpool->strata_dot_options,nth);
    strata_down_pair=strata_glm_cpu_iq4_down_pair_shape(src0,ids,&params->threadpool->strata_dot_options,nth);
    strata_schedule=!strata_static_down&&!strata_iq4_packed&&!strata_down_pair&&
        strata_glm_cpu_r16_down_schedule(src0,ids,params->threadpool->strata_dot_options.pair_schedule,nth);
    if (nth==16 && params->threadpool->strata_iq4_down_pair_capture &&
        params->threadpool->strata_iq4_down_pair_capture->bank==src0) {
        strata_down_capture=params->threadpool->strata_iq4_down_pair_capture;
        if (ith==0) { ++strata_down_capture->worker[0].nodes;
            strata_down_capture->worker[0].pair_enabled_nodes+=strata_down_pair; }
    }
#endif
#ifdef STRATA_GLM53_CPU_PHASE_PROFILE
    bool strata_snapshot=false,strata_static_requested=false,strata_row_option=false,strata_vnni_option=false;
#ifdef STRATA_GLM53_CPU_PAIR_DOT
    strata_snapshot=params->threadpool->strata_dot_options.graph_snapshot;
    strata_static_requested=params->threadpool->strata_dot_options.static_down;
    strata_row_option=params->threadpool->strata_dot_options.row_dot;
    strata_vnni_option=params->threadpool->strata_dot_options.vnni_dot;
#endif
    strata_cpu_phase_paths(strata_phase, ith, strata_pair, strata_wide, strata_snapshot, strata_static_requested, strata_row_option, strata_vnni_option, params->threadpool);
#ifdef STRATA_GLM53_CPU_PAIR_DOT
    strata_cpu_phase_pair4_path(strata_phase,ith,params->threadpool->strata_dot_options.pair4,strata_pair4);
#endif
#endif

    enum ggml_type    const vec_dot_type    = type_traits_cpu[type].vec_dot_type;
    ggml_from_float_t const from_float      = type_traits_cpu[vec_dot_type].from_float;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // row groups
    const int n_ids = ids->ne[0]; // n_expert_used
    const int n_as  = ne02;       // n_expert

    void * wdata_cur = params->wdata;

    if (src1->type != vec_dot_type) {
        incr_ptr_aligned(&wdata_cur, ggml_row_size(vec_dot_type, ggml_nelements(src1)), sizeof(int64_t));
    }

    int64_t * matrix_row_counts = // [n_as]
        incr_ptr_aligned(&wdata_cur, n_as*sizeof(int64_t), sizeof(int64_t));

    struct mmid_row_mapping * matrix_rows = // [n_as][ids->ne[0]*ids->ne[1]]
        incr_ptr_aligned(&wdata_cur, n_as*ids->ne[0]*ids->ne[1]*sizeof(struct mmid_row_mapping), sizeof(int64_t));

    char (*atomic_current_chunk)[CACHE_LINE_SIZE] = // [n_as]
        incr_ptr_aligned(&wdata_cur, CACHE_LINE_SIZE * n_as, CACHE_LINE_SIZE);

    // Tiled matmul (see tiled.h); per-thread work buffers, 0 bytes when disabled. The
    // reservation is unconditional, the per expert eligibility is decided at dispatch time
    char * tiled_scratch = incr_ptr_aligned(&wdata_cur, ggml_tiled_wdata_size(nth, dst), 64);

    void * strata_dual_workspace = NULL;
#ifdef STRATA_GLM53_CPU_PAIR_DOT
    if (lut_next_node >= 0)
        strata_dual_workspace = incr_ptr_aligned(&wdata_cur, STRATA_GU_DUAL_WORK_BYTES, 128);
#else
    GGML_UNUSED(strata_dual_workspace);
#endif
    GGML_ASSERT(params->wsize >= (size_t)((char *) wdata_cur - (char *) params->wdata));

#ifdef STRATA_GLM53_CPU_PHASE_PROFILE
    strata_cpu_phase_stamp(strata_phase, ith, 1);
#endif
    if (src1->type != vec_dot_type) {
        char * wdata = params->wdata;

        const size_t nbw0 = ggml_type_size(vec_dot_type);
        const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        const size_t nbw2 = nbw1*ne11;
        const size_t nbw3 = nbw2*ne12;

        assert(params->wsize >= ne13*nbw3);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);

#if 0
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = ith; i12 < ne12; i12 += nth) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                               ne10);
                }
            }
        }
#else
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    size_t bs = ggml_blck_size(vec_dot_type);
                    int64_t ne10_block_start = (ith * ne10/bs) / nth;
                    int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + ne10_block_start*bs*nb10),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0),
                               (ne10_block_end - ne10_block_start) * bs);
                }
            }
        }
#endif
    }

#ifdef STRATA_GLM53_CPU_PHASE_PROFILE
    strata_cpu_phase_stamp(strata_phase, ith, 2);
#endif

    if (ith == 0) {
        // initialize matrix_row_counts
        memset(matrix_row_counts, 0, n_as*sizeof(int64_t));

        // group rows by src0 matrix
        for (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            for (int id = 0; id < n_ids; ++id) {
                const int32_t i02 = *(const int32_t *) ((const char *) ids->data + iid1*ids->nb[1] + id*ids->nb[0]);

                assert(i02 >= 0 && i02 < n_as);

                MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = (struct mmid_row_mapping) {id, iid1};
                matrix_row_counts[i02] += 1;
            }
        }
    }

    // reset current_chunk
    for (int cur_a = ith; cur_a < n_as; cur_a += nth) {
        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);
        *current_chunk_ctr = nth;
    }

#ifdef STRATA_GLM53_CPU_PHASE_PROFILE
    strata_cpu_phase_stamp(strata_phase, ith, 3);
#endif
    ggml_barrier(params->threadpool);
#ifdef STRATA_GLM53_CPU_PHASE_PROFILE
    strata_cpu_phase_stamp(strata_phase, ith, 4);
#endif

#ifdef STRATA_GLM53_CPU_PAIR_DOT
    if (fused_up && params->threadpool->strata_gu_fusion_capture) {
        struct strata_cpu_gu_fusion_capture * capture=params->threadpool->strata_gu_fusion_capture;
        if (capture->gate==src0 && capture->up==fused_up->src[0]) {
            struct strata_cpu_gu_fusion_stats * value=&capture->worker[ith];
            value->quantized_blocks += ne12;
            if(ith==0) { ++value->fused_nodes; value->mapped_routes += ids->ne[1]; }
        }
    }
#endif
#ifdef STRATA_GLM53_CPU_PAIR_DOT
    if (lut_next_node >= 0 && strata_gu_dual_lut_compute(params,dst,fused_up,fused_glu,
            matrix_row_counts,matrix_rows,params->wdata,strata_dual_workspace,lut_next_node)) return;
#endif
    for (int cur_a = 0; cur_a < n_as; ++cur_a) {
        const int64_t cne1 = matrix_row_counts[cur_a];

        if (cne1 == 0) {
            continue;
        }

        // tiled takes over if profitable for this expert (see tiled.h)
        if (ggml_compute_forward_mul_mat_id_tiled(params, dst, cur_a, cne1, (const int32_t *) &MMID_MATRIX_ROW(cur_a, 0), tiled_scratch)) {
            continue;
        }

        const struct strata_cpu_derived_expert * strata_derived_expert = NULL;
        const struct strata_cpu_iq3_r4_expert * strata_r4_expert = NULL;
#ifdef STRATA_GLM53_CPU_PAIR_DOT
        if (strata_derived_view && type == GGML_TYPE_IQ3_S)
            strata_derived_expert = strata_cpu_derived_find(strata_derived_view,src0,cur_a,ith);
        if (strata_r4_view && type == GGML_TYPE_IQ3_S)
            strata_r4_expert = strata_cpu_iq3_r4_find(strata_r4_view,src0,cur_a,ith);
#endif
        const char * src0_cur = (const char *) src0->data + cur_a * nb02;
        const void * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
        const size_t row_size = ggml_row_size(vec_dot_type, ne10);

        const int64_t nr0 = ne01;
        const int64_t nr1 = cne1;

        const bool strata_static_expert = strata_static_down && cne1 <= 2;
#ifdef STRATA_GLM53_CPU_PHASE_PROFILE
        strata_cpu_phase_down_chunk(strata_phase, ith, strata_static_expert);
#endif
        if (strata_static_expert) {
            // Same quant dot/order and rank mapping, one contiguous256-row
            // range per native worker. Tiled dispatch above remains first.
            const int64_t first = ith * 256;
            ggml_compute_forward_mul_mat_id_one_chunk(dst, src0, src1, ids, cur_a,
                first, first + 256, 0, cne1, src0_cur, matrix_rows, row_size,
                src1_cont, wdata, strata_pair, strata_wide, strata_row_dot, strata_vnni, strata_iq4_packed, strata_pair4, strata_pair4_shape, strata_phase, ith,
                strata_derived_view, strata_derived_expert, strata_r4_view, strata_r4_expert, strata_lut_view, strata_down_pair, strata_down_capture, strata_schedule);
            continue;
        }

        int chunk_size = 16;
        if (nr0 == 1 || nr1 == 1) {
            chunk_size = 64;
        }

#ifdef STRATA_GLM53_CPU_PAIR_DOT
        if((strata_pair||strata_wide)&&nr1==2)chunk_size=64;
        if(strata_schedule&&nr1==2)chunk_size=64;
#endif
        // disable for NUMA
        const bool disable_chunking = ggml_is_numa();

        int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
        int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

        if (nchunk0 * nchunk1 < nth * 4 || disable_chunking) {
            nchunk0 = nr0 > nr1 ? nth : 1;
            nchunk1 = nr0 > nr1 ? 1 : nth;
        }

        const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
        const int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

        int current_chunk = ith;

        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);

        while (current_chunk < nchunk0 * nchunk1) {
            const int64_t ith0 = current_chunk % nchunk0;
            const int64_t ith1 = current_chunk / nchunk0;

            const int64_t ir0_start = dr0 * ith0;
            const int64_t ir0_end = MIN(ir0_start + dr0, nr0);

            const int64_t ir1_start = dr1 * ith1;
            const int64_t ir1_end = MIN(ir1_start + dr1, nr1);

            ggml_compute_forward_mul_mat_id_one_chunk(
                dst, src0, src1, ids, cur_a,
                ir0_start, ir0_end, ir1_start, ir1_end,
                src0_cur, matrix_rows, row_size, src1_cont, wdata, strata_pair, strata_wide, strata_row_dot, strata_vnni, strata_iq4_packed, strata_pair4, strata_pair4_shape, strata_phase, ith,
                strata_derived_view, strata_derived_expert, strata_r4_view, strata_r4_expert, strata_lut_view, strata_down_pair, strata_down_capture, strata_schedule
            );
#ifdef STRATA_GLM53_CPU_PAIR_DOT
            if (fused_up) strata_gu_fusion_up_and_glu(params,fused_up,fused_glu,cur_a,
                ir0_start,ir0_end,ir1_start,ir1_end,matrix_rows,row_size,wdata,strata_pair);
#endif

            if (nth >= nchunk0 * nchunk1) {
                break;
            }

            current_chunk = atomic_fetch_add_explicit(current_chunk_ctr, 1, memory_order_relaxed);
        }
    }
#ifdef STRATA_GLM53_CPU_PHASE_PROFILE
    strata_cpu_phase_stamp(strata_phase, ith, 5);
#endif
}

static void ggml_compute_forward_mul_mat_id(
        const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    ggml_compute_forward_mul_mat_id_impl(params,dst,NULL,NULL,-1);
}

