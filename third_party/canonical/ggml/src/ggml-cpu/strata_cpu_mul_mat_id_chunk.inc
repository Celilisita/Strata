static void ggml_compute_forward_mul_mat_id_one_chunk(
    struct ggml_tensor * dst,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * ids,
    const int64_t cur_a,
    const int64_t ir0_start,
    const int64_t ir0_end,
    const int64_t ir1_start,
    const int64_t ir1_end,
    const char * src0_cur,
    const struct mmid_row_mapping * matrix_rows,
    const size_t row_size,
    const bool src1_cont,
    const void * wdata, const bool strata_pair, const bool strata_wide,
    const bool strata_row_dot, const bool strata_vnni, const bool strata_iq4_packed, const bool strata_pair4, const bool strata_pair4_shape,
    const int strata_phase, const int strata_ith,
    const struct strata_cpu_derived_view * strata_derived_view,
    const struct strata_cpu_derived_expert * strata_derived_expert,
    const struct strata_cpu_iq3_r4_view * strata_r4_view,
    const struct strata_cpu_iq3_r4_expert * strata_r4_expert,
    const struct strata_cpu_iq3_lut_view * strata_lut_view,
    bool strata_down_pair, struct strata_iq4_down_pair_capture * strata_down_capture,
    bool strata_schedule) {

    GGML_TENSOR_BINARY_OP_LOCALS

    const enum ggml_type type = src0->type;

    ggml_vec_dot_t const vec_dot =
#ifdef STRATA_GLM53_CPU_PAIR_DOT
        strata_iq4_packed?strata_glm_cpu_iq4_packed_dot:
        strata_vnni&&ir1_end-ir1_start==1?strata_glm_cpu_iq3_s_vnni:
        strata_wide?ggml_vec_dot_iq3_s_q8_K_wide:
#endif
        type_traits_cpu[type].vec_dot;
    enum ggml_type    const vec_dot_type = type_traits_cpu[type].vec_dot_type;

#ifdef STRATA_GLM53_CPU_PAIR_DOT
    if (strata_down_pair || strata_down_capture) {
        struct strata_iq4_down_pair_stats * stats = strata_down_capture ? &strata_down_capture->worker[strata_ith] : NULL;
        if (strata_cpu_iq4_down_pair_chunk(dst,src0,src1,ids,cur_a,ir0_start,ir0_end,ir1_start,ir1_end,
                src0_cur,matrix_rows,row_size,src1_cont,wdata,strata_down_pair,stats,
                strata_down_capture && strata_down_capture->verify)) return;
    }
    if (strata_lut_view && !strata_wide && !strata_row_dot && !strata_vnni && !strata_pair4 &&
        strata_cpu_iq3_lut_dispatch(dst,src0,src1,ids,cur_a,ir0_start,ir0_end,ir1_start,ir1_end,
            src0_cur,matrix_rows,row_size,src1_cont,wdata,strata_lut_view,strata_ith)) return;
    if (strata_r4_expert && !strata_wide && !strata_row_dot && !strata_vnni && !strata_pair4 &&
        strata_cpu_iq3_r4_chunk(dst,src0,src1,ids,cur_a,ir0_start,ir0_end,ir1_start,ir1_end,
            src0_cur,matrix_rows,row_size,src1_cont,wdata,strata_r4_view,strata_r4_expert,strata_ith)) return;
    if(strata_derived_expert && !strata_wide && !strata_row_dot && !strata_vnni && !strata_pair4 &&
        strata_cpu_derived_chunk(dst,src0,src1,ids,cur_a,ir0_start,ir0_end,ir1_start,ir1_end,
            src0_cur,matrix_rows,row_size,src1_cont,wdata,strata_derived_view,strata_derived_expert,strata_pair,strata_ith)) return;
    if(strata_cpu_pair4_chunk(dst,src0,src1,ids,cur_a,ir0_start,ir0_end,ir1_start,ir1_end,
            src0_cur,matrix_rows,row_size,src1_cont,wdata,strata_pair4,strata_phase,strata_ith))return;
    if(strata_cpu_row_dot_chunk(dst,src0,src1,ids,cur_a,ir0_start,ir0_end,ir1_start,ir1_end,
            src0_cur,matrix_rows,row_size,src1_cont,wdata,strata_row_dot,strata_pair,strata_wide,strata_phase,strata_ith))return;
    if((strata_pair||strata_wide||strata_vnni) && ir1_end-ir1_start==2) {
        const void* activation[2];float* output[2];
        for(int p=0;p<2;++p) {
            const struct mmid_row_mapping row=MMID_MATRIX_ROW(cur_a,ir1_start+p);
            const int64_t i11=row.i1%ne11,i12=row.i2;
            activation[p]=(const char*)wdata+(src1_cont||src1->type!=vec_dot_type?
                (i11+i12*ne11)*row_size:i11*nb11+i12*nb12);
            output[p]=(float*)((char*)dst->data+row.i1*nb1+i12*nb2);
        }
        if(strata_vnni) {
            for(int64_t ir0=ir0_start;ir0<ir0_end;++ir0)
                strata_glm_cpu_iq3_s_vnni_pair(ne00,output[0]+ir0,output[1]+ir0,src0_cur+ir0*nb01,activation[0],activation[1]);
#ifdef STRATA_GLM53_CPU_PHASE_PROFILE
            strata_cpu_phase_vnni_dot(strata_phase,strata_ith,2,ir0_end-ir0_start);
#endif
        } else {
            for(int64_t ir0=ir0_start;ir0<ir0_end;++ir0)
                (strata_wide?ggml_vec_dot_iq3_s_q8_K_wide_pair:ggml_vec_dot_iq3_s_q8_K_pair)(ne00,output[0]+ir0,output[1]+ir0,src0_cur+ir0*nb01,activation[0],activation[1]);
        }
        return;
    }
#else
    GGML_UNUSED(strata_pair);GGML_UNUSED(strata_wide);
    GGML_UNUSED(strata_row_dot);GGML_UNUSED(strata_vnni);GGML_UNUSED(strata_iq4_packed);GGML_UNUSED(strata_phase);GGML_UNUSED(strata_ith);
    GGML_UNUSED(strata_pair4);
    GGML_UNUSED(strata_r4_view);GGML_UNUSED(strata_r4_expert);GGML_UNUSED(strata_lut_view);
    GGML_UNUSED(strata_down_pair);GGML_UNUSED(strata_down_capture);GGML_UNUSED(strata_schedule);
#endif
    const int64_t blck_0 =
#ifdef STRATA_GLM53_CPU_PAIR_DOT
        strata_glm_cpu_r16_down_block(ir1_end-ir1_start,strata_schedule);
#else
        16;
#endif
    const int64_t blck_1 = 16;

    float tmp[16];

    for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ++ir1) {
                const int64_t _i12 = ir1; // logical row index for this expert

                struct mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, _i12);
                const int id       = row_mapping.i1; // selected expert index

                const int64_t  i11 = id % ne11;
                const int64_t  i12 = row_mapping.i2; // row index in src1

                const int64_t  i1 = id;  // selected expert index
                const int64_t  i2 = i12; // row

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                const char * src1_col = (const char *) wdata +
                    (src1_cont || src1->type != vec_dot_type
                    ? (i11      + i12*ne11)*row_size
                    : (i11*nb11 + i12*nb12));

                float * dst_col = (float *) ((char *) dst->data + (i1*nb1 + i2*nb2));

                for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                    vec_dot(ne00, &tmp[ir0 - iir0], 0, src0_cur + ir0*nb01, 0, src1_col, 0, 1);
                }

                memcpy(&dst_col[iir0], tmp, (MIN(iir0 + blck_0, ir0_end) - iir0)*sizeof(float));
            }
        }
    }
#if defined(STRATA_GLM53_CPU_PAIR_DOT) && defined(STRATA_GLM53_CPU_PHASE_PROFILE)
    if(strata_vnni&&ir1_end-ir1_start==1)strata_cpu_phase_vnni_dot(strata_phase,strata_ith,1,ir0_end-ir0_start);
    if(strata_pair4_shape)strata_cpu_phase_pair4_rows(strata_phase,strata_ith,0,
        (uint64_t)(ir0_end-ir0_start)*(ir1_end-ir1_start),0);
#else
    GGML_UNUSED(strata_pair4_shape);
#endif
}

