#pragma once
// MMID_MATRIX_ROW needs the explicit ids argument below.
static inline bool strata_cpu_iq4_down_pair_chunk(struct ggml_tensor * dst,
        const struct ggml_tensor * src0, const struct ggml_tensor * src1,
        const struct ggml_tensor * ids, int64_t expert, int64_t first, int64_t end,
        int64_t col_first, int64_t col_end, const char * weights,
        const struct mmid_row_mapping * matrix_rows, size_t row_size, bool contiguous,
        const void * packed, bool enabled, struct strata_iq4_down_pair_stats * stats, bool verify) {
    const int columns = (int)(col_end-col_first);
    if (!enabled || columns != 2) {
        if (stats) { ++stats->scalar_chunks; stats->scalar_output_rows += (end-first)*columns; }
        return false;
    }
    const void * activation[2]; float * output[2];
    for (int p = 0; p < 2; ++p) {
        const struct mmid_row_mapping row = MMID_MATRIX_ROW(expert,col_first+p);
        const int64_t i11 = row.i1%src1->ne[1], i12 = row.i2;
        activation[p] = (const char *)packed+(contiguous || src1->type != GGML_TYPE_Q8_K ?
            (i11+i12*src1->ne[1])*row_size : i11*src1->nb[1]+i12*src1->nb[2]);
        output[p] = (float *)((char *)dst->data+row.i1*dst->nb[1]+i12*dst->nb[2]);
    }
    double error2 = 0, reference2 = 0, maximum = 0, reference_maximum = 0;
    uint64_t nonfinite = 0;
    for (int64_t row = first; row < end; ++row) {
        strata_glm_cpu_iq4_down_pair((int)src0->ne[0],output[0]+row,output[1]+row,
            weights+row*src0->nb[1],activation[0],activation[1]);
        if (stats && verify) {
            float reference[2];
            type_traits_cpu[GGML_TYPE_IQ4_XS].vec_dot((int)src0->ne[0],reference,0,
                weights+row*src0->nb[1],0,activation[0],0,1);
            type_traits_cpu[GGML_TYPE_IQ4_XS].vec_dot((int)src0->ne[0],reference+1,0,
                weights+row*src0->nb[1],0,activation[1],0,1);
            for (int p = 0; p < 2; ++p) {
                const double a = output[p][row], b = reference[p];
                if (!isfinite(a) || !isfinite(b)) { ++nonfinite; continue; }
                const double delta = a-b; error2 += delta*delta; reference2 += b*b;
                if (fabs(delta) > maximum) maximum = fabs(delta);
                if (fabs(b) > reference_maximum) reference_maximum = fabs(b);
            }
        }
    }
    if (stats) {
        ++stats->paired_chunks; stats->paired_weight_rows += end-first;
        stats->paired_output_rows += 2*(end-first); stats->nonfinite_outputs += nonfinite;
        stats->error_squared += error2; stats->reference_squared += reference2;
        if (maximum > stats->max_abs_error) stats->max_abs_error = maximum;
        if (reference_maximum > stats->reference_max_abs) stats->reference_max_abs = reference_maximum;
    }
    return true;
}
