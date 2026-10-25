#pragma once
#include "cpu_derived_view.h"

// Included after the existing expert row mapping in ggml-cpu.c.
static bool strata_cpu_derived_chunk(struct ggml_tensor * dst, const struct ggml_tensor * src0,
        const struct ggml_tensor * src1, const struct ggml_tensor * ids, int64_t key, int64_t row_begin, int64_t row_end,
        int64_t column_begin, int64_t column_end, const char * original,
        const struct mmid_row_mapping * matrix_rows, size_t row_size, bool contiguous,
        const void * packed, const struct strata_cpu_derived_view * view,
        const struct strata_cpu_derived_expert * expert, bool pair, int ith) {
    if (!expert) return false;
    const int64_t columns = column_end-column_begin;
    if (!view || !view->counters || ith < 0 || ith >= 16 || expert->role < 0 || expert->role >= 2 ||
        expert->original != original || src0->ne[0] != 4096 || src0->nb[1] != 1760 ||
        row_begin < 0 || row_end > 2048 || row_begin >= row_end || columns < 1 || columns > 2) return false;
    const void * activation[2]; float * output[2];
    for (int64_t column = 0; column < columns; ++column) {
        const struct mmid_row_mapping mapping = MMID_MATRIX_ROW(key,column_begin+column);
        const int64_t i11 = mapping.i1%src1->ne[1], i12 = mapping.i2;
        activation[column] = (const char *) packed + (contiguous || src1->type != GGML_TYPE_Q8_K ?
            (i11+i12*src1->ne[1])*row_size : i11*src1->nb[1]+i12*src1->nb[2]);
        output[column] = (float *) ((char *) dst->data+mapping.i1*dst->nb[1]+i12*dst->nb[2]);
    }
    struct strata_cpu_derived_worker_counts * counts = &view->counters[ith];
    for (int64_t row = row_begin; row < row_end; ++row) {
        const struct strata_iq3_derived_row values = strata_cpu_derived_row(expert,(size_t) row);
        if (columns == 2 && pair) {
            if (!strata_iq3_derived_dot_pair(4096,output[0]+row,output[1]+row,values,activation[0],activation[1])) {
                counts->fallback_rows[expert->role] += (uint64_t) (row_end-row_begin)*2;
                return false;
            }
        } else for (int64_t column = 0; column < columns; ++column) {
            if (!strata_iq3_derived_dot_single(4096,output[column]+row,values,activation[column])) {
                counts->fallback_rows[expert->role] += (uint64_t) (row_end-row_begin)*columns;
                return false;
            }
        }
    }
    if (columns == 2 && pair) counts->pair_rows[expert->role] += (uint64_t) (row_end-row_begin);
    else counts->single_rows[expert->role] += (uint64_t) (row_end-row_begin)*columns;
    return true;
}
