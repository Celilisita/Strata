#pragma once
#include "cpu_iq3_r4_view.h"

// Uses the current GGML expert/rank mapping, task boundaries and canonical Q8_K.
static bool strata_cpu_iq3_r4_chunk(struct ggml_tensor * dst, const struct ggml_tensor * src0,
        const struct ggml_tensor * src1, const struct ggml_tensor * ids, int64_t key,
        int64_t row_begin, int64_t row_end, int64_t column_begin, int64_t column_end,
        const char * original, const struct mmid_row_mapping * matrix_rows, size_t row_size,
        bool contiguous, const void * packed_q8, const struct strata_cpu_iq3_r4_view * view,
        const struct strata_cpu_iq3_r4_expert * expert, int ith) {
    if (!expert) return false;
    const int64_t columns = column_end-column_begin;
    if (!view || !view->counters || ith < 0 || ith >= 16 || expert->role < 0 || expert->role >= 2) return false;
    struct strata_cpu_iq3_r4_worker_counts * counts = view->counters+ith;
    if (expert->original != original || src0->ne[0] != 4096 || src0->ne[1] != 2048 ||
        src0->nb[1] != 1760 || row_begin < 0 || row_end > 2048 || row_begin >= row_end ||
        row_begin%4 || row_end%4 || columns < 1 || columns > 2) {
        counts->fallback_rows[expert->role] += (uint64_t)(row_end-row_begin)*(uint64_t)columns;
        return false;
    }
    const void * activation[2];
    float * output[2];
    for (int64_t column = 0; column < columns; ++column) {
        const struct mmid_row_mapping mapping = MMID_MATRIX_ROW(key,column_begin+column);
        const int64_t i11 = mapping.i1%src1->ne[1], i12 = mapping.i2;
        activation[column] = (const char *)packed_q8+(contiguous || src1->type != GGML_TYPE_Q8_K ?
            (i11+i12*src1->ne[1])*row_size : i11*src1->nb[1]+i12*src1->nb[2]);
        output[column] = (float *)((char *)dst->data+mapping.i1*dst->nb[1]+i12*dst->nb[2]);
    }
    for (int64_t row = row_begin; row < row_end; row += 4) {
        const struct strata_iq3_r4_rows group = strata_cpu_iq3_r4_group(expert,(size_t)row);
        if (!strata_iq3_r4_dot(4096,output[0]+row,sizeof(float),
                columns == 2 ? output[1]+row : NULL,columns == 2 ? sizeof(float) : 0,
                group,activation[0],columns == 2 ? activation[1] : NULL)) {
            counts->fallback_rows[expert->role] += (uint64_t)(row_end-row_begin)*(uint64_t)columns;
            return false;
        }
    }
    if (columns == 2) counts->ny2_pairs[expert->role] += (uint64_t)(row_end-row_begin);
    else counts->ny1_rows[expert->role] += (uint64_t)(row_end-row_begin);
    return true;
}
