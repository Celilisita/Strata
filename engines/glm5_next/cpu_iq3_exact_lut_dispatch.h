#pragma once
#include "cpu_iq3_exact_lut_view.h"

static bool strata_cpu_iq3_lut_dispatch(struct ggml_tensor * dst, const struct ggml_tensor * src0,
        const struct ggml_tensor * src1, const struct ggml_tensor * ids, int64_t key,
        int64_t row_begin, int64_t row_end, int64_t column_begin, int64_t column_end,
        const char * original, const struct mmid_row_mapping * matrix_rows, size_t row_size,
        bool contiguous, const void * packed_q8, const struct strata_cpu_iq3_lut_view * view, int ith) {
    if (!view || !view->counters || ith < 0 || ith >= 16) return false;
    int role = -1;
    for (int candidate = 0; candidate < 2; ++candidate) if (view->banks[candidate] == src0) role = candidate;
    if (role < 0) return false; // Down and unrelatedoperations remain original.
    const int64_t columns = column_end-column_begin, rows = row_end-row_begin;
    struct strata_cpu_iq3_lut_worker_counts * count = view->counters+ith;
    if (src0->type != GGML_TYPE_IQ3_S || src0->ne[0] != 4096 || src0->ne[1] != 2048 ||
        src0->nb[1] != 1760 || src0->nb[2] != 3604480 || key < 0 || key >= 288 ||
        original != (const char *)src0->data+(size_t)key*src0->nb[2] ||
        row_begin < 0 || row_end > 2048 || rows != 128 || columns < 1 || columns > 2) {
        count->fallback_rows[role] += rows > 0 && columns > 0 ? (uint64_t)rows*(uint64_t)columns : 1;
        return false;
    }
    const void * activation[2]; float * output[2];
    for (int64_t column = 0; column < columns; ++column) {
        const struct mmid_row_mapping mapping = MMID_MATRIX_ROW(key,column_begin+column);
        const int64_t i11 = mapping.i1%src1->ne[1], i12 = mapping.i2;
        activation[column] = (const char *)packed_q8+(contiguous || src1->type != GGML_TYPE_Q8_K ?
            (i11+i12*src1->ne[1])*row_size : i11*src1->nb[1]+i12*src1->nb[2]);
        output[column] = (float *)((char *)dst->data+mapping.i1*dst->nb[1]+i12*dst->nb[2])+row_begin;
    }
    if (!strata_iq3_exact_lut_chunk(4096,output[0],columns == 2 ? output[1] : NULL,
            original+row_begin*src0->nb[1],src0->nb[1],activation[0],columns == 2 ? activation[1] : NULL,(size_t)rows)) {
        count->fallback_rows[role] += (uint64_t)rows*(uint64_t)columns;
        return false;
    }
    if (columns == 2) count->ny2_pairs[role] += (uint64_t)rows;
    else count->ny1_rows[role] += (uint64_t)rows;
    ++count->chunk_calls[role];
    count->lut_blocks[role] += 16*(uint64_t)columns;
    return true;
}
