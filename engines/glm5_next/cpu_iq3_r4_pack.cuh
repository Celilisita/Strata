#pragma once
#include "cpu_iq3_r4.h"
#include "cpu_iq3_r4_license.h"

_Static_assert(sizeof(struct strata_iq3_r4_block) == 4*sizeof(block_iq3_s), "R4 preserves IQ3_S bytes");
_Static_assert(sizeof(struct strata_iq3_r4_block) == 440, "R4 block padding differs");

bool strata_iq3_r4_payload_bytes(size_t rows, size_t row_blocks, size_t * bytes) {
    if (!bytes || !rows || rows > 2048 || rows%4 || !row_blocks || row_blocks > 16) return false;
    *bytes = rows*row_blocks*sizeof(block_iq3_s);
    return true;
}
static bool strata_iq3_r4_ranges(const void * original, const void * packed, size_t bytes) {
    const uintptr_t a = (uintptr_t) original, b = (uintptr_t) packed;
    return a && b && a <= UINTPTR_MAX-bytes && b <= UINTPTR_MAX-bytes &&
           (a+bytes <= b || b+bytes <= a);
}
bool strata_iq3_r4_pack(const void * original, size_t rows, size_t row_blocks,
                      void * packed, size_t packed_bytes) {
    size_t bytes;
    if (!strata_iq3_r4_payload_bytes(rows,row_blocks,&bytes) || bytes != packed_bytes ||
        !strata_iq3_r4_ranges(original,packed,bytes)) return false;
    const block_iq3_s * x = original;
    struct strata_iq3_r4_block * y = packed;
    for (size_t row = 0; row < rows; row += 4) {
        for (size_t block = 0; block < row_blocks; ++block) {
            struct strata_iq3_r4_block * dst = y+block;
            memset(dst,0,sizeof(*dst));
            for (size_t k = 0; k < 4; ++k) {
                const block_iq3_s * src = x+k*row_blocks+block;
                dst->d[k] = src->d;
                for (size_t ib = 0; ib < 4; ++ib) {
                    const size_t j = 8*ib+k;
                    dst->scales[j%16] |= (src->scales[ib]&15) << (4*(j/16));
                    dst->scales[(j+4)%16] |= (src->scales[ib]>>4) << (4*((j+4)/16));
                }
                for (size_t ib = 0; ib < 8; ++ib) {
                    dst->qh[4*ib+k] = src->qh[ib];
                    for (size_t i = 0; i < 4; ++i) {
                        dst->qs[32*ib+k+8*i] = src->qs[8*ib+i];
                        dst->qs[32*ib+k+8*i+4] = src->qs[8*ib+i+4];
                        uint8_t signs = 0;
                        for (size_t q = 0; q < 4; ++q) {
                            signs |= ((src->signs[4*ib+q]>>i)&1) << (2*q);
                            signs |= ((src->signs[4*ib+q]>>(4+i))&1) << (2*q+1);
                        }
                        dst->signs[16*ib+4*k+i] = signs;
                    }
                }
            }
        }
        x += 4*row_blocks;
        y += row_blocks;
    }
    return true;
}
bool strata_iq3_r4_identity(const void * original, size_t rows, size_t row_blocks,
                          const void * packed, size_t packed_bytes) {
    size_t bytes;
    if (!strata_iq3_r4_payload_bytes(rows,row_blocks,&bytes) || bytes != packed_bytes ||
        !strata_iq3_r4_ranges(original,packed,bytes)) return false;
    const block_iq3_s * x = original;
    const struct strata_iq3_r4_block * y = packed;
    for (size_t row = 0; row < rows; row += 4) {
        for (size_t block = 0; block < row_blocks; ++block) for (size_t k = 0; k < 4; ++k) {
            const block_iq3_s * src = x+k*row_blocks+block;
            const struct strata_iq3_r4_block * dst = y+block;
            if (src->d != dst->d[k]) return false;
            for (size_t ib = 0; ib < 8; ++ib) {
                const size_t s = 4*ib+k;
                if (((src->scales[ib/2]>>(4*(ib%2)))&15) !=
                    ((dst->scales[s%16]>>(4*(s/16)))&15) || src->qh[ib] != dst->qh[4*ib+k]) return false;
                for (size_t i = 0; i < 8; ++i) {
                    if (src->qs[8*ib+i] != dst->qs[32*ib+k+8*(i%4)+4*(i/4)]) return false;
                    for (size_t j = 0; j < 4; ++j) {
                        const size_t original_bit = 4*i+j;
                        const unsigned a = (src->signs[4*ib+original_bit/8]>>(original_bit%8))&1;
                        const unsigned b = (dst->signs[16*ib+4*k+j]>>i)&1;
                        if (a != b) return false;
                    }
                }
            }
        }
        x += 4*row_blocks;
        y += row_blocks;
    }
    return true;
}
