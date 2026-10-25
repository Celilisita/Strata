#pragma once
// Included only by the original x86 quants TU: original blocks/grid/hsum apply.
#include "cpu_iq3_dual_lut.h"
#if defined(__AVX2__)
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

unsigned strata_iq3_dual_lut_capabilities(void) {
#if defined(__AVX2__)
    unsigned c1,b7,maxleaf;uint64_t xcr0=0;
#if defined(_MSC_VER) && !defined(__clang__)
    int regs[4];__cpuid(regs,0);maxleaf=(unsigned)regs[0];if(maxleaf<7)return 1;
    __cpuidex(regs,1,0);c1=(unsigned)regs[2];
    if((c1&0x1c000000u)==0x1c000000u)xcr0=_xgetbv(0);
    __cpuidex(regs,7,0);b7=(unsigned)regs[1];
#else
    unsigned a,b,c,d;maxleaf=__get_cpuid_max(0,NULL);if(maxleaf<7)return 1;
    __cpuid_count(1,0,a,b,c,d);c1=c;
    if((c&0x1c000000u)==0x1c000000u){unsigned lo,hi;
        __asm__ volatile("xgetbv":"=a"(lo),"=d"(hi):"c"(0));xcr0=((uint64_t)hi<<32)|lo;}
    __cpuid_count(7,0,a,b,c,d);b7=b;
#endif
    const unsigned need=(1u<<12)|(1u<<28)|(1u<<29);
    const bool cpu=(c1&need)==need&&(b7&(1u<<5));
    const bool os=(c1&0x1c000000u)==0x1c000000u&&(xcr0&6u)==6u;
    return 1u|(cpu?2u:0u)|(os?4u:0u);
#else
    return 0;
#endif
}
bool strata_iq3_dual_lut_q8_supported(const void * fullrow) {
#if defined(__AVX2__)
    if(!fullrow)return false;
    const block_q8_K * q=(const block_q8_K *)fullrow;
    for(int block=0;block<16;++block){
        if(!isfinite(q[block].d))return false;
        for(int j=0;j<256;++j)if(q[block].qs[j]==INT8_MIN)return false;
    }
    return true;
#else
    (void)fullrow;return false;
#endif
}

bool strata_iq3_dual_lut_build_block(uint32_t * table,const void * q8rep0,
        const void * q8rep1,int block,int ith,int nth) {
    if(!table||!q8rep0||!q8rep1||block<0||block>=16||nth!=16||ith<0||ith>=nth)return false;
#if defined(__AVX2__)
    const block_q8_K * q0=(const block_q8_K *)q8rep0+block;
    const block_q8_K * q1=(const block_q8_K *)q8rep1+block;
    const __m256i byte_mask=_mm256_set1_epi32(255),word_mask=_mm256_set1_epi32(65535);
    // Disjoint contiguous quartet ownership,128KiB stores/worker/block.
    for(int quartet=ith*4;quartet<(ith+1)*4;++quartet){
        const int8_t * a=q0->qs+quartet*4,*b=q1->qs+quartet*4;
        for(int grid=0;grid<512;grid+=8){
            const __m256i packed=_mm256_loadu_si256((const __m256i *)(iq3s_grid+grid));
            __m256i mag[4],da[4],db[4];
            mag[0]=_mm256_and_si256(packed,byte_mask);
            mag[1]=_mm256_and_si256(_mm256_srli_epi32(packed,8),byte_mask);
            mag[2]=_mm256_and_si256(_mm256_srli_epi32(packed,16),byte_mask);
            mag[3]=_mm256_srli_epi32(packed,24);
            __m256i va=_mm256_setzero_si256(),vb=_mm256_setzero_si256();
            for(int j=0;j<4;++j){
                const __m256i ca=_mm256_mullo_epi32(mag[j],_mm256_set1_epi32((int)a[j]));
                const __m256i cb=_mm256_mullo_epi32(mag[j],_mm256_set1_epi32((int)b[j]));
                va=_mm256_add_epi32(va,ca);vb=_mm256_add_epi32(vb,cb);
                da[j]=_mm256_slli_epi32(ca,1);db[j]=_mm256_slli_epi32(cb,1);
            }
            unsigned previous=0;
            for(unsigned step=0;step<16;++step){
                const unsigned sign=step^(step>>1);
                if(step){const unsigned changed=sign^previous;unsigned lane=0;
                    while(((changed>>lane)&1u)==0)++lane;
                    if(sign&changed){va=_mm256_sub_epi32(va,da[lane]);vb=_mm256_sub_epi32(vb,db[lane]);}
                    else {va=_mm256_add_epi32(va,da[lane]);vb=_mm256_add_epi32(vb,db[lane]);}}
                const __m256i value=_mm256_or_si256(_mm256_and_si256(va,word_mask),_mm256_slli_epi32(vb,16));
                _mm256_storeu_si256((__m256i *)(table+(size_t)quartet*8192+sign*512+grid),value);
                previous=sign;
            }
        }
    }
    return true;
#else
    return false;
#endif
}

#if defined(__AVX2__)
static inline __m256i strata_dual_lut_half(__m256i value,unsigned which) {
    return which?_mm256_srai_epi32(value,16):_mm256_srai_epi32(_mm256_slli_epi32(value,16),16);
}
static inline __m256i strata_dual_lut_indices(const block_iq3_s * weight,int group) {
    const __m256i shift=_mm256_setr_epi32(0,1,2,3,4,5,6,7);
    const __m256i signs_shift=_mm256_slli_epi32(shift,2);
    const __m256i base=_mm256_setr_epi32(0,8192,16384,24576,32768,40960,49152,57344);
    const __m256i low=_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(weight->qs+group*8)));
    const __m256i high=_mm256_slli_epi32(_mm256_and_si256(
        _mm256_srlv_epi32(_mm256_set1_epi32(weight->qh[group]),shift),_mm256_set1_epi32(1)),8);
    uint32_t signs;memcpy(&signs,weight->signs+group*4,sizeof(signs));
    const __m256i sign=_mm256_slli_epi32(_mm256_and_si256(
        _mm256_srlv_epi32(_mm256_set1_epi32((int32_t)signs),signs_shift),_mm256_set1_epi32(15)),9);
    return _mm256_add_epi32(_mm256_add_epi32(base,_mm256_set1_epi32(group*65536)),
        _mm256_or_si256(sign,_mm256_or_si256(high,low)));
}
#endif
bool strata_iq3_dual_lut_chunk(size_t rows,const void * weight_row0,size_t weight_stride_bytes,
        const uint32_t * table,const void * q8a,const void * q8b,unsigned class0,unsigned class1,
        int block,float * acc0,float * acc1,float * out0,float * out1) {
    const bool pair=q8b!=NULL;
    if(rows!=128||weight_stride_bytes!=1760||!weight_row0||!table||!q8a||!acc0||!out0||
       class0>1||(pair&&class1>1)||block<0||block>=16||
       ((acc1!=NULL)!=pair)||((out1!=NULL)!=pair))return false;
#if defined(__AVX2__)
    const block_q8_K * a=(const block_q8_K *)q8a+block;
    const block_q8_K * b=pair?(const block_q8_K *)q8b+block:NULL;
    for(size_t row=0;row<rows;++row){
        const block_iq3_s * weight=(const block_iq3_s *)((const char *)weight_row0+row*weight_stride_bytes)+block;
        __m256i even0=_mm256_setzero_si256(),odd0=_mm256_setzero_si256();
        __m256i even1=_mm256_setzero_si256(),odd1=_mm256_setzero_si256();
        for(int group=0;group<8;++group){
            // One packed lookup/index for both actual columns. Each SIMD lane
            // remains the original quartet0..7 within its32value scale group.
            const __m256i indices=strata_dual_lut_indices(weight,group);
            // Eight full32bit scalar loads retain both activation halves.
            // Index extraction and vector rebuild remain inside paid consumer.
            const int * entries=(const int *)table;
            const __m256i packed=_mm256_set_epi32(
                entries[_mm256_extract_epi32(indices,7)],entries[_mm256_extract_epi32(indices,6)],
                entries[_mm256_extract_epi32(indices,5)],entries[_mm256_extract_epi32(indices,4)],
                entries[_mm256_extract_epi32(indices,3)],entries[_mm256_extract_epi32(indices,2)],
                entries[_mm256_extract_epi32(indices,1)],entries[_mm256_extract_epi32(indices,0)]);
            const int scale=2*((weight->scales[group/2]>>(4*(group%2)))&15)+1;
            const __m256i factor=_mm256_set1_epi32(scale);
            const __m256i p0=_mm256_mullo_epi32(strata_dual_lut_half(packed,class0),factor);
            if(group&1)odd0=_mm256_add_epi32(odd0,p0);else even0=_mm256_add_epi32(even0,p0);
            if(pair){const __m256i p1=_mm256_mullo_epi32(strata_dual_lut_half(packed,class1),factor);
                if(group&1)odd1=_mm256_add_epi32(odd1,p1);else even1=_mm256_add_epi32(even1,p1);}
        }
        const float dx=GGML_CPU_FP16_TO_FP32(weight->d);
        const __m256 old0=block?_mm256_loadu_ps(acc0+row*8):_mm256_setzero_ps();
        const __m256 value0=_mm256_fmadd_ps(_mm256_set1_ps(dx*a->d),
            _mm256_cvtepi32_ps(_mm256_add_epi32(even0,odd0)),old0);
        if(block==15)out0[row]=hsum_float_8(value0);else _mm256_storeu_ps(acc0+row*8,value0);
        if(pair){const __m256 old1=block?_mm256_loadu_ps(acc1+row*8):_mm256_setzero_ps();
            const __m256 value1=_mm256_fmadd_ps(_mm256_set1_ps(dx*b->d),
                _mm256_cvtepi32_ps(_mm256_add_epi32(even1,odd1)),old1);
            if(block==15)out1[row]=hsum_float_8(value1);else _mm256_storeu_ps(acc1+row*8,value1);}
    }
    return true;
#else
    return false;
#endif
}
