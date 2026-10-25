#pragma once
// Included after GGML's existing expert row mapping. Chunk boundaries and
// original route writes stay intact; only two actual activation dots share x.
static inline bool strata_cpu_pair4_chunk(struct ggml_tensor* dst,const struct ggml_tensor* src0,
        const struct ggml_tensor* src1,const struct ggml_tensor* ids,int64_t expert,
        int64_t first,int64_t end,int64_t col_first,int64_t col_end,const char* weights,
        const struct mmid_row_mapping* map,size_t row_size,bool contiguous,const void* packed,
        bool enabled,int phase,int worker) {
    const int columns=(int)(col_end-col_first);
    if(!enabled||end<=first||columns<2||columns>4)return false;
    const void* activation[4];float* output[4];
    for(int col=0;col<columns;++col) {
        const struct mmid_row_mapping row=map[expert*ids->ne[0]*ids->ne[1]+col_first+col];
        const int64_t i11=row.i1%src1->ne[1],i12=row.i2;
        activation[col]=(const char*)packed+(contiguous||src1->type!=GGML_TYPE_Q8_K?
            (i11+i12*src1->ne[1])*row_size:i11*src1->nb[1]+i12*src1->nb[2]);
        output[col]=(float*)((char*)dst->data+row.i1*dst->nb[1]+i12*dst->nb[2]);
    }
    for(int col=0;col+1<columns;col+=2)for(int64_t row=first;row<end;++row)
        ggml_vec_dot_iq3_s_q8_K_pair((int)src0->ne[0],output[col]+row,output[col+1]+row,
            weights+row*src0->nb[1],activation[col],activation[col+1]);
    if(columns%2) {
        ggml_vec_dot_t const dot=type_traits_cpu[GGML_TYPE_IQ3_S].vec_dot;
        for(int64_t row=first;row<end;++row)
            dot((int)src0->ne[0],output[columns-1]+row,0,weights+row*src0->nb[1],0,activation[columns-1],0,1);
    }
#ifdef STRATA_GLM53_CPU_PHASE_PROFILE
    strata_cpu_phase_pair4_rows(phase,worker,(uint64_t)(columns/2)*(end-first),
        (uint64_t)(columns%2)*(end-first),(uint64_t)(columns%2)*(end-first));
#else
    GGML_UNUSED(phase);GGML_UNUSED(worker);
#endif
    return true;
}
