#pragma once
// Included after GGML's original mmid_row_mapping; no chunk/rank reordering.
static inline bool strata_cpu_row_dot_chunk(struct ggml_tensor* dst,const struct ggml_tensor* src0,
        const struct ggml_tensor* src1,const struct ggml_tensor* ids,int64_t expert,
        int64_t first,int64_t end,int64_t col_first,int64_t col_end,const char* weights,
        const struct mmid_row_mapping* map,size_t row_size,bool contiguous,const void* packed,
        bool enabled,bool pair,bool wide,int phase,int worker) {
    const int columns=(int)(col_end-col_first);
    if(!enabled||wide||end<=first||columns<1||columns>2)return false;
    const void* activation[2];float* output[2];
    for(int col=0;col<columns;++col) {
        const struct mmid_row_mapping row=map[expert*ids->ne[0]*ids->ne[1]+col_first+col];
        const int64_t i11=row.i1%src1->ne[1],i12=row.i2;
        activation[col]=(const char*)packed+(contiguous||src1->type!=GGML_TYPE_Q8_K?
            (i11+i12*src1->ne[1])*row_size:i11*src1->nb[1]+i12*src1->nb[2]);
        output[col]=(float*)((char*)dst->data+row.i1*dst->nb[1]+i12*dst->nb[2])+first;
    }
    uint64_t counts=0;
    if(columns==2&&pair)
        counts=strata_glm_cpu_row_dot((int)src0->ne[0],output[0],output[1],weights+first*src0->nb[1],
            src0->nb[1],activation[0],activation[1],(int)(end-first));
    else for(int col=0;col<columns;++col)
        counts+=strata_glm_cpu_row_dot((int)src0->ne[0],output[col],NULL,weights+first*src0->nb[1],
            src0->nb[1],activation[col],NULL,(int)(end-first));
#ifdef STRATA_GLM53_CPU_PHASE_PROFILE
    strata_cpu_phase_row_dot(phase,worker,columns==2&&pair?2:1,(uint32_t)counts,counts>>32);
#else
    GGML_UNUSED(phase);GGML_UNUSED(worker);GGML_UNUSED(counts);
#endif
    return true;
}
