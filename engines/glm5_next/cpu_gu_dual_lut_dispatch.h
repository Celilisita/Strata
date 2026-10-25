#pragma once
// Existing Gate-node continuation only. Original Q8 preparation precedes this.
#include "cpu_iq3_dual_lut.h"
enum {
    STRATA_GU_DUAL_ACC_BYTES = 2*10*2048*8*sizeof(float),
    STRATA_GU_DUAL_META_BYTES = 4096,
    STRATA_GU_DUAL_WORK_BYTES = STRATA_IQ3_DUAL_LUT_BYTES+STRATA_GU_DUAL_ACC_BYTES+STRATA_GU_DUAL_META_BYTES
};
struct strata_gu_dual_job {
    atomic_int failed;
    bool usable;
    unsigned classes;
    int representatives[2], keys[7];
    unsigned row_class[10];
};
static bool strata_gu_dual_lut_workspace_shape(const struct ggml_tensor * gate,int threads) {
    if (!gate || threads!=16 || gate->op!=GGML_OP_MUL_MAT_ID) return false;
    const struct ggml_tensor * bank=gate->src[0], *input=gate->src[1], *ids=gate->src[2];
    return bank && input && ids && bank->type==GGML_TYPE_IQ3_S && bank->ne[0]==4096 &&
        bank->ne[1]==2048 && bank->ne[2]==288 && bank->ne[3]==1 &&
        input->type==GGML_TYPE_F32 && input->ne[0]==4096 && input->ne[1]==1 &&
        input->ne[2]==10 && input->ne[3]==1 && ids->type==GGML_TYPE_I32 &&
        ids->ne[0]==1 && ids->ne[1]==10 && ids->ne[2]==1 && ids->ne[3]==1;
}
static bool strata_gu_dual_lut_job_shape(struct ggml_tensor * gate,
        const struct ggml_compute_params * params) {
    if (!params->threadpool->strata_gu_dual_lut_capability ||
        !strata_gu_dual_lut_workspace_shape(gate,params->nth) || !params->wdata) return false;
    const struct ggml_tensor * bank=gate->src[0], *input=gate->src[1], *ids=gate->src[2];
    if (sizeof(block_q8_K)!=292 || bank->nb[1]!=1760 || bank->nb[2]!=3604480 ||
        !ids->data || !ggml_is_contiguous(input) || !ggml_is_contiguous(ids)) return false;
    // Same conservative MMID planner layout; reject an old cached short plan
    // before advancing its workspace pointer or entering candidate barriers.
    size_t need=ggml_row_size(GGML_TYPE_Q8_K,ggml_nelements(input))+sizeof(int64_t);
    need+=288*sizeof(int64_t)+sizeof(int64_t);
    need+=288*10*sizeof(struct mmid_row_mapping)+sizeof(int64_t);
    need+=CACHE_LINE_SIZE*288+CACHE_LINE_SIZE;
    need=GGML_PAD(need,64)+ggml_tiled_wdata_size(16,gate);
    if (need>SIZE_MAX-STRATA_GU_DUAL_WORK_BYTES-127 ||
        params->wsize<need+STRATA_GU_DUAL_WORK_BYTES+127) return false;
    const int32_t * key=(const int32_t *)ids->data;
    int unique=0,pairs=0;
    for (int row=0;row<10;++row) {
        if (key[row]<0 || key[row]>=288) return false;
        bool prior=false;int count=0;
        for (int j=0;j<10;++j) { count+=key[j]==key[row]; if(j<row && key[j]==key[row]) prior=true; }
        if (count>2) return false;
        if (!prior) { ++unique; pairs+=count==2; }
    }
    return unique==7 && pairs==3;
}
static bool strata_gu_dual_q8_equal(const block_q8_K * a,const block_q8_K * b) {
    for (int block=0;block<16;++block)
        if (memcmp(&a[block].d,&b[block].d,sizeof(float)) || memcmp(a[block].qs,b[block].qs,256)) return false;
    return true; // bsums are dead for this original IQ3 dot, including zero blocks.
}
static struct strata_cpu_gu_fusion_stats * strata_gu_dual_stats(
        const struct ggml_compute_params * params,const struct ggml_tensor * gate,const struct ggml_tensor * up) {
    struct strata_cpu_gu_fusion_capture * c=params->threadpool->strata_gu_fusion_capture;
    return c && c->gate==gate->src[0] && c->up==up->src[0] ? &c->worker[params->ith] : NULL;
}
static void strata_gu_dual_barrier(const struct ggml_compute_params * params,
        struct strata_cpu_gu_fusion_stats * stats) {
    if (stats) ++stats->lut_barrier_arrivals;
    ggml_barrier(params->threadpool);
}
static void strata_gu_dual_fail(const struct ggml_compute_params * params,
        struct strata_gu_dual_job * job,int next_node) {
    // Only the first error writes ec; every writer publishes the same abort
    // boundary. Existing graph-tail barrier retires this node before Down.
    if (!atomic_fetch_add_explicit(&job->failed,1,memory_order_acq_rel))
        params->threadpool->ec=GGML_STATUS_FAILED;
    atomic_store_explicit(&params->threadpool->abort,next_node,memory_order_release);
}
static void strata_gu_dual_glu(const struct ggml_compute_params * params,
        struct ggml_tensor * up,struct ggml_tensor * glu,int row0,int cpurow) {
    struct ggml_tensor gate_span=*glu->src[0], up_span=*up, result_span=*glu;
    for (int axis=0;axis<4;++axis) {
        gate_span.ne[axis]=up_span.ne[axis]=result_span.ne[axis]=axis==0?128:1;
        gate_span.nb[axis]=up_span.nb[axis]=result_span.nb[axis]=axis==0?sizeof(float):128*sizeof(float);
    }
    const size_t offset=(size_t)cpurow*up->nb[2]+(size_t)row0*sizeof(float);
    gate_span.data=(char *)glu->src[0]->data+offset; up_span.data=(char *)up->data+offset;
    result_span.data=(char *)glu->data+offset;result_span.src[0]=&gate_span;result_span.src[1]=&up_span;
    struct ggml_compute_params local=*params;local.ith=0;local.nth=1;
    ggml_compute_forward_glu(&local,&result_span);
}
static bool strata_gu_dual_lut_compute(const struct ggml_compute_params * params,
        struct ggml_tensor * gate,struct ggml_tensor * up,struct ggml_tensor * glu,
        const int64_t * counts,const struct mmid_row_mapping * rows,const void * q8,
        void * workspace,int next_node) {
    uint32_t * table=(uint32_t *)workspace;
    float * accum=(float *)((char *)workspace+STRATA_IQ3_DUAL_LUT_BYTES);
    struct strata_gu_dual_job * job=(struct strata_gu_dual_job *)((char *)accum+STRATA_GU_DUAL_ACC_BYTES);
    GGML_ASSERT(sizeof(*job)<=STRATA_GU_DUAL_META_BYTES && workspace);
    const size_t q8_stride=ggml_row_size(GGML_TYPE_Q8_K,4096);
    struct strata_cpu_gu_fusion_stats * stats=strata_gu_dual_stats(params,gate,up);
    struct strata_cpu_gu_fusion_capture * capture=params->threadpool->strata_gu_fusion_capture;
    struct strata_cpu_gu_fusion_phase_worker * phase=stats && capture->phase_enabled ? &capture->phase[params->ith] : NULL;
    if (phase) phase->helper_enter_us=(uint64_t)ggml_time_us();
    if (phase) phase->class_enter_us=(uint64_t)ggml_time_us();
    if (params->ith==0) {
        memset(job,0,sizeof(*job));atomic_store_explicit(&job->failed,0,memory_order_relaxed);job->usable=true;
        for (int row=0;row<10 && job->usable;++row) {
            const block_q8_K * current=(const block_q8_K *)((const char *)q8+(size_t)row*q8_stride);
            unsigned cls=0;
            for (;cls<job->classes;++cls) {
                const block_q8_K * representative=(const block_q8_K *)((const char *)q8+(size_t)job->representatives[cls]*q8_stride);
                if (strata_gu_dual_q8_equal(current,representative)) break;
            }
            if (cls==job->classes) {
                if (cls==2 || !strata_iq3_dual_lut_q8_supported(current)) { job->usable=false;break; }
                job->representatives[job->classes++]=row;
            }
            job->row_class[row]=cls;
        }
        int unique=0;
        for (int key=0;key<288 && job->usable;++key) if (counts[key]) {
            if (unique==7 || counts[key]<1 || counts[key]>2) { job->usable=false;break; }
            job->keys[unique++]=key;
            for (int col=0;col<counts[key];++col) {
                const struct mmid_row_mapping m=rows[key*10+col];
                if (m.i1!=0 || m.i2<0 || m.i2>=10) job->usable=false;
            }
        }
        job->usable &= unique==7 && job->classes>=1 && job->classes<=2;
        if (job->classes==1) job->representatives[1]=job->representatives[0];
    }
    if (phase) phase->class_return_us=(uint64_t)ggml_time_us();
    strata_gu_dual_barrier(params,stats); // Added classification publication.
    if (!job->usable) { // No GU output/accumulator/table write committed.
        if (phase) phase->helper_return_us=(uint64_t)ggml_time_us();
        return false;
    }
    if (stats && params->ith==0) { ++stats->lut_nodes;stats->lut_classes+=job->classes;stats->lut_workspace_bytes=params->wsize; }
    const void * representative[2]={
        (const char *)q8+(size_t)job->representatives[0]*q8_stride,
        (const char *)q8+(size_t)job->representatives[1]*q8_stride};
    const int row0=params->ith*128;
    for (int block=0;block<16;++block) {
        if (stats) ++stats->lut_worker_build_calls;
        if (stats && params->ith==0) ++stats->lut_table_builds;
        if (phase) phase->block[block].build_enter_us=(uint64_t)ggml_time_us();
        if (!strata_iq3_dual_lut_build_block(table,representative[0],representative[1],block,params->ith,16)) {
            memset(table+(size_t)params->ith*4*8192,0,4*8192*sizeof(uint32_t));
            strata_gu_dual_fail(params,job,next_node);
        }
        if (phase) phase->block[block].build_return_us=(uint64_t)ggml_time_us();
        strata_gu_dual_barrier(params,stats); // Publish complete2MiB table.
        if (phase) phase->block[block].publish_return_us=(uint64_t)ggml_time_us();
        if (atomic_load_explicit(&job->failed,memory_order_acquire)) {
            if (phase) phase->helper_return_us=(uint64_t)ggml_time_us();
            return true;
        }
        bool local_ok=true;
        for (int index=0;index<7;++index) {
            const int key=job->keys[index],n=(int)counts[key];
            const int a=rows[key*10].i2,b=n==2?rows[key*10+1].i2:0;
            for (int role=0;role<2;++role) {
                struct ggml_tensor * out=role?up:gate;
                const struct ggml_tensor * bank=out->src[0];
                float * acc0=accum+((size_t)role*10+a)*2048*8+(size_t)row0*8;
                float * acc1=n==2?accum+((size_t)role*10+b)*2048*8+(size_t)row0*8:NULL;
                float * dst0=(float *)((char *)out->data+(size_t)a*out->nb[2])+row0;
                float * dst1=n==2?(float *)((char *)out->data+(size_t)b*out->nb[2])+row0:NULL;
                if (local_ok) {
                    if (stats) ++stats->lut_chunk_calls;
                    local_ok=strata_iq3_dual_lut_chunk(128,(const char *)bank->data+(size_t)key*bank->nb[2]+(size_t)row0*bank->nb[1],
                        bank->nb[1],table,(const char *)q8+(size_t)a*q8_stride,
                        n==2?(const char *)q8+(size_t)b*q8_stride:NULL,job->row_class[a],job->row_class[b],block,
                        acc0,acc1,dst0,dst1);
                    if (!local_ok) strata_gu_dual_fail(params,job,next_node);
                }
            }
            if (block==15 && local_ok) {
                strata_gu_dual_glu(params,up,glu,row0,a);
                if (n==2) strata_gu_dual_glu(params,up,glu,row0,b);
                if (stats) {
                    ++stats->chunks;stats->gate_rows+=128*n;stats->up_rows+=128*n;stats->glu_elements+=128*n;
                    stats->ny1_chunks+=n==1;stats->ny2_chunks+=n==2;
                }
            }
        }
        if (phase) phase->block[block].consume_return_us=(uint64_t)ggml_time_us();
        if (block<15) {
            strata_gu_dual_barrier(params,stats); // All readers retire before overwrite.
            if (phase) phase->block[block].retire_return_us=(uint64_t)ggml_time_us();
            if (atomic_load_explicit(&job->failed,memory_order_acquire)) {
                if (phase) phase->helper_return_us=(uint64_t)ggml_time_us();
                return true;
            }
        }
        // Finalblock errors publish the same abort before the existing graph
        // tail barrier. Every worker returns normally; no late stock replay.
    }
    if (phase) phase->helper_return_us=(uint64_t)ggml_time_us();
    return true;
}
