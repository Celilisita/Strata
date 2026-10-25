#pragma once
// Included after the original one_chunk helper. No dot/activation math is copied.
#include "cpu_gu_fusion.h"
static bool strata_gu_disjoint(const struct ggml_tensor * a, const struct ggml_tensor * b) {
    const uintptr_t x=(uintptr_t)a->data, y=(uintptr_t)b->data;
    const size_t nx=ggml_nbytes(a), ny=ggml_nbytes(b);
    return x && y && nx <= UINTPTR_MAX-x && ny <= UINTPTR_MAX-y && (x+nx <= y || y+ny <= x);
}
static bool strata_gu_fusion_pattern(const struct ggml_cgraph * graph,int first,
        const struct ggml_compute_params * params) {
    const struct strata_glm_cpu_dot_options * option=&params->threadpool->strata_dot_options;
    if (!option->graph_snapshot || !option->gu_fusion || params->nth != 16 || params->use_ref ||
        !option->pair || option->wide || option->row_dot || option->vnni_dot || option->pair4 ||
        option->iq4_packed || option->iq4_down_pair || option->pair_schedule || option->static_down ||
        params->threadpool->strata_derived_view || params->threadpool->strata_iq3_r4_view ||
        params->threadpool->strata_exact_lut_view || first+2 >= graph->n_nodes) return false;
#ifdef STRATA_GLM53_CPU_PHASE_PROFILE
    if (strata_cpu_phase_enabled) return false; // Old role clocks do not describe fusion.
#endif
    const struct ggml_tensor * gate=graph->nodes[first], *up=graph->nodes[first+1], *glu=graph->nodes[first+2];
    if (gate->op != GGML_OP_MUL_MAT_ID || up->op != GGML_OP_MUL_MAT_ID || glu->op != GGML_OP_GLU ||
        glu->src[0] != gate || glu->src[1] != up || ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU_CLAMP ||
        ggml_get_op_params_i32(glu,1) != 0 || ggml_get_op_params_f32(glu,3) != 10.0f ||
        !ggml_node_has_n_uses(graph,first,1) || !ggml_node_has_n_uses(graph,first+1,1) ||
        !(gate->flags & up->flags & glu->flags & GGML_TENSOR_FLAG_COMPUTE) ||
        gate->view_src || up->view_src || glu->view_src ||
        !ggml_are_same_shape(gate,up) || !ggml_are_same_shape(gate,glu) ||
        !ggml_is_contiguous(gate) || !ggml_is_contiguous(up) || !ggml_is_contiguous(glu) ||
        gate->type != GGML_TYPE_F32 || up->type != GGML_TYPE_F32 || glu->type != GGML_TYPE_F32 ||
        !strata_gu_disjoint(gate,up) || !strata_gu_disjoint(gate,glu) || !strata_gu_disjoint(up,glu)) return false;
    const struct ggml_tensor * weight=gate->src[0], *other=up->src[0], *input=gate->src[1], *ids=gate->src[2];
    if (!weight || !other || weight==other || !input || !ids || up->src[1]!=input || up->src[2]!=ids ||
        weight->type != GGML_TYPE_IQ3_S || other->type != GGML_TYPE_IQ3_S ||
        weight->ne[0]!=4096 || weight->ne[1]!=2048 || weight->ne[2]!=288 || weight->ne[3]!=1 ||
        !ggml_are_same_shape(weight,other) || !ggml_are_same_stride(weight,other) ||
        !ggml_is_contiguous(weight) || !ggml_is_contiguous(other) || weight->extra || other->extra ||
        !weight->buffer || !other->buffer || !weight->data || !other->data || !ggml_backend_buffer_is_host(weight->buffer) ||
        !ggml_backend_buffer_is_host(other->buffer) || input->type!=GGML_TYPE_F32 ||
        !ggml_is_contiguous(input) || input->ne[0]!=4096 || input->ne[1]!=1 ||
        input->ne[2]<1 || input->ne[2]>16 || input->ne[3]!=1 || !input->data ||
        ids->type!=GGML_TYPE_I32 || !ggml_is_contiguous(ids) || !ids->data ||
        ids->ne[0]!=1 || ids->ne[1]!=input->ne[2] || ids->ne[2]!=1 || ids->ne[3]!=1 ||
        gate->ne[0]!=2048 || gate->ne[1]!=1 || gate->ne[2]!=input->ne[2] || gate->ne[3]!=1) return false;
    // All workers read the same immutable job IDs before any barrier/write.
    // Max two columns per key excludes tiled's batch>=8 path, and native
    // caller disables this option if the *actual cached* tiled-force is true.
    const int32_t * keys=(const int32_t *)ids->data;
    for (int64_t i=0;i<ids->ne[1];++i) {
        if (keys[i]<0 || keys[i]>=288) return false;
        int count=1;
        for (int64_t j=0;j<i;++j) count += keys[j]==keys[i];
        if (count>2) return false;
    }
    return true;
}
static void strata_gu_fusion_up_and_glu(const struct ggml_compute_params * params,
        struct ggml_tensor * up, struct ggml_tensor * glu,int64_t key,
        int64_t r0,int64_t r1,int64_t c0,int64_t c1,
        const struct mmid_row_mapping * matrix_rows,size_t row_size,const void * q8,bool pair) {
    const struct ggml_tensor * weight=up->src[0], *input=up->src[1], *ids=up->src[2];
    const char * bank=(const char *)weight->data+key*weight->nb[2];
    ggml_compute_forward_mul_mat_id_one_chunk(up,weight,input,ids,key,r0,r1,c0,c1,bank,
        matrix_rows,row_size,true,q8,pair,false,false,false,false,false,false,-1,params->ith,
        NULL,NULL,NULL,NULL,NULL,false,NULL,false);
    // Local descriptors keep the original op parameters/float implementation.
    // Each span is written by this worker's two dots before it is consumed.
    struct ggml_tensor gate_span=*glu->src[0], up_span=*up, result_span=*glu;
    const int64_t width=r1-r0;
    for (int axis=0;axis<4;++axis) {
        gate_span.ne[axis]=up_span.ne[axis]=result_span.ne[axis]=axis==0?width:1;
        gate_span.nb[axis]=up_span.nb[axis]=result_span.nb[axis]=axis==0?sizeof(float):width*sizeof(float);
    }
    result_span.src[0]=&gate_span; result_span.src[1]=&up_span;
    struct ggml_compute_params local=*params; local.ith=0; local.nth=1;
    for (int64_t column=c0;column<c1;++column) {
        const struct mmid_row_mapping map=matrix_rows[key*ids->ne[0]*ids->ne[1]+column];
        const size_t offset=(size_t)map.i1*up->nb[1]+(size_t)map.i2*up->nb[2]+(size_t)r0*sizeof(float);
        gate_span.data=(char *)glu->src[0]->data+offset;
        up_span.data=(char *)up->data+offset; result_span.data=(char *)glu->data+offset;
        ggml_compute_forward_glu(&local,&result_span);
    }
    struct strata_cpu_gu_fusion_capture * capture=params->threadpool->strata_gu_fusion_capture;
    if (capture && capture->gate==glu->src[0]->src[0] && capture->up==weight) {
        struct strata_cpu_gu_fusion_stats * value=&capture->worker[params->ith];
        const uint64_t elements=(uint64_t)width*(uint64_t)(c1-c0);
        ++value->chunks; value->gate_rows+=elements; value->up_rows+=elements; value->glu_elements+=elements;
        value->ny1_chunks += c1-c0==1; value->ny2_chunks += c1-c0==2;
    }
}
