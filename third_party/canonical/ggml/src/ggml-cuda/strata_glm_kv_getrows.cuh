#pragma once
#include "strata_glm_kv_window.cuh"

// Registered F16 keeps its scalar cast. Q8_0 reads the original packed block
// from the resident row or full authoritative history and uses canonical d*qs.
// All unregistered tensors continue through getrows.cu unchanged.
template<typename dst_t, bool q8>
static __global__ void strata_glm_kv_get_rows_kernel(
        StrataGlmKvWindow view, int64_t base, const int32_t* indices, dst_t* dst,
        int64_t columns, int64_t rows, int64_t groups, int64_t batches,
        size_t nb01, size_t nb02, size_t nb03,
        size_t ib1, size_t ib2, size_t db1, size_t db2, size_t db3) {
    ggml_cuda_pdl_sync();
    for (int64_t z=blockIdx.z; z<groups*batches; z+=gridDim.z) {
        const int64_t group=z/batches, batch=z%batches;
        const int64_t row=blockIdx.x;
        const int32_t selected=*(const int32_t*)((const char*)indices+row*sizeof(int32_t)+group*ib1+batch*ib2);
        dst_t* out=(dst_t*)((char*)dst+row*db1+group*db2+batch*db3);
        const int64_t offset=base+int64_t(selected)*nb01+group*nb02+batch*nb03;
        for (int64_t col=blockIdx.y*blockDim.x+threadIdx.x; col<columns; col+=gridDim.y*blockDim.x) {
            if constexpr(q8) {
                const int64_t at=offset+(col/QK8_0)*sizeof(block_q8_0);
                assert(at>=0 && at+int64_t(sizeof(block_q8_0))<=view.logical_bytes);
                const auto* packed=(const block_q8_0*)strata_glm_kv_read_ptr(view,at);
                const float value=__half2float(packed->d)*packed->qs[col%QK8_0];
                out[col]=ggml_cuda_cast<dst_t>(value);
            } else {
                const int64_t at=offset+col*sizeof(half);
                assert(at>=0 && at+int64_t(sizeof(half))<=view.logical_bytes);
                const half value=*(const half*)strata_glm_kv_read_ptr(view,at);
                out[col]=ggml_cuda_cast<dst_t>(value);
            }
        }
    }
    GGML_UNUSED(rows);
}

template<typename dst_t, bool q8>
static void strata_glm_kv_get_rows_launch(ggml_backend_cuda_context& ctx,
        const ggml_tensor* src, const ggml_tensor* indices, ggml_tensor* dst,
        const StrataGlmKvWindow& view, int64_t base) {
    const dim3 block(256);
    const dim3 grid(indices->ne[0],MIN((src->ne[0]+255)/256,int64_t(UINT16_MAX)),
                    MIN(indices->ne[1]*indices->ne[2],int64_t(UINT16_MAX)));
    strata_glm_kv_get_rows_kernel<dst_t,q8><<<grid,block,0,ctx.stream()>>>(view,base,
        (const int32_t*)indices->data,(dst_t*)dst->data,
        src->ne[0],indices->ne[0],indices->ne[1],indices->ne[2],
        src->nb[1],src->nb[2],src->nb[3],indices->nb[1],indices->nb[2],dst->nb[1],dst->nb[2],dst->nb[3]);
}

template<typename dst_t>
static void strata_glm_kv_get_rows_dispatch(ggml_backend_cuda_context& ctx,
        const ggml_tensor* src,const ggml_tensor* indices,ggml_tensor* dst,
        const StrataGlmKvWindow& view,int64_t base) {
    if(src->type==GGML_TYPE_Q8_0)
        strata_glm_kv_get_rows_launch<dst_t,true>(ctx,src,indices,dst,view,base);
    else
        strata_glm_kv_get_rows_launch<dst_t,false>(ctx,src,indices,dst,view,base);
}

static bool strata_glm_kv_get_rows(ggml_backend_cuda_context& ctx,ggml_tensor* dst) {
    const ggml_tensor* src=dst->src[0];const ggml_tensor* indices=dst->src[1];
    StrataGlmKvWindow view{};
    if (!strata_glm_kv_window_view(src,view)) return false;
    GGML_ASSERT((src->type==GGML_TYPE_F16||src->type==GGML_TYPE_Q8_0) &&
        src->type==view.type && src->nb[0]==ggml_type_size(src->type));
    GGML_ASSERT(src->ne[0]%ggml_blck_size(src->type)==0);
    GGML_ASSERT(indices->type==GGML_TYPE_I32 && indices->nb[0]==sizeof(int32_t) && indices->ne[3]==1);
    GGML_ASSERT(view.host && view.device && view.tags && view.row_bytes>0 && view.resident_tokens>0 && view.logical_bytes>0);
    const int64_t base=(const unsigned char*)src->data-view.host;
    GGML_ASSERT(base>=0 && base<view.logical_bytes && dst->nb[0]==ggml_type_size(dst->type));
    GGML_ASSERT(base%ggml_type_size(src->type)==0);
    switch(dst->type) {
        case GGML_TYPE_F32:strata_glm_kv_get_rows_dispatch<float>(ctx,src,indices,dst,view,base);break;
        case GGML_TYPE_F16:strata_glm_kv_get_rows_dispatch<half>(ctx,src,indices,dst,view,base);break;
        case GGML_TYPE_I32:strata_glm_kv_get_rows_dispatch<int32_t>(ctx,src,indices,dst,view,base);break;
        case GGML_TYPE_BF16:strata_glm_kv_get_rows_dispatch<nv_bfloat16>(ctx,src,indices,dst,view,base);break;
        default:GGML_ABORT("unsupported registered GLM KV GET_ROWS destination");
    }
    return true;
}
