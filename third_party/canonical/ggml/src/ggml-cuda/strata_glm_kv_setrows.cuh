#pragma once
#include "strata_glm_kv_window.cuh"

// The original SET_ROWS kernel has already written authoritative mapped host
// storage. Refresh the complete owner row, including untouched partial columns.
// A slot lock prevents two logical rows with equal modulo from interleaving.
template<typename idx_t>
static __global__ void strata_glm_kv_refresh_rows_kernel(
        StrataGlmKvWindow view,int64_t base,const idx_t* indices,
        int64_t rows,int64_t groups,int64_t batches,int64_t index_groups,int64_t index_batches,
        size_t ib1,size_t ib2,size_t db1,size_t db2,size_t db3) {
    __shared__ int acquired;
    __shared__ int64_t logical_row;
    for (int64_t flat=blockIdx.x;flat<rows*groups*batches;flat+=gridDim.x) {
        if (threadIdx.x==0) {
            const int64_t row=flat%rows,group=(flat/rows)%groups,batch=flat/(rows*groups);
            const idx_t selected=*(const idx_t*)((const char*)indices+row*sizeof(idx_t)+(group%index_groups)*ib1+(batch%index_batches)*ib2);
            const int64_t offset=base+int64_t(selected)*db1+group*db2+batch*db3;
            assert(offset>=0 && offset<view.logical_bytes);
            logical_row=offset/view.row_bytes;
            auto* tag=(unsigned long long*)(view.tags+logical_row%view.resident_tokens);
            const unsigned long long old=atomicCAS(tag,0ULL,0ULL);
            acquired=old!=~1ULL && atomicCAS(tag,old,~1ULL)==old;
        }
        __syncthreads();
        if (acquired) {
            const int64_t slot=logical_row%view.resident_tokens;
            const unsigned char* in=view.host+logical_row*view.row_bytes;
            unsigned char* out=view.device+slot*view.row_bytes;
            for (int64_t col=threadIdx.x;col<view.row_bytes;col+=blockDim.x) out[col]=in[col];
            __syncthreads();
            if (threadIdx.x==0) {
                __threadfence();
                atomicExch((unsigned long long*)(view.tags+slot),(unsigned long long)logical_row);
            }
        }
        __syncthreads();
    }
}

static void strata_glm_kv_set_rows_refresh(ggml_backend_cuda_context& ctx,ggml_tensor* dst) {
    StrataGlmKvWindow view{};
    if (!strata_glm_kv_window_view(dst,view)) return;
    const ggml_tensor* src=dst->src[0];const ggml_tensor* indices=dst->src[1];
    GGML_ASSERT((dst->type==GGML_TYPE_F16||dst->type==GGML_TYPE_Q8_0) &&
        dst->type==view.type && dst->nb[0]==ggml_type_size(dst->type));
    GGML_ASSERT(view.host && view.device && view.tags && view.row_bytes>0 && view.resident_tokens>0 && view.logical_bytes>0);
    const int64_t base=(const unsigned char*)dst->data-view.host;
    GGML_ASSERT(base>=0 && base<view.logical_bytes && view.logical_bytes%view.row_bytes==0);
    GGML_ASSERT(src->ne[0]%ggml_blck_size(dst->type)==0);
    GGML_ASSERT(base%ggml_type_size(dst->type)==0);
    GGML_ASSERT(size_t(base%view.row_bytes)+ggml_row_size(dst->type,src->ne[0])<=size_t(view.row_bytes));
    GGML_ASSERT(dst->nb[1]%view.row_bytes==0 && dst->nb[2]%view.row_bytes==0 && dst->nb[3]%view.row_bytes==0);
    GGML_ASSERT(indices->ne[1]>0 && indices->ne[2]>0 && indices->nb[0]==ggml_type_size(indices->type));
    const int64_t rows=src->ne[1]*src->ne[2]*src->ne[3];
    if (!rows) return;
    const dim3 grid(MIN(rows,int64_t(UINT16_MAX))),block(256);
    if (indices->type==GGML_TYPE_I64) {
        strata_glm_kv_refresh_rows_kernel<int64_t><<<grid,block,0,ctx.stream()>>>(view,base,(const int64_t*)indices->data,
            src->ne[1],src->ne[2],src->ne[3],indices->ne[1],indices->ne[2],indices->nb[1],indices->nb[2],dst->nb[1],dst->nb[2],dst->nb[3]);
    } else {
        GGML_ASSERT(indices->type==GGML_TYPE_I32);
        strata_glm_kv_refresh_rows_kernel<int32_t><<<grid,block,0,ctx.stream()>>>(view,base,(const int32_t*)indices->data,
            src->ne[1],src->ne[2],src->ne[3],indices->ne[1],indices->ne[2],indices->nb[1],indices->nb[2],dst->nb[1],dst->nb[2],dst->nb[3]);
    }
}
