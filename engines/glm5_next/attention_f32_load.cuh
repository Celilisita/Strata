#pragma once
// Same shared-tile layout as the existing F32 tile kernel. Each source
// float2 contains two original F32 values, without a half conversion.
template<int Warp,int Warps,int Rows,int Columns,int Padding,bool Check>
static __device__ __forceinline__ void flash_attn_tile_load_tile(
    const float2* __restrict__ source,float* __restrict__ tile,int stride,int limit,const int32_t* indices=nullptr) {
    for(int row=threadIdx.y;row<Rows;row+=Warps) {
        for(int col=2*threadIdx.x;col<Columns;col+=2*Warp) {
            const auto values=(!Check||row<limit)?source[(indices?indices[row]:row)*stride+col/2]:make_float2(0.f,0.f);
            tile[row*(Columns+Padding)+col]=values.x;
            tile[row*(Columns+Padding)+col+1]=values.y;
        }
    }
}

// Only the private F32-compute tile calls this overload. Sparse rows retain
// their original compacted IDs; stored halves widen exactly before F32 math.
template<int Warp,int Warps,int Rows,int Columns,int Padding,bool Check>
static __device__ __forceinline__ void flash_attn_tile_load_tile(
    const half2* __restrict__ source,float* __restrict__ tile,int stride,int limit,const int32_t* indices) {
    const auto* window=strata_glm_kv_attention_window(source);
    for(int row=threadIdx.y;row<Rows;row+=Warps) {
        for(int col=2*threadIdx.x;col<Columns;col+=2*Warp) {
            const half2* from=source;
            if(!Check||row<limit) {
                from=source+int64_t(indices?indices[row]:row)*stride+col/2;
                if(window)from=reinterpret_cast<const half2*>(strata_glm_kv_read_ptr(*window,
                    reinterpret_cast<const unsigned char*>(from)-window->host));
            }
            const auto values=(!Check||row<limit)?__half22float2(*from):make_float2(0.f,0.f);
            tile[row*(Columns+Padding)+col]=values.x;
            tile[row*(Columns+Padding)+col+1]=values.y;
        }
    }
}

// T_KV is the actual canonical34-byte block: stride is blocks/row and
// the K iterator already advances its column origin in32-value blocks.
template<int Warp,int Warps,int Rows,int Columns,int Padding,bool Check>
static __device__ __forceinline__ void flash_attn_tile_load_tile(
    const block_q8_0* __restrict__ source,float* __restrict__ tile,int stride,int limit,const int32_t* indices) {
    const auto* window=strata_glm_kv_attention_window(source);
    assert(window&&window->type==GGML_TYPE_Q8_0&&window->row_elements==512);
    static_assert(Columns%QK8_0==0,"Q8 attention columns must cover whole blocks");
    for(int row=threadIdx.y;row<Rows;row+=Warps) {
        for(int col=2*threadIdx.x;col<Columns;col+=2*Warp) {
            float2 values=make_float2(0.f,0.f);
            if(!Check||row<limit) {
                const auto* from=source+int64_t(indices?indices[row]:row)*stride+col/QK8_0;
                const auto offset=reinterpret_cast<const unsigned char*>(from)-window->host;
                const auto* block=reinterpret_cast<const block_q8_0*>(strata_glm_kv_read_ptr(*window,offset));
                const float d=block->d;
                values.x=d*block->qs[col%QK8_0];values.y=d*block->qs[col%QK8_0+1];
            }
            tile[row*(Columns+Padding)+col]=values.x;
            tile[row*(Columns+Padding)+col+1]=values.y;
        }
    }
}

// Existing native compactor stores bounded per-query lists followed by counts.
// Every finite mask entry is retained; the graph passes the true selection bound.
static __device__ __forceinline__ const int32_t* strata_glm_attention_indices(
    const int32_t* compact,int bound,int queries,int queries_per_tile,int sequence,int tile,int* count) {
    const int tiles=(queries+queries_per_tile-1)/queries_per_tile;
    const int64_t list=int64_t(sequence)*tiles+tile;
    *count=compact[int64_t(bound)*tiles+list];
    return compact+list*bound;
}
