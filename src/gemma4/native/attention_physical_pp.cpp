#include "attention_physical_pp_ops.hpp"
#include <algorithm>

namespace strata::gemma4::native {
namespace {
void check_blas(cublasStatus_t status){
    if(status!=CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("Gemma241 physical KV PP cuBLAS attention");
}
void validate(const float* q,float* output,CacheView c,const int* position,
    int heads,int tokens,AttentionWorkspace w,PhysicalPpLease lease,
    cudaStream_t stream,const PhysicalPpLayout& l){
    if(!stream||!lease.owner||lease.stream!=stream||heads!=32||c.heads!=4||
       c.dim!=512||c.window!=0||c.capacity<l.end||c.capacity>262144||
       !w.handle||w.bytes<l.arena_bytes||!w.blas_bytes||
       lease.dead_k_bytes<l.dead_bytes||lease.dead_v_bytes<l.dead_bytes||
       lease.dead_k_bytes>PhysicalPpDeadLimit||lease.dead_v_bytes>PhysicalPpDeadLimit)
        throw std::invalid_argument("Gemma241 fixed global PP/borrowed lease contract");
    const size_t q_bytes=size_t(tokens)*heads*c.dim*sizeof(float);
    const size_t cache_bytes=size_t(c.capacity)*c.heads*(c.dim/32)*34;
    const PhysicalPpSpan spans[]={
        {q,q_bytes,16},{output,q_bytes,16},{c.key,cache_bytes,2},
        {c.value,cache_bytes,2},{position,sizeof(int),alignof(int)},
        {w.data,w.bytes,16},{w.blas_data,w.blas_bytes,256},
        {lease.dead_k,lease.dead_k_bytes,16},{lease.dead_v,lease.dead_v_bytes,16}};
    physical_pp_validate_spans(spans,sizeof(spans)/sizeof(spans[0]));
}
}

void physical_global_pp_attention(const float* q,float* output,CacheView c,
    const int* position,int past,int heads,int tokens,AttentionWorkspace w,
    PhysicalPpLease lease,cudaStream_t stream){
    const auto l=physical_pp_layout(past,tokens);
    validate(q,output,c,position,heads,tokens,w,lease,stream,l);
    using namespace detail241;
    auto* base=static_cast<unsigned char*>(w.data);
    auto floats=[&](size_t offset){return reinterpret_cast<float*>(base+offset);};
    AttentionTile tile{floats(l.scores),floats(l.partial),floats(l.acc),
        floats(l.maximum),floats(l.sum),floats(l.a),floats(l.b)};
    PhysicalKv kv{floats(0),static_cast<float*>(lease.dead_k),
        static_cast<float*>(lease.dead_v),l.end};
    PointerTables tables{reinterpret_cast<const float**>(base+l.key_table),
        reinterpret_cast<const float**>(base+l.value_table),
        reinterpret_cast<const float**>(base+l.query_table),
        reinterpret_cast<float**>(base+l.score_table),
        reinterpret_cast<float**>(base+l.partial_table)};
    check_blas(cublasSetStream(w.handle,stream));
    check_blas(cublasSetWorkspace(w.handle,w.blas_data,w.blas_bytes));
    check_blas(cublasSetPointerMode(w.handle,CUBLAS_POINTER_MODE_HOST));
    check_blas(cublasSetMathMode(w.handle,CUBLAS_DEFAULT_MATH));
    const float one=1.f,zero=0.f;
    // Physical KV is decoded once per invocation, with no Qhead expansion.
    unpack_physical(kv,c,stream);
    for(int query_first=0;query_first<tokens;query_first+=QueryTile){
        const int queries=(std::min)(QueryTile,tokens-query_first);
        const int last=past+query_first+queries-1;
        initialize_tile(tile,heads,c.dim,stream);
        for(int key_first=0;key_first<=last;key_first+=KeyTile){
            const int keys=(std::min)(KeyTile,last-key_first+1);
            prepare_pointers(tables,kv,tile,q,query_first,key_first,stream);
            check_blas(cublasSgemmBatched(w.handle,CUBLAS_OP_T,CUBLAS_OP_N,
                keys,queries,c.dim,&one,tables.key,c.dim,tables.query,heads*c.dim,
                &zero,tables.score,KeyTile,heads));
            normalize_tile(tile,c,position,query_first,key_first,keys,queries,stream);
            check_blas(cublasSgemmBatched(w.handle,CUBLAS_OP_N,CUBLAS_OP_N,
                c.dim,queries,keys,&one,tables.value,c.dim,tables.score,KeyTile,
                &zero,tables.partial,c.dim,heads));
            accumulate_tile(tile,heads,queries,c.dim,stream);
        }
        finish_tile(tile,output,heads,queries,c.dim,query_first,stream);
    }
    primitive::check_cuda(cudaGetLastError(),"Gemma241 bounded physical KV PP attention");
}
}
