#include "attention_three_product.hpp"
#include "attention_compensated_pp_k1024_ops.hpp"
#include <limits>

namespace strata::gemma4::native {
namespace {
void check_blas(cublasStatus_t status){
    if(status!=CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("Gemma324 compensated physical PP cuBLAS");
}
struct Span {const void* data;std::size_t bytes,alignment;};
void validate_spans(const Span* spans,std::size_t count){
    for(std::size_t i=0;i<count;++i){
        const auto& s=spans[i];const auto p=reinterpret_cast<std::uintptr_t>(s.data);
        if(!s.data||!s.bytes||p%s.alignment||
           s.bytes>(std::numeric_limits<std::uintptr_t>::max)()-p)
            throw std::invalid_argument("Gemma324 extent/alignment/overflow");
        for(std::size_t j=0;j<i;++j){
            const auto q=reinterpret_cast<std::uintptr_t>(spans[j].data);
            if(p<q+spans[j].bytes&&q<p+s.bytes)
                throw std::invalid_argument("Gemma324 advertised spans overlap");
        }
    }
}
void validate(const float* q,float* out,CacheView c,const int* p,int heads,
    int tokens,AttentionWorkspace w,CompensatedPpK1024Lease lease,
    CompensatedPpK1024Binding live,const CompensatedPpK1024Layout& l){
    const auto b=lease.binding;
    if(!live.owner||!live.epoch||!live.stream||b.owner!=live.owner||
       !b.epoch||b.epoch!=live.epoch||b.stream!=live.stream||heads!=32||
       c.heads!=4||c.dim!=512||c.window!=0||c.capacity<l.end||c.capacity>262144||
       !w.handle||w.bytes<l.arena_bytes||w.bytes>CompensatedPpK1024ArenaLimit||
       !w.blas_bytes||lease.dead_k_bytes<l.dead_bytes||
       lease.dead_v_bytes<l.dead_bytes||lease.dead_k_bytes>CompensatedPpK1024DeadLimit||
       lease.dead_v_bytes>CompensatedPpK1024DeadLimit)
        throw std::invalid_argument("Gemma324 fixed shape/live borrowed lease");
    const auto qbytes=std::size_t(tokens)*32*512*sizeof(float);
    const auto cachebytes=std::size_t(c.capacity)*4*16*34;
    const Span spans[]={{q,qbytes,16},{out,qbytes,16},
        {c.key,cachebytes,2},{c.value,cachebytes,2},{p,sizeof(int),alignof(int)},
        {w.data,w.bytes,16},{w.blas_data,w.blas_bytes,256},
        {lease.dead_k,lease.dead_k_bytes,16},{lease.dead_v,lease.dead_v_bytes,16}};
    validate_spans(spans,sizeof(spans)/sizeof(spans[0]));
}
// Three products retain HH/HL/LH and omit LL as a precision candidate.
// Both residual planes, F32 C and all complete-attention preparation remain.
void products(cublasHandle_t handle,cublasOperation_t op,int m,int n,int k,
    const void** a0,const void** a1,const void** b0,const void** b1,
    int lda,int ldb,void** output,int ldc){
    const float scales[]={1.f,1.f/4096.f,1.f/4096.f,1.f/16777216.f};
    const float zero=0.f,one=1.f;
    const void** aa[]={a0,a0,a1,a1};const void** bb[]={b0,b1,b0,b1};
    for(int term=0;term<3;++term)
        check_blas(cublasGemmBatchedEx(handle,op,CUBLAS_OP_N,m,n,k,
            scales+term,aa[term],CUDA_R_16F,lda,bb[term],CUDA_R_16F,ldb,
            term?&one:&zero,output,CUDA_R_32F,ldc,4,CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}
}

void three_product_physical_pp_333_attention(const float* q,float* output,CacheView c,
    const int* position,int past,int heads,int tokens,AttentionWorkspace w,
    CompensatedPpK1024Lease lease,CompensatedPpK1024Binding live){
    const auto l=compensated_pp_k1024_layout(past,tokens);
    validate(q,output,c,position,heads,tokens,w,lease,live,l);
    using namespace detail324;
    const auto stream=live.stream;auto* base=static_cast<unsigned char*>(w.data);
    auto f=[&](std::size_t at){return reinterpret_cast<float*>(base+at);};
    AttentionTile tile{reinterpret_cast<half*>(base+l.q),
        reinterpret_cast<half*>(base+l.p),f(l.work),f(l.acc),f(l.maximum),
        f(l.sum),f(l.a),f(l.b)};
    PhysicalKv kv{reinterpret_cast<half*>(base),static_cast<half*>(lease.dead_k),
        static_cast<half*>(lease.dead_v),l.end};
    auto** table=reinterpret_cast<const void**>(base+l.tables);
    PointerTables t{};
    for(int plane=0;plane<2;++plane){
        t.key[plane]=table;table+=4;t.value[plane]=table;table+=4;
        t.query[plane]=table;table+=4;t.probability[plane]=table;table+=4;
    }
    t.score_work=reinterpret_cast<void**>(base+l.tables+8*4*sizeof(void*));
    t.partial_work=reinterpret_cast<void**>(base+l.tables+9*4*sizeof(void*));
    check_blas(cublasSetStream(w.handle,stream));
    check_blas(cublasSetWorkspace(w.handle,w.blas_data,w.blas_bytes));
    check_blas(cublasSetPointerMode(w.handle,CUBLAS_POINTER_MODE_HOST));
    check_blas(cublasSetMathMode(w.handle,CUBLAS_TENSOR_OP_MATH));
    try {
        unpack_physical(kv,c,stream);
        // Q tiles stay512; the final K tile has512 or1024 causal-prefix rows.
        for(int first=0;first<tokens;first+=QueryTile){
            initialize_tile(tile,stream);pack_query(tile,q,first,stream);
            const int last=past+first+QueryTile-1;
            for(int key=0;key<=last;key+=KeyTile){
                const int keys=last-key+1<KeyTile?last-key+1:KeyTile;
                prepare_pointers(t,kv,tile,key,stream);
                products(w.handle,CUBLAS_OP_T,keys,HeadsPerKv*QueryTile,Dim,
                    t.key[0],t.key[1],t.query[0],t.query[1],Dim,Dim,t.score_work,KeyTile);
                normalize_tile(tile,position,past,first,key,keys,stream);
                products(w.handle,CUBLAS_OP_N,Dim,HeadsPerKv*QueryTile,keys,
                    t.value[0],t.value[1],t.probability[0],t.probability[1],
                    Dim,KeyTile,t.partial_work,Dim);
                accumulate_tile(tile,stream);
            }
            finish_tile(tile,output,first,stream);
        }
    } catch(...) {
        (void)cublasSetMathMode(w.handle,CUBLAS_DEFAULT_MATH);throw;
    }
    check_blas(cublasSetMathMode(w.handle,CUBLAS_DEFAULT_MATH));
    primitive::check_cuda(cudaGetLastError(),"Gemma324 compensated physical PP");
}
}
