#pragma once
#include "strata/qwen35/completed_v100/native/recurrent.hpp"
#include "strata/qwen35/completed_v100/native/attention_prefill.hpp"
#include "strata/qwen35/completed_v100/native/attention_decode.hpp"

namespace strata::completed_v100::qwen35::native {
struct RamKvView;
struct AttentionWeights {
    Matrix q,k,v,out,ffn_gate,ffn_up,ffn_down;
    const float *norm,*q_norm,*k_norm,*post_norm;
};
struct AttentionShape { int hidden,dim,query_heads,kv_heads,rotary_dim; float epsilon,rotary_base; };
struct AttentionSchedule { ReductionSchedule projections[4]{}; PrefillRounding rounding; DecodeRounding decode_rounding; };
struct AttentionCache { void *key,*value; int start,valid,capacity;const DecodeCursor *cursor=nullptr;int graph_partitions=0;bool q8=false;const int *graph_start=nullptr;int graph_shift=0;int window=0;RamKvView *ram=nullptr;bool full_history=false; };
struct AttentionBuffers {
    float *norm,*q_raw,*k_raw,*v_raw,*q_norm,*k_norm,*q_rope,*k_rope;
    float *flash_q,*flash_k,*flash_v,*attention,*pregate,*gate,*sigmoid,*gated;
    float *projected,*residual,*post_norm,*output;
};
using PrefillAttentionKernel=void (*)(const float *,const void *,const void *,float *,void *,size_t,
                                      PrefillAttentionShape,int,cudaStream_t);
// Same signature; the integer is KV partitions for this provider.
using VerifyAttentionKernel=PrefillAttentionKernel;
// One common weight/staging/activation workspace with recurrent layers.
// Borrowed objects/cache/stream survive execution; split methods permit a
// benchmark checkpoint before the FFN overwrites its dead attention scratch.
class AttentionLayer {
public:
    AttentionLayer(RecurrentLayer &shared,AttentionWeights,AttentionShape,AttentionSchedule);
    void rebind(AttentionWeights);
    void visual_positions(const int* table){visual_positions_=table;}
    void attention(const float *hidden,const int *positions,AttentionCache,int tokens,cudaStream_t);
    void cache_only(const float *,const int *,AttentionCache,int,cudaStream_t);
    void finish(const float *hidden,int tokens,cudaStream_t);
    void execute(const float *hidden,const int *positions,AttentionCache c,int tokens,cudaStream_t s){
        attention(hidden,positions,c,tokens,s);finish(hidden,tokens,s);
    }
    AttentionBuffers buffers(int tokens) const;
    DecodeRounding decode_rounding() const { return schedule_.decode_rounding; }
    void use_pp_attention_gemm(DenseGemm *service){attention_gemm_=service;}
    void use_verify_split(bool enabled){verify_split_=enabled;}
    void use_verify_attention_kernel(VerifyAttentionKernel kernel){verify_kernel_=kernel;}
    void use_verify_grid(bool enabled){verify_grid_=enabled;}
    bool verify_grid()const{return verify_grid_;}
    // Cold benchmark capture; caller owns sufficient query storage to completion.
    void probe_pp_query(float *output){pp_query_probe_=output;}
    // Cold borrowed lease; service/arena survive every queued PP/Graph replay.
    void use_pp_blocked(DenseGemm *service,void *arena,size_t bytes,int rows=128,int min_start=0){
        if(service&&(!arena||!bytes))throw std::invalid_argument("missing blocked PP arena");
        if(rows<32||rows>256||rows%32)throw std::invalid_argument("invalid blocked PP query rows");
        if(min_start<0||min_start>262144)throw std::invalid_argument("invalid blocked PP prefix threshold");
        blocked_gemm_=service;blocked_arena_=arena;blocked_bytes_=bytes;blocked_query_rows_=rows;
        blocked_min_start_=min_start;
    }
    // Caller configures the kernel before capture; all scratch remains borrowed.
    void use_pp_attention_kernel(PrefillAttentionKernel kernel,int rows=32){
        if(rows<16||rows>256||rows%16)throw std::invalid_argument("invalid PP attention query rows");
        pp_kernel_=kernel;pp_kernel_rows_=rows;
    }
    bool pp_blocked()const{return blocked_gemm_!=nullptr;}
    bool verify_split()const{return verify_split_;}
private:
    const int* visual_positions_=nullptr;
    bool verify_split_=false,verify_grid_=false;
    float *pp_query_probe_=nullptr;
    DenseGemm *attention_gemm_=nullptr;
    DenseGemm *blocked_gemm_=nullptr;void *blocked_arena_=nullptr;size_t blocked_bytes_=0;
    int blocked_query_rows_=128,blocked_min_start_=0;
    PrefillAttentionKernel pp_kernel_=nullptr;int pp_kernel_rows_=32;
    VerifyAttentionKernel verify_kernel_=nullptr;
    RecurrentLayer &shared_;AttentionWeights weights_;AttentionShape shape_;AttentionSchedule schedule_;
};
}
