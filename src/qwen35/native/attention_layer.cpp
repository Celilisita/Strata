#include <algorithm>
#include "strata/qwen35/native/attention_ram.hpp"
#include "strata/qwen35/native/attention_window.hpp"
#include "strata/qwen35/native/mtp_position.hpp"
#include "strata/qwen35/native/tiny_batch.hpp"
#include "strata/qwen35/native/small_prefill.hpp"
#include "strata/qwen35/native/attention_layer.hpp"
#include "strata/qwen35/native/attention_decode.hpp"
#include "strata/qwen35/native/attention_ops.hpp"
#include "strata/qwen35/native/ops.hpp"
#include "strata/qwen35/native/attention_gemm.hpp"
#include <cmath>
#include "strata/qwen35/native/cache_q8.hpp"
#include "strata/qwen35/native/attention_verify.hpp"
#include "strata/qwen35/native/attention_blocked.hpp"

namespace strata::qwen35::native {
AttentionLayer::AttentionLayer(RecurrentLayer &w,AttentionWeights weights,AttentionShape s,AttentionSchedule p)
    :shared_(w),weights_(weights),shape_(s),schedule_(p){
    const auto a=w.shape();
    if(s.hidden!=a.hidden || s.dim!=256 || s.query_heads<1 || s.kv_heads<1 || s.query_heads%s.kv_heads ||
       s.query_heads*s.dim!=a.dim*a.value_heads || 2*s.kv_heads*s.dim>a.dim*a.query_heads ||
       2*s.query_heads*s.dim>(a.mixed_capacity>0?a.mixed_capacity:a.dim*(2*a.query_heads+a.value_heads)) ||
       s.rotary_dim<2 || s.rotary_dim>s.dim || s.rotary_dim%2 || !(s.epsilon>0) || !(s.rotary_base>1))throw std::invalid_argument("unsupported shared Dense attention geometry");
    if(!schedule_.decode_rounding.processors){
        int device=0;check_cuda(cudaGetDevice(&device),"attention policy device");
        schedule_.decode_rounding.processors=capabilities(device).multiprocessors;
    }
    rebind(weights);
}
void AttentionLayer::rebind(AttentionWeights w){
    if(!w.norm || !w.q_norm || !w.k_norm || !w.post_norm)throw std::invalid_argument("missing own attention norms");
    const Matrix matrices[]={w.q,w.k,w.v,w.out,w.ffn_gate,w.ffn_up,w.ffn_down};
    const int intermediate=shared_.shared_ffn(2).intermediate_features();
    const int rows[]={2*shape_.query_heads*shape_.dim,shape_.kv_heads*shape_.dim,shape_.kv_heads*shape_.dim,
                      shape_.hidden,intermediate,intermediate,shape_.hidden};
    const int columns[]={shape_.hidden,shape_.hidden,shape_.hidden,shape_.query_heads*shape_.dim,
                         shape_.hidden,shape_.hidden,intermediate};
    for(int i=0;i<7;++i){matrix_bytes(matrices[i],false,false,i==4||i==5);if(matrices[i].rows!=rows[i] || matrices[i].columns!=columns[i])
        throw std::invalid_argument("own attention matrix geometry changed");}
    shared_.shared_ffn(2).rebind(w.ffn_gate,w.ffn_up,w.ffn_down);
    shared_.shared_ffn(1).rebind(w.ffn_gate,w.ffn_up,w.ffn_down);weights_=w;
}
AttentionBuffers AttentionLayer::buffers(int tokens) const{
    const auto &a=shared_.buffers();const auto &f=shared_.shared_ffn(tokens).buffers();
    const size_t offset=size_t(shared_.max_tokens())*shape_.kv_heads*shape_.dim;
    // Compact q_raw/attention/pregate borrow FFN gate/up/activation. At gate
    // execution q_norm/q_rope/flash_q are dead, and provide distinct outputs.
    const bool compact=shared_.shape().compact_workspace;
    return {a.norm,a.qkv,a.k,a.v,a.z,a.q,a.attention,a.q+offset,a.gated,a.k+offset,a.v+offset,
            a.conv_raw,a.conv_silu,compact?a.z:f.gate,compact?a.attention:f.up,compact?a.gated:f.activation,
            a.projected,a.residual,a.post_norm,a.output};
}
void AttentionLayer::attention(const float *hidden,const int *positions,AttentionCache c,int tokens,cudaStream_t stream){
    const auto &s=shape_;const auto &w=weights_;auto b=buffers(tokens);const bool vector=tokens==1;
    if(!hidden || !positions || !c.key || !c.value || tokens<1 || tokens>shared_.max_tokens() ||
       c.start<0 || c.valid!=c.start+tokens || (!c.window&&!c.ram&&c.capacity<c.valid) || c.capacity>262144)
        throw std::invalid_argument("invalid own attention layer invocation");
    const auto &scratch=shared_.shared_ffn(tokens).buffers();void *prepared=vector?scratch.quant_input:scratch.half_input;
    rms_norm(hidden,w.norm,b.norm,s.hidden,tokens,s.epsilon,stream);
    const bool native32=tokens==32&&shared_.native32_enabled();
    if(native32)shared_.project_native32(b.norm,{{w.q,b.q_raw},{w.k,b.k_raw},{w.v,b.v_raw}},stream);
    else{
    if(vector)vector_quantize(b.norm,prepared,s.hidden,stream);
    else if(shared_.tiny_batches(tokens))tiny_batch_quantize(b.norm,prepared,s.hidden,tokens,stream);
    else if(tokens<64)small_prefill_quantize(b.norm,prepared,s.hidden,tokens,stream);
    else to_half(b.norm,prepared,size_t(s.hidden)*tokens,stream);
    shared_.project(w.q,prepared,b.q_raw,tokens,schedule_.projections[0],stream);
    shared_.project(w.k,prepared,b.k_raw,tokens,schedule_.projections[1],stream);
    shared_.project(w.v,prepared,b.v_raw,tokens,schedule_.projections[2],stream);
    }
    rms_norm(b.q_raw,w.q_norm,b.q_norm,s.dim,tokens*s.query_heads,s.epsilon,stream,nullptr,2*s.dim);
    rms_norm(b.k_raw,w.k_norm,b.k_norm,s.dim,tokens*s.kv_heads,s.epsilon,stream);
    if(visual_positions_){
        rotary_visual(b.q_norm,positions,b.q_rope,s.dim,s.rotary_dim,s.query_heads,tokens,s.rotary_base,stream,visual_positions_);
        rotary_visual(b.k_norm,positions,b.k_rope,s.dim,s.rotary_dim,s.kv_heads,tokens,s.rotary_base,stream,visual_positions_);
    }else{
        rotary_text(b.q_norm,positions,b.q_rope,s.dim,s.rotary_dim,s.query_heads,tokens,s.rotary_base,stream);
        rotary_text(b.k_norm,positions,b.k_rope,s.dim,s.rotary_dim,s.kv_heads,tokens,s.rotary_base,stream);
    }
    hadamard(b.q_rope,b.flash_q,256,size_t(tokens)*s.query_heads,stream);
    hadamard(b.k_rope,b.flash_k,256,size_t(tokens)*s.kv_heads,stream);
    hadamard(b.v_raw,b.flash_v,64,size_t(tokens)*s.kv_heads*4,stream);
    if(pp_query_probe_&&tokens>=128)check_cuda(cudaMemcpyAsync(pp_query_probe_,b.flash_q,
        size_t(tokens)*s.query_heads*s.dim*4,cudaMemcpyDeviceToDevice,stream),"bounded PP query fixture capture");
    if(c.ram){
        ram_write(b.flash_k,*c.ram,true,s.dim,s.kv_heads,tokens,positions,c.graph_shift,stream);
        ram_write(b.flash_v,*c.ram,false,s.dim,s.kv_heads,tokens,positions,c.graph_shift,stream);
    }else if(c.window){
        if(!c.q8||c.capacity<std::min(c.valid,c.window+tokens-1))throw std::invalid_argument("insufficient sliding KV transaction guard");
        q8_window_write(b.flash_k,c.key,s.dim,s.kv_heads,tokens,positions,c.graph_shift,c.capacity,stream);
        q8_window_write(b.flash_v,c.value,s.dim,s.kv_heads,tokens,positions,c.graph_shift,c.capacity,stream);
    }else if(c.graph_start){
        if(!c.q8)throw std::invalid_argument("MTP graph cache requires Q8");
        q8_cache_write_positions(b.flash_k,c.key,s.dim,s.kv_heads,tokens,positions,c.graph_shift,c.capacity,stream);
        q8_cache_write_positions(b.flash_v,c.value,s.dim,s.kv_heads,tokens,positions,c.graph_shift,c.capacity,stream);
    }else if(c.cursor){
        if(!vector)throw std::invalid_argument("device KV cursor is single-token only");
        (c.q8?q8_cache_write_position:q4_cache_write_position)(b.flash_k,c.key,s.dim,s.kv_heads,positions,stream);
        (c.q8?q8_cache_write_position:q4_cache_write_position)(b.flash_v,c.value,s.dim,s.kv_heads,positions,stream);
    }else{
        (c.q8?q8_cache_write:q4_cache_write)(b.flash_k,c.key,s.dim,s.kv_heads,tokens,c.start,c.capacity,stream);
        (c.q8?q8_cache_write:q4_cache_write)(b.flash_v,c.value,s.dim,s.kv_heads,tokens,c.start,c.capacity,stream);
    }
    // PP decoded weights are dead here. Borrow the same staging for either phase.
    const auto &pp=shared_.shared_ffn(2);void *workspace=pp.buffers().half_weights;
    const size_t capacity=pp.half_weight_capacity();
    if(c.ram&&!c.full_history&&tokens<=3&&c.ram->host_capacity>c.ram->resident){
        ram_attention(b.flash_q,*c.ram,b.attention,workspace,capacity,positions,c.graph_shift,tokens,s.query_heads,schedule_.decode_rounding,stream);
    }else if(c.window){
        PrefillAttentionShape p{s.dim,s.query_heads,s.kv_heads,tokens,c.start,c.valid,c.capacity,1.f/std::sqrt(float(s.dim)),schedule_.rounding,true};
        if(vector)window_decode_attention(b.flash_q,c.key,c.value,b.attention,workspace,capacity,c,positions,
                                          c.window,s.query_heads,s.kv_heads,p.scale,schedule_.decode_rounding,stream);
        else if(tokens<=3)window_verify_attention(b.flash_q,c.key,c.value,b.attention,workspace,capacity,p,
                    positions,c.graph_shift,c.window,verify_grid_?64:32,stream);
        else window_prefill_attention(b.flash_q,c.key,c.value,b.attention,workspace,capacity,p,c.window,stream);
    }else if(vector){
        const bool host=c.ram&&c.full_history&&c.ram->host_capacity>c.ram->resident;
        const void *key=host?c.ram->host_key:c.key,*value=host?c.ram->host_value:c.value;
        DecodeAttentionShape p{s.dim,s.query_heads,s.kv_heads,c.valid,host?c.ram->host_capacity:c.capacity,128,1.0f/std::sqrt(float(s.dim))};
        p.q8=c.q8;
        p.partitions=decode_partition_count(c.valid,s.query_heads,schedule_.decode_rounding);
        if(c.cursor)decode_attention_graph(b.flash_q,key,value,b.attention,workspace,capacity,p,c.cursor,c.graph_partitions,stream);
        else decode_attention(b.flash_q,key,value,b.attention,workspace,capacity,p,stream);
    }else{
        const bool host=c.ram&&c.ram->host_capacity>c.ram->resident;
        const void *key=host?c.ram->host_key:c.key,*value=host?c.ram->host_value:c.value;
        const int logical_capacity=c.ram?c.ram->host_capacity:c.capacity;
        auto r=schedule_.rounding;r.padded_keys=((c.valid+255)/256)*256;
        PrefillAttentionShape p{s.dim,s.query_heads,s.kv_heads,tokens,c.start,c.valid,logical_capacity,1.0f/std::sqrt(float(s.dim)),r};
        p.q8=c.q8;p.graph_start=c.graph_start;
        if(verify_split_&&c.q8&&tokens<=3){
            // A fixed request-capacity decision gives eager and Graph identical grids.
            const int parts=verify_grid_&&logical_capacity>=16384?(logical_capacity+511)/512:
                verify_attention_partitions(c.graph_start?logical_capacity:c.valid,s.query_heads,schedule_.decode_rounding);
            const auto kernel=verify_kernel_?verify_kernel_:verify_attention_q8;
            kernel(b.flash_q,key,value,b.attention,workspace,capacity,p,parts,stream);
        }else if(pp_kernel_&&c.q8&&tokens>=128&&!c.graph_start){
            pp_kernel_(b.flash_q,key,value,b.attention,workspace,capacity,p,pp_kernel_rows_,stream);
        }else if(blocked_gemm_&&c.q8&&tokens>=128&&!c.graph_start&&c.start>=blocked_min_start_){
            blocked_attention(*blocked_gemm_,b.flash_q,key,value,b.attention,
                              blocked_arena_,blocked_bytes_,p,blocked_query_rows_,stream);
        }else if(!attention_gemm_||!gemm_prefill_attention(*attention_gemm_,b.flash_q,key,value,b.attention,
                                                   b.q_norm,workspace,capacity,p,stream))
            prefill_attention(b.flash_q,key,value,b.attention,workspace,capacity,p,stream);
    }
    hadamard(b.attention,b.pregate,64,size_t(tokens)*s.query_heads*4,stream);
    attention_gate(b.q_raw,b.pregate,b.gate,b.sigmoid,b.gated,s.dim,s.query_heads,tokens,stream);
}
void AttentionLayer::finish(const float *hidden,int tokens,cudaStream_t stream){
    const auto &s=shape_;const auto &w=weights_;auto b=buffers(tokens);const bool vector=tokens==1;
    auto &ffn=shared_.shared_ffn(tokens);const auto &scratch=ffn.buffers();void *prepared=vector?scratch.quant_input:scratch.half_input;
    if(tokens==32&&shared_.native32_enabled())shared_.project_native32(b.gated,{{w.out,b.projected}},stream);
    else{
    if(vector)vector_quantize(b.gated,prepared,s.dim*s.query_heads,stream);
    else if(shared_.tiny_batches(tokens))tiny_batch_quantize(b.gated,prepared,s.dim*s.query_heads,tokens,stream);
    else if(tokens<64)small_prefill_quantize(b.gated,prepared,s.dim*s.query_heads,tokens,stream);
    else to_half(b.gated,prepared,size_t(tokens)*s.dim*s.query_heads,stream);
    shared_.project(w.out,prepared,b.projected,tokens,schedule_.projections[3],stream);
    }
    add(b.projected,hidden,b.residual,size_t(tokens)*s.hidden,stream);
    rms_norm(b.residual,w.post_norm,b.post_norm,s.hidden,tokens,s.epsilon,stream);
    ffn.execute(b.post_norm,tokens,stream);
    add(scratch.output,b.residual,b.output,size_t(tokens)*s.hidden,stream);
}
}
