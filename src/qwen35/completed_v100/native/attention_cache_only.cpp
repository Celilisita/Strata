#include <algorithm>
#include "strata/qwen35/completed_v100/native/kv_ram.hpp"
#include "strata/qwen35/completed_v100/native/attention_window.hpp"
#include "strata/qwen35/completed_v100/native/mtp_position.hpp"
#include "strata/qwen35/completed_v100/native/tiny_batch.hpp"
#include "strata/qwen35/completed_v100/native/attention_layer.hpp"
#include "strata/qwen35/completed_v100/native/attention_ops.hpp"
#include "strata/qwen35/completed_v100/native/cache_q8.hpp"
#include "strata/qwen35/completed_v100/native/ops.hpp"
#include "strata/qwen35/completed_v100/native/small_prefill.hpp"
#include "strata/qwen35/completed_v100/native/original_formats.hpp"
namespace strata::completed_v100::qwen35::native {
void AttentionLayer::cache_only(const float *hidden,const int *positions,AttentionCache c,int tokens,cudaStream_t stream){
    const auto &s=shape_;const auto &w=weights_;auto b=buffers(tokens);
    if(!hidden||!positions||!c.key||!c.value||c.cursor||tokens<1||tokens>shared_.max_tokens()||
       c.start<0||c.valid!=c.start+tokens||(!c.window&&!c.ram&&c.capacity<c.valid))
        throw std::invalid_argument("invalid own draft KV catch-up lease");
    const auto &scratch=shared_.shared_ffn(tokens).buffers();
    void *prepared=tokens==1?scratch.quant_input:scratch.half_input;
    rms_norm(hidden,w.norm,b.norm,s.hidden,tokens,s.epsilon,stream);
    if(original_projection(w.k))prepared=b.norm;
    else if(tokens==1)vector_quantize(b.norm,prepared,s.hidden,stream);
    else if(shared_.tiny_batches(tokens))tiny_batch_quantize(b.norm,prepared,s.hidden,tokens,stream);
    else if(tokens<64)small_prefill_quantize(b.norm,prepared,s.hidden,tokens,stream);
    else to_half(b.norm,prepared,size_t(s.hidden)*tokens,stream);
    shared_.project(w.k,prepared,b.k_raw,tokens,schedule_.projections[1],stream);
    shared_.project(w.v,prepared,b.v_raw,tokens,schedule_.projections[2],stream);
    rms_norm(b.k_raw,w.k_norm,b.k_norm,s.dim,tokens*s.kv_heads,s.epsilon,stream);
    if(visual_positions_)rotary_visual(b.k_norm,positions,b.k_rope,s.dim,s.rotary_dim,s.kv_heads,tokens,s.rotary_base,stream,visual_positions_);
    else rotary_text(b.k_norm,positions,b.k_rope,s.dim,s.rotary_dim,s.kv_heads,tokens,s.rotary_base,stream);
    hadamard(b.k_rope,b.flash_k,256,size_t(tokens)*s.kv_heads,stream);
    hadamard(b.v_raw,b.flash_v,64,size_t(tokens)*s.kv_heads*4,stream);
    if(c.ram){
        ram_write(b.flash_k,*c.ram,true,s.dim,s.kv_heads,tokens,positions,c.graph_shift,stream);
        ram_write(b.flash_v,*c.ram,false,s.dim,s.kv_heads,tokens,positions,c.graph_shift,stream);return;
    }else if(c.window){
        if(!c.q8||c.capacity<std::min(c.valid,c.window+tokens-1))throw std::invalid_argument("insufficient sliding KV transaction guard");
        q8_window_write(b.flash_k,c.key,s.dim,s.kv_heads,tokens,positions,c.graph_shift,c.capacity,stream);
        q8_window_write(b.flash_v,c.value,s.dim,s.kv_heads,tokens,positions,c.graph_shift,c.capacity,stream);
    }else if(c.graph_start){
        if(!c.q8)throw std::invalid_argument("MTP graph catch-up requires Q8");
        q8_cache_write_positions(b.flash_k,c.key,s.dim,s.kv_heads,tokens,positions,c.graph_shift,c.capacity,stream);
        q8_cache_write_positions(b.flash_v,c.value,s.dim,s.kv_heads,tokens,positions,c.graph_shift,c.capacity,stream);
        return;
    }
    if(c.window)return;
    const auto write=c.q8?q8_cache_write:q4_cache_write;
    write(b.flash_k,c.key,s.dim,s.kv_heads,tokens,c.start,c.capacity,stream);
    write(b.flash_v,c.value,s.dim,s.kv_heads,tokens,c.start,c.capacity,stream);
}
}
