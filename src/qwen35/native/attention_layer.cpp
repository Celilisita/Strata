#include "device_selected_request.hpp"
#include "fp4_residual_norm_scope.hpp"
#include "inputnorm_scope.hpp"
#include "ffn_down_add.hpp"
#include "ram_six.hpp"
#include "t1_tc.hpp"
#include "regp_policy.hpp"
#include "fp8_pp_policy.hpp"
#include "ram_score_policy.hpp"
#include "ram40_prefix512_policy.hpp"
#include "operator_fusion_policy.hpp"
#include <algorithm>
#include "strata/qwen35/native/nvfp4_runtime.hpp"
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
#include "ram_six_policy.hpp"
#include "strata/qwen35/native/cache_q8.hpp"
#include "strata/qwen35/native/attention_verify.hpp"
#include <attention_score_reuse_policy.hpp>
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
AttentionLayer::~AttentionLayer(){
    // The staging lease is process-cold; a failed teardown must not throw.
    if(pp_stage_)cudaFree(pp_stage_);
}
void *AttentionLayer::pp_staging(size_t bytes){
    if(pp_stage_bytes_>=bytes)return pp_stage_;
    const size_t unit=size_t(64)<<20,grown=((bytes+unit-1)/unit)*unit;
    void *fresh=nullptr;
    check_cuda(cudaMalloc(&fresh,grown),"owned PP staging growth");
    if(pp_stage_)check_cuda(cudaFree(pp_stage_),"owned PP staging release");
    pp_stage_=fresh;pp_stage_bytes_=grown;return fresh;
}
void AttentionLayer::release_staging(){
    if(!pp_stage_)return;
    cudaFree(pp_stage_);pp_stage_=nullptr;pp_stage_bytes_=0;
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
    attention_impl(hidden,positions,c,tokens,stream,false);
}
void AttentionLayer::attention_impl(const float *hidden,const int *positions,AttentionCache c,int tokens,cudaStream_t stream,bool closed){
    const auto &s=shape_;const auto &w=weights_;auto b=buffers(tokens);const bool vector=tokens==1;
    if(!hidden || !positions || !c.key || !c.value || tokens<1 || tokens>shared_.max_tokens() ||
       c.start<0 || c.valid!=c.start+tokens || (!c.window&&!c.ram&&c.capacity<c.valid) || c.capacity>262144)
        throw std::invalid_argument("invalid own attention layer invocation");
    if(c.ram&&ram_request_bound_&&!c.full_history&&tokens<=4){
        if(ram_request_bound_>c.ram->resident||ram_request_bound_>c.ram->host_capacity||
           (!c.cursor&&!c.graph_start&&c.valid>ram_request_bound_))
            throw std::logic_error("RAM request resident admission exceeded");
    }
    const int prefix_bound=c.ram&&c.full_history?ram_prefix_bound_:0;
    if(prefix_bound&&(prefix_bound>c.ram->resident||prefix_bound>c.ram->host_capacity||c.valid>prefix_bound))
        throw std::logic_error("RAM fresh prefix resident admission exceeded");
    const int resident_bound=ram_request_bound_?ram_request_bound_:prefix_bound;
    {
    const auto &scratch=shared_.shared_ffn(tokens).buffers();void *prepared=vector?scratch.quant_input:scratch.half_input;
    const bool share_nv=w.q.quant==Quant::NVFP4&&w.k.quant==Quant::NVFP4&&w.v.quant==Quant::NVFP4&&nvfp4_shared_projections_enabled();
    InputNormScope input_norm(closed&&!pp_query_probe_&&share_nv,hidden,w.norm,b.norm,nullptr,s.hidden,tokens,s.epsilon,stream);
    if(!input_norm.deferred())rms_norm(hidden,w.norm,b.norm,s.hidden,tokens,s.epsilon,stream);
    const bool native32=tokens==32&&shared_.native32_enabled();
    if(native32)shared_.project_native32(b.norm,{{w.q,b.q_raw},{w.k,b.k_raw},{w.v,b.v_raw}},stream);
    else if(share_nv)
        shared_.project_shared_nvfp4(b.norm,{{w.q,b.q_raw},{w.k,b.k_raw},{w.v,b.v_raw}},tokens,stream);
    else{
    if(w.q.quant==Quant::NVFP4&&w.k.quant==Quant::NVFP4&&w.v.quant==Quant::NVFP4){}
    else if(vector)vector_quantize(b.norm,prepared,s.hidden,stream);
    else if(shared_.tiny_batches(tokens))tiny_batch_quantize(b.norm,prepared,s.hidden,tokens,stream);
    else if(tokens<64)small_prefill_quantize(b.norm,prepared,s.hidden,tokens,stream);
    else to_half(b.norm,prepared,size_t(s.hidden)*tokens,stream);
    shared_.project(w.q,w.q.quant==Quant::NVFP4?b.norm:prepared,b.q_raw,tokens,schedule_.projections[0],stream);
    shared_.project(w.k,w.k.quant==Quant::NVFP4?b.norm:prepared,b.k_raw,tokens,schedule_.projections[1],stream);
    shared_.project(w.v,w.v.quant==Quant::NVFP4?b.norm:prepared,b.v_raw,tokens,schedule_.projections[2],stream);
    }
    input_norm.verify();
    } // No deferred norm descriptor can escape the closed Q/K/V projection group.
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
    // Q/K/V projection is complete on this ordered stream. Its decoded-weight
    // arena is dead until finish() starts output projection/FFN on that stream.
    // Attention partials and optional F32 scores share disjoint subranges here;
    // carry stays in registers/shared memory. Q/K/V/output and Runtime codecs
    // are distinct live allocations, checked by the score entry before launch.
    const auto &pp=shared_.shared_ffn(2);void *workspace=pp.buffers().half_weights;
    size_t capacity=pp.half_weight_capacity();
    const bool draft_tc=draft_t1_tc_&&vector&&closed&&!visual_positions_&&!pp_query_probe_&&c.q8&&!c.window&&!c.full_history&&
        s.dim==256&&s.query_heads==24&&s.kv_heads==4&&capacity>=T1TcWorkspace;
    bool tc_used=false;
    if(c.ram&&!ram_request_bound_&&!c.full_history&&tokens<=4&&(c.ram->host_key!=c.ram->key||c.ram->host_value!=c.ram->value||device_selected_attention(*c.ram))){
        const int six=ram_six_mode();
        const auto &ram=*c.ram;
        if(draft_tc&&ram.resident==40960&&ram.slots==2560&&ram.heads==4&&ram.selected_tokens==8192&&ram.selected_slots==512&&
           ram.host_capacity<=40960&&ram.pages*RamPage==ram.host_capacity&&ram.sorted_ids&&ram.selected_slot&&c.key==ram.key&&c.value==ram.value){
            ram_select(b.flash_q,ram,positions,c.graph_shift,s.query_heads,stream);
            t1_tc_attention(b.flash_q,ram.key,ram.value,ram,b.attention,workspace,capacity,
                T1TcShape{c.valid,ram.host_capacity,c.graph_shift,true,nullptr,positions},stream);
            ++draft_t1_tc_stats().selected_host_calls;tc_used=true;
        }else if(six&&tokens>1&&(s.query_heads/s.kv_heads)%6==0){
            const int parts=six==1?64:128;
            if(closed&&!visual_positions_&&!pp_query_probe_&&use_ram40_score_reuse(*c.ram,tokens,s.query_heads,parts,capacity)){
                Ram40Prefix512ClosedScope prefix512_scope(true);
                ram40_score_attention(b.flash_q,*c.ram,b.attention,workspace,capacity,positions,c.graph_shift,tokens,s.query_heads,parts,stream);
                ram_score_request().submissions+=tokens;ram_score40_submissions()+=tokens;
            }else if(closed&&!visual_positions_&&!pp_query_probe_&&use_ram_score_reuse(*c.ram,tokens,s.query_heads,parts,capacity))
                ram_score_attention(b.flash_q,*c.ram,b.attention,workspace,capacity,positions,c.graph_shift,tokens,s.query_heads,parts,stream);
            else ram_attention_six(b.flash_q,*c.ram,b.attention,workspace,capacity,positions,c.graph_shift,tokens,s.query_heads,schedule_.decode_rounding,parts,stream);
        }
        else ram_attention(b.flash_q,*c.ram,b.attention,workspace,capacity,positions,c.graph_shift,tokens,s.query_heads,schedule_.decode_rounding,stream);
    }else if(c.window){
        PrefillAttentionShape p{s.dim,s.query_heads,s.kv_heads,tokens,c.start,c.valid,c.capacity,1.f/std::sqrt(float(s.dim)),schedule_.rounding,true};
        if(vector)window_decode_attention(b.flash_q,c.key,c.value,b.attention,workspace,capacity,c,positions,
                                          c.window,s.query_heads,s.kv_heads,p.scale,schedule_.decode_rounding,stream);
        else if(tokens<=4)window_verify_attention(b.flash_q,c.key,c.value,b.attention,workspace,capacity,p,
                    positions,c.graph_shift,c.window,verify_grid_?64:32,stream);
        else window_prefill_attention(b.flash_q,c.key,c.value,b.attention,workspace,capacity,p,c.window,stream);
    }else if(vector){
        const bool host=c.ram&&!resident_bound&&c.full_history&&(c.ram->host_key!=c.ram->key||c.ram->host_value!=c.ram->value);
        const void *key=host?c.ram->host_key:c.key,*value=host?c.ram->host_value:c.value;
        DecodeAttentionShape p{s.dim,s.query_heads,s.kv_heads,c.valid,host?c.ram->host_capacity:c.capacity,128,1.0f/std::sqrt(float(s.dim))};
        p.q8=c.q8;
        p.partitions=decode_partition_count(c.valid,s.query_heads,schedule_.decode_rounding);
        const int tc_capacity=c.ram?ram_request_bound_:c.capacity;
        if(draft_tc&&!host&&tc_capacity>=c.valid&&tc_capacity<=16384&&tc_capacity>0&&(!c.graph_start||c.cursor)){
            t1_tc_attention(b.flash_q,key,value,c.ram?*c.ram:RamKvView{},b.attention,workspace,capacity,
                T1TcShape{c.valid,tc_capacity,0,false,c.cursor,nullptr},stream);
            ++draft_t1_tc_stats().direct_host_calls;tc_used=true;
        }else if(c.cursor)decode_attention_graph(b.flash_q,key,value,b.attention,workspace,capacity,p,c.cursor,c.graph_partitions,stream);
        else decode_attention(b.flash_q,key,value,b.attention,workspace,capacity,p,stream);
    }else{
        const bool host=c.ram&&!resident_bound&&(c.ram->host_key!=c.ram->key||c.ram->host_value!=c.ram->value);
        const void *key=host?c.ram->host_key:c.key,*value=host?c.ram->host_value:c.value;
        const int logical_capacity=c.ram&&resident_bound?
            std::min(c.capacity,((resident_bound+8191)/8192)*8192):
            (c.ram?c.ram->host_capacity:c.capacity);
        auto r=schedule_.rounding;r.padded_keys=((c.valid+255)/256)*256;
        PrefillAttentionShape p{s.dim,s.query_heads,s.kv_heads,tokens,c.start,c.valid,logical_capacity,1.0f/std::sqrt(float(s.dim)),r};
        p.q8=c.q8;p.graph_start=c.graph_start;
        const bool register_pp=closed&&!visual_positions_&&!pp_query_probe_&&pp_kernel_&&c.ram&&resident_bound>0&&
            !host&&!c.window&&!c.cursor&&!c.graph_start&&!c.graph_shift&&c.key==c.ram->key&&c.value==c.ram->value&&
            c.ram->host_key==c.key&&c.ram->host_value==c.value&&c.valid<=resident_bound;
        const bool fp8_full_history=closed&&!visual_positions_&&!pp_query_probe_&&pp_kernel_&&c.ram&&c.q8&&c.full_history&&
            !host&&!c.window&&!c.cursor&&!c.graph_start&&!c.graph_shift&&c.key==c.ram->key&&c.value==c.ram->value&&
            c.ram->host_key==c.key&&c.ram->host_value==c.value&&c.ram->host_capacity>0&&c.ram->host_capacity<=40960&&
            c.ram->resident>=c.ram->host_capacity&&c.valid<=c.ram->host_capacity;
        const bool direct_q8=pp_q8_kernel_&&c.q8&&tokens>=128&&!c.window&&!c.cursor&&!c.graph_start&&!c.graph_shift&&
            !register_pp&&!fp8_full_history;
        if(!direct_q8){
            // A long prefix stages one half K/V row set per live position; the
            // borrowed FFN arena only holds this layer's weights. Lease an owned
            // buffer when the live extent does not fit it, so a request is not
            // refused at the PP staging guard for exceeding the FFN weight size.
            const size_t staging=prefill_attention_workspace(p);
            if(staging>capacity){workspace=pp_staging(staging);capacity=staging;}
        }
        Fp8PpScope fp8_scope(register_pp||fp8_full_history);
        if(register_pp&&use_regp_attention(p)){
            regp_attention(b.flash_q,key,value,b.attention,workspace,capacity,p,stream);
            ++regp_stats().host_calls[regp_case(p)];
        }else if(verify_split_&&c.q8&&tokens<=4){
            // A fixed request-capacity decision gives eager and Graph identical grids.
            const int parts=verify_grid_&&logical_capacity>=16384?(logical_capacity+511)/512:
                verify_attention_partitions(c.graph_start?logical_capacity:c.valid,s.query_heads,schedule_.decode_rounding);
            const auto kernel=verify_kernel_?verify_kernel_:verify_attention_q8;
            const bool direct_resident=!host&&!c.window&&!c.cursor&&!c.full_history&&!c.graph_shift;
            AttentionScoreReuseScope score_scope(direct_resident);
            kernel(b.flash_q,key,value,b.attention,workspace,capacity,p,parts,stream);
        }else if(direct_q8){
            pp_q8_kernel_(b.flash_q,key,value,b.attention,p,pp_q8_rows_,stream);
        }else if(pp_kernel_&&c.q8&&tokens>=128&&!c.graph_start){
            pp_kernel_(b.flash_q,key,value,b.attention,workspace,capacity,p,pp_kernel_rows_,stream);
        }else if(blocked_gemm_&&c.q8&&tokens>=128&&!c.graph_start&&c.start>=blocked_min_start_){
            blocked_attention(*blocked_gemm_,b.flash_q,key,value,b.attention,
                              blocked_arena_,blocked_bytes_,p,blocked_query_rows_,stream);
        }else if(!attention_gemm_||!gemm_prefill_attention(*attention_gemm_,b.flash_q,key,value,b.attention,
                                                   b.q_norm,workspace,capacity,p,stream))
            prefill_attention(b.flash_q,key,value,b.attention,workspace,capacity,p,stream);
    }
    if(draft_t1_tc_&&vector&&!tc_used)++draft_t1_tc_stats().fallback_host_calls;
    hadamard(b.attention,b.pregate,64,size_t(tokens)*s.query_heads*4,stream);
    attention_gate(b.q_raw,b.pregate,b.gate,b.sigmoid,b.gated,s.dim,s.query_heads,tokens,stream);
}
void AttentionLayer::finish(const float *hidden,int tokens,cudaStream_t stream){
    const auto &s=shape_;const auto &w=weights_;auto b=buffers(tokens);const bool vector=tokens==1;
    auto &ffn=shared_.shared_ffn(tokens);const auto &scratch=ffn.buffers();void *prepared=vector?scratch.quant_input:scratch.half_input;
    if(tokens==32&&shared_.native32_enabled())shared_.project_native32(b.gated,{{w.out,b.projected}},stream);
    else{
    if(w.out.quant==Quant::NVFP4){}
    else if(vector)vector_quantize(b.gated,prepared,s.dim*s.query_heads,stream);
    else if(shared_.tiny_batches(tokens))tiny_batch_quantize(b.gated,prepared,s.dim*s.query_heads,tokens,stream);
    else if(tokens<64)small_prefill_quantize(b.gated,prepared,s.dim*s.query_heads,tokens,stream);
    else to_half(b.gated,prepared,size_t(tokens)*s.dim*s.query_heads,stream);
    shared_.project(w.out,w.out.quant==Quant::NVFP4?b.gated:prepared,b.projected,tokens,schedule_.projections[3],stream);
    }
    // The MTP conditioning lease may be exactly this residual buffer. The
    // existing two-operation path permits its in-place add; fused leases do not.
    const bool separate_residual=hidden!=b.residual;
    const bool fused_residual=separate_residual&&w.out.quant==Quant::NVFP4&&use_native_operator_fusion(1,tokens);
    ResidualNormScope norm_scope(separate_residual&&w.ffn_gate.quant==Quant::NVFP4,b.projected,hidden,w.post_norm,b.residual,b.post_norm,
        s.hidden,tokens,s.epsilon,stream,fused_residual);
    if(!norm_scope.deferred()){
    if(fused_residual)
        add_rms_norm(b.projected,hidden,w.post_norm,b.residual,b.post_norm,s.hidden,tokens,s.epsilon,stream);
    else {
    add(b.projected,hidden,b.residual,size_t(tokens)*s.hidden,stream);
    rms_norm(b.residual,w.post_norm,b.post_norm,s.hidden,tokens,s.epsilon,stream);
    }
    }
    DownAddScope down_scope(w.ffn_gate.quant==Quant::NVFP4&&s.hidden==5120&&w.ffn_gate.rows==17408,
        b.residual,b.output,scratch.output,tokens,stream);
    ffn.execute(b.post_norm,tokens,stream);
    norm_scope.verify();down_scope.verify();
    if(!down_scope.consumed())add(scratch.output,b.residual,b.output,size_t(tokens)*s.hidden,stream);
}
}
