#include "strata/gemma4/native/assistant.hpp"
#include "strata/gemma4/native/assistant_split.hpp"
#include "strata/qwen35/native/ops.hpp"
#include <climits>
#include <cmath>
#include <stdexcept>
#include <utility>
namespace strata::gemma4::native::assistant {
namespace {
struct Pick {float score;int id;};
__device__ Pick better(Pick a,Pick b){return b.score>a.score||(b.score==a.score&&b.id<a.id)?b:a;}
__device__ Pick reduce(Pick p){
    for(int d=16;d;d/=2){Pick other{__shfl_down_sync(0xffffffff,p.score,d),__shfl_down_sync(0xffffffff,p.id,d)};
        if((threadIdx.x%32)+d<32)p=better(p,other);}return p;
}
__global__ void finish(const float* logits,const float* state,int* proposal,unsigned* invalid){
    __shared__ Pick warps[8];Pick p{-INFINITY,INT_MAX};
    for(int i=threadIdx.x;i<262144;i+=256){const float v=logits[i];if(!isfinite(v))atomicOr(invalid,8u);else p=better(p,{v,i});}
    for(int i=threadIdx.x;i<5376;i+=256)if(!isfinite(state[i]))atomicOr(invalid,8u);
    p=reduce(p);if(!(threadIdx.x%32))warps[threadIdx.x/32]=p;__syncthreads();
    if(threadIdx.x<32){p=threadIdx.x<8?warps[threadIdx.x]:Pick{-INFINITY,INT_MAX};p=reduce(p);
        if(!threadIdx.x){if(p.id==INT_MAX)atomicOr(invalid,8u);*proposal=*invalid?0:p.id;}}
}
}
void Draft::run(const float* embedding,const float* seed,CacheView local,CacheView global,const Bounds* bounds,const HistoryView* history){
    validate_lease(embedding,seed,local,global,bounds);
    if(history&&(history->binding.stream!=stream_||history->logical_capacity!=global.capacity||
        history->host_key!=global.key||history->host_value!=global.value))
        throw std::invalid_argument("Assistant explicit mapped history lease");
    primitive::check_cuda(cudaMemsetAsync(invalid_,0,sizeof(unsigned),stream_),"Assistant per-run diagnostic reset");
    condition_concat(embedding,seed,condition_,invalid_,stream_);project(pre_projection_,condition_,hidden_[0]);
    float* input=hidden_[0];float* output=hidden_[1];
    for(int i=0;i<4;++i){const auto& w=layers_[i];const auto& s=spec_.layers[i];
        primitive::rms_norm(input,w.pre,normalized_,1024,1,spec_.epsilon,stream_);project(w.q,normalized_,query_);
        head_norm_rope(query_,query_,w.q_norm,s.dimension,32,1,spec_.epsilon,i<3?spec_.local_rope:spec_.global_rope,
            i<3?nullptr:frequencies_,&bounds->rope_position,true,stream_);
        if(i==3&&history){
            AttentionWorkspace work{};work.data=decoded_;work.bytes=decoded_values_*sizeof(float);
            // Query RoPE uses rope_position; shared KV ends at P-1.
            history_attention(query_,attended_,*history,&bounds->kv_end,32,1,work,history->binding);
        }else if(split_attention_){
            // Q projection consumed decoded_. Attention finishes on the same
            // stream before output projection reuses this exclusive arena.
            const SplitAttentionLease lease{this,stream_,decoded_,decoded_values_*sizeof(float)};
            split_cross_attention(query_,attended_,i<3?local:global,bounds,i<3,runtime_context_,invalid_,lease,stream_);
        }else cross_attention(query_,attended_,i<3?local:global,bounds,i<3,runtime_context_,invalid_,stream_);
        project(w.out,attended_,projected_);primitive::rms_norm(projected_,w.post,post_,1024,1,spec_.epsilon,stream_);
        primitive::add(input,post_,after_,1024,stream_);
        primitive::rms_norm(after_,w.ffn_pre,normalized_,1024,1,spec_.epsilon,stream_);
        project(w.gate,normalized_,gate_);project(w.up,normalized_,up_);
        gelu_product(gate_,up_,activation_,8192,stream_);project(w.down,activation_,down_);
        primitive::rms_norm(down_,w.ffn_post,post_,1024,1,spec_.epsilon,stream_);
        residual_scale(post_,after_,w.scale,output,1024,stream_);std::swap(input,output);
    }
    primitive::rms_norm(input,output_norm_,features_,1024,1,spec_.epsilon,stream_);
    project(head_,features_,logits_);project(post_projection_,features_,post_state_);
    finish<<<1,256,0,stream_>>>(logits_,post_state_,proposal_,invalid_);
    primitive::check_cuda(cudaGetLastError(),"Assistant full-vocabulary greedy/finite output");
}
void Draft::select_runtime_context(int context){
    if(context!=spec_.context&&context!=262144)throw std::invalid_argument("Assistant explicit runtime context");
    if(runtime_context_==context)return;
    primitive::check_cuda(cudaStreamSynchronize(stream_),"Assistant runtime context cold join");
    runtime_context_=context;
}
void Draft::select_split_attention(bool enabled){
    if(split_attention_==enabled)return;
    primitive::check_cuda(cudaStreamSynchronize(stream_),"Assistant split policy cold join");
    split_attention_=enabled;
}
}
