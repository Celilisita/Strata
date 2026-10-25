#include "strata/gemma4/native/model.hpp"
#include "attention_swa.hpp"
#include "strata/gemma4/native/pp_hh_only349_policy.hpp"
#include "attention_half_ops.hpp"
#include "attention_fused.hpp"
#include "strata/gemma4/native/vector_single.hpp"
#include "strata/gemma4/native/vector_striped_q4.hpp"
#include "strata/gemma4/native/vector_grouped_aligned_q4.hpp"
#include "strata/gemma4/native/qkv_single_t1.hpp"
#include "strata/qwen35/native/ops.hpp"
#include <algorithm>
#include <cmath>
namespace strata::gemma4::native {
float* Model::forward_hidden(const int* ids,int tokens){
    mark("begin",-1,tokens);
    embed(embedding_,ids,hidden_[0],tokens,std::sqrt(float(spec_.hidden)),invalid_,stream_);
    mark("embedding",-1,tokens);
    float* input=hidden_[0];float* output=hidden_[1];
    for(size_t index=0;index<layers_.size();++index){
        const auto& w=layers_[index];const auto& layer=spec_.layers[index];const auto c=cache_[index];
        primitive::rms_norm(input,w.pre,normalized_,spec_.hidden,tokens,spec_.epsilon,stream_);mark("attn_pre_norm",int(index),tokens);
        const bool single264=tokens==1&&(t1_paths_&16u);
        const bool single=single_projection(layer.sliding?0:1,tokens);
        const auto encoded=single?ffn_->prepare_linear_single(normalized_,spec_.hidden,tokens,stream_):SingleQ8View{};
        const auto arena=ffn_->attention_workspace();
        const QkvSingleBinding qkv_live{this,single264?++t1_arena_epoch_:t1_arena_epoch_,stream_};
        const auto qkv_single=single264?prepare_qkv_single_t1(normalized_,arena.data,arena.bytes,qkv_live):QkvSingleLease{};
        const bool prepared=!single&&!single264&&ffn_->prepare_vector_input(normalized_,spec_.hidden,tokens,stream_);
        const auto project_qkv=[&](primitive::Matrix matrix,float* output){
            if(single264){project_qkv_single_t1(matrix,qkv_single,output,qkv_live);++t1_single_qkv_enqueues_;}
            else if(single)ffn_->linear_single(matrix,encoded,output,stream_);
            else if(tokens==1&&(t1_paths_&1u)&&prepared&&matrix.quant==primitive::Quant::Q4K){
                linear_vector_grouped_aligned_q4_t1(matrix,arena.data,output,tokens,stream_);++t1_qkv_enqueues_;
            }else ffn_->linear(matrix,normalized_,output,tokens,stream_,false,prepared);
        };
        project_qkv(w.q,q_);mark("Q_projection",int(index),tokens);
        project_qkv(w.k,k_);mark("K_projection",int(index),tokens);
        if(layer.shared_raw_kv)head_norm_rope(k_,v_,nullptr,layer.head_dim,layer.kv_heads,tokens,spec_.epsilon,0,nullptr,position_,false,stream_);
        else{project_qkv(w.v,v_);head_norm_rope(v_,v_,nullptr,layer.head_dim,layer.kv_heads,tokens,spec_.epsilon,0,nullptr,position_,false,stream_);}
        mark("V_projection_norm",int(index),tokens);
        const float base=layer.sliding?spec_.local_rope:spec_.global_rope;const float* freq=layer.sliding?nullptr:frequencies_;
        head_norm_rope(q_,q_,w.qnorm,layer.head_dim,spec_.heads,tokens,spec_.epsilon,base,freq,position_,true,stream_);
        head_norm_rope(k_,k_,w.knorm,layer.head_dim,layer.kv_heads,tokens,spec_.epsilon,base,freq,position_,true,stream_);
        mark("QK_norm_rope",int(index),tokens);
        if(global_history_enabled_&&!layer.sliding&&history_owners_.at(index)){
            const auto& history=history_views_.at(index);
            history_write(k_,v_,history,position_,tokens,invalid_,history.binding);
        }else store_kv(k_,v_,c,position_,tokens,invalid_,stream_);
        mark("Q8_append",int(index),tokens);
        const auto* selected=global_history_view(index);
        if(tokens<=3&&selected){
            history_attention(q_,attended_,*selected,position_,spec_.heads,tokens,
                ffn_->attention_workspace(),selected->binding);
        }else if(tokens<=3&&split_decode_){
            if(!enqueue_parallel_global_verify(c,tokens)&&!enqueue_swa_warp_t1(c,tokens)&&!enqueue_parallel_global_t1(c,tokens)&&!enqueue_shared_global_attention(c,tokens)){
                split_attention(q_,attended_,c,position_,spec_.heads,tokens,ffn_->attention_workspace(),stream_);
                ++parent_decode_attention_enqueues_;
            }
        }
        else if(tokens==1280&&host_position_==0&&fused_prefill_)fused_float_attention(q_,attended_,c,position_,host_position_,spec_.heads,tokens,ffn_->attention_workspace(),stream_);
        else if(tokens==1280&&host_position_==0&&blas_prefill_&&half_prefill_)half_blas_attention(q_,attended_,c,position_,host_position_,spec_.heads,tokens,ffn_->attention_workspace(),stream_);
        else if(tokens>3&&blas_prefill_){
            const swa403::Binding live{this,reset_epoch_,stream_};
            if(c.window==swa403::Window&&(tokens==8192||tokens==6144)&&swa403::request_enabled(live))
                swa403::attention(q_,attended_,c,position_,host_position_,spec_.heads,tokens,
                    {ffn_->attention_workspace(),live},live);
            else if(!enqueue_physical_global_pp(c,tokens))
                blas_attention(q_,attended_,c,position_,host_position_,spec_.heads,tokens,ffn_->attention_workspace(),stream_);
        }
        else attention(q_,attended_,c,position_,spec_.heads,tokens,stream_);
        mark("attention",int(index),tokens);
        if(single_projection(layer.sliding?2:3,tokens)){
            const auto encoded_out=ffn_->prepare_linear_single(attended_,w.out.columns,tokens,stream_);
            if(tokens==1&&(t1_paths_&1024u)){
                linear_single_q4_striped_t1(w.out,encoded_out,projected_,stream_);++t1_out302_enqueues_;
            }else ffn_->linear_single(w.out,encoded_out,projected_,stream_);
        }else ffn_->linear(w.out,attended_,projected_,tokens,stream_,false);
        mark("Out_projection",int(index),tokens);
        primitive::rms_norm(projected_,w.post,post_,spec_.hidden,tokens,spec_.epsilon,stream_);
        primitive::add(input,post_,after_,size_t(tokens)*spec_.hidden,stream_);
        mark("attn_residual",int(index),tokens);
        ffn_->rebind(w.ffn);ffn_->run(after_,output,tokens,stream_);mark("FFN",int(index),tokens);std::swap(input,output);
    }
    return input;
}
void Model::forward(const int* ids,int tokens,bool head){
    float* input=forward_hidden(ids,tokens);
    if(head){
        primitive::rms_norm(input+size_t(tokens-1)*spec_.hidden,final_norm_,features_,spec_.hidden,1,spec_.epsilon,stream_);
        mark("head_norm",-1,tokens);
        project_head(features_,logits_,1,tokens==1);
        mark("head_projection",-1,tokens);
        if (sampling_active()) enqueue_sampling(tokens);
        else pick(logits_,partials_,next_,position_,invalid_,spec_.vocabulary,tokens,spec_.logit_cap,stream_);
        mark("sample_position",-1,tokens);
    }else advance(position_,tokens,capacity_,invalid_,stream_);
}
void Model::release_graph(){
    if(executable_)cudaGraphExecDestroy(executable_);if(graph_)cudaGraphDestroy(graph_);graph_=nullptr;executable_=nullptr;graph_replays_=0;
}
Model::~Model(){if(stream_)cudaStreamSynchronize(stream_);release_graph();}
int Model::prefill(const std::vector<int>& ids){
    if (sampling_) sampling_->begin(ids);
    if(ids.empty()||ids.size()+size_t(host_position_)>size_t(capacity_))throw std::invalid_argument("Gemma prompt exceeds reserved actual context");
    for(int id:ids)if(id<0||id>=spec_.vocabulary)throw std::invalid_argument("Gemma prompt token range");
    // Keep dispatch fixed for this request's Graph lifetime. Its kernels read
    // P+t on the GPU; they never branch on a capture-time host cursor.
    if(host_position_==0){
        global_shared_long_request_=ids.size()==30720&&capacity_==32768;
        global_verify321_request_=maximum_batch_==8192&&((ids.size()==8192&&capacity_==10240)||
            (ids.size()==30720&&capacity_==32768));
        // Record identity independently of the flag, for cold per-request A/B.
        physical_global_pp_request_=maximum_batch_==8192&&((ids.size()==30720&&capacity_==32768)||
            (pp349::is_k1024(pp_key_tile_)&&compensated_physical_global_pp_&&ids.size()==8192&&capacity_==10240));
    }
    for(size_t first=0;first<ids.size();first+=maximum_batch_){
        const int count=int((std::min)(size_t(maximum_batch_),ids.size()-first));
        primitive::check_cuda(cudaMemcpyAsync(input_,ids.data()+first,count*4,cudaMemcpyHostToDevice,stream_),"Gemma prompt IDs");
        forward(input_,count,first+count==ids.size());host_position_+=count;
    }
    int token;unsigned bad;primitive::check_cuda(cudaMemcpyAsync(&token,next_,4,cudaMemcpyDeviceToHost,stream_),"Gemma anchor");
    primitive::check_cuda(cudaMemcpyAsync(&bad,invalid_,4,cudaMemcpyDeviceToHost,stream_),"Gemma finite state");
    primitive::check_cuda(cudaStreamSynchronize(stream_),"Gemma full PP completion");if(bad)throw std::runtime_error("Gemma invalid full-model state");
    if (sampling_active()) sampling_->observe(token);
    return token;
}
int Model::step(bool graph_enabled){
    // Captured greedy picks carry fixed draw metadata. Sampled requests use
    // the normal eager path so every emitted token advances its own counter.
    if (sampling_active()) graph_enabled = false;
    if(profile_&&graph_enabled)throw std::invalid_argument("Gemma event attribution is eager only");
    if(host_position_>=capacity_)throw std::runtime_error("Gemma generation exceeds reserved context");
    if(graph_enabled&&!executable_){
        primitive::check_cuda(cudaStreamBeginCapture(stream_,cudaStreamCaptureModeThreadLocal),"Gemma decode Graph begin");
        forward(next_,1,true);primitive::check_cuda(cudaStreamEndCapture(stream_,&graph_),"Gemma decode Graph end");
        primitive::check_cuda(cudaGraphInstantiateWithFlags(&executable_,graph_,0),"Gemma decode Graph instantiate");
    }
    if(graph_enabled){primitive::check_cuda(cudaGraphLaunch(executable_,stream_),"Gemma complete decoder Graph");++graph_replays_;}
    else forward(next_,1,true);
    ++host_position_;
    int token;unsigned bad;primitive::check_cuda(cudaMemcpyAsync(&token,next_,4,cudaMemcpyDeviceToHost,stream_),"Gemma token");
    primitive::check_cuda(cudaMemcpyAsync(&bad,invalid_,4,cudaMemcpyDeviceToHost,stream_),"Gemma finite flag");
    primitive::check_cuda(cudaStreamSynchronize(stream_),"Gemma TG completion");if(bad)throw std::runtime_error("Gemma nonfinite decoder output");
    if (sampling_active()) sampling_->observe(token);
    return token;
}
}
