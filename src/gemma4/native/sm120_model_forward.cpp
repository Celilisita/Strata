#include "strata/gemma4/native/sm120_model.hpp"
#include "strata/qwen35/native/ops.hpp"
#include "sm120_rope_table.hpp"
#include "sm120_t1_rope_policy.hpp"
#include "sm120_t1_norm_cache.hpp"
#include "original_input_reuse.hpp"
#include "original_q6_head.hpp"
#include "sm120_native_group_codec.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace strata::gemma4::native {
float* Sm120Model::forward_hidden(const int* ids,int tokens){
    const bool input_reuse=original_input_reuse_requested();
    if(input_reuse&&projections_->format()!=GemmaWeightFormat::OriginalQuant)
        throw std::invalid_argument("Gemma original input reuse requires original_quant");
    const bool t1_table=t1_rope_table_requested();
    if(t1_table&&(!rope_table_||(projections_->format()!=GemmaWeightFormat::OriginalQuant&&
        projections_->format()!=GemmaWeightFormat::Nvfp4)))
        throw std::invalid_argument("Gemma T1 RoPE table requires its native backend and the owned PP table");
    const bool fusion=t1_norm_cache_fusion_requested();
    if(fusion&&!t1_table)throw std::invalid_argument("Gemma T1 norm/cache fusion requires explicit T1 owned-table mode");
    const bool fused=fusion&&tokens==1;
    const bool table=rope_table_&&(tokens>3||(tokens==1&&t1_table));
    if(table)prepare_rope_table(rope_table_,batch_,tokens,spec_.local_rope,spec_.global_rope,
        frequencies_,position_,invalid_,stream_);
    embed(embedding_,ids,hidden_[0],tokens,std::sqrt(5376.f),invalid_,stream_);
    auto* input=hidden_[0];auto* output=hidden_[1];
    for(std::size_t i=0;i<layers_.size();++i){
        const auto& w=layers_[i];const auto& layer=spec_.layers[i];
        primitive::rms_norm(input,w.pre,normalized_,5376,tokens,spec_.epsilon,stream_);
        {
            const bool reuse_qkv=input_reuse&&(tokens==1||(tokens>3&&
                (layer.shared_raw_kv||w.v.quant!=primitive::Quant::Q6K||tokens==batch_)));
            OriginalInputReuseScope group(projections_.get(),normalized_,tokens,5376,stream_,reuse_qkv);
            const bool native_group=projections_->format()==GemmaWeightFormat::Nvfp4&&projections_->tiny_compensated()&&native_group_codec_tokens(tokens);
            if(native_group){
                if(layer.shared_raw_kv)projections_->project_group(normalized_,{{w.q,q_},{w.k,k_}},tokens,stream_);
                else projections_->project_group(normalized_,{{w.q,q_},{w.k,k_},{w.v,v_}},tokens,stream_);
            }else{
                projections_->project(w.q,normalized_,q_,tokens,stream_);
                projections_->project(w.k,normalized_,k_,tokens,stream_);
            }
            if(layer.shared_raw_kv){
                // Same raw K projection, distinct normalized V. This precedes K's
                // weighted norm/RoPE and never borrows another layer's cache.
                if(!fused)head_norm_rope(k_,v_,nullptr,512,4,tokens,spec_.epsilon,0,nullptr,position_,false,stream_);
            }else{
                if(!native_group)projections_->project(w.v,normalized_,v_,tokens,stream_);
                if(!fused)head_norm_rope(v_,v_,nullptr,256,16,tokens,spec_.epsilon,0,nullptr,position_,false,stream_);
            }
        }
        const auto base=layer.sliding?spec_.local_rope:spec_.global_rope;
        const auto* frequencies=layer.sliding?nullptr:frequencies_;
        if(fused){
            t1_norm_cache_fused(q_,k_,v_,q_,k_,v_,w.qnorm,w.knorm,caches_[i],layer.shared_raw_kv,
                spec_.epsilon,rope_table_,batch_,position_,capacity_,invalid_,stream_);
        }else if(table){
            head_norm_rope_table(q_,q_,w.qnorm,layer.head_dim,32,tokens,spec_.epsilon,rope_table_,batch_,stream_);
            head_norm_rope_table(k_,k_,w.knorm,layer.head_dim,layer.kv_heads,tokens,spec_.epsilon,rope_table_,batch_,stream_);
        }else{
            head_norm_rope(q_,q_,w.qnorm,layer.head_dim,32,tokens,spec_.epsilon,base,frequencies,position_,true,stream_);
            head_norm_rope(k_,k_,w.knorm,layer.head_dim,layer.kv_heads,tokens,spec_.epsilon,base,frequencies,position_,true,stream_);
        }
        if(!fused)store_kv(k_,v_,caches_[i],position_,tokens,invalid_,stream_);
        attention_->enqueue(q_,attended_,caches_[i],position_,tokens,host_position_,stream_);
        projections_->project(w.out,attended_,projected_,tokens,stream_);
        primitive::rms_norm(projected_,w.post,post_,5376,tokens,spec_.epsilon,stream_);
        primitive::add(input,post_,after_,std::size_t(tokens)*5376,stream_);
        ffn_->run(w.ffn,after_,output,tokens,stream_);std::swap(input,output);
    }
    return input;
}
void Sm120Model::project_head_rows(const float* features,float* logits,int rows){
    if(rows>1&&original_q6_head_requested()){
        if(projections_->format()!=GemmaWeightFormat::OriginalQuant&&projections_->format()!=GemmaWeightFormat::Nvfp4)
            throw std::invalid_argument("Tied Q6 head requires its explicit native weight backend");
        original_q6_head_rows(embedding_,features,logits,rows,stream_);
        for(int row=0;row<rows;++row)softcap_logits_checked(logits+std::size_t(row)*262144,262144,30.f,invalid_,stream_);
        return;
    }
    for(int row=0;row<rows;++row){
        const auto* seed=features+std::size_t(row)*5376;
        auto* output=logits+std::size_t(row)*262144;
        if(original_q6_head_requested()){
            if(projections_->format()!=GemmaWeightFormat::OriginalQuant&&projections_->format()!=GemmaWeightFormat::Nvfp4)
                throw std::invalid_argument("Tied Q6 head requires its explicit native weight backend");
            original_q6_head(embedding_,seed,output,stream_);
        }else project_vector(embedding_,seed,output,stream_);
        softcap_logits_checked(output,262144,30.f,invalid_,stream_);
    }
}
void Sm120Model::forward(const int* ids,int tokens,bool head){
    auto* input=forward_hidden(ids,tokens);
    if(head){
        primitive::rms_norm(input+std::size_t(tokens-1)*5376,final_norm_,features_,5376,1,spec_.epsilon,stream_);
        project_head_rows(features_,logits_,1);
    }
    // MTP0 processed chunks retain their original single cursor advance.
    advance(position_,tokens,capacity_,invalid_,stream_);
}
void Sm120Model::enqueue_prefill(const std::vector<int>& ids){
    if(!capacity_||host_position_||ids.empty()||ids.size()>std::size_t(capacity_))
        throw std::invalid_argument("Gemma cold prefill extent");
    for(int id:ids)if(id<0||id>=262144)throw std::invalid_argument("Gemma prompt token range");
    const int tail=int(ids.size()%std::size_t(batch_));
    projections_->prepare(tail?std::vector<int>{1,batch_,tail}:std::vector<int>{1,batch_});
    prompt_=ids; // Own the asynchronous upload source until release/reset drains it.
    for(std::size_t first=0;first<prompt_.size();first+=batch_){
        const int count=int(std::min(std::size_t(batch_),prompt_.size()-first));
        primitive::check_cuda(cudaMemcpyAsync(input_,prompt_.data()+first,count*4,cudaMemcpyHostToDevice,stream_),"Gemma prompt IDs upload");
        forward(input_,count,first+count==prompt_.size());host_position_+=count;
    }
}
void Sm120Model::enqueue_step(const int* token,bool graph_enabled){
    if(!token||host_position_<1||host_position_>=capacity_)throw std::invalid_argument("Gemma decode input/context");
    if(!graph_enabled||!eager_decode_seen_){
        forward(token,1,true);eager_decode_seen_=true;
    }else{
        if(executable_&&graph_input_!=token)throw std::invalid_argument("Gemma captured token pointer changed");
        if(!executable_){
            primitive::check_cuda(cudaStreamBeginCapture(stream_,cudaStreamCaptureModeThreadLocal),"Gemma decode capture begin");
            try{forward(token,1,true);
                primitive::check_cuda(cudaStreamEndCapture(stream_,&graph_),"Gemma decode capture end");
                primitive::check_cuda(cudaGraphInstantiateWithFlags(&executable_,graph_,0),"Gemma decode instantiate");
                graph_input_=token;
            }catch(...){cudaGraph_t discarded=nullptr;cudaStreamEndCapture(stream_,&discarded);
                if(discarded)cudaGraphDestroy(discarded);release_graph();throw;}
        }
        primitive::check_cuda(cudaGraphLaunch(executable_,stream_),"Gemma decode replay");
    }
    ++host_position_;
}
void Sm120Model::finish(){
    if(!capacity_)throw std::invalid_argument("Gemma finish requires a prepared request");
    unsigned invalid;int position;
    primitive::check_cuda(cudaMemcpyAsync(&invalid,invalid_,sizeof(invalid),cudaMemcpyDeviceToHost,stream_),"Gemma finite state");
    primitive::check_cuda(cudaMemcpyAsync(&position,position_,sizeof(position),cudaMemcpyDeviceToHost,stream_),"Gemma cursor state");
    primitive::check_cuda(cudaStreamSynchronize(stream_),"Gemma queued operation completion");
    if(invalid||position!=host_position_)throw std::runtime_error("Gemma invalid numerical/cursor state");
}
}
