#include "strata/qwen35/completed_v100/native/tiny_batch.hpp"
#include "strata/qwen35/completed_v100/native/mtp.hpp"
#include "strata/qwen35/completed_v100/native/ops.hpp"
#include "strata/qwen35/completed_v100/native/small_prefill.hpp"
#include <cmath>
#include <algorithm>
#include "mtp_graph.hpp"
#include "strata/qwen35/completed_v100/native/original_formats.hpp"
#include "strata/qwen35/completed_v100/native/raw_q8_projection.hpp"
namespace strata::completed_v100::qwen35::native {
void NativeMtp::condition(const int *ids,const float *main,int count,int position,cudaStream_t stream){
    const int h=int(model_.spec_.n_embd);
    if(!ids||!main||count<1||count>model_.max_tokens()||position<1||position+count-1>model_.logical_context())
        throw std::out_of_range("own MTP conditioning span");
    reserve_cache(position+count-1,stream);
    gather_embedding(model_.weights_.embedding,ids,embedding_,count,stream);
    model_.visual_embedding(embedding_,position,count,stream);
    attention_->visual_positions(model_.visual_positions());
    rms_norm(embedding_,enorm_,normed_,h,count,float(model_.spec_.rms_epsilon),stream);
    check_cuda(cudaMemcpy2DAsync(concat_,size_t(2*h)*4,normed_,size_t(h)*4,size_t(h)*4,count,cudaMemcpyDeviceToDevice,stream),"draft embedding concat");
    rms_norm(main,hnorm_,normed_,h,count,float(model_.spec_.rms_epsilon),stream);
    check_cuda(cudaMemcpy2DAsync(concat_+h,size_t(2*h)*4,normed_,size_t(h)*4,size_t(h)*4,count,cudaMemcpyDeviceToDevice,stream),"draft target-state concat");
    counters_.D2D_bytes+=size_t(2*h)*count*4;
    const auto &scratch=model_.shared_->shared_ffn(count).buffers();
    void *prepared=count==1?scratch.quant_input:scratch.half_input;
    if(original_projection(projection_))prepared=concat_;
    else if(count==1)vector_quantize(concat_,prepared,2*h,stream);
    else if(model_.shared_->tiny_batches(count))tiny_batch_quantize(concat_,prepared,2*h,count,stream);
    else if(count<64)small_prefill_quantize(concat_,prepared,2*h,count,stream);
    else to_half(concat_,prepared,size_t(2*h)*count,stream);
    model_.shared_->project(projection_,prepared,conditioned_,count,{},stream);
    if(graph_building_&&graph_batch_.start)mtp_positions(model_.positions_,graph_batch_.start,graph_offset_,count,1,
        graph_batch_.cursor,int(model_.spec_.n_head),attention_->decode_rounding(),stream);
    else set_positions(model_.positions_,position,count,stream);
}
void NativeMtp::enqueue_pick(const float *hidden,const float *norm,Matrix head,float *seed,cudaStream_t stream){
    rms_norm(hidden,norm,seed,int(model_.spec_.n_embd),1,float(model_.spec_.rms_epsilon),stream);
    void *quant=model_.shared_->shared_ffn(1).buffers().quant_input;
    if(head.quant==Quant::Q8)model_.shared_->project(head,seed,logits_,1,{},stream);
    else {vector_quantize(seed,quant,int(model_.spec_.n_embd),stream);vector_linear(head,quant,logits_,stream);}
    greedy_pick(logits_,int(model_.spec_.n_vocab),greedy_,greedy_bytes_,picked_,invalid_,stream,model_.token_mask_);
}
int NativeMtp::read_draft(cudaStream_t stream){
    check_cuda(cudaMemcpyAsync(host_->p->ids,picked_,4,cudaMemcpyDeviceToHost,stream),"draft actual pick");
    check_cuda(cudaMemcpyAsync(host_->p->flags,invalid_,4,cudaMemcpyDeviceToHost,stream),"draft finite pick");
    check_cuda(cudaStreamSynchronize(stream),"draft actual pick completion");
    counters_.D2H_bytes+=8;++counters_.completion_joins;
    const int token=host_->p->ids[0];const unsigned flag=host_->p->flags[0];
    if(flag||token<0||token>=int(model_.spec_.n_vocab))throw std::runtime_error("nonfinite native draft logits");
    return token;
}
int NativeMtp::draft(const int *ids,const float *main,int position,cudaStream_t stream){
    if(graph_enabled()&&!graph_building_){
        const int step=main==seed_?0:1;graphs_->draft(step,stream);counters_.D2D_bytes+=size_t(2*model_.spec_.n_embd)*4;
        return read_draft(stream);
    }
    enqueue_draft(ids,main,position,stream);
    if(graph_building_)return -1;
    return read_draft(stream);
}
void NativeMtp::enqueue_draft(const int *ids,const float *main,int position,cudaStream_t stream){
    condition(ids,main,1,position,stream);attention_->rebind(decoder_);
    AttentionCache cache{key_,value_,position-1,position,capacity_,nullptr,0,model_.q8_cache()};
    if(graph_building_){cache.cursor=graph_batch_.cursor;cache.graph_partitions=graph_batch_.maximum_partitions;
        cache.graph_start=graph_batch_.start;cache.graph_shift=1;}
    cache.window=capacity_>model_.kv_window()?model_.kv_window():0;cache.graph_shift=1;
    cache.ram=ram_?&ram_->view():nullptr;
    attention_->execute(conditioned_,model_.positions_,cache,1,stream);
    enqueue_pick(model_.shared_->buffers().output,head_norm_,model_.weights_.head,draft_seed_,stream);
    if(sampling_) {
        const int row=int(ids-device_ids_);
        sampling_->pick(logits_,device_ids_,row,position-row,
            graph_building_?graph_batch_.start:nullptr,picked_,invalid_,stream);
    }
}
void NativeMtp::catch_up(const int *ids,const float *main,int count,int position,cudaStream_t stream){
    condition(ids,main,count,position,stream);attention_->rebind(decoder_);
    AttentionCache cache{key_,value_,position-1,position-1+count,capacity_,nullptr,0,model_.q8_cache()};
    if(graph_building_&&graph_batch_.start){cache.graph_start=graph_batch_.start;cache.graph_shift=1;}
    cache.window=capacity_>model_.kv_window()?model_.kv_window():0;cache.graph_shift=1;
    cache.ram=ram_?&ram_->view():nullptr;
    attention_->cache_only(conditioned_,model_.positions_,cache,count,stream);
}
void NativeMtp::prefill(const int *ids,int tokens,cudaStream_t stream){
    if(tokens<2||tokens>=model_.logical_context())throw std::out_of_range("native MTP prefix span");
    reserve_cache(tokens,stream);
    if(ram_&&(!graph_enabled()||graph_building_))ram_->reset(stream);
    if(graph_enabled()&&!graph_building_){
        const int chunks=(tokens+model_.max_tokens()-1)/model_.max_tokens();
        graphs_->prefix(stream);counters_.D2D_bytes+=size_t(model_.spec_.n_embd)*4*(chunks+2*(tokens-1));return;
    }
    if(tokens>model_.max_tokens()){
        const int h=int(model_.spec_.n_embd),width=model_.max_tokens();
        for(int start=0;start<tokens;start+=width){
            const int n=std::min(width,tokens-start);
            model_.prefill_chunk(ids+start,start,n,start+n==tokens,stream);
            rms_norm(model_.shared_->buffers().output,model_.weights_.output_norm,prefix_,h,n,
                     float(model_.spec_.rms_epsilon),stream);
            // Bridge h_(start-1) with x_start before reusing the retained seed.
            // Every MTP KV cell is written once, including chunk boundaries.
            if(start)catch_up(ids+start,seed_,1,start,stream);
            if(n>1)catch_up(ids+start+1,prefix_,n-1,start+1,stream);
            check_cuda(cudaMemcpyAsync(seed_,prefix_+size_t(n-1)*h,size_t(h)*4,cudaMemcpyDeviceToDevice,stream),"retain chunk target draft seed");
            counters_.D2D_bytes+=size_t(h)*4;
        }
        set_positions(model_.positions_,tokens-1,1,stream);return;
    }
    model_.prefill(ids,tokens,stream);
    const int h=int(model_.spec_.n_embd);
    rms_norm(model_.shared_->buffers().output,model_.weights_.output_norm,prefix_,h,tokens,float(model_.spec_.rms_epsilon),stream);
    check_cuda(cudaMemcpyAsync(seed_,prefix_+size_t(tokens-1)*h,size_t(h)*4,cudaMemcpyDeviceToDevice,stream),"retain real target draft seed");
    counters_.D2D_bytes+=size_t(h)*4;
    catch_up(ids+1,prefix_,tokens-1,1,stream);
    set_positions(model_.positions_,tokens-1,1,stream);
}
}
