#include "mtp_state_owner.hpp"
#include "strata/qwen35/native/tiny_batch.hpp"
#include "strata/qwen35/native/mtp.hpp"
#include "strata/qwen35/native/ops.hpp"
#include "strata/qwen35/native/small_prefill.hpp"
#include <cmath>
#include <algorithm>
#include "mtp_graph.hpp"
#include "rmsconcat.hpp"
#include "rmsconcat_policy.hpp"
#include "prefill_trace.hpp"
#include "strata/qwen35/native/mtp_sampling.hpp"
#include "strata/qwen35/native/output_head.hpp"
namespace strata::qwen35::native {
void NativeMtp::condition(const int *ids,const float *main,int count,int position,cudaStream_t stream){
    const int h=int(model_.spec_.n_embd);
    if(!ids||!main||count<1||count>model_.max_tokens()||position<1||position+count-1>model_.logical_context())
        throw std::out_of_range("own MTP conditioning span");
    reserve_cache(position+count-1,stream);
    gather_embedding(model_.weights_.embedding,ids,embedding_,count,stream);
    model_.visual_embedding(embedding_,position,count,stream);
    attention_->visual_positions(model_.visual_positions());
    if(rmsconcat_uses(h,count)){
        rmsconcat(embedding_,enorm_,concat_,h,count,0,float(model_.spec_.rms_epsilon),stream);
        rmsconcat(main,hnorm_,concat_,h,count,h,float(model_.spec_.rms_epsilon),stream);
        ++rmsconcat_host_submissions();
    }else{
        rms_norm(embedding_,enorm_,normed_,h,count,float(model_.spec_.rms_epsilon),stream);
        check_cuda(cudaMemcpy2DAsync(concat_,size_t(2*h)*4,normed_,size_t(h)*4,size_t(h)*4,count,cudaMemcpyDeviceToDevice,stream),"draft embedding concat");
        rms_norm(main,hnorm_,normed_,h,count,float(model_.spec_.rms_epsilon),stream);
        check_cuda(cudaMemcpy2DAsync(concat_+h,size_t(2*h)*4,normed_,size_t(h)*4,size_t(h)*4,count,cudaMemcpyDeviceToDevice,stream),"draft target-state concat");
        counters_.D2D_bytes+=size_t(2*h)*count*4;
    }
    const auto &scratch=model_.shared_->shared_ffn(count).buffers();
    void *prepared=count==1?scratch.quant_input:scratch.half_input;
    const bool fp4=projection_.quant==Quant::NVFP4;
    if(fp4){}
    else if(count==1)vector_quantize(concat_,prepared,2*h,stream);
    else if(model_.shared_->tiny_batches(count))tiny_batch_quantize(concat_,prepared,2*h,count,stream);
    else if(count<64)small_prefill_quantize(concat_,prepared,2*h,count,stream);
    else to_half(concat_,prepared,size_t(2*h)*count,stream);
    model_.shared_->project(projection_,fp4?concat_:prepared,conditioned_,count,{},stream);
    if(graph_building_&&graph_batch_.start)mtp_positions(model_.positions_,graph_batch_.start,graph_offset_,count,1,
        graph_batch_.cursor,int(model_.spec_.n_head),attention_->decode_rounding(),stream);
    else set_positions(model_.positions_,position,count,stream);
}
void NativeMtp::enqueue_pick(const float *hidden,const float *norm,Matrix head,float *seed,cudaStream_t stream,unsigned offset){
    rms_norm(hidden,norm,seed,int(model_.spec_.n_embd),1,float(model_.spec_.rms_epsilon),stream);
    void *quant=model_.shared_->shared_ffn(1).buffers().quant_input;
    output_head_linear(head,model_.head_service_,seed,quant,logits_,1,stream);
    if(sampling_)sampling_->pick(logits_,offset,device_ids_,picked_,invalid_,stream);
    else greedy_pick(logits_,int(model_.spec_.n_vocab),greedy_,greedy_bytes_,picked_,invalid_,stream,model_.token_mask_);
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
int NativeMtp::draft(const int *ids,const float *main,int position,cudaStream_t stream,unsigned offset){
    if(graph_enabled()&&!graph_building_){
        graphs_->draft(int(offset),stream);counters_.D2D_bytes+=rmsconcat_copy_bytes(int(model_.spec_.n_embd),1);
        return read_draft(stream);
    }
    enqueue_draft(ids,main,position,stream,offset);
    if(graph_building_)return -1;
    return read_draft(stream);
}
void NativeMtp::enqueue_draft(const int *ids,const float *main,int position,cudaStream_t stream,unsigned offset){
    condition(ids,main,1,position,stream);attention_->rebind(decoder_);
    AttentionCache cache{key_,value_,position-1,position,capacity_,nullptr,0,model_.q8_cache()};
    if(graph_building_){cache.cursor=graph_batch_.cursor;cache.graph_partitions=graph_batch_.maximum_partitions;
        cache.graph_start=graph_batch_.start;cache.graph_shift=1;}
    cache.window=capacity_>model_.kv_window()?model_.kv_window():0;cache.graph_shift=1;
    cache.ram=ram_?&ram_->view():nullptr;
    attention_->execute(conditioned_,model_.positions_,cache,1,stream);
    enqueue_pick(model_.shared_->buffers().output,head_norm_,model_.weights_.head,draft_seed_,stream,offset);
}
void NativeMtp::catch_up(const int *ids,const float *main,int count,int position,cudaStream_t stream){
    condition(ids,main,count,position,stream);attention_->rebind(decoder_);
    AttentionCache cache{key_,value_,position-1,position-1+count,capacity_,nullptr,0,model_.q8_cache()};
    if(graph_building_&&graph_batch_.start){cache.graph_start=graph_batch_.start;cache.graph_shift=1;}
    cache.window=capacity_>model_.kv_window()?model_.kv_window():0;cache.graph_shift=1;
    cache.ram=ram_?&ram_->view():nullptr;
    attention_->cache_only(conditioned_,model_.positions_,cache,count,stream);
}
void NativeMtp::prefill(const int *ids,int tokens,cudaStream_t stream,int offset){
    normalize_state_owner();
    const int total=offset+tokens;
    if(tokens<1||offset<0||total>=model_.logical_context())throw std::out_of_range("native MTP prefix span");
    reserve_cache(total,stream);
    if(ram_&&(!graph_enabled()||graph_building_)&&!offset)ram_->reset(stream);
    if(graph_enabled()&&!graph_building_){
        graphs_->prefix(stream);
        counters_.D2D_bytes+=rmsconcat_prefix_copy_bytes(int(model_.spec_.n_embd),tokens,model_.max_tokens());return;
    }
    auto *trace=graph_building_?nullptr:active_prefill_trace;
    if(!graph_building_)model_.begin_prefill_progress(total,offset);
    if(tokens>model_.max_tokens()){
        const int h=int(model_.spec_.n_embd),width=model_.max_tokens();
        for(int i=0;i<tokens;i+=width){
            const int chunk=offset+i,n=std::min(width,tokens-i);
            const int timing=trace?trace->begin(chunk,n,stream):-1;
            model_.prefill_chunk(ids+i,chunk,n,chunk+n==total,stream);
            if(trace)trace->mark(timing,1,stream);
            rms_norm(model_.shared_->buffers().output,model_.weights_.output_norm,prefix_,h,n,
                     float(model_.spec_.rms_epsilon),stream);
            // Bridge h_(chunk-1) with x_chunk before reusing the retained seed.
            // Every MTP KV cell is written once, including chunk boundaries.
            if(chunk)catch_up(ids+i,seed_,1,chunk,stream);
            if(n>1)catch_up(ids+i+1,prefix_,n-1,chunk+1,stream);
            check_cuda(cudaMemcpyAsync(seed_,prefix_+size_t(n-1)*h,size_t(h)*4,cudaMemcpyDeviceToDevice,stream),"retain chunk target draft seed");
            counters_.D2D_bytes+=size_t(h)*4;
            if(trace)trace->mark(timing,2,stream);
        }
        set_positions(model_.positions_,total-1,1,stream);return;
    }
    const int timing=trace?trace->begin(offset,tokens,stream):-1;
    model_.prefill(ids,tokens,stream,offset);
    if(trace)trace->mark(timing,1,stream);
    const int h=int(model_.spec_.n_embd);
    rms_norm(model_.shared_->buffers().output,model_.weights_.output_norm,prefix_,h,tokens,float(model_.spec_.rms_epsilon),stream);
    check_cuda(cudaMemcpyAsync(seed_,prefix_+size_t(tokens-1)*h,size_t(h)*4,cudaMemcpyDeviceToDevice,stream),"retain real target draft seed");
    counters_.D2D_bytes+=size_t(h)*4;
    if(tokens>1)catch_up(ids+1,prefix_,tokens-1,offset+1,stream);
    set_positions(model_.positions_,total-1,1,stream);
    if(trace)trace->mark(timing,2,stream);
}
}
