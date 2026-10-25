#include "mtp_session.hpp"
#include "mtp_options.hpp"
#include "mtp_finite_trace.hpp"
#include "phase_options.hpp"
#include "llama-context.h"
#include "llama-model.h"
#include "llama-memory.h"
#include "llama-memory-hybrid-idx.h"
#include "mtp_probe_io.hpp"
#include "llama-moe-cache.h"
#include "llama-batch.h"
#include "device_cache.hpp"
#include "phase_allocator.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
namespace strata::glm5_next {
struct MtpSession::Impl {
    llama_context& primary;
    const llama_model& model;
    llama_cparams cparams;
    llama_memory_ptr memory;
    ggml_backend_sched_t scheduler=nullptr;
    llm_graph_result result{16384};
    std::vector<ggml_backend_t> backends;
    std::vector<float> output;
    llama_pos next=0;
    bool retirement_registered=false;
    MtpDraftProfile draft_profile{mtp_draft_profile_requested()};
    MtpFiniteTrace finite_trace;
    explicit Impl(llama_context& source,ggml_type kv_type):primary(source),model(source.get_model()),cparams(source.get_cparams()) {
        if(model.arch!=LLM_ARCH_GLM5_NEXT||model.layers.size()!=46||!model.layers[45].nextn.eh_proj||
           !primary.strata_moe_cache()||cparams.n_seq_max!=1)
            throw std::runtime_error("GLM NextN requires the loaded original block and shared cache");
        primary.synchronize();
        try {
            auto* parent=primary.get_sched();
            for(int i=0;i<ggml_backend_sched_get_n_backends(parent);++i)
                backends.push_back(ggml_backend_sched_get_backend(parent,i));
            scheduler=ggml_backend_sched_new(backends.data(),nullptr,int(backends.size()),16384,false,false);
            if(!scheduler)throw std::runtime_error("NextN borrowed-backend scheduler unavailable");
            ggml_backend_sched_set_copy_callback(scheduler,[](ggml_backend_t backend,const ggml_tensor* src,
                ggml_tensor* dst,ggml_cgraph* graph,void* cache){
                return static_cast<llama_moe_cache*>(cache)->copy(backend,src,dst,graph);
            },primary.strata_moe_cache());
            cparams.ctx_type=LLAMA_CONTEXT_TYPE_MTP;cparams.embeddings=false;cparams.embeddings_nextn=false;
            cparams.embeddings_nextn_masked=true;cparams.nextn_layer_offset=0;
            memory.reset(model.create_memory({kv_type,kv_type,0,true,LLAMA_CONTEXT_TYPE_MTP,nullptr},cparams));
            if(!memory)throw std::runtime_error("NextN filtered memory unavailable");
            output.resize(model.vocab.n_tokens());
            // The shared cache calls this only at an actual capacity/phase
            // mutation boundary, before primary/CUDA graph retirement.
            if(phase_cache_requested()) {
                deepseek_v4::register_device_cache_graph_retirement(&primary,[](void* owner){
                    static_cast<Impl*>(owner)->retire();
                },this);
                retirement_registered=true;
            }
        }catch(...){if(scheduler)ggml_backend_sched_free(scheduler);scheduler=nullptr;throw;}
    }
    ~Impl(){
        // Runtime destroys this session before primary context/cache. Remove
        // its borrowed pointer before releasing the private scheduler/results.
        if(retirement_registered)deepseek_v4::register_device_cache_graph_retirement(&primary,nullptr,nullptr);
        if(scheduler){ggml_backend_sched_synchronize(scheduler);ggml_backend_sched_free(scheduler);}
    }
    void retire() {
        ggml_backend_sched_synchronize(scheduler);result.reset();ggml_backend_sched_release_buffers(scheduler);
    }
    void compute(const llama_token* ids,const float* hidden,int count,llama_pos first) {
        if(!ids||!hidden||count<1||count>int(cparams.n_batch)||first!=next||uint64_t(first)+count>cparams.n_ctx)
            throw std::runtime_error("NextN input position/size differs");
        const bool trace=finite_trace.begin(first,count,hidden);
        if(finite_trace.enabled())ggml_backend_sched_set_eval_callback(scheduler,trace?MtpFiniteTrace::callback:nullptr,trace?&finite_trace:nullptr);
        MtpDraftPhaseScope phase(draft_profile,MtpDraftStage::BatchMemory);
        llama_batch_ext input(count,4096,4096,1,memory.get(),model.vocab.n_tokens(),1);
        for(int row=0;row<count;++row) {
            const llama_pos pos=first+row;const int idx=input.add_token(0);
            llama_embd embd{};embd.data=hidden+size_t(row)*4096;embd.n_embd=4096;embd.n_rows=1;
            if(!input.set_token_id(idx,ids[row])||!input.set_token_embd(idx,embd)||
               !input.set_token_pos(idx,&pos)||!input.set_output(idx,row==count-1))
                throw std::runtime_error("NextN input construction failed");
        }
        llama_batch_allocr allocator(1);
        if(!allocator.init(input,model.vocab,false))throw std::runtime_error("NextN batch allocation failed");
        auto state=memory->init_batch(allocator,cparams.n_ubatch,false);
        if(!state||llama_memory_status_is_fail(state->get_status()))throw std::runtime_error("NextN memory batch failed");
        do {
            if(!state->apply())throw std::runtime_error("NextN state application failed");
            draft_profile.count(MtpDraftCount::Ubatches);phase.next(MtpDraftStage::GraphBuildAlloc);
            const auto& batch=state->get_ubatch();
            auto params=primary.strata_borrow_graph_params(&result,batch,state.get(),LLM_GRAPH_TYPE_DECODER_MTP);
            params.cparams=cparams;params.sched=scheduler;params.samplers.clear();params.n_outputs=1;
            params.cb=[this](const llama_ubatch& batch,ggml_tensor* tensor,const char* name,int layer) {
                if(layer>=0)ggml_format_name(tensor,"%s-%d",name,layer);else ggml_set_name(tensor,name);
                if(layer>=0&&(batch.n_tokens<32||model.n_gpu_layers()>model.hparams.n_layer_all)&&
                   (!std::strcmp(name,"norm")||!std::strcmp(name,"l_last")))
                    for(auto* backend:backends)if(ggml_backend_get_device(backend)==model.dev_layer(layer)&&ggml_backend_supports_op(backend,tensor))
                        ggml_backend_sched_set_tensor_backend(scheduler,tensor,backend);
            };
            if(!result.can_reuse(params)) {
                draft_profile.count(MtpDraftCount::RebuildAttempts);
                result.reset();ggml_backend_sched_reset(scheduler);
                // Quiescent borrowing changes graph placement only. Expert
                // inventory, directory, worker and lease owner stay shared.
                auto* old=primary.strata_borrow_cache_scheduler(scheduler);
                ggml_cgraph* graph=nullptr;
                try{graph=model.build_graph(params);}catch(...){primary.strata_borrow_cache_scheduler(old);throw;}
                primary.strata_borrow_cache_scheduler(old);
                if(!graph||!ggml_backend_sched_alloc_graph(scheduler,graph))throw std::runtime_error("NextN graph allocation failed");
                draft_profile.count(MtpDraftCount::BuildCompleted);
            }else draft_profile.count(MtpDraftCount::Reuse);
            phase.next(MtpDraftStage::SetInputs);
            result.set_inputs(&batch);
            phase.next(MtpDraftStage::GraphCompute);
            const auto compute_status=ggml_backend_sched_graph_compute(scheduler,result.get_gf());
            finite_trace.check();
            if(compute_status!=GGML_STATUS_SUCCESS)
                throw std::runtime_error("NextN graph computation failed");
            phase.next(MtpDraftStage::Sync);
            ggml_backend_sched_synchronize(scheduler);
            phase.next(MtpDraftStage::Logits);
            auto* logits=result.get_logits();
            if(!logits||logits->type!=GGML_TYPE_F32||ggml_nelements(logits)!=int64_t(output.size()))
                throw std::runtime_error("NextN output shape differs");
            ggml_backend_tensor_get(logits,output.data(),0,output.size()*sizeof(float));
            for(float value:output)if(std::isnan(value)||value==INFINITY)throw std::runtime_error("NextN logits nonfinite");
            finite_trace.finish(output);
            phase.next(MtpDraftStage::BatchMemory);
        }while(state->next());
        next=first+count;
    }
};
MtpSession::MtpSession(llama_context& primary,ggml_type kv_type):impl_(std::make_unique<Impl>(primary,kv_type)){}
MtpSession::~MtpSession()=default;
void MtpSession::reset(){impl_->retire();impl_->memory->clear(true);impl_->next=0;impl_->draft_profile.reset();}
MtpDraftProfile& MtpSession::draft_profile(){return impl_->draft_profile;}
void MtpSession::retire_graph(){impl_->retire();}
llama_pos MtpSession::position()const{return impl_->next;}
const std::vector<float>& MtpSession::logits()const{return impl_->output;}
void MtpSession::warm(const std::vector<llama_token>& ids,const std::vector<float>& hidden) {
    if(ids.empty()||hidden.size()!=ids.size()*4096||impl_->next)
        throw std::runtime_error("NextN full-prefix hidden capture differs");
    std::vector<float> previous;
    for(size_t first=0;first<ids.size();) {
        const int count=int(std::min<size_t>(impl_->cparams.n_batch,ids.size()-first));
        previous.resize(size_t(count)*4096);
        if(!first){std::fill_n(previous.data(),4096,0.f);if(count>1)std::copy_n(hidden.data(),size_t(count-1)*4096,previous.data()+4096);}
        else std::copy_n(hidden.data()+(first-1)*4096,size_t(count)*4096,previous.data());
        impl_->compute(ids.data()+first,previous.data(),count,llama_pos(first));first+=count;
    }
    impl_->retire(); // release PP scheduler workspace before primary cache grows
}
llama_token MtpSession::advance(llama_token token,const float* previous,llama_pos position) {
    impl_->draft_profile.count(MtpDraftCount::OneRow);
    impl_->compute(&token,previous,1,position);
    MtpDraftPhaseScope phase(impl_->draft_profile,MtpDraftStage::Greedy);
    return llama_token(std::max_element(impl_->output.begin(),impl_->output.end())-impl_->output.begin());
}
llama_token MtpSession::advance_pair(const llama_token* tokens,const float* previous,llama_pos position) {
    // Both positions update EH/attention/KV/indexer. The existing canonical
    // output mask crops the position-wise FFN/head to the consumed last row.
    impl_->draft_profile.count(MtpDraftCount::TwoRow);
    impl_->compute(tokens,previous,2,position);
    MtpDraftPhaseScope phase(impl_->draft_profile,MtpDraftStage::Greedy);
    return llama_token(std::max_element(impl_->output.begin(),impl_->output.end())-impl_->output.begin());
}
MtpProbeStep MtpSession::probe_step(const llama_token* ids,const float* previous,int count,llama_pos first) {
    if(count<1||count>8||count>int(impl_->cparams.n_ubatch))
        throw std::runtime_error("recursive NextN step must fit one existing ubatch, at most8 rows");
    impl_->compute(ids,previous,count,first);
    auto* hidden=impl_->result.get_h_nextn();
    if(!hidden||hidden->type!=GGML_TYPE_F32||!ggml_is_contiguous(hidden)||ggml_nelements(hidden)!=4096)
        throw std::runtime_error("recursive NextN post-norm hidden shape differs");
    MtpProbeStep out;
    ggml_backend_tensor_get(hidden,out.hidden.data(),0,sizeof(out.hidden));
    for(float value:out.hidden)if(!std::isfinite(value))throw std::runtime_error("recursive NextN hidden nonfinite");
    const auto best=std::max_element(impl_->output.begin(),impl_->output.end());
    if(best==impl_->output.end()||!std::isfinite(*best))throw std::runtime_error("recursive NextN has no finite logit");
    out.token=llama_token(best-impl_->output.begin());return out;
}
llama_token MtpSession::probe_advance(const llama_token* ids,const float* previous,int count,llama_pos first) {
    if(count<1||count>8||count>int(impl_->cparams.n_ubatch))throw std::runtime_error("diagnostic NextN last-head batch exceeds one ubatch");
    impl_->compute(ids,previous,count,first);
    const auto best=std::max_element(impl_->output.begin(),impl_->output.end());
    if(best==impl_->output.end()||!std::isfinite(*best))throw std::runtime_error("diagnostic NextN has no finite logit");
    return llama_token(best-impl_->output.begin());
}
MtpProbeSnapshot MtpSession::probe_snapshot() const {
    ggml_backend_sched_synchronize(impl_->scheduler);
    return save_mtp_probe_prefix(*impl_->memory,impl_->next);
}
void MtpSession::probe_restore(const MtpProbeSnapshot& snapshot) {
    impl_->retire();restore_mtp_probe_prefix(*impl_->memory,snapshot);impl_->next=snapshot.next;
}
void MtpSession::probe_rewind(llama_pos next) {
    if(next<0||next>impl_->next)throw std::runtime_error("recursive NextN rewind range differs");
    ggml_backend_sched_synchronize(impl_->scheduler);
    if(!impl_->memory->seq_rm(0,next,-1))throw std::runtime_error("recursive NextN MLA/indexer rewind failed");
    impl_->next=next;
}
MtpProbeState inspect_mtp_probe_memory(llama_memory_i& source,llama_pos next) {
    auto* memory=dynamic_cast<llama_memory_hybrid_idx*>(&source);
    if(!memory)throw std::runtime_error("recursive diagnostic memory type is not hybrid indexer");
    auto* attn=memory->get_mem_attn();auto* index=memory->get_mem_idx();
    if(!attn||!index)throw std::runtime_error("recursive NextN MLA/indexer inventory absent");
    MtpProbeState out;out.next=next;
    out.attn_min=attn->seq_pos_min(0);out.attn_max=attn->seq_pos_max(0);
    out.index_min=index->seq_pos_min(0);out.index_max=index->seq_pos_max(0);
    out.recurrent_empty=true;
    auto* recurrent=memory->get_mem_recr();
    // An empty module retains the configured member despite its zero-snapshot
    // constructor log; effective rollback capacity comes from physical storage.
    out.recurrent_configured_snapshots=recurrent->n_rs_seq;
    for(const auto& entry:recurrent->memory_breakdown())out.recurrent_storage_bytes+=entry.second;
    size_t kind=0;
    for(const auto* rows:{&recurrent->r_l,&recurrent->s_l,&recurrent->p_l}) {
        for(auto* tensor:*rows)if(tensor) {
            out.recurrent_empty=false;++out.recurrent_tensor_counts[kind];
            if(!recurrent->size||tensor->ne[1]<recurrent->size||tensor->ne[1]%recurrent->size)
                throw std::runtime_error("recursive diagnostic recurrent storage shape differs");
            const auto snapshots=uint32_t(tensor->ne[1]/recurrent->size-1);
            out.recurrent_snapshots=std::max(out.recurrent_snapshots,snapshots);
            out.recurrent_rollback_storage_bytes+=ggml_row_size(tensor->type,tensor->ne[0])*
                uint64_t(tensor->ne[1]-recurrent->size);
        }
        ++kind;
    }
    for(const auto& entry:memory->memory_breakdown())out.memory_bytes+=entry.second;
    out.attn_stream=attn->get_stream(0);out.index_stream=index->get_stream(0);
    out.attn_layers=attn->get_layer_ids();out.index_layers=index->get_layer_ids();
    const auto& a=attn->get_cells(0).seq_pos_get(0);const auto& b=index->get_cells(0).seq_pos_get(0);
    out.layout_valid=a==b&&attn->get_stream(0)==index->get_stream(0)&&out.recurrent_empty;
    for(const auto& cell:a)out.positions.push_back(cell.first);
    for(const auto& cell:b)out.index_positions.push_back(cell.first);
    auto ai=a.begin();auto bi=b.begin();
    while(ai!=a.end()&&bi!=b.end()&&*ai==*bi){++ai;++bi;}
    if(ai!=a.end()||bi!=b.end())out.first_layout_difference={
        ai==a.end()?-1:ai->first,ai==a.end()?-1:int64_t(ai->second),
        bi==b.end()?-1:bi->first,bi==b.end()?-1:int64_t(bi->second)};
    for(uint32_t i=0;i<attn->get_cells(0).size();++i)out.attn_scan_rows+=attn->get_cells(0).seq_has(i,0);
    for(uint32_t i=0;i<index->get_cells(0).size();++i)out.index_scan_rows+=index->get_cells(0).seq_has(i,0);
    return out;
}
MtpProbeState MtpSession::probe_state(bool payload) const {
    ggml_backend_sched_synchronize(impl_->scheduler);
    auto out=inspect_mtp_probe_memory(*impl_->memory,impl_->next);
    auto* memory=static_cast<llama_memory_hybrid_idx*>(impl_->memory.get());
    auto* attn=memory->get_mem_attn();auto* index=memory->get_mem_idx();
    for(auto* backend:impl_->backends)out.workspace_bytes+=ggml_backend_sched_get_buffer_size(impl_->scheduler,backend);
    if(payload)for(auto* cache:{attn,index}) {
        const auto layers=cache->get_layer_ids();
        if(layers!=std::vector<uint32_t>{45})throw std::runtime_error("recursive NextN owns an unexpected layer");
        auto* tensor=cache->get_k_storage(45);
        const auto* traits=ggml_get_type_traits(tensor->type);
        const size_t width=size_t(tensor->ne[0]);
        const size_t keep=cache==attn?width:size_t(impl_->model.hparams.indexer_head_size)*2;
        if(!traits->to_float||keep>width)throw std::runtime_error("recursive NextN KV format differs");
        std::vector<uint8_t> raw(ggml_row_size(tensor->type,width));std::vector<float> row(width);
        for(const auto& cell:cache->get_cells(0).seq_pos_get(0)) {
            const size_t offset=size_t(cell.second)*tensor->nb[1]+size_t(cache->get_stream(0))*tensor->nb[2];
            ggml_backend_tensor_get(tensor,raw.data(),offset,raw.size());traits->to_float(raw.data(),row.data(),width);
            for(size_t i=0;i<keep;++i)if(!std::isfinite(row[i]))throw std::runtime_error("recursive NextN live KV nonfinite");
            out.payload.insert(out.payload.end(),row.begin(),row.begin()+keep);
        }
    }
    return out;
}
MtpProbeWarm MtpSession::probe_warm(const std::vector<llama_token>& ids,const std::vector<float>& hidden) {
    if(ids.empty()||hidden.size()!=ids.size()*4096||impl_->next)
        throw std::runtime_error("recursive diagnostic warm prefix differs");
    MtpProbeWarm out;out.chunk_limit=int(std::min({impl_->cparams.n_batch,impl_->cparams.n_ubatch,uint32_t(32)}));
    constexpr uint64_t MiB=uint64_t(1)<<20,workspace_limit=3072*MiB;
    auto workspace=[&]{uint64_t bytes=0;for(auto* backend:impl_->backends)
        bytes+=ggml_backend_sched_get_buffer_size(impl_->scheduler,backend);return bytes;};
    std::vector<float> previous;
    for(size_t first=0;first<ids.size();) {
        const int count=int(std::min<size_t>(out.chunk_limit,ids.size()-first));
        size_t free=0,total=0;ggml_backend_dev_memory(impl_->model.dev_layer(45),&free,&total);
        // Admission estimate for attention slabs plus fixed one-row FFN/head.
        // 3GiB is the existing phase selection workspace allowance, not added VRAM.
        const uint64_t estimate=128*MiB+uint64_t(impl_->cparams.n_ctx)*count*impl_->model.hparams.n_head(45)*sizeof(float)*4;
        out.shape_estimate=std::max(out.shape_estimate,estimate);out.free_min=std::min(out.free_min,uint64_t(free));
        const auto current=workspace();
        if(estimate>workspace_limit||uint64_t(free)<64*MiB+(estimate>current?estimate-current:0)) {
            out.available=false;break;
        }
        previous.resize(size_t(count)*4096);
        if(!first){std::fill_n(previous.data(),4096,0.f);if(count>1)std::copy_n(hidden.data(),size_t(count-1)*4096,previous.data()+4096);}
        else std::copy_n(hidden.data()+(first-1)*4096,size_t(count)*4096,previous.data());
        impl_->compute(ids.data()+first,previous.data(),count,llama_pos(first));
        ++out.calls;out.input_rows+=count;first+=count;
        out.workspace_peak=std::max(out.workspace_peak,workspace());
        ggml_backend_dev_memory(impl_->model.dev_layer(45),&free,&total);out.free_min=std::min(out.free_min,uint64_t(free));
        if(out.workspace_peak>workspace_limit||free<64*MiB){out.available=false;break;}
    }
    impl_->retire();return out;
}
}
