#include "mtp_drafter.hpp"
#include "mtp_window.hpp"
#include "mtp_embedding.hpp"
#include "ggml-alloc.h"
#include "llama-context.h"
#include "llama-moe-cache.h"
#include "device_cache.hpp"
#include "native_tg_graph.hpp"
#include "router_prefetch.hpp"
#include "llama-model.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
namespace strata::deepseek_v4 {
struct MtpDrafter::Impl {
    MtpSidecar& weights;ggml_backend_t backend;Mxfp4CpuService& cpu;int tokens;
    ggml_context* meta=nullptr;ggml_cgraph *front=nullptr,*finish=nullptr;
    ggml_gallocr_t front_allocator=nullptr,finish_allocator=nullptr;
    ggml_tensor *embedded=nullptr,*embedding_source=nullptr,*hidden=nullptr,*input_ids=nullptr,*positions=nullptr,*mask=nullptr,*contributions=nullptr,*output=nullptr;
    ggml_tensor *confidence_probabilities=nullptr,*confidence_logits=nullptr;MtpConfidence confidence{};
    llama_context* target=nullptr;bool native=false,hashing=false;MtpPrepared prepared{};bool traced=false;std::unique_ptr<MtpWindow> window;ggml_cgraph* commit=nullptr;
    bool stage_trace=std::getenv("STRATA_DSV4_MTP_STAGE_TRACE")!=nullptr;
    bool async_input=false;ggml_backend_buffer_t input_stage=nullptr;
    char *stage_mask=nullptr,*stage_hidden=nullptr,*stage_embedding=nullptr,*stage_ids=nullptr,*stage_positions=nullptr;
    uint64_t stage_calls=0;std::array<double,5> stage_ms{};
    MtpStageMetrics request_stages{};
    std::array<uint64_t,2> stage_mode_calls{};std::array<std::array<double,5>,2> stage_mode_ms{};
    Impl(MtpSidecar& w,ggml_backend_t b,Mxfp4CpuService& pool,int n):weights(w),backend(b),cpu(pool),tokens(n){}
    ~Impl(){if(stage_trace&&stage_calls)std::fprintf(stderr,
        "strata_mtp_stage calls=%llu input_ms=%.4f front_ms=%.4f finish_ms=%.4f output_ms=%.4f commit_ms=%.4f native=%d\n",
        (unsigned long long)stage_calls,stage_ms[0],stage_ms[1],stage_ms[2],stage_ms[3],stage_ms[4],native?1:0);
        if(stage_trace)for(int mode=0;mode<2;++mode)if(stage_mode_calls[mode])std::fprintf(stderr,
            "strata_mtp_stage_mode async=%d calls=%llu input_ms=%.4f front_ms=%.4f finish_ms=%.4f output_ms=%.4f commit_ms=%.4f native=%d\n",
            mode,(unsigned long long)stage_mode_calls[mode],stage_mode_ms[mode][0],stage_mode_ms[mode][1],
            stage_mode_ms[mode][2],stage_mode_ms[mode][3],stage_mode_ms[mode][4],native?1:0);
        if(input_stage){ggml_backend_synchronize(backend);ggml_backend_buffer_free(input_stage);}
        if(finish_allocator)ggml_gallocr_free(finish_allocator);if(front_allocator)ggml_gallocr_free(front_allocator);if(meta)ggml_free(meta);}
    ggml_tensor* alias(ggml_tensor* t){auto* a=ggml_new_tensor(meta,t->type,ggml_n_dims(t),t->ne);a->data=t->data;a->buffer=t->buffer;ggml_set_input(a);return a;}
};
MtpDrafter::MtpDrafter(MtpSidecar& weights,ggml_backend_t backend,ggml_tensor* embedding,ggml_tensor* head,Mxfp4CpuService& cpu,int tokens,llama_context* target,llama_moe_cache* unified_cache,const MtpHashPrefetchInputs* hash_fixture)
    :impl_(std::make_unique<Impl>(weights,backend,cpu,tokens)) {
    auto& p=*impl_;p.target=target;if(target)prepare_device_expert_cache(target,tokens);if(tokens<1||tokens>2||!embedding||!head||!embedding->data||!head->data)throw std::runtime_error("MTP borrowed target weights invalid");
    p.meta=ggml_init({4096*ggml_tensor_overhead()+2*ggml_graph_overhead_custom(4096,false),nullptr,true});if(!p.meta)throw std::runtime_error("MTP graph metadata allocation failed");
    auto* c=p.meta;p.hidden=ggml_new_tensor_3d(c,GGML_TYPE_F32,4096,4,tokens);p.input_ids=ggml_new_tensor_1d(c,GGML_TYPE_I32,tokens);p.positions=ggml_new_tensor_1d(c,GGML_TYPE_I32,tokens);
    for(auto* t:{p.hidden,p.input_ids,p.positions})ggml_set_input(t);
    p.window=std::make_unique<MtpWindow>(weights,backend,embedding);
    p.mask=ggml_new_tensor_2d(c,GGML_TYPE_F16,256,64);ggml_set_input(p.mask);
    p.embedding_source=embedding;p.embedded=mtp_embedding_input(c,embedding,tokens);
    p.prepared=mtp_prepare(c,weights,embedding,p.input_ids,p.hidden,p.positions,p.window->tensor(),p.mask,p.embedded);
    p.front=ggml_new_graph_custom(c,4096,false);
    std::array<ggml_tensor*,3> hash_tables{},hash_directories{};
    p.hashing=(target||hash_fixture)&&tokens==1&&router_prefetch_enabled();
    if(p.hashing){
        // The finish graph borrows this input after the front graph returns.
        // INPUT alone allows gallocr to reuse its storage after its last front use.
        if(target||hash_fixture->retain_pending_token)ggml_set_output(p.input_ids);
        for(int layer=0;layer<3;++layer){
            hash_tables[layer]=p.alias(target?target->get_model().layers[layer].ffn_gate_tid2eid:hash_fixture->tables[layer]);
            hash_directories[layer]=p.alias(target?target->strata_moe_cache()->get_slot_map(layer,1,6):hash_fixture->directories[layer]);}
    }
    if(hash_tables[0])build_router_hash_prefetch(c,p.front,p.hidden,p.input_ids,p.input_ids,hash_tables.data(),hash_directories.data(),backend,false);
    for(auto* t:{p.prepared.activation,p.prepared.ids,p.prepared.weights,p.prepared.shared,p.prepared.residual,p.prepared.hc.post,p.prepared.hc.comb,p.prepared.next_kv}){ggml_set_output(t);ggml_build_forward_expand(p.front,t);}
    for(int i=0;i<ggml_graph_n_nodes(p.front);++i)if(!ggml_backend_supports_op(backend,ggml_graph_node(p.front,i)))throw std::runtime_error(std::string("MTP front unsupported CUDA op: ")+ggml_op_name(ggml_graph_node(p.front,i)->op));
    p.front_allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));if(!ggml_gallocr_alloc_graph(p.front_allocator,p.front))throw std::runtime_error("MTP front GPU allocation failed");
    auto inputs=p.prepared;inputs.weights=p.alias(inputs.weights);inputs.shared=p.alias(inputs.shared);inputs.residual=p.alias(inputs.residual);inputs.hc.post=p.alias(inputs.hc.post);inputs.hc.comb=p.alias(inputs.hc.comb);
    p.contributions=ggml_new_tensor_3d(c,GGML_TYPE_F32,4096,6,tokens);ggml_set_input(p.contributions);
    ggml_tensor* logits=nullptr;
    if(target||unified_cache){auto* cache=target ? target->strata_moe_cache() : unified_cache;if(!cache)throw std::runtime_error("MTP unified cache absent");
        auto* ids=p.alias(p.prepared.ids);auto* x=p.alias(p.prepared.activation);x=ggml_reshape_3d(c,x,4096,1,tokens);
        auto* lookup=ggml_get_rows(c,cache->get_slot_map(43,tokens,6),ggml_reshape_1d(c,ids,6*tokens));cache->strata_decorate_lookup(lookup,43);
        auto* gate=weights.tensor("blk.43.ffn_gate_exps.weight");auto* up=weights.tensor("blk.43.ffn_up_exps.weight");auto* down=weights.tensor("blk.43.ffn_down_exps.weight");
        auto* start=build_native_tg(c,x,ids,inputs.weights,ggml_reshape_2d(c,lookup,6,tokens),cache->get_experts(gate),cache->get_experts(up),cache->get_experts(down),gate,up,down,10,43);
        auto* joined=finish_native_tg(c,start,inputs.shared);if(!joined)throw std::runtime_error("MTP native expert graph absent");
        auto* hidden=mtp_hc_post(c,ggml_reshape_2d(c,joined,4096,tokens),inputs.residual,inputs.hc);
        logits=cache->strata_guard_logits(c,ggml_mul_mat(c,head,mtp_head_hidden(c,weights,hidden)));p.native=true;
    }else logits=mtp_finish(c,weights,inputs,p.contributions,head);p.output=ggml_argmax(c,logits);ggml_set_output(p.output);
    const char* confidence=std::getenv("STRATA_DSV4_MTP_CONFIDENCE_TRACE");
    if(tokens==1&&confidence&&!std::strcmp(confidence,"1")){
        const auto* vocab=logits;const int64_t count=vocab->ne[0];
        auto* top2=ggml_top_k(c,logits,2);
        p.confidence_logits=ggml_get_rows(c,ggml_reshape_2d(c,logits,1,count),top2);
        p.confidence_probabilities=ggml_get_rows(c,ggml_reshape_2d(c,ggml_soft_max(c,logits),1,count),top2);
        ggml_set_output(p.confidence_logits);ggml_set_output(p.confidence_probabilities);
    }
    p.finish=ggml_new_graph_custom(c,4096,false);ggml_build_forward_expand(p.finish,p.output);
    if(p.confidence_logits){ggml_build_forward_expand(p.finish,p.confidence_logits);ggml_build_forward_expand(p.finish,p.confidence_probabilities);}
    if(hash_tables[0])build_router_hash_prefetch(c,p.finish,inputs.shared,p.alias(p.input_ids),p.output,hash_tables.data(),hash_directories.data(),backend,true);
    for(int i=0;i<ggml_graph_n_nodes(p.finish);++i)if(!ggml_backend_supports_op(backend,ggml_graph_node(p.finish,i)))throw std::runtime_error(std::string("MTP head unsupported CUDA op: ")+ggml_op_name(ggml_graph_node(p.finish,i)->op));
    p.commit=ggml_new_graph_custom(c,4096,false);ggml_build_forward_expand(p.commit,p.window->append(c,p.alias(p.prepared.next_kv),tokens));
    p.finish_allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));if(!ggml_gallocr_alloc_graph(p.finish_allocator,p.finish))throw std::runtime_error("MTP finish GPU allocation failed");
    const char* async=std::getenv("STRATA_DSV4_MTP_ASYNC_INPUT");
    const bool requested=async&&std::strcmp(async,"1")==0;
    const char* aba=std::getenv("STRATA_DSV4_MTP_INPUT_ABA");
    if(requested||(aba&&std::strcmp(aba,"1")==0)){
        const auto type=ggml_backend_dev_host_buffer_type(ggml_backend_get_device(backend));
        if(!type||std::strcmp(ggml_backend_buft_name(type),"CUDA_Host"))throw std::runtime_error("MTP async input requires pinned CUDA host memory");
        const size_t mask_bytes=256*64*sizeof(ggml_fp16_t),hidden_bytes=size_t(tokens)*16384*sizeof(float);
        const size_t embedding_bytes=p.embedded?size_t(tokens)*4096*sizeof(float):0;
        const size_t ids_bytes=size_t(tokens)*sizeof(int32_t),positions_bytes=ids_bytes;
        const size_t total=mask_bytes+hidden_bytes+embedding_bytes+ids_bytes+positions_bytes;
        p.input_stage=ggml_backend_buft_alloc_buffer(type,total);
        if(!p.input_stage||!ggml_backend_buffer_is_host(p.input_stage)||ggml_backend_buffer_get_size(p.input_stage)<total)
            throw std::runtime_error("MTP pinned input allocation failed");
        auto* cursor=static_cast<char*>(ggml_backend_buffer_get_base(p.input_stage));
        if(!cursor)throw std::runtime_error("MTP pinned input address unavailable");
        p.stage_mask=cursor;cursor+=mask_bytes;p.stage_hidden=cursor;cursor+=hidden_bytes;
        p.stage_embedding=cursor;cursor+=embedding_bytes;p.stage_ids=cursor;cursor+=ids_bytes;p.stage_positions=cursor;
        p.async_input=requested;
    }
}
MtpDrafter::~MtpDrafter()=default;
void MtpDrafter::set_async_input(bool enabled){
    auto& p=*impl_;if(enabled&&!p.input_stage)throw std::runtime_error("MTP async input stage was not allocated");
    if(p.async_input!=enabled){ggml_backend_synchronize(p.backend);p.async_input=enabled;}
}
std::vector<int32_t> MtpDrafter::forward(const float* hidden,const int32_t* tokens,const int32_t* positions) {
    auto& p=*impl_;if(!hidden||!tokens||!positions)throw std::runtime_error("MTP input missing");
    using Clock=std::chrono::steady_clock;auto stage_start=p.stage_trace?Clock::now():Clock::time_point{};
    auto record=[&](int index){if(!p.stage_trace)return;const auto now=Clock::now();
        const auto ms=std::chrono::duration<double,std::milli>(now-stage_start).count();
        p.stage_ms[index]+=ms;p.stage_mode_ms[p.async_input?1:0][index]+=ms;
        p.request_stages.host_ms[index]+=ms;stage_start=now;};
    const auto cpu_before=p.stage_trace?mtp_cpu_service_snapshot():MtpCpuServiceSnapshot{};
    auto sub_start=p.stage_trace?Clock::now():Clock::time_point{};
    auto input_sub=[&](int index){if(!p.stage_trace)return;
        const auto now=Clock::now();p.request_stages.input_sub_ms[index]+=
            std::chrono::duration<double,std::milli>(now-sub_start).count();sub_start=now;};
    p.window->check(positions,p.tokens);input_sub(0);
    const size_t mask_bytes=256*64*sizeof(ggml_fp16_t),hidden_bytes=size_t(p.tokens)*16384*sizeof(float),ids_bytes=size_t(p.tokens)*sizeof(int32_t);
    // The front graph allocator can reuse the mask input storage after the
    // graph returns. Refresh it even when the sliding window length is stable.
    const int mask_rows=p.window->rows();const bool mask_changed=true;
    std::vector<ggml_fp16_t> owned_mask;
    auto* mask=p.async_input?reinterpret_cast<ggml_fp16_t*>(p.stage_mask):nullptr;
    if(mask_changed){
        if(!mask){owned_mask.assign(256*64,ggml_fp32_to_fp16(-INFINITY));mask=owned_mask.data();}
        else std::fill_n(mask,256*64,ggml_fp32_to_fp16(-INFINITY));
        for(int query=0;query<p.tokens;++query)for(int key=std::max(127-mask_rows,query);key<=127+query;++key)mask[query*256+key]=ggml_fp32_to_fp16(0);
    }
    input_sub(1);
    if(p.async_input){
        std::memcpy(p.stage_hidden,hidden,hidden_bytes);std::memcpy(p.stage_ids,tokens,ids_bytes);std::memcpy(p.stage_positions,positions,ids_bytes);
        if(p.embedded)mtp_prepare_embedding(reinterpret_cast<float*>(p.stage_embedding),p.embedding_source,tokens,p.tokens);
        input_sub(7);
        if(mask_changed)ggml_backend_tensor_set_async(p.backend,p.mask,p.stage_mask,0,mask_bytes);input_sub(2);
        ggml_backend_tensor_set_async(p.backend,p.hidden,p.stage_hidden,0,hidden_bytes);input_sub(3);
        if(p.hashing||!p.embedded)ggml_backend_tensor_set_async(p.backend,p.input_ids,p.stage_ids,0,ids_bytes);
        input_sub(5);
        if(p.embedded)ggml_backend_tensor_set_async(p.backend,p.embedded,p.stage_embedding,0,size_t(p.tokens)*4096*sizeof(float));
        input_sub(4);
        ggml_backend_tensor_set_async(p.backend,p.positions,p.stage_positions,0,ids_bytes);input_sub(6);
    }else{
        if(mask_changed)ggml_backend_tensor_set(p.mask,mask,0,mask_bytes);input_sub(2);
        if(p.hashing)ggml_backend_tensor_set(p.input_ids,tokens,0,sizeof(int32_t));input_sub(5);
        ggml_backend_tensor_set(p.hidden,hidden,0,hidden_bytes);input_sub(3);
        if(p.embedded)mtp_set_embedding_input(p.embedded,p.embedding_source,tokens,p.tokens);input_sub(4);
        if(!p.embedded)ggml_backend_tensor_set(p.input_ids,tokens,0,ids_bytes);
        input_sub(7);
        ggml_backend_tensor_set(p.positions,positions,0,ids_bytes);input_sub(6);
    }
    record(0);
    if(ggml_backend_graph_compute(p.backend,p.front)!=GGML_STATUS_SUCCESS)throw std::runtime_error("MTP forward front failed");
    record(1);
    if(!p.traced&&std::getenv("STRATA_DSV4_MTP_COMPONENT_TRACE"))for(int i=0;i<ggml_graph_n_nodes(p.front);++i){auto* t=ggml_graph_node(p.front,i);if(t->type!=GGML_TYPE_F32||std::string(ggml_get_name(t)).rfind("strata_mtp_",0)!=0)continue;
        std::vector<float> values(ggml_nelements(t));ggml_backend_tensor_get(t,values.data(),0,values.size()*sizeof(float));size_t invalid=0;float maximum=0;
        for(float value:values)if(!std::isfinite(value))++invalid;else maximum=std::max(maximum,std::abs(value));
        std::cerr<<ggml_get_name(t)<<" invalid="<<invalid<<" max_abs="<<maximum<<'\n';}
    p.traced=true;
    if(!p.native){
    std::array<float,8192> x{};std::array<int32_t,12> ids{};std::vector<float> contribution(p.tokens*6*4096);
    ggml_backend_tensor_get(p.prepared.activation,x.data(),0,p.tokens*4096*sizeof(float));ggml_backend_tensor_get(p.prepared.ids,ids.data(),0,p.tokens*6*sizeof(int32_t));
    const Mxfp4CpuBank bank{p.weights.tensor("blk.43.ffn_gate_exps.weight"),p.weights.tensor("blk.43.ffn_up_exps.weight"),p.weights.tensor("blk.43.ffn_down_exps.weight"),10};
    if(p.tokens==1)p.cpu.compute(bank,x.data(),ids.data(),6,contribution.data());else{const std::array<uint32_t,2> masks{63,63};p.cpu.compute_pair(bank,x.data(),ids.data(),masks.data(),contribution.data());}
    ggml_backend_tensor_set(p.contributions,contribution.data(),0,contribution.size()*sizeof(float));
    }
    if(ggml_backend_graph_compute(p.backend,p.finish)!=GGML_STATUS_SUCCESS)throw std::runtime_error("MTP forward head failed");
    record(2);
    std::vector<int32_t> result(p.tokens);ggml_backend_tensor_get(p.output,result.data(),0,result.size()*sizeof(int32_t));
    if(p.confidence_logits){float scores[2]{},probabilities[2]{};
        ggml_backend_tensor_get(p.confidence_logits,scores,0,sizeof(scores));
        ggml_backend_tensor_get(p.confidence_probabilities,probabilities,0,sizeof(probabilities));
        p.confidence={probabilities[0],scores[0]-scores[1],true};
        if(!std::isfinite(p.confidence.top1_probability)||!std::isfinite(p.confidence.top1_top2_logit_gap)||
           p.confidence.top1_probability<0||p.confidence.top1_probability>1||p.confidence.top1_top2_logit_gap<0)
            throw std::runtime_error("MTP diagnostic confidence is invalid");
    }
    record(3);
    if(ggml_backend_graph_compute(p.backend,p.commit)!=GGML_STATUS_SUCCESS)throw std::runtime_error("MTP KV commit failed");p.window->committed(positions,p.tokens);record(4);
    if(p.stage_trace){const auto cpu_after=mtp_cpu_service_snapshot();
        p.request_stages.cpu_miss_jobs+=cpu_after.jobs-cpu_before.jobs;
        p.request_stages.cpu_miss_compute_ns+=cpu_after.compute_ns-cpu_before.compute_ns;
        ++p.request_stages.calls;}
    ++p.stage_calls;++p.stage_mode_calls[p.async_input?1:0];return result;
}
void MtpDrafter::catch_up(const float* hidden,const int32_t* tokens,const int32_t* positions,int count){impl_->window->catch_up(hidden,tokens,positions,count);}
void MtpDrafter::reset(){impl_->window->reset();impl_->request_stages={};}
MtpStageMetrics MtpDrafter::stage_metrics() const{return impl_->request_stages;}
MtpConfidence MtpDrafter::last_confidence() const{return impl_->confidence;}
int MtpDrafter::history_rows()const{return impl_->window->rows();}
std::vector<int32_t> MtpDrafter::diagnostic_routes()const{std::vector<int32_t> ids(6*impl_->tokens);ggml_backend_tensor_get(impl_->prepared.ids,ids.data(),0,ids.size()*sizeof(int32_t));return ids;}
int32_t MtpDrafter::diagnostic_pending_token()const{int32_t token;ggml_backend_tensor_get(impl_->input_ids,&token,0,sizeof(token));return token;}
size_t MtpDrafter::device_scratch_bytes()const{return impl_->window->bytes()+ggml_gallocr_get_buffer_size(impl_->front_allocator,0)+ggml_gallocr_get_buffer_size(impl_->finish_allocator,0);}
}
