#include "router_prediction.hpp"
#include "router_prefetch.hpp"
#include "cpu_weight_fixture.hpp"
#include "checkpoint.hpp"
#include "ggml-cuda.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>

using namespace strata::deepseek_v4;
int main(int argc,char** argv)try {
    if(argc!=4)throw std::runtime_error("Expected dense GGUF shard, checkpoints, report");
    ggml_context* source=nullptr;auto* metadata=gguf_init_from_file(argv[1],{true,&source});if(!metadata||!source)throw std::runtime_error("Fixture metadata absent");
    CpuShardMap mapping(argv[1]);auto tensor=[&](const char* name) {auto* t=ggml_get_tensor(source,name);const auto id=gguf_find_tensor(metadata,name);
        if(!t||id<0)throw std::runtime_error(std::string("Fixture missing:")+name);
        const size_t offset=gguf_get_data_offset(metadata)+gguf_get_tensor_offset(metadata,id);
        if(offset>mapping.size()||ggml_nbytes(t)>mapping.size()-offset)throw std::runtime_error("Fixture span invalid");t->data=static_cast<char*>(mapping.addr())+offset;return t;};
    auto* router_host=tensor("blk.3.ffn_gate_inp.weight");auto* norm_host=tensor("blk.3.ffn_norm.weight");auto* fn_host=tensor("blk.3.hc_ffn_fn.weight");
    auto* scale_host=tensor("blk.3.hc_ffn_scale.weight");auto* base_host=tensor("blk.3.hc_ffn_base.weight");auto* bias_host=tensor("blk.3.exp_probs_b.bias");
    auto raw_data=read_checkpoint_data(argv[2],"hc_attn_post-0");auto merged_data=read_checkpoint_data(argv[2],"hc_ffn_pre-0");
    if(raw_data.data.size()<2*16384*4||merged_data.data.size()<2*4096*4)throw std::runtime_error("Authentic hidden fixture too short");
    auto* backend=ggml_backend_cuda_init(0);if(!backend)throw std::runtime_error("CUDA unavailable");auto* reg=ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    using Reserve=bool(*)(ggml_backend_t);using Epoch=bool(*)(ggml_backend_t,uint32_t);using Read=bool(*)(ggml_backend_t,RouterPredictionTrace*,bool);
    auto reserve=reinterpret_cast<Reserve>(ggml_backend_reg_get_proc_address(reg,"strata_native_tg_reserve"));auto set_epoch=reinterpret_cast<Epoch>(ggml_backend_reg_get_proc_address(reg,"strata_router_prediction_set_probe_epoch"));
    auto read=reinterpret_cast<Read>(ggml_backend_reg_get_proc_address(reg,"strata_router_prediction_trace_read"));
    if(!reserve||!set_epoch||!read||!reserve(backend)||!set_epoch(backend,2))throw std::runtime_error("Prediction fixture registry absent");
    auto* ctx=ggml_init({512*ggml_tensor_overhead()+2*ggml_graph_overhead_custom(256,false),nullptr,true});
    auto clone=[&](const ggml_tensor* host){auto* t=ggml_new_tensor(ctx,host->type,ggml_n_dims(host),host->ne);ggml_set_input(t);ggml_set_output(t);return t;};
    auto* router=clone(router_host);auto* norm=clone(norm_host);auto* fn=clone(fn_host);auto* scale=clone(scale_host);auto* base=clone(base_host);auto* bias=clone(bias_host);
    auto* raw=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,4096,4,2);auto* merged=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,4096,2);auto* directory=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,256);
    for(auto* t:{raw,merged,directory}){ggml_set_input(t);ggml_set_output(t);}
    std::array<ggml_tensor*,3> normalized{},reference{};for(auto& t:normalized)t=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,4096,2);
    for(auto* t:normalized){ggml_set_input(t);ggml_set_output(t);}
    std::string hash_path=argv[1];const auto shard=hash_path.find("00004-of-00004.gguf");if(shard==std::string::npos)throw std::runtime_error("Dense fixture shard path invalid");hash_path.replace(shard,5,"00001");
    ggml_context* hash_source=nullptr;auto* hash_metadata=gguf_init_from_file(hash_path.c_str(),{true,&hash_source});
    if(!hash_metadata||!hash_source)throw std::runtime_error("Hash fixture metadata absent");CpuShardMap hash_mapping(hash_path.c_str());
    std::array<ggml_tensor*,3> tables{},host_tables{},directories{};
    for(int layer=0;layer<3;++layer){const auto name="blk."+std::to_string(layer)+".ffn_gate_tid2eid.weight";
        auto* host=ggml_get_tensor(hash_source,name.c_str());const auto id=gguf_find_tensor(hash_metadata,name.c_str());
        if(!host||id<0)throw std::runtime_error("Hash fixture tensor absent");const size_t offset=gguf_get_data_offset(hash_metadata)+gguf_get_tensor_offset(hash_metadata,id);
        if(offset>hash_mapping.size()||ggml_nbytes(host)>hash_mapping.size()-offset)throw std::runtime_error("Hash fixture bounds invalid");
        host->data=static_cast<char*>(hash_mapping.addr())+offset;host_tables[layer]=host;tables[layer]=clone(host);directories[layer]=directory;}
    auto* pending=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,1);auto* draft=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,1);
    for(auto* t:{pending,draft}){ggml_set_input(t);ggml_set_output(t);}
    auto* graph=ggml_new_graph_custom(ctx,256,false);RouterPredictionParams p{1e-20f,1e-6f,3,4,0};
    build_router_prediction(ctx,graph,nullptr,raw,merged,router,norm,fn,scale,base,bias,directory,backend,p);
    for(int v=0;v<3;++v){auto* scores=ggml_add(ctx,ggml_sqrt(ctx,ggml_softplus(ctx,ggml_mul_mat(ctx,router,normalized[v]))),bias);
        reference[v]=ggml_cont(ctx,ggml_top_k(ctx,scores,10));ggml_set_output(reference[v]);ggml_build_forward_expand(graph,reference[v]);}
    p.oracle=1;build_router_prediction(ctx,graph,nullptr,normalized[0],normalized[0],router,nullptr,nullptr,nullptr,nullptr,bias,directory,backend,p);
    p.oracle=2;build_router_prediction(ctx,graph,nullptr,raw,merged,router,norm,fn,scale,base,bias,directory,backend,p);
    build_router_hash_prefetch(ctx,graph,raw,pending,draft,tables.data(),directories.data(),backend,true);
    auto* allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));if(!ggml_gallocr_alloc_graph(allocator,graph))throw std::runtime_error("Prediction graph allocation failed");
    for(auto pair:{std::make_pair(router,router_host),{norm,norm_host},{fn,fn_host},{scale,scale_host},{base,base_host},{bias,bias_host}})
        ggml_backend_tensor_set(pair.first,pair.second->data,0,ggml_nbytes(pair.first));
    for(int layer=0;layer<3;++layer)ggml_backend_tensor_set(tables[layer],host_tables[layer]->data,0,ggml_nbytes(tables[layer]));
    const int32_t pending_id=123,draft_id=456;ggml_backend_tensor_set(pending,&pending_id,0,4);ggml_backend_tensor_set(draft,&draft_id,0,4);
    ggml_backend_tensor_set(raw,raw_data.data.data(),0,2*16384*4);ggml_backend_tensor_set(merged,merged_data.data.data(),0,2*4096*4);
    std::array<int32_t,256> resident;for(int i=0;i<256;++i)resident[i]=(i%3)?-1:i;ggml_backend_tensor_set(directory,resident.data(),0,sizeof(resident));
    std::array<std::array<float,8192>,3> values{};const auto* raw_x=reinterpret_cast<const float*>(raw_data.data.data());const auto* merge_x=reinterpret_cast<const float*>(merged_data.data.data());
    const auto* weights=static_cast<const float*>(fn_host->data);const auto* scales=static_cast<const float*>(scale_host->data);const auto* biases=static_cast<const float*>(base_host->data);
    for(int token=0;token<2;++token){const float* x=raw_x+token*16384;double sum=0;for(int i=0;i<16384;++i)sum+=double(x[i])*x[i];const double rms=1/std::sqrt(sum/16384+1e-20);
        std::array<double,4> hc{};for(int c=0;c<4;++c){double mix=0;for(int i=0;i<16384;++i)mix+=double(x[i])*weights[c*16384+i];hc[c]=1/(1+std::exp(-(mix*rms*scales[0]+biases[c])))+1e-6;}
        for(int v=0;v<3;++v){sum=0;for(int i=0;i<4096;++i){double value=0;if(v==1)value=merge_x[token*4096+i];else for(int c=0;c<4;++c)value+=x[c*4096+i]*(v==0?.25:hc[c]);values[v][token*4096+i]=float(value);sum+=value*value;}
            const double inverse=1/std::sqrt(sum/4096+1e-20);for(int i=0;i<4096;++i)values[v][token*4096+i]*=float(inverse)*static_cast<const float*>(norm_host->data)[i];}}
    for(int v=0;v<3;++v)ggml_backend_tensor_set(normalized[v],values[v].data(),0,sizeof(values[v]));
    for(auto pair:{std::make_pair(raw,raw_data.data.data()),std::make_pair(merged,merged_data.data.data())}){
        std::vector<uint8_t> copy(ggml_nbytes(pair.first));ggml_backend_tensor_get(pair.first,copy.data(),0,copy.size());
        if(std::memcmp(copy.data(),pair.second,copy.size()))throw std::runtime_error("Probe hidden GPU payload overwritten before compute");}
    {std::vector<uint8_t> copy(ggml_nbytes(norm));ggml_backend_tensor_get(norm,copy.data(),0,copy.size());if(std::memcmp(copy.data(),norm_host->data,copy.size()))throw std::runtime_error("Probe norm GPU payload overwritten");}
    for(int trial=0;trial<3;++trial)if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("Prediction GPU compute failed");
    for(auto pair:{std::make_pair(raw,raw_data.data.data()),std::make_pair(merged,merged_data.data.data())}){
        std::vector<uint8_t> copy(ggml_nbytes(pair.first));ggml_backend_tensor_get(pair.first,copy.data(),0,copy.size());
        if(std::memcmp(copy.data(),pair.second,copy.size()))throw std::runtime_error("Immutable probe hidden payload changed during compute");}
    ggml_backend_synchronize(backend);RouterPredictionTrace trace;if(!read(backend,&trace,true))throw std::runtime_error("Prediction fixture trace absent");
    for(int v=0;v<3;++v){std::array<float,8192> copy{};ggml_backend_tensor_get(normalized[v],copy.data(),0,sizeof(copy));
        if(copy!=values[v])throw std::runtime_error("Immutable numerical reference input changed during compute");}
    using ReadNormalized=bool(*)(ggml_backend_t,float*);
    auto get_normalized=reinterpret_cast<ReadNormalized>(ggml_backend_reg_get_proc_address(reg,"strata_router_prediction_probe_normalized"));
    std::array<std::array<float,8192>,3> actual{};
    if(!get_normalized||!get_normalized(backend,actual[0].data()))throw std::runtime_error("Prediction normalized fixture absent");
    for(int v=0;v<3;++v){double difference=0,signal=0;for(int i=0;i<8192;++i){difference+=std::pow(double(actual[v][i])-values[v][i],2);signal+=double(values[v][i])*values[v][i];}
        const auto rms=std::sqrt(difference/std::max(signal,1e-20));
        if(rms>1e-5)throw std::runtime_error("Normalization mismatch v="+std::to_string(v)+" rms="+std::to_string(rms)+" got="+std::to_string(actual[v][0])+" ref="+std::to_string(values[v][0])+" got1="+std::to_string(actual[1][0])+" ref1="+std::to_string(values[1][0])+" got2="+std::to_string(actual[2][0])+" ref2="+std::to_string(values[2][0]));}
    auto* packet=router_prefetch_packet(backend);if(!packet||!packet->sequence||!packet->hash_sequence)throw std::runtime_error("Mapped prefetch publication missing");
    const auto& learned=packet->learned[packet->sequence&1];
    if(learned.layer!=3||learned.epoch!=1||learned.tokens!=2)throw std::runtime_error("Mapped predictor metadata wrong");
    for(int token=0;token<2;++token)for(int rank=0;rank<6;++rank){const int id=trace.records[token*43+3].top[2][rank];
        if(learned.ids[token*6+rank]!=(resident[id]<0?id:-1)||bool(learned.masks[token]&(1u<<rank))!=(resident[id]<0))throw std::runtime_error("Mapped missing-only predictor IDs wrong");}
    for(int layer=0;layer<3;++layer){const auto& entry=packet->hash[packet->hash_sequence&1][layer];if(entry.epoch!=3||entry.layer!=layer||entry.tokens!=2||entry.source!=2)throw std::runtime_error("Hash metadata wrong");
        const auto* table=static_cast<const int32_t*>(host_tables[layer]->data);
        for(int token=0;token<2;++token)for(int rank=0;rank<6;++rank){const int id=table[(token?draft_id:pending_id)*6+rank];
            if(entry.ids[token*6+rank]!=(resident[id]<0?id:-1))throw std::runtime_error("Hash exact token/expert mapping wrong");}}
    double max_score_error=0;int reference_rank_differences=0;
    for(int v=0;v<3;++v)for(int token=0;token<2;++token){
        std::array<double,256> scores{};const auto* bf16=static_cast<const uint16_t*>(router_host->data);
        const auto* correction=static_cast<const float*>(bias_host->data);
        for(int expert=0;expert<256;++expert){double dot=0;
            for(int i=0;i<4096;++i){const uint32_t bits=uint32_t(bf16[expert*4096+i])<<16;float weight;std::memcpy(&weight,&bits,4);dot+=double(weight)*values[v][token*4096+i];}
            scores[expert]=std::sqrt(dot>20?dot:std::log1p(std::exp(dot)))+correction[expert];}
        std::array<int,256> order;std::iota(order.begin(),order.end(),0);std::sort(order.begin(),order.end(),[&](int a,int b){return scores[a]>scores[b]||(scores[a]==scores[b]&&a<b);});
        const auto& row=trace.records[token*43+3];
        std::array<int32_t,20> ids{};ggml_backend_tensor_get(reference[v],ids.data(),0,sizeof(ids));
        for(int rank=0;rank<10;++rank){const int id=row.top[v][rank];if(id<0||id>=256||!std::isfinite(row.score[v][rank]))throw std::runtime_error("Invalid prediction result");
            const double error=std::abs(row.score[v][rank]-scores[id]);max_score_error=std::max(max_score_error,error);
            const double tolerance=1e-4*(1+std::abs(scores[id]));
            if(error>tolerance||scores[order[rank]]-scores[id]>2*tolerance)throw std::runtime_error("BF16/F32 prediction numerical reference failed");
            reference_rank_differences+=id!=ids[token*10+rank];
            if(v==0&&(row.top[3][rank]!=id||std::abs(row.score[3][rank]-row.score[v][rank])>tolerance))throw std::runtime_error("Same-input oracle differs");}}
    using Timing=bool(*)(ggml_backend_t,uint64_t*);auto timing=reinterpret_cast<Timing>(ggml_backend_reg_get_proc_address(reg,"strata_router_prefetch_probe_timing"));
    if(!timing)throw std::runtime_error("Prefetch timing API absent");
    auto* timed_graph=ggml_new_graph_custom(ctx,256,false);p.oracle=2;
    build_router_prediction(ctx,timed_graph,nullptr,raw,merged,router,norm,fn,scale,base,bias,directory,backend,p);
    auto* timed_allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));if(!ggml_gallocr_alloc_graph(timed_allocator,timed_graph))throw std::runtime_error("Timing graph allocation failed");
    std::array<std::vector<double>,4> spans;
    for(int trial=0;trial<35;++trial){if(ggml_backend_graph_compute(backend,timed_graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("Timing compute failed");uint64_t clock[4];if(!timing(backend,clock))throw std::runtime_error("Clock not enabled");
        if(!(clock[0]<=clock[1]&&clock[1]<=clock[2]&&clock[2]<=clock[3]))throw std::runtime_error("Router GPUclock order invalid");
        if(trial>=3){spans[0].push_back(double(clock[1]-clock[0])/1000);spans[1].push_back(double(clock[2]-clock[1])/1000);spans[2].push_back(double(clock[3]-clock[2])/1000);spans[3].push_back(double(clock[3]-clock[0])/1000);}}
    for(auto& v:spans)std::sort(v.begin(),v.end());ggml_gallocr_free(timed_allocator);
    std::ofstream result(argv[3]);result<<"{\"whole_model_loaded\":false,\"actual_router_bf16_and_mhc_f32\":true,\"authentic_hidden_states\":true,\"positions\":2,\"prediction_cases\":3,\"bf16_f32_top10_score_and_order_tolerance_passed\":true,\"ggml_reference_rank_differences\":"<<reference_rank_differences<<",\"max_score_absolute_error\":"<<max_score_error<<",\"mapped_missing_only_ids_checked\":true,\"three_hash_layers_checked\":true,\"formal_GPU_median_us\":{\"normalization_block0\":"<<spans[0][16]<<",\"projection_and_kernel_gap\":"<<spans[1][16]<<",\"top6_and_mapped_publication\":"<<spans[2][16]<<",\"total\":"<<spans[3][16]<<"},\"oracle_matches\":true,\"immutable_payload_check\":true,\"passed\":true}";
    if(!result)throw std::runtime_error("Prediction receipt write failed");ggml_gallocr_free(allocator);ggml_free(ctx);ggml_backend_free(backend);ggml_free(source);gguf_free(metadata);ggml_free(hash_source);gguf_free(hash_metadata);return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
