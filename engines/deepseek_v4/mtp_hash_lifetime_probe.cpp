#include "mtp_drafter.hpp"
#include "mtp_shared_weights.hpp"
#include "cpu_weight_fixture.hpp"
#include "router_prefetch.hpp"
#include "ggml-cuda.h"
#include "checkpoint.hpp"
#include <fstream>
#include <iostream>
#include <memory>
using namespace strata::deepseek_v4;
int main(int argc,char** argv)try{
    if(argc!=5)throw std::runtime_error("Expected MTP weights, activation, dense shard, report");
    auto* backend=ggml_backend_cuda_init(0);if(!backend)throw std::runtime_error("CUDA absent");
    const auto reg=ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));using Reserve=bool(*)(ggml_backend_t);
    auto reserve=reinterpret_cast<Reserve>(ggml_backend_reg_get_proc_address(reg,"strata_native_tg_reserve"));if(!reserve||!reserve(backend))throw std::runtime_error("Prefetch reserve absent");
    auto* packet=router_prefetch_packet(backend);if(!packet)throw std::runtime_error("Mapped packet absent");
    MtpSidecar mtp(argv[1],backend);MtpSharedWeights shared(argv[3],backend);Mxfp4CpuService cpu(8);
    auto fixture=read_checkpoint_data(argv[2],"ffn_norm-0");if(fixture.data.size()<4096*4)throw std::runtime_error("Activation absent");
    std::array<float,16384> hidden;const auto* x=reinterpret_cast<const float*>(fixture.data.data());for(int c=0;c<4;++c)for(int i=0;i<4096;++i)hidden[c*4096+i]=x[i]*(1+c*.025f);
    std::string hash_path=argv[3];const auto offset=hash_path.find("00004-of-00004.gguf");if(offset==std::string::npos)throw std::runtime_error("Dense fixture path invalid");hash_path.replace(offset,5,"00001");
    ggml_context* source=nullptr;auto* metadata=gguf_init_from_file(hash_path.c_str(),{true,&source});if(!metadata||!source)throw std::runtime_error("Hash metadata absent");CpuShardMap map(hash_path.c_str());
    auto* c=ggml_init({16*ggml_tensor_overhead(),nullptr,true});MtpHashPrefetchInputs inputs;
    std::array<ggml_tensor*,3> originals{};
    for(int layer=0;layer<3;++layer){auto name="blk."+std::to_string(layer)+".ffn_gate_tid2eid.weight";auto* t=ggml_get_tensor(source,name.c_str());const auto id=gguf_find_tensor(metadata,name.c_str());
        if(!t||id<0)throw std::runtime_error("Hash table absent");const size_t start=gguf_get_data_offset(metadata)+gguf_get_tensor_offset(metadata,id);
        if(start>map.size()||ggml_nbytes(t)>map.size()-start)throw std::runtime_error("Hash bounds invalid");t->data=static_cast<char*>(map.addr())+start;originals[layer]=t;
        inputs.tables[layer]=ggml_dup_tensor(c,t);inputs.directories[layer]=ggml_new_tensor_1d(c,GGML_TYPE_I32,256);}
    auto* buffer=ggml_backend_alloc_ctx_tensors(c,backend);if(!buffer)throw std::runtime_error("Hash allocation failed");
    std::array<int32_t,256> directory;directory.fill(-1);for(int layer=0;layer<3;++layer){ggml_backend_tensor_set(inputs.tables[layer],originals[layer]->data,0,ggml_nbytes(originals[layer]));ggml_backend_tensor_set(inputs.directories[layer],directory.data(),0,sizeof(directory));}
    const int32_t token=123,position=0;std::ofstream out(argv[4]);out<<"{\"whole_model_loaded\":false,\"real_mtp_front_expert_head_executed\":true,\"variants\":[";
    bool reproduced=false,fixed=false;
    for(int mode=0;mode<2;++mode){inputs.retain_pending_token=mode!=0;MtpDrafter draft(mtp,backend,shared.embedding,shared.head,cpu,1,nullptr,nullptr,&inputs);
        const auto before=packet->hash_sequence;const auto result=draft.forward(hidden.data(),&token,&position);const auto after=packet->hash_sequence;
        const auto actual=draft.diagnostic_pending_token();const auto& hash=packet->hash[after&1][0];
        if(mode==0)reproduced=actual!=token&&after-before==1&&hash.source==1;
        else {fixed=actual==token&&after-before==2&&hash.source==2;
            for(int layer=0;layer<3;++layer){const auto& entry=packet->hash[after&1][layer];const auto* table=static_cast<const int32_t*>(originals[layer]->data);
                for(int t=0;t<2;++t)for(int r=0;r<6;++r)fixed&=entry.ids[t*6+r]==table[(t?result[0]:token)*6+r];}}
        if(mode)out<<',';out<<"{\"retain_pending_token\":"<<(mode?"true":"false")<<",\"input_token\":"<<token<<",\"stored_token_after_front_finish\":"<<actual<<",\"hash_publications\":"<<after-before<<",\"latest_source\":"<<hash.source<<",\"draft_id_diagnostic\":"<<result[0]<<'}';
    }
    out<<"],\"unretained_lifetime_bug_reproduced\":"<<(reproduced?"true":"false")<<",\"retained_three_hash_layers_both_positions_correct\":"<<(fixed?"true":"false")<<",\"passed\":"<<(reproduced&&fixed?"true":"false")<<'}';
    if(!out)throw std::runtime_error("Report write failed");ggml_backend_synchronize(backend);ggml_backend_buffer_free(buffer);ggml_free(c);ggml_free(source);gguf_free(metadata);
    return reproduced&&fixed?0:2;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
