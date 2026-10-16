#include "mtp_sidecar.hpp"
#include "gguf.h"
#include "ggml-cpu.h"
#include "llama-mmap.h"
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <vector>
namespace strata::deepseek_v4 {
struct MtpSidecar::Impl {
    std::unique_ptr<llama_file> file;
    std::unique_ptr<llama_mmap> mapping;
    gguf_context* metadata=nullptr;
    ggml_context* source=nullptr;
    ggml_context* dense=nullptr;
    ggml_backend_buffer_t host=nullptr,device=nullptr;
    std::map<std::string,ggml_tensor*> tensors;
    ~Impl(){if(device)ggml_backend_buffer_free(device);if(host)ggml_backend_buffer_free(host);
        if(dense)ggml_free(dense);if(source)ggml_free(source);if(metadata)gguf_free(metadata);}
};
MtpSidecar::MtpSidecar(const std::string& path,ggml_backend_t backend):impl_(std::make_unique<Impl>()) {
    auto& p=*impl_;if(!backend)throw std::runtime_error("MTP GPU backend missing");
    p.metadata=gguf_init_from_file(path.c_str(),{true,&p.source});
    if(!p.metadata||!p.source||gguf_get_n_tensors(p.metadata)!=32)throw std::runtime_error("MTP sidecar inventory differs");
    const auto arch=gguf_find_key(p.metadata,"general.architecture");
    if(arch<0||std::string(gguf_get_val_str(p.metadata,arch))!="strata-deepseek4-mtp")throw std::runtime_error("MTP source is not original single-layer sidecar");
    p.file=std::make_unique<llama_file>(path.c_str(),"rb");
    const char* priority=std::getenv("STRATA_DSV4_MTP_PIN_PRIORITY");
    if(priority&&std::strcmp(priority,"1")==0)
        p.mapping=std::make_unique<llama_mmap>(p.file.get(),0,false,llama_mmap::ranges{{0,p.file->size()}});
    else p.mapping=std::make_unique<llama_mmap>(p.file.get(),0);
    p.host=ggml_backend_cpu_buffer_from_ptr(p.mapping->addr(),p.mapping->size());
    if(!p.host)throw std::runtime_error("MTP immutable expert mapping unavailable");
    ggml_backend_buffer_set_usage(p.host,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    p.dense=ggml_init({64*ggml_tensor_overhead(),nullptr,true});
    if(!p.dense)throw std::runtime_error("MTP dense metadata unavailable");
    std::vector<std::pair<ggml_tensor*,ggml_tensor*>> uploads;
    for(int64_t i=0;i<gguf_get_n_tensors(p.metadata);++i) {
        const std::string name=gguf_get_tensor_name(p.metadata,i);
        auto* source=ggml_get_tensor(p.source,name.c_str());
        const size_t offset=gguf_get_data_offset(p.metadata)+gguf_get_tensor_offset(p.metadata,i);
        if(!source||offset>p.mapping->size()||ggml_nbytes(source)>p.mapping->size()-offset)throw std::runtime_error("MTP tensor span invalid");
        source->data=static_cast<char*>(p.mapping->addr())+offset;source->buffer=p.host;
        if(name.rfind("blk.43.ffn_",0)==0) {
            if(source->type!=GGML_TYPE_MXFP4||source->ne[2]!=256||ggml_nbytes(source)!=1140850688ull)throw std::runtime_error("MTP expert bank geometry differs");
            p.tensors.emplace(name,source);continue;
        }
        // Norms and router need F32 operands for existing fused CUDA kernels.
        const auto type=source->type==GGML_TYPE_BF16?GGML_TYPE_F32:source->type;
        auto* target=ggml_new_tensor(p.dense,type,ggml_n_dims(source),source->ne);
        ggml_set_name(target,name.c_str());p.tensors.emplace(name,target);uploads.emplace_back(target,source);
    }
    p.device=ggml_backend_alloc_ctx_tensors(p.dense,backend);
    if(!p.device||ggml_backend_buffer_get_size(p.device)>512ull<<20)throw std::runtime_error("MTP dense residency exceeds budget");
    ggml_backend_buffer_set_usage(p.device,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for(auto [target,source]:uploads) {
        if(source->type==GGML_TYPE_BF16){std::vector<float> values(ggml_nelements(source));
            ggml_bf16_to_fp32_row(static_cast<const ggml_bf16_t*>(source->data),values.data(),values.size());
            ggml_backend_tensor_set(target,values.data(),0,values.size()*sizeof(float));}
        else ggml_backend_tensor_set(target,source->data,0,ggml_nbytes(source));
    }
    ggml_backend_synchronize(backend);
}
MtpSidecar::~MtpSidecar()=default;
ggml_tensor* MtpSidecar::tensor(const std::string& name)const {
    const auto it=impl_->tensors.find(name);if(it==impl_->tensors.end())throw std::runtime_error("MTP tensor missing: "+name);return it->second;
}
size_t MtpSidecar::dense_device_bytes()const{return ggml_backend_buffer_get_size(impl_->device);}
}
