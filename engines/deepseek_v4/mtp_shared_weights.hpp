#pragma once
#include "gguf.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "llama-mmap.h"
#include <memory>
#include <stdexcept>
#include <string>
namespace strata::deepseek_v4 {
// Diagnostic target owner: only its embedding/head, each allocated once.
struct MtpSharedWeights {
    gguf_context* metadata=nullptr;ggml_context *source=nullptr,*dense=nullptr;
    std::unique_ptr<llama_file> file;std::unique_ptr<llama_mmap> mapping;
    ggml_backend_buffer_t host=nullptr,device=nullptr;ggml_tensor *embedding=nullptr,*head=nullptr;
    MtpSharedWeights(const std::string& shard,ggml_backend_t backend) {
        metadata=gguf_init_from_file(shard.c_str(),{true,&source});if(!metadata||!source)throw std::runtime_error("Target head shard metadata missing");
        file=std::make_unique<llama_file>(shard.c_str(),"rb");mapping=std::make_unique<llama_mmap>(file.get(),0);
        dense=ggml_init({8*ggml_tensor_overhead(),nullptr,true});if(!dense)throw std::runtime_error("Target shared weight metadata failed");
        auto make=[&](const char* name){auto* t=ggml_get_tensor(source,name);if(!t||t->ne[0]!=4096||t->ne[1]!=129280)throw std::runtime_error("Target shared weight geometry differs");return ggml_dup_tensor(dense,t);};
        embedding=ggml_get_tensor(source,"token_embd.weight");if(!embedding||embedding->type!=GGML_TYPE_BF16||embedding->ne[0]!=4096||embedding->ne[1]!=129280)throw std::runtime_error("Target host embedding invalid");
        const size_t embedding_offset=gguf_get_data_offset(metadata)+gguf_get_tensor_offset(metadata,gguf_find_tensor(metadata,"token_embd.weight"));
        if(embedding_offset>mapping->size()||ggml_nbytes(embedding)>mapping->size()-embedding_offset)throw std::runtime_error("Target host embedding span invalid");
        host=ggml_backend_cpu_buffer_from_ptr(mapping->addr(),mapping->size());if(!host)throw std::runtime_error("Target host embedding buffer failed");ggml_backend_buffer_set_usage(host,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        embedding->data=static_cast<char*>(mapping->addr())+embedding_offset;embedding->buffer=host;
        head=make("output.weight");device=ggml_backend_alloc_ctx_tensors(dense,backend);if(!device)throw std::runtime_error("Target shared weight GPU allocation failed");
        ggml_backend_buffer_set_usage(device,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for(auto item:{std::make_pair(head,"output.weight")}){
            const auto id=gguf_find_tensor(metadata,item.second);const size_t offset=gguf_get_data_offset(metadata)+gguf_get_tensor_offset(metadata,id);
            if(offset>mapping->size()||ggml_nbytes(item.first)>mapping->size()-offset)throw std::runtime_error("Target shared tensor span invalid");
            ggml_backend_tensor_set(item.first,static_cast<const char*>(mapping->addr())+offset,0,ggml_nbytes(item.first));}
        ggml_backend_synchronize(backend);
    }
    ~MtpSharedWeights(){if(host)ggml_backend_buffer_free(host);if(device)ggml_backend_buffer_free(device);if(dense)ggml_free(dense);if(source)ggml_free(source);if(metadata)gguf_free(metadata);}
};
}
