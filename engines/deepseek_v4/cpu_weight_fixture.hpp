#pragma once
#include "mxfp4_cpu_service.hpp"
#include "gguf.h"
#include "ggml-cpu.h"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include "llama-mmap.h"
#endif
#include <memory>
#include <stdexcept>
namespace strata::deepseek_v4 {
#ifdef _WIN32
struct CpuShardMap {
    HANDLE file=INVALID_HANDLE_VALUE,mapping=nullptr;void* address=nullptr;size_t bytes=0;
    explicit CpuShardMap(const char* path) {
        file=CreateFileA(path,GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr);
        LARGE_INTEGER length{};
        if(file==INVALID_HANDLE_VALUE||!GetFileSizeEx(file,&length)||length.QuadPart<=0)throw std::runtime_error("CPU shard open failed");
        bytes=size_t(length.QuadPart);mapping=CreateFileMappingW(file,nullptr,PAGE_READONLY,0,0,nullptr);
        if(mapping)address=MapViewOfFile(mapping,FILE_MAP_READ,0,0,0);
        if(!address)throw std::runtime_error("CPU read-only shard mapping failed");
    }
    ~CpuShardMap(){if(address)UnmapViewOfFile(address);if(mapping)CloseHandle(mapping);if(file!=INVALID_HANDLE_VALUE)CloseHandle(file);}
    void* addr() const {return address;}
    size_t size() const {return bytes;}
};
#endif
// Map one shard. Only selected layer-zero expert pages are read; no model,
// context, dense GPU weights or GGUF host registration is constructed.
struct CpuWeightFixture {
    gguf_context* metadata=nullptr;ggml_context* tensors=nullptr;ggml_backend_buffer_t buffer=nullptr;
#ifdef _WIN32
    std::unique_ptr<CpuShardMap> mapping;
#else
    std::unique_ptr<llama_file> file;std::unique_ptr<llama_mmap> mapping;
#endif
    Mxfp4CpuBank bank{};
    explicit CpuWeightFixture(const char* path){
        metadata=gguf_init_from_file(path,{true,&tensors});if(!metadata||!tensors)throw std::runtime_error("Expert fixture GGUF metadata missing");
#ifdef _WIN32
        mapping=std::make_unique<CpuShardMap>(path);
#else
        file=std::make_unique<llama_file>(path,"rb");mapping=std::make_unique<llama_mmap>(file.get(),0);
#endif
        buffer=ggml_backend_cpu_buffer_from_ptr(mapping->addr(),mapping->size());if(!buffer)throw std::runtime_error("Expert fixture CPU buffer failed");
        ggml_backend_buffer_set_usage(buffer,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        auto tensor=[&](const char* name){auto* t=ggml_get_tensor(tensors,name);const auto id=gguf_find_tensor(metadata,name);
            if(!t||id<0||t->type!=GGML_TYPE_MXFP4)throw std::runtime_error("Real main MXFP4 expert tensor absent");
            const size_t offset=gguf_get_data_offset(metadata)+gguf_get_tensor_offset(metadata,id);
            if(offset>mapping->size()||ggml_nbytes(t)>mapping->size()-offset)throw std::runtime_error("Expert fixture payload span invalid");
            t->data=static_cast<char*>(mapping->addr())+offset;t->buffer=buffer;return t;};
        bank={tensor("blk.0.ffn_gate_exps.weight"),tensor("blk.0.ffn_up_exps.weight"),tensor("blk.0.ffn_down_exps.weight"),10};
    }
    ~CpuWeightFixture(){if(buffer)ggml_backend_buffer_free(buffer);if(tensors)ggml_free(tensors);if(metadata)gguf_free(metadata);}
    uint64_t prefault(){uint64_t checksum=0;for(auto* t:{bank.gate,bank.up,bank.down}){const auto* p=static_cast<const volatile uint8_t*>(t->data);for(size_t i=0;i<ggml_nbytes(t);i+=4096)checksum+=p[i];}return checksum;}
};
}
