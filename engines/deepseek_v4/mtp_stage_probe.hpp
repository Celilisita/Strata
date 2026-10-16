#pragma once
#include "device_cache_backend.hpp"
#include "mtp_sidecar.hpp"
#include "ggml-backend.h"
#include <array>
#include <chrono>
#include <cstring>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace strata::deepseek_v4 {
// Authentic MXFP4 payload, but no trunk model allocation or inference.
inline bool mtp_stage_probe(MtpSidecar& mtp,ggml_backend_t backend,std::ostream& out){
    std::array<ggml_tensor*,3> source{};
    size_t bytes=0;
    for(int i=0;i<3;++i){const char* role=i==0?"gate":i==1?"up":"down";
        source[i]=mtp.tensor(std::string("blk.43.ffn_")+role+"_exps.weight");bytes+=source[i]->nb[2];}
    std::unique_ptr<ggml_context,decltype(&ggml_free)> meta(ggml_init({ggml_tensor_overhead(),nullptr,true}),ggml_free);
    if(!meta)throw std::runtime_error("staging probe metadata failed");
    auto* dst=ggml_new_tensor_1d(meta.get(),GGML_TYPE_I8,bytes);
    std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(meta.get(),backend),ggml_backend_buffer_free);
    if(!buffer)throw std::runtime_error("staging probe device allocation failed");
    DeviceCacheDma dma(backend,nullptr);
    size_t offset=0;
    for(auto* tensor:source){dma.upload(dst,tensor->data,offset,tensor->nb[2]);offset+=tensor->nb[2];}
    dma.seal_background();const auto begin=std::chrono::steady_clock::now();
    while(!dma.background_ready()){
        if(std::chrono::steady_clock::now()-begin>std::chrono::seconds(3))throw std::runtime_error("staging probe background DMA timeout");
        std::this_thread::yield();
    }
    std::vector<unsigned char> actual(bytes);ggml_backend_tensor_get(dst,actual.data(),0,bytes);
    bool same=true;offset=0;
    for(auto* tensor:source){same&=std::memcmp(actual.data()+offset,tensor->data,tensor->nb[2])==0;offset+=tensor->nb[2];}
    const auto metrics=dma.metrics();const bool ok=same&&metrics.host_copy_bytes==bytes&&metrics.bytes==bytes&&metrics.calls==3;
    out<<"{\"whole_model_loaded\":false,\"background_only\":true,\"two_16mib_pinned_slabs\":true,\"payload_bytes\":"<<bytes
       <<",\"staged_bytes\":"<<metrics.host_copy_bytes<<",\"dma_calls\":"<<metrics.calls
       <<",\"payload_equal\":"<<(same?"true":"false")<<",\"passed\":"<<(ok?"true":"false")<<'}';
    return ok;
}
}
