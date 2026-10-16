#pragma once
#include "llama-kv-cache-dsv4.h"
#include "llama-model.h"
#include "ggml-backend.h"
#include <array>
#include <memory>
#include <stdexcept>
namespace strata::deepseek_v4 {
inline void mtp_main_kv_probe(const llama_model& model,ggml_backend_t backend) {
    // Real 44-layer metadata, main compression ratios and rollback planes;
    // no target dense/expert weights are loaded or evaluated here.
    llama_kv_cache_dsv4 kv(model,GGML_TYPE_F16,GGML_TYPE_F16,false,true,false,true,12288,1,8192,256,2,0,nullptr,nullptr);
    kv.clear(true);ggml_backend_synchronize(backend);
    std::unique_ptr<ggml_context,decltype(&ggml_free)> meta(ggml_init({128*ggml_tensor_overhead(),nullptr,true}),ggml_free);
    const std::array<llama_dsv4_comp_state*,3> states{kv.get_csa_state(),kv.get_hca_state(),kv.get_lid_state()};
    std::array<float,512> values{};
    for(auto entry:states){int layer=0;while(layer<43&&model.hparams.dsv4_compress_ratios[layer]!=uint32_t(entry->get_ratio()))++layer;
        if(layer==43)throw std::runtime_error("MTP rollback state fixture layer absent");
        for(auto* t:{entry->get_kv_all(meta.get(),layer),entry->get_score_all(meta.get(),layer)}){
            if(!t||t->type!=GGML_TYPE_F32||ggml_nelements(t)<512)throw std::runtime_error("MTP rollback state geometry differs");
            ggml_backend_tensor_get(t,values.data(),ggml_nbytes(t)-sizeof(values),sizeof(values));
            for(float value:values)if(value!=0)throw std::runtime_error("MTP rollback snapshot tail not cleared");}}
    kv.clear(true);ggml_backend_synchronize(backend);
}
}
