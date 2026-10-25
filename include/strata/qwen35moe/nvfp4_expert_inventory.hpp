#pragma once
#include "strata/qwen35moe/spec.hpp"
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::qwen35moe {
// The one model mapping remains authoritative for CPU, cache copies and GPU
// controls. Nothing here allocates/copies a weight or changes its byte layout.
struct Nvfp4ExpertSlice {
    const uint8_t *gate=nullptr,*up=nullptr,*down=nullptr;
    float gate_scale=1,up_scale=1,down_scale=1;
};
struct ExpertPayloadLayout {
    size_t gu=0,down=0;
    size_t bytes() const { return 2*gu+down; }
    size_t stride() const { return (bytes()+255)&~size_t(255); }
};
inline bool nvfp4_expert_flag(const char* raw) {
    if (!raw || !std::strcmp(raw,"0")) return false;
    if (!std::strcmp(raw,"1")) return true;
    throw std::invalid_argument("STRATA_QWEN35_MOE_NVFP4 requires0 or1");
}
inline bool nvfp4_main_stack(const Spec& spec) {
    size_t converted=0;
    for (const auto& layer:spec.layers) {
        if (!layer.gate_exps || !layer.up_exps || !layer.down_exps)
            throw std::invalid_argument("missing authoritative expert inventory");
        converted+=layer.gate_exps->type==40;
        converted+=layer.up_exps->type==40;
        converted+=layer.down_exps->type==40;
    }
    if (converted && converted!=spec.layers.size()*3)
        throw std::invalid_argument("NVFP4 requires the complete routed main stack");
    return converted!=0;
}
class Nvfp4ExpertInventory {
public:
    static constexpr size_t experts=256, matrix_bytes=589824;
    explicit Nvfp4ExpertInventory(const GgufFile& file,const Spec& spec) {
        if (spec.layers.size()!=40 || spec.n_expert!=experts || spec.n_embd!=2048 ||
            spec.n_ff_exp!=512 || !nvfp4_main_stack(spec))
            throw std::invalid_argument("NVFP4 routed inventory geometry");
        metadata(file);
        layers_.reserve(spec.layers.size());
        for (const auto& layer:spec.layers) {
            Layer target;
            target.gate=bind(file,*layer.gate_exps,2048,512,target.gate_scales);
            target.up=bind(file,*layer.up_exps,2048,512,target.up_scales);
            target.down=bind(file,*layer.down_exps,512,2048,target.down_scales);
            layers_.push_back(target);
        }
    }
    size_t layers() const { return layers_.size(); }
    Nvfp4ExpertSlice view(size_t layer,int expert) const {
        if (layer>=layers_.size() || expert<0 || expert>=int(experts))
            throw std::out_of_range("NVFP4 expert key");
        const auto& l=layers_[layer];const size_t i=size_t(expert),off=i*matrix_bytes;
        return {l.gate+off,l.up+off,l.down+off,
            l.gate_scales[i],l.up_scales[i],l.down_scales[i]};
    }
    static ExpertPayloadLayout layout() { return {matrix_bytes,matrix_bytes}; }
private:
    struct Layer {
        const uint8_t *gate=nullptr,*up=nullptr,*down=nullptr;
        std::array<float,experts> gate_scales{},up_scales{},down_scales{};
    };
    std::vector<Layer> layers_;
    static void metadata(const GgufFile& file) {
        auto integer=[&](const char* name,MetaType type,uint64_t expected){const auto* v=file.get(name);
            if(!v||v->type!=type||v->u!=expected)throw std::invalid_argument(std::string("NVFP4 metadata: ")+name);};
        auto text=[&](const char* name,const char* expected){const auto* v=file.get(name);
            if(!v||v->type!=MetaType::STRING||v->s!=expected)throw std::invalid_argument(std::string("NVFP4 metadata: ")+name);};
        integer("strata.moe.nvfp4.version",MetaType::U32,1);
        integer("strata.moe.nvfp4.scale_axis",MetaType::U32,2);
        text("strata.moe.nvfp4.scope","main_routed_experts");
        text("strata.moe.nvfp4.scale_suffix",".scale");
        for(const char* key:{"strata.source.gguf.sha256","strata.source.gguf.header_sha256"}) {
            const auto* v=file.get(key);
            if(!v||v->type!=MetaType::STRING||v->s.size()!=64||
               v->s.find_first_not_of("0123456789abcdef")!=std::string::npos)
                throw std::invalid_argument(std::string("NVFP4 source identity: ")+key);
        }
        const auto* size=file.get("strata.source.gguf.bytes");
        if(!size||size->type!=MetaType::U64||!size->u)throw std::invalid_argument("NVFP4 source size identity");
    }
    static void extent(const GgufFile& file,const TensorInfo& tensor,size_t bytes) {
        if (file.data_start()>file.file_size() ||
            tensor.offset>file.file_size()-file.data_start() ||
            bytes>file.file_size()-file.data_start()-tensor.offset)
            throw std::invalid_argument("NVFP4 tensor exceeds mapped GGUF: "+tensor.name);
    }
    static const uint8_t* bind(const GgufFile& file,const TensorInfo& tensor,
        uint64_t columns,uint64_t rows,std::array<float,experts>& scales) {
        if (tensor.type!=40 || tensor.shape!=std::vector<uint64_t>{columns,rows,experts} ||
            tensor.name.size()<7 || tensor.name.substr(tensor.name.size()-7)!=".weight")
            throw std::invalid_argument("NVFP4 routed tensor shape/type/name");
        extent(file,tensor,matrix_bytes*experts);
        const auto name=tensor.name.substr(0,tensor.name.size()-7)+".scale";
        const auto* global=file.find(name);
        if (!global || global->type!=0 || global->shape!=std::vector<uint64_t>{experts})
            throw std::invalid_argument("NVFP4 requires F32 per-expert scale: "+name);
        extent(file,*global,experts*sizeof(float));
        std::memcpy(scales.data(),file.tensor_data(*global),experts*sizeof(float));
        for (float value:scales) if (!(value>0) || !std::isfinite(value))
            throw std::invalid_argument("invalid NVFP4 per-expert global: "+name);
        return file.tensor_data(tensor);
    }
};
} // namespace strata::qwen35moe
