#include "sm120_mtp_assistant_spec.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
namespace strata::gemma4::native::mtp_assistant {
namespace {
const MetaValue& meta(const GgufFile& f,const std::string& name){
    const auto* v=f.get("gemma4-assistant."+name);
    if(!v)throw std::runtime_error("Assistant missing metadata: "+name);return *v;
}
void expect(const GgufFile& f,const std::string& name,int expected){
    const auto& v=meta(f,name);
    if(!v.is_num()||v.type==MetaType::BOOL||v.type==MetaType::F32||v.type==MetaType::F64||v.u!=uint64_t(expected))
        throw std::runtime_error("Assistant metadata mismatch: "+name);
}
float positive(const GgufFile& f,const std::string& name){
    const auto& v=meta(f,name);const double n=v.num();
    if(!v.is_num()||!std::isfinite(n)||n<=0||n>(std::numeric_limits<float>::max)())
        throw std::runtime_error("Assistant numeric metadata: "+name);return float(n);
}
const TensorInfo* tensor(const GgufFile& f,const std::string& name,std::initializer_list<uint64_t> shape,uint32_t type){
    const auto* t=f.find(name);
    if(!t||t->shape!=std::vector<uint64_t>(shape)||t->type!=type)
        throw std::runtime_error("Assistant tensor shape/type: "+name);
    tensor_payload_bytes(f,*t);return t;
}
}
size_t tensor_payload_bytes(const GgufFile& f,const TensorInfo& t){
    size_t n=1;
    for(uint64_t d:t.shape){if(!d||d>(std::numeric_limits<size_t>::max)()/n)throw std::runtime_error("Assistant tensor overflow");n*=size_t(d);}
    const size_t block=t.type==0?1:32,bytes=t.type==0?4:18;
    if((t.type!=0&&t.type!=2)||n%block||n/block>(std::numeric_limits<size_t>::max)()/bytes)
        throw std::runtime_error("Assistant raw tensor encoding");
    n=n/block*bytes;
    if(f.data_start()>f.file_size()||t.offset>f.file_size()-f.data_start()||n>f.file_size()-f.data_start()-t.offset)
        throw std::runtime_error("Assistant tensor outside GGUF");return n;
}
Spec bind_spec(const GgufFile& f){
    const auto* arch=f.get("general.architecture");
    if(!arch||arch->type!=MetaType::STRING||arch->s!="gemma4-assistant"||f.tensors().size()!=49)
        throw std::runtime_error("Assistant exact Dense49 inventory required");
    for(auto pair:{std::pair<const char*,int>{"block_count",4},{"embedding_length",1024},{"feed_forward_length",8192},
        {"attention.head_count",32},{"attention.sliding_window",1024},{"attention.shared_kv_layers",4},
        {"embedding_length_out",5376},{"embedding_length_per_layer_input",0},{"nextn_predict_layers",4},
        {"attention.key_length",512},{"attention.value_length",512},{"attention.key_length_swa",256},
        {"attention.value_length_swa",256},{"rope.dimension_count",512},{"rope.dimension_count_swa",256}})
        expect(f,pair.first,pair.second);
    // Do not silently claim the official config's 256K for this 128K GGUF.
    expect(f,"context_length",131072);
    Spec s;s.context=131072;s.epsilon=positive(f,"attention.layer_norm_rms_epsilon");
    s.local_rope=positive(f,"rope.freq_base_swa");s.global_rope=positive(f,"rope.freq_base");
    const auto& pattern=meta(f,"attention.sliding_window_pattern");const auto& kv=meta(f,"attention.head_count_kv");
    if(pattern.type!=MetaType::ARRAY||kv.type!=MetaType::ARRAY||pattern.items.size()!=4||kv.items.size()!=4)
        throw std::runtime_error("Assistant attention topology arrays");
    s.head=tensor(f,"token_embd.weight",{1024,262144},2);
    s.pre_projection=tensor(f,"nextn.pre_projection.weight",{10752,1024},2);
    s.post_projection=tensor(f,"nextn.post_projection.weight",{1024,5376},2);
    s.output_norm=tensor(f,"output_norm.weight",{1024},0);s.frequencies=tensor(f,"rope_freqs.weight",{256},0);
    std::set<std::string> names;
    auto remember=[&](const TensorInfo* t){names.insert(t->name);};
    for(auto* t:{s.head,s.pre_projection,s.post_projection,s.output_norm,s.frequencies})remember(t);
    for(int i=0;i<4;++i){
        const bool local=i<3;
        if(pattern.items[i].type!=MetaType::BOOL||bool(pattern.items[i].u)!=local||!kv.items[i].is_num()||kv.items[i].u!=(local?16:4))
            throw std::runtime_error("Assistant type-shared KV topology mismatch");
        auto& l=s.layers[i];l.dimension=local?256:512;l.kv_heads=local?16:4;const std::string p="blk."+std::to_string(i)+".";
        auto m=[&](const char* name,int columns,int rows){auto* t=tensor(f,p+name+".weight",{uint64_t(columns),uint64_t(rows)},2);remember(t);return t;};
        auto v=[&](const char* name,int count){auto* t=tensor(f,p+name+".weight",{uint64_t(count)},0);remember(t);return t;};
        l.q=m("attn_q",1024,32*l.dimension);l.out=m("attn_output",32*l.dimension,1024);
        l.q_norm=v("attn_q_norm",l.dimension);l.pre=v("attn_norm",1024);l.post=v("post_attention_norm",1024);
        l.gate=m("ffn_gate",1024,8192);l.up=m("ffn_up",1024,8192);l.down=m("ffn_down",8192,1024);
        l.ffn_pre=v("ffn_norm",1024);l.ffn_post=v("post_ffw_norm",1024);l.scale=v("layer_output_scale",1);
    }
    std::vector<std::pair<uint64_t,uint64_t>> ranges;size_t total=0;
    for(const auto& t:f.tensors()){
        if(names.erase(t.name)!=1||t.offset%32)throw std::runtime_error("Assistant unknown/duplicate/unaligned tensor");
        const size_t bytes=tensor_payload_bytes(f,t);ranges.push_back({t.offset,t.offset+bytes});total+=bytes;
    }
    std::sort(ranges.begin(),ranges.end());
    for(size_t i=1;i<ranges.size();++i)if(ranges[i].first<ranges[i-1].second)throw std::runtime_error("Assistant payload overlap");
    if(!names.empty()||total!=264169488)throw std::runtime_error("Assistant actual payload inventory changed");
    return s;
}
}
