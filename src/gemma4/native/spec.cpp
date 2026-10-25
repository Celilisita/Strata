#include "strata/gemma4/native/spec.hpp"
#include "strata/qwen35/spec.hpp"
#include <cmath>
#include <stdexcept>
namespace strata::gemma4::native {
namespace {
const MetaValue& meta(const GgufFile& f,const std::string& key){
    const auto* value=f.get("gemma4."+key);
    if(!value)throw std::runtime_error("Gemma4 missing metadata: "+key);return *value;
}
int integer(const GgufFile& f,const std::string& key){
    const auto& v=meta(f,key);
    if(!v.is_num()||v.type==MetaType::F32||v.type==MetaType::F64||v.type==MetaType::BOOL||v.u>262144)
        throw std::runtime_error("Gemma4 integer metadata: "+key);return int(v.u);
}
float number(const GgufFile& f,const std::string& key){
    const auto& v=meta(f,key);
    if(!v.is_num()||!std::isfinite(v.num())||v.num()<=0)throw std::runtime_error("Gemma4 numeric metadata: "+key);
    return float(v.num());
}
const TensorInfo* tensor(const GgufFile& f,const std::string& name,std::initializer_list<uint64_t> shape,bool scalar=false){
    const auto* t=f.find(name);
    if(!t||t->shape!=std::vector<uint64_t>(shape)||(scalar?t->type!=0:(t->type!=12&&t->type!=14)))
        throw std::runtime_error("Gemma4 tensor shape/type: "+name);
    qwen35::tensor_bytes(f,*t);return t;
}
}
Spec bind_spec(const GgufFile& f){
    const auto* arch=f.get("general.architecture");
    if(!arch||arch->type!=MetaType::STRING||arch->s!="gemma4")throw std::runtime_error("Gemma4 Dense main architecture required");
    if(const auto* ex=f.get("gemma4.expert_count");ex&&(!ex->is_num()||ex->u))throw std::runtime_error("Gemma4 Dense cannot route experts");
    for(const auto& t:f.tensors())if(t.name.find("_exps.")!=std::string::npos||t.name.find("router")!=std::string::npos)
        throw std::runtime_error("Gemma4 Dense contains expert/router payload");
    Spec s;s.hidden=integer(f,"embedding_length");s.intermediate=integer(f,"feed_forward_length");
    s.heads=integer(f,"attention.head_count");s.context=integer(f,"context_length");s.window=integer(f,"attention.sliding_window");
    s.epsilon=number(f,"attention.layer_norm_rms_epsilon");s.logit_cap=number(f,"final_logit_softcapping");
    s.local_rope=number(f,"rope.freq_base_swa");s.global_rope=number(f,"rope.freq_base");
    if(integer(f,"block_count")!=60||s.hidden!=5376||s.intermediate!=21504||s.heads!=32||s.context!=262144||s.window!=1024||
        integer(f,"attention.shared_kv_layers")||integer(f,"embedding_length_per_layer_input")||
        integer(f,"attention.key_length")!=512||integer(f,"attention.value_length")!=512||
        integer(f,"attention.key_length_swa")!=256||integer(f,"attention.value_length_swa")!=256||
        integer(f,"rope.dimension_count")!=512||integer(f,"rope.dimension_count_swa")!=256)
        throw std::runtime_error("Gemma4 unsupported Dense main geometry");
    const auto* embedding=f.find("token_embd.weight");
    if(!embedding||embedding->shape.size()!=2||embedding->shape[1]!=262144)throw std::runtime_error("Gemma4 vocabulary extent");
    s.vocabulary=int(embedding->shape[1]);s.embedding=tensor(f,"token_embd.weight",{5376,262144});
    s.output=f.find("output.weight")?tensor(f,"output.weight",{5376,262144}):s.embedding;
    s.output_norm=tensor(f,"output_norm.weight",{5376},true);s.rope_frequencies=tensor(f,"rope_freqs.weight",{256},true);
    const auto& pattern=meta(f,"attention.sliding_window_pattern");const auto& kv=meta(f,"attention.head_count_kv");
    if(pattern.type!=MetaType::ARRAY||kv.type!=MetaType::ARRAY||pattern.items.size()!=60||kv.items.size()!=60)
        throw std::runtime_error("Gemma4 layer metadata arrays");
    for(int i=0;i<60;++i){
        const bool local=i%6!=5;
        if(pattern.items[i].type!=MetaType::BOOL||bool(pattern.items[i].u)!=local||!kv.items[i].is_num()||kv.items[i].u!=(local?16:4))
            throw std::runtime_error("Gemma4 actual SWA/global pattern mismatch");
        Layer l;l.sliding=local;l.head_dim=local?256:512;l.kv_heads=local?16:4;l.shared_raw_kv=!local;
        const std::string p="blk."+std::to_string(i)+".";
        auto matrix=[&](const char* name,int columns,int rows){return tensor(f,p+name+".weight",{uint64_t(columns),uint64_t(rows)});};
        auto norm=[&](const char* name,int count){return tensor(f,p+name+".weight",{uint64_t(count)},true);};
        l.q=matrix("attn_q",s.hidden,s.heads*l.head_dim);l.k=matrix("attn_k",s.hidden,l.kv_heads*l.head_dim);
        if(local)l.v=matrix("attn_v",s.hidden,l.kv_heads*l.head_dim);
        else{if(f.find(p+"attn_v.weight"))throw std::runtime_error("Gemma4 global raw K/V sharing changed");l.v=l.k;}
        l.out=matrix("attn_output",s.heads*l.head_dim,s.hidden);
        l.q_norm=norm("attn_q_norm",l.head_dim);l.k_norm=norm("attn_k_norm",l.head_dim);
        l.attn_norm=norm("attn_norm",s.hidden);l.attn_post=norm("post_attention_norm",s.hidden);
        l.gate=matrix("ffn_gate",s.hidden,s.intermediate);l.up=matrix("ffn_up",s.hidden,s.intermediate);
        l.down=matrix("ffn_down",s.intermediate,s.hidden);l.ffn_norm=norm("ffn_norm",s.hidden);
        l.ffn_post=norm("post_ffw_norm",s.hidden);l.scale=norm("layer_output_scale",1);s.layers.push_back(l);
    }
    return s;
}
}
