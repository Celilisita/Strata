#pragma once
#include "strata/qwen35/native/weights.hpp"
#include <array>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strata::qwen35::native {
inline bool q36_nextn_nvfp4_requested(){
    const char* raw=std::getenv("STRATA_DENSE_Q36_NEXTN_NVFP4");
    const std::string mode=raw?raw:"0";
    if(mode!="0"&&mode!="1")throw std::invalid_argument("Q36 nextN NVFP4 requires0or1");
    return mode=="1";
}
inline bool configure_q36_nextn_nvfp4(const std::string& variant,int proposals){
    const bool enabled=q36_nextn_nvfp4_requested();
    if(!enabled)return false;
    if(variant!="qwen36_27b"||proposals<1)
        throw std::invalid_argument("Q36 nextN NVFP4 is an explicit Q36 MTP-only option");
    const char* planning=std::getenv("STRATA_DENSE_Q36_STARTUP_BODY_PLAN");
    if(planning&&std::string(planning)!="0")
        throw std::invalid_argument("Q36 nextN NVFP4 requires startup400 body planning off");
    return true;
}
inline void validate_q36_nextn_nvfp4_asset(const strata::GgufFile& file,const strata::qwen35::Spec& s,bool ab_nvfp4=false){
    auto require=[](bool ok,const char* text){if(!ok)throw std::invalid_argument(text);};
    using Tensor=const strata::TensorInfo*;
    auto matrix=[&](Tensor t,int type,int columns,int rows){
        require(t&&t==file.find(t->name)&&t->type==type&&
            t->shape==std::vector<uint64_t>({uint64_t(columns),uint64_t(rows)}),
            "Q36 nextN exact owned matrix geometry/type");
    };
    auto fp4=[&](Tensor t){require(t&&t==file.find(t->name)&&t->type==40&&t->shape.size()==2&&
        t->shape[0]%256==0,"Q36 nextN requires original400 main NVFP4 matrices");};
    auto norm=[&](Tensor t,int columns){require(t&&t==file.find(t->name)&&t->type==0&&
        t->shape==std::vector<uint64_t>({uint64_t(columns)}),"Q36 nextN original F32 norm identity");};
    require(s.n_layers==64&&s.layers.size()==64&&s.n_mtp_layers==1&&s.mtp.size()==1&&
        s.n_embd==5120&&s.n_vocab==248320&&s.n_ff==17408&&s.n_head==24&&s.n_head_kv==4&&s.head_dim==256&&
        s.ssm_state==128&&s.ssm_group==16&&s.ssm_rank==48&&s.ssm_inner==6144&&s.ssm_conv==4&&
        s.rope_dim==64&&s.context_length==262144,"Q36 nextN original Dense geometry");
    matrix(s.embedding,12,5120,248320);matrix(s.output,14,5120,248320);norm(s.output_norm,5120);
    require(s.embedding==file.find("token_embd.weight")&&s.output==file.find("output.weight"),
            "Q36 nextN original embedding12/rawhead14 names");
    int main=0,recurrent=0;
    for(size_t layer=0;layer<s.layers.size();++layer){
        const auto& l=s.layers[layer];
        for(Tensor t:{l.ffn_gate,l.ffn_up,l.ffn_down}){fp4(t);++main;}
        if(l.recurrent){++recurrent;
            for(Tensor t:{l.qkv,l.gate,l.ssm_out}){fp4(t);++main;}
            matrix(l.alpha,ab_nvfp4?40:8,5120,48);matrix(l.beta,ab_nvfp4?40:8,5120,48);
            if(ab_nvfp4){
                const auto prefix="blk."+std::to_string(layer);
                require(l.alpha->name==prefix+".ssm_alpha.weight"&&l.beta->name==prefix+".ssm_beta.weight"&&
                    l.alpha!=l.beta,"Q36 AB exact independent alpha/beta role names");
                for(Tensor t:{l.alpha,l.beta}){
                    const auto scale=t->name.substr(0,t->name.size()-7)+".scale";
                    norm(file.find(scale),1);
                }
            }
        }else for(Tensor t:{l.q,l.k,l.v,l.attn_out}){fp4(t);++main;}
    }
    require(main==400&&recurrent==48,"Q36 nextN requires complete400 main and48 alpha/beta pairs");
    const auto& m=s.mtp[0];const auto& d=m.decoder;
    require(m.embedding==s.embedding&&m.output==s.output,"Q36 nextN uses original shared embedding/head");
    const std::array<Tensor,8> roles={d.q,d.k,d.v,d.attn_out,d.ffn_gate,d.ffn_up,d.ffn_down,m.eh_proj};
    const std::array<const char*,8> names={"blk.64.attn_q.weight","blk.64.attn_k.weight","blk.64.attn_v.weight",
        "blk.64.attn_output.weight","blk.64.ffn_gate.weight","blk.64.ffn_up.weight","blk.64.ffn_down.weight",
        "blk.64.nextn.eh_proj.weight"};
    const std::array<int,8> columns={5120,5120,5120,6144,5120,5120,17408,10240};
    const std::array<int,8> rows={12288,1024,1024,5120,17408,17408,5120,5120};
    for(size_t i=0;i<roles.size();++i){matrix(roles[i],40,columns[i],rows[i]);
        require(roles[i]->name==names[i],"Q36 nextN role name identity");
        for(size_t j=0;j<i;++j)require(roles[i]!=roles[j],"Q36 nextN roles must be distinct");
        const auto* scale=file.find(std::string(names[i]).substr(0,std::string(names[i]).size()-7)+".scale");
        norm(scale,1);
    }
    for(Tensor t:{m.enorm,m.hnorm,m.output_norm,d.attn_norm,d.post_attn_norm})norm(t,5120);
    norm(d.q_norm,256);norm(d.k_norm,256);
    std::array<size_t,5> counts{};
    for(const auto& t:file.tensors()){
        if(t.type==0)++counts[0];else if(t.type==8)++counts[1];else if(t.type==12)++counts[2];
        else if(t.type==14)++counts[3];else if(t.type==40)++counts[4];
        else require(false,"Q36 nextN unexpected GGUF quantization type");
    }
    const std::array<size_t,5> expected=ab_nvfp4?std::array<size_t,5>{864,0,1,1,504}:
        std::array<size_t,5>{768,96,1,1,408};
    require(file.tensors().size()==size_t(ab_nvfp4?1370:1274)&&counts==expected,
            "Q36 nextN exact inventory must match explicit AB format and original12and14");
}
}
