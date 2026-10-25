#pragma once
#include "gdn_output_elision.hpp"
#include "strata/qwen35/native/ffn.hpp"
#include <cstdint>
#include <stdexcept>
#include <utility>
namespace strata::qwen35::native {
bool gdn_output_elision_capability(DenseGemm&,Matrix,Nvfp4View&,Nvfp4Input* =nullptr);
class GdnOutputElisionScope;
inline GdnOutputElisionScope *&gdn_output_elision_pending(){static thread_local GdnOutputElisionScope *value=nullptr;return value;}
inline bool gdn_output_elision_overlap(const void *a,size_t n,const void *b,size_t m){
    const auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);return x<=y?y-x<n:x-y<m;
}
class GdnOutputElisionScope {
    GdnOutputElisionScope *prior_;Matrix weight_;Nvfp4View expected_;Nvfp4Input expected_encoded_{};const float *attention_,*gamma_,*gate_;float *gated_;
    int tokens_;float epsilon_;cudaStream_t stream_;bool active_,consumed_=false;Nvfp4Input encoded_{};
public:
    GdnOutputElisionScope(bool enabled,DenseGemm&service,Matrix weight,const float *attention,const float *gamma,const float *gate,
              float *gated,int tokens,float epsilon,cudaStream_t stream,bool observer=false)
        :prior_(gdn_output_elision_pending()),weight_(weight),expected_{} ,attention_(attention),gamma_(gamma),gate_(gate),gated_(gated),
         tokens_(tokens),epsilon_(epsilon),stream_(stream),active_(enabled&&!observer&&(tokens==8192||tokens==5424)&&
             weight.quant==Quant::NVFP4&&weight.layout==BlockLayout::Nvfp4WarpFragments&&weight.rows==5120&&weight.columns==6144){
        if(active_&&!gdn_output_elision_capability(service,weight,expected_,&expected_encoded_))active_=false;
        if(active_&&(!weight.data||!attention||!gamma||!gate||!gated))throw std::invalid_argument("GDN output elision missing source lease");
        gdn_output_elision_pending()=this;
    }
    ~GdnOutputElisionScope(){gdn_output_elision_pending()=prior_;}
    GdnOutputElisionScope(const GdnOutputElisionScope&)=delete;GdnOutputElisionScope& operator=(const GdnOutputElisionScope&)=delete;
    bool deferred()const{return active_;}
    bool consumed()const{return consumed_;}
    Nvfp4Input encoded()const{return encoded_;}
    void verify()const{if(active_&&!consumed_)throw std::logic_error("GDN output elision deferred norm was not consumed");}
    bool prepare(Matrix weight,Nvfp4View view,const float *input,float *output,Nvfp4Input encoded,int tokens,cudaStream_t stream){
        if(!active_)return false;
        if(consumed_||encoded.values!=expected_encoded_.values||encoded.scales!=expected_encoded_.scales||
           encoded.capacity_rows!=expected_encoded_.capacity_rows||encoded.capacity_columns!=expected_encoded_.capacity_columns||weight.data!=weight_.data||weight.quant!=weight_.quant||weight.layout!=weight_.layout||
           weight.rows!=weight_.rows||weight.columns!=weight_.columns||view.values!=weight.data||
           view.rows!=5120||view.columns!=6144||view.values!=expected_.values||view.scales!=expected_.scales||
           view.weight_scale!=expected_.weight_scale||view.input_scale!=expected_.input_scale||tokens!=tokens_||stream!=stream_||input!=gated_||!output)
            throw std::logic_error("GDN output elision one-shot projection/stream lease mismatch");
        gdn_output_elision_validate(attention_,gamma_,gate_,gated_,encoded,tokens,epsilon_,view.input_scale);
        const size_t n=size_t(tokens)*6144*4,out=size_t(tokens)*5120*4;
        for(const auto &p:std::initializer_list<std::pair<const void*,size_t>>{{attention_,n},{gamma_,512},{gate_,n},
            {gated_,n},{encoded.values,nvfp4_value_bytes(tokens,6144)},{encoded.scales,nvfp4_scale_bytes(tokens,6144)},
            {view.values,nvfp4_value_bytes(5120,6144)},{view.scales,nvfp4_scale_bytes(5120,6144)}})
            if(gdn_output_elision_overlap(output,out,p.first,p.second))throw std::logic_error("GDN output elision projection output aliases live storage");
        for(const auto &p:std::initializer_list<std::pair<const void*,size_t>>{{gated_,n},
            {encoded.values,nvfp4_value_bytes(tokens,6144)},{encoded.scales,nvfp4_scale_bytes(tokens,6144)}})
            if(gdn_output_elision_overlap(p.first,p.second,view.values,nvfp4_value_bytes(5120,6144))||
               gdn_output_elision_overlap(p.first,p.second,view.scales,nvfp4_scale_bytes(5120,6144)))
                throw std::logic_error("GDN output elision codec output aliases immutable weights");
        gdn_output_elision_encode(attention_,gamma_,gate_,gated_,encoded,tokens,epsilon_,view.input_scale,stream);
        encoded_=encoded;consumed_=true;return true;
    }
};
inline bool gdn_output_elision_prepare(Matrix weight,Nvfp4View view,const float *input,float *output,Nvfp4Input encoded,int tokens,cudaStream_t stream){
    auto *scope=gdn_output_elision_pending();return scope&&scope->prepare(weight,view,input,output,encoded,tokens,stream);
}
}
