#pragma once
#include "fp4_compensated.hpp"
#include "strata/qwen35/native/nvfp4_runtime.hpp"
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
namespace strata::qwen35::native {
// Capability and ordered scratch lease only. Runtime owns the one registry.
class TinyHeadCapability {
    Matrix identity_{};bool admitted_=false,batched_=true;TinyHeadCounters counters_{};
    static bool overlap(const void *a,size_t n,const void *b,size_t m){
        const auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);
        if(n>std::numeric_limits<uintptr_t>::max()-x||m>std::numeric_limits<uintptr_t>::max()-y)
            throw std::invalid_argument("head operand extent overflow");
        return x<y+m&&y<x+n;
    }
    static void lease(Nvfp4View w,CompensatedFp4Input x,const float *in,float *out,int tokens){
        if(!in||!out||tokens<1||tokens>8192||reinterpret_cast<uintptr_t>(in)%4||reinterpret_cast<uintptr_t>(out)%4)
            throw std::invalid_argument("head input/output span");
        for(auto plane:{x.high,x.low})if(plane.capacity_rows<16||plane.capacity_columns<5120)
            throw std::invalid_argument("head encoded capacity");
        const void *p[]={w.values,w.scales,x.high.values,x.high.scales,x.low.values,x.low.scales};
        const size_t n[]={nvfp4_value_bytes(w.rows,w.columns),nvfp4_scale_bytes(w.rows,w.columns),
            nvfp4_value_bytes(x.high.capacity_rows,x.high.capacity_columns),nvfp4_scale_bytes(x.high.capacity_rows,x.high.capacity_columns),
            nvfp4_value_bytes(x.low.capacity_rows,x.low.capacity_columns),nvfp4_scale_bytes(x.low.capacity_rows,x.low.capacity_columns)};
        const size_t inputs=size_t(tokens)*w.columns*4,outputs=size_t(tokens)*w.rows*4;
        if(overlap(in,inputs,out,outputs))throw std::invalid_argument("head normalized/logits alias");
        for(int i=0;i<6;++i){
            if(!p[i]||reinterpret_cast<uintptr_t>(p[i])%4||overlap(in,inputs,p[i],n[i])||overlap(out,outputs,p[i],n[i]))
                throw std::invalid_argument("head weight/encoded/input/output lease");
            for(int j=i+1;j<6;++j)if(overlap(p[i],n[i],p[j],n[j]))throw std::invalid_argument("head encoded operands alias");
        }
    }
public:
    explicit TinyHeadCapability(Matrix identity={}):identity_(identity){
        if(identity.data&&(identity.quant!=Quant::NVFP4||identity.layout!=BlockLayout::Nvfp4WarpFragments||
           identity.rows!=248320||identity.columns!=5120||reinterpret_cast<uintptr_t>(identity.data)%16))
            throw std::invalid_argument("explicit tiny-only head identity");
    }
    bool present()const{return identity_.data!=nullptr;}
    bool contains(Nvfp4View w)const{return present()&&w.values==identity_.data;}
    bool contains(Matrix w)const{return present()&&w.data==identity_.data;}
    bool admit(Nvfp4View w){
        if(!contains(w))return false;
        if(w.rows!=identity_.rows||w.columns!=identity_.columns||!w.scales||reinterpret_cast<uintptr_t>(w.scales)%4||
           !(w.weight_scale>0)||!std::isfinite(w.weight_scale)||w.input_scale!=1||
           overlap(w.values,nvfp4_value_bytes(w.rows,w.columns),w.scales,nvfp4_scale_bytes(w.rows,w.columns)))
            throw std::invalid_argument("tiny-only head registered view");
        admitted_=true;return true;
    }
    void finish_admission()const{if(present()&&!admitted_)throw std::invalid_argument("tiny-only head absent from inventory");}
    void configure(bool batched){batched_=batched;counters_={};}
    TinyHeadCounters counters()const{auto out=counters_;if(present())out.weight_bytes=nvfp4_gguf_bytes(248320,5120);return out;}
    void validate(Nvfp4View w,CompensatedFp4Input x,const float *input,float *output,int tokens)const{
        if(!admitted_||!contains(w))throw std::invalid_argument("projection requires its tiny-only head entry");
        lease(w,x,input,output,tokens);
    }
    void execute(Nvfp4View w,CompensatedFp4Input x,const float *input,float *output,int tokens,cudaStream_t stream){
        validate(w,x,input,output,tokens);
        if(tokens==1||(tokens==3&&batched_)){
            fp4_compensated_quantize(input,x,tokens,w.columns,w.input_scale,stream);
            fp4_compensated_linear({w},x,output,tokens,stream);
            if(tokens==1)++counters_.scalar_submissions;else ++counters_.batch_submissions;
        }else{
            for(int row=0;row<tokens;++row){
                fp4_compensated_quantize(input+size_t(row)*w.columns,x,1,w.columns,w.input_scale,stream);
                fp4_compensated_linear({w},x,output+size_t(row)*w.rows,1,stream);
            }
            counters_.fallback_rows+=tokens;
        }
    }
};
}
