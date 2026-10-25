#pragma once
#include "ggml.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::glm5_next {
// Diagnostic only: first warm summary and first draft / 256 activations.
// No weights, full KV tensors, values or generation policy are changed.
class MtpFiniteTrace {
    bool enabled_=false,active_=false,failed_=false,warm_seen_=false,draft_seen_=false,report_=false;
    unsigned computes_=0,observed_=0;
    int count_=0;int64_t first_=0;
    std::string failure_;
    static bool selected(const ggml_tensor* tensor) {
        if(tensor->op==GGML_OP_FLASH_ATTN_EXT)return true;
        const char* name=ggml_get_name(tensor);
        const char* stages[]{"mtp_","q_resid-45","kv_cmpr-45","q_absorbed-45",
            "indexer_","kq_mask_dsa-45","kq-45","kq_soft_max-45","kqv",
            "attn_out-45","h_nextn","result_output"};
        for(const char* stage:stages)if(!std::strncmp(name,stage,std::strlen(stage)))return true;
        return false;
    }
    static bool allow_negative_infinity(const char* name) {
        return !std::strncmp(name,"indexer_score",13)||!std::strncmp(name,"indexer_sel",11)||
            !std::strncmp(name,"kq_mask_dsa",11)||!std::strcmp(name,"result_output");
    }
    void observe(ggml_tensor* tensor,const char* role="activation",bool mask=false) {
        constexpr size_t max_bytes=32ull<<20;
        const char* name=ggml_get_name(tensor);const size_t bytes=ggml_nbytes(tensor);
        const int64_t elements=ggml_nelements(tensor);
        if(elements<=0||elements>(4ll<<20)||bytes>max_bytes||
           (tensor->type!=GGML_TYPE_F32&&tensor->type!=GGML_TYPE_F16)) {
            std::fprintf(stderr,"strata_nextn_finite skip=bounded_or_type name=%s type=%s elements=%lld bytes=%llu\n",
                name,ggml_type_name(tensor->type),(long long)elements,(unsigned long long)bytes);
            std::fflush(stderr);return;
        }
        std::vector<unsigned char> raw(bytes);ggml_backend_tensor_get(tensor,raw.data(),0,bytes);
        const size_t width=tensor->type==GGML_TYPE_F32?sizeof(float):sizeof(ggml_fp16_t);
        uint64_t finite=0,nan=0,posinf=0,neginf=0;int64_t first_bad=-1;
        float minimum=std::numeric_limits<float>::infinity(),maximum=-minimum;
        const bool allow=mask||allow_negative_infinity(name);
        for(int64_t linear=0;linear<elements;++linear) {
            int64_t remaining=linear;size_t offset=0;
            for(int axis=0;axis<4;++axis) {offset+=size_t(remaining%tensor->ne[axis])*tensor->nb[axis];remaining/=tensor->ne[axis];}
            if(offset>bytes||width>bytes-offset)throw std::runtime_error("NextN finite trace tensor span invalid");
            float value=0;
            if(tensor->type==GGML_TYPE_F32)std::memcpy(&value,raw.data()+offset,sizeof(value));
            else {ggml_fp16_t half;std::memcpy(&half,raw.data()+offset,sizeof(half));value=ggml_fp16_to_fp32(half);}
            if(std::isfinite(value)){++finite;minimum=std::min(minimum,value);maximum=std::max(maximum,value);}
            else {if(std::isnan(value))++nan;else if(value>0)++posinf;else ++neginf;
                if(first_bad<0&&(!allow||std::isnan(value)||value>0))first_bad=linear;}
        }
        ++observed_;
        std::fprintf(stderr,"strata_nextn_finite compute=%u first=%lld rows=%d stage=%s role=%s op=%s type=%s shape=%lld,%lld,%lld,%lld finite=%llu nan=%llu posinf=%llu neginf=%llu allmasked=%d min=%g max=%g firstbad=%lld\n",
            computes_,(long long)first_,count_,name,role,ggml_op_name(tensor->op),ggml_type_name(tensor->type),
            (long long)tensor->ne[0],(long long)tensor->ne[1],(long long)tensor->ne[2],(long long)tensor->ne[3],
            (unsigned long long)finite,(unsigned long long)nan,(unsigned long long)posinf,(unsigned long long)neginf,
            int(allow&&neginf==uint64_t(elements)),finite?minimum:0.f,finite?maximum:0.f,(long long)first_bad);
        if(first_bad>=0) {
            failed_=true;failure_="NextN first nonfinite activation: "+std::string(name)+" role="+role+" op="+ggml_op_name(tensor->op)+
                " first="+std::to_string(first_)+" rows="+std::to_string(count_)+" index="+std::to_string(first_bad);
            for(int i=0;i<4;++i)if(const auto* src=tensor->src[i])
                std::fprintf(stderr,"strata_nextn_finite upstream=%d name=%s op=%s type=%s shape=%lld,%lld,%lld,%lld\n",
                    i,ggml_get_name(src),ggml_op_name(src->op),ggml_type_name(src->type),
                    (long long)src->ne[0],(long long)src->ne[1],(long long)src->ne[2],(long long)src->ne[3]);
        }
        std::fflush(stderr);
    }
public:
    MtpFiniteTrace() {
        const char* flag=std::getenv("STRATA_GLM53_NEXTN_FINITE_PROBE");
        if(flag&&std::strcmp(flag,"0")&&std::strcmp(flag,"1"))throw std::runtime_error("NextN finite trace must be0 or1");
        enabled_=flag&&!std::strcmp(flag,"1");
    }
    bool enabled()const {return enabled_;}
    bool begin(int64_t first,int count,const float* hidden) {
        ++computes_;first_=first;count_=count;active_=false;report_=false;
        if(enabled_&&observed_<256) {
            if(!warm_seen_&&first==0){warm_seen_=true;report_=true;}
            else if(!draft_seen_&&first>0&&count<=2){draft_seen_=true;active_=report_=true;}
        }
        if(report_){std::fprintf(stderr,"strata_nextn_finite begin compute=%u first=%lld rows=%d phase=%s diagnostic_timing=excluded\n",computes_,(long long)first,count,active_?"first_draft":"warm_summary");std::fflush(stderr);}
        if(active_)for(size_t i=0;i<size_t(count)*4096;++i)if(!std::isfinite(hidden[i]))
            throw std::runtime_error("NextN first draft input hidden nonfinite index="+std::to_string(i));
        return active_;
    }
    void finish(const std::vector<float>& output)const {
        if(!report_)return;
        size_t finite=0;float low=INFINITY,high=-INFINITY;
        for(float value:output)if(std::isfinite(value)){++finite;low=std::min(low,value);high=std::max(high,value);}
        std::fprintf(stderr,"strata_nextn_finite end compute=%u first=%lld rows=%d final_finite=%llu final_total=%llu min=%g max=%g diagnostic_timing=excluded\n",
            computes_,(long long)first_,count_,(unsigned long long)finite,(unsigned long long)output.size(),finite?low:0.f,finite?high:0.f);
        std::fflush(stderr);
    }
    static bool callback(ggml_tensor* tensor,bool ask,void* opaque) {
        auto& self=*static_cast<MtpFiniteTrace*>(opaque);
        if(ask)return self.active_&&!self.failed_&&self.observed_<256&&selected(tensor);
        try {
            if(tensor->op==GGML_OP_FLASH_ATTN_EXT) {
                const char* roles[]{"flash_q","flash_k","flash_v","flash_mask"};
                for(int i=0;i<4&&!self.failed_;++i)if(tensor->src[i])self.observe(tensor->src[i],roles[i],i==3);
            }
            if(!self.failed_)self.observe(tensor);
        }catch(const std::exception& error){self.failed_=true;self.failure_=error.what();}
        return !self.failed_;
    }
    void check()const {if(failed_)throw std::runtime_error(failure_);}
};
}
