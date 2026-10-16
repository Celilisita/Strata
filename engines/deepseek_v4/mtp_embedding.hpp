#pragma once
#include "ggml-backend.h"
#include "ggml.h"
#include <stdexcept>
#include <vector>
#include <cstring>
namespace strata::deepseek_v4 {
inline ggml_tensor* mtp_embedding_input(ggml_context* c,ggml_tensor* source,int count) {
    if(!source||!source->data||!source->buffer||source->ne[0]!=4096)throw std::runtime_error("MTP shared embedding source invalid");
    if(!ggml_backend_buffer_is_host(source->buffer))return nullptr;
    if(source->type!=GGML_TYPE_BF16&&source->type!=GGML_TYPE_F16&&source->type!=GGML_TYPE_F32)throw std::runtime_error("MTP shared host embedding format unsupported");
    auto* input=ggml_new_tensor_2d(c,GGML_TYPE_F32,4096,count);ggml_set_input(input);return input;
}
inline void mtp_prepare_embedding(float* values,ggml_tensor* source,const int32_t* ids,int count) {
    for(int t=0;t<count;++t){if(ids[t]<0||ids[t]>=source->ne[1])throw std::runtime_error("MTP embedding token out of range");
        const auto* row=static_cast<const char*>(source->data)+ids[t]*source->nb[1];float* out=values+t*4096;
        if(source->type==GGML_TYPE_BF16)ggml_bf16_to_fp32_row(reinterpret_cast<const ggml_bf16_t*>(row),out,4096);
        else if(source->type==GGML_TYPE_F16)ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t*>(row),out,4096);
        else std::memcpy(out,row,4096*sizeof(float));}
}
inline void mtp_set_embedding_input(ggml_tensor* input,ggml_tensor* source,const int32_t* ids,int count) {
    if(!input)return;std::vector<float> values(size_t(count)*4096);
    mtp_prepare_embedding(values.data(),source,ids,count);
    ggml_backend_tensor_set(input,values.data(),0,values.size()*sizeof(float));
}
}
