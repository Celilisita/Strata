#include "strata/qwen35/completed_v100/native/weights.hpp"
#include "small_prefill_q4_tile_codes.hpp"
#include "strata/qwen35/completed_v100/native/original_upload.hpp"
#include <algorithm>
#include <cstdlib>

namespace strata::completed_v100::qwen35::native {
DenseWeights::DenseWeights(const strata::GgufFile &file,const strata::completed_v100::qwen35::Spec &s,cudaStream_t stream,bool tiled_ffn):tiled_ffn_(tiled_ffn){
    auto matrix=[&](const strata::TensorInfo *t,bool tile=false){
        if(t&&t->type==40){if(tile)throw std::invalid_argument("NVFP4 has its own GPU provider");return upload_original_nvfp4(file,*t,storage_,stream);}
        if(!t || t->shape.size()!=2 || (t->type!=8 && t->type!=12 && t->type!=14))
            throw std::invalid_argument("unsupported own Dense weight format");
        Matrix m{storage_.allocate(strata::completed_v100::qwen35::tensor_bytes(file,*t)),Quant(t->type),int(t->shape[1]),int(t->shape[0])};
        const size_t n=original_matrix_bytes(m);
        if(m.quant==Quant::Q4K){
            void *raw=nullptr;check_cuda(cudaMalloc(&raw,n),"own temporary lossless weight staging");
            try{
                check_cuda(cudaMemcpyAsync(raw,file.tensor_data(*t),n,cudaMemcpyHostToDevice,stream),"own setup Q4 upload");
                Matrix original=m;original.data=raw;m=prepare_q4_pairs(original,const_cast<void *>(m.data),stream);
                if(tile){
                    prepare_q4_tile_codes(m,raw,stream);
                    check_cuda(cudaMemcpyAsync(const_cast<void *>(m.data),raw,n,cudaMemcpyDeviceToDevice,stream),"cold tiled FFN inventory replacement");
                    m.layout=BlockLayout::Q4Tiles;
                }
                check_cuda(cudaStreamSynchronize(stream),"own setup prepack lifetime");
            }catch(...){cudaStreamSynchronize(stream);cudaFree(raw);throw;}
            cudaFree(raw);
        }else{
            if(tile)throw std::invalid_argument("tiled FFN inventory requires Q4 Gate/Up");
            check_cuda(cudaMemcpyAsync(const_cast<void *>(m.data),file.tensor_data(*t),n,cudaMemcpyHostToDevice,stream),"own setup compressed matrix upload");
        }
        return m;
    };
    auto scalar=[&](const strata::TensorInfo *t){
        if(!t || t->type!=0)throw std::invalid_argument("own Dense parameter requires F32");
        const size_t n=strata::completed_v100::qwen35::tensor_bytes(file,*t);auto *p=static_cast<float *>(storage_.allocate(n));
        check_cuda(cudaMemcpyAsync(p,file.tensor_data(*t),n,cudaMemcpyHostToDevice,stream),"own setup scalar upload");return p;
    };
    embedding=matrix(s.embedding);head=matrix(s.output);output_norm=scalar(s.output_norm);
    recurrent.resize(s.layers.size());attention.resize(s.layers.size());
    for(size_t i=0;i<s.layers.size();++i){
        const auto &l=s.layers[i];
        const Matrix gate=matrix(l.ffn_gate,tiled_ffn),up=matrix(l.ffn_up,tiled_ffn),down=matrix(l.ffn_down);
        const float *norm=scalar(l.attn_norm),*post=scalar(l.post_attn_norm);
        if(l.recurrent)recurrent[i]={matrix(l.qkv),matrix(l.gate),matrix(l.alpha),matrix(l.beta),matrix(l.ssm_out),
            gate,up,down,norm,post,scalar(l.conv),scalar(l.dt),scalar(l.a),scalar(l.ssm_norm)};
        else attention[i]={matrix(l.q),matrix(l.k),matrix(l.v),matrix(l.attn_out),gate,up,down,
            norm,scalar(l.q_norm),scalar(l.k_norm),post};
    }
    // No next-N/MTP tensor is loaded. Mmap sources remain live through this join.
    check_cuda(cudaStreamSynchronize(stream),"own Dense inventory setup completion");
}
}
