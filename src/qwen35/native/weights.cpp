#include "strata/qwen35/native/weights.hpp"
#include "small_prefill_q4_tile_codes.hpp"
#include "strata/qwen35/native/nvfp4_runtime.hpp"
#include <algorithm>
#include <cstdlib>

namespace strata::qwen35::native {
DeviceStorage::~DeviceStorage(){for(const auto &p:allocations_)cudaFree(p.first);}
size_t DeviceStorage::reserve_bytes(){
    static const size_t reserve=[] {
        const char* text=std::getenv("STRATA_DENSE_RESERVE_MIB");
        if(!text)return size_t(4096)*1024*1024;
        size_t used=0;auto mib=std::stoull(text,&used);
        if(text[used]||*text=='-'||mib>1048576)throw std::invalid_argument("Dense reserve MiB");
        return size_t(mib)*1024*1024;
    }();
    return reserve;
}
void *DeviceStorage::allocate(size_t bytes){
    if(!bytes)throw std::invalid_argument("empty own device allocation");
    size_t free=0,total=0;check_cuda(cudaMemGetInfo(&free,&total),"own setup reserve query");
    if(bytes>free || free-bytes<reserve_bytes())
        throw std::runtime_error("own setup reserve: request="+std::to_string(bytes)+" free="+std::to_string(free)+" owned="+std::to_string(bytes_));
    void *p=nullptr;check_cuda(cudaMalloc(&p,bytes),"own Dense setup allocation");
    try{allocations_.push_back({p,bytes});}catch(...){cudaFree(p);throw;}
    bytes_+=bytes;return p;
}
void DeviceStorage::release(void *p){
    auto found=std::find_if(allocations_.begin(),allocations_.end(),[&](const auto &a){return a.first==p;});
    if(found==allocations_.end())throw std::invalid_argument("foreign own allocation release");
    bytes_-=found->second;
    check_cuda(cudaFree(p),"release idle own allocation");allocations_.erase(found);
}
DenseWeights::DenseWeights(const strata::GgufFile &f,const strata::qwen35::Spec &s,cudaStream_t stream,bool tiled)
    :DenseWeights(f,s,stream,tiled,{}){}
DenseWeights::DenseWeights(const strata::GgufFile &file,const strata::qwen35::Spec &s,cudaStream_t stream,bool tiled_ffn,ColdHeadBinding binding):tiled_ffn_(tiled_ffn){
    if(binding.fixture)validate_cold_head_binding(file,s,binding);
    else if(s.output&&s.output->type==40)validate_shared_head_identity(file,s);
    auto matrix=[&](const strata::TensorInfo *t,bool tile=false,bool allow_q8=false){
        if(t && t->type==40){
            if(tile)throw std::invalid_argument("NVFP4 inventory cannot use a Q4 tiled policy");
            Nvfp4View view;auto m=load_nvfp4_matrix(file,*t,storage_,view,stream);
            fp4_weights.push_back(view);return m;
        }
        if(!t || t->shape.size()!=2 || (t->type!=12 && t->type!=14 && !(allow_q8 && t->type==8)))
            throw std::invalid_argument("unsupported own Dense weight format");
        Matrix m{storage_.allocate(strata::qwen35::tensor_bytes(file,*t)),Quant(t->type),int(t->shape[1]),int(t->shape[0])};
        const size_t n=matrix_bytes(m);
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
        const size_t n=strata::qwen35::tensor_bytes(file,*t);auto *p=static_cast<float *>(storage_.allocate(n));
        check_cuda(cudaMemcpyAsync(p,file.tensor_data(*t),n,cudaMemcpyHostToDevice,stream),"own setup scalar upload");return p;
    };
    embedding=matrix(s.embedding,false,true);
    if(binding.fixture){Nvfp4View view;head=load_cold_head_binding(file,s,binding,storage_,view,stream);fp4_weights.push_back(view);}
    else if(s.output&&s.output->type==40){Nvfp4View view;head=load_nvfp4_shared_head(file,s,storage_,view,stream);fp4_weights.push_back(view);}
    else head=matrix(s.output);
    output_norm=scalar(s.output_norm);
    recurrent.resize(s.layers.size());attention.resize(s.layers.size());
    for(size_t i=0;i<s.layers.size();++i){
        const auto &l=s.layers[i];
        const Matrix gate=matrix(l.ffn_gate,tiled_ffn),up=matrix(l.ffn_up,tiled_ffn),down=matrix(l.ffn_down);
        const float *norm=scalar(l.attn_norm),*post=scalar(l.post_attn_norm);
        if(l.recurrent)recurrent[i]={matrix(l.qkv),matrix(l.gate),matrix(l.alpha,false,true),matrix(l.beta,false,true),matrix(l.ssm_out),
            gate,up,down,norm,post,scalar(l.conv),scalar(l.dt),scalar(l.a),scalar(l.ssm_norm)};
        else attention[i]={matrix(l.q),matrix(l.k),matrix(l.v),matrix(l.attn_out),gate,up,down,
            norm,scalar(l.q_norm),scalar(l.k_norm),post};
    }
    // No next-N/MTP tensor is loaded. Mmap sources remain live through this join.
    check_cuda(cudaStreamSynchronize(stream),"own Dense inventory setup completion");
}
}
