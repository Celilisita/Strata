#include "strata/gemma4/native/assistant.hpp"
#include "strata/gemma4/native/assistant_head_direct.hpp"
#include <limits>
#include <cmath>
#include <cstdint>
#include <stdexcept>
namespace strata::gemma4::native::assistant {
namespace {
void blas(cublasStatus_t s){if(s!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("Assistant cold cuBLAS setup");}
bool overlap(const void* a,size_t na,const void* b,size_t nb){
    const auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);return x<=y?y-x<na:x-y<nb;
}
}
const void* Draft::payload(const TensorInfo* t)const{
    for(const auto& item:inventory_)if(item.name==t->name)return item.device;
    throw std::runtime_error("Assistant uploaded tensor missing");
}
Q40Matrix Draft::matrix(const TensorInfo* t)const{
    if(!t||t->type!=2||t->shape.size()!=2)throw std::runtime_error("Assistant raw Q4_0 matrix");
    return {payload(t),int(t->shape[1]),int(t->shape[0])};
}
const float* Draft::scalar(const TensorInfo* t)const{
    if(!t||t->type!=0)throw std::runtime_error("Assistant F32 parameter");return static_cast<const float*>(payload(t));
}
Draft::Draft(const GgufFile& file,cudaStream_t stream,void* borrowed,size_t decoded_bytes):
    spec_(bind_spec(file)),stream_(stream){
    constexpr size_t decode_bytes=128u*1024*1024;
    if(!stream||(borrowed&&(decoded_bytes<decode_bytes||reinterpret_cast<uintptr_t>(borrowed)%256))||(!borrowed&&decoded_bytes))
        throw std::invalid_argument("Assistant cold decoded arena lease");
    inventory_.reserve(file.tensors().size());
    for(const auto& t:file.tensors()){
        const size_t bytes=tensor_payload_bytes(file,t);
        // Parameter validation is file integrity checking, not CPU inference.
        if(t.type==0){const auto* f=reinterpret_cast<const float*>(file.tensor_data(t));
            for(size_t i=0;i<bytes/4;++i)if(t.name=="rope_freqs.weight"?!(f[i]>0):!std::isfinite(f[i]))
                throw std::runtime_error("Assistant invalid norm/rotary parameter");}
        void* data=weights_.allocate(bytes);
        primitive::check_cuda(cudaMemcpyAsync(data,file.tensor_data(t),bytes,cudaMemcpyHostToDevice,stream_),"Assistant raw payload upload");
        inventory_.push_back({t.name,data,bytes});
    }
    pre_projection_=matrix(spec_.pre_projection);post_projection_=matrix(spec_.post_projection);head_=matrix(spec_.head);
    output_norm_=scalar(spec_.output_norm);frequencies_=scalar(spec_.frequencies);
    for(size_t i=0;i<layers_.size();++i){const auto& s=spec_.layers[i];auto& w=layers_[i];
        w.q=matrix(s.q);w.out=matrix(s.out);w.gate=matrix(s.gate);w.up=matrix(s.up);w.down=matrix(s.down);
        w.pre=scalar(s.pre);w.post=scalar(s.post);w.q_norm=scalar(s.q_norm);w.ffn_pre=scalar(s.ffn_pre);
        w.ffn_post=scalar(s.ffn_post);w.scale=scalar(s.scale);
    }
    auto floats=[&](size_t n){return static_cast<float*>(memory_.allocate(n*4));};
    borrowed_decode_=borrowed!=nullptr;decoded_=borrowed?static_cast<float*>(borrowed):floats(decode_bytes/4);decoded_values_=decode_bytes/4;
    workspace_=memory_.allocate(32u*1024*1024);condition_=floats(10752);hidden_[0]=floats(1024);hidden_[1]=floats(1024);
    normalized_=floats(1024);query_=floats(32*512);attended_=floats(32*512);
    projected_=floats(1024);post_=floats(1024);after_=floats(1024);gate_=floats(8192);up_=floats(8192);activation_=floats(8192);
    down_=floats(1024);features_=floats(1024);post_state_=floats(5376);logits_=floats(262144);
    proposal_=static_cast<int*>(memory_.allocate(4));invalid_=static_cast<unsigned*>(memory_.allocate(4));
    if(borrowed)for(const auto& t:inventory_)if(overlap(decoded_,decode_bytes,t.device,t.bytes))
        throw std::invalid_argument("Assistant borrowed decode overlaps uploaded weights");
    primitive::check_cuda(cudaStreamSynchronize(stream_),"Assistant cold immutable upload completion");
    try{
        blas(cublasCreate(&handle_));blas(cublasSetStream(handle_,stream_));
        blas(cublasSetWorkspace(handle_,workspace_,32u*1024*1024));blas(cublasSetPointerMode(handle_,CUBLAS_POINTER_MODE_HOST));
        blas(cublasSetMathMode(handle_,CUBLAS_DEFAULT_MATH));
    }catch(...){if(handle_)cublasDestroy(handle_);handle_=nullptr;throw;}
}
Draft::~Draft(){if(stream_)cudaStreamSynchronize(stream_);if(handle_)cublasDestroy(handle_);}
void Draft::select_head_backend(HeadBackend mode){
    if((mode!=HeadBackend::DecodeSgemm&&mode!=HeadBackend::Direct399)||
       !spec_.head||spec_.head->name!="token_embd.weight"||spec_.head->type!=2||
       head_.data!=payload(spec_.head)||head_.rows!=head399::OutputValues||head_.columns!=head399::InputValues||
       head_epoch_==(std::numeric_limits<std::uint64_t>::max)())
        throw std::invalid_argument("401 exact validated Assistant Q4_0 head/cold policy");
    primitive::check_cuda(cudaStreamSynchronize(stream_),"401 head policy cold lifetime join");
    head_backend_=mode;++head_epoch_;head_parent_enqueues_=0;head_direct_enqueues_=0;
}
void Draft::project(Q40Matrix m,const float* input,float* output){
    const bool head=m.data==head_.data;
    if(head&&head_backend_==HeadBackend::Direct399){
        // The constructor bound this pointer to the actual type2 Assistant head.
        // Epoch0 is valid for immutable instance storage; policy switches advance it.
        const head399::Binding binding{this,head_epoch_,stream_};
        head399::project(input,output,{m,head399::WeightBytes,binding},binding);
        ++head_direct_enqueues_;return;
    }
    q40_project(m,input,output,decoded_,decoded_values_,handle_,workspace_,32u*1024*1024,stream_);
    if(head)++head_parent_enqueues_;
}
void Draft::validate_lease(const float* embedding,const float* seed,CacheView local,CacheView global,const Bounds* bounds)const{
    if(!embedding||!seed||!bounds||!local.key||!local.value||!global.key||!global.value||
       local.capacity<1||global.capacity<1||local.heads!=16||local.dim!=256||local.window!=1024||
       global.heads!=4||global.dim!=512||global.window!=0)
        throw std::invalid_argument("Assistant borrowed main58/59 shape");
    const void* read[]{embedding,seed,local.key,local.value,global.key,global.value,bounds};
    const size_t lkv=size_t(local.capacity)*16*8*34,gkv=size_t(global.capacity)*4*16*34;
    const size_t rb[]{5376*4,5376*4,lkv,lkv,gkv,gkv,sizeof(Bounds)};
    const void* write[]{decoded_,workspace_,condition_,hidden_[0],hidden_[1],normalized_,query_,attended_,projected_,post_,
        after_,gate_,up_,activation_,down_,features_,post_state_,logits_,proposal_,invalid_};
    const size_t wb[]{decoded_values_*4,32u*1024*1024,10752*4,1024*4,1024*4,1024*4,32*512*4,32*512*4,1024*4,1024*4,
        1024*4,8192*4,8192*4,8192*4,1024*4,1024*4,5376*4,262144*4,4,4};
    for(int i=0;i<7;++i){
        if(reinterpret_cast<uintptr_t>(read[i])%(i>=2&&i<=5?2:4))throw std::invalid_argument("Assistant input alignment");
        for(int j=0;j<20;++j){
            if(i==1&&j==16&&seed==post_state_)continue; // Prior poststate is copied before overwriting it.
            if(overlap(read[i],rb[i],write[j],wb[j]))throw std::invalid_argument("Assistant read/write lease overlap");
        }
    }
    for(int i=2;i<7;++i)for(int j=i+1;j<7;++j)if(overlap(read[i],rb[i],read[j],rb[j]))
        throw std::invalid_argument("Assistant distinct KV/bounds leases");
}
}
