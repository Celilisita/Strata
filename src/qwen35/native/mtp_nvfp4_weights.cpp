#include "strata/qwen35/native/weights.hpp"
#include "strata/qwen35/native/nvfp4_runtime.hpp"
#include <array>
#include <vector>
#include <iostream>
#include "q36_nextn_nvfp4_policy.hpp"
#include "q36_ab_nvfp4_policy.hpp"

namespace strata::qwen35::native {
namespace {
using Tensor = const strata::TensorInfo *;
std::array<Tensor,8> draft_roles(const strata::qwen35::Spec &s) {
    if(s.mtp.size()!=1)throw std::invalid_argument("NVFP4 draft requires one nextN block");
    const auto &m=s.mtp[0];const auto &d=m.decoder;
    if(m.embedding!=s.embedding||m.output!=s.output)
        throw std::invalid_argument("NVFP4 draft requires shared embedding and output identities");
    return {d.q,d.k,d.v,d.attn_out,d.ffn_gate,d.ffn_up,d.ffn_down,m.eh_proj};
}
void require_fp4(Tensor tensor) {
    if(!tensor||tensor->type!=40||tensor->shape.size()!=2||
       !tensor->shape[0]||!tensor->shape[1]||tensor->shape[0]%256)
        throw std::invalid_argument("Qwen3.5 derived matrix must be canonical NVFP4");
}
void require_norm(Tensor tensor) {
    if(!tensor||tensor->type!=0)
        throw std::invalid_argument("Qwen3.5 derived norm must retain its original F32 payload");
}
}
void validate_qwen35_nvfp4_asset(const strata::qwen35::Spec &s) {
    if(s.n_layers!=64||s.layers.size()!=64||s.n_mtp_layers!=1||s.n_embd!=5120||
       s.n_vocab!=248320||s.n_ff!=17408||s.n_head!=24||s.n_head_kv!=4||s.head_dim!=256||
       s.ssm_state!=128||s.ssm_group!=16||s.ssm_rank!=48||s.ssm_inner!=6144||s.ssm_conv!=4||
       s.rope_dim!=64||!s.embedding||s.embedding->type!=8)
        throw std::invalid_argument("Qwen3.5 derived profile requires its original Q8 embedding and Dense geometry");
    require_fp4(s.output);require_norm(s.output_norm);
    for(const auto &l:s.layers) {
        for(Tensor t:{l.ffn_gate,l.ffn_up,l.ffn_down})require_fp4(t);
        if(l.recurrent) {
            for(Tensor t:{l.qkv,l.gate,l.alpha,l.beta,l.ssm_out})require_fp4(t);
        } else {
            for(Tensor t:{l.q,l.k,l.v,l.attn_out})require_fp4(t);
        }
    }
    const auto roles=draft_roles(s);
    for(size_t i=0;i<roles.size();++i) {
        require_fp4(roles[i]);
        for(size_t j=0;j<i;++j)if(roles[i]==roles[j])
            throw std::invalid_argument("Qwen3.5 nextN roles must have distinct tensor identities");
    }
    const auto &m=s.mtp[0];const auto &d=m.decoder;
    for(Tensor t:{m.enorm,m.hnorm,m.output_norm,d.attn_norm,d.q_norm,d.k_norm,d.post_attn_norm})
        require_norm(t);
}
void DenseWeights::prepare_mtp_nvfp4(const strata::GgufFile &file,
        const strata::qwen35::Spec &s,cudaStream_t stream) {
    if(mtp_fp4_ready_||mtp_fp4_failed_)
        throw std::logic_error("NVFP4 nextN cold inventory may be prepared only once");
    const bool q36_nextn=q36_nextn_nvfp4_requested();
    if(q36_nextn)validate_q36_nextn_nvfp4_asset(file,s,q36_ab_nvfp4_requested());
    else validate_qwen35_nvfp4_asset(s);
    cudaStreamCaptureStatus capture{};
    check_cuda(cudaStreamIsCapturing(stream,&capture),"NVFP4 nextN cold inventory capture check");
    if(capture!=cudaStreamCaptureStatusNone)
        throw std::logic_error("NVFP4 nextN inventory must precede Runtime and Graph construction");
    const auto roles=draft_roles(s);
    for(Tensor t:roles)if(file.find(t->name)!=t)
        throw std::invalid_argument("NVFP4 nextN tensor belongs to another source inventory");
    std::array<Matrix,8> matrices{};
    std::array<Nvfp4View,8> views{};
    const size_t previous=fp4_weights.size();
    const size_t q36_expected=q36_ab_nvfp4_requested()?497:401;
    if(q36_nextn&&(previous!=q36_expected||head.quant!=Quant::NVFP4))
        throw std::invalid_argument("Q36 nextN preloads after explicit main/AB inventory plus bound external head");
    fp4_weights.reserve(previous+roles.size());
    try {
        for(size_t i=0;i<roles.size();++i)
            matrices[i]=load_nvfp4_matrix(file,*roles[i],storage_,views[i],stream);
        // Publish only a complete eight-role inventory. Runtime copies these views
        // after this method returns; NativeMtp only borrows the resulting handles.
        for(const auto &view:views)fp4_weights.push_back(view);
        mtp_fp4_tensors_=roles;mtp_fp4_matrices_=matrices;
        mtp_fp4_stream_=stream;mtp_fp4_ready_=true;
        if(q36_nextn)std::cerr<<"DENSE_Q36_NEXTN_NVFP4 enabled=1 matrices=8 fp4_inventory_before="<<previous
            <<" fp4_inventory_after="<<fp4_weights.size()<<" canonical_payload_bytes=238878720"
            <<" owner=DenseWeights borrowed_by=NativeMtp startup_body_plan=0 ab_nvfp4="<<q36_ab_nvfp4_requested()<<'\n';
    } catch(...) {
        mtp_fp4_failed_=true;cudaStreamSynchronize(stream);fp4_weights.resize(previous);
        for(const auto &m:matrices)if(m.data) {
            try {storage_.release(const_cast<void *>(m.data));}catch(...){}
        }
        throw;
    }
}
Matrix DenseWeights::borrowed_mtp_nvfp4(const strata::TensorInfo &tensor,cudaStream_t stream) const {
    if(!mtp_fp4_ready_||mtp_fp4_failed_||stream!=mtp_fp4_stream_)
        throw std::logic_error("NVFP4 nextN requires its preloaded ordered-stream inventory");
    for(size_t i=0;i<mtp_fp4_tensors_.size();++i)if(mtp_fp4_tensors_[i]==&tensor) {
        const auto m=mtp_fp4_matrices_[i];matrix_bytes(m);
        if(tensor.type!=40||tensor.shape.size()!=2||m.quant!=Quant::NVFP4||
           m.columns!=int(tensor.shape[0])||m.rows!=int(tensor.shape[1]))
            throw std::logic_error("NVFP4 nextN borrowed identity or geometry changed");
        return m;
    }
    throw std::invalid_argument("unregistered NVFP4 nextN tensor role");
}
}
