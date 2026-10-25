#include "strata/qwen35/completed_v100/native/ffn.hpp"
#include "strata/qwen35/completed_v100/native/dense_gemm.hpp"
#include "strata/qwen35/completed_v100/native/weight_overlap.hpp"

namespace strata::completed_v100::qwen35::native {
void Ffn::enable_pp_weight_overlap(cudaStream_t stream,bool enabled){
    if(pp_weight_overlap_||!pp_gemm_||provider_!=Provider::OwnedStaged||max_tokens_<128)
        throw std::invalid_argument("invalid PP weight overlap selection");
    pp_weight_overlap_=std::make_unique<PpWeightOverlap>(2*size_t(down_.rows)*down_.columns,stream);
    workspace_bytes_+=pp_weight_overlap_->bytes();pp_weight_overlap_enabled_=enabled;
}
size_t Ffn::pp_weight_overlap_bytes()const{return pp_weight_overlap_?pp_weight_overlap_->bytes():0;}
void Ffn::select_pp_weight_overlap(bool enabled){
    if(!pp_weight_overlap_)throw std::invalid_argument("weight overlap workspace not configured");
    pp_weight_overlap_enabled_=enabled;
}
void Ffn::use_pp_gemm(DenseGemm *service){
    if(service&&(provider_!=Provider::OwnedStaged||!buffers_.half_weights||!buffers_.half_input||
                 tiled_pair_||group_pair_||gate_.layout==BlockLayout::Q4Tiles||down_.layout==BlockLayout::Q6Codes))
        throw std::invalid_argument("Dense GEMM requires unleased staged operands");
    pp_gemm_=service;
}
void Ffn::execute_pp_gemm(const float *input,int tokens,cudaStream_t stream,bool diagnostic){
    const bool fused=pp_gemm_->can_fuse_swiglu(gate_.rows,gate_.columns,tokens);
    const bool direct=fused&&pp_gemm_->can_fuse_quant_swiglu(gate_,up_,tokens);
    if(fused&&!direct)decode_pair_interleaved(gate_,up_,buffers_.half_weights,stream);
    else if(!fused)decode_pair(gate_,up_,buffers_.half_weights,stream);
    to_half(input,buffers_.half_input,size_t(gate_.columns)*tokens,stream);
    const bool overlap=pp_weight_overlap_enabled_&&!diagnostic;
    if(overlap)pp_weight_overlap_->begin(down_,stream);
    const auto *second=static_cast<const unsigned char *>(buffers_.half_weights)+
                       2*size_t(gate_.rows)*gate_.columns;
    const size_t pair_bytes=8*size_t(gate_.rows)*tokens;
    // Concatenated output borrows the contiguous Gate/Up pool. Debug unpacking
    // uses dead decoded weights and is excluded from all normal cost cases.
    if(fused){
        // Output cannot alias the converted input while GEMM CTAs still read it.
        if(direct)pp_gemm_->project_quant_swiglu(gate_,up_,buffers_.half_input,buffers_.activation,
            buffers_.half_weights,buffers_.gate,diagnostic?buffers_.gate:nullptr,
            diagnostic?buffers_.up:nullptr,tokens,stream);
        else pp_gemm_->project_swiglu(buffers_.half_weights,buffers_.half_input,buffers_.activation,
            buffers_.gate,diagnostic?buffers_.gate:nullptr,diagnostic?buffers_.up:nullptr,
            gate_.rows,gate_.columns,tokens,stream);
        if(diagnostic){
            // Preserve the actual fused half payload before the cold FP32 view.
            check_cuda(cudaMemcpyAsync(buffers_.half_input,buffers_.activation,
                size_t(gate_.rows)*tokens*2,cudaMemcpyDeviceToDevice,stream),"fused activation diagnostic lease");
            swiglu(buffers_.gate,buffers_.up,buffers_.activation,size_t(gate_.rows)*tokens,stream);
        }
    }else{
    const bool pair=(!diagnostic||pair_bytes<=half_weight_capacity())&&
        pp_gemm_->project_pair(buffers_.half_weights,buffers_.half_input,buffers_.gate,
                               gate_.rows,gate_.columns,tokens,stream);
    if(pair){
        if(diagnostic){
            check_cuda(cudaMemcpyAsync(buffers_.half_weights,buffers_.gate,pair_bytes,cudaMemcpyDeviceToDevice,stream),"paired GEMM cold materialization");
            split_swiglu_pair(static_cast<float *>(buffers_.half_weights),buffers_.gate,buffers_.up,
                              buffers_.activation,buffers_.half_input,gate_.rows,tokens,stream);
        }else swiglu_pair_half(buffers_.gate,buffers_.half_input,gate_.rows,tokens,stream);
    }else{
        pp_gemm_->project(buffers_.half_weights,buffers_.half_input,buffers_.gate,
                          gate_.rows,gate_.columns,tokens,stream);
        pp_gemm_->project(second,buffers_.half_input,buffers_.up,
                          up_.rows,up_.columns,tokens,stream);
        if(diagnostic){
            swiglu(buffers_.gate,buffers_.up,buffers_.activation,size_t(gate_.rows)*tokens,stream);
            to_half(buffers_.activation,buffers_.half_input,size_t(down_.columns)*tokens,stream);
        }else swiglu_half(buffers_.gate,buffers_.up,buffers_.half_input,size_t(gate_.rows)*tokens,stream);
    }
    }
    intermediates_materialized_=diagnostic;
    const void *down_weights=buffers_.half_weights;
    if(overlap)down_weights=pp_weight_overlap_->ready(stream);
    else decode_linear(down_,buffers_.half_weights,stream);
    const void *down_input=fused&&!diagnostic?static_cast<void *>(buffers_.activation):buffers_.half_input;
    pp_gemm_->project(down_weights,down_input,buffers_.output,
                      down_.rows,down_.columns,tokens,stream);
}
}
