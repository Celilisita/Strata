#include "strata/qwen35/native/model.hpp"
#include "strata/qwen35/native/ops.hpp"
#include "strata/qwen35/native/tiny_batch.hpp"
#include "strata/qwen35/native/output_head.hpp"
#include "q6_warp_policy.hpp"
#include "strata/qwen35/native/nvfp4_runtime.hpp"
namespace strata::qwen35::native {
void output_head_linear(Matrix weight,DenseGemm *service,const float *input,void *quantized,float *logits,int rows,cudaStream_t stream){
    matrix_bytes(weight);
    if(!input||!logits||rows<1||rows>8192)throw std::invalid_argument("invalid normalized head span");
    if(weight.quant==Quant::NVFP4){
        if(!service)throw std::invalid_argument("NVFP4 head requires its authoritative runtime");
        nvfp4_project_head(*service,weight,input,logits,rows,stream);return;
    }
    if((weight.quant!=Quant::Q4K&&weight.quant!=Quant::Q6K)||!quantized||rows>4)
        throw std::invalid_argument("legacy head requires Q4_K/Q6_K and its codec span");
    if(rows==1){vector_quantize(input,quantized,weight.columns,stream);vector_linear(weight,quantized,logits,stream);}
    else{tiny_batch_quantize(input,quantized,weight.columns,rows,stream);
        if(use_q6_warp_head(weight,rows))q6_warp_logical(weight,quantized,logits,rows,stream);
        else tiny_batch_linear(weight,quantized,logits,rows,stream);}
}
void output_head_batch(Matrix weight,const float *norm,const float *hidden,int rows,
                       float epsilon,OutputHeadBatch b,cudaStream_t stream,TokenMask mask){
    output_head_batch(weight,norm,hidden,rows,epsilon,b,stream,mask,nullptr);
}
void output_head_batch(Matrix weight,const float *norm,const float *hidden,int rows,
                       float epsilon,OutputHeadBatch b,cudaStream_t stream,TokenMask mask,DenseGemm *service){
    if(!norm||!hidden||!b.normalized||!b.logits||!b.quantized||!b.greedy||!b.ids||!b.flags||
       (rows<2||rows>4)||epsilon<=0||b.greedy_bytes<size_t(rows)*greedy_workspace(weight.rows))
        throw std::invalid_argument("invalid native head batch buffers");
    matrix_bytes(weight);
    rms_norm(hidden,norm,b.normalized,weight.columns,rows,epsilon,stream);
    output_head_linear(weight,service,b.normalized,b.quantized,b.logits,rows,stream);
    greedy_pick_batch(b.logits,weight.rows,rows,b.greedy,b.greedy_bytes,b.ids,b.flags,stream,mask);
}
void DenseModel::head_batch(const float *hidden,int tokens,float *normed,int *ids,unsigned *flags,cudaStream_t stream){
    if(!hidden||!normed||!ids||!flags||(tokens<2||tokens>4))throw std::invalid_argument("invalid native batched head");
    const int h=int(spec_.n_embd),v=int(spec_.n_vocab);
    // The wide shared activation arena is dead after the final model layer.
    // Its bytes now hold the same Q8 activation format used by the scalar head.
    void *prepared=shared_->shared_ffn(tokens).buffers().half_input;
    output_head_batch(weights_.head,weights_.output_norm,hidden,tokens,float(spec_.rms_epsilon),
                      {normed,batch_logits_,prepared,batch_greedy_,4*greedy_bytes_,ids,flags},stream,token_mask_,head_service_);
    // Keep the ordinary model's diagnostic/output state coherent with the last
    // computed row, without changing any captured pointer on the host.
    check_cuda(cudaMemcpyAsync(logits_,batch_logits_+size_t(tokens-1)*v,size_t(v)*4,cudaMemcpyDeviceToDevice,stream),"retain batched final head logits");
    check_cuda(cudaMemcpyAsync(final_norm_,normed+size_t(tokens-1)*h,size_t(h)*4,cudaMemcpyDeviceToDevice,stream),"retain batched final head norm");
    check_cuda(cudaMemcpyAsync(picked_,ids+tokens-1,4,cudaMemcpyDeviceToDevice,stream),"retain batched final head pick");
    check_cuda(cudaMemcpyAsync(invalid_,flags+tokens-1,4,cudaMemcpyDeviceToDevice,stream),"retain batched final head finite flag");
}
}
