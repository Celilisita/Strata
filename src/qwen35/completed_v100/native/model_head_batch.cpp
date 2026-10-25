#include "q6_soa.hpp"
#include "q6_head_policy.hpp"
#include "strata/qwen35/completed_v100/native/model.hpp"
#include "strata/qwen35/completed_v100/native/ops.hpp"
#include "strata/qwen35/completed_v100/native/tiny_batch.hpp"
#include "strata/qwen35/completed_v100/native/output_head.hpp"
#include "strata/qwen35/completed_v100/native/raw_q8_projection.hpp"
namespace strata::completed_v100::qwen35::native {
void output_head_batch(Matrix weight,const float *norm,const float *hidden,int rows,
                       float epsilon,OutputHeadBatch b,cudaStream_t stream,TokenMask mask){
    if(!norm||!hidden||!b.normalized||!b.logits||!b.quantized||!b.greedy||!b.ids||!b.flags||
       (rows!=2&&rows!=3)||epsilon<=0||b.greedy_bytes<size_t(rows)*greedy_workspace(weight.rows))
        throw std::invalid_argument("invalid native head batch buffers");
    original_matrix_bytes(weight);
    rms_norm(hidden,norm,b.normalized,weight.columns,rows,epsilon,stream);
    if(weight.quant==Quant::Q8)raw_q8_projection(weight,b.normalized,b.logits,rows,stream);
    else {tiny_batch_quantize(b.normalized,b.quantized,weight.columns,rows,stream);
        tiny_batch_linear(weight,b.quantized,b.logits,rows,stream);}
    greedy_pick_batch(b.logits,weight.rows,rows,b.greedy,b.greedy_bytes,b.ids,b.flags,stream,mask);
}
void DenseModel::head_batch(const float *hidden,int tokens,float *normed,int *ids,unsigned *flags,cudaStream_t stream){
    if(!hidden||!normed||!ids||!flags||(tokens!=2&&tokens!=3))throw std::invalid_argument("invalid native batched head");
    const int h=int(spec_.n_embd),v=int(spec_.n_vocab);
    // The wide shared activation arena is dead after the final model layer.
    // Its bytes now hold the same Q8 activation format used by the scalar head.
    void *prepared=shared_->shared_ffn(tokens).buffers().half_input;
    if(q6_soa_enabled()&&weights_.head.quant==Quant::Q6K&&weights_.head.layout==BlockLayout::Gguf) {
        rms_norm(hidden,weights_.output_norm,normed,h,tokens,float(spec_.rms_epsilon),stream);
        q6_soa_prepare(normed,prepared,h,tokens,stream);
        const int warps=q6_head_warps();
        if(warps&&tokens==3)project_q6_head(weights_.head,prepared,batch_logits_,tokens,warps,stream);
        else q6_soa_project(weights_.head,prepared,batch_logits_,tokens,stream);
        greedy_pick_batch(batch_logits_,v,tokens,batch_greedy_,3*greedy_bytes_,ids,flags,stream,token_mask_);
    }else if(weights_.head.quant==Quant::Q8) {
        rms_norm(hidden,weights_.output_norm,normed,h,tokens,float(spec_.rms_epsilon),stream);
        shared_->project(weights_.head,normed,batch_logits_,tokens,{},stream);
        greedy_pick_batch(batch_logits_,v,tokens,batch_greedy_,3*greedy_bytes_,ids,flags,stream,token_mask_);
    }else output_head_batch(weights_.head,weights_.output_norm,hidden,tokens,float(spec_.rms_epsilon),
                      {normed,batch_logits_,prepared,batch_greedy_,3*greedy_bytes_,ids,flags},stream,token_mask_);
    // Keep the ordinary model's diagnostic/output state coherent with the last
    // computed row, without changing any captured pointer on the host.
    check_cuda(cudaMemcpyAsync(logits_,batch_logits_+size_t(tokens-1)*v,size_t(v)*4,cudaMemcpyDeviceToDevice,stream),"retain batched final head logits");
    check_cuda(cudaMemcpyAsync(final_norm_,normed+size_t(tokens-1)*h,size_t(h)*4,cudaMemcpyDeviceToDevice,stream),"retain batched final head norm");
    check_cuda(cudaMemcpyAsync(picked_,ids+tokens-1,4,cudaMemcpyDeviceToDevice,stream),"retain batched final head pick");
    check_cuda(cudaMemcpyAsync(invalid_,flags+tokens-1,4,cudaMemcpyDeviceToDevice,stream),"retain batched final head finite flag");
}
}
