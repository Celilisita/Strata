#include "strata/gemma4/native/nvfp4_projection.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/qwen35/spec.hpp"
#include "original_quant_projection.hpp"
#include "original_input_reuse.hpp"
#include <algorithm>
#include <stdexcept>
namespace strata::gemma4::native {
namespace p=strata::qwen35::native;
namespace mmq=strata::prefill::mmq;
std::size_t original_projection_input_bytes(int batch,int columns){
    if(batch<1||batch>8192||columns<256||columns>21504||columns%256)
        throw std::invalid_argument("Gemma original input arena geometry");
    return std::max(mmq::q8_bytes(batch,columns),strata::kernels::native_q8_1_bytes(columns,3));
}
void Nvfp4ProjectionService::initialize_original(const GgufFile& file,const Nvfp4BodyPlan& plan){
    if(plan.format!=GemmaWeightFormat::OriginalQuant||!mmq::built())
        throw std::invalid_argument("Gemma original requires the existing GPU MMQ provider");
    // Complete capability admission precedes every upload. No slow/dequant fallback.
    for(const auto& selected:plan.tensors){const auto& t=*selected.tensor;
        if((t.type!=12&&t.type!=14)||!mmq::supported(int(t.type))||!mmq::fits(int(t.type),int64_t(t.shape[1])))
            throw std::invalid_argument("Gemma original Q4_K/Q6_K MMQ type/tile unavailable");
    }
    original_input_=storage_.allocate(original_projection_input_bytes(maximum_batch_,plan.maximum_columns));
    tensors_.reserve(plan.tensors.size());values_.reserve(plan.tensors.size());
    for(const auto& selected:plan.tensors){const auto& t=*selected.tensor;const auto bytes=qwen35::tensor_bytes(file,t);
        void* gpu=storage_.allocate(bytes);
        p::check_cuda(cudaMemcpyAsync(gpu,file.tensor_data(t),bytes,cudaMemcpyHostToDevice,stream_),"Gemma original raw GGUF body upload");
        Matrix matrix{gpu,p::Quant(t.type),int(t.shape[1]),int(t.shape[0]),p::BlockLayout::Gguf};
        auto [it,inserted]=tensors_.emplace(&t,Entry{matrix,{}});
        if(!inserted||!values_.emplace(matrix.data,&it->second).second)
            throw std::invalid_argument("Gemma original duplicate projection owner");
    }
    original_mmq_=std::make_unique<mmq::Context>();
}
void Nvfp4ProjectionService::project_original(Matrix matrix,const float* input,float* output,int tokens){
    if(!original_mmq_||!original_input_||
       (matrix.quant!=p::Quant::Q4K&&matrix.quant!=p::Quant::Q6K)||matrix.layout!=p::BlockLayout::Gguf)
        throw std::logic_error("Gemma original projection format/owner mismatch");
    const int type=int(matrix.quant);
    const auto codec=original_input_codec(tokens,type);
    const bool reused=original_input_reuse_hit(this,input,tokens,matrix.columns,stream_,original_input_,codec);
    if(tokens<=3){
        if(!reused){
            strata::kernels::native_quantize_q8_1(input,original_input_,matrix.columns,tokens,stream_);
            original_input_reuse_publish(this,input,tokens,matrix.columns,stream_,original_input_,codec);
        }
        strata::kernels::native_mmvq(type,matrix.data,original_input_,output,matrix.columns,matrix.rows,tokens,stream_);
        return;
    }
    if(!reused){
        mmq::quantize(input,nullptr,original_input_,type,matrix.columns,matrix.columns,tokens,stream_);
        original_input_reuse_publish(this,input,tokens,matrix.columns,stream_,original_input_,codec);
    }
    mmq::Product product;
    product.w=matrix.data;product.type=type;product.w_rows=matrix.rows;product.w_cols=matrix.columns;
    product.expert_bytes=std::size_t(matrix.rows)*(matrix.columns/256)*(type==12?144:210);
    product.n=1;product.xq=original_input_; // Dense product: ids/bounds stay null, as existing GDN MMQ.
    product.total_rows=tokens;product.max_rows=tokens;product.dst=output;product.ld_dst=matrix.rows;
    original_mmq_->run(product,stream_);
}
}
