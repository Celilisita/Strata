#include "strata/gemma4/native/nvfp4_projection.hpp"
#include "strata/qwen35/native/nvfp4_runtime.hpp"
#include "fp4_compensated_service.hpp"
#include "sm120_tiny_projection.hpp"
#include "sm120_native_group_codec.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace strata::gemma4::native {
namespace p=strata::qwen35::native;
Nvfp4ProjectionService::Nvfp4ProjectionService(const GgufFile& file,const Spec& spec,int batch,cudaStream_t stream):stream_(stream),maximum_batch_(batch){
    if(!stream||batch<1||batch>8192)throw std::invalid_argument("Gemma projection stream/batch");
    const auto plan=plan_nvfp4_body(file,spec);
    format_=plan.format;tiny_compensated_=gemma_tiny_compensated_requested();
    if(format_==GemmaWeightFormat::OriginalQuant){
        if(tiny_compensated_)throw std::invalid_argument("Gemma original format forbids NVFP4 tiny policy");
        try{initialize_original(file,plan);prepare({1,batch});
            p::check_cuda(cudaStreamSynchronize(stream),"Gemma original immutable inventory lease");}
        catch(...){cudaStreamSynchronize(stream);throw;}
        return;
    }
    const int rows=std::max(16,batch),columns=plan.maximum_columns;
    try{
    input_={storage_.allocate(p::nvfp4_value_bytes(rows,columns)),
        storage_.allocate(p::nvfp4_scale_bytes(rows,columns)),rows,columns};
    p::check_cuda(cudaMemsetAsync(input_.scales,0,p::nvfp4_scale_bytes(rows,columns),stream),"Gemma NVFP4 padded scales");
    if(tiny_compensated_){
        const auto resources=p::fp4_compensated_resources(false);
        if(resources.blocks<1||resources.local_bytes)throw std::runtime_error("Gemma tiny pure-linear resource admission");
        low_={storage_.allocate(p::nvfp4_value_bytes(16,columns)),
            storage_.allocate(p::nvfp4_scale_bytes(16,columns)),16,columns};
        p::check_cuda(cudaMemsetAsync(low_.scales,0,p::nvfp4_scale_bytes(16,columns),stream),"Gemma tiny low padded scales");
        canonical_values_=storage_.allocate(gemma_tiny_canonical_scratch(plan));
    }
    tensors_.reserve(plan.tensors.size());values_.reserve(plan.tensors.size());
    for(const auto& selected:plan.tensors){
        p::Nvfp4View view;p::Matrix matrix;
        {p::CompensatedFfnLoadScope canonical(0);matrix=p::load_nvfp4_matrix(file,*selected.tensor,storage_,view,stream);}
        if(matrix.layout!=p::BlockLayout::Nvfp4Planes||view.input_scale!=1)
            throw std::invalid_argument("Gemma requires canonical planes without Qwen physical packing");
        if(tiny_compensated_){
            const auto values=p::nvfp4_value_bytes(view.rows,view.columns),scales=p::nvfp4_scale_bytes(view.rows,view.columns);
            auto* packed=static_cast<unsigned char*>(storage_.allocate(values+scales));
            p::fp4_pack_values(view,packed,stream);
            p::check_cuda(cudaMemcpyAsync(packed+values,view.scales,scales,cudaMemcpyDeviceToDevice,stream),"Gemma immutable tiny SF copy");
            p::check_cuda(cudaStreamSynchronize(stream),"Gemma canonical to packed ownership handoff");
            storage_.release(const_cast<void*>(matrix.data));
            matrix.data=packed;matrix.layout=p::BlockLayout::Nvfp4WarpFragments;
            view.values=packed;view.scales=packed+values;
        }
        auto [it,inserted]=tensors_.emplace(selected.tensor,Entry{matrix,view});
        if(!inserted||!values_.emplace(matrix.data,&it->second).second)
            throw std::invalid_argument("Gemma duplicate projection owner");
    }
    gemm_=std::make_unique<p::Nvfp4Gemm>(stream,32u*1024*1024,plan.maximum_rows);
    prepare({1,batch});
    p::check_cuda(cudaStreamSynchronize(stream),"Gemma NVFP4 immutable inventory lease");
    }catch(...){cudaStreamSynchronize(stream);throw;}
}
Nvfp4ProjectionService::~Nvfp4ProjectionService(){cudaStreamSynchronize(stream_);}
Nvfp4ProjectionService::Matrix Nvfp4ProjectionService::matrix(const TensorInfo& tensor)const{
    const auto found=tensors_.find(&tensor);
    if(found==tensors_.end())throw std::invalid_argument("Gemma tensor outside owned projection body");
    return found->second.matrix;
}
void Nvfp4ProjectionService::prepare(const std::vector<int>& counts){
    cudaStreamCaptureStatus capture{};
    p::check_cuda(cudaStreamIsCapturing(stream_,&capture),"Gemma projection PREPARE capture state");
    if(capture!=cudaStreamCaptureStatusNone)throw std::logic_error("Gemma projection PREPARE precedes capture");
    for(int count:counts){
        if(count<1||count>maximum_batch_)throw std::invalid_argument("Gemma projection prepared token extent");
        const int padded=std::max(16,count);
        if(std::find(prepared_.begin(),prepared_.end(),padded)!=prepared_.end())continue;
        if(format_==GemmaWeightFormat::Nvfp4)for(const auto& tensor:tensors_){
            auto view=tensor.second.view;if(tiny_compensated_)view.values=canonical_values_;
            gemm_->prepare(view,input_,count);
        }
        prepared_.push_back(padded);
    }
}
void Nvfp4ProjectionService::project(Matrix matrix,const float* input,float* output,int tokens,cudaStream_t stream){
    const auto found=values_.find(matrix.data);
    if(found==values_.end())throw std::invalid_argument("Gemma projection inventory identity");
    const auto& entry=*found->second;const auto& admitted=entry.matrix;
    if(matrix.quant!=admitted.quant||matrix.layout!=admitted.layout||matrix.rows!=admitted.rows||
       matrix.columns!=admitted.columns||stream!=stream_||!input||!output||tokens<1||
       tokens>maximum_batch_||std::find(prepared_.begin(),prepared_.end(),std::max(16,tokens))==prepared_.end())
        throw std::invalid_argument("Gemma projection geometry/stream/prepared shape");
    const auto x=reinterpret_cast<std::uintptr_t>(input),y=reinterpret_cast<std::uintptr_t>(output);
    const auto input_bytes=std::size_t(tokens)*matrix.columns*4,output_bytes=std::size_t(tokens)*matrix.rows*4;
    if(input_bytes>std::numeric_limits<std::uintptr_t>::max()-x||output_bytes>std::numeric_limits<std::uintptr_t>::max()-y||
       (x<y+output_bytes&&y<x+input_bytes))
        throw std::invalid_argument("Gemma projection input/output alias");
    if(format_==GemmaWeightFormat::OriginalQuant){
        project_original(matrix,input,output,tokens);++calls_;return;
    }
    if(tiny_compensated_&&tokens<=3){
        p::fp4_compensated_quantize(input,{input_,low_},tokens,matrix.columns,entry.view.input_scale,stream);
        p::fp4_compensated_linear({entry.view},{input_,low_},output,tokens,stream);++tiny_calls_;
    }else{
        auto view=entry.view;
        if(tiny_compensated_){
            // One scratch lease, read completely by this GEMM before the next
            // projection unpacks into it on the same ordered stream.
            p::fp4_unpack_values({entry.view},canonical_values_,stream);view.values=canonical_values_;
            ++canonical_materializations_;
        }
        p::nvfp4_quantize(input,input_,tokens,matrix.columns,view.input_scale,stream);
        gemm_->project(view,input_,output,tokens,stream);
    }
    ++calls_;
}
void Nvfp4ProjectionService::project_group(const float* input,std::initializer_list<std::pair<Matrix,float*>> group,int tokens,cudaStream_t stream){
    if(format_!=GemmaWeightFormat::Nvfp4||!tiny_compensated_||!native_group_codec_tokens(tokens)||stream!=stream_||!input||
       tokens>maximum_batch_||(group.size()!=2&&group.size()!=3)||
       std::find(prepared_.begin(),prepared_.end(),std::max(16,tokens))==prepared_.end())
        throw std::invalid_argument("Gemma explicit native group format/stream/prepared extent");
    const auto overlaps=[](const void* a,std::size_t an,const void* b,std::size_t bn){const auto x=reinterpret_cast<std::uintptr_t>(a),y=reinterpret_cast<std::uintptr_t>(b);
        if(!a||!b||an>std::numeric_limits<std::uintptr_t>::max()-x||bn>std::numeric_limits<std::uintptr_t>::max()-y)
            throw std::invalid_argument("Gemma native group span overflow");return x<y+bn&&y<x+an;};
    std::array<NativeGroupOperand,3> operands{};std::size_t index=0;
    // Finish all owner/geometry/alias validation before the first GPU submission.
    for(const auto& item:group){const auto& matrix=item.first;const auto found=values_.find(matrix.data);
        if(found==values_.end())throw std::invalid_argument("Gemma group outside owned inventory");const auto& e=*found->second;const auto& a=e.matrix;
        if(matrix.quant!=a.quant||matrix.layout!=a.layout||matrix.rows!=a.rows||matrix.columns!=a.columns||
           matrix.quant!=p::Quant::NVFP4||matrix.columns!=5376||e.view.input_scale!=1||!item.second||
           overlaps(input,std::size_t(tokens)*5376*4,item.second,std::size_t(tokens)*matrix.rows*4))
            throw std::invalid_argument("Gemma native group owned geometry/output span");
        for(const auto& earlier:group){if(&earlier==&item)break;
            if(overlaps(earlier.second,std::size_t(tokens)*earlier.first.rows*4,item.second,std::size_t(tokens)*matrix.rows*4))
                throw std::invalid_argument("Gemma native group output/output alias");}
        operands[index++]={e.view,item.second};
    }
    NativeGroupCounts count;
    if(index==2)count=execute_native_group(input,{operands[0],operands[1]},input_,low_,canonical_values_,*gemm_,tokens,stream);
    else count=execute_native_group(input,{operands[0],operands[1],operands[2]},input_,low_,canonical_values_,*gemm_,tokens,stream);
    calls_+=count.projections;tiny_calls_+=count.tiny;canonical_materializations_+=count.canonical_materializations;
}
std::size_t Nvfp4ProjectionService::bytes()const{return storage_.bytes()+(gemm_?gemm_->bytes():0);}
}
