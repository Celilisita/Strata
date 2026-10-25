#include "strata/gemma4/native/sm120_attention.hpp"
#include "pp_tc_attention.hpp"
#include "sm120_global_merge.hpp"
#include "tg_local270.hpp"
#include "sm120_global_gqa.hpp"
#include "sm120_global_flash.hpp"
#include "whole_history.hpp"
#include "sm120_cached_global.hpp"
#include "global_keys1024_binding.hpp"
#include <algorithm>
#include <cstdlib>
#include <string>
#include <stdexcept>
namespace strata::gemma4::native {
namespace local_gqa2_candidate {
// Already-qualified checker COW host/CUDA objects; old PP APIs remain untouched.
void pp_tc_attention(const float*,float*,CacheView,const int*,int,int,int,AttentionWorkspace,cudaStream_t);
}
namespace {
bool local_gqa2_requested(){
    static const bool enabled=[](){
        const char* raw=std::getenv("STRATA_GEMMA_PP_TC_LOCAL_GQA2");
        const std::string value=raw?raw:"0";
        if(value!="0"&&value!="1")throw std::invalid_argument("STRATA_GEMMA_PP_TC_LOCAL_GQA2 requires0or1");
        return value=="1";
    }();
    return enabled;
}
}
Sm120AttentionService::Sm120AttentionService(int reserved,int batch,cudaStream_t stream):
    stream_(stream),reserved_(reserved),batch_(batch){
    sm120_attention_choice(1,0,reserved,batch);
    if(!stream)throw std::invalid_argument("Gemma attention ordered stream");
    pp_tc_=pp_tc_requested();global_t1_merge_=tg_global_merge_requested();local270_=tg_local270_requested();
    if(tg_global_gqa_requested()){
        if(!global_t1_merge_||gemma_weight_format_requested()!=GemmaWeightFormat::OriginalQuant)
            throw std::invalid_argument("Gemma global GQA requires original_quant and qualified global merge");
        prepare_global_t1_gqa(stream_);
    }
    const bool global_flash=global_flash_requested(),whole=whole_history_requested();
    const bool cached=cached_global_requested();
    const bool keys1024=global_keys1024::pp_tc_requested();
    const auto format=gemma_weight_format_requested();
    const bool native_backend=format==GemmaWeightFormat::OriginalQuant||format==GemmaWeightFormat::Nvfp4;
    if(local_gqa2_requested()&&(!pp_tc_||batch!=2048||
       !native_backend||global_flash||whole||cached))
        throw std::invalid_argument("Gemma local GQA2 requires a native backend, PP_TC, batch2048 and other PP leaves off");
    if(keys1024&&(!pp_tc_||batch!=2048||global_flash||whole||cached||
       !native_backend))
        throw std::invalid_argument("Gemma global Keys1024 requires a native backend, PP_TC, batch2048 and other PP leaves off");
    if(cached&&(!pp_tc_||global_flash||whole||gemma_weight_format_requested()!=GemmaWeightFormat::OriginalQuant))
        throw std::invalid_argument("Gemma cached global requires original_quant, PP_TC and other PP leaves off");
    if(whole&&(global_flash||gemma_weight_format_requested()!=GemmaWeightFormat::OriginalQuant))
        throw std::invalid_argument("Gemma wholehistory requires original_quant and global_flash=0");
    if(global_flash&&gemma_weight_format_requested()!=GemmaWeightFormat::OriginalQuant)
        throw std::invalid_argument("Gemma global flash candidate requires original_quant");
    constexpr std::size_t pp=4*(2*32*512*512+2*32*128*512+32*512*128+4*32*128);
    const auto split=std::size_t((reserved+127)/128)*32*std::min(3,batch)*(512+2)*4;
    workspace_.bytes=std::max(pp,split);
    if(pp_tc_)workspace_.bytes=std::max(workspace_.bytes,pp_tc_workspace_bytes(batch));
    if(keys1024)workspace_.bytes=std::max(workspace_.bytes,global_keys1024::pp_tc_workspace_bytes(batch));
    if(global_flash){
        workspace_.bytes=std::max(workspace_.bytes,global_flash_workspace_bytes(batch,reserved));
        prepare_global_flash(stream_);
    }
    if(whole)workspace_.bytes=std::max(workspace_.bytes,whole_history_workspace_bytes(batch,reserved));
    if(cached)workspace_.bytes=std::max(workspace_.bytes,cached_global_workspace_bytes(batch,reserved));
    workspace_.data=storage_.allocate(workspace_.bytes);
    workspace_.blas_bytes=32u*1024*1024;workspace_.blas_data=storage_.allocate(workspace_.blas_bytes);
    if(cublasCreate(&handle_)!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("Gemma FP32 attention handle");
    workspace_.handle=handle_;
}
Sm120AttentionService::~Sm120AttentionService(){cudaStreamSynchronize(stream_);if(handle_)cublasDestroy(handle_);}
void Sm120AttentionService::enqueue(const float* q,float* out,CacheView cache,const int* position,int tokens,int past,cudaStream_t stream){
    const bool local=cache.dim==256&&cache.heads==16&&cache.window==1024;
    const bool global=cache.dim==512&&cache.heads==4&&cache.window==0;
    if(stream!=stream_||(!local&&!global)||cache.capacity<tokens||
       cache.capacity!=(local?std::min(reserved_,1024+batch_):reserved_))
        throw std::invalid_argument("Gemma attention cache/stream identity");
    switch(sm120_attention_choice(tokens,past,reserved_,batch_)){
    case Sm120Attention::PackedQ8Split:
        if(local270_&&local&&tokens==1&&cache.capacity>=1024)
            tg_local270_attention(q,out,cache,position,32,tokens,reserved_,workspace_,stream);
        else if(global&&tokens==1&&tg_global_gqa_requested())split_attention_global_t1_gqa(q,out,cache,position,32,tokens,workspace_,stream);
        else if(global_t1_merge_&&global&&tokens==1)split_attention_global_t1_merge(q,out,cache,position,32,tokens,workspace_,stream);
        else split_attention(q,out,cache,position,32,tokens,workspace_,stream);
        break;
    case Sm120Attention::Float32Tiled:
        if(local_gqa2_requested()&&local&&pp_tc_&&batch_==2048&&past>=2048&&past%2048==0&&
           (tokens==2048||tokens==1328))
            local_gqa2_candidate::pp_tc_attention(q,out,cache,position,past,tokens,batch_,workspace_,stream);
        else if(global&&global_keys1024::pp_tc_requested()&&past>=2048&&(tokens==2048||tokens==1328))
            global_keys1024::pp_tc_attention(q,out,cache,position,past,tokens,batch_,workspace_,stream);
        else if(global&&cached_global_requested())cached_global_attention(q,out,cache,position,past,tokens,batch_,workspace_,stream);
        else if(global&&whole_history_requested())whole_history_attention(q,out,cache,position,past,tokens,batch_,workspace_,stream);
        else if(global&&global_flash_requested())global_flash_attention(q,out,cache,position,past,tokens,batch_,workspace_,stream);
        else if(pp_tc_)pp_tc_attention(q,out,cache,position,past,tokens,batch_,workspace_,stream);
        else blas_attention(q,out,cache,position,past,32,tokens,workspace_,stream);
        break;
    }
}
}
