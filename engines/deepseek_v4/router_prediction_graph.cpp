#include "router_prediction.hpp"
#include "router_prefetch.hpp"
#include <cstdlib>
#include <stdexcept>
namespace strata::deepseek_v4 {
static void reject_cpu(ggml_tensor*,const ggml_tensor*,const ggml_tensor*,const ggml_tensor*,int,int,void*) {
    GGML_ABORT("GPU router prediction cannot run on CPU");
}
ggml_tensor* build_router_prediction(ggml_context* ctx,ggml_cgraph* graph,ggml_backend_sched_t sched,
    ggml_tensor* raw,ggml_tensor* merged,ggml_tensor* router,ggml_tensor* norm,
    ggml_tensor* hc_fn,ggml_tensor* hc_scale,ggml_tensor* hc_base,ggml_tensor* bias,
    ggml_tensor* directory,ggml_backend_t backend,const RouterPredictionParams& p) {
    if(!router_prediction_enabled())return nullptr;
    if(!raw||!merged||!router||!backend||p.layer<3||p.layer>=43||p.gating!=4||p.norm_eps<=0||p.hc_eps<=0)
        throw std::runtime_error("Prediction model contract invalid");
    const int64_t rows=p.oracle==1?raw->ne[1]:raw->ne[2];
    if(rows<1||rows>2||raw->type!=GGML_TYPE_F32||merged->type!=GGML_TYPE_F32||!ggml_is_contiguous(raw)||!ggml_is_contiguous(merged)||
        router->type!=GGML_TYPE_BF16||router->ne[0]!=4096||router->ne[1]!=256||!ggml_is_contiguous(router)||
        raw->ne[0]!=4096||(p.oracle!=1&&raw->ne[1]!=4))throw std::runtime_error("Prediction hidden/router geometry invalid");
    if(p.oracle!=1&&(!norm||!hc_fn||!hc_scale||!hc_base||norm->type!=GGML_TYPE_F32||norm->ne[0]!=4096||
        hc_fn->type!=GGML_TYPE_F32||hc_fn->ne[0]!=16384||hc_fn->ne[1]!=24||hc_scale->type!=GGML_TYPE_F32||hc_base->type!=GGML_TYPE_F32))
        throw std::runtime_error("Prediction mHC geometry invalid");
    auto* scalar=ggml_view_1d(ctx,raw,1,0);
    auto* result=ggml_map_custom3(ctx,scalar,merged,router,reject_cpu,1,reinterpret_cast<void*>(router_prediction_tag));
    result->src[3]=raw;result->src[4]=norm;result->src[5]=hc_fn;result->src[6]=hc_scale;result->src[7]=hc_base;result->src[8]=bias;
    if(!directory||directory->type!=GGML_TYPE_I32||ggml_nelements(directory)!=256||!ggml_is_contiguous(directory))throw std::runtime_error("Prediction directory invalid");
    result->src[9]=directory;
    std::memcpy(reinterpret_cast<char*>(result->op_params)+24,&p,sizeof(p));
    ggml_set_name(result,p.oracle==2?"strata_early_router_prefetch":(p.oracle==1?"strata_router_oracle_probe":"strata_early_router_probe"));
    if(sched)ggml_backend_sched_set_tensor_backend(sched,result,backend);ggml_build_forward_expand(graph,result);return result;
}
ggml_tensor* build_router_hash_prefetch(ggml_context* ctx,ggml_cgraph* graph,ggml_tensor* raw,
    ggml_tensor* pending,ggml_tensor* draft,ggml_tensor* const* tables,ggml_tensor* const* directories,ggml_backend_t backend,bool both) {
    if(!router_prefetch_enabled())return nullptr;
    if(!raw||raw->type!=GGML_TYPE_F32||!pending||pending->type!=GGML_TYPE_I32||ggml_nelements(pending)!=1||
        !draft||draft->type!=GGML_TYPE_I32||ggml_nelements(draft)!=1)throw std::runtime_error("Hash prefetch token contract invalid");
    auto* scalar=ggml_view_1d(ctx,raw,1,0);auto* result=ggml_map_custom3(ctx,scalar,pending,draft,reject_cpu,1,reinterpret_cast<void*>(router_prediction_tag));
    for(int i=0;i<3;++i){if(!tables[i]||tables[i]->type!=GGML_TYPE_I32||tables[i]->ne[0]!=6||tables[i]->ne[1]!=129280||!ggml_is_contiguous(tables[i])||
            !directories[i]||directories[i]->type!=GGML_TYPE_I32||ggml_nelements(directories[i])!=256)throw std::runtime_error("Hash prefetch table contract invalid");
        result->src[3+i]=tables[i];result->src[6+i]=directories[i];}
    RouterPredictionParams params{1e-20f,1e-6f,0,4,uint16_t(both?4:3)};std::memcpy(reinterpret_cast<char*>(result->op_params)+24,&params,sizeof(params));
    ggml_set_name(result,"strata_hash_prefetch");ggml_set_output(result);ggml_build_forward_expand(graph,result);return result;
}
}
