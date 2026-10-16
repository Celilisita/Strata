#include "router_prediction_cuda.cuh"
#include "router_prefetch_kernels.cuh"
#include "router_prediction_fused.cuh"
#include "router_prediction_cooperative.cuh"
#include "ggml-cuda/common.cuh"
#include "ggml-cuda.h"
#include "ggml-backend-impl.h"
#include <algorithm>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
namespace strata::deepseek_v4 {
namespace {
struct PredictionScratch {
    float *normalized=nullptr,*scores=nullptr,*partial=nullptr,*norm_partial=nullptr;
    uint64_t* clocks=nullptr;uint32_t* completed=nullptr;
    RouterPredictionRecord *host=nullptr,*device=nullptr;
    RouterPrefetchPacket *prefetch_host=nullptr,*prefetch_device=nullptr;
    PredictionScratch() {
        const char* profiling=std::getenv("STRATA_DSV4_ROUTER_PREFETCH_CLOCK");if(profiling&&!std::strcmp(profiling,"1"))CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&clocks),4*sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&normalized),3*2*4096*sizeof(float)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&scores),4*2*256*sizeof(float)));
        if(router_prefetch_enabled()){CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&partial),2*32*5*sizeof(float)));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&norm_partial),2*32*sizeof(float)));}
        if(router_prefetch_enabled()){CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&prefetch_host),sizeof(RouterPrefetchPacket),cudaHostAllocMapped));CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&prefetch_device),prefetch_host,0));new(prefetch_host)RouterPrefetchPacket{};
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&completed),sizeof(uint32_t)));CUDA_CHECK(cudaMemset(completed,0,sizeof(uint32_t)));}
        const char* tracing=std::getenv("STRATA_DSV4_ROUTER_PREDICTION_TRACE");if(!tracing||std::strcmp(tracing,"1"))return;
        constexpr size_t bytes=8192ull*43*sizeof(RouterPredictionRecord);
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&host),bytes,cudaHostAllocMapped));
        CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&device),host,0));
        std::memset(host,0,bytes);
    }
    ~PredictionScratch(){if(completed)cudaFree(completed);if(clocks)cudaFree(clocks);if(prefetch_host)cudaFreeHost(prefetch_host);if(host)cudaFreeHost(host);if(norm_partial)cudaFree(norm_partial);if(partial)cudaFree(partial);if(scores)cudaFree(scores);if(normalized)cudaFree(normalized);}
};
std::mutex prediction_mutex;
std::unordered_map<void*,std::unique_ptr<PredictionScratch>> prediction_scratch;
}
bool router_prediction_reserve(ggml_backend_t backend) {
    if(!router_prediction_enabled())return true;
    if(!backend||!ggml_backend_is_cuda(backend))return false;
    auto* context=static_cast<ggml_backend_cuda_context*>(backend->context);ggml_cuda_set_device(context->device);
    std::lock_guard<std::mutex> lock(prediction_mutex);
    if(!prediction_scratch.count(context))prediction_scratch.emplace(context,std::make_unique<PredictionScratch>());
    return true;
}
void router_prediction_release(ggml_backend_cuda_context& context) {
    std::lock_guard<std::mutex> lock(prediction_mutex);prediction_scratch.erase(&context);
}
void router_prediction_run(ggml_backend_cuda_context& context,ggml_tensor* tensor) {
    PredictionScratch* scratch=nullptr;
    {std::lock_guard<std::mutex> lock(prediction_mutex);auto found=prediction_scratch.find(&context);
        if(found==prediction_scratch.end())throw std::runtime_error("Prediction scratch absent");scratch=found->second.get();}
    const auto p=router_prediction_params(tensor);const int tokens=p.oracle>=3?1:int(p.oracle==1?tensor->src[3]->ne[1]:tensor->src[3]->ne[2]);
    auto* epoch=native_tg_epoch_pointer(context);if(!epoch)throw std::runtime_error("Prediction requires native route epoch");
    const auto data=[&](int n){return tensor->src[n]?static_cast<const float*>(tensor->src[n]->data):nullptr;};
    if(p.oracle>=3){const auto idata=[&](int n){return static_cast<const int32_t*>(tensor->src[n]->data);};
        prefetch_hash<<<1,32,0,context.stream()>>>(idata(1),idata(2),idata(3),idata(4),idata(5),idata(6),idata(7),idata(8),scratch->prefetch_device,epoch,p.oracle==4,static_cast<float*>(tensor->data));CUDA_CHECK(cudaGetLastError());return;}
    if(p.oracle==2&&router_prefetch_enabled()){
        const float* raw=data(3);const float* norm=data(4);const float* hc_fn=data(5);
        const float* scale=data(6);const float* base=data(7);const float* bias=data(8);
        const auto* weight=static_cast<const __nv_bfloat16*>(tensor->src[2]->data);
        const auto* directory=static_cast<const int32_t*>(tensor->src[9]->data);
        float* merged=scratch->normalized+2*2*4096;float* partial=scratch->partial;float* norm_partial=scratch->norm_partial;
        float* scores=scratch->scores;uint32_t* completed=scratch->completed;
        auto* packet=scratch->prefetch_device;auto* records=scratch->device;
        const uint32_t* epochs=epoch;float* sentinel=static_cast<float*>(tensor->data);uint64_t* clocks=scratch->clocks;
        RouterPredictionParams params=p;int rows=tokens;
        void* args[]{&raw,&norm,&hc_fn,&scale,&base,&weight,&bias,&directory,&merged,&partial,&norm_partial,
            &scores,&completed,&packet,&records,&epochs,&params,&rows,&sentinel,&clocks};
        CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(cooperative_router_prefetch),
            dim3(32,tokens),dim3(256),args,0,context.stream()));return;
    }
    if(p.oracle!=1)prediction_normalize<<<tokens,256,0,context.stream()>>>(data(3),data(1),data(4),data(5),data(6),data(7),
        scratch->normalized,scratch->device,epoch,p,tokens,scratch->clocks);
    const int candidates=p.oracle?1:3;
    prediction_project<<<dim3(64,candidates*tokens),128,0,context.stream()>>>(static_cast<const __nv_bfloat16*>(tensor->src[2]->data),
        p.oracle==1?data(3):scratch->normalized,data(8),scratch->scores,scratch->device,epoch,p,tokens);
    if(p.oracle==2){prefetch_select<<<1,256,0,context.stream()>>>(scratch->scores,static_cast<const int32_t*>(tensor->src[9]->data),scratch->prefetch_device,epoch,p.layer,tokens,static_cast<float*>(tensor->data),scratch->clocks);CUDA_CHECK(cudaGetLastError());return;}
    prediction_select<<<candidates*tokens,256,0,context.stream()>>>(scratch->scores,static_cast<const int32_t*>(tensor->src[9]->data),
        scratch->device,epoch,p,tokens,static_cast<float*>(tensor->data));CUDA_CHECK(cudaGetLastError());
}
bool router_prediction_trace_read(ggml_backend_t backend,RouterPredictionTrace* result,bool payload) {
    if(!backend||!result||!ggml_backend_is_cuda(backend))return false;
    auto* context=static_cast<ggml_backend_cuda_context*>(backend->context);
    auto* epoch=native_tg_epoch_pointer(*context);if(!epoch)return false;
    std::lock_guard<std::mutex> lock(prediction_mutex);auto found=prediction_scratch.find(context);if(found==prediction_scratch.end()||!found->second->host)return false;
    CUDA_CHECK(cudaMemcpy(&result->epochs,epoch,sizeof(uint32_t),cudaMemcpyDeviceToHost));
    if(result->epochs>8192)throw std::runtime_error("Prediction epoch capacity exhausted");
    if(payload){CUDA_CHECK(cudaStreamSynchronize(context->stream()));auto* first=found->second->host;
        result->records.assign(first,first+size_t(result->epochs)*43);}
    return true;
}
RouterPrefetchPacket* router_prefetch_packet(ggml_backend_t backend){
    std::lock_guard<std::mutex> lock(prediction_mutex);auto found=prediction_scratch.find(backend->context);return found==prediction_scratch.end()?nullptr:found->second->prefetch_host;
}
bool router_prefetch_probe_timing(ggml_backend_t backend,uint64_t* output){
    std::lock_guard<std::mutex> lock(prediction_mutex);auto found=prediction_scratch.find(backend->context);
    if(found==prediction_scratch.end()||!found->second->clocks)return false;CUDA_CHECK(cudaMemcpy(output,found->second->clocks,4*sizeof(uint64_t),cudaMemcpyDeviceToHost));return true;
}
bool router_prediction_set_probe_epoch(ggml_backend_t backend,uint32_t value) {
    if(!backend||!ggml_backend_is_cuda(backend)||value>2)return false;
    auto* context=static_cast<ggml_backend_cuda_context*>(backend->context);auto* epoch=native_tg_epoch_pointer(*context);if(!epoch)return false;
    CUDA_CHECK(cudaMemcpy(epoch,&value,sizeof(value),cudaMemcpyHostToDevice));return true;
}
bool router_prediction_probe_normalized(ggml_backend_t backend,float* output) {
    if(!backend||!output||!ggml_backend_is_cuda(backend))return false;
    std::lock_guard<std::mutex> lock(prediction_mutex);auto found=prediction_scratch.find(backend->context);if(found==prediction_scratch.end())return false;
    CUDA_CHECK(cudaMemcpy(output,found->second->normalized,3*2*4096*sizeof(float),cudaMemcpyDeviceToHost));return true;
}
}
