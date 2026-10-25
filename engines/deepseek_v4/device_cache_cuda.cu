#include "device_cache_cuda.hpp"
#include "device_cache_protocol.hpp"
#include "../glm5_next/route_trace_cuda.cuh"
#include "../glm5_next/hybrid_join.cuh"
#include "../glm5_next/causal_phases.cuh"
#include "ggml-cuda/common.cuh"
#include "../glm5_next/cold_gpu_rows.cuh"
#include "../glm5_next/partial_down_cuda.cuh"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include <climits>
#include <cstring>
#include <type_traits>

namespace {
using strata::deepseek_v4::DeviceMissPacket;
using strata::deepseek_v4::device_frequency_scores;
__device__ unsigned long long nanoseconds() {
    unsigned long long value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}
__device__ bool probe_routes(const volatile int32_t* directory, const int32_t* ids, int32_t* slots,
                             int32_t* status, unsigned long long* counters, int count, int experts, int layer, int first, int last) {
    atomicAdd(counters, 1ull);
    bool missing = false;
    int hits = 0;uint32_t cold_mask=0;
    for (int i = 0; i < count; ++i) {
        const int id = ids[i];
        if (id < 0 || id >= experts) { atomicExch(status, 1); slots[i] = 0; missing = true; }
        else { slots[i] = directory[id]; missing |= slots[i] < 0; hits += slots[i] >= 0;
            if(slots[i]<0)cold_mask|=1u<<(experts==288?i/8:0); }
    }
    atomicAdd(counters + 2, static_cast<unsigned long long>(hits));
    const bool cold = missing || *status;
    if (!cold) atomicAdd(counters + 1, 1ull);
    atomicAdd(counters + 8 + layer*3, 1ull);
    if (!cold) atomicAdd(counters + 9 + layer*3, 1ull);
    atomicAdd(counters + 10 + layer*3, static_cast<unsigned long long>(count-hits));
    if(layer>=first&&layer<=last) {
        if(layer==first)counters[7]=0;
        counters[7]|=cold_mask;
        if(layer==last){const unsigned positions=experts==288?count/8:1;
            atomicAdd(counters+5,static_cast<unsigned long long>(positions));
            for(unsigned p=0;p<positions;++p)if(!(counters[7]&(1ull<<p)))atomicAdd(counters+6,1ull);}
    }
    return cold;
}
__device__ void observe_frequency(uint32_t* frequency, int layer, const int32_t* ids, size_t score_count, int count, int experts) {
    for (int i = 0; i < count; ++i) {
        if (++frequency[score_count] == 65536) {
            for (size_t key = 0; key < score_count; ++key) frequency[key] /= 2;
            frequency[score_count] = 0;
        }
        auto& score = frequency[layer * experts + ids[i]];
        if (score < 65535) ++score;
    }
}
template<bool CausalPhases=false,bool ColdGpu=false,unsigned ColdRows=2,bool PartialDown=false>
__global__ void lookup_kernel(const volatile int32_t* directory, const int32_t* ids, int32_t* slots,
                              int32_t* status, unsigned long long* counters,
                              DeviceMissPacket* packet, int layer, unsigned long long timeout, uint32_t* frequency, size_t score_count,
                              int count, int experts, int first, int last, uint64_t epoch, const float* activation,
                              strata::glm5_next::RouteTraceHeader* trace,uint32_t* state,bool overlap,
                              strata::glm5_next::ColdGpuLayer* cold_gpu=nullptr,const strata::glm5_next::PartialDownBridge* partial_bridge=nullptr,const int32_t* partial_indices=nullptr) {
    static_assert(!PartialDown||(ColdGpu&&ColdRows==2),"partial Down requires original cold2");
    if(state&&threadIdx.x==0){state[0]=0;state[1]=0;}
    if (!frequency) {
        if(threadIdx.x!=0)return;
        const bool cold=probe_routes(directory,ids,slots,status,counters,count,experts,layer,first,last);
        strata::glm5_next::capture_routes(trace,layer,ids,slots,count);
        if constexpr(ColdGpu)strata::glm5_next::cold_gpu_select<ColdRows>(cold_gpu,ids,slots,status,epoch,layer);
        if(!cold)return;
    } else {
        __shared__ bool cold;
        if (threadIdx.x == 0) {
            cold = probe_routes(directory, ids, slots, status, counters, count, experts, layer, first, last);
            strata::glm5_next::capture_routes(trace,layer,ids,slots,count);
            if (!*status) observe_frequency(frequency, layer, ids, score_count, count, experts);
            if constexpr(ColdGpu)strata::glm5_next::cold_gpu_select<ColdRows>(cold_gpu,ids,slots,status,epoch,layer);
        }
        __syncthreads();
        if (!cold) return; // Allhit keeps history entirely in device memory.
        if (!*status) {
            // Publish from this same lookup. Enqueueing a separate D2H copy
            // while the compute stream awaits this miss can block in WDDM.
            if(!overlap)for (size_t key = threadIdx.x; key <= score_count; key += blockDim.x)
                packet->frequency[key] = frequency[key];
            if(activation)for(int i=threadIdx.x;i<(count/8)*4096;i+=blockDim.x)packet->activation[i]=activation[i];
            // Each writer must flush its own mapped writes before thread0
            // publishes the ticket; a fence in thread0 alone is insufficient.
            __threadfence_system();
            __syncthreads();
        }
        if (threadIdx.x != 0) return;
    }
    // Only the cold path touches the host mailbox, including its epoch.
    if(packet->owner_epoch!=epoch)atomicExch(status,1);
    if (!*status) {
        packet->native_request = overlap?5:activation?4:0;
        packet->routed_count = count;
        if constexpr(ColdGpu){packet->gpu_cold_mask=cold_gpu->active_mask;
            for(int rank=0;rank<16;++rank)if(cold_gpu->active_mask&(1u<<rank))packet->reply_slots[rank]=slots[rank];}
        packet->cpu_mask=0;
        packet->layer = unsigned(layer);
        for (int i = 0; i < count; ++i) packet->ids[i] = ids[i];
        const unsigned sequence = packet->sequence + 1;
        if(state&&!sequence){atomicExch(status,1);return;}
        if constexpr(PartialDown)strata::glm5_next::partial_down_publish_candidate(partial_bridge,cold_gpu,ids,slots,partial_indices,sequence,epoch,status);
        if constexpr(PartialDown)if(*status)return;
        if(overlap){state[1]=sequence;for(int i=0;i<count;++i)if(slots[i]<0){slots[i]=-1;state[0]|=1u<<i;}}
        // Keep the original cold ticket even if all missing rows moved to GPU.
        if constexpr(ColdGpu)if(overlap)state[0]=cold_gpu->original_missing_mask;
        __threadfence_system();
        strata::glm5_next::causal_phase_publish<CausalPhases>(counters,sequence,layer,epoch);
        if(state)strata::glm5_next::HybridWord(strata::glm5_next::hybrid_plain(packet->sequence)).store(sequence,cuda::memory_order_release);
        else packet->sequence = sequence;
        __threadfence_system();
        atomicAdd(counters + 3, 1ull);
        if(overlap)return; // D4 schedule: resident/shared products precede CPU join.
        const unsigned long long begin = nanoseconds();
        while (packet->response != sequence) {
            if (nanoseconds() - begin >= timeout) {
                atomicExch(status, 1);
                atomicAdd(counters + 4, 1ull);
                break;
            }
            __nanosleep(1000);
        }
        __threadfence_system();
        if (packet->error) atomicExch(status, 1);
    }
    for (int i = 0; i < count; ++i) {
        const int id = ids[i];
        const int slot = activation ? packet->reply_slots[i] : id >= 0 && id < experts ? directory[id] : -1;
        if (*status || (slot < 0 && !(activation&&slot==-1))) { slots[i] = 0; atomicExch(status, 1); }
        else slots[i] = slot;
    }
}
__global__ void guard_kernel(const float* input, float* output, const int32_t* status, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) output[i] = *status ? __int_as_float(0x7fc00000) : input[i];
}
}
namespace {
bool partial_down_mid_cuda(ggml_backend_cuda_context& context,ggml_tensor* tensor){
    using namespace strata::glm5_next;using strata::deepseek_v4::DeviceMissPacket;
    if(!partial_down_mid(tensor))return false;
    GGML_ASSERT(partial_down_mid_supported(tensor)&&strata::deepseek_v4::device_cache_lookup(tensor->src[1])&&context.curr_stream_no==0);
    uintptr_t packet=0;uint64_t epoch=0,timeout=0;
    std::memcpy(&packet,tensor->op_params,sizeof(packet));std::memcpy(&epoch,tensor->op_params+6,sizeof(epoch));
    std::memcpy(&timeout,tensor->op_params+4,sizeof(timeout));GGML_ASSERT(packet&&epoch&&timeout);
    const auto* bridge=static_cast<const PartialDownBridge*>(tensor->src[2]->data);
    const auto* state=static_cast<const uint32_t*>(tensor->src[4]->data);
    auto* status=static_cast<int32_t*>(tensor->src[3]->data);auto* mailbox=reinterpret_cast<DeviceMissPacket*>(packet);
    partial_down_copy_rows<<<dim3(48,2),256,0,context.stream()>>>(bridge,state,mailbox,status,epoch,timeout);
    partial_down_read_mid<<<32,256,0,context.stream()>>>(bridge,state,mailbox,status,epoch,timeout,static_cast<float*>(tensor->data));
    partial_down_mid_consumed<<<1,1,0,context.stream()>>>(bridge,state,status);
    CUDA_CHECK(cudaGetLastError());return true;
}
}
bool strata_device_cache_lookup_cuda(ggml_backend_cuda_context& context, ggml_tensor* tensor) {
    if(partial_down_mid_cuda(context,tensor))return true;
    using namespace strata::deepseek_v4;
    if (!device_cache_lookup(tensor)) return false;
    const int count=int(ggml_nelements(tensor)), experts=int(ggml_nelements(tensor->src[0]));
    GGML_ASSERT(tensor->type == GGML_TYPE_I32 && (count==6||count==8||((count==16||count==32)&&experts==288)) && (experts==256||experts==288) && ggml_is_contiguous(tensor));
    if(count==32)GGML_ASSERT(device_cache_supported(tensor));
    GGML_ASSERT(tensor->src[0]->type == GGML_TYPE_I32 && ggml_nelements(tensor->src[0]) == experts &&
                tensor->src[1]->type == GGML_TYPE_I32 && ggml_nelements(tensor->src[1]) == count &&
                tensor->src[2]->type == GGML_TYPE_I32 && ggml_nelements(tensor->src[2]) == 1 &&
                tensor->src[3]->type == GGML_TYPE_I64 && ggml_nelements(tensor->src[3]) == device_cache_counter_count);
    for (int i = 0; i < 4; ++i) GGML_ASSERT(ggml_is_contiguous(tensor->src[i]));
    if(tensor->src[6])GGML_ASSERT(tensor->src[6]->type==GGML_TYPE_I8&&ggml_is_contiguous(tensor->src[6])&&
        count<=16&&ggml_nbytes(tensor->src[6])==strata::glm5_next::route_trace_bytes);
    if(tensor->src[7])GGML_ASSERT(tensor->src[7]->type==GGML_TYPE_I32&&ggml_is_contiguous(tensor->src[7])&&ggml_nelements(tensor->src[7])==2);
    GGML_ASSERT(!tensor->op_params[11]||(experts==288&&tensor->src[5]&&tensor->src[4]&&tensor->src[7]));
    uintptr_t packet = 0;
    uint64_t timeout = 0, epoch = 0;
    std::memcpy(&packet, tensor->op_params, sizeof(packet));
    std::memcpy(&timeout, tensor->op_params + 4, sizeof(timeout));
    std::memcpy(&epoch, tensor->op_params + 6, sizeof(epoch));
    GGML_ASSERT(packet && timeout && tensor->op_params[2] >= 0 && tensor->op_params[2] < device_cache_max_layers);
    const bool causal=experts==288&&tensor->op_params[11]&&(count==8||count==16)&&strata::glm5_next::causal_phases_requested();
    if(causal)GGML_ASSERT(strata::glm5_next::causal_phase_layer_valid(tensor->op_params[2])&&tensor->op_params[9]<46);
    strata::glm5_next::ColdGpuLayer* cold_gpu=nullptr;
    const unsigned cold_rows=strata::glm5_next::cold_gpu_magic_rows(tensor->op_params[strata::glm5_next::cold_gpu_magic_word]);
    if(cold_rows){
        GGML_ASSERT(device_cache_supported(tensor));uintptr_t metadata=0;
        std::memcpy(&metadata,tensor->op_params+strata::glm5_next::cold_gpu_pointer_word,sizeof(metadata));
        cold_gpu=reinterpret_cast<strata::glm5_next::ColdGpuLayer*>(metadata);}
    auto launch=[&](auto tag){lookup_kernel<decltype(tag)::value><<<1, 32, 0, context.stream()>>>(static_cast<const int32_t*>(tensor->src[0]->data),
        static_cast<const int32_t*>(tensor->src[1]->data), static_cast<int32_t*>(tensor->data),
        static_cast<int32_t*>(tensor->src[2]->data), static_cast<unsigned long long*>(tensor->src[3]->data),
        reinterpret_cast<DeviceMissPacket*>(packet), tensor->op_params[2], timeout,
        tensor->src[4] ? static_cast<uint32_t*>(tensor->src[4]->data) : nullptr, tensor->src[4] ? size_t(tensor->src[4]->ne[0]-1) : 0,
        count, experts, tensor->op_params[8], tensor->op_params[9], epoch, tensor->src[5] ? static_cast<const float*>(tensor->src[5]->data) : nullptr,
        tensor->src[6]?static_cast<strata::glm5_next::RouteTraceHeader*>(tensor->src[6]->data):nullptr,
        tensor->src[7]?static_cast<uint32_t*>(tensor->src[7]->data):nullptr,tensor->op_params[11]!=0);};
    if(cold_gpu){auto launch_cold=[&](auto tag,auto rows,auto partial){lookup_kernel<decltype(tag)::value,true,decltype(rows)::value,decltype(partial)::value><<<1,32,0,context.stream()>>>(
            static_cast<const int32_t*>(tensor->src[0]->data),static_cast<const int32_t*>(tensor->src[1]->data),static_cast<int32_t*>(tensor->data),
            static_cast<int32_t*>(tensor->src[2]->data),static_cast<unsigned long long*>(tensor->src[3]->data),reinterpret_cast<DeviceMissPacket*>(packet),
            tensor->op_params[2],timeout,tensor->src[4]?static_cast<uint32_t*>(tensor->src[4]->data):nullptr,tensor->src[4]?size_t(tensor->src[4]->ne[0]-1):0,
            count,experts,tensor->op_params[8],tensor->op_params[9],epoch,tensor->src[5]?static_cast<const float*>(tensor->src[5]->data):nullptr,
            tensor->src[6]?static_cast<strata::glm5_next::RouteTraceHeader*>(tensor->src[6]->data):nullptr,
            tensor->src[7]?static_cast<uint32_t*>(tensor->src[7]->data):nullptr,true,cold_gpu,
            decltype(partial)::value?static_cast<const strata::glm5_next::PartialDownBridge*>(tensor->src[8]->data):nullptr,
            decltype(partial)::value?static_cast<const int32_t*>(tensor->src[9]->data):nullptr);};
        if(tensor->src[8]){
            GGML_ASSERT(cold_rows==2&&tensor->src[9]&&context.curr_stream_no==0);
            if(causal)launch_cold(std::true_type{},std::integral_constant<unsigned,2>{},std::true_type{});
            else launch_cold(std::false_type{},std::integral_constant<unsigned,2>{},std::true_type{});
        }else if(cold_rows==4){
            if(causal)launch_cold(std::true_type{},std::integral_constant<unsigned,4>{},std::false_type{});
            else launch_cold(std::false_type{},std::integral_constant<unsigned,4>{},std::false_type{});
        }else{
            if(causal)launch_cold(std::true_type{},std::integral_constant<unsigned,2>{},std::false_type{});
            else launch_cold(std::false_type{},std::integral_constant<unsigned,2>{},std::false_type{});
        }
        CUDA_CHECK(cudaGetLastError());
        if(cold_rows==4)strata::glm5_next::cold_gpu_copy_rows4<<<dim3(48,12),256,0,context.stream()>>>(cold_gpu,static_cast<int32_t*>(tensor->src[2]->data),epoch,tensor->op_params[2]);
        else strata::glm5_next::cold_gpu_copy_rows<<<dim3(48,6),256,0,context.stream()>>>(cold_gpu,static_cast<int32_t*>(tensor->src[2]->data),epoch,tensor->op_params[2]);
    }else{if(causal)launch(std::true_type{});else launch(std::false_type{});}
    CUDA_CHECK(cudaGetLastError());
    return true;
}
bool strata_device_cache_guard_cuda(ggml_backend_cuda_context& context, ggml_tensor* tensor) {
    using namespace strata::deepseek_v4;
    if (!device_cache_guard(tensor)) return false;
    GGML_ASSERT(tensor->type == GGML_TYPE_F32 && tensor->src[0]->type == GGML_TYPE_F32 &&
                tensor->src[1]->type == GGML_TYPE_I32 && ggml_nelements(tensor->src[1]) == 1 &&
                ggml_is_contiguous(tensor) && ggml_is_contiguous(tensor->src[0]) && ggml_is_contiguous(tensor->src[1]) &&
                ggml_nelements(tensor) == ggml_nelements(tensor->src[0]) && ggml_nelements(tensor) <= INT_MAX);
    const int count = int(ggml_nelements(tensor));
    guard_kernel<<<(count + 255) / 256, 256, 0, context.stream()>>>(static_cast<const float*>(tensor->src[0]->data),
        static_cast<float*>(tensor->data), static_cast<const int32_t*>(tensor->src[1]->data), count);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

// Prepare the canonical pool before a device lookup can wait for host DMA.
// Expanding VMM after that lookup may synchronize the device and block its miss worker.
bool strata_device_cache_reserve_cuda(ggml_backend_t backend, size_t bytes, bool all_streams) {
    if (!backend || !ggml_backend_is_cuda(backend) || bytes < (1ull << 20) || bytes > (1ull << 30)) return false;
    auto* context = static_cast<ggml_backend_cuda_context*>(backend->context);
    if (!ggml_cuda_info().devices[context->device].vmm) return false;
    ggml_cuda_set_device(context->device);
    using namespace strata::deepseek_v4;
    const int previous_stream = context->curr_stream_no;
    const int count = all_streams ? GGML_CUDA_MAX_STREAMS : 1;
    cache_graph_phase(context, CacheGraphPhase::Prepare, 0);
    for (int stream = 0; stream < count; ++stream) {
        context->curr_stream_no = stream;
        cache_graph_stream(uint32_t(stream));
        context->stream();
        if (all_streams) context->cublas_handle();
        auto& pool = context->pool();
        size_t actual = 0;
        const size_t reserve = stream == 0 ? bytes : std::min(bytes, size_t(32ull << 20));
        void* storage = pool.alloc(reserve, &actual);
        pool.free(storage, actual);
    }
    context->curr_stream_no = previous_stream;
    cache_graph_phase(context, CacheGraphPhase::Idle, uint32_t(previous_stream));
    return true;
}

bool strata_device_cache_trace_cuda(ggml_backend_t backend, strata::deepseek_v4::CacheGraphTrace* trace) {
    if (!backend || !ggml_backend_is_cuda(backend)) return false;
    return strata::deepseek_v4::bind_cache_graph_trace(backend->context, trace);
}

namespace {
template<bool SampleIntervals,bool CausalPhases=false,bool PartialDown=false>
__global__ void hybrid_weight_kernel(const float* down,const float* weights,const int32_t* slots,
                                     strata::deepseek_v4::DeviceMissPacket* packet,float* output,int count,
                                     const uint32_t* state,int32_t* status,unsigned long long* counters,uint64_t epoch,int layer,uint64_t timeout,
                                     const strata::glm5_next::PartialDownBridge* partial_bridge,const float* partial_down){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count*4096)return;
    if(!strata::glm5_next::hybrid_join<SampleIntervals,CausalPhases>(state,status,counters,packet,epoch,layer,timeout)){output[i]=__int_as_float(0x7fc00000);return;}
    const int rank=i/4096;
    // The all-hit branch reads GPU tensors only; it never reads the mailbox.
    float value;
    if constexpr(PartialDown){
        if(!strata::glm5_next::partial_down_join_valid(partial_bridge,state,packet,status,epoch)){output[i]=__int_as_float(0x7fc00000);return;}
        if(!strata::glm5_next::partial_down_value(partial_bridge,state,unsigned(rank),unsigned(i%4096),partial_down,value))
            value=slots[rank]<0?reinterpret_cast<const volatile float*>(packet->contribution)[i]:down[i];
    }else value=slots[rank]<0?reinterpret_cast<const volatile float*>(packet->contribution)[i]:down[i];
    output[i]=value*weights[rank];
}
template<bool SampleIntervals,bool CausalPhases=false,bool PartialDown=false>
void launch_hybrid_weight(ggml_backend_cuda_context& context,ggml_tensor* tensor,int count,
                         uintptr_t packet,bool overlap,uint64_t epoch,uint64_t timeout) {
    hybrid_weight_kernel<SampleIntervals,CausalPhases,PartialDown><<<(count*4096+255)/256,256,0,context.stream()>>>(static_cast<const float*>(tensor->src[0]->data),
        static_cast<const float*>(tensor->src[1]->data),static_cast<const int32_t*>(tensor->src[2]->data),
        reinterpret_cast<DeviceMissPacket*>(packet),static_cast<float*>(tensor->data),count,
        overlap?static_cast<const uint32_t*>(tensor->src[3]->data):nullptr,
        overlap?static_cast<int32_t*>(tensor->src[4]->data):nullptr,
        overlap?static_cast<unsigned long long*>(tensor->src[5]->data):nullptr,epoch,tensor->op_params[2],timeout,
        tensor->src[7]?static_cast<const strata::glm5_next::PartialDownBridge*>(tensor->src[7]->data):nullptr,
        tensor->src[6]?static_cast<const float*>(tensor->src[6]->data):nullptr);
}
}
bool strata_device_cache_hybrid_weight_cuda(ggml_backend_cuda_context& context,ggml_tensor* tensor){
    using namespace strata::deepseek_v4;
    if(!device_cache_hybrid_weight(tensor))return false;
    GGML_ASSERT(device_cache_supported(tensor));uintptr_t packet=0;
    std::memcpy(&packet,tensor->op_params,sizeof(packet));GGML_ASSERT(packet);
    uint64_t epoch=0,timeout=0;std::memcpy(&epoch,tensor->op_params+6,sizeof(epoch));std::memcpy(&timeout,tensor->op_params+4,sizeof(timeout));
    const bool overlap=tensor->op_params[11]!=0;
    if(overlap)GGML_ASSERT(tensor->src[3]&&tensor->src[4]&&tensor->src[5]);
    const int count=int(ggml_nelements(tensor->src[2]));
    const bool sample=overlap&&(count==8||count==16)&&strata::glm5_next::critical_intervals_requested();
    const bool causal=overlap&&(count==8||count==16)&&strata::glm5_next::causal_phases_requested();
    if(tensor->src[7]){
        if(causal){
            if(sample)launch_hybrid_weight<true,true,true>(context,tensor,count,packet,overlap,epoch,timeout);
            else launch_hybrid_weight<false,true,true>(context,tensor,count,packet,overlap,epoch,timeout);
        }else if(sample)launch_hybrid_weight<true,false,true>(context,tensor,count,packet,overlap,epoch,timeout);
        else launch_hybrid_weight<false,false,true>(context,tensor,count,packet,overlap,epoch,timeout);
        CUDA_CHECK(cudaGetLastError());return true;
    }
    if(causal){
        GGML_ASSERT(strata::glm5_next::causal_phase_layer_valid(tensor->op_params[2])&&tensor->op_params[9]<46);
        if(sample)launch_hybrid_weight<true,true>(context,tensor,count,packet,overlap,epoch,timeout);
        else launch_hybrid_weight<false,true>(context,tensor,count,packet,overlap,epoch,timeout);
    }else if(sample) {
        // Model contract:45 primary blocks and at most NextN45. Layer63's
        // triplet is unused; reject a future layout before recording into it.
        GGML_ASSERT(tensor->op_params[2]>=3&&tensor->op_params[2]<46&&tensor->op_params[9]<46);
        launch_hybrid_weight<true>(context,tensor,count,packet,overlap,epoch,timeout);
    } else launch_hybrid_weight<false>(context,tensor,count,packet,overlap,epoch,timeout);
    CUDA_CHECK(cudaGetLastError());return true;
}

bool strata_glm53_probe_cold_gpu_copy(ggml_backend_t backend,uintptr_t metadata,int32_t* status,uint64_t epoch,int layer){
    using namespace strata::glm5_next;
    if(!backend||!ggml_backend_is_cuda(backend)||!metadata||metadata%alignof(ColdGpuLayer)||!status||
        reinterpret_cast<uintptr_t>(status)%alignof(int32_t)||!epoch||!cold_gpu_layer_valid(layer))return false;
    auto* context=static_cast<ggml_backend_cuda_context*>(backend->context);
    if(context->curr_stream_no!=0||cudaSetDevice(context->device)!=cudaSuccess)return false;
    cudaPointerAttributes meta_attributes{},status_attributes{};
    if(cudaPointerGetAttributes(&meta_attributes,reinterpret_cast<void*>(metadata))!=cudaSuccess||
        cudaPointerGetAttributes(&status_attributes,status)!=cudaSuccess){(void)cudaGetLastError();return false;}
    if(meta_attributes.type!=cudaMemoryTypeDevice||status_attributes.type!=cudaMemoryTypeDevice||
        meta_attributes.device!=context->device||status_attributes.device!=context->device)return false;
    cold_gpu_copy_rows<<<dim3(48,6),256,0,context->stream()>>>(reinterpret_cast<ColdGpuLayer*>(metadata),status,epoch,layer);
    return cudaGetLastError()==cudaSuccess;
}

// Explicit fixture only: same five-argument diagnostic ABI, fixed capacity4.
bool strata_glm53_probe_cold_gpu_copy4(ggml_backend_t backend,uintptr_t metadata,int32_t* status,uint64_t epoch,int layer){
    using namespace strata::glm5_next;
    if(!backend||!ggml_backend_is_cuda(backend)||!metadata||metadata%alignof(ColdGpuLayer)||!status||
        reinterpret_cast<uintptr_t>(status)%alignof(int32_t)||!epoch||!cold_gpu_layer_valid(layer))return false;
    auto* context=static_cast<ggml_backend_cuda_context*>(backend->context);
    if(context->curr_stream_no!=0||cudaSetDevice(context->device)!=cudaSuccess)return false;
    cudaPointerAttributes meta_attributes{},status_attributes{};
    if(cudaPointerGetAttributes(&meta_attributes,reinterpret_cast<void*>(metadata))!=cudaSuccess||
        cudaPointerGetAttributes(&status_attributes,status)!=cudaSuccess){(void)cudaGetLastError();return false;}
    if(meta_attributes.type!=cudaMemoryTypeDevice||status_attributes.type!=cudaMemoryTypeDevice||
        meta_attributes.device!=context->device||status_attributes.device!=context->device)return false;
    cold_gpu_copy_rows4<<<dim3(48,12),256,0,context->stream()>>>(reinterpret_cast<ColdGpuLayer*>(metadata),status,epoch,layer);
    return cudaGetLastError()==cudaSuccess;
}
