#include "moe-weighted-reduction.cuh"
#include <cstring>
#include <type_traits>
#ifdef STRATA_DEEPSEEK_CACHE_LEASE
#include "device_cache_protocol.hpp"
#include "../glm5_next/hybrid_join.cuh"
#include "../glm5_next/partial_down_cuda.cuh"
#endif

template<bool SampleIntervals=false,bool CausalPhases=false,bool PartialDown=false>
static __global__ void moe_weighted_reduction_f32(const float * __restrict__ experts,
                                                  const float * __restrict__ expert_scale,
                                                  const float * __restrict__ weights,
                                                  float * __restrict__ dst,
                                                  const int64_t n_embd,
                                                  const int     n_expert_used,
                                                  const int32_t* slots,const volatile float* contribution
#ifdef STRATA_DEEPSEEK_CACHE_LEASE
                                                  ,const uint32_t* state,int32_t* status,unsigned long long* counters,
                                                  strata::deepseek_v4::DeviceMissPacket* packet,uint64_t epoch,int layer,uint64_t timeout,
                                                  const strata::glm5_next::PartialDownBridge* partial_bridge,const float* partial_down
#endif
                                                  ) {
    const int64_t token = blockIdx.x;
    const int64_t col   = (int64_t) blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= n_embd) {
        return;
    }
#ifdef STRATA_DEEPSEEK_CACHE_LEASE
    if(!strata::glm5_next::hybrid_join<SampleIntervals,CausalPhases>(state,status,counters,packet,epoch,layer,timeout)) {
        dst[token*n_embd+col]=__int_as_float(0x7fc00000);return;
    }
    if constexpr(PartialDown)if(!strata::glm5_next::partial_down_join_valid(partial_bridge,state,packet,status,epoch)){
        dst[token*n_embd+col]=__int_as_float(0x7fc00000);return;
    }
#endif

    const uint64_t first_row   = (uint64_t) token * n_expert_used;
    const float    first_scale = expert_scale != nullptr ? expert_scale[first_row] : 1.0f;
    float first_value;
#ifdef STRATA_DEEPSEEK_CACHE_LEASE
    if constexpr(PartialDown){
        if(!strata::glm5_next::partial_down_value(partial_bridge,state,unsigned(first_row),unsigned(col),partial_down,first_value))
            first_value=slots&&slots[first_row]<0?contribution[first_row*n_embd+col]:experts[first_row*n_embd+col];
    }else
#endif
        first_value = slots&&slots[first_row]<0?contribution[first_row*n_embd+col]:experts[first_row*n_embd+col];
    float sum=(first_value*first_scale)*weights[first_row];

    for (int expert = 1; expert < n_expert_used; ++expert) {
        const uint64_t row   = first_row + expert;
        const float   scale = expert_scale != nullptr ? expert_scale[row] : 1.0f;
        float value;
#ifdef STRATA_DEEPSEEK_CACHE_LEASE
        if constexpr(PartialDown){
            if(!strata::glm5_next::partial_down_value(partial_bridge,state,unsigned(row),unsigned(col),partial_down,value))
                value=slots&&slots[row]<0?contribution[row*n_embd+col]:experts[row*n_embd+col];
        }else
#endif
            value=slots&&slots[row]<0?contribution[row*n_embd+col]:experts[row*n_embd+col];
        sum += (value * scale) * weights[row];
    }
    dst[token * n_embd + col] = sum;
}

static void launch_moe_weighted_reduction(const float * experts,
                                          const float * expert_scale,
                                          const float * weights,
                                          float *       dst,
                                          int64_t       n_embd,
                                          int64_t       n_tokens,
                                          int           n_expert_used,
                                          cudaStream_t  stream,const int32_t* slots,const volatile float* contribution,const ggml_tensor* weighted) {
#ifdef STRATA_DEEPSEEK_CACHE_LEASE
    const uint32_t* state=nullptr;int32_t* status=nullptr;unsigned long long* counters=nullptr;
    strata::deepseek_v4::DeviceMissPacket* packet=nullptr;uint64_t epoch=0,timeout=0;int layer=0;
    const strata::glm5_next::PartialDownBridge* partial_bridge=nullptr;const float* partial_down=nullptr;
    bool sample=false,causal=false;
    if(strata::deepseek_v4::device_cache_hybrid_weight(weighted)&&weighted->op_params[11]) {
        GGML_ASSERT(n_embd==4096&&weighted->src[3]&&weighted->src[4]&&weighted->src[5]);
        uintptr_t address=0;std::memcpy(&address,weighted->op_params,sizeof(address));
        std::memcpy(&epoch,weighted->op_params+6,sizeof(epoch));std::memcpy(&timeout,weighted->op_params+4,sizeof(timeout));
        packet=reinterpret_cast<strata::deepseek_v4::DeviceMissPacket*>(address);layer=weighted->op_params[2];
        state=static_cast<const uint32_t*>(weighted->src[3]->data);status=static_cast<int32_t*>(weighted->src[4]->data);
        counters=static_cast<unsigned long long*>(weighted->src[5]->data);
        if(weighted->src[7]){GGML_ASSERT(strata::deepseek_v4::device_cache_supported(weighted));
            partial_bridge=static_cast<const strata::glm5_next::PartialDownBridge*>(weighted->src[7]->data);
            partial_down=static_cast<const float*>(weighted->src[6]->data);}
        sample=strata::glm5_next::critical_intervals_requested();
        causal=(n_tokens==1||n_tokens==2)&&n_expert_used==8&&strata::glm5_next::causal_phases_requested();
        if(causal)GGML_ASSERT(strata::glm5_next::causal_phase_layer_valid(layer)&&weighted->op_params[9]<46);
        if(sample)GGML_ASSERT((n_tokens==1||n_tokens==2||n_tokens==4)&&n_expert_used==8&&layer>=3&&layer<46&&weighted->op_params[9]<46);
    }
#else
    GGML_UNUSED(weighted);
#endif
    constexpr int threads = 256;
    const dim3 blocks(n_tokens, (n_embd + threads - 1) / threads, 1);
    auto launch=[&](auto tag,auto phase,auto partial) {
    moe_weighted_reduction_f32<decltype(tag)::value,decltype(phase)::value,decltype(partial)::value>
        <<<blocks, threads, 0, stream>>>(experts, expert_scale, weights, dst, n_embd, n_expert_used,slots,contribution
#ifdef STRATA_DEEPSEEK_CACHE_LEASE
            ,state,status,counters,packet,epoch,layer,timeout,partial_bridge,partial_down
#endif
        );
    };
#ifdef STRATA_DEEPSEEK_CACHE_LEASE
    if(partial_bridge){
        if(causal){if(sample)launch(std::true_type{},std::true_type{},std::true_type{});else launch(std::false_type{},std::true_type{},std::true_type{});}
        else if(sample)launch(std::true_type{},std::false_type{},std::true_type{});
        else launch(std::false_type{},std::false_type{},std::true_type{});return;
    }
    if(causal){if(sample)launch(std::true_type{},std::true_type{},std::false_type{});else launch(std::false_type{},std::true_type{},std::false_type{});return;}
    if(sample){launch(std::true_type{},std::false_type{},std::false_type{});return;}
#endif
    launch(std::false_type{},std::false_type{},std::false_type{});
}

void ggml_cuda_op_moe_weighted_reduction(ggml_backend_cuda_context & ctx,
                                         const ggml_tensor *         experts,
                                         const ggml_tensor *         expert_scale,
                                         const ggml_tensor *         weights,
                                         ggml_tensor *               dst,const ggml_tensor* weighted) {
    GGML_ASSERT(experts->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(expert_scale == nullptr || expert_scale->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(experts));
    GGML_ASSERT(ggml_is_contiguous(weights));
    GGML_ASSERT(expert_scale == nullptr || ggml_is_contiguous(expert_scale));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t n_embd        = experts->ne[0];
    const int64_t n_expert_used = experts->ne[1];
    const int64_t n_tokens      = experts->ne[2] * experts->ne[3];
    cudaStream_t  stream        = ctx.stream();

    const int32_t* slots=nullptr;const volatile float* contribution=nullptr;
#ifdef STRATA_DEEPSEEK_CACHE_LEASE
    if(strata::deepseek_v4::device_cache_hybrid_weight(weighted)){
        GGML_ASSERT((n_tokens==1||n_tokens==2||(n_tokens==4&&weighted->op_params[11]==1))&&n_expert_used==8&&n_embd==4096&&!expert_scale);
        uintptr_t packet=0;std::memcpy(&packet,weighted->op_params,sizeof(packet));GGML_ASSERT(packet);
        slots=static_cast<const int32_t*>(weighted->src[2]->data);
        contribution=reinterpret_cast<const strata::deepseek_v4::DeviceMissPacket*>(packet)->contribution;
    }
#else
    GGML_ASSERT(!weighted||!weighted->src[2]); // reject a contribution seam compiled out
    GGML_UNUSED(weighted);
#endif
    launch_moe_weighted_reduction((const float *) experts->data,
                                  expert_scale ? (const float *) expert_scale->data : nullptr,
                                  (const float *) weights->data,
                                  (float *) dst->data, n_embd, n_tokens, (int) n_expert_used, stream,slots,contribution,weighted);
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_moe_hybrid_ready(){
#ifdef STRATA_DEEPSEEK_CACHE_LEASE
    return true;
#else
    return false;
#endif
}
