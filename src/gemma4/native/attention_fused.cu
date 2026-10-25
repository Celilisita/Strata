#include "attention_fused.hpp"
#include "kernel_forward.h"
#include <cuda_fp16.h>
#include <math_constants.h>
#include <cstdint>
#include <type_traits>

namespace strata::gemma4::native {
namespace {
constexpr int Tokens=1280,Heads=32;
using Fused=::AttentionKernel<float,cutlass::arch::Sm70,true,64,64,512,false,true>;
static_assert(std::is_same<typename Fused::scalar_t,float>::value);
static_assert(std::is_same<typename Fused::output_t,float>::value);
static_assert(std::is_same<typename Fused::accum_t,float>::value);
static_assert(std::is_same<typename Fused::output_accum_t,float>::value);
static_assert(!Fused::kNeedsOutputAccumulatorBuffer);
static_assert(sizeof(typename Fused::SharedStorage)<=48u*1024,
    "Gemma float attention requires bounded SM70 shared storage");

void validate_shape(CacheView cache,int heads,int tokens){
    const bool local=cache.dim==256&&cache.heads==16&&cache.window==1024;
    const bool global=cache.dim==512&&cache.heads==4&&cache.window==0;
    if(heads!=Heads||tokens!=Tokens||cache.capacity<Tokens||(!local&&!global))
        throw std::invalid_argument("Gemma fused float attention cold1280 shape");
}
bool overlaps(const void* first,size_t first_bytes,const void* second,size_t second_bytes){
    const auto a=reinterpret_cast<std::uintptr_t>(first),b=reinterpret_cast<std::uintptr_t>(second);
    return a<=b?b-a<first_bytes:a-b<second_bytes;
}
__global__ void unpack_duplicate(CacheView cache,const int* position,float* key,float* value,size_t count){
    const size_t i=size_t(blockIdx.x)*256+threadIdx.x;
    if(i>=count)return;
    // This primitive is deliberately restricted to a cold request. Preserve
    // device position as an authority: an invalid lease propagates NaN.
    if(*position!=0){key[i]=CUDART_NAN_F;value[i]=CUDART_NAN_F;return;}
    const int column=int(i%cache.dim),head=int((i/cache.dim)%Heads);
    const int token=int(i/(Heads*cache.dim)),kv_head=head/(Heads/cache.heads);
    const size_t at=((size_t(token%cache.capacity)*cache.heads+kv_head)*
        (cache.dim/32)+column/32)*34;
    const auto* k=static_cast<const unsigned char*>(cache.key)+at;
    const auto* v=static_cast<const unsigned char*>(cache.value)+at;
    key[i]=__half2float(*reinterpret_cast<const half*>(k))*
        float(reinterpret_cast<const signed char*>(k+2)[column%32]);
    value[i]=__half2float(*reinterpret_cast<const half*>(v))*
        float(reinterpret_cast<const signed char*>(v+2)[column%32]);
}
__global__ void sliding_bias(float* bias,int window){
    const size_t i=size_t(blockIdx.x)*256+threadIdx.x;
    if(i>=size_t(Tokens)*Tokens)return;
    const int query=int(i/Tokens),key=int(i%Tokens);
    // Upper causal masking is supplied by the unchanged CUTLASS primitive.
    bias[i]=key>=query-window+1?0.f:-CUDART_INF_F;
}
}

size_t fused_float_attention_bytes(CacheView cache,int heads,int tokens){
    validate_shape(cache,heads,tokens);
    return (2*size_t(tokens)*heads*cache.dim+
        (cache.window?size_t(tokens)*tokens:0))*sizeof(float);
}
size_t fused_float_attention_shared_bytes(){return sizeof(typename Fused::SharedStorage);}

void fused_float_attention(const float* query,float* output,CacheView cache,const int* position,
    int past,int heads,int tokens,AttentionWorkspace workspace,cudaStream_t stream){
    const size_t bytes=fused_float_attention_bytes(cache,heads,tokens);
    if(!query||!output||!cache.key||!cache.value||!position||past!=0||
       !workspace.data||workspace.bytes<bytes||!stream)
        throw std::invalid_argument("Gemma fused float attention operand lease");
    const size_t count=size_t(tokens)*heads*cache.dim,output_bytes=count*sizeof(float);
    const size_t cache_bytes=size_t(cache.capacity)*cache.heads*(cache.dim/32)*34;
    if(overlaps(query,output_bytes,output,output_bytes)||
       overlaps(query,output_bytes,workspace.data,bytes)||
       overlaps(output,output_bytes,workspace.data,bytes)||
       overlaps(output,output_bytes,cache.key,cache_bytes)||
       overlaps(output,output_bytes,cache.value,cache_bytes)||
       overlaps(workspace.data,bytes,cache.key,cache_bytes)||
       overlaps(workspace.data,bytes,cache.value,cache_bytes)||
       overlaps(workspace.data,bytes,position,sizeof(int))||
       overlaps(output,output_bytes,position,sizeof(int)))
        throw std::invalid_argument("Gemma fused float attention arena alias");
    auto* key=static_cast<float*>(workspace.data);
    auto* value=key+count;
    auto* bias=cache.window?value+count:nullptr;
    typename Fused::Params params;
    params.query_ptr=const_cast<float*>(query);
    params.key_ptr=key;params.value_ptr=value;params.attn_bias_ptr=bias;
    params.output_ptr=output;params.output_accum_ptr=nullptr;
    params.logsumexp_ptr=nullptr;params.scale=1.f;
    params.head_dim=cache.dim;params.head_dim_value=cache.dim;
    params.num_queries=tokens;params.num_keys=tokens;params.num_keys_absolute=tokens;
    params.custom_mask_type=Fused::CausalFromTopLeft;
    params.q_strideM=heads*cache.dim;params.k_strideM=heads*cache.dim;
    params.v_strideM=heads*cache.dim;params.o_strideM=heads*cache.dim;
    params.q_strideH=cache.dim;params.k_strideH=cache.dim;params.v_strideH=cache.dim;
    params.q_strideB=int64_t(tokens)*heads*cache.dim;
    params.k_strideB=params.q_strideB;params.v_strideB=params.q_strideB;
    params.bias_strideM=tokens;params.bias_strideH=0;params.bias_strideB=0;
    params.num_batches=1;params.num_heads=heads;params.use_dropout=false;
    if(!Fused::check_supported(params))throw std::invalid_argument("Gemma CUTLASS float attention unsupported");
    unpack_duplicate<<<unsigned((count+255)/256),256,0,stream>>>(cache,position,key,value,count);
    if(bias)sliding_bias<<<unsigned((size_t(tokens)*tokens+255)/256),256,0,stream>>>(bias,cache.window);
    ::attention_kernel_batched_impl<Fused><<<params.getBlocksGrid(),params.getThreadsGrid(),
        sizeof(typename Fused::SharedStorage),stream>>>(params);
    primitive::check_cuda(cudaGetLastError(),"Gemma bounded CUTLASS F32 fused attention");
}
}
