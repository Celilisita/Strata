#include "sm120_global_flash.hpp"
// Exact dependency hashes are checked by the parent native-family CMake hook.
// Only this kernel template is instantiated; no ggml host launcher is called.
#include "fattn-mma-f16.cuh"
#if defined(GGML_CUDA_NO_FA)
#error "Gemma flash leaf cannot compile a NO_DEVICE_CODE substitute"
#endif
#if defined(__CUDA_ARCH__) && !defined(TURING_MMA_AVAILABLE)
#error "Gemma flash leaf requires the existing NVIDIA MMA implementation"
#endif
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include "sm120_global_flash_f32pv.cuh"
namespace strata::gemma4::native {
namespace p=strata::qwen35::native;
namespace {
constexpr int Dim=512,QueryHeads=32,KvHeads=4,QueryTile=8,Gqa=8;
constexpr int Threads=256,Warps=8,KeyTile=32;
// Ampere-style config selected by the existing primitive on SM120:
// swizzled K/V stride128half2, Q stride260half2, mask stride20half2.
constexpr std::size_t Shared=64*260*4+KeyTile*128*4+8*20*4;
static_assert(Shared==83584,"D512 existing leaf shared layout");
bool overlap(const void* a,std::size_t an,const void* b,std::size_t bn){
    const auto x=reinterpret_cast<std::uintptr_t>(a),y=reinterpret_cast<std::uintptr_t>(b);
    if(an>std::numeric_limits<std::uintptr_t>::max()-x||bn>std::numeric_limits<std::uintptr_t>::max()-y)return true;
    return x<y+bn&&y<x+an;
}
__device__ bool valid_position(const int* position,int tokens,int capacity,int keys,unsigned* invalid){
    const int first=*position;
    const bool valid=first>=0&&tokens<=capacity&&first<=capacity-tokens&&first<=keys-tokens;
    if(!valid)atomicOr(invalid,1u);return valid;
}
__global__ void stage_q8(CacheView cache,const int* position,int tokens,int keys,half* key,half* value,unsigned* invalid){
    const auto count=std::size_t(keys)*KvHeads*Dim;
    const bool valid=valid_position(position,tokens,cache.capacity,keys,invalid);
    const int active=valid?*position+tokens:0;
    for(std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=std::size_t(blockDim.x)*gridDim.x){
        const int d=int(i%Dim),row=int(i/Dim),k=row%keys,h=row/keys;
        half a=__float2half_rn(0.f),b=a;
        if(k<active){
            const auto block=(std::size_t(k)*KvHeads+h)*(Dim/32)+d/32;
            const auto* kr=static_cast<const unsigned char*>(cache.key)+block*34;
            const auto* vr=static_cast<const unsigned char*>(cache.value)+block*34;
            // Stored Q8 half scale times signed byte, then explicit F16 staging.
            const float kf=__half2float(*reinterpret_cast<const half*>(kr))*
                float(reinterpret_cast<const signed char*>(kr+2)[d%32]);
            const float vf=__half2float(*reinterpret_cast<const half*>(vr))*
                float(reinterpret_cast<const signed char*>(vr+2)[d%32]);
            a=__float2half_rn(kf);b=__float2half_rn(vf);
            if(!isfinite(kf)||!isfinite(vf)||!isfinite(__half2float(a))||!isfinite(__half2float(b)))atomicOr(invalid,2u);
        }
        // All padding is written even when the same arena held a longer history.
        key[i]=a;value[i]=b;
    }
}
__global__ void causal_mask(const int* position,int tokens,int capacity,int keys,half* mask,unsigned* invalid){
    const bool valid=valid_position(position,tokens,capacity,keys,invalid);
    const int first=valid?*position:0;const auto count=std::size_t(tokens)*keys;
    for(std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=std::size_t(blockDim.x)*gridDim.x){
        const int query=int(i/keys),key=int(i%keys);
        mask[i]=__float2half_rn(valid&&key<=first+query?0.f:-INFINITY);
    }
}
__global__ void check_output(const float* out,std::size_t count,unsigned* invalid){
    for(std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=std::size_t(blockDim.x)*gridDim.x){
        if(!isfinite(out[i]))atomicOr(invalid,4u);
    }
}
__global__ void propagate_invalid(float* out,std::size_t count,const unsigned* invalid){
    // Separate ordered kernel: every output observes the complete finite scan.
    // Propagate invalid leases into the model's existing checked-logits path.
    if(!*invalid)return;
    for(std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=std::size_t(blockDim.x)*gridDim.x)out[i]=nanf("");
}
auto leaf(){return flash_attn_ext_f16<Dim,Dim,QueryTile,Gqa,false,false,false>;}
}
void prepare_global_flash(cudaStream_t stream){
    // Attribute preparation only: no Q/K/V producer, attention, state or prefix.
    if(p::DeviceStorage::reserve_bytes()<std::size_t(4)*1024*1024*1024)
        throw std::runtime_error("Gemma global flash requires actual4GiB reserve");
    cudaStreamCaptureStatus capture{};
    p::check_cuda(cudaStreamIsCapturing(stream,&capture),"Gemma flash cold capture status");
    if(capture!=cudaStreamCaptureStatusNone)throw std::logic_error("Gemma flash prepare precedes capture");
    int device=0;cudaDeviceProp prop{};p::check_cuda(cudaGetDevice(&device),"Gemma flash device");
    p::check_cuda(cudaGetDeviceProperties(&prop,device),"Gemma flash device properties");
    if(prop.major!=12||prop.minor!=0||prop.sharedMemPerBlockOptin<Shared)
        throw std::runtime_error("Gemma global flash requires SM120 and83584B opt-in shared memory");
    if(f32pv_detail::requested()){f32pv_detail::prepare();return;}
    const auto config=ggml_cuda_fattn_mma_get_config_ampere(Dim,Dim,64);
    if(config.nthreads!=Threads||config.nbatch_fa!=KeyTile||config.nbatch_K2!=128||config.nbatch_V2!=128||config.Q_in_reg)
        throw std::runtime_error("Pinned Gemma flash primitive config changed");
    p::check_cuda(cudaFuncSetAttribute(reinterpret_cast<const void*>(leaf()),cudaFuncAttributeMaxDynamicSharedMemorySize,int(Shared)),"Gemma flash dynamic shared preparation");
    cudaFuncAttributes attributes{};p::check_cuda(cudaFuncGetAttributes(&attributes,reinterpret_cast<const void*>(leaf())),"Gemma flash actual kernel resources");
    std::fprintf(stderr,"GEMMA_GLOBAL_FLASH_PREP dtype=F16_PV_accum threads=%d dynamic_shared=%zu registers=%d local_bytes=%zu no_forward=1\n",
        Threads,Shared,attributes.numRegs,attributes.localSizeBytes);
}
void global_flash_attention(const float* q,float* out,CacheView cache,const int* position,
        int past_upper,int tokens,int maximum_tokens,AttentionWorkspace w,cudaStream_t stream){
    if(!q||!out||!position||!stream||tokens<=3||tokens>maximum_tokens||maximum_tokens>2048||cache.dim!=Dim||cache.heads!=KvHeads||cache.window!=0||
       !cache.key||!cache.value||cache.key==cache.value||past_upper<0||past_upper>cache.capacity-tokens)
        throw std::invalid_argument("Gemma global-only flash scope/actual capacity");
    const auto out_bytes=std::size_t(tokens)*QueryHeads*Dim*4;
    if(overlap(q,out_bytes,out,out_bytes)||overlap(q,out_bytes,w.data,w.bytes)||overlap(out,out_bytes,w.data,w.bytes))
        throw std::invalid_argument("Gemma flash Q/output/workspace alias");
    const auto b=global_flash_detail::bind(w,tokens,past_upper+tokens);
    const auto kv_bytes=std::size_t(cache.capacity)*KvHeads*(Dim/32)*34;
    if(overlap(w.data,w.bytes,cache.key,kv_bytes)||overlap(w.data,w.bytes,cache.value,kv_bytes)||
       overlap(cache.key,kv_bytes,cache.value,kv_bytes)||overlap(out,out_bytes,cache.key,kv_bytes)||overlap(out,out_bytes,cache.value,kv_bytes))
        throw std::invalid_argument("Gemma flash immutable Q8 cache alias");
    p::check_cuda(cudaMemsetAsync(b.invalid,0,sizeof(unsigned),stream),"Gemma flash fresh validity");
    stage_q8<<<256,Threads,0,stream>>>(cache,position,tokens,b.keys,static_cast<half*>(b.key),static_cast<half*>(b.value),b.invalid);
    p::check_cuda(cudaGetLastError(),"Gemma flash full Q8/F16 staging and zero pad");
    causal_mask<<<256,Threads,0,stream>>>(position,tokens,cache.capacity,b.keys,static_cast<half*>(b.mask),b.invalid);
    p::check_cuda(cudaGetLastError(),"Gemma flash fresh device-position causal mask");
    if(f32pv_detail::requested())f32pv_detail::launch(q,b.key,b.value,b.mask,out,tokens,b.keys,stream);
    else{
    // Exactly one whole output tile per CTA; no split-K seams, metadata or fixup.
    const int tiles=((tokens+QueryTile-1)/QueryTile)*KvHeads;
    cudaLaunchAttribute attribute{};attribute.id=cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed=1;
    cudaLaunchConfig_t launch{};launch.gridDim=dim3(tiles);launch.blockDim=dim3(32,Warps);
    launch.dynamicSmemBytes=Shared;launch.stream=stream;launch.attrs=&attribute;launch.numAttrs=1;
    const int q_token=QueryHeads*Dim*4,q_head=Dim*4,k_token=Dim*2,k_head=b.keys*k_token;
    const int mask_token=b.keys*2,mask_sequence=tokens*mask_token;
    p::check_cuda(cudaLaunchKernelEx(&launch,leaf(),
        reinterpret_cast<const char*>(q),static_cast<const char*>(b.key),static_cast<const char*>(b.value),
        static_cast<const char*>(b.mask),static_cast<const char*>(nullptr),static_cast<const int*>(nullptr),out,static_cast<float2*>(nullptr),
        1.f,0.f,1.f,1.f,std::uint32_t(32),0.f,
        Dim,init_fastdiv_values(tokens),QueryHeads,1,q_token,q_head,tokens*q_token,
        Dim,b.keys,KvHeads,1,k_token,k_head,std::int64_t(KvHeads)*k_head,
        k_token,k_head,std::int64_t(KvHeads)*k_head,
        tokens,1,1,mask_token,mask_sequence,std::int64_t(mask_sequence)),"Gemma existing D512 flash leaf");
    }
    check_output<<<256,Threads,0,stream>>>(out,out_bytes/4,b.invalid);
    p::check_cuda(cudaGetLastError(),"Gemma flash complete finite output scan");
    propagate_invalid<<<256,Threads,0,stream>>>(out,out_bytes/4,b.invalid);
    p::check_cuda(cudaGetLastError(),"Gemma flash invalid lease propagation");
}
}
