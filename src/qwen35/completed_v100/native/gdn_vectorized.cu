#include "strata/qwen35/completed_v100/native/gdn_vectorized.hpp"
#include "strata/qwen35/completed_v100/native/ffn.hpp"
#include <cmath>
#include <type_traits>

namespace strata::completed_v100::qwen35::native {
namespace {
template<int Keys>using Vector=std::conditional_t<Keys==128,float4,float2>;
template<int Keys>struct Record {Vector<Keys> q,k;float4 v;float2 parameters;};
template<int Keys>
__device__ __forceinline__ Record<Keys> load_record(GdnInputs x,GdnShape s,const float *work,int t,int head,int query,int first,int lane){
    Record<Keys> r{};
    if(t<s.tokens){
        const size_t values=size_t(s.tokens)*s.query_heads*Keys;
        const size_t at=(size_t(t)*s.query_heads+query)*32+lane;
        r.q=__ldg(reinterpret_cast<const Vector<Keys> *>(work)+at);
        r.k=__ldg(reinterpret_cast<const Vector<Keys> *>(work+values)+at);
        r.v=__ldg(reinterpret_cast<const float4 *>(x.v+(size_t(t)*s.value_heads+head)*s.value_dim+first));
        if(!lane)r.parameters=__ldg(reinterpret_cast<const float2 *>(work+2*values)+size_t(t)*s.value_heads+head);
    }
    return r;
}
template<int Keys>
__global__ void prepare_vectors(GdnInputs x,GdnShape s,float *work){
    constexpr int Columns=Keys/32;
    const size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const size_t vectors=size_t(s.tokens)*s.query_heads*32;
    const size_t values=size_t(s.tokens)*s.query_heads*Keys;
    if(index<vectors){
        const size_t base=(index/32)*Keys+index%32;
        Vector<Keys> q,k;
        if constexpr(Keys==128){
            q=make_float4(x.q[base],x.q[base+32],x.q[base+64],x.q[base+96]);
            k=make_float4(x.k[base],x.k[base+32],x.k[base+64],x.k[base+96]);
        }else{
            q=make_float2(x.q[base],x.q[base+32]);k=make_float2(x.k[base],x.k[base+32]);
        }
        reinterpret_cast<Vector<Keys> *>(work)[index]=q;
        reinterpret_cast<Vector<Keys> *>(work+values)[index]=k;
    }
    if(index<size_t(s.tokens)*s.value_heads){
        auto *parameters=reinterpret_cast<float2 *>(work+2*values);
        parameters[index]=make_float2(__expf(x.log_decay[index]),x.beta[index]);
    }
}
template<int Keys>
__global__ void vector_recurrence(GdnInputs x,GdnShape s,float *out,float *final_state,const float *work,float scale){
    constexpr int Columns=Keys/32,Rows=4;
    const int lane=threadIdx.x%32,warp=threadIdx.x/32,head=blockIdx.y;
    const int first=int(blockIdx.x)*16+warp*4,query=head%s.query_heads;
    float state[Rows][Columns];
    #pragma unroll
    for(int r=0;r<Rows;++r){
        #pragma unroll
        for(int j=0;j<Columns;++j)state[r][j]=x.state[(size_t(head)*s.value_dim+first+r)*Keys+lane+32*j];
    }
    auto current=load_record<Keys>(x,s,work,0,head,query,first,lane);
    auto ahead=load_record<Keys>(x,s,work,1,head,query,first,lane);
    for(int t=0;t<s.tokens;++t){
        // The following inputs are independent of the current FP32 state.
        const auto future=load_record<Keys>(x,s,work,t+2,head,query,first,lane);
        const auto qq=current.q,kk=current.k;
        float q[Columns],k[Columns];q[0]=qq.x;q[1]=qq.y;k[0]=kk.x;k[1]=kk.y;
        if constexpr(Keys==128){q[2]=qq.z;q[3]=qq.w;k[2]=kk.z;k[3]=kk.w;}
        float decay=current.parameters.x,beta=current.parameters.y;
        decay=__shfl_sync(0xffffffff,decay,0);beta=__shfl_sync(0xffffffff,beta,0);
        const size_t value_at=(size_t(t)*s.value_heads+head)*s.value_dim+first;
        const auto vv=current.v;
        const float v[Rows]={vv.x,vv.y,vv.z,vv.w};
        float key_memory[Rows]{},query_memory[Rows]{},delta[Rows];
        #pragma unroll
        for(int j=0;j<Columns;++j){
            #pragma unroll
            for(int r=0;r<Rows;++r)key_memory[r]=__fmaf_rn(state[r][j],k[j],key_memory[r]);
        }
        #pragma unroll
        for(int d=16;d;d/=2){
            #pragma unroll
            for(int r=0;r<Rows;++r)key_memory[r]=__fadd_rn(key_memory[r],__shfl_xor_sync(0xffffffff,key_memory[r],d));
        }
        #pragma unroll
        for(int r=0;r<Rows;++r)delta[r]=__fmul_rn(__fmaf_rn(-decay,key_memory[r],v[r]),beta);
        #pragma unroll
        for(int j=0;j<Columns;++j){
            #pragma unroll
            for(int r=0;r<Rows;++r){
                state[r][j]=__fmaf_rn(decay,state[r][j],__fmul_rn(k[j],delta[r]));
                query_memory[r]=__fmaf_rn(state[r][j],q[j],query_memory[r]);
            }
        }
        #pragma unroll
        for(int d=16;d;d/=2){
            #pragma unroll
            for(int r=0;r<Rows;++r)query_memory[r]=__fadd_rn(query_memory[r],__shfl_xor_sync(0xffffffff,query_memory[r],d));
        }
        if(!lane)*reinterpret_cast<float4 *>(out+value_at)=make_float4(
            __fmul_rn(query_memory[0],scale),__fmul_rn(query_memory[1],scale),
            __fmul_rn(query_memory[2],scale),__fmul_rn(query_memory[3],scale));
        current=ahead;ahead=future;
    }
    #pragma unroll
    for(int r=0;r<Rows;++r){
        #pragma unroll
        for(int j=0;j<Columns;++j)final_state[(size_t(head)*s.value_dim+first+r)*Keys+lane+32*j]=state[r][j];
    }
}
template<int Keys>
void launch(GdnInputs x,GdnShape s,float *out,float *state,float *work,cudaStream_t stream){
    const size_t qk=size_t(s.tokens)*s.query_heads*32,ab=size_t(s.tokens)*s.value_heads;
    prepare_vectors<Keys><<<unsigned(((qk>ab?qk:ab)+255)/256),256,0,stream>>>(x,s,work);
    check_cuda(cudaGetLastError(),"vectorGDN transpose/parameters");
    vector_recurrence<Keys><<<dim3(s.value_dim/16,s.value_heads),128,0,stream>>>(x,s,out,state,work,1.f/std::sqrt(float(Keys)));
    check_cuda(cudaGetLastError(),"vectorGDN FP32 state/output");
}
}
size_t gdn_vectorized_workspace_bytes(GdnShape s){
    if((s.key_dim!=64&&s.key_dim!=128)||s.value_dim<32||s.value_dim>256||s.value_dim%32||
       s.query_heads<1||s.value_heads<1||s.value_heads>256||s.value_heads%s.query_heads||s.tokens<1||s.tokens>8192)
        throw std::invalid_argument("vectorGDN geometry outside supported bounds");
    return size_t(s.tokens)*(2*size_t(s.query_heads)*s.key_dim+2*size_t(s.value_heads))*4;
}
void gdn_recurrence_vectorized(GdnInputs x,GdnShape s,float *out,float *state,void *work,size_t bytes,cudaStream_t stream){
    const size_t needed=gdn_vectorized_workspace_bytes(s);
    if(!x.q||!x.k||!x.v||!x.log_decay||!x.beta||!x.state||!out||!state||!work||bytes<needed||
       reinterpret_cast<uintptr_t>(work)%16||reinterpret_cast<uintptr_t>(x.v)%16||reinterpret_cast<uintptr_t>(out)%16||
       work==out||work==state||work==x.q||work==x.k||work==x.v||work==x.log_decay||work==x.beta||work==x.state)
        throw std::invalid_argument("vectorGDN borrowed buffers invalid");
    if(s.key_dim==128)launch<128>(x,s,out,state,static_cast<float *>(work),stream);
    else launch<64>(x,s,out,state,static_cast<float *>(work),stream);
}
}
