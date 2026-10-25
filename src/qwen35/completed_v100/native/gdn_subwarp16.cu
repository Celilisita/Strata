#include "strata/qwen35/completed_v100/native/gdn_subwarp16.hpp"
#include "strata/qwen35/completed_v100/native/gdn_vectorized.hpp"
#include "strata/qwen35/completed_v100/native/ffn.hpp"
#include <cmath>
namespace strata::completed_v100::qwen35::native {
namespace n=strata::completed_v100::qwen35::native;
namespace {
// Prepared Q/K layout is identical to the current vector recurrence.
__global__ void prepare(n::GdnInputs x,n::GdnShape s,float* work){
    const size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const size_t vectors=size_t(s.tokens)*s.query_heads*32,values=size_t(s.tokens)*s.query_heads*128;
    if(index<vectors){
        const size_t base=(index/32)*128+index%32;
        reinterpret_cast<float4*>(work)[index]=make_float4(x.q[base],x.q[base+32],x.q[base+64],x.q[base+96]);
        reinterpret_cast<float4*>(work+values)[index]=make_float4(x.k[base],x.k[base+32],x.k[base+64],x.k[base+96]);
    }
    if(index<size_t(s.tokens)*s.value_heads)reinterpret_cast<float2*>(work+2*values)[index]=make_float2(__expf(x.log_decay[index]),x.beta[index]);
}
struct Record {float4 q0,k0,q1,k1,v;float2 parameters;};
__device__ __forceinline__ Record load(n::GdnInputs x,n::GdnShape s,const float* work,int t,int head,int query,int first,int lane){
    Record r{};
    if(t<s.tokens){
        const size_t values=size_t(s.tokens)*s.query_heads*128,at=(size_t(t)*s.query_heads+query)*32+lane;
        r.q0=__ldg(reinterpret_cast<const float4*>(work)+at);r.q1=__ldg(reinterpret_cast<const float4*>(work)+at+16);
        r.k0=__ldg(reinterpret_cast<const float4*>(work+values)+at);r.k1=__ldg(reinterpret_cast<const float4*>(work+values)+at+16);
        r.v=__ldg(reinterpret_cast<const float4*>(x.v+(size_t(t)*s.value_heads+head)*s.value_dim+first));
        if(!lane)r.parameters=__ldg(reinterpret_cast<const float2*>(work+2*values)+size_t(t)*s.value_heads+head);
    }
    return r;
}
__global__ void recurrence(n::GdnInputs x,n::GdnShape s,float* output,float* final,const float* work,float scale){
    constexpr int Rows=4,Columns=8;
    const int lane=threadIdx.x%16,head=blockIdx.y,first=blockIdx.x*32+(threadIdx.x/16)*Rows,query=head%s.query_heads;
    float state[Rows][Columns];
    #pragma unroll
    for(int r=0;r<Rows;++r){
        #pragma unroll
        for(int j=0;j<Columns;++j){const int column=lane+(j%4)*32+(j/4)*16;
            state[r][j]=x.state[(size_t(head)*s.value_dim+first+r)*128+column];}
    }
    auto current=load(x,s,work,0,head,query,first,lane);
    for(int t=0;t<s.tokens;++t){
        // One lookahead bounds live Q/K registers while each lane owns eight keys.
        const auto ahead=load(x,s,work,t+1,head,query,first,lane);
        const float q[Columns]{current.q0.x,current.q0.y,current.q0.z,current.q0.w,current.q1.x,current.q1.y,current.q1.z,current.q1.w};
        const float k[Columns]{current.k0.x,current.k0.y,current.k0.z,current.k0.w,current.k1.x,current.k1.y,current.k1.z,current.k1.w};
        const float decay=__shfl_sync(0xffffffff,current.parameters.x,0,16),beta=__shfl_sync(0xffffffff,current.parameters.y,0,16);
        const float v[Rows]{current.v.x,current.v.y,current.v.z,current.v.w};float km[Rows]{},qm[Rows]{},delta[Rows];
        #pragma unroll
        for(int j=0;j<Columns;++j){
            #pragma unroll
            for(int r=0;r<Rows;++r)km[r]=__fmaf_rn(state[r][j],k[j],km[r]);
        }
        #pragma unroll
        for(int d=8;d;d/=2){
            #pragma unroll
            for(int r=0;r<Rows;++r)km[r]=__fadd_rn(km[r],__shfl_xor_sync(0xffffffff,km[r],d,16));
        }
        #pragma unroll
        for(int r=0;r<Rows;++r)delta[r]=__fmul_rn(__fmaf_rn(-decay,km[r],v[r]),beta);
        #pragma unroll
        for(int j=0;j<Columns;++j){
            #pragma unroll
            for(int r=0;r<Rows;++r){state[r][j]=__fmaf_rn(decay,state[r][j],__fmul_rn(k[j],delta[r]));qm[r]=__fmaf_rn(state[r][j],q[j],qm[r]);}
        }
        #pragma unroll
        for(int d=8;d;d/=2){
            #pragma unroll
            for(int r=0;r<Rows;++r)qm[r]=__fadd_rn(qm[r],__shfl_xor_sync(0xffffffff,qm[r],d,16));
        }
        if(!lane)*reinterpret_cast<float4*>(output+(size_t(t)*s.value_heads+head)*s.value_dim+first)=
            make_float4(__fmul_rn(qm[0],scale),__fmul_rn(qm[1],scale),__fmul_rn(qm[2],scale),__fmul_rn(qm[3],scale));
        current=ahead;
    }
    #pragma unroll
    for(int r=0;r<Rows;++r){
        #pragma unroll
        for(int j=0;j<Columns;++j){const int column=lane+(j%4)*32+(j/4)*16;
            final[(size_t(head)*s.value_dim+first+r)*128+column]=state[r][j];}
    }
}
}
size_t gdn_subwarp16_workspace_bytes(n::GdnShape s){return n::gdn_vectorized_workspace_bytes(s);}
void gdn_recurrence_subwarp16(n::GdnInputs x,n::GdnShape s,float* output,float* final,void* work,size_t bytes,cudaStream_t stream){
    // In-place state is legal: each CTA owns disjoint rows and reads its state
    // into registers before writing final state after the token loop.
    const size_t needed=gdn_subwarp16_workspace_bytes(s);
    if(!x.q||!x.k||!x.v||!x.beta||!x.log_decay||!x.state||!output||!final||!work||bytes<needed||
       reinterpret_cast<uintptr_t>(work)%16||reinterpret_cast<uintptr_t>(x.v)%16||reinterpret_cast<uintptr_t>(output)%16||
       output==x.state||output==final||work==output||work==final||work==x.q||work==x.k||work==x.v||work==x.state)
        throw std::invalid_argument("GDN subwarp16 input/state/work lease");
    if(s.key_dim!=128){n::gdn_recurrence_vectorized(x,s,output,final,work,bytes,stream);return;}
    const size_t qk=size_t(s.tokens)*s.query_heads*32,ab=size_t(s.tokens)*s.value_heads;
    prepare<<<unsigned(((qk>ab?qk:ab)+255)/256),256,0,stream>>>(x,s,static_cast<float*>(work));
    n::check_cuda(cudaGetLastError(),"GDN subwarp16 unchanged preparation");
    recurrence<<<dim3(s.value_dim/32,s.value_heads),128,0,stream>>>(x,s,output,final,static_cast<const float*>(work),1.f/std::sqrt(128.f));
    n::check_cuda(cudaGetLastError(),"GDN subwarp16 sixteen-lane FP32 recurrence");
}
}
