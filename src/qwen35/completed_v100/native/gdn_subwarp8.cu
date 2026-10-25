#include "gdn_subwarp8.hpp"
#include "strata/qwen35/completed_v100/native/gdn_subwarp16.hpp"
#include "strata/qwen35/completed_v100/native/gdn_vectorized.hpp"
#include "strata/qwen35/completed_v100/native/ffn.hpp"
#include <cmath>
namespace strata::completed_v100::qwen35::native {
namespace n=strata::completed_v100::qwen35::native;
namespace {
// Prepared Q/K layout is identical to the current vector recurrence.
__global__ void prepare_dual(n::GdnInputs x,n::GdnShape s,float* work){
    const size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const size_t vectors=size_t(s.tokens)*s.query_heads*32,values=size_t(s.tokens)*s.query_heads*128;
    if(index<vectors){
        const size_t base=(index/32)*128+index%32;
        reinterpret_cast<float4*>(work)[index]=make_float4(x.q[base],x.q[base+32],x.q[base+64],x.q[base+96]);
        reinterpret_cast<float4*>(work+values)[index]=make_float4(x.k[base],x.k[base+32],x.k[base+64],x.k[base+96]);
    }
    if(index<size_t(s.tokens)*s.value_heads)reinterpret_cast<float2*>(work+2*values)[index]=make_float2(__expf(x.log_decay[index]),x.beta[index]);
}
struct Record {float4 q[4],k[4];float2 v,parameters;};
__device__ __forceinline__ Record load(n::GdnInputs x,n::GdnShape s,const float* work,int t,int head,int query,int first,int lane){
    Record r{};
    if(t<s.tokens){
        const size_t values=size_t(s.tokens)*s.query_heads*128,at=(size_t(t)*s.query_heads+query)*32+lane;
        #pragma unroll
        for(int j=0;j<4;++j)r.q[j]=__ldg(reinterpret_cast<const float4*>(work)+at+8*j);
        #pragma unroll
        for(int j=0;j<4;++j)r.k[j]=__ldg(reinterpret_cast<const float4*>(work+values)+at+8*j);
        r.v=__ldg(reinterpret_cast<const float2*>(x.v+(size_t(t)*s.value_heads+head)*s.value_dim+first));
        if(!lane)r.parameters=__ldg(reinterpret_cast<const float2*>(work+2*values)+size_t(t)*s.value_heads+head);
    }
    return r;
}
__global__ void recurrence_dual(n::GdnInputs x,n::GdnShape s,float* output,float* final,const float* work,float scale){
    constexpr int Rows=2,Columns=16;
    const int lane=threadIdx.x%8,head=blockIdx.y,first=blockIdx.x*32+(threadIdx.x/8)*Rows,query=head%s.query_heads;
    float state[Rows][Columns];
    #pragma unroll
    for(int r=0;r<Rows;++r){
        #pragma unroll
        for(int j=0;j<Columns;++j){const int column=lane+(j%4)*32+(j/4)*8;
            state[r][j]=x.state[(size_t(head)*s.value_dim+first+r)*128+column];}
    }
    auto current=load(x,s,work,0,head,query,first,lane);
    for(int t=0;t<s.tokens;++t){
        // One lookahead bounds live Q/K registers while each lane owns eight keys.
        const auto ahead=load(x,s,work,t+1,head,query,first,lane);
        const float q[16]{current.q[0].x,current.q[0].y,current.q[0].z,current.q[0].w,current.q[1].x,current.q[1].y,current.q[1].z,current.q[1].w,
            current.q[2].x,current.q[2].y,current.q[2].z,current.q[2].w,current.q[3].x,current.q[3].y,current.q[3].z,current.q[3].w};
        const float k[16]{current.k[0].x,current.k[0].y,current.k[0].z,current.k[0].w,current.k[1].x,current.k[1].y,current.k[1].z,current.k[1].w,
            current.k[2].x,current.k[2].y,current.k[2].z,current.k[2].w,current.k[3].x,current.k[3].y,current.k[3].z,current.k[3].w};
        const float decay=__shfl_sync(0xffffffff,current.parameters.x,0,8),beta=__shfl_sync(0xffffffff,current.parameters.y,0,8);
        const float v[Rows]{current.v.x,current.v.y};float km[Rows]{},qm[Rows]{},a[Rows][4]{},b[Rows][4]{},delta[Rows];
        #pragma unroll
        for(int j=0;j<4;++j){
            #pragma unroll
            for(int r=0;r<Rows;++r){
                #pragma unroll
                for(int p=0;p<4;++p)a[r][p]=__fmaf_rn(state[r][j+4*p],k[j+4*p],a[r][p]);
            }
        }
        #pragma unroll
        for(int r=0;r<Rows;++r)km[r]=__fadd_rn(__fadd_rn(a[r][0],a[r][1]),__fadd_rn(a[r][2],a[r][3]));
        #pragma unroll
        for(int d=4;d;d/=2){
            #pragma unroll
            for(int r=0;r<Rows;++r)km[r]=__fadd_rn(km[r],__shfl_xor_sync(0xffffffff,km[r],d,8));
        }
        #pragma unroll
        for(int r=0;r<Rows;++r)delta[r]=__fmul_rn(__fmaf_rn(-decay,km[r],v[r]),beta);
        #pragma unroll
        for(int j=0;j<4;++j){
            #pragma unroll
            for(int r=0;r<Rows;++r){
                #pragma unroll
                for(int p=0;p<4;++p){
                    const int at=j+4*p;
                    state[r][at]=__fmaf_rn(decay,state[r][at],__fmul_rn(k[at],delta[r]));
                    b[r][p]=__fmaf_rn(state[r][at],q[at],b[r][p]);
                }
            }
        }
        #pragma unroll
        for(int r=0;r<Rows;++r)qm[r]=__fadd_rn(__fadd_rn(b[r][0],b[r][1]),__fadd_rn(b[r][2],b[r][3]));
        #pragma unroll
        for(int d=4;d;d/=2){
            #pragma unroll
            for(int r=0;r<Rows;++r)qm[r]=__fadd_rn(qm[r],__shfl_xor_sync(0xffffffff,qm[r],d,8));
        }
        if(!lane)*reinterpret_cast<float2*>(output+(size_t(t)*s.value_heads+head)*s.value_dim+first)=
            make_float2(__fmul_rn(qm[0],scale),__fmul_rn(qm[1],scale));
        current=ahead;
    }
    #pragma unroll
    for(int r=0;r<Rows;++r){
        #pragma unroll
        for(int j=0;j<Columns;++j){const int column=lane+(j%4)*32+(j/4)*8;
            final[(size_t(head)*s.value_dim+first+r)*128+column]=state[r][j];}
    }
}
}
void gdn_recurrence_subwarp8_dual(n::GdnInputs x,n::GdnShape s,float* output,float* final,void* work,size_t bytes,cudaStream_t stream){
    const size_t needed=n::gdn_subwarp16_workspace_bytes(s);
    if(!x.q||!x.k||!x.v||!x.beta||!x.log_decay||!x.state||!output||!final||!work||bytes<needed||
       reinterpret_cast<uintptr_t>(work)%16||reinterpret_cast<uintptr_t>(x.v)%8||reinterpret_cast<uintptr_t>(output)%8||
       output==x.state||output==final||work==output||work==final||work==x.q||work==x.k||work==x.v||work==x.state)
        throw std::invalid_argument("GDN subwarp8 two-row input/state/work lease");
    if(s.key_dim!=128){n::gdn_recurrence_vectorized(x,s,output,final,work,bytes,stream);return;}
    const size_t qk=size_t(s.tokens)*s.query_heads*32,ab=size_t(s.tokens)*s.value_heads;
    prepare_dual<<<unsigned(((qk>ab?qk:ab)+255)/256),256,0,stream>>>(x,s,static_cast<float*>(work));
    n::check_cuda(cudaGetLastError(),"GDN subwarp8 two-row unchanged preparation");
    recurrence_dual<<<dim3(s.value_dim/32,s.value_heads),128,0,stream>>>(x,s,output,final,static_cast<const float*>(work),1.f/std::sqrt(128.f));
    n::check_cuda(cudaGetLastError(),"GDN subwarp8 two-row eight-lane FP32 recurrence");
}
}
