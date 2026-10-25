#include "regp.hpp"
#include <strata/qwen35/native/ffn.hpp>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace strata::qwen35::native {
namespace {
constexpr int Dim=256,Keys=64,Rows=64,Threads=256,Pitch=272;
struct __align__(32) RegisterTile {__half query[Rows*Pitch],kv[Keys*Pitch];};
static_assert(sizeof(RegisterTile)==69632,"one original Q tile and one aliased K/V tile");
struct Acc {float x[4];};
struct OperandA {unsigned x[4];};
struct OperandB {unsigned x[2];};
__device__ __forceinline__ unsigned address(const __half *p){return unsigned(__cvta_generic_to_shared(p));}
__device__ __forceinline__ OperandA load_a(const __half *p,int lane){
    // m16n8 A(row-major): the four 8x8 matrices are TL, BL, TR, BR.
    OperandA a;const unsigned at=address(p+(lane%16)*Pitch+(lane/16)*8);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        :"=r"(a.x[0]),"=r"(a.x[1]),"=r"(a.x[2]),"=r"(a.x[3]):"r"(at));return a;
}
__device__ __forceinline__ OperandB load_query(const __half *p,int lane){
    // Q[query8][dim16] is B[dim16][query8] in column-major order.
    OperandB b;const unsigned at=address(p+(lane%8)*Pitch+((lane/8)%2)*8);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
        :"=r"(b.x[0]),"=r"(b.x[1]):"r"(at));return b;
}
__device__ __forceinline__ OperandA load_value(const __half *p,int lane){
    // Transpose V[key16][feature16] into A[feature16][key16].
    OperandA a;const int matrix=lane/8;
    const unsigned at=address(p+(lane%8+(matrix/2)*8)*Pitch+(matrix%2)*8);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
        :"=r"(a.x[0]),"=r"(a.x[1]),"=r"(a.x[2]),"=r"(a.x[3]):"r"(at));return a;
}
__device__ __forceinline__ void mma(Acc &c,OperandA a,OperandB b){
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        :"+f"(c.x[0]),"+f"(c.x[1]),"+f"(c.x[2]),"+f"(c.x[3])
        :"r"(a.x[0]),"r"(a.x[1]),"r"(a.x[2]),"r"(a.x[3]),"r"(b.x[0]),"r"(b.x[1]));
}
__device__ __forceinline__ void load_half(const __half *x,__half *tile,int first,int head,int heads,int valid){
    #pragma unroll
    for(int i=threadIdx.x;i<Keys*Dim/8;i+=Threads){
        const int token=first+i/32,d=(i%32)*8;const bool active=token<valid;
        const unsigned destination=address(tile+(i/32)*Pitch+d);
        const auto *source=active?x+(size_t(token)*heads+head)*Dim+d:x;
        const int bytes=active?16:0;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"::"r"(destination),"l"(source),"r"(bytes):"memory");
    }
    asm volatile("cp.async.commit_group;":::"memory");
}
__device__ __forceinline__ float reduce_max(float x){
    #pragma unroll
    for(int d=4;d<=16;d*=2)x=fmaxf(x,__shfl_xor_sync(0xffffffff,x,d));return x;
}
__device__ __forceinline__ float reduce_sum(float x){
    #pragma unroll
    for(int d=4;d<=16;d*=2)x+=__shfl_xor_sync(0xffffffff,x,d);return x;
}
__device__ __forceinline__ unsigned half_pair(float a,float b){
    return unsigned(__half_as_ushort(__float2half_rn(a)))|(unsigned(__half_as_ushort(__float2half_rn(b)))<<16);
}
__device__ __forceinline__ OperandB probability(unsigned low,unsigned high,int lane){
    // Score C: key=group/+8, query=2*tid/+1. PV B needs
    // key=2*tid/+1/+8, query=group. This is a warp transpose, not a bitcast.
    const int group=lane/4,tid=lane%4,source=(2*tid)*4+group/2,shift=(group%2)*16;
    const unsigned l0=__shfl_sync(0xffffffff,low,source),l1=__shfl_sync(0xffffffff,low,source+4);
    const unsigned h0=__shfl_sync(0xffffffff,high,source),h1=__shfl_sync(0xffffffff,high,source+4);
    return {{((l0>>shift)&65535u)|(((l1>>shift)&65535u)<<16),
             ((h0>>shift)&65535u)|(((h1>>shift)&65535u)<<16)}};
}
__global__ __launch_bounds__(Threads,1) void regp_kernel(const float *query,const __half *key,const __half *value,
                                                        float *out,PrefillAttentionShape s){
    extern __shared__ __align__(32) unsigned char arena[];auto &tile=*reinterpret_cast<RegisterTile *>(arena);
    const int lane=threadIdx.x%32,warp=threadIdx.x/32,group=lane/4,tid=lane%4;
    const int head=blockIdx.x,first_query=int(blockIdx.y)*Rows,kh=head/6;
    const int end=min(s.valid,s.query_start+min(s.tokens,first_query+Rows));
    for(int i=threadIdx.x;i<Rows*Dim;i+=Threads){
        const int row=first_query+i/Dim;
        tile.query[(i/Dim)*Pitch+i%Dim]=row<s.tokens?
            __hmul(__float2half_rn(query[(size_t(row)*24+head)*Dim+i%Dim]),__float2half_rn(s.scale)):__float2half_rn(0.f);
    }
    Acc result[16],history[16];
    #pragma unroll
    for(int f=0;f<16;++f)result[f]=history[f]={{0,0,0,0}};
    float maximum[2]={-CUDART_INF_F,-CUDART_INF_F},denominator[2]={0,0},seam[2]={-CUDART_INF_F,-CUDART_INF_F};
    __syncthreads();load_half(key,tile.kv,0,kh,4,s.valid);
    for(int first=0;first<end;first+=Keys){
        asm volatile("cp.async.wait_group 0;":::"memory");__syncthreads();
        Acc score[4]={};
        #pragma unroll
        for(int d=0;d<Dim;d+=16){
            const OperandB b=load_query(tile.query+warp*8*Pitch+d,lane);
            #pragma unroll
            for(int k=0;k<4;++k)mma(score[k],load_a(tile.kv+k*16*Pitch+d,lane),b);
        }
        // Retire K readers before the original aliased V async copy starts.
        __syncthreads();load_half(value,tile.kv,first,kh,4,s.valid);
        float adjust[2];
        #pragma unroll
        for(int q=0;q<2;++q){
            const int query_row=first_query+warp*8+2*tid+q,boundary=s.query_start+query_row;
            float next=-CUDART_INF_F;
            #pragma unroll
            for(int k=0;k<4;++k){
                #pragma unroll
                for(int h=0;h<2;++h){const int key_row=first+k*16+group+h*8;
                    float &x=score[k].x[2*h+q];if(query_row>=s.tokens||key_row>=end||key_row>boundary)x=-CUDART_INF_F;next=fmaxf(next,x);}
            }
            next=fmaxf(maximum[q],reduce_max(next));adjust[q]=isfinite(maximum[q])?__expf(maximum[q]-next):0.f;
            float total=0;
            #pragma unroll
            for(int k=0;k<4;++k){
                #pragma unroll
                for(int h=0;h<2;++h){const int key_row=first+k*16+group+h*8;float &x=score[k].x[2*h+q];
                    x=query_row<s.tokens&&key_row<end&&key_row<=boundary?__expf(x-next):0.f;total+=x;}
            }
            denominator[q]=fmaf(denominator[q],adjust[q],reduce_sum(total));maximum[q]=next;
        }
        unsigned packed[4][2];
        #pragma unroll
        for(int k=0;k<4;++k){packed[k][0]=half_pair(score[k].x[0],score[k].x[1]);packed[k][1]=half_pair(score[k].x[2],score[k].x[3]);}
        const bool skip_policy=s.tokens==8192&&(s.query_start==8192||s.query_start==16384);
        const bool rescale=!skip_policy||__any_sync(0xffffffff,adjust[0]!=1.f||adjust[1]!=1.f);
        if(rescale){
            #pragma unroll
            for(int f=0;f<16;++f){result[f].x[0]*=adjust[0];result[f].x[2]*=adjust[0];result[f].x[1]*=adjust[1];result[f].x[3]*=adjust[1];}
        }
        asm volatile("cp.async.wait_group 0;":::"memory");__syncthreads();
        #pragma unroll
        for(int k=0;k<4;++k){
            const OperandB p=probability(packed[k][0],packed[k][1],lane);
            #pragma unroll
            for(int f=0;f<16;++f)mma(result[f],load_value(tile.kv+k*16*Pitch+f*16,lane),p);
        }
        __syncthreads();
        if(first+Keys<end)load_half(key,tile.kv,first+Keys,kh,4,s.valid);
        if((first+Keys)%256==0||first+Keys>=end){
            float scale[2];
            #pragma unroll
            for(int q=0;q<2;++q){scale[q]=isfinite(seam[q])?__expf(seam[q]-maximum[q]):0.f;seam[q]=maximum[q];}
            #pragma unroll
            for(int f=0;f<16;++f){
                #pragma unroll
                for(int i=0;i<4;++i){history[f].x[i]=fmaf(history[f].x[i],scale[i%2],result[f].x[i]);result[f].x[i]=0.f;}
            }
        }
    }
    const float inverse[2]={__frcp_rn(denominator[0]),__frcp_rn(denominator[1])};
    #pragma unroll
    for(int f=0;f<16;++f){
        #pragma unroll
        for(int i=0;i<4;++i){const int row=first_query+warp*8+2*tid+i%2,feature=f*16+group+(i/2)*8;
            if(row<s.tokens)out[(size_t(row)*24+head)*Dim+feature]=history[f].x[i]*inverse[i%2];}
    }
}
struct Range {uintptr_t first,last;};
Range range(const void *p,size_t bytes,size_t alignment){
    const auto first=reinterpret_cast<uintptr_t>(p);
    if(!p||first%alignment||bytes>std::numeric_limits<uintptr_t>::max()-first)throw std::invalid_argument("regp operand extent/alignment");
    return {first,first+bytes};
}
bool overlap(Range a,Range b){return a.first<b.last&&b.first<a.last;}
}
void configure_regp_attention(){
    int device=0;check_cuda(cudaGetDevice(&device),"regp device");const auto caps=capabilities(device);
    if(caps.major!=12||caps.minor!=0)throw std::invalid_argument("regp requires SM120");
    check_cuda(cudaFuncSetAttribute(regp_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(RegisterTile)),"regp shared optin");
    check_cuda(cudaFuncSetAttribute(regp_kernel,cudaFuncAttributePreferredSharedMemoryCarveout,100),"regp shared carveout");
}
PrefillOnlineResources regp_attention_resources(){
    cudaFuncAttributes a{};int blocks=0;check_cuda(cudaFuncGetAttributes(&a,regp_kernel),"regp resources");
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,regp_kernel,Threads,sizeof(RegisterTile)),"regp occupancy");
    return {a.numRegs,blocks,a.maxThreadsPerBlock,a.localSizeBytes,sizeof(RegisterTile)};
}
int regp_binary_version(){cudaFuncAttributes a{};check_cuda(cudaFuncGetAttributes(&a,regp_kernel),"regp binary");return a.binaryVersion;}
void regp_attention(const float *query,const void *key,const void *value,float *out,void *workspace,size_t bytes,
                    PrefillAttentionShape s,cudaStream_t stream){
    const bool shape=(s.tokens==8192&&(s.query_start==0||s.query_start==8192||s.query_start==16384))||
                     (s.tokens==5424&&s.query_start==24576);
    if(!shape||!s.q8||s.dim!=Dim||s.query_heads!=24||s.kv_heads!=4||s.valid!=s.query_start+s.tokens||
       s.valid>s.capacity||s.capacity>262144||s.graph_start||!(s.scale>0)||!std::isfinite(s.scale))throw std::invalid_argument("regp fixed causal geometry");
    const size_t needed=prefill_attention_workspace(s),vectors=size_t(s.tokens)*24*Dim,cache=size_t(s.valid)*4*272;
    if(bytes<needed||bytes>size_t(5120)*17408*4)throw std::invalid_argument("regp borrowed arena capacity");
    const Range live[]={range(query,vectors*4,4),range(key,cache,2),range(value,cache,2),range(out,vectors*4,16),range(workspace,bytes,16)};
    for(int i=0;i<5;++i)for(int j=i+1;j<5;++j)if(overlap(live[i],live[j]))throw std::invalid_argument("regp live alias");
    prefill_decode_kv(key,value,workspace,needed,s,stream);
    const auto *k=static_cast<const __half*>(workspace),*v=k+needed/(2*sizeof(__half));
    regp_kernel<<<dim3(24,(s.tokens+Rows-1)/Rows),Threads,sizeof(RegisterTile),stream>>>(query,k,v,out,s);
    check_cuda(cudaGetLastError(),"regp result");
}
}
