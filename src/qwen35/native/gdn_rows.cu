#include "strata/qwen35/native/gdn.hpp"
#include "strata/qwen35/native/ffn.hpp"
#include <cmath>

namespace strata::qwen35::native {
namespace {
template<int Keys,int Rows>
__global__ void interleaved_rows(GdnInputs x,GdnShape s,float *out,float *final_state,float scale){
    constexpr int Columns=Keys/32;
    const int lane=threadIdx.x%32,warp=threadIdx.x/32,head=blockIdx.y;
    const int first=int(blockIdx.x)*(4*Rows)+warp*Rows,query_head=head%s.query_heads;
    float state[Rows][Columns];
    #pragma unroll
    for(int r=0;r<Rows;++r){
        #pragma unroll
        for(int j=0;j<Columns;++j)
            state[r][j]=x.state[(size_t(head)*s.value_dim+first+r)*Keys+lane+32*j];
    }
    for(int t=0;t<s.tokens;++t){
        const size_t q_at=(size_t(t)*s.query_heads+query_head)*Keys;
        float decay=0,beta=0;
        if(!lane){decay=__expf(x.log_decay[size_t(t)*s.value_heads+head]);beta=x.beta[size_t(t)*s.value_heads+head];}
        decay=__shfl_sync(0xffffffff,decay,0);beta=__shfl_sync(0xffffffff,beta,0);
        float q[Columns],k[Columns],key_memory[Rows]{},query_memory[Rows]{},delta[Rows];
        #pragma unroll
        for(int j=0;j<Columns;++j){
            q[j]=x.q[q_at+lane+32*j];k[j]=x.k[q_at+lane+32*j];
            #pragma unroll
            for(int r=0;r<Rows;++r)key_memory[r]=__fmaf_rn(state[r][j],k[j],key_memory[r]);
        }
        #pragma unroll
        for(int d=16;d;d/=2){
            #pragma unroll
            for(int r=0;r<Rows;++r)key_memory[r]=__fadd_rn(key_memory[r],__shfl_xor_sync(0xffffffff,key_memory[r],d));
        }
        #pragma unroll
        for(int r=0;r<Rows;++r){
            const size_t at=(size_t(t)*s.value_heads+head)*s.value_dim+first+r;
            delta[r]=__fmul_rn(__fmaf_rn(-decay,key_memory[r],x.v[at]),beta);
        }
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
        if(!lane){
            #pragma unroll
            for(int r=0;r<Rows;++r)out[(size_t(t)*s.value_heads+head)*s.value_dim+first+r]=__fmul_rn(query_memory[r],scale);
        }
    }
    #pragma unroll
    for(int r=0;r<Rows;++r){
        #pragma unroll
        for(int j=0;j<Columns;++j)
            final_state[(size_t(head)*s.value_dim+first+r)*Keys+lane+32*j]=state[r][j];
    }
}
}
void gdn_recurrence_rows(GdnInputs x,GdnShape s,float *out,float *state,int rows,cudaStream_t stream){
    if(!x.q||!x.k||!x.v||!x.log_decay||!x.beta||!x.state||!out||!state||
       (s.key_dim!=64&&s.key_dim!=128)||s.value_dim<32||s.value_dim%32||s.value_dim>256||
       s.query_heads<1||s.value_heads<1||s.value_heads>256||s.value_heads%s.query_heads||s.tokens<1||s.tokens>8192||
       (rows!=1&&rows!=2&&rows!=4&&rows!=8))throw std::invalid_argument("invalid interleaved GDN geometry");
    const dim3 grid(s.value_dim/(4*rows),s.value_heads);const float scale=1.0f/std::sqrt(float(s.key_dim));
    if(s.key_dim==128){
        if(rows==1)interleaved_rows<128,1><<<grid,128,0,stream>>>(x,s,out,state,scale);
        else if(rows==2)interleaved_rows<128,2><<<grid,128,0,stream>>>(x,s,out,state,scale);
        else if(rows==4)interleaved_rows<128,4><<<grid,128,0,stream>>>(x,s,out,state,scale);
        else interleaved_rows<128,8><<<grid,128,0,stream>>>(x,s,out,state,scale);
    }else{
        if(rows==1)interleaved_rows<64,1><<<grid,128,0,stream>>>(x,s,out,state,scale);
        else if(rows==2)interleaved_rows<64,2><<<grid,128,0,stream>>>(x,s,out,state,scale);
        else if(rows==4)interleaved_rows<64,4><<<grid,128,0,stream>>>(x,s,out,state,scale);
        else interleaved_rows<64,8><<<grid,128,0,stream>>>(x,s,out,state,scale);
    }
    check_cuda(cudaGetLastError(),"Strata interleaved GDN rows");
}
}
