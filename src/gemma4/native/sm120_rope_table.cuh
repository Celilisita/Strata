#pragma once
#include "sm120_rope_table.hpp"
#include <cmath>
namespace strata::gemma4::native {
namespace rope_table_detail {
__device__ float sum(float x){for(int d=16;d;d/=2)x=__fadd_rn(x,__shfl_xor_sync(0xffffffff,x,d));return x;}
__global__ void prepare(float* table,int capacity,int tokens,float local_base,float global_base,
    const float* frequencies,const int* position,unsigned* invalid){
    const std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=std::size_t(tokens)*384)return;
    const int token=int(i/384),mode_pair=int(i%384),dim=mode_pair<128?256:512;
    const int pair=mode_pair<128?mode_pair:mode_pair-128,start=*position;
    const float base=mode_pair<128?local_base:global_base;
    const auto at=mode_pair<128?(std::size_t(token)*128+pair)*2:
        (std::size_t(capacity)*128+std::size_t(token)*256+pair)*2;
    if(start<0||start>262144-tokens){atomicExch(invalid,1u);table[at]=nanf("");table[at+1]=nanf("");return;}
    const int absolute=start+token;
    // Keep the exact original GPU expression and evaluation order. In
    // particular, do not premultiply a reciprocal frequency or use powf.
    double theta=double(absolute)*pow(double(base),-2.0*pair/dim);
    if(mode_pair>=128)theta/=double(frequencies[pair]);
    double sine,cosine;sincos(theta,&sine,&cosine);
    table[at]=float(sine);table[at+1]=float(cosine);
}
__global__ void norm(const float* x,float* y,const float* w,int dim,int heads,float epsilon,const float* table){
    __shared__ float partial[4],inverse;
    const int row=blockIdx.x,tid=threadIdx.x,lane=tid%32,warp=tid/32;const std::size_t first=std::size_t(row)*dim;
    float total=0;for(int k=tid;k<dim;k+=128)total=__fmaf_rn(x[first+k],x[first+k],total);
    total=sum(total);if(!lane)partial[warp]=total;__syncthreads();
    if(!tid)inverse=rsqrtf(((partial[0]+partial[1])+(partial[2]+partial[3]))/dim+epsilon);__syncthreads();
    for(int pair=tid;pair<dim/2;pair+=128){
        float a=x[first+pair]*inverse,b=x[first+pair+dim/2]*inverse;
        if(w){a*=w[pair];b*=w[pair+dim/2];}
        const auto at=(std::size_t(row/heads)*(dim/2)+pair)*2;
        const float sn=table[at],cs=table[at+1];
        y[first+pair]=__fmaf_rn(-b,sn,a*cs);y[first+pair+dim/2]=__fmaf_rn(a,sn,b*cs);
    }
}
}
void prepare_rope_table(float* table,int capacity,int tokens,float local_base,float global_base,
    const float* frequencies,const int* position,unsigned* invalid,cudaStream_t stream){
    rope_table_bytes(capacity);
    if(!table||!frequencies||!position||!invalid||!stream||tokens<1||tokens>capacity||
       local_base!=10000.f||global_base!=1000000.f)
        throw std::invalid_argument("Gemma PP RoPE table exact geometry/metadata");
    rope_table_detail::prepare<<<unsigned((std::size_t(tokens)*384+255)/256),256,0,stream>>>(
        table,capacity,tokens,local_base,global_base,frequencies,position,invalid);
    primitive::check_cuda(cudaGetLastError(),"Gemma PP per-position RoPE table");
}
void head_norm_rope_table(const float* x,float* y,const float* weight,int dim,int heads,int tokens,
    float epsilon,const float* table,int capacity,cudaStream_t stream){
    rope_table_bytes(capacity);
    if(!x||!y||!table||!stream||(dim!=256&&dim!=512)||heads<1||tokens<1||tokens>capacity||epsilon<=0)
        throw std::invalid_argument("Gemma cached weighted QK norm/RoPE geometry");
    const float* selected=dim==256?table:table+std::size_t(capacity)*128*2;
    rope_table_detail::norm<<<heads*tokens,128,0,stream>>>(x,y,weight,dim,heads,epsilon,selected);
    primitive::check_cuda(cudaGetLastError(),"Gemma PP table weighted QK norm/RoPE");
}
}
