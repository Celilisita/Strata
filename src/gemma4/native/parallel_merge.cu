#include "strata/gemma4/native/attention_parallel_merge.hpp"
#include <math_constants.h>
#include <limits>
#include <stdexcept>
namespace strata::gemma4::native::merge321 {
namespace {
constexpr int Threads=128,Warps=4,Tile=128;
size_t mul(size_t a,size_t b){if(b&&a>(std::numeric_limits<size_t>::max)()/b)throw std::invalid_argument("321 stride overflow");return a*b;}
size_t add(size_t a,size_t b){if(a>(std::numeric_limits<size_t>::max)()-b)throw std::invalid_argument("321 extent overflow");return a+b;}
void shape(Layout l){
    if(l.parts<1||l.parts>MaxParts||l.queries<1||l.queries>3||l.head_stride<mul(l.parts,Row)||
       l.query_stride<mul(Heads,l.head_stride)||l.output_query_stride<size_t(Heads)*Dim)
        throw std::invalid_argument("321 explicit parts/query/head stride");
}
bool span(const void* p,size_t n,size_t alignment){
    const auto x=reinterpret_cast<std::uintptr_t>(p);
    return p&&n&&x%alignment==0&&n<=(std::numeric_limits<std::uintptr_t>::max)()-x;
}
bool overlap(const void* a,size_t an,const void* b,size_t bn){
    const auto x=reinterpret_cast<std::uintptr_t>(a),y=reinterpret_cast<std::uintptr_t>(b);return x<=y?y-x<an:x-y<bn;
}
__device__ float sum(float x){for(int d=16;d;d/=2)x+=__shfl_xor_sync(0xffffffff,x,d);return x;}
__device__ float maximum(float x){for(int d=16;d;d/=2)x=fmaxf(x,__shfl_xor_sync(0xffffffff,x,d));return x;}
__global__ void merge(const float* partial,float* output,Layout l,State s,std::uint64_t epoch){
    __shared__ float weights[MaxParts],reduction[Warps],vectors[Warps*Tile],top,denominator;
    __shared__ unsigned invalid[Warps],bad_stats;
    const int tid=threadIdx.x,lane=tid%32,warp=tid/32,head=blockIdx.x,tile=blockIdx.y,query=blockIdx.z;
    const long long p=static_cast<long long>(*s.position)+s.query_offset+query;
    const size_t out=size_t(query)*l.output_query_stride+size_t(head)*Dim+tile*Tile+tid;
    if(p<0||p>=s.logical_capacity||(s.valid_end&&p>=*s.valid_end)||(s.invalid&&*s.invalid)||
       (s.device_epoch&&*s.device_epoch!=epoch)){output[out]=CUDART_NAN_F;return;}
    const float* source=partial+size_t(query)*l.query_stride+size_t(head)*l.head_stride;
    float value=-CUDART_INF_F;bool bad=false;
    for(int part=tid;part<l.parts;part+=Threads){const float z=source[size_t(part)*Row+1],m=source[size_t(part)*Row];
        bad|=!isfinite(z)||(z>0&&!isfinite(m));if(z>0)value=fmaxf(value,m);}
    const unsigned mask=__ballot_sync(0xffffffff,bad);value=maximum(value);
    if(!lane){invalid[warp]=mask;reduction[warp]=value;}__syncthreads();
    if(!warp){const float all=maximum(lane<Warps?reduction[lane]:-CUDART_INF_F);
        if(!lane){top=all;bad_stats=invalid[0]|invalid[1]|invalid[2]|invalid[3];}}
    __syncthreads();float total=0;
    for(int part=tid;part<l.parts;part+=Threads){const float z=source[size_t(part)*Row+1];
        const float weight=z==0?0:expf(source[size_t(part)*Row]-top);
        weights[part]=weight;total=fmaf(z,weight,total);}
    total=sum(total);if(!lane)reduction[warp]=total;__syncthreads();
    if(!warp){const float all=sum(lane<Warps?reduction[lane]:0.f);if(!lane)denominator=all;}
    __syncthreads();float accum[Tile/32]{};
    // Preserve174/317 zero-global-weight skip, including an underflowed part.
    // No read of its numerator when its global weight is exactly zero.
    for(int part=warp;part<l.parts;part+=Warps)if(weights[part]!=0){const float weight=weights[part];
        #pragma unroll
        for(int j=0;j<Tile/32;++j)accum[j]=fmaf(weight,source[size_t(part)*Row+2+tile*Tile+lane+j*32],accum[j]);}
    #pragma unroll
    for(int j=0;j<Tile/32;++j)vectors[warp*Tile+lane+j*32]=accum[j];
    __syncthreads();value=0;
    #pragma unroll
    for(int w=0;w<Warps;++w)value=__fadd_rn(value,vectors[w*Tile+tid]);
    output[out]=bad_stats||!(denominator>0)?CUDART_NAN_F:value/denominator;
}
}
Layout layout(int parts,int queries){Layout l{parts,queries,size_t(parts)*Row,size_t(Heads)*parts*Row,size_t(Heads)*Dim};shape(l);return l;}
size_t partial_bytes(Layout l){shape(l);return mul(add(add(mul(l.queries-1,l.query_stride),mul(Heads-1,l.head_stride)),mul(l.parts,Row)),sizeof(float));}
size_t output_bytes(Layout l){shape(l);return mul(add(mul(l.queries-1,l.output_query_stride),size_t(Heads)*Dim),sizeof(float));}
void merge_aos(float* output,Layout l,State s,Lease lease,Binding live){
    const size_t input_bytes=partial_bytes(l),out_bytes=output_bytes(l);const auto owned=lease.binding;
    if(!live.owner||!live.epoch||!live.stream||owned.owner!=live.owner||owned.epoch!=live.epoch||owned.stream!=live.stream||
       s.query_offset<0||s.query_offset>3-l.queries||s.logical_capacity<1||s.logical_capacity>262144||
       !span(s.position,4,4)||!span(output,out_bytes,4)||!span(lease.partial.data,lease.partial.bytes,4)||lease.partial.bytes<input_bytes)
        throw std::invalid_argument("321 merge owner/epoch/stream/state/extent");
    const void* state[]{s.position,s.valid_end,s.invalid,s.device_epoch};const size_t bytes[]{4,4,4,8};
    if(overlap(output,out_bytes,lease.partial.data,lease.partial.bytes))throw std::invalid_argument("321 output/partial alias");
    for(int i=0;i<4;++i)if(state[i]&&(!span(state[i],bytes[i],bytes[i])||overlap(output,out_bytes,state[i],bytes[i])||
        overlap(lease.partial.data,lease.partial.bytes,state[i],bytes[i])))throw std::invalid_argument("321 merge state alias/alignment");
    merge<<<dim3(Heads,Dim/Tile,l.queries),Threads,0,live.stream>>>(static_cast<const float*>(lease.partial.data),output,l,s,live.epoch);
    primitive::check_cuda(cudaGetLastError(),"321 explicit-parts parallel AoS merge");
}
}
