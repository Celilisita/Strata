#include "fp8_p_fusion.hpp"
#include <pipeline_attention.hpp>
#include <strata/qwen35/native/ffn.hpp>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
namespace strata::qwen35::native::fp8_p_fusion_unit { namespace {
constexpr int D=256,Q=32,K=64,T=256,Pitch=272,SP=72,VP=68;
union Row {float score[SP];unsigned char p[SP*4];};
struct Shared {
    unsigned char query[Q*Pitch],kv[K*Pitch];Row row[Q];
    float qs[Q],ks[K],maximum[Q],denominator[Q],adjust[Q],seam_maximum[Q],seam_scale[Q],vs[K],pscale[Q],previous_pscale[Q];
};
__device__ float sum_warp(float x){for(int d=16;d;d/=2)x+=__shfl_xor_sync(0xffffffff,x,d);return x;}
__device__ float max_warp(float x){for(int d=16;d;d/=2)x=fmaxf(x,__shfl_xor_sync(0xffffffff,x,d));return x;}
__device__ float scale(float a){
    if(!(a>0.f)||!isfinite(a))return 1.f;
    const unsigned u=__float_as_uint(a*(1.f/256.f));return __uint_as_float(((u>>23)+bool(u&0x7fffff))<<23);
}
template<bool Diag>__device__ unsigned char encode(float x,Trace tr){
    if(!isfinite(x)){if constexpr(Diag)atomicAdd(tr.flags,1ULL);return 0x7f;}
    if constexpr(Diag)if(fabsf(x)>448.f)atomicAdd(tr.flags+1,1ULL);
    unsigned short v;asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0,%1,%1;":"=h"(v):"f"(x));return static_cast<unsigned char>(v);
}
__device__ unsigned pack(const unsigned char*p){return *reinterpret_cast<const unsigned*>(p);}
__device__ void mma(float (&c)[4],unsigned a0,unsigned a1,unsigned a2,unsigned a3,unsigned b0,unsigned b1){
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
__device__ int sample(int row,int head,int tokens){
    const int h=head==0?0:head==5?1:head==6?2:head==23?3:-1;if(h<0)return -1;
    const int r=row==0?0:row==31?1:row==32?2:row==63?3:row==tokens-1?4:-1;return r<0?-1:h*5+r;
}
struct Layout {size_t value_offset,k_scale_offset,v_scale_offset,total;int tiles;};
Layout layout(PrefillAttentionShape s){
    const auto align=[](size_t n){return(n+127)&~size_t(127);};const int tiles=(s.valid+63)/64;
    const size_t v=align(size_t(s.valid)*4*D),ks=align(v+size_t(tiles)*4*D*VP),vs=align(ks+size_t(s.valid)*4*4);
    return{v,ks,vs,align(vs+size_t(s.valid)*4*4),tiles};
}
struct PrepareShared {__half half[K*D];unsigned char code[D*VP];float scales[K];};
static_assert(sizeof(Shared)==36864&&sizeof(PrepareShared)==50432,"fixed Q32/K64 shared layout");
__global__ __launch_bounds__(256,1) void prepare_fp8(const unsigned char*key,const unsigned char*value,unsigned char*k8,unsigned char*v8,float*ks,float*vs,int valid,Trace tr){
    extern __shared__ __align__(32) unsigned char raw[];auto&t=*reinterpret_cast<PrepareShared*>(raw);
    const int first=int(blockIdx.x)*K,head=blockIdx.y,qr=threadIdx.x/4,qd=threadIdx.x%4;
    for(int val=0;val<2;++val){
        const auto*source=val?value:key;
        for(int i=threadIdx.x;i<K*D;i+=256){const int token=first+i/D,d=i%D;float x=0.f;
            if(token<valid){const auto*p=source+(size_t(token)*4+head)*272+(d/32)*34;
                x=__half2float(*reinterpret_cast<const __half*>(p))*float(static_cast<signed char>(p[2+d%32]));}
            t.half[i]=__float2half_rn(x);}
        if(val)for(int i=threadIdx.x;i<D*4;i+=256)t.code[(i/4)*VP+64+i%4]=0;
        __syncthreads();float maximum=0.f;
        for(int d=qd;d<D;d+=4)maximum=fmaxf(maximum,fabsf(__half2float(t.half[qr*D+d])));
        for(int d=2;d;d/=2)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,d,4));const float sf=scale(maximum);
        if(!qd){t.scales[qr]=sf;if(first+qr<valid){(val?vs:ks)[size_t(first+qr)*4+head]=sf;
            if(tr.flags)(val?tr.v_scales:tr.k_scales)[size_t(first+qr)*4+head]=sf;}}
        __syncthreads();
        for(int i=threadIdx.x;i<K*D;i+=256){const int token=first+i/D,d=i%D;const float x=__half2float(t.half[i])/t.scales[i/D];
            const auto code=tr.flags?encode<true>(x,tr):encode<false>(x,tr);t.code[val?d*VP+i/D:i]=code;
            if(tr.flags&&token<valid)(val?tr.v_codes:tr.k_codes)[(size_t(token)*4+head)*D+d]=code;}
        __syncthreads();
        if(val){auto*dst=v8+(size_t(blockIdx.x)*4+head)*D*VP;
            for(int i=threadIdx.x*16;i<D*VP;i+=256*16)*reinterpret_cast<uint4*>(dst+i)=*reinterpret_cast<const uint4*>(t.code+i);}
        else for(int i=threadIdx.x*16;i<K*D;i+=256*16){const int token=first+i/D,d=i%D;
            if(token<valid)*reinterpret_cast<uint4*>(k8+(size_t(token)*4+head)*D+d)=*reinterpret_cast<const uint4*>(t.code+i);}
        __syncthreads();
    }
}
__device__ void load_key(const unsigned char*x,const float*scales,Shared&t,int first,int head,int valid){
    #pragma unroll
    for(int i=threadIdx.x;i<K*D/16;i+=T){const int token=first+i/16,d=(i%16)*16;const bool live=token<valid;
        const auto*src=live?x+(size_t(token)*4+head)*D+d:x;const unsigned p=unsigned(__cvta_generic_to_shared(t.kv+(i/16)*Pitch+d));const int n=live?16:0;
        asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;"::"r"(p),"l"(src),"r"(n):"memory");}
    if(threadIdx.x<K)t.ks[threadIdx.x]=first+threadIdx.x<valid?scales[size_t(first+threadIdx.x)*4+head]:1.f;
    asm volatile("cp.async.commit_group;":::"memory");
}
__device__ void publish_value_scales(const float*scales,Shared&t,int first,int head,int valid){
    if(threadIdx.x<K)t.vs[threadIdx.x]=first+threadIdx.x<valid?scales[size_t(first+threadIdx.x)*4+head]:1.f;
}
__device__ void load_value_codes(const unsigned char*x,Shared&t,int first,int head){
    const auto*base=x+(size_t(first/K)*4+head)*D*VP;
    for(int i=threadIdx.x*16;i<D*VP;i+=T*16){const unsigned p=unsigned(__cvta_generic_to_shared(t.kv+i));
        asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(p),"l"(base+i):"memory");}
    asm volatile("cp.async.commit_group;":::"memory");
}
template<bool Diag>__global__ __launch_bounds__(T,2) void fp8_kernel(const float*query,const unsigned char*key,const unsigned char*value,const float*ks,const float*vs,float*out,PrefillAttentionShape s,bool skip,Trace tr){
    extern __shared__ __align__(32) unsigned char arena[];auto&t=*reinterpret_cast<Shared*>(arena);
    const int lane=threadIdx.x%32,warp=threadIdx.x/32,group=warp/4,localwarp=warp%4,gid=lane/4,tid=lane%4;
    const int head=blockIdx.x,kh=head/6,first_query=int(blockIdx.y)*Q,end=min(s.valid,s.query_start+min(s.tokens,first_query+Q));
    const int qr=threadIdx.x/8,qd=threadIdx.x%8,actual=first_query+qr;float amax=0.f;
    for(int d=qd;d<D;d+=8){const float x=actual<s.tokens?__half2float(__hmul(__float2half_rn(query[(size_t(actual)*24+head)*D+d]),__float2half_rn(s.scale))):0.f;amax=fmaxf(amax,fabsf(x));}
    for(int d=4;d;d/=2)amax=fmaxf(amax,__shfl_xor_sync(0xffffffff,amax,d,8));const float qscale=scale(amax);
    if(!qd){t.qs[qr]=qscale;if constexpr(Diag){const int at=sample(actual,head,s.tokens);if(at>=0)tr.q_scales[at]=qscale;}}
    for(int d=qd;d<D;d+=8){const float x=actual<s.tokens?__half2float(__hmul(__float2half_rn(query[(size_t(actual)*24+head)*D+d]),__float2half_rn(s.scale))):0.f;
        const auto code=encode<Diag>(x/qscale,tr);t.query[qr*Pitch+d]=code;if constexpr(Diag){const int at=sample(actual,head,s.tokens);if(at>=0)tr.q_codes[at*D+d]=code;}}
    if(threadIdx.x<Q){t.maximum[threadIdx.x]=-CUDART_INF_F;t.denominator[threadIdx.x]=0;t.seam_maximum[threadIdx.x]=-CUDART_INF_F;t.previous_pscale[threadIdx.x]=1.f;}
    float result[8][4]={};
    float history[8][4]={};
    if constexpr(Diag){
        for(int r=warp;r<Q;r+=8){const int at=sample(first_query+r,head,s.tokens);if(at>=0){
            for(int i=lane;i<s.valid;i+=32){const size_t j=size_t(at)*s.valid+i;tr.scores[j]=-CUDART_INF_F;tr.p_codes[j]=0;tr.p_half[j]=0;}
            for(int i=lane;i<(s.valid+63)/64;i+=32){const size_t j=size_t(at)*((s.valid+63)/64)+i;tr.maximum[j]=-CUDART_INF_F;tr.adjust[j]=0;tr.denominator[j]=0;tr.p_scales[j]=1.f;}
        }}
    }
    __syncthreads();load_key(key,ks,t,0,kh,s.valid);
    for(int first=0;first<end;first+=K){
        asm volatile("cp.async.wait_group 0;":::"memory");__syncthreads();
        // Official m16n8k32 byte layout: A rows gid,gid+8; K halves0,16; B column gid.
        {float dots[2][4]={};
            #pragma unroll
            for(int d=0;d<D;d+=32){const int r=group*16+gid,c=d+tid*4;const auto*p=t.query+r*Pitch+c;
                const unsigned a0=pack(p),a1=pack(p+8*Pitch),a2=pack(p+16),a3=pack(p+8*Pitch+16);
                #pragma unroll
                for(int j=0;j<2;++j){const auto*b=t.kv+(localwarp*16+j*8+gid)*Pitch+c;mma(dots[j],a0,a1,a2,a3,pack(b),pack(b+16));}}
            #pragma unroll
            for(int j=0;j<2;++j){
                #pragma unroll
                for(int i=0;i<4;++i){const int r=group*16+gid+(i>=2?8:0),c=localwarp*16+j*8+tid*2+(i&1);
                    t.row[r].score[c]=(dots[j][i]*t.qs[r])*t.ks[c];}}
        }
        // Publish only V scales before the existing QK barrier: they never alias K codes.
        publish_value_scales(vs,t,first,kh,s.valid);
        __syncthreads();
        // The single K/V bank may be overwritten only after every warp finished QK.
        load_value_codes(value,t,first,kh);
        for(int r=warp;r<Q;r+=8){const int tok0=first+lane,tok1=tok0+32,bound=s.query_start+first_query+r;
            const bool live=first_query+r<s.tokens,a0=live&&tok0<end&&tok0<=bound,a1=live&&tok1<end&&tok1<=bound;
            const float x0=a0?t.row[r].score[lane]:-CUDART_INF_F,x1=a1?t.row[r].score[lane+32]:-CUDART_INF_F;
            __syncwarp();const float next=fmaxf(t.maximum[r],max_warp(fmaxf(x0,x1))),adjust=isfinite(t.maximum[r])?__expf(t.maximum[r]-next):0.f;
            const float p0=a0?__expf(x0-next):0.f,p1=a1?__expf(x1-next):0.f;const __half h0=__float2half_rn(p0),h1=__float2half_rn(p1);
            const float total=sum_warp(p0+p1);
            if(!lane){t.denominator[r]=fmaf(t.denominator[r],adjust,total);t.maximum[r]=next;t.adjust[r]=adjust;}
            if constexpr(Diag){const int at=sample(first_query+r,head,s.tokens);if(at>=0){
                if(tok0<s.valid){const size_t i=size_t(at)*s.valid+tok0;tr.scores[i]=x0;tr.p_half[i]=__half_as_ushort(h0);}
                if(tok1<s.valid){const size_t i=size_t(at)*s.valid+tok1;tr.scores[i]=x1;tr.p_half[i]=__half_as_ushort(h1);}
                if(!lane){const size_t i=size_t(at)*((s.valid+63)/64)+first/64;tr.maximum[i]=next;tr.adjust[i]=adjust;tr.denominator[i]=t.denominator[r];}}}
            // Same RN HalfP values, kept in this row-owner warp instead of shared memory.
            const float w0=a0?__half2float(h0)*t.vs[lane]:0.f;
            const float w1=a1?__half2float(h1)*t.vs[lane+32]:0.f;
            __syncwarp();const float ps=scale(max_warp(fmaxf(fabsf(w0),fabsf(w1))));
            const auto b0=encode<Diag>(w0/ps,tr),b1=encode<Diag>(w1/ps,tr);t.row[r].p[lane]=b0;t.row[r].p[lane+32]=b1;if(!lane)t.pscale[r]=ps;
            if constexpr(Diag){const int at=sample(first_query+r,head,s.tokens);if(at>=0){
                if(first+lane<s.valid)tr.p_codes[size_t(at)*s.valid+first+lane]=b0;
                if(first+lane+32<s.valid)tr.p_codes[size_t(at)*s.valid+first+lane+32]=b1;
                if(!lane)tr.p_scales[size_t(at)*((s.valid+63)/64)+first/64]=ps;}}
        }
        __syncthreads();
        // PV folds each causal Vscale into P, with one independent row/tile Pscale.
        // Neither its amax nor the V encoding observes another token's future value.
        const bool rescale=!skip||__any_sync(0xffffffff,lane<16&&(t.adjust[group*16+lane]!=1.f||t.previous_pscale[group*16+lane]!=t.pscale[group*16+lane]));
        if(rescale){
            #pragma unroll
            for(int f=0;f<8;++f){
                #pragma unroll
                for(int i=0;i<4;++i){const int r=group*16+(f%2)*8+tid*2+(i&1);result[f][i]*=t.adjust[r]*(t.previous_pscale[r]/t.pscale[r]);}}}
        asm volatile("cp.async.wait_group 0;":::"memory");__syncthreads();
        #pragma unroll
        for(int k=0;k<K;k+=32){
            #pragma unroll
            for(int f=0;f<8;++f){const int d=localwarp*64+(f/2)*16+gid,c=k+tid*4;const auto*a=t.kv+d*VP+c;
                const auto*b=t.row[group*16+(f%2)*8+gid].p+c;
                mma(result[f],pack(a),pack(a+8*VP),pack(a+16),pack(a+8*VP+16),pack(b),pack(b+16));}}
        __syncthreads();if(first+K<end)load_key(key,ks,t,first+K,kh,s.valid);
        if(threadIdx.x<Q)t.previous_pscale[threadIdx.x]=t.pscale[threadIdx.x];
        if((first+K)%256==0||first+K>=end){
            if(threadIdx.x<Q){const int r=threadIdx.x;t.seam_scale[r]=isfinite(t.seam_maximum[r])?__expf(t.seam_maximum[r]-t.maximum[r]):0.f;t.seam_maximum[r]=t.maximum[r];}
            __syncthreads();
            #pragma unroll
            for(int f=0;f<8;++f){
                #pragma unroll
                for(int i=0;i<4;++i){const int r=group*16+(f%2)*8+tid*2+(i&1);
                    history[f][i]=fmaf(history[f][i],t.seam_scale[r],result[f][i]*t.pscale[r]);result[f][i]=0.f;}}
            __syncthreads();
        }
    }
    if(threadIdx.x<Q)t.denominator[threadIdx.x]=__frcp_rn(t.denominator[threadIdx.x]);__syncthreads();
    #pragma unroll
    for(int f=0;f<8;++f){
        #pragma unroll
        for(int i=0;i<4;++i){const int row=first_query+group*16+(f%2)*8+tid*2+(i&1),d=localwarp*64+(f/2)*16+gid+(i>=2?8:0);
            if(row<s.tokens)out[(size_t(row)*24+head)*D+d]=history[f][i]*t.denominator[row-first_query];}}
}
void require(bool b,const char*m){if(!b)throw std::invalid_argument(m);}
struct Range {uintptr_t start,end;};
Range range(const void*p,size_t n,size_t a){const auto x=reinterpret_cast<uintptr_t>(p);require(x&&x%a==0&&n<=UINTPTR_MAX-x,"FP8 PP span/alignment");return{x,x+n};}
bool overlap(Range a,Range b){return a.start<b.end&&b.start<a.end;}
}
void configure(){
    const auto caps=capabilities(0);require(caps.major==12&&caps.minor==0&&sizeof(Shared)<=caps.shared_per_block,"FP8 PP SM120/shared gate");
    check_cuda(cudaFuncSetAttribute(prepare_fp8,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(PrepareShared)),"FP8 PP prepare shared");
    check_cuda(cudaFuncSetAttribute(prepare_fp8,cudaFuncAttributePreferredSharedMemoryCarveout,100),"FP8 PP prepare carveout");
    check_cuda(cudaFuncSetAttribute(fp8_kernel<false>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared)),"FP8 PP shared");
    check_cuda(cudaFuncSetAttribute(fp8_kernel<true>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared)),"FP8 PP diagnostic shared");
    check_cuda(cudaFuncSetAttribute(fp8_kernel<false>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"FP8 PP carveout");
    check_cuda(cudaFuncSetAttribute(fp8_kernel<true>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"FP8 PP diagnostic carveout");
}
Resources resources(){
    cudaFuncAttributes a{},b{},c{};int x=0,y=0,z=0;
    check_cuda(cudaFuncGetAttributes(&a,fp8_kernel<false>),"FP8 PP attrs");check_cuda(cudaFuncGetAttributes(&b,fp8_kernel<true>),"FP8 PP trace attrs");
    check_cuda(cudaFuncGetAttributes(&c,prepare_fp8),"FP8 PP prepare attrs");
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&x,fp8_kernel<false>,T,sizeof(Shared)),"FP8 PP occupancy");
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&y,fp8_kernel<true>,T,sizeof(Shared)),"FP8 PP trace occupancy");
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&z,prepare_fp8,256,sizeof(PrepareShared)),"FP8 PP prepare occupancy");
    return{std::max({a.numRegs,b.numRegs,c.numRegs}),std::min(x,y),std::min({a.binaryVersion,b.binaryVersion,c.binaryVersion}),3,sizeof(Shared),a.localSizeBytes+b.localSizeBytes+c.localSizeBytes,x,y,z,a.numRegs,b.numRegs,c.numRegs};
}
size_t workspace_bytes(PrefillAttentionShape s){require(s.valid>0&&s.valid<=40960,"FP8 workspace valid extent");return layout(s).total;}

void attention(const float*q,const void*k,const void*v,float*out,void*work,size_t bytes,PrefillAttentionShape s,bool skip,cudaStream_t stream,Trace trace){
    require(q&&k&&v&&out&&work&&s.q8&&s.dim==D&&s.query_heads==24&&s.kv_heads==4&&(s.tokens==8192||s.tokens==5424)&&
        s.query_start>=0&&s.valid==s.query_start+s.tokens&&s.valid<=s.capacity&&s.capacity<=40960&&!s.graph_start&&s.scale>0&&std::isfinite(s.scale),"FP8 PP geometry");
    const auto l=layout(s);require(bytes>=l.total,"FP8 PP encoded borrowed lease");
    std::vector<Range> spans;const auto add=[&](const void*p,size_t n,size_t a){const auto r=range(p,n,a);for(auto old:spans)require(!overlap(r,old),"FP8 PP live alias");spans.push_back(r);};
    add(q,size_t(s.tokens)*24*D*4,4);add(k,size_t(s.capacity)*4*272,2);add(v,size_t(s.capacity)*4*272,2);add(out,size_t(s.tokens)*24*D*4,4);add(work,bytes,16);
    const bool diag=trace.flags;const size_t tiles=(s.valid+63)/64;
    const void*ptr[]={trace.q_codes,trace.k_codes,trace.v_codes,trace.p_codes,trace.q_scales,trace.k_scales,trace.v_scales,trace.p_scales,trace.scores,trace.maximum,trace.adjust,trace.denominator,trace.p_half};
    if(diag){const size_t extent[]={20*D,size_t(s.valid)*4*D,size_t(s.valid)*4*D,size_t(20)*s.valid,20*4,size_t(s.valid)*4*4,size_t(s.valid)*4*4,20*tiles*4,size_t(20)*s.valid*4,20*tiles*4,20*tiles*4,20*tiles*4,size_t(20)*s.valid*2};
        for(int i=0;i<13;++i)add(ptr[i],extent[i],i<4?1:i==12?2:4);add(trace.flags,4*8,8);}
    else for(auto p:ptr)require(!p,"FP8 PP partial diagnostic lease");
    auto*raw=static_cast<unsigned char*>(work);auto*key=raw;auto*value=raw+l.value_offset;
    auto*ks=reinterpret_cast<float*>(raw+l.k_scale_offset),*vs=reinterpret_cast<float*>(raw+l.v_scale_offset);
    prepare_fp8<<<dim3(l.tiles,4),256,sizeof(PrepareShared),stream>>>(static_cast<const unsigned char*>(k),static_cast<const unsigned char*>(v),key,value,ks,vs,s.valid,trace);
    check_cuda(cudaGetLastError(),"FP8 PP prepare result");const dim3 grid(24,(s.tokens+31)/32);
    if(diag)fp8_kernel<true><<<grid,T,sizeof(Shared),stream>>>(q,key,value,ks,vs,out,s,skip,trace);
    else fp8_kernel<false><<<grid,T,sizeof(Shared),stream>>>(q,key,value,ks,vs,out,s,skip,trace);
    check_cuda(cudaGetLastError(),"FP8 PP result");
}
}
