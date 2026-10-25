#include "t1_tc.hpp"
#include <strata/qwen35/native/ffn.hpp>
#include <mma.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
namespace strata::qwen35::native {namespace {
namespace wmma=nvcuda::wmma;
constexpr int D=256,Q=6,Rows=16,Keys=64,Pitch=272,PP=80,Threads=128,Parts=T1TcParts;
struct Shared {
    union {__half value[Keys*Pitch];int key[Keys*64];float tile[Rows*D];};
    int query[Q*64],physical[Keys],bad_query[Q],bad_payload;
    float delta[Q*8],key_delta[Keys*8],score[Q*Keys],maximum[Q],denominator[Q],history[Q];
    __align__(32) __half probability[2*Rows*PP];
};
__device__ __forceinline__ float wsum(float x){for(int d=16;d;d/=2)x=__fadd_rn(x,__shfl_xor_sync(0xffffffff,x,d));return x;}
__device__ __forceinline__ float wmax(float x){for(int d=16;d;d/=2)x=fmaxf(x,__shfl_xor_sync(0xffffffff,x,d));return x;}
__device__ __forceinline__ int logical_valid(T1TcShape s){return s.positions?*s.positions-s.shift+1:s.cursor?s.cursor->position+1:s.valid;}
// PTX ISA9.7.16.5.10 m16n8k32: A rows g,g+8 and columns4t+[0..3],[16..19]; B columns g.
__device__ __forceinline__ void dot32(int &d0,int &d1,int &d2,int &d3,unsigned a0,unsigned a1,unsigned a2,unsigned a3,unsigned b0,unsigned b1){
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32.satfinite {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
        :"=r"(d0),"=r"(d1),"=r"(d2),"=r"(d3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(0));
}
__global__ void segments(const float *q,const unsigned char *key,const unsigned char *value,RamKvView ram,float *partial,T1TcShape s,T1TcDiagnostic diagnostic){
    __shared__ __align__(32) Shared a;
    const int lane=threadIdx.x%32,warp=threadIdx.x/32,kh=blockIdx.x,head=kh*Q,valid=logical_valid(s);
    if(valid<1||valid>s.capacity){for(int row=0;row<Q;++row){float *dst=partial+(size_t(head+row)*Parts+blockIdx.y)*258;
        for(int d=threadIdx.x;d<D;d+=Threads)dst[d]=CUDART_NAN_F;if(!threadIdx.x){dst[D]=0;dst[D+1]=1;}}return;}
    if(threadIdx.x<Q){a.bad_query[threadIdx.x]=0;a.maximum[threadIdx.x]=-CUDART_INF_F;a.denominator[threadIdx.x]=0;}
    if(!threadIdx.x)a.bad_payload=0;__syncthreads();
    // Exact old block32 query codec: four serial F32 values, eight-worker max tree, RN scale, roundf.
    if(threadIdx.x<64)for(int row=0;row<Q;++row){const int block=threadIdx.x/8,worker=threadIdx.x%8;float x[4],largest=0;int bad=0;
        #pragma unroll
        for(int j=0;j<4;++j){x[j]=__fmul_rn(q[(head+row)*D+block*32+worker*4+j],1.f/16.f);bad|=!isfinite(x[j]);largest=fmaxf(largest,fabsf(x[j]));}
        for(int step=4;step;step/=2){largest=fmaxf(largest,__shfl_xor_sync(0xffffffff,largest,step,8));bad|=__shfl_xor_sync(0xffffffff,bad,step,8);}
        const float delta=__fmul_rn(largest,1.f/127.f);unsigned packed=0;
        #pragma unroll
        for(int j=0;j<4;++j){const int code=delta==0||!isfinite(x[j])?0:int(roundf(__fdividef(x[j],delta)));packed|=(unsigned(code)&255)<<(8*j);}
        a.query[row*64+block*8+worker]=int(packed);if(!worker){a.delta[row*8+block]=delta;atomicOr(a.bad_query+row,bad);}
        if(!blockIdx.y&&diagnostic.codes)diagnostic.codes[(head+row)*64+block*8+worker]=int(packed);
        if(!blockIdx.y&&diagnostic.delta&&!worker)diagnostic.delta[(head+row)*8+block]=delta;
    }
    __syncthreads();
    float carry[Q*2]={};
    const int extent=s.selected?min((valid+RamPage-1)/RamPage,ram.selected_slots)*RamPage:valid;
    for(int first=int(blockIdx.y)*Keys;first<extent;first+=Parts*Keys){
        if(threadIdx.x<Keys){const int token=first+threadIdx.x;int physical=-1;
            if(token<extent){if(!s.selected)physical=token;
                else {const int rank=token/RamPage,inpage=token%RamPage;
                    const int logical=ram.sorted_ids[size_t(kh)*ram.pages+rank]*RamPage+inpage;
                    const int slot=ram.selected_slot[size_t(kh)*ram.slots+rank];
                    if(logical<0||logical>=ram.pages*RamPage||(logical<valid&&(slot<0||slot>=ram.slots)))atomicExch(&a.bad_payload,1);
                    else if(logical<valid)physical=slot*RamPage+inpage;
                }}a.physical[threadIdx.x]=physical;}
        __syncthreads();
        for(int i=threadIdx.x;i<Keys*64;i+=Threads){const int row=i/64,word=i%64,block=word/8,physical=a.physical[row];unsigned packed=0;float delta=0;
            if(physical>=0){const auto *p=key+(size_t(physical)*4+kh)*272+block*34;memcpy(&packed,p+2+(word%8)*4,4);delta=__half2float(*reinterpret_cast<const __half*>(p));}
            a.key[i]=int(packed);if(!(word%8)){a.key_delta[row*8+block]=delta;if(!isfinite(delta))atomicExch(&a.bad_payload,1);}}
        __syncthreads();
        // Each 32-dimensional block has its own Q/K scale; never combine integer accumulators across blocks.
        for(int tile=warp;tile<Keys/8;tile+=4){const int row=lane/4,t=lane%4,col=tile*8+t*2;float r0=0,r1=0;
            #pragma unroll
            for(int block=0;block<8;++block){const unsigned a0=row<Q?unsigned(a.query[row*64+block*8+t]):0,a2=row<Q?unsigned(a.query[row*64+block*8+t+4]):0;
                const unsigned b0=unsigned(a.key[(tile*8+row)*64+block*8+t]),b1=unsigned(a.key[(tile*8+row)*64+block*8+t+4]);int d0,d1,d2,d3;
                dot32(d0,d1,d2,d3,a0,0,a2,0,b0,b1);
                if(row<Q){r0=__fmaf_rn(a.key_delta[col*8+block],__fmul_rn(float(d0),a.delta[row*8+block]),r0);
                    r1=__fmaf_rn(a.key_delta[(col+1)*8+block],__fmul_rn(float(d1),a.delta[row*8+block]),r1);}}
            if(row<Q){a.score[row*Keys+col]=a.physical[col]>=0?r0:-CUDART_INF_F;a.score[row*Keys+col+1]=a.physical[col+1]>=0?r1:-CUDART_INF_F;}}
        __syncthreads();
        for(int row=warp;row<Q;row+=4){const int i=row*Keys+lane;const float fresh=__fadd_rn(wmax(fmaxf(a.score[i],a.score[i+32])),3.f*0.6931f);
            const float next=fmaxf(a.maximum[row],fresh),history=isinf(next)?1.f:__expf(__fsub_rn(a.maximum[row],next));
            const float p0=a.physical[lane]>=0?__expf(__fsub_rn(a.score[i],next)):0.f,p1=a.physical[lane+32]>=0?__expf(__fsub_rn(a.score[i+32],next)):0.f;
            a.score[i]=p0;a.score[i+32]=p1;const float total=wsum(__fadd_rn(p0,p1));
            if(!lane){a.maximum[row]=next;a.history[row]=history;a.denominator[row]=__fmaf_rn(a.denominator[row],history,total);}}
        __syncthreads();
        for(int i=threadIdx.x;i<Rows*Keys;i+=Threads){const int row=i/Keys,col=i%Keys;const float p=row<Q?a.score[i]:0.f;const __half high=__float2half_rn(p);
            a.probability[row*PP+col]=high;a.probability[Rows*PP+row*PP+col]=__float2half_rn(__fsub_rn(p,__half2float(high)));}
        // K staging is now dead. Decode original Q8 V only to shared FP16, with no global cache/layout conversion.
        for(int i=threadIdx.x;i<Keys*D/8;i+=Threads){const int row=i/32,block=(i%32)/4,offset=(i%4)*8,physical=a.physical[row];__align__(16) __half decoded[8];
            #pragma unroll
            for(int j=0;j<8;++j){float v=0;if(physical>=0){const auto *p=value+(size_t(physical)*4+kh)*272+block*34;v=__fmul_rn(__half2float(*reinterpret_cast<const __half*>(p)),float(reinterpret_cast<const signed char*>(p+2)[offset+j]));}
                decoded[j]=__float2half_rn(v);if(!isfinite(__half2float(decoded[j])))atomicExch(&a.bad_payload,1);}
            *reinterpret_cast<uint4*>(a.value+row*Pitch+block*32+offset)=*reinterpret_cast<const uint4*>(decoded);}
        __syncthreads();
        wmma::fragment<wmma::accumulator,16,16,16,float> result[4];
        #pragma unroll
        for(int f=0;f<4;++f)wmma::fill_fragment(result[f],0.f);
        #pragma unroll
        for(int k=0;k<Keys;k+=16){
            #pragma unroll
            for(int p=0;p<2;++p){wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> b;wmma::load_matrix_sync(b,a.probability+p*Rows*PP+k,PP);
                #pragma unroll
                for(int f=0;f<4;++f){wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::col_major> v;wmma::load_matrix_sync(v,a.value+k*Pitch+(warp*4+f)*16,Pitch);wmma::mma_sync(result[f],v,b,result[f]);}}}
        __syncthreads();
        #pragma unroll
        for(int f=0;f<4;++f)wmma::store_matrix_sync(a.tile+(warp*4+f)*16,result[f],D,wmma::mem_col_major);
        __syncthreads();
        // Fixed64-key seam: only six live rows persist, as ordinary FP32 carry.
        #pragma unroll
        for(int row=0;row<Q;++row){
            #pragma unroll
            for(int d=0;d<2;++d)carry[row*2+d]=__fmaf_rn(a.history[row],carry[row*2+d],a.tile[row*D+threadIdx.x+d*Threads]);}
        __syncthreads();
    }
    #pragma unroll
    for(int row=0;row<Q;++row){const bool bad=a.bad_query[row]||a.bad_payload;float *dst=partial+(size_t(head+row)*Parts+blockIdx.y)*258;
        #pragma unroll
        for(int d=0;d<2;++d)dst[threadIdx.x+d*Threads]=bad?CUDART_NAN_F:carry[row*2+d];
        if(!threadIdx.x){dst[D]=bad?0.f:a.maximum[row];dst[D+1]=bad?1.f:a.denominator[row];}}
}
__global__ void merge(const float *partial,float *out){const int head=blockIdx.x,d=threadIdx.x;__shared__ float maximum;
    if(!d){float m=-CUDART_INF_F;for(int p=0;p<Parts;++p)m=fmaxf(m,partial[(size_t(head)*Parts+p)*258+256]);maximum=m;}__syncthreads();
    float numerator=0,denominator=0;for(int p=0;p<Parts;++p){const size_t i=(size_t(head)*Parts+p)*258;const float scale=__expf(__fsub_rn(partial[i+256],maximum));
        numerator=__fmaf_rn(scale,partial[i+d],numerator);denominator=__fmaf_rn(scale,partial[i+257],denominator);}out[size_t(head)*D+d]=__fdividef(numerator,denominator);}
struct Range {uintptr_t first,last;};
Range span(const void *p,size_t bytes,size_t alignment){const auto a=reinterpret_cast<uintptr_t>(p);if(!p||a%alignment||bytes>std::numeric_limits<uintptr_t>::max()-a)throw std::invalid_argument("T1TC buffer range/alignment");return {a,a+bytes};}
bool overlap(Range a,Range b){return a.first<b.last&&b.first<a.last;}
}
std::array<T1TcResource,2> t1_tc_resources(){std::array<T1TcResource,2> r{};const void *kernels[]={reinterpret_cast<void*>(segments),reinterpret_cast<void*>(merge)};
    for(int i=0;i<2;++i){cudaFuncAttributes a{};int blocks=0;check_cuda(cudaFuncGetAttributes(&a,kernels[i]),"T1TC attributes");check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernels[i],i?256:Threads,0),"T1TC occupancy");r[i]={a.numRegs,blocks,a.binaryVersion,a.sharedSizeBytes,a.localSizeBytes};}return r;}
void t1_tc_configure(){int device=0;check_cuda(cudaGetDevice(&device),"T1TC device");const auto c=capabilities(device);if(c.major!=12||c.minor!=0)throw std::invalid_argument("T1TC SM120 only");
    for(const auto r:t1_tc_resources())if(r.binary_version!=120||r.blocks_per_sm<2||r.local_bytes)throw std::runtime_error("T1TC zero-local2CTA resource rejection");}
bool t1_tc_enabled(){static const bool enabled=[](){const char *raw=std::getenv("STRATA_DENSE_T1_INT8_TC");const std::string mode=raw?raw:"0";if(mode!="0"&&mode!="1")throw std::invalid_argument("T1TC flag must be0or1");if(mode=="0")return false;t1_tc_configure();return true;}();return enabled;}
void t1_tc_validate(const float *q,const void *k,const void *v,RamKvView ram,float *out,void *work,size_t bytes,T1TcShape s,T1TcDiagnostic diag){
    if(s.capacity<1||s.capacity>262144||s.valid<1||s.valid>s.capacity||bytes<T1TcWorkspace||(!s.selected&&s.positions)||(s.selected&&(!s.positions||s.cursor)))throw std::invalid_argument("T1TC fixed geometry/extent");
    const size_t rows=s.selected?size_t(ram.resident):size_t(s.capacity);if(s.selected&&(ram.heads!=4||ram.selected_tokens!=8192||ram.selected_slots!=512||ram.pages<512||ram.slots<512||ram.slots>16384||ram.pages>16384||ram.resident!=ram.slots*RamPage||s.capacity>ram.pages*RamPage||k!=ram.key||v!=ram.value||!ram.sorted_ids||!ram.selected_slot))throw std::invalid_argument("T1TC selected lease geometry");
    const auto a=span(q,24*D*4,4),b=span(k,rows*4*272,2),c=span(v,rows*4*272,2),d=span(out,24*D*4,4),e=span(work,T1TcWorkspace,16);
    std::array<Range,8> ranges{};int count=0;for(auto r:{a,b,c,d,e})ranges[count++]=r;
    if(s.cursor)ranges[count++]=span(s.cursor,sizeof(DecodeCursor),4);if(s.positions)ranges[count++]=span(s.positions,4,4);
    if(s.selected){ranges[count++]=span(ram.sorted_ids,size_t(4)*ram.pages*4,4);ranges[count++]=span(ram.selected_slot,size_t(4)*ram.slots*4,4);}
    for(int i=0;i<count;++i)for(int j=i+1;j<count;++j)if(overlap(ranges[i],ranges[j]))throw std::invalid_argument("T1TC input/output/work lease overlap");
    if(bool(diag.codes)!=bool(diag.delta))throw std::invalid_argument("T1TC diagnostic plane pair");
    if(diag.codes){const auto f=span(diag.codes,24*64*4,4),g=span(diag.delta,24*8*4,4);if(overlap(f,g))throw std::invalid_argument("T1TC diagnostic alias");for(int i=0;i<count;++i)if(overlap(ranges[i],f)||overlap(ranges[i],g))throw std::invalid_argument("T1TC diagnostic input alias");}
}
void t1_tc_attention(const float *q,const void *k,const void *v,RamKvView ram,float *out,void *work,size_t bytes,T1TcShape s,cudaStream_t stream,T1TcDiagnostic diag){
    if(!t1_tc_enabled())throw std::invalid_argument("T1TC provider disabled by default");t1_tc_validate(q,k,v,ram,out,work,bytes,s,diag);
    segments<<<dim3(4,Parts),Threads,0,stream>>>(q,static_cast<const unsigned char*>(k),static_cast<const unsigned char*>(v),ram,static_cast<float*>(work),s,diag);check_cuda(cudaGetLastError(),"T1TC grouped segments");
    merge<<<24,256,0,stream>>>(static_cast<const float*>(work),out);check_cuda(cudaGetLastError(),"T1TC complete merge");
}
}
