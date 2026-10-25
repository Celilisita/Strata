#include "strata/qwen35/native/nvfp4.hpp"
#include "nvfp4_layout.cuh"
#include <cmath>
#include <stdexcept>

namespace strata::qwen35::native {
using namespace nvfp4_detail;
namespace {
void launched(){const auto e=cudaGetLastError();if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
__global__ void repack(const unsigned char *raw,unsigned char *q,unsigned char *sf,
                       int r,int k,unsigned *invalid){
    const size_t b=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(b>=size_t(r)*(k/64))return;
    const auto *src=raw+b*36;auto *dst=q+b*32;
    const int row=int(b/(k/64)),col=int(b%(k/64));
    for(int s=0;s<4;++s){
        const auto d=src[s];if(d>126&&invalid)atomicAdd(invalid,1u);
        sf[scale_index(row,col*4+s,k)]=d;
        const auto *p=src+4+s*8;
        for(int j=0;j<8;++j){
            const int a=2*j,c=a+1;
            const unsigned lo=(p[a%8]>>(a>=8?4:0))&15;
            const unsigned hi=(p[c%8]>>(c>=8?4:0))&15;
            dst[s*8+j]=static_cast<unsigned char>(lo|(hi<<4));
        }
    }
}
template<bool SwiGLU> __global__ void quantize(const float *input,unsigned char *q,unsigned char *sf,
                                               int t,int k,float global){
    // One warp handles four independent block16 groups. Consecutive lanes load
    // adjacent input pairs; XOR reductions stay inside their eight-lane group.
    const size_t pair=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(pair>=size_t(t)*k/2)return;
    const size_t element=2*pair;const int row=int(element/k),col=int(element%k);
    float x,y;
    if constexpr(SwiGLU){
        const auto *p=input+size_t(row)*2*k;
        x=silu_product(p[col],p[k+col]);y=silu_product(p[col+1],p[k+col+1]);
    }else{x=input[element];y=input[element+1];}
    float a=fmaxf(fabsf(x),fabsf(y));
    unsigned bad=unsigned(!isfinite(x)||!isfinite(y));
    for(int d=4;d;d>>=1){
        a=fmaxf(a,__shfl_xor_sync(0xffffffff,a,d,8));
        bad|=__shfl_xor_sync(0xffffffff,bad,d,8);
    }
    // E2M1 has no NaN. Preserve upstream invalids in the E4M3 group scale;
    // the native MMA must propagate it, as checked by the runtime gate.
    const unsigned char sb=bad?127:__nv_cvt_float_to_fp8(a/(6.f*global),__NV_SATFINITE,__NV_E4M3);
    const float s=scale_float(sb)*global;
    const float inv=s>0?1.f/s:0.f;
    if((threadIdx.x&7)==0)sf[scale_index(row,col/16,k)]=sb;
    q[pair]=__nv_cvt_float2_to_fp4x2(make_float2(x*inv,y*inv),__NV_E2M1,cudaRoundNearest);
}
__global__ void decode(const unsigned char *q,const unsigned char *sf,float *out,int r,int k,float global){
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=size_t(r)*k)return;
    const int row=int(i/k),col=int(i%k);
    out[i]=fp4_float((q[i/2]>>((i&1)*4))&15)*scale_float(sf[scale_index(row,col/16,k)])*global;
}
void geometry(int rows,int columns){
    if(rows<1||columns<64||columns%64)throw std::invalid_argument("invalid NVFP4 geometry");
}
void input_geometry(Nvfp4Input x,int t,int k,float global){
    geometry(t,k);
    if(!x.values||!x.scales||t>x.capacity_rows||k>x.capacity_columns||!(global>0)||!std::isfinite(global))
        throw std::invalid_argument("invalid NVFP4 input storage/global scale");
}
}
void nvfp4_repack(const void *raw,void *q,void *sf,int r,int k,unsigned *invalid,cudaStream_t stream){
    geometry(r,k);if(!raw||!q||!sf)throw std::invalid_argument("missing NVFP4 payload");
    repack<<<unsigned((size_t(r)*(k/64)+255)/256),256,0,stream>>>(
        static_cast<const unsigned char *>(raw),static_cast<unsigned char *>(q),static_cast<unsigned char *>(sf),r,k,invalid);
    launched();
}
void nvfp4_quantize(const float *p,Nvfp4Input x,int t,int k,float global,cudaStream_t stream){
    input_geometry(x,t,k,global);if(!p)throw std::invalid_argument("missing NVFP4 input");
    quantize<false><<<unsigned((size_t(t)*k/2+255)/256),256,0,stream>>>(p,
        static_cast<unsigned char *>(x.values),static_cast<unsigned char *>(x.scales),t,k,global);launched();
}
void nvfp4_swiglu_quantize(const float *p,Nvfp4Input x,int t,int k,float global,cudaStream_t stream){
    input_geometry(x,t,k,global);if(!p)throw std::invalid_argument("missing NVFP4 paired output");
    quantize<true><<<unsigned((size_t(t)*k/2+255)/256),256,0,stream>>>(p,
        static_cast<unsigned char *>(x.values),static_cast<unsigned char *>(x.scales),t,k,global);launched();
}
void nvfp4_decode(Nvfp4View w,float *out,cudaStream_t stream){
    geometry(w.rows,w.columns);
    decode<<<unsigned((size_t(w.rows)*w.columns+255)/256),256,0,stream>>>(
        static_cast<const unsigned char *>(w.values),static_cast<const unsigned char *>(w.scales),out,w.rows,w.columns,w.weight_scale);launched();
}
void nvfp4_decode_input(Nvfp4Input x,float *out,int t,int k,float global,cudaStream_t stream){
    input_geometry(x,t,k,global);
    nvfp4_decode({x.values,x.scales,t,k,global,global},out,stream);
}
}
