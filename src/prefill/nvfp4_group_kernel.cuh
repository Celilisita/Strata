#pragma once
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <cfloat>
namespace strata::prefill::mmq::nvfp4_detail {
__device__ __forceinline__ float scale_value(unsigned char b) {
    return __half2float(__nv_cvt_fp8_to_halfraw(b,__NV_E4M3));
}
// Same split-nibble GGUF->adjacent E2M1 permutation as the mature NVFP4 repack.
// Read four-byte words: raw block36 is only4-byte aligned, never assume uint2.
__device__ __forceinline__ unsigned compact4(unsigned v) {
    v&=0x0f0f0f0fu;
    return (v&15u)|((v>>4)&240u)|((v>>8)&3840u)|((v>>12)&61440u);
}
__device__ __forceinline__ unsigned weight_word(const uint8_t* block,int word) {
    const auto* p=reinterpret_cast<const unsigned*>(block+4+(word/2)*8);
    const int shift=(word&1)*4;
    return compact4(p[0]>>shift)|(compact4(p[1]>>shift)<<16);
}
__device__ __forceinline__ void mma(float (&c)[4],const unsigned (&a)[4],
                                    const unsigned (&b)[2],unsigned sa,unsigned sb) {
#if __CUDA_ARCH__ >= 1200
    asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};"
        : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
        : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),"r"(sa),"r"(sb));
#endif
}
__global__ void encode(const float* source,uint8_t* values,uint8_t* scales,float* globals,int columns,
                      const unsigned long long* active) {
    __shared__ float maxima[4],global;
    __shared__ unsigned invalid[4],bad;
    const int tid=int(threadIdx.x),warp=tid/32,lane=tid%32,row=int(blockIdx.x);
    if(active && !active[row]) { // CTA-uniform, before any source read/barrier.
        for(int c=tid;c<columns/2;c+=128)values[size_t(row)*(columns/2)+c]=0;
        for(int c=tid;c<columns/16;c+=128)scales[size_t(row)*(columns/16)+c]=0;
        if(!tid)globals[row]=1.f;
        return;
    }
    const auto* input=source+size_t(row)*columns;float maximum=0;unsigned nonfinite=0;
    for(int c=tid;c<columns;c+=128){const float x=input[c];maximum=fmaxf(maximum,fabsf(x));nonfinite|=!isfinite(x);}
    for(int d=16;d;d>>=1){maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffffu,maximum,d));
        nonfinite|=__shfl_xor_sync(0xffffffffu,nonfinite,d);}
    if(!lane){maxima[warp]=maximum;invalid[warp]=nonfinite;}__syncthreads();
    if(!tid){float m=0;bad=0;for(int w=0;w<4;++w){m=fmaxf(m,maxima[w]);bad|=invalid[w];}
        global=bad?CUDART_NAN_F:m==0?1.f:fmaxf(__fdiv_rn(m,2688.f),FLT_MIN);globals[row]=global;}
    __syncthreads();
    for(int pair=tid;pair<columns/2;pair+=128){
        const float x=input[2*pair],y=input[2*pair+1];
        // Divide by the row global first: finite extreme inputs cannot overflow an inverse.
        const float nx=bad?0.f:__fdiv_rn(x,global),ny=bad?0.f:__fdiv_rn(y,global);
        float a=fmaxf(fabsf(nx),fabsf(ny));
        for(int d=4;d;d>>=1)a=fmaxf(a,__shfl_xor_sync(0xffffffffu,a,d,8));
        const uint8_t sf=bad?127:__nv_cvt_float_to_fp8(__fdiv_rn(a,6.f),__NV_SATFINITE,__NV_E4M3);
        const float s=scale_value(sf),inv=s>0?__fdiv_rn(1.f,s):0.f;
        if(!(tid&7))scales[size_t(row)*(columns/16)+pair/8]=sf;
        values[size_t(row)*(columns/2)+pair]=__nv_cvt_float2_to_fp4x2(make_float2(nx*inv,ny*inv),__NV_E2M1,cudaRoundNearest);
    }
}
template<int Rows> __global__ __launch_bounds__(128,2)
void product(const NativeNvfp4Matrix* matrices,const int32_t* output_ids,float* output,int entries,
             int outputs,const NativeMmqTile* tiles,NativeNvfp4Input input) {
    const auto tile=tiles[blockIdx.y];const auto m=matrices[tile.matrix];
    const int first=m.begin+tile.tile*Rows,lane=int(threadIdx.x)%32,warp=int(threadIdx.x)/32;
    const int feature=int(blockIdx.x)*64+warp*16,part=int(blockIdx.z),k=input.columns;
    if(feature>=outputs||first>=m.end||!(part?m.up:m.gate))return; // Warp-uniform, no CTA barriers.
    const auto* weights=reinterpret_cast<const uint8_t*>(part?m.up:m.gate);
    const float weight_global=part?m.up_scale:m.gate_scale;
    float accum[Rows/8][4]{};
    for(int base=0;base<k;base+=64){
        const size_t block=size_t(feature+lane/4)*(k/64)+base/64;
        const auto* a0=weights+block*36;const auto* a1=a0+size_t(8)*(k/64)*36;
        const int local=lane%4;
        const unsigned a[4]={weight_word(a0,local),weight_word(a1,local),
                            weight_word(a0,local+4),weight_word(a1,local+4)};
        unsigned sa=0;if(local<2){const auto* sf=weights+
            (size_t(feature+lane/4+8*local)*(k/64)+base/64)*36;sa=*reinterpret_cast<const unsigned*>(sf);}
#pragma unroll
        for(int f=0;f<Rows/8;++f){
            const int at=first+f*8+lane/4;unsigned b[2]{},sb=0;
            if(at<m.end){const int source=input.source_ids[at];
                const auto* words=reinterpret_cast<const unsigned*>(input.values)+size_t(source)*(k/8)+base/8+local;
                b[0]=words[0];b[1]=words[4];
                if(!local)sb=*reinterpret_cast<const unsigned*>(input.scales+size_t(source)*(k/16)+base/16);}
            mma(accum[f],a,b,sa,sb);
        }
    }
#pragma unroll
    for(int f=0;f<Rows/8;++f)
#pragma unroll
        for(int i=0;i<4;++i){
            const int at=first+f*8+(lane%4)*2+(i&1),column=feature+lane/4+(i/2)*8;
            if(at<m.end&&column<outputs){const int source=input.source_ids[at],destination=output_ids[at];
                output[(size_t(part)*entries+destination)*outputs+column]=
                    __fmul_rn(accum[f][i],__fmul_rn(weight_global,input.globals[source]));}
        }
}
}
