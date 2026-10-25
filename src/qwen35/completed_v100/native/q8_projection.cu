#include "strata/qwen35/completed_v100/native/q8_projection.hpp"
#include "strata/qwen35/completed_v100/native/ffn.hpp"
#include <cuda_fp16.h>

namespace strata::completed_v100::qwen35::native {
namespace {
template<int Tile>
__global__ void project_pair(const unsigned char* a, const unsigned char* b,
                             const float* x, float* ya, float* yb,
                             int rows, int columns, int tokens) {
    const int row = blockIdx.x, first = blockIdx.y * Tile;
    float sa[Tile]{}, sb[Tile]{};
    const int blocks = columns / 32;
    for(int c = threadIdx.x; c < columns; c += blockDim.x) {
        const size_t offset = (size_t(row)*blocks + c/32)*34;
        const float wa = __half2float(*reinterpret_cast<const half*>(a+offset)) *
                         float(reinterpret_cast<const signed char*>(a+offset+2)[c%32]);
        const float wb = __half2float(*reinterpret_cast<const half*>(b+offset)) *
                         float(reinterpret_cast<const signed char*>(b+offset+2)[c%32]);
        #pragma unroll
        for(int t=0;t<Tile;++t) if(first+t<tokens) {
            const float value=x[size_t(first+t)*columns+c];
            sa[t]=fmaf(wa,value,sa[t]); sb[t]=fmaf(wb,value,sb[t]);
        }
    }
    __shared__ float partial[2*Tile][4];
    #pragma unroll
    for(int t=0;t<Tile;++t) {
        #pragma unroll
        for(int d=16;d;d/=2) {
            sa[t]+=__shfl_down_sync(0xffffffff,sa[t],d);
            sb[t]+=__shfl_down_sync(0xffffffff,sb[t],d);
        }
        if((threadIdx.x&31)==0) {
            partial[2*t][threadIdx.x/32]=sa[t];
            partial[2*t+1][threadIdx.x/32]=sb[t];
        }
    }
    __syncthreads();
    if(threadIdx.x==0) {
        #pragma unroll
        for(int t=0;t<Tile;++t) if(first+t<tokens) {
            ya[size_t(first+t)*rows+row]=(partial[2*t][0]+partial[2*t][1])+
                                       (partial[2*t][2]+partial[2*t][3]);
            yb[size_t(first+t)*rows+row]=(partial[2*t+1][0]+partial[2*t+1][1])+
                                       (partial[2*t+1][2]+partial[2*t+1][3]);
        }
    }
}
}
void q8_projection_pair(Matrix a,Matrix b,const float* x,float* ya,float* yb,
                        int tokens,cudaStream_t stream) {
    matrix_bytes(a); matrix_bytes(b);
    if(a.quant!=Quant::Q8||b.quant!=Quant::Q8||a.layout!=BlockLayout::Gguf||
       b.layout!=BlockLayout::Gguf||a.rows!=b.rows||a.columns!=b.columns||
       !x||!ya||!yb||ya==yb||tokens<1)
        throw std::invalid_argument("Q8 paired projection inventory/extent mismatch");
    if(tokens<=3) project_pair<1><<<dim3(a.rows,tokens),128,0,stream>>>(
        static_cast<const unsigned char*>(a.data),static_cast<const unsigned char*>(b.data),
        x,ya,yb,a.rows,a.columns,tokens);
    else project_pair<4><<<dim3(a.rows,(tokens+3)/4),128,0,stream>>>(
        static_cast<const unsigned char*>(a.data),static_cast<const unsigned char*>(b.data),
        x,ya,yb,a.rows,a.columns,tokens);
    check_cuda(cudaGetLastError(),"Strata Q8 GDN paired projection");
}
}
