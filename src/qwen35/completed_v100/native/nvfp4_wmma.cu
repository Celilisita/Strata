#include "nvfp4_wmma.cuh"
#include "original_fused_codec.cuh"
#include "strata/qwen35/completed_v100/native/ffn.hpp"
#include <mma.h>
#include <climits>
namespace strata::completed_v100::qwen35::native {
namespace {
template<int Tokens,bool Tiled,bool FullRows>
__global__ void compact_wmma(const unsigned char* raw,const half* input,float* partial,
    int rows,int columns,int chunk) {
    using namespace nvcuda;
    constexpr int N=64,K=32,Pitch=48; // 16-half skew keeps WMMAalignment and separates banks.
    __shared__ __align__(32) half a[16*Pitch];
    __shared__ __align__(32) unsigned char storage[N*Pitch*sizeof(half)];
    auto* b=reinterpret_cast<half*>(storage);auto* result=reinterpret_cast<float*>(storage);
    const int tid=threadIdx.x,warp=tid/32,first=blockIdx.x*N;
    const int begin=blockIdx.y*chunk,end=begin+chunk<columns?begin+chunk:columns;
    unsigned code_offset[2],scale_offset[2];
    #pragma unroll
    for(int stripe=0;stripe<2;++stripe) {
        const unsigned row=unsigned(first+tid/4+stripe*32);
        if constexpr(Tiled) {
            const unsigned tile=(row/8)*unsigned(columns/32)+unsigned(begin/32);
            code_offset[stripe]=tile*128+(row&7)*16+(tid&3)*4;
            scale_offset[stripe]=unsigned(rows)*unsigned(columns)/2+tile*16+(row&7)*2+(tid&3)/2;
        }else {
            const unsigned block=(row*unsigned(columns/64)+unsigned(begin/64))*36;
            code_offset[stripe]=block+4+unsigned(begin&32)/2+(tid&3)*4;
            scale_offset[stripe]=block+unsigned(begin&32)/16+(tid&3)/2;
        }
    }
    wmma::fragment<wmma::accumulator,16,16,16,float> accum;
    wmma::fill_fragment(accum,0.f);
    // Padding rows are initialized once and never touched in the K loop.
    #pragma unroll
    for(int i=tid;i<16*Pitch;i+=128)a[i]=__float2half_rn(0.f);
    __syncthreads();
    for(int k=begin;k<end;k+=K) {
        #pragma unroll
        for(int i=tid;i<2*Tokens*K;i+=128)a[(i/K)*Pitch+i%K]=input[size_t(i/K)*columns+k+i%K];
        #pragma unroll
        for(int stripe=0;stripe<2;++stripe) {
            const int local_row=tid/4+stripe*32,column=k+(tid&3)*8,row=first+local_row;
            __align__(16) half decoded[8];
            if(FullRows||row<rows) {
                const unsigned word=__ldg(reinterpret_cast<const unsigned*>(raw+code_offset[stripe]));
                const unsigned scale=__ldg(raw+scale_offset[stripe]);
                original_fused::nvfp4_eight_values(scale,word,decoded);
            }else {
                #pragma unroll
                for(int j=0;j<8;++j)decoded[j]=__float2half_rn(0.f);
            }
            code_offset[stripe]+=Tiled?128:((k&32)?20:16);
            scale_offset[stripe]+=Tiled?16:((k&32)?34:2);
            *reinterpret_cast<int4*>(b+local_row*Pitch+(tid&3)*8)=*reinterpret_cast<const int4*>(decoded);
        }
        __syncthreads();
        #pragma unroll
        for(int z=0;z<K;z+=16) {
            wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> av;
            wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> bv;
            wmma::load_matrix_sync(av,a+z,Pitch);wmma::load_matrix_sync(bv,b+warp*16*Pitch+z,Pitch);
            wmma::mma_sync(accum,av,bv,accum);
        }
        __syncthreads(); // B and A storage cannot be reused while another warp reads it.
    }
    wmma::store_matrix_sync(result+warp*16*16,accum,16,wmma::mem_row_major);
    __syncthreads();
    #pragma unroll
    for(int i=tid;i<2*Tokens*N;i+=128) {
        const int row=first+i%N,m=i/N,n=i%N;
        if(FullRows||row<rows)partial[(size_t(blockIdx.y)*2*Tokens+m)*rows+row]=result[(n/16)*256+m*16+n%16];
    }
}
template<bool Tiled,bool FullRows> void launch(const void* payload,const half* combined,float* partial,
    int rows,int columns,int tokens,int splits,cudaStream_t stream) {
    const int chunk=((columns+splits-1)/splits+31)/32*32;
    const dim3 grid(unsigned((rows+63)/64),unsigned(splits));
    if(tokens==1)compact_wmma<1,Tiled,FullRows><<<grid,128,0,stream>>>(static_cast<const unsigned char*>(payload),combined,partial,rows,columns,chunk);
    else if(tokens==2)compact_wmma<2,Tiled,FullRows><<<grid,128,0,stream>>>(static_cast<const unsigned char*>(payload),combined,partial,rows,columns,chunk);
    else compact_wmma<3,Tiled,FullRows><<<grid,128,0,stream>>>(static_cast<const unsigned char*>(payload),combined,partial,rows,columns,chunk);
}
}
void nvfp4_wmma_partial(bool tiled,const void* payload,const half* combined,float* partial,
    int rows,int columns,int tokens,int splits,cudaStream_t stream) {
    if(!payload||!combined||!partial||rows<1||columns<64||columns%64||tokens<1||tokens>3||splits<1||splits>32||size_t(rows)*columns>INT_MAX)
        throw std::invalid_argument("NVFP4 WMMA owner/extent");
    if(tiled) {
        if(rows%64==0)launch<true,true>(payload,combined,partial,rows,columns,tokens,splits,stream);
        else launch<true,false>(payload,combined,partial,rows,columns,tokens,splits,stream);
    }else {
        if(rows%64==0)launch<false,true>(payload,combined,partial,rows,columns,tokens,splits,stream);
        else launch<false,false>(payload,combined,partial,rows,columns,tokens,splits,stream);
    }
    check_cuda(cudaGetLastError(),"NVFP4 NVIDIA WMMA compact16 high/low partial");
}
}
