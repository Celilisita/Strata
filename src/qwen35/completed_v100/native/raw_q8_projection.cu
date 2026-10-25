#include "strata/qwen35/completed_v100/native/raw_q8_projection.hpp"
#include "strata/qwen35/completed_v100/native/ffn.hpp"
#include "q8_raw.cuh"
namespace strata::completed_v100::qwen35::native {
namespace {
__global__ void project(Matrix matrix,const float* input,float* output,int tokens) {
    const int row=blockIdx.x;if(row>=matrix.rows)return;
    float values[3]{0.f,0.f,0.f};__shared__ float partial[3][4];
    for(int c=threadIdx.x;c<matrix.columns;c+=blockDim.x) {
        const float weight=q8_raw_value(matrix.data,matrix.columns,row,c);
        #pragma unroll
        for(int t=0;t<3;++t)if(t<tokens)values[t]=fmaf(weight,input[size_t(t)*matrix.columns+c],values[t]);
    }
    #pragma unroll
    for(int t=0;t<3;++t)if(t<tokens) {
        for(int d=16;d;d/=2)values[t]+=__shfl_down_sync(0xffffffff,values[t],d);
        if(!(threadIdx.x&31))partial[t][threadIdx.x/32]=values[t];
    }
    __syncthreads();
    if(threadIdx.x<32)for(int t=0;t<tokens;++t) {
        float value=threadIdx.x<4?partial[t][threadIdx.x]:0.f;
        for(int d=16;d;d/=2)value+=__shfl_down_sync(0xffffffff,value,d);
        if(threadIdx.x==0)output[size_t(t)*matrix.rows+row]=value;
    }
}
// One or more warps own a row; groups are original32-value Q8 blocks. Keep
// byte pointers advancing by34 instead of rebuilding signed div/mod addresses
// for every scalar. Compile-time Tokens removes per-value token predicates.
template<int Tokens,int WarpsPerRow>
__global__ void project_warp(Matrix matrix,const float* input,float* output) {
    constexpr int Warps=8,Rows=Warps/WarpsPerRow;
    const unsigned lane=threadIdx.x&31,warp=threadIdx.x/32,stripe=warp%WarpsPerRow;
    const unsigned row=blockIdx.x*Rows+warp/WarpsPerRow,groups=unsigned(matrix.columns)/32;
    float values[Tokens]{};
    const auto* block=static_cast<const unsigned char*>(matrix.data)+size_t(row)*groups*34+stripe*34;
    const float* x=input+stripe*32+lane;
    if(row<unsigned(matrix.rows))for(unsigned group=stripe;group<groups;group+=WarpsPerRow) {
        const float delta=__half2float(__ldg(reinterpret_cast<const half*>(block)));
        const float weight=delta*float(__ldg(reinterpret_cast<const signed char*>(block+2)+lane));
        #pragma unroll
        for(int t=0;t<Tokens;++t)values[t]=fmaf(weight,x[size_t(t)*matrix.columns],values[t]);
        block+=WarpsPerRow*34;x+=WarpsPerRow*32;
    }
    #pragma unroll
    for(int t=0;t<Tokens;++t)
        for(int d=16;d;d/=2)values[t]+=__shfl_down_sync(0xffffffff,values[t],d);
    if constexpr(WarpsPerRow==1) {
        if(!lane&&row<unsigned(matrix.rows)) {
            #pragma unroll
            for(int t=0;t<Tokens;++t)output[size_t(t)*matrix.rows+row]=values[t];
        }
    }else {
        __shared__ float partial[Tokens][Warps];
        if(!lane) {
            #pragma unroll
            for(int t=0;t<Tokens;++t)partial[t][warp]=values[t];
        }
        __syncthreads();
        if(!stripe) {
            #pragma unroll
            for(int t=0;t<Tokens;++t) {
                float value=lane<WarpsPerRow?partial[t][warp+lane]:0.f;
                for(int d=16;d;d/=2)value+=__shfl_down_sync(0xffffffff,value,d);
                if(!lane&&row<unsigned(matrix.rows))output[size_t(t)*matrix.rows+row]=value;
            }
        }
    }
}
template<int WarpsPerRow>
void launch_warp(Matrix matrix,const float* input,float* output,int tokens,cudaStream_t stream) {
    constexpr int Rows=8/WarpsPerRow;const unsigned blocks=(matrix.rows+Rows-1)/Rows;
    if(tokens==1)project_warp<1,WarpsPerRow><<<blocks,256,0,stream>>>(matrix,input,output);
    else if(tokens==2)project_warp<2,WarpsPerRow><<<blocks,256,0,stream>>>(matrix,input,output);
    else project_warp<3,WarpsPerRow><<<blocks,256,0,stream>>>(matrix,input,output);
}

}
void raw_q8_projection(Matrix matrix,const float* input,float* output,int tokens,cudaStream_t stream) {
    original_matrix_bytes(matrix);
    if(matrix.quant!=Quant::Q8||matrix.layout!=BlockLayout::Gguf||!input||!output||input==output||tokens<1||tokens>3)
        throw std::invalid_argument("Raw Q8 projection requires original format, F32 input, T1..3");
    project<<<matrix.rows,128,0,stream>>>(matrix,input,output,tokens);
    check_cuda(cudaGetLastError(),"Raw Q8 GPU projection");
}
void raw_q8_projection_warp(Matrix matrix,const float* input,float* output,int tokens,
                            int warps_per_row,cudaStream_t stream) {
    original_matrix_bytes(matrix);
    if(matrix.quant!=Quant::Q8||matrix.layout!=BlockLayout::Gguf||!input||!output||
       input==output||tokens<1||tokens>3||(warps_per_row!=1&&warps_per_row!=2&&warps_per_row!=4))
        throw std::invalid_argument("Q8 warp projection original format/extent");
    if(warps_per_row==1)launch_warp<1>(matrix,input,output,tokens,stream);
    else if(warps_per_row==2)launch_warp<2>(matrix,input,output,tokens,stream);
    else launch_warp<4>(matrix,input,output,tokens,stream);
    check_cuda(cudaGetLastError(),"Q8 block-pointer compiletime-token GPU projection");
}

}
