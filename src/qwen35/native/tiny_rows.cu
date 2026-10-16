#include "strata/qwen35/native/tiny_rows.hpp"
#include "strata/qwen35/native/ffn.hpp"
#include "vector_fragment.cuh"
#include "math.cuh"
namespace strata::qwen35::native {
namespace {
using detail::ActivationBlock;
template<Quant Type,bool Pair,int Tokens,bool Packed>
__global__ void row_product(Matrix a,Matrix b,const ActivationBlock *input,float *first,float *second,float *activation){
    // Four independent K accumulators expose ILP and remove the old CTA shared reduction.
    constexpr int Rows=4,Stripes=Type==Quant::Q4K&&Pair?2:4,Workers=Type==Quant::Q4K?16:32;
    constexpr int Bytes=Type==Quant::Q4K?144:210;
    const int lane=threadIdx.x%32,row=int(blockIdx.x)*Rows+threadIdx.x/32;
    if(row>=a.rows)return; // Entire warp is inactive; other rows need no CTA barrier.
    const int blocks=a.columns/256;
    float one[Tokens][Stripes]{},two[Tokens][Stripes]{};
    for(int base=lane/Workers;base<blocks;base+=Stripes*(32/Workers)){
        #pragma unroll
        for(int stripe=0;stripe<Stripes;++stripe){
            const int block=base+stripe*(32/Workers);if(block>=blocks)continue;
            const size_t offset=(size_t(row)*blocks+block)*Bytes;
            const auto *ap=static_cast<const unsigned char *>(a.data)+offset;
            const auto *bp=static_cast<const unsigned char *>(b.data)+offset;
            #pragma unroll
            for(int t=0;t<Tokens;++t){
                const auto *x=input+size_t(t)*(a.columns/32)+block*8;
                one[t][stripe]=detail::accumulate_fragment<Type,Packed>(ap,x,one[t][stripe],nullptr);
                if constexpr(Pair)two[t][stripe]=detail::accumulate_fragment<Type,Packed>(bp,x,two[t][stripe],nullptr);
            }
        }
    }
    #pragma unroll
    for(int t=0;t<Tokens;++t){
        float x=one[t][0],y=0;if constexpr(Pair)y=two[t][0];
        #pragma unroll
        for(int stripe=1;stripe<Stripes;++stripe){x=__fadd_rn(x,one[t][stripe]);if constexpr(Pair)y=__fadd_rn(y,two[t][stripe]);}
        for(int d=16;d;d/=2){
            x=__fadd_rn(x,__shfl_xor_sync(0xffffffff,x,d));
            if constexpr(Pair)y=__fadd_rn(y,__shfl_xor_sync(0xffffffff,y,d));
        }
        if(!lane){const size_t at=size_t(t)*a.rows+row;first[at]=x;
            if constexpr(Pair){second[at]=y;activation[at]=swiglu_value(x,y);}}
    }
}
// Two warps cooperate on a row. Separate K stripes keep ILP without the
// register footprint of putting all four partitions in a single warp.
template<Quant Type,bool Pair,int Tokens,bool Packed>
__global__ void cooperative_product(Matrix a,Matrix b,const ActivationBlock *input,float *first,float *second,float *activation){
    constexpr int Rows=2,Threads=64,Logical=128,Workers=Type==Quant::Q4K?16:32;
    constexpr int Bytes=Type==Quant::Q4K?144:210;
    __shared__ float partial[Rows][Tokens][Pair?2:1][Logical];
    const int local=threadIdx.x%Threads,slot=threadIdx.x/Threads,row=int(blockIdx.x)*Rows+slot;
    const bool valid=row<a.rows;const int blocks=a.columns/256;
    float one[Tokens][2]{},two[Tokens][2]{};
    if(valid)for(int base=local/Workers;base<blocks;base+=Logical/Workers){
        #pragma unroll
        for(int stripe=0;stripe<2;++stripe){
            const int block=base+stripe*(Threads/Workers);if(block>=blocks)continue;
            const size_t offset=(size_t(row)*blocks+block)*Bytes;
            const auto *ap=static_cast<const unsigned char *>(a.data)+offset;
            const auto *bp=static_cast<const unsigned char *>(b.data)+offset;
            #pragma unroll
            for(int t=0;t<Tokens;++t){
                const auto *x=input+size_t(t)*(a.columns/32)+block*8;
                one[t][stripe]=detail::accumulate_fragment<Type,Packed>(ap,x,one[t][stripe],nullptr);
                if constexpr(Pair)two[t][stripe]=detail::accumulate_fragment<Type,Packed>(bp,x,two[t][stripe],nullptr);
            }
        }
    }
    #pragma unroll
    for(int t=0;t<Tokens;++t){
        #pragma unroll
        for(int stripe=0;stripe<2;++stripe){
            partial[slot][t][0][local+stripe*Threads]=one[t][stripe];
            if constexpr(Pair)partial[slot][t][1][local+stripe*Threads]=two[t][stripe];
        }
    }
    __syncthreads(); // Tail rows still participate; no invalid matrix reads.
    if(valid&&local<32){
        #pragma unroll
        for(int t=0;t<Tokens;++t){
            float x=partial[slot][t][0][local],y=0;if constexpr(Pair)y=partial[slot][t][1][local];
            #pragma unroll
            for(int part=1;part<4;++part){
                x=__fadd_rn(x,partial[slot][t][0][local+part*32]);
                if constexpr(Pair)y=__fadd_rn(y,partial[slot][t][1][local+part*32]);
            }
            for(int d=16;d;d/=2){
                x=__fadd_rn(x,__shfl_xor_sync(0xffffffff,x,d));
                if constexpr(Pair)y=__fadd_rn(y,__shfl_xor_sync(0xffffffff,y,d));
            }
            if(!local){const size_t at=size_t(t)*a.rows+row;first[at]=x;
                if constexpr(Pair){second[at]=y;activation[at]=swiglu_value(x,y);}}
        }
    }
}
void validate(Matrix a,const void *input,float *out,int tokens){
    matrix_bytes(a);
    if(!input||!out||(tokens!=2&&tokens!=3))throw std::invalid_argument("invalid row-tiny invocation");
}
template<bool Pair,int Tokens> void launch(Matrix a,Matrix b,const void *input,float *out,float *second,float *act,cudaStream_t s){
    constexpr int Rows=Tokens==3?2:4;const int grid=(a.rows+Rows-1)/Rows;
    const auto *x=static_cast<const ActivationBlock *>(input);
    if constexpr(Tokens==3){
        if(a.quant==Quant::Q4K){
            if(a.layout==BlockLayout::Q4Pairs)cooperative_product<Quant::Q4K,Pair,Tokens,true><<<grid,128,0,s>>>(a,b,x,out,second,act);
            else cooperative_product<Quant::Q4K,Pair,Tokens,false><<<grid,128,0,s>>>(a,b,x,out,second,act);
        }else cooperative_product<Quant::Q6K,Pair,Tokens,false><<<grid,128,0,s>>>(a,b,x,out,second,act);
    }else if(a.quant==Quant::Q4K){
        if(a.layout==BlockLayout::Q4Pairs)row_product<Quant::Q4K,Pair,Tokens,true><<<grid,128,0,s>>>(a,b,x,out,second,act);
        else row_product<Quant::Q4K,Pair,Tokens,false><<<grid,128,0,s>>>(a,b,x,out,second,act);
    }else row_product<Quant::Q6K,Pair,Tokens,false><<<grid,128,0,s>>>(a,b,x,out,second,act);
    check_cuda(cudaGetLastError(),"owned warp-row tiny projection");
}
}
void tiny_row_linear(Matrix a,const void *input,float *out,int tokens,cudaStream_t s){
    validate(a,input,out,tokens);
    if(tokens==2)launch<false,2>(a,a,input,out,nullptr,nullptr,s);else launch<false,3>(a,a,input,out,nullptr,nullptr,s);
}
void tiny_row_pair(Matrix a,Matrix b,const void *input,float *out,float *second,float *act,int tokens,cudaStream_t s){
    validate(a,input,out,tokens);validate(b,input,second,tokens);
    if(!act||a.quant!=b.quant||a.layout!=b.layout||a.rows!=b.rows||a.columns!=b.columns)
        throw std::invalid_argument("invalid owned warp-row pair");
    if(tokens==2)launch<true,2>(a,b,input,out,second,act,s);else launch<true,3>(a,b,input,out,second,act,s);
}
}
