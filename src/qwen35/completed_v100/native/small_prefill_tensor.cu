#include "strata/qwen35/completed_v100/native/small_prefill.hpp"
#include "strata/qwen35/completed_v100/native/ffn.hpp"
#include "small_prefill_values.cuh"
#include <mma.h>

namespace strata::completed_v100::qwen35::native {
namespace {
namespace wmma=nvcuda::wmma;
using detail::TokenBlock;
constexpr int Tokens=32;
thread_local int configured_rows=0;
template<Quant Type,int Rows>
constexpr int tensor_shared_bytes(){
    constexpr int Bytes=Type==Quant::Q4K?144:210,Stride=(Bytes+3)&~3;
    return Rows*Stride+Tokens*8*40+Rows*32*2+Tokens*32*2+(Type==Quant::Q6K?2:1)*Rows*Tokens*4;
}
// Quantized codes are exactly representable as half operands. Each16-value
// dot has magnitude <=65024: every product and F32 integer sum is exact.
// Apply actual GGUF/activation scales only after retrieving these integer dots.
template<Quant Type,int Rows>
__global__ __launch_bounds__(Rows*8) void tensor_project(Matrix weight,const TokenBlock *input,float *output,int tokens,int units){
    constexpr int Bytes=Type==Quant::Q4K?144:210,Stride=(Bytes+3)&~3;
    constexpr int Threads=Rows*8,Outputs=Rows*Tokens;
    extern __shared__ __align__(32) unsigned char arena[];
    auto *raw=reinterpret_cast<unsigned char (*)[Stride]>(arena);
    auto *activation_codes=reinterpret_cast<int *>(arena+Rows*Stride);
    auto *activation_delta=reinterpret_cast<float *>(activation_codes+8*Tokens*8);
    auto *activation_ds=reinterpret_cast<unsigned *>(activation_delta+8*Tokens);
    auto *lhs=reinterpret_cast<half *>(activation_ds+8*Tokens);
    auto *rhs=lhs+Rows*32;
    auto *dots=reinterpret_cast<float (*)[Outputs]>(rhs+Tokens*32);
    const int tid=threadIdx.x,warp=tid/32;
    const int first_row=int(blockIdx.x)*Rows,first_token=int(blockIdx.y)*Tokens;
    const int blocks=weight.columns/256,total=((weight.rows+127)/128)*blocks;
    const int start=(first_row/128)*blocks;
    int end=start+blocks;bool first=true;
    float tail[4]{},preceding[4]{};
    do{
        const int owner=int((int64_t(end)*units-1)/total);
        const int begin=max(start,int(int64_t(owner)*total/units));
        float result[4]{};
        for(int block=begin-start;block<end-start;++block){
            for(int i=tid;i<Rows*(Bytes/2);i+=Threads){
                const int row=i/(Bytes/2),position=i%(Bytes/2);
                unsigned short value=0;
                if(first_row+row<weight.rows){
                    const auto *p=static_cast<const unsigned char *>(weight.data)+
                                  (size_t(first_row+row)*blocks+block)*Bytes;
                    value=reinterpret_cast<const unsigned short *>(p)[position];
                }
                reinterpret_cast<unsigned short *>(raw[row])[position]=value;
            }
            for(int i=tid;i<Tokens*8*8;i+=Threads){
                const int token=i/64,group=(i/8)%8,word=i%8;
                int code=0;
                if(first_token+token<tokens){
                    const auto &x=input[size_t(first_token+token)*(weight.columns/32)+block*8+group];
                    code=reinterpret_cast<const int *>(x.codes)[word];
                }
                activation_codes[(group*Tokens+token)*8+word]=code;
            }
            for(int i=tid;i<8*Tokens;i+=Threads){
                const int group=i/Tokens,token=i%Tokens;
                float delta=0;unsigned ds=0;
                if(first_token+token<tokens){
                    const auto &x=input[size_t(first_token+token)*(weight.columns/32)+block*8+group];
                    delta=x.delta;ds=*reinterpret_cast<const unsigned *>(&x.rounded_delta);
                }
                activation_delta[i]=delta;activation_ds[i]=ds;
            }
            __syncthreads();
            for(int pair=0;pair<4;++pair){
                float scaled[4]{},correction[4]{};
                for(int part=0;part<2;++part){
                    const int group=pair*2+part;
                    // Contiguous four-code loads, shared row-major weight and
                    // token-major input operands. No floating-input conversion.
                    for(int i=tid;i<Rows*8;i+=Threads){
                        const int row=i/8,word_index=i%8;
                        const unsigned packed=unsigned(detail::weight_codes<Type>(raw[row],group,word_index*4));
                        #pragma unroll
                        for(int j=0;j<4;++j){
                            const int code=Type==Quant::Q6K?int(static_cast<signed char>(packed>>(j*8))):int((packed>>(j*8))&255);
                            lhs[row*32+word_index*4+j]=__int2half_rn(code);
                        }
                    }
                    for(int i=tid;i<Tokens*8;i+=Threads){
                        const int token=i/8,word_index=i%8;
                        const unsigned packed=unsigned(activation_codes[(group*Tokens+token)*8+word_index]);
                        #pragma unroll
                        for(int j=0;j<4;++j)
                            rhs[token*32+word_index*4+j]=__int2half_rn(int(static_cast<signed char>(packed>>(j*8))));
                    }
                    __syncthreads();
                    if(Type==Quant::Q6K || warp<Rows/8){
                        const int token_group=warp%2;
                        const int row_group=Type==Quant::Q6K?warp/4:warp/2;
                        wmma::fragment<wmma::accumulator,16,16,16,float> sum;
                        wmma::fill_fragment(sum,0.0f);
                        const int first_k=Type==Quant::Q6K?((warp/2)%2)*16:0;
                        const int last_k=Type==Quant::Q6K?first_k+16:32;
                        for(int k=first_k;k<last_k;k+=16){
                            wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a;
                            wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> b;
                            wmma::load_matrix_sync(a,lhs+row_group*16*32+k,32);
                            wmma::load_matrix_sync(b,rhs+token_group*16*32+k,32);
                            wmma::mma_sync(sum,a,b,sum);
                        }
                        wmma::store_matrix_sync(dots[Type==Quant::Q6K?(warp/2)%2:0]+row_group*16*Tokens+token_group*16,
                                                sum,Tokens,wmma::mem_row_major);
                    }
                    __syncthreads();
                    #pragma unroll
                    for(int slot=0;slot<4;++slot){
                        const int index=tid+slot*Threads,row=index/Tokens,token=index%Tokens;
                        if(first_row+row>=weight.rows||first_token+token>=tokens)continue;
                        const auto *p=raw[row];
                        const int activation_index=group*Tokens+token;
                        if constexpr(Type==Quant::Q6K){
                            const auto *scale=reinterpret_cast<const signed char *>(p+192)+group*2;
                            const int weighted=int(scale[0])*int(dots[0][index])+int(scale[1])*int(dots[1][index]);
                            scaled[slot]=__fmaf_rn(activation_delta[activation_index],float(weighted),scaled[slot]);
                        }else{
                            const unsigned meta=detail::metadata(p,group,weight.layout==BlockLayout::Q4Pairs);
                            const int weighted=int(meta&63)*int(dots[0][index]);
                            const auto *ds=reinterpret_cast<const half *>(activation_ds+activation_index);
                            scaled[slot]=__fmaf_rn(__half2float(ds[0]),float(weighted),scaled[slot]);
                            correction[slot]=__fmaf_rn(__half2float(ds[1]),float((meta>>6)&63),correction[slot]);
                        }
                    }
                    __syncthreads();
                }
                #pragma unroll
                for(int slot=0;slot<4;++slot){
                    const int row=(tid+slot*Threads)/Tokens;
                    const auto *p=raw[row];
                    const float d=__half2float(*reinterpret_cast<const half *>(p+(Type==Quant::Q6K?208:0)));
                    if constexpr(Type==Quant::Q6K)result[slot]=__fmaf_rn(d,scaled[slot],result[slot]);
                    else{
                        const float minimum=__half2float(*reinterpret_cast<const half *>(p+2));
                        const float term=__fmaf_rn(d,scaled[slot],-__fmul_rn(minimum,correction[slot]));
                        result[slot]=__fadd_rn(result[slot],term);
                    }
                }
            }
            __syncthreads();
        }
        #pragma unroll
        for(int slot=0;slot<4;++slot){
            if(first)tail[slot]=result[slot];else preceding[slot]=__fadd_rn(preceding[slot],result[slot]);
        }
        first=false;end=begin;
    }while(end>start);
    #pragma unroll
    for(int slot=0;slot<4;++slot){
        const int index=tid+slot*Threads,row=first_row+index/Tokens,token=first_token+index%Tokens;
        if(row<weight.rows&&token<tokens)output[size_t(token)*weight.rows+row]=__fadd_rn(tail[slot],preceding[slot]);
    }
}
template<int Rows>
void launch_tensor(Matrix weight,const void *input,float *output,int tokens,cudaStream_t stream,int units){
    const dim3 grid((weight.rows+Rows-1)/Rows,(tokens+Tokens-1)/Tokens);
    if(weight.quant==Quant::Q6K)
        tensor_project<Quant::Q6K,Rows><<<grid,Rows*8,tensor_shared_bytes<Quant::Q6K,Rows>(),stream>>>(
            weight,static_cast<const TokenBlock *>(input),output,tokens,units);
    else tensor_project<Quant::Q4K,Rows><<<grid,Rows*8,tensor_shared_bytes<Quant::Q4K,Rows>(),stream>>>(
            weight,static_cast<const TokenBlock *>(input),output,tokens,units);
}
}
void configure_small_prefill_tensor(){
    int device=0;check_cuda(cudaGetDevice(&device),"tensor provider setup device");
    const auto caps=capabilities(device);
    configured_rows=caps.shared_per_block>=tensor_shared_bytes<Quant::Q6K,128>()?128:64;
    if(configured_rows==128){
        check_cuda(cudaFuncSetAttribute(tensor_project<Quant::Q6K,128>,cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       tensor_shared_bytes<Quant::Q6K,128>()),"tensor Q6 shared capacity setup");
        check_cuda(cudaFuncSetAttribute(tensor_project<Quant::Q4K,128>,cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       tensor_shared_bytes<Quant::Q4K,128>()),"tensor Q4 shared capacity setup");
    }
}
void small_prefill_tensor_linear(Matrix weight,const void *input,float *output,int tokens,
                                 cudaStream_t stream,int processors){
    matrix_bytes(weight);
    if(tokens<2||tokens>=64||processors<=0||!configured_rows)
        throw std::invalid_argument("small PP tensor shape/setup/policy");
    const int tiles=(weight.rows+127)/128,waves=(tiles+processors-1)/processors;
    const int units=100*tiles/(processors*waves)>=90?tiles:processors;
    if(weight.rows<128)launch_tensor<16>(weight,input,output,tokens,stream,units);
    else if(configured_rows==128)launch_tensor<128>(weight,input,output,tokens,stream,units);
    else launch_tensor<64>(weight,input,output,tokens,stream,units);
    check_cuda(cudaGetLastError(),"owned integer-code Tensor Core projection");
}
}
