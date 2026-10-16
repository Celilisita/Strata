#include "strata/qwen35/native/small_prefill.hpp"
#include "strata/qwen35/native/ffn.hpp"
#include "small_prefill_values.cuh"
#include "small_prefill_blocked.cuh"
#include "small_prefill_scaled.cuh"
#include "small_prefill_half_input.cuh"

namespace strata::qwen35::native {
namespace {
using namespace detail::scaled;
thread_local bool configured=false;
// Lossless split: large odd integer W becomes the next lower even H and
// R=+1; every other W has H=W,R=0. All H values are exact half integers.
__global__ void prepare_scaled(const unsigned char *source,unsigned char *output,size_t count){
    const size_t block=size_t(blockIdx.x)*4+threadIdx.x/32;
    const int lane=threadIdx.x%32;
    if(block>=count)return;
    const auto *p=source+block*210;
    auto *high=reinterpret_cast<half *>(output)+block*256;
    auto *bitmap=reinterpret_cast<unsigned *>(output+count*512)+block*8;
    for(int group=0;group<8;++group){
        const unsigned packed=unsigned(detail::weight_codes<Quant::Q6K>(p,group,(lane/4)*4));
        const int code=int(static_cast<signed char>(packed>>((lane%4)*8)));
        const int scale=int(reinterpret_cast<const signed char *>(p+192)[group*2+lane/16]);
        const int weighted=code*scale;
        const bool residual=abs(weighted)>2048&&(weighted&1);
        const int integer_high=residual?(weighted&~1):weighted;
        high[group*32+lane]=__int2half_rn(integer_high);
        const unsigned bits=__ballot_sync(0xffffffffu,residual);
        if(lane==0)bitmap[group]=bits;
    }
    if(lane==0)reinterpret_cast<unsigned short *>(output+count*544)[block]=
        *reinterpret_cast<const unsigned short *>(p+208);
}
template<bool HalfInput>
__global__ __launch_bounds__(Threads,3) void scaled_project(Matrix weight,const void *input,
                                                          float *output,int blocks){
#if __CUDA_ARCH__ == 700
    extern __shared__ __align__(32) unsigned char arena[];
    auto *high=reinterpret_cast<unsigned *>(arena);
    auto *encoded=high+Rows*detail::HalfOperandPitch;
    auto *delta=reinterpret_cast<float *>(encoded+Tokens*detail::HalfOperandPitch);
    auto *bitmaps=reinterpret_cast<unsigned *>(delta+8*Tokens);
    auto *indices=bitmaps+Rows*8;
    auto *counts=indices+RowTiles*8*8;
    auto *weight_delta=reinterpret_cast<float *>(counts+RowTiles*8);
    const int tid=threadIdx.x,warp=tid/32,lane=tid%32,first_row=int(blockIdx.x)*Rows;
    const size_t total=size_t(weight.rows)*blocks;
    const auto *source_high=static_cast<const unsigned *>(weight.data);
    const auto *source_bits=reinterpret_cast<const unsigned *>(static_cast<const unsigned char *>(weight.data)+total*512);
    const auto *source_delta=reinterpret_cast<const half *>(static_cast<const unsigned char *>(weight.data)+total*544);
    const auto *raw_input=static_cast<const detail::VectorBlock *>(input);
    const auto *half_input=static_cast<const detail::HalfInputBlock *>(input);
    float result[8]{};
    for(int block=0;block<blocks;++block){
        for(int i=tid;i<Rows*8;i+=Threads){
            const int row=i/8,group=i%8;
            bitmaps[i]=source_bits[(size_t(first_row+row)*blocks+block)*8+group];
        }
        if(tid<Rows)weight_delta[tid]=__half2float(source_delta[size_t(first_row+tid)*blocks+block]);
        const int group=tid/Tokens,token=tid%Tokens;
        if constexpr(HalfInput)delta[tid]=half_input[block].delta[group][token];
        else delta[tid]=raw_input[block].delta[group][token];
        __syncthreads();
        detail::scaled::build_lists(bitmaps,indices,counts,tid);
        for(int chunk=0;chunk<2;++chunk){
            for(int row=warp;row<Rows;row+=Threads/32){
                const auto *source=source_high+(size_t(first_row+row)*blocks+block)*128+chunk*64;
                const uint2 value=reinterpret_cast<const uint2 *>(source)[lane];
                *reinterpret_cast<uint2 *>(high+row*detail::HalfOperandPitch+lane*2)=value;
            }
            if constexpr(HalfInput){
                for(int t=warp;t<Tokens;t+=Threads/32){
                    const uint2 value=*reinterpret_cast<const uint2 *>(half_input[block].codes[t]+chunk*64+lane*2);
                    *reinterpret_cast<uint2 *>(encoded+t*detail::HalfOperandPitch+lane*2)=value;
                }
            }else{
                for(int i=tid;i<32*Tokens;i+=Threads){
                    const int word=i/Tokens,t=i%Tokens;
                    const unsigned packed=unsigned(raw_input[block].codes[chunk*32+word][t]);
                    const uint2 value=make_uint2(detail::half_codes(packed,true,0),detail::half_codes(packed,true,2));
                    *reinterpret_cast<uint2 *>(encoded+t*detail::HalfOperandPitch+word*2)=value;
                }
            }
            __syncthreads();
            #pragma unroll
            for(int pair=0;pair<2;++pair){
                float scaled[8]{};
                #pragma unroll
                for(int part=0;part<2;++part){
                    const int local_group=pair*2+part,absolute_group=chunk*4+local_group;
                    float dot[8]{};
                    detail::code_dot16<Rows>(high,encoded,local_group,0,warp,lane,dot);
                    detail::code_dot16<Rows>(high,encoded,local_group,1,warp,lane,dot);
                    detail::scaled::residual_dot(bitmaps,indices,counts,encoded,absolute_group,local_group,warp,lane,dot);
                    float activation_delta[4];
                    #pragma unroll
                    for(int t=0;t<4;++t)
                        activation_delta[t]=delta[absolute_group*Tokens+detail::result_token(warp,lane,(t&1)+(t/2)*4)];
                    #pragma unroll
                    for(int i=0;i<8;++i){
                        scaled[i]=__fmaf_rn(activation_delta[(i&1)+(i/4)*2],dot[i],scaled[i]);
                    }
                }
                #pragma unroll
                for(int i=0;i<8;++i)
                    result[i]=__fmaf_rn(weight_delta[detail::result_row(warp,lane,i)],scaled[i],result[i]);
            }
            __syncthreads();
        }
    }
    #pragma unroll
    for(int i=0;i<8;++i){
        const int row=first_row+detail::result_row(warp,lane,i),token=detail::result_token(warp,lane,i);
        output[size_t(token)*weight.rows+row]=__fadd_rn(result[i],0.0f);
    }
#else
    // This experiment has no modern-device arithmetic dispatch. Setup rejects it.
    (void)weight;(void)input;(void)output;(void)blocks;
#endif
}
}
size_t prepared_q6_scaled_bytes(Matrix raw){
    matrix_bytes(raw);
    if(raw.quant!=Quant::Q6K||raw.layout!=BlockLayout::Gguf)
        throw std::invalid_argument("scaled preparation requires raw GGUF Q6");
    return size_t(raw.rows)*size_t(raw.columns/256)*Q6ScaledBlockBytes;
}
Matrix prepare_q6_scaled(Matrix raw,void *output,cudaStream_t stream){
    const size_t raw_bytes=matrix_bytes(raw),bytes=prepared_q6_scaled_bytes(raw);
    const uintptr_t source=reinterpret_cast<uintptr_t>(raw.data),destination=reinterpret_cast<uintptr_t>(output);
    if(!configured||!output||source%2||destination%16||
       (source<=destination?destination-source<raw_bytes:source-destination<bytes))
        throw std::invalid_argument("scaled preparation requires configured SM70 and aligned distinct buffers");
    const size_t count=size_t(raw.rows)*size_t(raw.columns/256),grid=(count+3)/4;
    if(grid>0x7fffffffu)throw std::invalid_argument("scaled preparation CUDA grid extent");
    prepare_scaled<<<unsigned(grid),128,0,stream>>>(static_cast<const unsigned char *>(raw.data),
                                                 static_cast<unsigned char *>(output),count);
    check_cuda(cudaGetLastError(),"native exact scaled Q6 preparation");
    return {output,raw.quant,raw.rows,raw.columns,BlockLayout::Q6Scaled};
}
void configure_small_prefill_scaled(){
    configured=false;
    int device=0;check_cuda(cudaGetDevice(&device),"scaled experiment setup device");
    const auto caps=capabilities(device);
    if(caps.major!=7||caps.minor!=0||caps.shared_per_block<SharedBytes)
        throw std::runtime_error("scaled experiment is restricted to SM70 with sufficient shared memory");
    check_cuda(cudaFuncSetAttribute(scaled_project<false>,cudaFuncAttributeMaxDynamicSharedMemorySize,SharedBytes),
               "scaled experiment shared capacity setup");
    check_cuda(cudaFuncSetAttribute(scaled_project<true>,cudaFuncAttributeMaxDynamicSharedMemorySize,SharedBytes),
               "scaled half-input shared capacity setup");
    configured=true;
}
template<bool HalfInput>
void scaled_linear(Matrix weight,const void *input,float *output,int tokens,
                                 cudaStream_t stream,int processors){
    matrix_bytes(weight,false,true);
    if(!configured||weight.quant!=Quant::Q6K||weight.layout!=BlockLayout::Q6Scaled||
       reinterpret_cast<uintptr_t>(weight.data)%16||!input||!output||tokens!=Tokens||
       weight.rows%Rows||processors<=0)
        throw std::invalid_argument("scaled experiment requires aligned Q6, complete PP32 row tiles");
    const int tiles=(weight.rows+127)/128,waves=(tiles+processors-1)/processors;
    const int units=100*tiles/(processors*waves)>=90?tiles:processors;
    if(units!=tiles)throw std::invalid_argument("scaled experiment does not implement segmented numerical intervals");
    scaled_project<HalfInput><<<weight.rows/Rows,Threads,SharedBytes,stream>>>(weight,input,output,weight.columns/256);
    check_cuda(cudaGetLastError(),"native exact scaled Q6 PP32 projection");
}
void small_prefill_scaled_linear(Matrix weight,const void *input,float *output,int tokens,
                                 cudaStream_t stream,int processors){
    scaled_linear<false>(weight,input,output,tokens,stream,processors);
}
void small_prefill_scaled_half_linear(Matrix weight,const void *input,float *output,int tokens,
                                      cudaStream_t stream,int processors){
    scaled_linear<true>(weight,input,output,tokens,stream,processors);
}
}
