#include "strata/qwen35/native/small_prefill.hpp"
#include "small_prefill_values.cuh"
#include "small_prefill_blocked.cuh"
#include "small_prefill_load.cuh"
#include "strata/qwen35/native/ffn.hpp"

namespace strata::qwen35::native {
namespace {
constexpr int Rows=32,Tokens=32,Pitch=36,Threads=256,Slots=4;
template<Quant Type,bool Segmented>
__global__ __launch_bounds__(Threads,4) void vector_project(Matrix weight,const detail::VectorBlock *input,
                                                          float *output,int tokens,int units){
    constexpr int Bytes=Type==Quant::Q4K?144:210,Stride=(((Bytes+3)/4)|1)*4;
    union __align__(16) Stage {
        unsigned char raw[Rows][Stride];
        int input_codes[64][Pitch];
    };
    __shared__ Stage stage;
    __shared__ int weight_codes[64][Rows];
    __shared__ __align__(4) unsigned char weight_meta[Rows][20];
    __shared__ float delta[8][Tokens];
    __shared__ unsigned ds[8][Tokens];
    const int tid=threadIdx.x,row=tid%Rows,token_base=tid/Rows;
    const int first_row=int(blockIdx.x)*Rows,first_token=int(blockIdx.y)*Tokens;
    const int blocks=weight.columns/256,total=((weight.rows+127)/128)*blocks;
    const int start=(first_row/128)*blocks;
    int end=start+blocks;bool first=true;
    float tail[Slots]{},preceding[Slots]{};
    do{
        int begin=start;
        if constexpr(Segmented){
            const int owner=int((int64_t(end)*units-1)/total);
            begin=max(start,int(int64_t(owner)*total/units));
        }
        float result[Slots]{};
        for(int block=begin-start;block<end-start;++block){
            constexpr int RawWords=(Bytes+3)/4;
            static_assert((Rows*RawWords)%32==0,"raw loads require whole participating warps");
            for(int i=tid;i<Rows*RawWords;i+=Threads){
                const int r=i/RawWords,word=i%RawWords;
                const bool active=first_row+r<weight.rows;
                const size_t index=active?size_t(first_row+r)*blocks+block:0;
                const auto *p=static_cast<const unsigned char *>(weight.data)+index*Bytes;
                reinterpret_cast<unsigned *>(stage.raw[r])[word]=detail::load_raw_word<Bytes>(p,word,active);
            }
            const auto &x=input[block*((tokens+31)/32)+int(blockIdx.y)];
            {
                const int group=tid/Tokens,token=tid%Tokens;
                delta[group][token]=x.delta[group][token];
                ds[group][token]=x.ds[group][token];
            }
            __syncthreads();
            for(int i=tid;i<64*Rows;i+=Threads){
                const int word=i/Rows,r=i%Rows;
                weight_codes[word][r]=detail::weight_codes<Type,true>(stage.raw[r],word/8,(word%8)*4);
            }
            constexpr int MetaBytes=Type==Quant::Q6K?18:16;
            for(int i=tid;i<Rows*MetaBytes;i+=Threads){
                const int r=i/MetaBytes,byte=i%MetaBytes;
                weight_meta[r][byte]=stage.raw[r][(Type==Quant::Q6K?192:0)+byte];
            }
            __syncthreads();
            // Global code and metadata reads have contiguous token addresses.
            for(int i=tid;i<64*Tokens;i+=Threads){
                const int word=i/Tokens,token=i%Tokens;
                stage.input_codes[word][(token%8)*Slots+token/8]=x.codes[word][token];
            }
            __syncthreads();
            const auto *p=weight_meta[row];
            const float d=__half2float(*reinterpret_cast<const half *>(p+(Type==Quant::Q6K?16:0)));
            const float minimum=Type==Quant::Q4K?__half2float(*reinterpret_cast<const half *>(p+2)):0;
            #pragma unroll
            for(int pair=0;pair<4;++pair){
                float scaled[Slots]{},correction[Slots]{};
                #pragma unroll
                for(int part=0;part<2;++part){
                    const int group=pair*2+part;
                    int one[Slots]{},two[Slots]{};
                    #pragma unroll
                    for(int word=0;word<8;++word){
                        const int q=weight_codes[group*8+word][row];
                        const int4 packed=*reinterpret_cast<const int4 *>(&stage.input_codes[group*8+word][token_base*Slots]);
                        #pragma unroll
                        for(int slot=0;slot<Slots;++slot){
                            const int x=slot==0?packed.x:(slot==1?packed.y:(slot==2?packed.z:packed.w));
                            if(word<4)one[slot]=__dp4a(q,x,one[slot]);
                            else two[slot]=__dp4a(q,x,two[slot]);
                        }
                    }
                    #pragma unroll
                    for(int slot=0;slot<Slots;++slot){
                        const int token=token_base+slot*8;
                        if constexpr(Type==Quant::Q6K){
                            const auto *scale=reinterpret_cast<const signed char *>(p)+group*2;
                            const int weighted=int(scale[0])*one[slot]+int(scale[1])*two[slot];
                            scaled[slot]=__fmaf_rn(delta[group][token],float(weighted),scaled[slot]);
                        }else{
                            const unsigned meta=detail::metadata(p,group,weight.layout==BlockLayout::Q4Pairs);
                            const int weighted=int(meta&63)*(one[slot]+two[slot]);
                            const auto *halves=reinterpret_cast<const half *>(&ds[group][token]);
                            scaled[slot]=__fmaf_rn(__half2float(halves[0]),float(weighted),scaled[slot]);
                            correction[slot]=__fmaf_rn(__half2float(halves[1]),float((meta>>6)&63),correction[slot]);
                        }
                    }
                }
                #pragma unroll
                for(int slot=0;slot<Slots;++slot){
                    if constexpr(Type==Quant::Q6K)result[slot]=__fmaf_rn(d,scaled[slot],result[slot]);
                    else result[slot]=__fadd_rn(result[slot],__fmaf_rn(d,scaled[slot],-__fmul_rn(minimum,correction[slot])));
                }
            }
            __syncthreads();
        }
        if constexpr(!Segmented){
            #pragma unroll
            for(int slot=0;slot<Slots;++slot){
                const int r=first_row+row,token=first_token+token_base+slot*8;
                if(r<weight.rows&&token<tokens)output[size_t(token)*weight.rows+r]=__fadd_rn(result[slot],0.0f);
            }
            return;
        }
        #pragma unroll
        for(int slot=0;slot<Slots;++slot){if(first)tail[slot]=result[slot];else preceding[slot]=__fadd_rn(preceding[slot],result[slot]);}
        first=false;end=begin;
    }while(end>start);
    #pragma unroll
    for(int slot=0;slot<Slots;++slot){
        const int r=first_row+row,token=first_token+token_base+slot*8;
        if(r<weight.rows&&token<tokens)output[size_t(token)*weight.rows+r]=__fadd_rn(tail[slot],preceding[slot]);
    }
}
template<bool Segmented>
void launch_vector(Matrix weight,const void *input,float *output,int tokens,cudaStream_t stream,int units){
    const dim3 grid((weight.rows+Rows-1)/Rows,(tokens+Tokens-1)/Tokens);
    if(weight.quant==Quant::Q6K)
        vector_project<Quant::Q6K,Segmented><<<grid,Threads,0,stream>>>(weight,static_cast<const detail::VectorBlock *>(input),output,tokens,units);
    else vector_project<Quant::Q4K,Segmented><<<grid,Threads,0,stream>>>(weight,static_cast<const detail::VectorBlock *>(input),output,tokens,units);
}
}
void small_prefill_vector_linear(Matrix weight,const void *input,float *output,int tokens,
                                 cudaStream_t stream,int processors){
    matrix_bytes(weight);
    if(reinterpret_cast<unsigned long long>(weight.data)&3)throw std::invalid_argument("word staging requires aligned weight base");
    if(tokens<2||tokens>=64||processors<=0)throw std::invalid_argument("vector small PP shape/policy");
    const int tiles=(weight.rows+127)/128,waves=(tiles+processors-1)/processors;
    const int units=100*tiles/(processors*waves)>=90?tiles:processors;
    if(units==tiles)launch_vector<false>(weight,input,output,tokens,stream,units);
    else launch_vector<true>(weight,input,output,tokens,stream,units);
    check_cuda(cudaGetLastError(),"owned DP4A vector projection");
}
}
