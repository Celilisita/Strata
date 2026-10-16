#include "strata/qwen35/native/small_prefill.hpp"
#include "strata/qwen35/native/ffn.hpp"
#include "small_prefill_values.cuh"
#include "small_prefill_boundary.hpp"
#include "small_prefill_direct_limits.cuh"

namespace strata::qwen35::native {
namespace {
using detail::TokenBlock;
using detail::weight_codes;
using detail::metadata;
__global__ void encode(const float *input, TokenBlock *output, int columns) {
    const int group=int(blockIdx.x)*16+threadIdx.x/8, lane=threadIdx.x%8;
    if(group>=columns/32)return;
    const auto values=reinterpret_cast<const float4 *>(input+size_t(blockIdx.y)*columns)[group*8+lane];
    float maximum=fmaxf(fmaxf(fabsf(values.x),fabsf(values.y)),fmaxf(fabsf(values.z),fabsf(values.w)));
    float sum=__fadd_rn(__fadd_rn(__fadd_rn(values.x,values.y),values.z),values.w);
    for(int offset=4;offset;offset/=2){
        maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,offset));
        sum=__fadd_rn(sum,__shfl_xor_sync(0xffffffff,sum,offset));
    }
    const float reciprocal=maximum?__fdividef(127.0f,maximum):0;
    const float delta=maximum?__fdividef(1.0f,reciprocal):0;
    auto &block=output[size_t(blockIdx.y)*(columns/32)+group];
    const char4 codes=make_char4(static_cast<signed char>(roundf(__fmul_rn(values.x,reciprocal))),
                                 static_cast<signed char>(roundf(__fmul_rn(values.y,reciprocal))),
                                 static_cast<signed char>(roundf(__fmul_rn(values.z,reciprocal))),
                                 static_cast<signed char>(roundf(__fmul_rn(values.w,reciprocal))));
    reinterpret_cast<char4 *>(block.codes)[lane]=codes;
    if(!lane){block.delta=delta;block.rounded_delta=__float2half_rn(delta);block.original_sum=__float2half_rn(sum);}
}
__device__ bool half_boundary(float value,float uncertainty){
    const unsigned short bits=__half_as_ushort(__float2half_rn(value));
    if((bits&0x7c00)==0x7c00)return true;
    const float rounded=__half2float(__ushort_as_half(bits));
    const unsigned short below=(bits&0x7fff)==0?0x8001:static_cast<unsigned short>(bits+(bits&0x8000?1:-1));
    const unsigned short above=(bits&0x7fff)==0?1:static_cast<unsigned short>(bits+(bits&0x8000?-1:1));
    const float lower=0.5f*__fadd_rn(rounded,__half2float(__ushort_as_half(below)));
    const float upper=0.5f*__fadd_rn(rounded,__half2float(__ushort_as_half(above)));
    return fabsf(value-lower)<=uncertainty||fabsf(value-upper)<=uncertainty;
}
__global__ void boundary_mask(const float *values,unsigned *mask,int groups){
    const int group=int(blockIdx.x)*4+threadIdx.x/32,lane=threadIdx.x%32;
    if(group>=groups)return;
    const float value=values[size_t(group)*32+lane];
    float maximum=fabsf(value);
    for(int offset=16;offset;offset/=2)
        maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,offset));
    const float scaled=fabsf(__fmul_rn(value,maximum?__fdividef(127.0f,maximum):0));
    const float distance=fabsf(__fsub_rn(__fsub_rn(scaled,floorf(scaled)),0.5f));
    // A conservative relative F32 uncertainty band selects recomputation;
    // the quantizer itself retains roundf and its original scale/sum policy.
    constexpr float error=32.0f*0x1p-23f;
    const int first=lane&~3;
    float sum=__fadd_rn(__fadd_rn(__fadd_rn(__shfl_sync(0xffffffff,value,first),
        __shfl_sync(0xffffffff,value,first+1)),__shfl_sync(0xffffffff,value,first+2)),
        __shfl_sync(0xffffffff,value,first+3));
    for(int offset=16;offset>=4;offset/=2)
        sum=__fadd_rn(sum,__shfl_xor_sync(0xffffffff,sum,offset));
    float absolute=fabsf(value);
    for(int offset=16;offset;offset/=2)
        absolute=__fadd_rn(absolute,__shfl_xor_sync(0xffffffff,absolute,offset));
    const float delta=maximum?__fdividef(1.0f,__fdividef(127.0f,maximum)):0;
    const bool uncertain=distance<=error*__fadd_rn(scaled,1.0f)||
        half_boundary(delta,error*fabsf(delta))||half_boundary(sum,error*absolute);
    const unsigned selected=__ballot_sync(0xffffffff,uncertain);
    if(!lane)mask[group]=selected!=0;
}
template<Quant Type,bool Masked=false>
__global__ void project(Matrix weight,const TokenBlock *input,float *output,int tokens,int units,const unsigned *mask=nullptr){
    constexpr int Rows=Masked?32:8,TokenTile=Masked?1:32,Bytes=Type==Quant::Q4K?144:210;
    constexpr int Stride=(Bytes+3)&~3;
    __shared__ __align__(16) unsigned char raw[Rows][Stride];
    __shared__ int codes[64][Rows];
    const int local_row=threadIdx.x%Rows;
    const int row=int(blockIdx.x)*Rows+local_row;
    const int token=int(blockIdx.y)*TokenTile+threadIdx.x/Rows;
    bool active=row<weight.rows&&token<tokens;
    if constexpr(Masked){
        active=active&&mask[size_t(token)*(weight.rows/32)+row/32];
        if(!__syncthreads_or(active))return;
    }
    const int blocks=weight.columns/256,total=((weight.rows+127)/128)*blocks;
    const int start=(int(blockIdx.x)*Rows/128)*blocks;
    int end=start+blocks;
    float tail=0,preceding=0;bool first=true;
    // A physical row tile shares one numerical interval. Each worker still
    // owns its output's rounding; the weight fetch/decode is shared by32 tokens.
    do{
        const int owner=int((int64_t(end)*units-1)/total);
        const int begin=max(start,int(int64_t(owner)*total/units));
        float result=0;
        for(int block=begin-start;block<end-start;++block){
            // Two-byte accesses support the210-byte Q6 GGUF extent. Adjacent
            // workers cooperate on contiguous words, rather than strided rows.
            for(int i=threadIdx.x;i<Rows*(Bytes/2);i+=blockDim.x){
                const int r=i/(Bytes/2),position=i%(Bytes/2);
                const int source_row=int(blockIdx.x)*Rows+r;
                unsigned short value=0;
                if(source_row<weight.rows){
                    const auto *p=static_cast<const unsigned char *>(weight.data)+
                                  (size_t(source_row)*blocks+block)*Bytes;
                    value=reinterpret_cast<const unsigned short *>(p)[position];
                }
                reinterpret_cast<unsigned short *>(raw[r])[position]=value;
            }
            __syncthreads();
            for(int i=threadIdx.x;i<64*Rows;i+=blockDim.x){
                const int r=i%Rows,word_index=i/Rows;
                codes[word_index][r]=weight_codes<Type>(raw[r],word_index/8,(word_index%8)*4);
            }
            __syncthreads();
            if(active){
                const auto *p=raw[local_row];
                const auto *activation=input+size_t(token)*(weight.columns/32)+block*8;
                const float d=__half2float(*reinterpret_cast<const half *>(p+(Type==Quant::Q6K?208:0)));
                const float minimum=Type==Quant::Q4K?__half2float(*reinterpret_cast<const half *>(p+2)):0;
                #pragma unroll
                for(int pair=0;pair<4;++pair){
                    float scaled=0,correction=0;
                    #pragma unroll
                    for(int part=0;part<2;++part){
                        const int group=pair*2+part;
                        const auto &x=activation[group];
                        int product[2]{};
                        #pragma unroll
                        for(int position=0;position<8;++position){
                            const int q=codes[group*8+position][local_row];
                            const int code=reinterpret_cast<const int *>(x.codes)[position];
                            product[position/4]=__dp4a(q,code,product[position/4]);
                        }
                        if constexpr(Type==Quant::Q6K){
                            const auto *scale=reinterpret_cast<const signed char *>(p+192)+group*2;
                            const int weighted=int(scale[0])*product[0]+int(scale[1])*product[1];
                            scaled=__fmaf_rn(x.delta,float(weighted),scaled);
                        }else{
                            const unsigned meta=metadata(p,group,weight.layout==BlockLayout::Q4Pairs);
                            scaled=__fmaf_rn(__half2float(x.rounded_delta),float(int(meta&63)*(product[0]+product[1])),scaled);
                            correction=__fmaf_rn(__half2float(x.original_sum),float((meta>>6)&63),correction);
                        }
                    }
                    if constexpr(Type==Quant::Q6K)result=__fmaf_rn(d,scaled,result);
                    else result=__fadd_rn(result,__fmaf_rn(d,scaled,-__fmul_rn(minimum,correction)));
                }
            }
            __syncthreads();
        }
        if(first){tail=result;first=false;}else preceding=__fadd_rn(preceding,result);
        end=begin;
    }while(end>start);
    if(active)output[size_t(token)*weight.rows+row]=__fadd_rn(tail,preceding);
}
}
void small_prefill_quantize(const float *input,void *output,int columns,int tokens,cudaStream_t stream){
    if(columns<=0||columns%256||tokens<2||tokens>=64)throw std::invalid_argument("small PP quantizer shape");
    encode<<<dim3((columns/32+15)/16,tokens),128,0,stream>>>(input,static_cast<TokenBlock *>(output),columns);
    check_cuda(cudaGetLastError(),"owned small PP encode");
}
void small_prefill_q4_boundary_mask(const float *values,unsigned *mask,int columns,int tokens,cudaStream_t stream){
    const size_t groups=size_t(columns/32)*tokens;
    if(!values||!mask||columns<=0||columns%32||tokens!=32||
       reinterpret_cast<uintptr_t>(values)%16||reinterpret_cast<uintptr_t>(mask)%16||
       detail::split_overlap(values,groups*32*4,mask,groups*4))
        throw std::invalid_argument("Q4 boundary mask requires distinct aligned PP32 spans");
    boundary_mask<<<unsigned((groups+3)/4),128,0,stream>>>(values,mask,int(groups));
    check_cuda(cudaGetLastError(),"owned Q4 boundary detector");
}
void small_prefill_q4_masked_linear(Matrix weight,const void *input,float *output,const unsigned *mask,
                                   int tokens,cudaStream_t stream,int processors){
    const size_t wb=matrix_bytes(weight),ib=size_t(weight.columns/32)*tokens*SmallPrefillActivationBytes;
    const size_t ob=size_t(weight.rows)*tokens*4,mb=size_t(weight.rows/32)*tokens*4;
    if(weight.quant!=Quant::Q4K||weight.layout!=BlockLayout::Q4Pairs||weight.rows%32||tokens!=32||processors<=0||
       !input||!output||!mask||reinterpret_cast<uintptr_t>(mask)%16||reinterpret_cast<uintptr_t>(input)%4||
       reinterpret_cast<uintptr_t>(output)%16||reinterpret_cast<uintptr_t>(weight.data)%16||
       detail::split_overlap(input,ib,output,ob)||detail::split_overlap(mask,mb,input,ib)||
       detail::split_overlap(mask,mb,output,ob)||detail::split_overlap(mask,mb,weight.data,wb)||
       detail::split_overlap(output,ob,weight.data,wb)||detail::split_overlap(input,ib,weight.data,wb))
        throw std::invalid_argument("masked Q4 projection requires distinct PP32 borrowed spans");
    const int tiles=(weight.rows+127)/128,waves=(tiles+processors-1)/processors;
    const int units=100*tiles/(processors*waves)>=90?tiles:processors;
    project<Quant::Q4K,true><<<dim3(weight.rows/32,tokens),32,0,stream>>>(
        weight,static_cast<const TokenBlock *>(input),output,tokens,units,mask);
    check_cuda(cudaGetLastError(),"owned masked Q4 projection");
}
void small_prefill_linear(Matrix weight,const void *input,float *output,int tokens,cudaStream_t stream,int processors){
    matrix_bytes(weight);
    if(tokens<2||tokens>=64)throw std::invalid_argument("small PP projection shape");
    if(processors<=0)throw std::invalid_argument("small PP rounding policy processors");
    const int tiles=(weight.rows+127)/128;
    const int waves=(tiles+processors-1)/processors;
    const int units=100*tiles/(processors*waves)>=90?tiles:processors;
    const dim3 grid((weight.rows+7)/8,(tokens+31)/32);
    if(weight.quant==Quant::Q6K)project<Quant::Q6K><<<grid,256,0,stream>>>(weight,static_cast<const TokenBlock *>(input),output,tokens,units);
    else project<Quant::Q4K><<<grid,256,0,stream>>>(weight,static_cast<const TokenBlock *>(input),output,tokens,units);
    check_cuda(cudaGetLastError(),"owned small PP integer projection");
}
}
