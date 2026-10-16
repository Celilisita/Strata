#pragma once
#include "small_prefill_values.cuh"
#include "small_prefill_mma.cuh"
#include "small_prefill_blocked.cuh"
#include "small_prefill_half_input.cuh"
#include "small_prefill_half_raw.hpp"
#include "small_prefill_fused_reduce.cuh"
#include "small_prefill_direct_limits.cuh"
#include "small_prefill_q6_copy.cuh"

namespace strata::qwen35::native::detail::direct {
constexpr int Rows=DirectRows,Tokens=DirectTokens,Threads=256;
template<Quant Type,bool Blocked,bool Complete,bool PackedCodes=false,bool Split=false,bool HalfInput=false,bool HalfWeight=false,bool Fused=false,bool CompactInput=false,int Partitions=DirectSplits>
__global__ __launch_bounds__(Threads,PackedCodes?3:2) void direct_project(Matrix weight,const void *input,
                                                         float *output,int tokens,int units,
                                                         float *final_output=nullptr,unsigned *counters=nullptr){
    static_assert(!Split||(Type==Quant::Q6K&&Blocked&&Complete));
    static_assert(Partitions==2||Partitions==3);
    static_assert(!HalfInput||Blocked);
    static_assert(!HalfWeight||(HalfInput&&Type==Quant::Q6K&&PackedCodes));
    static_assert(!Fused||(HalfWeight&&Split));
    static_assert(!CompactInput||HalfInput);
    #if __CUDA_ARCH__ >= 800
    constexpr int Words=1;
    #else
    constexpr int Words=2;
    #endif
    if constexpr(HalfInput&&Words==1)return;
    constexpr int Bytes=Type==Quant::Q4K?144:(PackedCodes?(Words==1?288:32):210);
    constexpr int Stride=Type==Quant::Q6K && !PackedCodes && HalfInput && CompactInput && Words==2?240:(Bytes+15)&~15;
    constexpr int ScaleOffset=PackedCodes?(Words==1?256:0):192;
    constexpr int DeltaOffset=PackedCodes?(Words==1?272:16):208;
    extern __shared__ __align__(32) unsigned char arena[];
    auto *raw=reinterpret_cast<unsigned char (*)[Stride]>(arena);
    auto *encoded_weight=reinterpret_cast<unsigned *>(arena+Rows*Stride);
    auto *encoded_input=encoded_weight+(Words==1?32*Rows:detail::HalfOperandPitch*Rows);
    constexpr int InputPitch=CompactInput?detail::CompactHalfOperandPitch:detail::HalfOperandPitch;
    auto *delta=reinterpret_cast<float *>(encoded_input+(Words==1?32*Tokens:InputPitch*Tokens));
    auto *ds=reinterpret_cast<unsigned *>(delta+8*Tokens);
    const int tid=threadIdx.x,warp=tid/32,lane=tid%32;
    const int first_row=int(blockIdx.x)*Rows,first_token=int(blockIdx.y)*Tokens;
    const int blocks=weight.columns/256,total=((weight.rows+127)/128)*blocks;
    const int start=(first_row/128)*blocks;
    const int first_block=Split?int(int64_t(blockIdx.z)*blocks/Partitions):0;
    const int last_block=Split?int(int64_t(blockIdx.z+1)*blocks/Partitions):blocks;
    int end=start+last_block;bool first=true;
    float tail[8]{},preceding[8]{};
    do{
        const int owner=Complete?0:int((int64_t(end)*units-1)/total);
        const int begin=Complete?start+first_block:max(start,int(int64_t(owner)*total/units));
        float result[8]{};
        for(int block=begin-start;block<end-start;++block){
            const size_t half_tile=size_t(first_row/Rows)*blocks+block;
            // A warp owns a row; no quotient/remainder by the 105-word Q6
            // extent for every copied half. Q4 is naturally 16-byte aligned.
            if constexpr(HalfWeight){
                const auto *metadata=static_cast<const unsigned char *>(weight.data)+
                                     size_t(weight.rows)*blocks*512+half_tile*Rows*Stride;
                if(tid<Rows*Stride/16)
                    reinterpret_cast<uint4 *>(raw)[tid]=reinterpret_cast<const uint4 *>(metadata)[tid];
            }else{ for(int r=warp;r<Rows;r+=Threads/32){
                const bool valid=first_row+r<weight.rows;
                const auto *p=static_cast<const unsigned char *>(weight.data)+
                              (size_t(first_row+r)*blocks+block)*(PackedCodes?256:Bytes);
                if constexpr(PackedCodes){
                    if constexpr(Words==1){
                        if(lane<16)
                            reinterpret_cast<uint4 *>(raw[r])[lane]=valid?reinterpret_cast<const uint4 *>(p)[lane]:make_uint4(0,0,0,0);
                    }
                    if(lane<9){
                        const auto *meta=static_cast<const unsigned char *>(weight.data)+size_t(weight.rows)*blocks*256+
                                         (size_t(first_row+r)*blocks+block)*18;
                        reinterpret_cast<unsigned short *>(raw[r])[ScaleOffset/2+lane]=valid?reinterpret_cast<const unsigned short *>(meta)[lane]:0;
                    }
                }else if constexpr(Type==Quant::Q4K){
                    if(lane<Bytes/16){
                        const uint4 value=valid?reinterpret_cast<const uint4 *>(p)[lane]:make_uint4(0,0,0,0);
                        reinterpret_cast<uint4 *>(raw[r])[lane]=value;
                    }
                }else if constexpr(HalfInput){
                    detail::copy_raw_q6_row(p,reinterpret_cast<unsigned *>(raw[r]),lane,valid);
                }else{
                    #pragma unroll
                    for(int pass=0;pass<2;++pass){
                        const int position=lane*2+pass*64;
                        if(position<Stride/2){
                            unsigned short first=0,second=0;
                            if(valid&&position<Bytes/2)first=reinterpret_cast<const unsigned short *>(p)[position];
                            if(valid&&position+1<Bytes/2)second=reinterpret_cast<const unsigned short *>(p)[position+1];
                            reinterpret_cast<unsigned *>(raw[r])[position/2]=unsigned(first)|(unsigned(second)<<16);
                        }
                    }
                }
            }}
            for(int i=tid;i<8*Tokens;i+=Threads){
                const int group=i/Tokens,token=i%Tokens;
                float d=0;unsigned metadata=0;
                if(first_token+token<tokens){
                    if constexpr(HalfInput){
                        if constexpr(Type==Quant::Q6K)d=static_cast<const detail::HalfInputBlock *>(input)[block].delta[group][token];
                        else metadata=static_cast<const detail::HalfAffineInputBlock *>(input)[block].ds[group][token];
                    }else if constexpr(Blocked){
                        const auto &x=static_cast<const detail::VectorBlock *>(input)[block*((tokens+31)/32)+int(blockIdx.y)];
                        d=x.delta[group][token];if constexpr(Type==Quant::Q4K)metadata=x.ds[group][token];
                    }else{
                        const auto &x=static_cast<const detail::TokenBlock *>(input)[size_t(first_token+token)*(weight.columns/32)+block*8+group];
                        d=x.delta;if constexpr(Type==Quant::Q4K)metadata=*reinterpret_cast<const unsigned *>(&x.rounded_delta);
                    }
                }
                delta[i]=d;if constexpr(Type==Quant::Q4K)ds[i]=metadata;
            }
            __syncthreads();
            float weight_delta[2];
            #pragma unroll
            for(int r=0;r<2;++r){
                const auto *p=raw[detail::result_row(warp,lane,r*2)];
                weight_delta[r]=__half2float(*reinterpret_cast<const half *>(p+(Type==Quant::Q6K?DeltaOffset:0)));
            }
            for(int chunk=0;chunk<2;++chunk){
                if constexpr(HalfInput){
                    for(int token=warp;token<Tokens;token+=Threads/32){
                        const unsigned *codes;
                        if constexpr(Type==Quant::Q6K)codes=static_cast<const detail::HalfInputBlock *>(input)[block].codes[token]+chunk*64;
                        else codes=static_cast<const detail::HalfAffineInputBlock *>(input)[block].codes[token]+chunk*64;
                        const uint2 value=reinterpret_cast<const uint2 *>(codes)[lane];
                        *reinterpret_cast<uint2 *>(encoded_input+token*InputPitch+lane*2)=value;
                    }
                }else{ for(int i=tid;i<32*Tokens;i+=Threads){
                    const int word=i/Tokens,token=i%Tokens;
                    unsigned packed=0;
                    if(first_token+token<tokens){
                        if constexpr(Blocked){
                            const auto &x=static_cast<const detail::VectorBlock *>(input)[block*((tokens+31)/32)+int(blockIdx.y)];
                            packed=unsigned(x.codes[chunk*32+word][token]);
                        }else{
                            const auto &x=static_cast<const detail::TokenBlock *>(input)[size_t(first_token+token)*(weight.columns/32)+block*8+chunk*4+word/8];
                            packed=unsigned(reinterpret_cast<const int *>(x.codes)[word%8]);
                        }
                    }
                    if constexpr(Words==1)encoded_input[i]=packed;
                    else{
                        const uint2 value=make_uint2(detail::half_codes(packed,true,0),detail::half_codes(packed,true,2));
                        *reinterpret_cast<uint2 *>(encoded_input+token*detail::HalfOperandPitch+word*2)=value;
                    }
                }}
                if constexpr(Words==1){
                    for(int i=tid;i<32*Rows;i+=Threads){
                        const int word=i/Rows,row=i%Rows;
                        if constexpr(PackedCodes)encoded_weight[i]=detail::word<true>(raw[row]+chunk*128+word*4);
                        else encoded_weight[i]=unsigned(detail::weight_codes<Type,true>(raw[row],chunk*4+word/8,(word%8)*4));
                    }
                }else if constexpr(HalfWeight){
                    const auto *codes=static_cast<const unsigned *>(weight.data)+half_tile*Rows*128+chunk*Rows*64;
                    for(int i=tid;i<Rows*16;i+=Threads){
                        const int row=i/16,word=(i%16)*4;
                        const uint4 value=reinterpret_cast<const uint4 *>(codes)[i];
                        *reinterpret_cast<uint4 *>(encoded_weight+row*detail::HalfOperandPitch+word)=value;
                    }
                }else{
                    // A warp writes a contiguous row of half2 pairs; two
                    // 128-byte transactions cover the32 aligned uint2 stores.
                    for(int row=warp;row<Rows;row+=Threads/32){
                        unsigned packed=0;
                        if constexpr(PackedCodes){
                            if(first_row+row<weight.rows){
                                const auto *codes=static_cast<const unsigned *>(weight.data)+
                                                  (size_t(first_row+row)*blocks+block)*64+chunk*32;
                                packed=codes[lane];
                            }
                        }else packed=unsigned(detail::weight_codes<Type,true>(raw[row],chunk*4+lane/8,(lane%8)*4));
                        const uint2 value=make_uint2(detail::half_codes(packed,Type==Quant::Q6K,0),
                                                    detail::half_codes(packed,Type==Quant::Q6K,2));
                        *reinterpret_cast<uint2 *>(encoded_weight+row*detail::HalfOperandPitch+lane*2)=value;
                    }
                }
                __syncthreads();
                #pragma unroll
                for(int pair=0;pair<2;++pair){
                    float scaled[8]{},correction[8]{};
                    #pragma unroll
                    for(int part=0;part<2;++part){
                        const int local_group=pair*2+part,group=chunk*4+local_group;
                        detail::IntegerDot one[8]{};
                        detail::code_dot16<Rows,CompactInput>(encoded_weight,encoded_input,local_group,0,warp,lane,one);
                        if constexpr(Type==Quant::Q6K){
                            detail::IntegerDot two[8]{};
                            detail::code_dot16<Rows,CompactInput>(encoded_weight,encoded_input,local_group,1,warp,lane,two);
                            int a[2],b[2];float activation_delta[4];
                            #pragma unroll
                            for(int r=0;r<2;++r){
                                const auto *p=raw[detail::result_row(warp,lane,r*2)];
                                const auto *scale=reinterpret_cast<const signed char *>(p+ScaleOffset)+group*2;
                                a[r]=int(scale[0]);b[r]=int(scale[1]);
                            }
                            #pragma unroll
                            for(int t=0;t<4;++t)
                                activation_delta[t]=delta[group*Tokens+detail::result_token(warp,lane,(t&1)+(t/2)*4)];
                            #pragma unroll
                            for(int i=0;i<8;++i){
                                const int r=(i/2)%2,t=(i&1)+(i/4)*2;
                                const float weighted=detail::weighted_q6(one[i],two[i],a[r],b[r]);
                                scaled[i]=__fmaf_rn(activation_delta[t],weighted,scaled[i]);
                            }
                        }else{
                            // One exact integer32-dot; two16-dots do not need
                            // separate accumulator banks before the Q4 scale.
                            detail::code_dot16<Rows,CompactInput>(encoded_weight,encoded_input,local_group,1,warp,lane,one);
                            #pragma unroll
                            for(int i=0;i<8;++i){
                                const int row=detail::result_row(warp,lane,i),token=detail::result_token(warp,lane,i);
                                const auto *p=raw[row];const int index=group*Tokens+token;
                                const unsigned meta=detail::metadata(p,group,weight.layout==BlockLayout::Q4Pairs);
                                const float weighted=detail::weighted_q4(one[i],detail::IntegerDot(0),int(meta&63));
                                const auto *halves=reinterpret_cast<const half *>(ds+index);
                                scaled[i]=__fmaf_rn(__half2float(halves[0]),weighted,scaled[i]);
                                correction[i]=__fmaf_rn(__half2float(halves[1]),float((meta>>6)&63),correction[i]);
                            }
                        }
                    }
                    #pragma unroll
                    for(int i=0;i<8;++i){
                        const auto *p=raw[detail::result_row(warp,lane,i)];
                        const float d=weight_delta[(i/2)%2];
                        if constexpr(Type==Quant::Q6K)result[i]=__fmaf_rn(d,scaled[i],result[i]);
                        else{
                            const float minimum=__half2float(*reinterpret_cast<const half *>(p+2));
                            result[i]=__fadd_rn(result[i],__fmaf_rn(d,scaled[i],-__fmul_rn(minimum,correction[i])));
                        }
                    }
                }
                __syncthreads();
            }
        }
        if constexpr(Complete){
            #pragma unroll
            for(int i=0;i<8;++i){
                const int row=first_row+detail::result_row(warp,lane,i),token=first_token+detail::result_token(warp,lane,i);
                const size_t plane=Split?size_t(blockIdx.z)*tokens*weight.rows:0;
                if(row<weight.rows&&token<tokens){
                    const size_t index=plane+size_t(token)*weight.rows+row;
                    if constexpr(Fused)static_cast<volatile float *>(output)[index]=__fadd_rn(result[i],0.0f);
                    else output[index]=__fadd_rn(result[i],0.0f);
                }
            }
            if constexpr(Fused)detail::finish_split3(output,final_output,counters,weight.rows,first_row,tid);
            return;
        }
        #pragma unroll
        for(int i=0;i<8;++i){if(first)tail[i]=result[i];else preceding[i]=__fadd_rn(preceding[i],result[i]);}
        first=false;end=begin;
    }while(end>start);
    #pragma unroll
    for(int i=0;i<8;++i){
        const int row=first_row+detail::result_row(warp,lane,i),token=first_token+detail::result_token(warp,lane,i);
        if(row<weight.rows&&token<tokens)output[size_t(token)*weight.rows+row]=__fadd_rn(tail[i],preceding[i]);
    }
}
}
