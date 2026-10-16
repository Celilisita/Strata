#include "small_prefill_q4_grouped.hpp"
#include "small_prefill_q4_half_codes.hpp"
#include "small_prefill_q4_scaled_codes.hpp"
#include "small_prefill_q4_tile_codes.hpp"
#include "small_prefill_q4_group_codes.hpp"
#include "small_prefill_q4_fused_reduce.cuh"
#include "small_prefill_half_raw.hpp"
#include "small_prefill_mma.cuh"
#include "math.cuh"
#include "small_prefill_values.cuh"
#include "small_prefill_half_input.cuh"
#include "small_prefill_half_group.cuh"
#include "small_prefill_direct_limits.cuh"
#include "strata/qwen35/native/ffn.hpp"
#include <algorithm>

namespace strata::qwen35::native {
#include "small_prefill_q4_tile_pack.cuh"
#include "small_prefill_q4_group_pack.cuh"
namespace {
constexpr int Rows=64,Tokens=32,Threads=256;
template<int Matrices,int TileRows=Rows>
struct Shared {
    static constexpr int Rows=TileRows;
    unsigned short coefficient[Matrices][Rows][10];
    unsigned weight_ds[Matrices][Rows];
    union {
        unsigned weight[Matrices][Rows][detail::HalfOperandPitch];
        float activation[Tokens][Rows];
    };
    unsigned input[Tokens][detail::CompactHalfOperandPitch];
    unsigned ds[8][Tokens];
    float2 affine[Matrices==1?8:1][Matrices==1?Rows:1];
};
static_assert(sizeof(Shared<2>)==47368,"paired Q4 shared extent");
static_assert(sizeof(Shared<1>)==32512,"single Q4 shared extent");
thread_local bool configured=false;
__device__ inline unsigned multiply_half_words(unsigned bits,unsigned scale){
    const half2 codes=__halves2half2(__ushort_as_half(static_cast<unsigned short>(bits)),__ushort_as_half(static_cast<unsigned short>(bits>>16)));
    const half s=__ushort_as_half(static_cast<unsigned short>(scale));const half2 v=__hmul2(codes,__halves2half2(s,s));
    return unsigned(__half_as_ushort(__low2half(v)))|(unsigned(__half_as_ushort(__high2half(v)))<<16);
}
#include "small_prefill_q4_mma.cuh"
#include "small_prefill_q4_register.cuh"
#include "small_prefill_q4_pair_block.cuh"
#include "small_prefill_q4_bias_pair.cuh"
#include "small_prefill_q4_group_pair.cuh"
template<int First,bool Bias>
__device__ __forceinline__ void single_header_pair(Shared<1> &s,uint4 raw,float positive,float negative,int row){
    #pragma unroll
    for(int group=First;group<8;group+=4){
        constexpr int stride=12;const int word=(group*stride)>>5,shift=(group*stride)&31;
        const unsigned low=word==0?raw.y:word==1?raw.z:raw.w,high=word==0?raw.z:word==1?raw.w:0;
        const unsigned descriptor=__funnelshift_r(low,high,shift)&4095u;
        s.affine[group][row]=make_float2(__fmul_rn(positive,float(descriptor&63)),__fmul_rn(negative,float(descriptor>>6)));
        if constexpr(Bias)s.coefficient[0][row][group]=__half_as_ushort(__float2half_rn(float(descriptor>>6)));
    }
}
template<int Matrices,int Partitions=1,bool Prepared=false,bool TensorBias=false,bool Fused=false,bool Scaled=false,bool Compact=false,bool Segmented=false,int TileRows=Rows,int Workers=Threads,bool CellSplit=false,bool Tiled=false,bool BlockAffine=false>
__global__ __launch_bounds__(Workers,TileRows==128?1:(Matrices==1?3:2))
void grouped_q4(Matrix gate,Matrix up,const detail::HalfAffineInputBlock *input,float *gout,float *uout,float *activation,bool materialize,detail::HalfInputBlock *quantized,float *final_output=nullptr,unsigned *counters=nullptr,int units=0){
    #if __CUDA_ARCH__ >= 800
    return;
    #else
    static_assert(Partitions==1||(Partitions==5&&Matrices==1)||(Partitions==2&&CellSplit));
    static_assert(!Prepared||Matrices==1);
    static_assert(!TensorBias||(Matrices==1&&(!Prepared||Scaled)));
    static_assert(!Fused||(TensorBias&&Partitions==5));
    static_assert(!Scaled||(Prepared&&TensorBias&&!Fused));
    static_assert(!Compact||(Scaled&&Matrices==1));
    static_assert(!Segmented||(Matrices==2&&(Partitions==1||CellSplit)));
    static_assert(!CellSplit||(Matrices==2&&Segmented&&Partitions==2));
    static_assert(!Tiled||(Matrices==2&&Segmented&&TileRows==64));
    static_assert(!BlockAffine||(Matrices==2&&!Segmented&&TileRows==64));
    static_assert(TileRows==64||(Matrices==2&&Segmented&&TileRows==128&&Workers==512));
    constexpr int Rows=TileRows,Threads=Workers;
    extern __shared__ __align__(16) unsigned char dynamic[];
    __shared__ __align__(16) unsigned char fixed[TileRows==64?sizeof(Shared<Matrices,TileRows>):16];
    auto &s=*reinterpret_cast<Shared<Matrices,TileRows> *>(TileRows==64?fixed:dynamic);
    const int tid=threadIdx.x,warp=tid/32,lane=tid%32,first=int(blockIdx.x)*Rows,blocks=gate.columns/256;
    float result[Matrices][8]{};
    int boundary=0;
    if constexpr(Segmented){
        const int total=((gate.rows+127)/128)*blocks,start=(first/128)*blocks;
        const int owner=int((int64_t(start+blocks)*units-1)/total);
        boundary=max(0,int(int64_t(owner)*total/units)-start);
    }
    float tail[Segmented&&!CellSplit?Matrices:1][Segmented&&!CellSplit?8:1]{};
    for(int interval=0;interval<(CellSplit?1:(boundary?2:1));++interval){
    if constexpr(Segmented&&!CellSplit)if(interval){
        #pragma unroll
        for(int m=0;m<Matrices;++m){
            #pragma unroll
            for(int i=0;i<8;++i){tail[m][i]=result[m][i];result[m][i]=0;}
        }
    }
    const int begin=CellSplit?(blockIdx.z?0:boundary):(Segmented?(interval?0:boundary):int(blockIdx.z)*blocks/Partitions);
    const int end=CellSplit?(blockIdx.z?boundary:blocks):(Segmented?(interval?boundary:blocks):int(blockIdx.z+1)*blocks/Partitions);
    for(int block=begin;block<end;++block){
        float positive_block[8]{};
        // Four warps cover all128 headers in one cooperative row pass. Each
        // thread decodes eight coefficients with constant bit positions.
        if constexpr(Compact){
            if(tid<Rows){const int row=first+tid;const auto *p=static_cast<const unsigned char *>(gate.data)+(size_t(row)*blocks+block)*176;
                s.weight_ds[0][tid]=row<gate.rows?__ldg(reinterpret_cast<const unsigned *>(p)):0;
                #pragma unroll
                for(int group=0;group<8;++group)s.coefficient[0][tid][group]=row<gate.rows?static_cast<unsigned short>(__ldg(reinterpret_cast<const unsigned *>(p+4)+group)>>16):0;
            }
        }else if constexpr(Prepared){
            const auto *metadata=reinterpret_cast<const float2 *>(static_cast<const unsigned char *>(gate.data)+size_t(gate.rows)*blocks*512);
            const size_t tile=size_t(blockIdx.x)*blocks+block;
            for(int i=tid;i<Rows*8;i+=Threads){const int group=i/Rows,row=i%Rows;const float2 value=metadata[tile*Rows*8+i];
                if constexpr(Scaled){
                    s.coefficient[0][row][group]=__half_as_ushort(__float2half_rn(value.x));
                    if(!group)s.weight_ds[0][row]=unsigned(__half_as_ushort(__float2half_rn(value.y)))|
                        (unsigned(__half_as_ushort(__float2half_rn(metadata[tile*Rows*8+Rows+row].y)))<<16);
                }else s.affine[group][row]=value;
            }
        }else if constexpr(Matrices==1){
            const int row=tid%Rows;const auto *p=static_cast<const unsigned char *>(gate.data)+(size_t(first+row)*blocks+block)*144;
            const uint4 raw=first+row<gate.rows?__ldg(reinterpret_cast<const uint4 *>(p)):make_uint4(0,0,0,0);
            if constexpr(TensorBias)if(tid<Rows)s.weight_ds[0][row]=raw.x;
            const float positive=__half2float(__ushort_as_half(static_cast<unsigned short>(raw.x)));
            const float negative=__half2float(__ushort_as_half(static_cast<unsigned short>(raw.x>>16)));
            if(tid<Rows)single_header_pair<0,TensorBias>(s,raw,positive,negative,row);
            else if(tid<2*Rows)single_header_pair<1,TensorBias>(s,raw,positive,negative,row);
            else if(tid<3*Rows)single_header_pair<2,TensorBias>(s,raw,positive,negative,row);
            else single_header_pair<3,TensorBias>(s,raw,positive,negative,row);
        }else if(tid<Matrices*Rows){
            const int m=tid/Rows,row=tid%Rows;
            const auto *base=static_cast<const unsigned char *>(m?up.data:gate.data);
            const auto *p=Tiled?tile_q4_header(base,int(blockIdx.x),blocks,block,row):base+(size_t(first+row)*blocks+block)*144;
            const uint4 raw=first+row<gate.rows?__ldg(reinterpret_cast<const uint4 *>(p)):make_uint4(0,0,0,0);
            if constexpr(Matrices==2||TensorBias)s.weight_ds[m][row]=raw.x;
            const float positive=Matrices==1?__half2float(__ushort_as_half(static_cast<unsigned short>(raw.x))):0;
            const float negative=Matrices==1?__half2float(__ushort_as_half(static_cast<unsigned short>(raw.x>>16))):0;
            #pragma unroll
            for(int group=0;group<8;++group){
                const int word=(group*12)>>5,shift=(group*12)&31;
                const unsigned low=word==0?raw.y:word==1?raw.z:raw.w;
                const unsigned high=word==0?raw.z:word==1?raw.w:0;
                const unsigned descriptor=__funnelshift_r(low,high,shift)&4095u;
                if constexpr(Matrices==2)s.coefficient[m][row][group]=static_cast<unsigned short>(descriptor);
                else s.affine[group][row]=make_float2(__fmul_rn(positive,float(descriptor&63)),__fmul_rn(negative,float(descriptor>>6)));
                if constexpr(TensorBias)s.coefficient[0][row][group]=__half_as_ushort(__float2half_rn(float(descriptor>>6)));
            }
        }
        if(tid<256)s.ds[tid/Tokens][tid%Tokens]=input[block].ds[tid/Tokens][tid%Tokens];
        float delta[Matrices][2]{},minimum[Matrices][2]{};
        for(int chunk=0;chunk<2;++chunk){
            // Issue all row reads before dependent half conversions. The16
            // scalar words have static indices and no shared-memory lifetime.
            if constexpr(Prepared&&!Compact){
                const auto *codes=static_cast<const unsigned *>(gate.data)+(size_t(blockIdx.x)*blocks+block)*Rows*128+chunk*Rows*64;
                #pragma unroll
                for(int r=0;r<Rows/(Threads/32);++r){const int row=warp+r*(Threads/32);
                    *reinterpret_cast<uint2 *>(s.weight[0][row]+lane*2)=__ldg(reinterpret_cast<const uint2 *>(codes+row*64)+lane);
                }
            }else if constexpr(Compact){
                #pragma unroll
                for(int batch=0;batch<Rows/(Threads/32);batch+=2){
                    unsigned packed[2],multiplier[2];const int group=chunk*4+lane/8,position=(lane%8)*4;
                    #pragma unroll
                    for(int r=0;r<2;++r){
                        const int row=warp+(batch+r)*(Threads/32);packed[r]=0;multiplier[r]=0;
                        if(first+row<gate.rows){
                            const auto *p=static_cast<const unsigned char *>(gate.data)+(size_t(first+row)*blocks+block)*176;
                            packed[r]=(__ldg(reinterpret_cast<const unsigned *>(p+48+(group/2)*32+position))>>((group%2)*4))&0x0f0f0f0fu;
                            multiplier[r]=__ldg(reinterpret_cast<const unsigned *>(p+4)+group);
                        }
                    }
                    #pragma unroll
                    for(int r=0;r<2;++r){const int row=warp+(batch+r)*(Threads/32);
                        const uint2 codes=make_uint2(multiply_half_words(detail::half_codes(packed[r],false,0),multiplier[r]),multiply_half_words(detail::half_codes(packed[r],false,2),multiplier[r]));
                        *reinterpret_cast<uint2 *>(s.weight[0][row]+lane*2)=codes;
                    }
                    asm volatile("" ::: "memory"); // End the two-row lookahead lifetime; no device fence.
                }
            }else if constexpr(Matrices==1){
                #pragma unroll
                for(int batch=0;batch<Rows/(Threads/32);batch+=2){
                    unsigned packed[2];const int group=chunk*4+lane/8,position=(lane%8)*4;
                    #pragma unroll
                    for(int r=0;r<2;++r){const int row=warp+(batch+r)*(Threads/32);packed[r]=0;
                        if(first+row<gate.rows){const auto *p=static_cast<const unsigned char *>(gate.data)+(size_t(first+row)*blocks+block)*144;
                            packed[r]=(__ldg(reinterpret_cast<const unsigned *>(p+16+(group/2)*32+position))>>((group%2)*4))&0x0f0f0f0fu;}
                    }
                    #pragma unroll
                    for(int r=0;r<2;++r){const int row=warp+(batch+r)*(Threads/32);
                        *reinterpret_cast<uint2 *>(s.weight[0][row]+lane*2)=make_uint2(detail::half_codes(packed[r],false,0),detail::half_codes(packed[r],false,2));}
                    asm volatile("" ::: "memory");
                }
            }else{
            unsigned packed[Matrices][Rows/(Threads/32)];
            #pragma unroll
            for(int m=0;m<Matrices;++m){
                const auto *base=static_cast<const unsigned char *>(m?up.data:gate.data);
                #pragma unroll
                for(int r=0;r<Rows/(Threads/32);++r){
                    const int row=warp+r*(Threads/32),group=chunk*4+lane/8,position=(lane%8)*4;
                    unsigned value=0;
                    if(first+row<gate.rows){
                        if constexpr(Tiled){
                            const auto *p=tile_q4_payload(base,gate.rows,blocks,int(blockIdx.x),block,chunk,row);
                            value=__ldg(reinterpret_cast<const unsigned *>(p+((group%4)/2)*32+position));
                        }else{
                            const auto *p=base+(size_t(first+row)*blocks+block)*144;
                            value=__ldg(reinterpret_cast<const unsigned *>(p+16+(group/2)*32+position));
                        }
                    }
                    packed[m][r]=(value>>((group%2)*4))&0x0f0f0f0fu;
                }
            }
            #pragma unroll
            for(int m=0;m<Matrices;++m){
                #pragma unroll
                for(int r=0;r<Rows/(Threads/32);++r){
                    const int row=warp+r*(Threads/32);const unsigned value=packed[m][r];
                    const uint2 codes=make_uint2(detail::half_codes(value,false,0),detail::half_codes(value,false,2));
                    *reinterpret_cast<uint2 *>(s.weight[m][row]+lane*2)=codes;
                }
            }
            }
            for(int token=warp;token<Tokens;token+=Threads/32)
                *reinterpret_cast<uint2 *>(s.input[token]+lane*2)=__ldg(reinterpret_cast<const uint2 *>(input[block].codes[token]+chunk*64)+lane);
            __syncthreads();
            if constexpr(Matrices==2)if(!chunk){
                #pragma unroll
                for(int m=0;m<Matrices;++m){
                    #pragma unroll
                    for(int r=0;r<2;++r){
                        const auto *p=reinterpret_cast<const half *>(&s.weight_ds[m][detail::result_row(warp,lane,r*2)]);
                        delta[m][r]=__half2float(p[0]);minimum[m][r]=__half2float(p[1]);
                    }
                }
            }
            if constexpr(BlockAffine)pair_block_affine(s,chunk,warp,lane,delta,minimum,result);
            else {
            #pragma unroll
            for(int pair=0;pair<2;++pair){
                float scaled[Matrices][8]{},correction[Matrices][8]{};
                #pragma unroll
                for(int part=0;part<2;++part){
                    const int local=pair*2+part,group=chunk*4+local;
                    float dot[Matrices][8]{};
                    grouped_dot16(s,local,0,warp,lane,dot);
                    grouped_dot16(s,local,1,warp,lane,dot);
                    float activation_delta[4],activation_sum[4],scale[Matrices][2],minimum_scale[Matrices][2];
                    #pragma unroll
                    for(int t=0;t<4;++t){
                        const int token=detail::result_token(warp,lane,(t&1)+(t/2)*4);
                        const auto *h=reinterpret_cast<const half *>(&s.ds[group][token]);
                        activation_delta[t]=__half2float(h[0]);activation_sum[t]=__half2float(h[1]);
                    }
                    if constexpr(Matrices==1){
                        if constexpr(Scaled){
                            #pragma unroll
                            for(int i=0;i<8;++i){const int t=(i&1)+(i/4)*2;positive_block[i]=__fmaf_rn(activation_delta[t],dot[0][i],positive_block[i]);}
                        }else{
                        float2 coefficient[2];
                        #pragma unroll
                        for(int r=0;r<2;++r)coefficient[r]=s.affine[group][detail::result_row(warp,lane,r*2)];
                        #pragma unroll
                        for(int i=0;i<8;++i){
                            const int r=(i/2)%2,t=(i&1)+(i/4)*2;
                            result[0][i]=__fmaf_rn(__fmul_rn(activation_delta[t],coefficient[r].x),dot[0][i],result[0][i]);
                            if constexpr(!TensorBias)result[0][i]=__fmaf_rn(-coefficient[r].y,activation_sum[t],result[0][i]);
                        }
                        }
                    }else{
                        #pragma unroll
                        for(int m=0;m<Matrices;++m){
                            #pragma unroll
                            for(int r=0;r<2;++r){
                                const unsigned meta=s.coefficient[m][detail::result_row(warp,lane,r*2)][group];
                                scale[m][r]=float(meta&63);
                                minimum_scale[m][r]=float((meta>>6)&63);
                            }
                        }
                        #pragma unroll
                        for(int i=0;i<8;++i){
                            const int r=(i/2)%2,t=(i&1)+(i/4)*2;
                            #pragma unroll
                            for(int m=0;m<Matrices;++m){
                                const float weighted=__fmul_rn(scale[m][r],dot[m][i]);
                                scaled[m][i]=__fmaf_rn(activation_delta[t],weighted,scaled[m][i]);
                                correction[m][i]=__fmaf_rn(activation_sum[t],minimum_scale[m][r],correction[m][i]);
                            }
                        }
                    }
                }
                if constexpr(Matrices==2){
                    #pragma unroll
                    for(int m=0;m<Matrices;++m){
                        #pragma unroll
                        for(int i=0;i<8;++i){
                            const int r=(i/2)%2;
                            result[m][i]=__fadd_rn(result[m][i],__fmaf_rn(delta[m][r],scaled[m][i],-__fmul_rn(minimum[m][r],correction[m][i])));
                        }
                    }
                }
            }
            }
            if(!TensorBias||!chunk)__syncthreads();
        }
        if constexpr(TensorBias){
            float bias[8]{};tensor_bias8(s,warp,lane,bias);
            float minimum[2],delta[2];
            #pragma unroll
            for(int r=0;r<2;++r)minimum[r]=__half2float(__ushort_as_half(static_cast<unsigned short>(s.weight_ds[0][detail::result_row(warp,lane,r*2)]>>16)));
            if constexpr(Scaled){
                #pragma unroll
                for(int r=0;r<2;++r)delta[r]=__half2float(__ushort_as_half(static_cast<unsigned short>(s.weight_ds[0][detail::result_row(warp,lane,r*2)])));
                #pragma unroll
                for(int i=0;i<8;++i)result[0][i]=__fmaf_rn(delta[(i/2)%2],positive_block[i],result[0][i]);
            }
            #pragma unroll
            for(int i=0;i<8;++i)result[0][i]=__fmaf_rn(-minimum[(i/2)%2],bias[i],result[0][i]);
            __syncthreads();
        }
    }
    }
    if constexpr(Segmented&&!CellSplit)if(boundary){
        #pragma unroll
        for(int m=0;m<Matrices;++m){
            #pragma unroll
            for(int i=0;i<8;++i)result[m][i]=__fadd_rn(tail[m][i],result[m][i]);
        }
    }
    if constexpr(CellSplit){
        #pragma unroll
        for(int i=0;i<8;++i){const int row=first+detail::result_row(warp,lane,i),token=detail::result_token(warp,lane,i);
            if(row<gate.rows){const size_t at=(size_t(blockIdx.z)*Tokens+token)*gate.rows+row;gout[at]=result[0][i];uout[at]=result[1][i];}}
        return;
    }
    #pragma unroll
    for(int i=0;i<8;++i){
        const int row=first+detail::result_row(warp,lane,i),token=detail::result_token(warp,lane,i);
        if(row<gate.rows){
            const size_t at=size_t(token)*gate.rows+row;
            if constexpr(Matrices==1){
                const size_t index=size_t(blockIdx.z)*Tokens*gate.rows+at;
                if constexpr(Fused)static_cast<volatile float *>(gout)[index]=result[0][i];
                else gout[index]=result[0][i];
            }
            else{
                if(materialize){gout[at]=result[0][i];uout[at]=result[1][i];}
                const float value=swiglu_value(result[0][i],result[1][i]);
                if(materialize || !quantized)activation[at]=value;
                if(quantized)s.activation[token][row-first]=value;
            }
        }
    }
    if constexpr(Fused)detail::finish_q4_split5(gout,final_output,counters,gate.rows,first,tid);
    if constexpr(Matrices==2)if(quantized){
        __syncthreads();
        for(int task=tid/8;task<Tokens*(Rows/32);task+=Threads/8){
            const int token=task/(Rows/32),group=task%(Rows/32);
            const float4 v=reinterpret_cast<const float4 *>(s.activation[token])[group*8+tid%8];
            detail::encode_half_group(v,quantized,first/32+group,token,1,tid%8);
        }
    }
    #endif
}
template<int Partitions>
__global__ void reduce_single(const float *parts,float *output,size_t vectors){
    const size_t i=size_t(blockIdx.x)*Threads+threadIdx.x;if(i>=vectors)return;
    const auto *p=reinterpret_cast<const float4 *>(parts);float4 result=p[i];
    if constexpr(Partitions==3){
        const float4 a=p[vectors+i],b=p[2*vectors+i];
        const float4 preceding=make_float4(__fadd_rn(a.x,b.x),__fadd_rn(a.y,b.y),__fadd_rn(a.z,b.z),__fadd_rn(a.w,b.w));
        result=make_float4(__fadd_rn(result.x,preceding.x),__fadd_rn(result.y,preceding.y),__fadd_rn(result.z,preceding.z),__fadd_rn(result.w,preceding.w));
        reinterpret_cast<float4 *>(output)[i]=result;return;
    }
    #pragma unroll
    for(int slice=1;slice<Partitions;++slice){const float4 v=p[size_t(slice)*vectors+i];
        result=make_float4(__fadd_rn(result.x,v.x),__fadd_rn(result.y,v.y),__fadd_rn(result.z,v.z),__fadd_rn(result.w,v.w));}
    reinterpret_cast<float4 *>(output)[i]=result;
}
__global__ void reduce_small_cells(const float *parts,float *output,size_t vectors,int cells){
    const size_t i=size_t(blockIdx.x)*Threads+threadIdx.x;if(i>=vectors)return;
    const auto *p=reinterpret_cast<const float4 *>(parts);float4 preceding=make_float4(0,0,0,0);
    for(int cell=1;cell<cells;++cell){const float4 v=p[size_t(cell)*vectors+i];
        preceding=make_float4(__fadd_rn(preceding.x,v.x),__fadd_rn(preceding.y,v.y),__fadd_rn(preceding.z,v.z),__fadd_rn(preceding.w,v.w));}
    const float4 tail=p[i];
    reinterpret_cast<float4 *>(output)[i]=make_float4(__fadd_rn(tail.x,preceding.x),__fadd_rn(tail.y,preceding.y),__fadd_rn(tail.z,preceding.z),__fadd_rn(tail.w,preceding.w));
}
}
#include "small_prefill_q4_launch.cuh"
#include "small_prefill_q4_group_launch.cuh"
#include "small_prefill_q4_small_pair.cuh"
}
