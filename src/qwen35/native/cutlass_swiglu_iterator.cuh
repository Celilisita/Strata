#pragma once
#include "math.cuh"
#include <cuda_fp16.h>

namespace strata::qwen35::native::cutlass_detail {
// CUTLASS's logical float columns are [G0,U0,G1,U1,...]. The owned Gate/Up
// pool supplies a valid logical view but is untouched during normal execution.
// Each float4 fragment becomes two half activations in a separate live arena.
template<class Underlying,bool Debug=true>
class SwiGluIterator:public Underlying {
    using Map=typename Underlying::ThreadMap;
public:
    using Layout=typename Underlying::Layout;
    using Fragment=typename Underlying::Fragment;
    using TensorCoord=typename Underlying::TensorCoord;
    using LongIndex=typename Underlying::LongIndex;
    using ParameterBase=typename Underlying::Params;
    struct Params:ParameterBase {
        half *half_output=nullptr;
        float *gate=nullptr,*up=nullptr;
        LongIndex logical_stride=0;
        CUTLASS_HOST_DEVICE Params()=default;
        CUTLASS_HOST_DEVICE Params(Layout const &layout):ParameterBase(layout),logical_stride(layout.stride(0)){}
    };
    static_assert(Underlying::kElementsPerAccess==4,"SwiGLU pairs require four float columns");
    static_assert(Map::Delta::kColumn%4==0,"SwiGLU vector columns must stay aligned");
    CUTLASS_DEVICE SwiGluIterator(Params const &p,float *logical_view,TensorCoord extent,
        int thread,TensorCoord tile=TensorCoord(),int const *indices=nullptr)
        :Underlying(p,logical_view,extent,thread,tile,indices),parameters_(p){}
    CUTLASS_DEVICE void add_pointer_offset(LongIndex offset){
        Underlying::add_pointer_offset(offset);logical_offset_+=offset;
    }
    CUTLASS_DEVICE void store_with_byte_offset(Fragment const &fragment,int64_t bytes)const{
        typename Underlying::Mask mask;Underlying::get_mask(mask);
        CUTLASS_PRAGMA_UNROLL
        for(int cluster=0;cluster<Map::Iterations::kCluster;++cluster){
            CUTLASS_PRAGMA_UNROLL
            for(int group=0;group<Map::Iterations::kGroup;++group){
                CUTLASS_PRAGMA_UNROLL
                for(int row=0;row<Map::Iterations::kRow;++row){
                    const int logical_row=Underlying::thread_start_row()+row*Map::Delta::kRow+
                        group*Map::Delta::kGroup+cluster*Map::Delta::kCluster;
                    const int fragment_row=row+Map::Iterations::kRow*(group+Map::Iterations::kGroup*cluster);
                    CUTLASS_PRAGMA_UNROLL
                    for(int column=0;column<Map::Iterations::kColumn;++column){
                        if(logical_row>=Underlying::extent_row()||!mask.predicates[column])continue;
                        const int logical_column=Underlying::thread_start_column()+column*Map::Delta::kColumn;
                        const int at=(fragment_row*Map::Iterations::kColumn+column)*4;
                        const float g0=fragment[at],u0=fragment[at+1],g1=fragment[at+2],u1=fragment[at+3];
                        const LongIndex logical=LongIndex(logical_row)*parameters_.logical_stride+logical_column+
                            logical_offset_+bytes/int(sizeof(float));
                        reinterpret_cast<half2 *>(parameters_.half_output)[logical/4]=
                            __floats2half2_rn(swiglu_value(g0,u0),swiglu_value(g1,u1));
                        if constexpr(Debug) if(parameters_.gate){
                            const LongIndex feature=logical/2;
                            parameters_.gate[feature]=g0;parameters_.gate[feature+1]=g1;
                            parameters_.up[feature]=u0;parameters_.up[feature+1]=u1;
                        }
                    }
                }
            }
        }
    }
    CUTLASS_DEVICE void store(Fragment const &fragment)const{store_with_byte_offset(fragment,0);}
private:
    Params parameters_;
    LongIndex logical_offset_=0;
};
}
