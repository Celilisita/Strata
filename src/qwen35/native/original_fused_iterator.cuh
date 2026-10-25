#pragma once
#include "original_fused_codec.cuh"
#include <cutlass/cutlass.h>

namespace strata::qwen35::native::original_fused {
// The CUTLASS thread map, fragments, shared-memory iterators and MMA stay
// unchanged. Only the global B view decodes the original compressed payload.
template<class Underlying,bool Q8,bool Tiled=false>
class QuantizedB {
public:
    using Shape=typename Underlying::Shape;
    using Element=typename Underlying::Element;
    using Layout=typename Underlying::Layout;
    using TensorRef=typename Underlying::TensorRef;
    using TensorCoord=typename Underlying::TensorCoord;
    using AccessType=typename Underlying::AccessType;
    using Fragment=typename Underlying::Fragment;
    using ThreadMap=typename Underlying::ThreadMap;
    struct Params {
        int columns=0;
        CUTLASS_HOST_DEVICE Params()=default;
        CUTLASS_HOST_DEVICE Params(Layout const& layout):columns(int(layout.stride(0))){}
    };
    static_assert(ThreadMap::kElementsPerAccess==8,"quantized loader needs eight aligned K values");
    CUTLASS_DEVICE QuantizedB(Params const& p,Element* pointer,TensorCoord extent,
        int thread,TensorCoord const& block,int const* =nullptr):
        raw_(reinterpret_cast<const unsigned char*>(pointer)),columns_(p.columns),extent_(extent) {
        const auto initial=ThreadMap::initial_offset(thread);
        k_=block.row()+initial.contiguous();row_=block.column()+initial.strided();
    }
    CUTLASS_DEVICE QuantizedB& operator++(){k_+=Shape::kRow;return *this;}
    CUTLASS_DEVICE void clear_mask(bool enable=true){if(enable)active_=false;}
    CUTLASS_DEVICE void load(Fragment& fragment) {
        fragment.clear();if(!active_)return;
        CUTLASS_PRAGMA_UNROLL
        for(int s=0;s<ThreadMap::Iterations::kStrided;++s) {
            const int row=row_+s*ThreadMap::Delta::kStrided;
            CUTLASS_PRAGMA_UNROLL
            for(int c=0;c<ThreadMap::Iterations::kContiguous;++c) {
                const int k=k_+c*ThreadMap::Delta::kContiguous;
                if(row>=extent_.column()||k>=extent_.row())continue;
                auto* out=reinterpret_cast<half*>(fragment.data())+
                    (s*ThreadMap::Iterations::kContiguous+c)*8;
                if constexpr(Tiled)nvfp4_tiled_eight(raw_,columns_,extent_.column(),row,k,out);
                else if constexpr(Q8)q8_eight(raw_,columns_,row,k,out);
                else nvfp4_eight(raw_,columns_,row,k,out);
            }
        }
    }
private:
    const unsigned char* raw_;
    int columns_=0,k_=0,row_=0;
    TensorCoord extent_;
    bool active_=true;
};
// Parallel K partitions use disjoint D tiles. The NVIDIA kernel handles its
// K ranges; no atomics, semaphores or in-kernel private matrix reduction.
template<class Underlying>
class PartialOutput:public Underlying {
public:
    using Layout=typename Underlying::Layout;
    using TensorCoord=typename Underlying::TensorCoord;
    struct Params:Underlying::Params {
        long long partition_stride=0;
        CUTLASS_HOST_DEVICE Params()=default;
        CUTLASS_HOST_DEVICE Params(Layout const& layout):Underlying::Params(layout){}
    };
    CUTLASS_DEVICE PartialOutput(Params const& p,float* pointer,TensorCoord extent,
        int thread,TensorCoord block=TensorCoord(),int const* indices=nullptr):
        Underlying(p,pointer+size_t(blockIdx.z)*p.partition_stride,extent,thread,block,indices){}
};
}
