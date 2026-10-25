#include "fp4_unpack_vector.hpp"
#include <cstdint>
#include <stdexcept>
#include "strata/qwen35/native/ffn.hpp"
namespace strata::qwen35::native {
namespace {
__device__ void exchange(unsigned &a,unsigned &b,int lane,int bit){
    const unsigned selected=(lane&bit)?a:b;
    const unsigned other=__shfl_xor_sync(0xffffffff,selected,bit);
    if(lane&bit)a=other;else b=other;
}
__global__ void unpack(const uint4 *input,uint4 *output,int rows,int columns){
    const size_t thread=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const size_t tile=thread/32;if(tile>=size_t(rows/16)*(columns/64))return;
    const int lane=threadIdx.x&31;
    const uint4 p=input[thread];unsigned q0=p.x,q1=p.z,q2=p.y,q3=p.w;
    exchange(q0,q2,lane,2);exchange(q1,q3,lane,2);
    exchange(q0,q1,lane,1);exchange(q2,q3,lane,1);
    const int row=int(tile/(columns/64))*16+lane/4+((lane&2)<<2);
    const int first=int(tile%(columns/64))*64;
    output[size_t(row)*(columns/32)+first/32+(lane&1)]={q0,q1,q2,q3};
}
}
void fp4_unpack_vector(PackedFp4 p,void *destination,cudaStream_t stream){
    const auto w=p.view;
    if(!w.values||!destination||w.rows<16||w.rows%16||w.columns<64||w.columns%64||
       reinterpret_cast<uintptr_t>(w.values)%16||reinterpret_cast<uintptr_t>(destination)%16)
        throw std::invalid_argument("vector unpack geometry/alignment");
    const size_t bytes=size_t(w.rows)*w.columns/2;
    const auto a=reinterpret_cast<uintptr_t>(w.values),b=reinterpret_cast<uintptr_t>(destination);
    if((a<=b?b-a:a-b)<bytes)throw std::invalid_argument("vector unpack overlapping planes");
    const size_t threads=size_t(w.rows/16)*(w.columns/64)*32;
    unpack<<<(threads+127)/128,128,0,stream>>>(static_cast<const uint4*>(w.values),static_cast<uint4*>(destination),w.rows,w.columns);
    check_cuda(cudaGetLastError(),"vector FP4 canonical materialization");
}
}
