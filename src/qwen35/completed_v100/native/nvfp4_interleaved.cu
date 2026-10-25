#include "strata/qwen35/completed_v100/native/original_fused.hpp"
#include "strata/qwen35/completed_v100/native/ffn.hpp"
#include <cstdint>
namespace strata::completed_v100::qwen35::native {
namespace {
__global__ void pack(const unsigned char* raw,unsigned char* output,int rows,int columns) {
    const size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(index>=size_t(rows)*columns/8)return;
    const size_t tile=index/32;
    const int row=int(tile/(columns/32))*8+int(index%32)/4;
    const int k=int(tile%(columns/32))*32+int(index%4)*8;
    const auto* source=raw+(size_t(row)*(columns/64)+k/64)*36;
    reinterpret_cast<unsigned*>(output)[index]=*reinterpret_cast<const unsigned*>(source+4+((k>>4)&3)*8+(k&8)/2);
    if(!(k&8))output[size_t(rows)*columns/2+tile*16+(row&7)*2+((k&31)/16)]=source[(k>>4)&3];
}
__global__ void unpack(const unsigned char* tiled,unsigned char* raw,int rows,int columns) {
    const size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(index>=size_t(rows)*columns/8)return;
    const size_t tile=index/32;
    const int row=int(tile/(columns/32))*8+int(index%32)/4;
    const int k=int(tile%(columns/32))*32+int(index%4)*8;
    auto* target=raw+(size_t(row)*(columns/64)+k/64)*36;
    *reinterpret_cast<unsigned*>(target+4+((k>>4)&3)*8+(k&8)/2)=reinterpret_cast<const unsigned*>(tiled)[index];
    if(!(k&8))target[(k>>4)&3]=tiled[size_t(rows)*columns/2+tile*16+(row&7)*2+((k&31)/16)];
}
void validate(const void* source,void* output,int rows,int columns) {
    if(!source||!output||source==output||rows<8||rows%8||columns<64||columns%64||
       reinterpret_cast<uintptr_t>(source)%4||reinterpret_cast<uintptr_t>(output)%4)
        throw std::invalid_argument("Lossless NVFP4 tile transport ownership/extent");
}
}
void nvfp4_tile_pack(const void* source,void* output,int rows,int columns,cudaStream_t stream) {
    validate(source,output,rows,columns);
    pack<<<unsigned((size_t(rows)*columns/8+255)/256),256,0,stream>>>(
        static_cast<const unsigned char*>(source),static_cast<unsigned char*>(output),rows,columns);
    check_cuda(cudaGetLastError(),"Lossless original NVFP4 tile pack");
}
void nvfp4_tile_unpack(const void* source,void* output,int rows,int columns,cudaStream_t stream) {
    validate(source,output,rows,columns);
    unpack<<<unsigned((size_t(rows)*columns/8+255)/256),256,0,stream>>>(
        static_cast<const unsigned char*>(source),static_cast<unsigned char*>(output),rows,columns);
    check_cuda(cudaGetLastError(),"Lossless original NVFP4 tile inverse");
}
}
