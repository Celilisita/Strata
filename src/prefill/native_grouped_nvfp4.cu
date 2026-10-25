#include "strata/prefill/native_grouped_nvfp4.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <limits>
#include <stdexcept>
#include "nvfp4_group_kernel.cuh"
namespace strata::prefill::mmq {
namespace {
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void geometry(int rows,int columns){
    if(rows<1||rows>65536||columns<64||columns>8192||columns%64)
        throw std::invalid_argument("bounded NVFP4 input geometry");
}
void span(const void* p,size_t bytes,size_t alignment){
    const auto a=reinterpret_cast<uintptr_t>(p);
    if(!p||a%alignment||bytes>(std::numeric_limits<uintptr_t>::max)()-a)
        throw std::invalid_argument("NVFP4 view alignment/overflow");
}
void separate(const void* a,size_t an,const void* b,size_t bn){
    const auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);
    if(x<y+bn&&y<x+an)throw std::invalid_argument("NVFP4 view alias");
}
void sm120(){int device=0,major=0,minor=0;ck(cudaGetDevice(&device));
    ck(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,device));
    ck(cudaDeviceGetAttribute(&minor,cudaDevAttrComputeCapabilityMinor,device));
    if(major!=12||minor!=0)throw std::invalid_argument("grouped NVFP4 requires SM120");}
}
size_t native_nvfp4_input_bytes(int rows,int columns){geometry(rows,columns);return size_t(rows)*(columns/2+columns/16+4);}
NativeNvfp4Input encode_native_nvfp4(const float* input,void* packed,size_t bytes,int rows,int columns,
                                    const int32_t* ids,void* stream){
    const size_t need=native_nvfp4_input_bytes(rows,columns);if(!stream||!ids||bytes<need)throw std::invalid_argument("NVFP4 reserved input workspace");
    span(input,size_t(rows)*columns*4,4);span(packed,need,16);span(ids,4,4);
    separate(input,size_t(rows)*columns*4,packed,need);
    auto* q=static_cast<uint8_t*>(packed);auto* sf=q+size_t(rows)*(columns/2);
    auto* global=reinterpret_cast<float*>(sf+size_t(rows)*(columns/16));
    nvfp4_detail::encode<<<rows,128,0,static_cast<cudaStream_t>(stream)>>>(input,q,sf,global,columns,nullptr);ck(cudaGetLastError());
    return{q,sf,global,ids,rows,columns};
}
NativeNvfp4Input encode_native_nvfp4_masked(const float* input,void* packed,size_t bytes,int rows,int columns,
    const int32_t* ids,const unsigned long long* active,void* stream) {
    const size_t need=native_nvfp4_input_bytes(rows,columns);
    if(!stream||bytes<need)throw std::invalid_argument("NVFP4 masked reserved workspace");
    span(input,size_t(rows)*columns*4,4);span(packed,need,16);span(ids,size_t(rows)*4,4);span(active,size_t(rows)*8,8);
    separate(input,size_t(rows)*columns*4,packed,need);separate(active,size_t(rows)*8,packed,need);
    auto* q=static_cast<uint8_t*>(packed);auto* sf=q+size_t(rows)*(columns/2);
    auto* global=reinterpret_cast<float*>(sf+size_t(rows)*(columns/16));
    nvfp4_detail::encode<<<rows,128,0,static_cast<cudaStream_t>(stream)>>>(input,q,sf,global,columns,active);ck(cudaGetLastError());
    return{q,sf,global,ids,rows,columns};
}
void native_grouped_nvfp4(const NativeNvfp4Matrix* matrices,int groups,int parts,const int32_t* output_ids,
    float* output,int entries,int max_rows,int outputs,const NativeMmqTile* tiles,int tile_count,int tile_rows,
    NativeNvfp4Input input,void* stream){
    geometry(input.source_rows,input.columns);
    if(!stream||groups<1||groups>256||(parts!=1&&parts!=2)||entries<1||entries>65536||max_rows<1||max_rows>entries||
       outputs<16||outputs>8192||outputs%16||tile_count<1||tile_count>(entries+31)/32+256||(tile_rows!=8&&tile_rows!=32&&tile_rows!=64))
        throw std::invalid_argument("NVFP4 grouped product geometry");
    if(tile_rows==8 && (max_rows!=1 || groups!=entries || tile_count!=groups))
        throw std::invalid_argument("NVFP4 tile8 is reserved for one-row expert matrices");
    span(matrices,size_t(groups)*sizeof(NativeNvfp4Matrix),8);span(output_ids,size_t(entries)*4,4);
    span(tiles,size_t(tile_count)*sizeof(NativeMmqTile),4);span(output,size_t(parts)*entries*outputs*4,4);
    span(input.values,size_t(input.source_rows)*input.columns/2,16);
    span(input.scales,size_t(input.source_rows)*input.columns/16,4);span(input.globals,size_t(input.source_rows)*4,4);
    span(input.source_ids,size_t(entries)*4,4);
    const size_t out_bytes=size_t(parts)*entries*outputs*4;
    separate(output,out_bytes,matrices,size_t(groups)*sizeof(NativeNvfp4Matrix));
    separate(output,out_bytes,output_ids,size_t(entries)*4);separate(output,out_bytes,tiles,size_t(tile_count)*sizeof(NativeMmqTile));
    separate(output,out_bytes,input.values,size_t(input.source_rows)*input.columns/2);
    separate(output,out_bytes,input.scales,size_t(input.source_rows)*input.columns/16);
    separate(output,out_bytes,input.globals,size_t(input.source_rows)*4);separate(output,out_bytes,input.source_ids,size_t(entries)*4);
    const dim3 grid((outputs+63)/64,tile_count,parts);auto s=static_cast<cudaStream_t>(stream);
    if(tile_rows==8)nvfp4_detail::product<8><<<grid,128,0,s>>>(matrices,output_ids,output,entries,outputs,tiles,input);
    else if(tile_rows==32)nvfp4_detail::product<32><<<grid,128,0,s>>>(matrices,output_ids,output,entries,outputs,tiles,input);
    else nvfp4_detail::product<64><<<grid,128,0,s>>>(matrices,output_ids,output,entries,outputs,tiles,input);
    ck(cudaGetLastError());
}
NativeNvfp4Resources native_nvfp4_resources(int rows){
    if(rows!=8&&rows!=32&&rows!=64)throw std::invalid_argument("NVFP4 resource tile8/32/64");sm120();
    const auto kernel=rows==8?nvfp4_detail::product<8>:rows==32?nvfp4_detail::product<32>:nvfp4_detail::product<64>;
    cudaFuncAttributes a{};int blocks=0;ck(cudaFuncGetAttributes(&a,kernel));
    ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel,128,0));
    return{a.numRegs,128,blocks,a.sharedSizeBytes,a.localSizeBytes};
}
}
