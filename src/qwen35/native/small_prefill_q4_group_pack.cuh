#pragma once
namespace {
__host__ __device__ inline size_t group_q4_source_word(size_t i,int rows,int blocks){
    const size_t headers=size_t(rows)*blocks;
    size_t tile_block,row,word;
    if(i<headers){tile_block=i/64;row=i%64;word=0;}
    else{
        const size_t payload=i-headers;
        tile_block=payload/512;
        const size_t within=payload%512;
        const size_t pair=within/128;
        row=(within%128)/2;word=1+pair*2+within%2;
    }
    const size_t raw_row=(tile_block/blocks)*64+row,block=tile_block%blocks;
    return (raw_row*blocks+block)*9+word;
}
__global__ void group_q4_bytes(Matrix source,uint4 *destination){
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const size_t words=size_t(source.rows)*(source.columns/256)*9;
    if(i<words)destination[i]=__ldg(static_cast<const uint4 *>(source.data)+group_q4_source_word(i,source.rows,source.columns/256));
}
__global__ void verify_group_q4_bytes(Matrix source,Q4GroupCodes view,unsigned long long *errors){
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=view.bytes/16)return;
    // Independent inverse mapping: traverse original row-major words.
    const size_t blocks=source.columns/256,raw_block=i/9,word=i%9;
    const size_t raw_row=raw_block/blocks,block=raw_block%blocks,row=raw_row%64;
    const size_t tile_block=(raw_row/64)*blocks+block,headers=size_t(source.rows)*blocks;
    const size_t mapped=word==0?tile_block*64+row:
        headers+tile_block*512+((word-1)/2)*128+row*2+(word-1)%2;
    const uint4 expected=__ldg(static_cast<const uint4 *>(source.data)+i);
    const uint4 actual=__ldg(static_cast<const uint4 *>(view.data)+mapped);
    if(expected.x!=actual.x||expected.y!=actual.y||expected.z!=actual.z||expected.w!=actual.w)atomicAdd(errors,1ULL);
}
}
Q4GroupCodes prepare_q4_group_codes(Matrix source,void *destination,cudaStream_t stream){
    const size_t bytes=matrix_bytes(source);
    if(source.quant!=Quant::Q4K||source.layout!=BlockLayout::Q4Pairs||source.rows%64||
       reinterpret_cast<uintptr_t>(source.data)%16||!destination||reinterpret_cast<uintptr_t>(destination)%16||
       detail::split_overlap(source.data,bytes,destination,bytes))
        throw std::invalid_argument("group Q4 preparation requires distinct aligned raw-pair buffers");
    group_q4_bytes<<<unsigned((bytes/16+255)/256),256,0,stream>>>(source,static_cast<uint4 *>(destination));
    check_cuda(cudaGetLastError(),"cold lossless group Q4 copy");
    return {destination,source.rows,source.columns,bytes};
}
void verify_q4_group_codes(Matrix source,Q4GroupCodes view,void *scratch,cudaStream_t stream){
    const size_t bytes=q4_group_codes_bytes(view);
    if(matrix_bytes(source)!=bytes||source.quant!=Quant::Q4K||source.layout!=BlockLayout::Q4Pairs||
       source.rows!=view.rows||source.columns!=view.columns||!scratch||reinterpret_cast<uintptr_t>(scratch)%8||
       detail::split_overlap(scratch,8,source.data,bytes)||detail::split_overlap(scratch,8,view.data,bytes))
        throw std::invalid_argument("group Q4 diagnostic geometry or scratch mismatch");
    check_cuda(cudaMemsetAsync(scratch,0,8,stream),"initialize exact payload diagnostic");
    verify_group_q4_bytes<<<unsigned((bytes/16+255)/256),256,0,stream>>>(source,view,static_cast<unsigned long long *>(scratch));
    check_cuda(cudaGetLastError(),"verify exact group Q4 payload");
    unsigned long long errors=0;
    check_cuda(cudaMemcpyAsync(&errors,scratch,8,cudaMemcpyDeviceToHost,stream),"read group Q4 byte errors");
    check_cuda(cudaStreamSynchronize(stream),"exact group Q4 diagnostic completion");
    if(errors)throw std::runtime_error("lossless group Q4 payload mismatch");
}
