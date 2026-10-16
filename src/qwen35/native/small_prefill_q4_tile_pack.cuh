#pragma once
// Included in the owned CUDA translation unit; byte copies have no FP math.
namespace {
__global__ void tile_q4_bytes(Matrix source,uint4 *destination){
    const size_t blocks=source.columns/256,headers=size_t(source.rows)*blocks;
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=headers*9)return;
    size_t tile_block,row,offset;
    if(i<headers){tile_block=i/64;row=i%64;offset=0;}
    else{
        const size_t payload=i-headers;
        const size_t raw_block=payload/8;
        const size_t raw_row=raw_block/blocks;
        tile_block=(raw_row/64)*blocks+raw_block%blocks;row=raw_row%64;
        offset=16+(payload%8)*16;
    }
    const size_t raw_row=(tile_block/blocks)*64+row,block=tile_block%blocks;
    const auto *p=static_cast<const unsigned char *>(source.data)+(raw_row*blocks+block)*144+offset;
    destination[i]=__ldg(reinterpret_cast<const uint4 *>(p));
}
__device__ inline const unsigned char *tile_q4_header(const unsigned char *base,int tile,int blocks,int block,int row){
    return base+(size_t(tile)*blocks+block)*64*16+row*16;
}
__device__ inline const unsigned char *tile_q4_payload(const unsigned char *base,int rows,int blocks,int tile,int block,int chunk,int row){
    return base+size_t(rows)*blocks*16+(size_t(tile*64+row)*blocks+block)*128+chunk*64;
}
}
Q4TileCodes prepare_q4_tile_codes(Matrix source,void *destination,cudaStream_t stream){
    const size_t bytes=matrix_bytes(source);
    if(source.quant!=Quant::Q4K||source.layout!=BlockLayout::Q4Pairs||source.rows%64||
       reinterpret_cast<uintptr_t>(source.data)%16||!destination||reinterpret_cast<uintptr_t>(destination)%16||
       detail::split_overlap(source.data,bytes,destination,bytes))
        throw std::invalid_argument("tiled Q4 preparation requires distinct aligned raw-pair buffers");
    tile_q4_bytes<<<unsigned((bytes/16+255)/256),256,0,stream>>>(source,static_cast<uint4 *>(destination));
    check_cuda(cudaGetLastError(),"cold lossless tiled Q4 copy");
    return {destination,source.rows,source.columns,bytes};
}
