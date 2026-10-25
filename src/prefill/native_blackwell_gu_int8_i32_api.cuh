// Keep the same finite product view contract as the existing I16 GU entry.
void run_blackwell_gu_int8_i32(const NativeMmqMatrix* matrices,int groups,const int32_t* ids,
    float* output,int entries,int max_rows,void* stream,const NativeMmqTile* tiles,
    int tile_count,const NativeMmqEncodedInput& encoded) {
    check_device();
    if(!matrices || !ids || !output || !stream || !encoded.codes || !encoded.metadata ||
        !encoded.source_ids || encoded.source_rows<1 || encoded.source_rows>4096 ||
        entries!=encoded.source_rows*8 || max_rows<1 || max_rows>8192 || max_rows>entries ||
        groups<1 || groups>256 || (encoded.tile_rows!=64 && encoded.tile_rows!=128) ||
        (tiles ? tile_count<1 || tile_count>1280 : tile_count!=0))
        throw std::invalid_argument("invalid paired sparse GU I32 product");
    const std::array<std::pair<uintptr_t,size_t>,6> reads{{{uintptr_t(matrices),size_t(groups)*sizeof(NativeMmqMatrix)},
        {uintptr_t(ids),size_t(entries)*4},{uintptr_t(encoded.source_ids),size_t(entries)*4},
        {uintptr_t(encoded.codes),size_t(encoded.source_rows)*2048},
        {uintptr_t(encoded.metadata),size_t(encoded.source_rows)*64*sizeof(GroupInputMeta)},
        {uintptr_t(tiles),size_t(tile_count)*sizeof(NativeMmqTile)}}};
    for(const auto [address,bytes]:reads) if(bytes)
        gu_int8_spans({{{address,bytes},{uintptr_t(output),size_t(entries)*4096},{0,0},{0,0}}});
    if(uintptr_t(output)%4 || uintptr_t(matrices)%8 || uintptr_t(ids)%4 ||
        uintptr_t(encoded.source_ids)%4 || uintptr_t(encoded.codes)%16 || uintptr_t(encoded.metadata)%8)
        throw std::invalid_argument("unaligned GU I32 product view");
    if(encoded.tile_rows==64) {
        if(tiles) launch_gu_int8_i32<64,true>(matrices,groups,ids,output,entries,max_rows,stream,tiles,tile_count,encoded);
        else launch_gu_int8_i32<64,false>(matrices,groups,ids,output,entries,max_rows,stream,tiles,tile_count,encoded);
    } else {
        if(tiles) launch_gu_int8_i32<128,true>(matrices,groups,ids,output,entries,max_rows,stream,tiles,tile_count,encoded);
        else launch_gu_int8_i32<128,false>(matrices,groups,ids,output,entries,max_rows,stream,tiles,tile_count,encoded);
    }
    CUDA_CHECK(cudaGetLastError());
}
NativeBlackwellGroupResources blackwell_gu_int8_i32_resources(int tile_rows,bool active) {
    check_device();
    if(tile_rows==64) return active ? gu_int8_i32_resources<64,true>() : gu_int8_i32_resources<64,false>();
    if(tile_rows==128) return active ? gu_int8_i32_resources<128,true>() : gu_int8_i32_resources<128,false>();
    throw std::invalid_argument("GU I32 resources require tile64 or128");
}
