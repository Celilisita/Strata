#pragma once
void validate_blackwell_gu_int8_views(const void* q8,void* controls,size_t control_bytes,
    void* packed,size_t packed_bytes,float* output,int entries,int source_rows,void* stream) {
    if(!stream || source_rows<1 || source_rows>4096 || entries!=source_rows*8 ||
        !q8 || !controls || !packed || !output || !control_bytes || uintptr_t(q8)%4 ||
        uintptr_t(controls)%16 || uintptr_t(packed)%16 || uintptr_t(output)%4)
        throw std::invalid_argument("invalid compact GU INT8 workspace");
    const size_t required=blackwell_input_bytes(12,2048,source_rows);
    if(packed_bytes<required) throw std::invalid_argument("short GU INT8 workspace");
    gu_int8_spans({{{uintptr_t(q8),size_t(source_rows)*64*sizeof(Q81Block)},
        {uintptr_t(controls),control_bytes},{uintptr_t(packed),required},
        {uintptr_t(output),size_t(entries)*1024*sizeof(float)}}});
}
NativeMmqEncodedInput encode_blackwell_gu_int8_input(const void* q8,void* packed,size_t packed_bytes,
    int source_rows,const int32_t* source_ids,void* stream) {
    if(!stream || source_rows<1 || source_rows>4096 || !q8 || !packed ||
        uintptr_t(q8)%4 || uintptr_t(packed)%16)
        throw std::invalid_argument("invalid GU INT8 encoder");
    const size_t required=blackwell_input_bytes(12,2048,source_rows);
    if(packed_bytes<required) throw std::invalid_argument("short GU INT8 encoder workspace");
    gu_int8_spans({{{uintptr_t(q8),size_t(source_rows)*64*sizeof(Q81Block)},
        {uintptr_t(packed),required},{0,0},{0,0}}});
    auto* meta=reinterpret_cast<GroupInputMeta*>(static_cast<unsigned char*>(packed)+size_t(source_rows)*4096);
    const int cells=source_rows*64;
    encode_dense_group_int8<<<(cells+127)/128,128,0,static_cast<cudaStream_t>(stream)>>>(
        static_cast<const Q81Block*>(q8),static_cast<unsigned*>(packed),meta,2048,source_rows);
    CUDA_CHECK(cudaGetLastError());
    return {packed,meta,source_ids,source_rows};
}
void run_blackwell_gu_int8(const NativeMmqMatrix* matrices,int groups,const int32_t* ids,
    float* output,int entries,int max_rows,void* stream,const NativeMmqTile* tiles,
    int tile_count,const NativeMmqEncodedInput& encoded) {
    check_device();
    if(!matrices || !ids || !output || !stream || !encoded.codes || !encoded.metadata ||
        !encoded.source_ids || encoded.source_rows<1 || encoded.source_rows>4096 ||
        entries!=encoded.source_rows*8 || max_rows<1 || max_rows>8192 || max_rows>entries ||
        groups<1 || groups>256 || (encoded.tile_rows!=64 && encoded.tile_rows!=128) ||
        (tiles ? tile_count<1 || tile_count>1280 : tile_count!=0))
        throw std::invalid_argument("invalid paired sparse GU INT8 product");
    const std::array<std::pair<uintptr_t,size_t>,6> reads{{{uintptr_t(matrices),size_t(groups)*sizeof(NativeMmqMatrix)},
        {uintptr_t(ids),size_t(entries)*4},{uintptr_t(encoded.source_ids),size_t(entries)*4},
        {uintptr_t(encoded.codes),size_t(encoded.source_rows)*2048},
        {uintptr_t(encoded.metadata),size_t(encoded.source_rows)*64*sizeof(GroupInputMeta)},
        {uintptr_t(tiles),size_t(tile_count)*sizeof(NativeMmqTile)}}};
    for(const auto [address,bytes]:reads) if(bytes)
        gu_int8_spans({{{address,bytes},{uintptr_t(output),size_t(entries)*4096},{0,0},{0,0}}});
    if(uintptr_t(output)%4 || uintptr_t(matrices)%8 || uintptr_t(ids)%4 ||
        uintptr_t(encoded.source_ids)%4 || uintptr_t(encoded.codes)%16 || uintptr_t(encoded.metadata)%8)
        throw std::invalid_argument("unaligned GU INT8 product view");
    if(encoded.tile_rows==64) {
        if(tiles) launch_gu_int8<64,true>(matrices,groups,ids,output,entries,max_rows,stream,tiles,tile_count,encoded);
        else launch_gu_int8<64,false>(matrices,groups,ids,output,entries,max_rows,stream,tiles,tile_count,encoded);
    } else {
        if(tiles) launch_gu_int8<128,true>(matrices,groups,ids,output,entries,max_rows,stream,tiles,tile_count,encoded);
        else launch_gu_int8<128,false>(matrices,groups,ids,output,entries,max_rows,stream,tiles,tile_count,encoded);
    }
    CUDA_CHECK(cudaGetLastError());
}
NativeBlackwellGroupResources blackwell_gu_int8_resources(int tile_rows,bool active) {
    check_device();
    if(tile_rows==64) return active ? gu_int8_resources<64,true>() : gu_int8_resources<64,false>();
    if(tile_rows==128) return active ? gu_int8_resources<128,true>() : gu_int8_resources<128,false>();
    throw std::invalid_argument("GU INT8 resources require tile64 or128");
}
