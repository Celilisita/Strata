// Public typed entrypoints; included inside mmq namespace after the private core.
void validate_blackwell_down_int8_views(int type,const void* q8,void* controls,size_t control_bytes,
    void* packed,size_t packed_bytes,float* output,int entries,int source_rows,void* stream) {
    if((type!=12 && type!=13) || !stream || source_rows<1 || source_rows>32768 || entries!=source_rows ||
        !q8 || !controls || !packed || !output || !control_bytes || uintptr_t(q8)%4 ||
        uintptr_t(controls)%16 || uintptr_t(packed)%16 || uintptr_t(output)%4)
        throw std::invalid_argument("invalid compact Down INT8 workspace");
    const size_t required=blackwell_input_bytes(type,512,source_rows);
    if(packed_bytes<required) throw std::invalid_argument("short Down INT8 workspace");
    gu_int8_spans({{{uintptr_t(q8),size_t(source_rows)*16*sizeof(Q81Block)},
        {uintptr_t(controls),control_bytes},{uintptr_t(packed),required},
        {uintptr_t(output),size_t(entries)*2048*sizeof(float)}}});
}
NativeMmqEncodedInput encode_blackwell_down_int8_input(int type,const void* q8,void* packed,
    size_t packed_bytes,int source_rows,const int32_t* source_ids,void* stream) {
    if((type!=12 && type!=13) || !stream || source_rows<1 || source_rows>32768 || !q8 || !packed ||
        uintptr_t(q8)%4 || uintptr_t(packed)%16)
        throw std::invalid_argument("invalid Down INT8 encoder");
    const size_t required=blackwell_input_bytes(type,512,source_rows);
    if(packed_bytes<required) throw std::invalid_argument("short Down INT8 encoder workspace");
    gu_int8_spans({{{uintptr_t(q8),size_t(source_rows)*16*sizeof(Q81Block)},
        {uintptr_t(packed),required},{0,0},{0,0}}});
    auto* meta=reinterpret_cast<GroupInputMeta*>(static_cast<unsigned char*>(packed)+size_t(source_rows)*1024);
    const int cells=source_rows*16;
    encode_dense_group_int8<<<(cells+127)/128,128,0,static_cast<cudaStream_t>(stream)>>>(
        static_cast<const Q81Block*>(q8),static_cast<unsigned*>(packed),meta,512,source_rows);
    CUDA_CHECK(cudaGetLastError());
    return {packed,meta,source_ids,source_rows};
}
void run_blackwell_down_int8(int type,const NativeMmqMatrix* matrices,int groups,const int32_t* ids,
    float* output,int entries,int max_rows,void* stream,const NativeMmqTile* tiles,
    int tile_count,const NativeMmqEncodedInput& encoded) {
    check_device();
    if((type!=12 && type!=13) || !matrices || !ids || !output || !stream || !encoded.codes || !encoded.metadata ||
        !encoded.source_ids || encoded.source_rows<1 || encoded.source_rows>32768 || entries!=encoded.source_rows ||
        max_rows<1 || max_rows>8192 || max_rows>entries || groups<1 || groups>256 ||
        (encoded.tile_rows!=64 && encoded.tile_rows!=128) || (tiles ? tile_count<1 || tile_count>1280 : tile_count!=0))
        throw std::invalid_argument("invalid sparse Down INT8 product");
    const std::array<std::pair<uintptr_t,size_t>,6> reads{{{uintptr_t(matrices),size_t(groups)*sizeof(NativeMmqMatrix)},
        {uintptr_t(ids),size_t(entries)*4},{uintptr_t(encoded.source_ids),size_t(entries)*4},
        {uintptr_t(encoded.codes),size_t(encoded.source_rows)*512},
        {uintptr_t(encoded.metadata),size_t(encoded.source_rows)*16*sizeof(GroupInputMeta)},
        {uintptr_t(tiles),size_t(tile_count)*sizeof(NativeMmqTile)}}};
    for(const auto [address,bytes]:reads) if(bytes)
        gu_int8_spans({{{address,bytes},{uintptr_t(output),size_t(entries)*8192},{0,0},{0,0}}});
    if(uintptr_t(output)%4 || uintptr_t(matrices)%8 || uintptr_t(ids)%4 || uintptr_t(encoded.source_ids)%4 ||
        uintptr_t(encoded.codes)%16 || uintptr_t(encoded.metadata)%8)
        throw std::invalid_argument("unaligned Down INT8 product view");
#define DOWN_INT8(F,T) if(tiles) launch_down_int8<F,T,true>(matrices,groups,ids,output,entries,max_rows,stream,tiles,tile_count,encoded); \
    else launch_down_int8<F,T,false>(matrices,groups,ids,output,entries,max_rows,stream,tiles,tile_count,encoded)
    if(type==12) { if(encoded.tile_rows==64) { DOWN_INT8(GroupQ4,64); } else { DOWN_INT8(GroupQ4,128); } }
    else { if(encoded.tile_rows==64) { DOWN_INT8(GroupQ5,64); } else { DOWN_INT8(GroupQ5,128); } }
#undef DOWN_INT8
    CUDA_CHECK(cudaGetLastError());
}
NativeBlackwellGroupResources blackwell_down_int8_resources(int type,int tile_rows,bool active) {
    check_device();
#define DOWN_RESOURCE(F,T) return active ? down_int8_resources<F,T,true>() : down_int8_resources<F,T,false>()
    if(type==12) { if(tile_rows==64) { DOWN_RESOURCE(GroupQ4,64); } if(tile_rows==128) { DOWN_RESOURCE(GroupQ4,128); } }
    if(type==13) { if(tile_rows==64) { DOWN_RESOURCE(GroupQ5,64); } if(tile_rows==128) { DOWN_RESOURCE(GroupQ5,128); } }
#undef DOWN_RESOURCE
    throw std::invalid_argument("Down INT8 resources require Q4/Q5 tile64 or128");
}
