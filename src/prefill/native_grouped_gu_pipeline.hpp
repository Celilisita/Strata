// Included in the existing provider TU: same four pinned control leases.
NativeGroupedMmqWork NativeGroupedMmqProvider::run_gu_pipeline(
    const unsigned long long* gate,const unsigned long long* up,const int32_t* source_ids,
    int entries,int source_rows,const void* q8,void* controls,size_t controls_bytes,
    void* packed,size_t packed_bytes,float* output,int matrices_per_wave,
    const std::function<void(const NativeMmqMatrix*,int,bool)>& ready) {
#ifndef STRATA_BLACKWELL_MMQ
    (void)gate;(void)up;(void)source_ids;(void)entries;(void)source_rows;(void)q8;
    (void)controls;(void)controls_bytes;(void)packed;(void)packed_bytes;(void)output;
    (void)matrices_per_wave;(void)ready;
    throw std::invalid_argument("GU copy pipeline requires Blackwell provider");
#else
    auto& p=*impl_;const char* panel=std::getenv("STRATA_BENCH_GROUP_PANEL_TOKENS");
    if(!p.gu_int8||!p.gu_i32||!up||!ready||!panel||std::strcmp(panel,"64")||
       source_rows<1024||source_rows>4096||entries!=source_rows*8||
       (matrices_per_wave!=32&&matrices_per_wave!=64)||blackwell_mmq_backend(12,2)!=6)
        throw std::invalid_argument("GU copy pipeline requires actual PP INT8/I32 domain");
    const int tile_rows=native_mmq_group_rows(12,6,entries);
    const auto plan=native_mmq_plan(gate,up,source_ids,entries,source_rows,true,tile_rows);
    if(plan.max_rows>8192)throw std::invalid_argument("GU pipeline expert row bound");
    const size_t matrix_bytes=plan.matrices.size()*sizeof(NativeMmqMatrix),map_bytes=size_t(entries)*4;
    const size_t tile_bytes=plan.tiles.size()*sizeof(NativeMmqTile),control_bytes=matrix_bytes+2*map_bytes+tile_bytes;
    const size_t required=blackwell_input_bytes(12,2048,source_rows);
    if(!controls||!packed||!q8||!output||control_bytes>controls_bytes||required>packed_bytes||
       control_bytes>control_capacity||plan.tiles.empty())
        throw std::invalid_argument("unreserved GU pipeline workspace");
    validate_blackwell_gu_int8_views(q8,controls,control_bytes,packed,packed_bytes,output,entries,source_rows,p.stream);
    auto& lease=p.leases[p.next++%p.leases.size()];
    if(lease.used){const auto status=cudaEventQuery(lease.done);
        if(status==cudaErrorNotReady)throw std::runtime_error("GU pipeline lease requires caller drain");ck(status);}
    std::memcpy(lease.bytes,plan.matrices.data(),matrix_bytes);
    std::memcpy(lease.bytes+matrix_bytes,plan.input_ids.data(),map_bytes);
    std::memcpy(lease.bytes+matrix_bytes+map_bytes,plan.output_ids.data(),map_bytes);
    std::memcpy(lease.bytes+matrix_bytes+2*map_bytes,plan.tiles.data(),tile_bytes);
    auto* device=static_cast<uint8_t*>(controls);int waves=0;
    try{
        ck(cudaMemcpyAsync(device,lease.bytes,control_bytes,cudaMemcpyHostToDevice,p.stream));
        auto encoded=encode_blackwell_gu_int8_input(q8,packed,packed_bytes,source_rows,
            reinterpret_cast<int32_t*>(device+matrix_bytes),p.stream);encoded.tile_rows=tile_rows;
        size_t tile_begin=0;
        for(size_t first=0;first<plan.matrices.size();first+=size_t(matrices_per_wave)){
            const size_t end=(std::min)(first+size_t(matrices_per_wave),plan.matrices.size());
            size_t tile_end=tile_begin;
            while(tile_end<plan.tiles.size()&&size_t(plan.tiles[tile_end].matrix)<end)++tile_end;
            if(tile_end==tile_begin)throw std::runtime_error("empty GU matrix wave");
            ready(plan.matrices.data()+first,int(end-first),end==plan.matrices.size());
            const auto* tiles=reinterpret_cast<NativeMmqTile*>(device+matrix_bytes+2*map_bytes)+tile_begin;
            // Preserve the measured ordinary I64 domain; all shorter PP uses I32.
            if(p.gu_i64&&source_rows==4096)run_blackwell_gu_int8_i64(reinterpret_cast<NativeMmqMatrix*>(device),
                int(plan.matrices.size()),reinterpret_cast<int32_t*>(device+matrix_bytes+map_bytes),output,entries,
                plan.max_rows,p.stream,tiles,int(tile_end-tile_begin),encoded);
            else run_blackwell_gu_int8_i32(reinterpret_cast<NativeMmqMatrix*>(device),int(plan.matrices.size()),
                reinterpret_cast<int32_t*>(device+matrix_bytes+map_bytes),output,entries,plan.max_rows,p.stream,
                tiles,int(tile_end-tile_begin),encoded);
            tile_begin=tile_end;++waves;
        }
        if(tile_begin!=plan.tiles.size())throw std::runtime_error("GU pipeline tile coverage");
        ck(cudaEventRecord(lease.done,p.stream));lease.used=true;
    }catch(...){const auto status=cudaEventRecord(lease.done,p.stream);lease.used=status==cudaSuccess;
        if(status!=cudaSuccess)cudaStreamSynchronize(p.stream);throw;}
    return {control_bytes,required,int(plan.matrices.size()),1+waves,plan.active_count,plan.active_count};
#endif
}
