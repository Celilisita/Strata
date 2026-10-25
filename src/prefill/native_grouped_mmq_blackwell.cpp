#include "native_grouped_mmq_state.hpp"
namespace strata::prefill::mmq {
NativeGroupedMmqWork NativeGroupedMmqProvider::run_input(int type, const unsigned long long* gate,
    const unsigned long long* up, const int32_t* source_ids, int entries, int source_rows, const void* q8,
    void* controls, size_t controls_bytes, void* packed, size_t packed_bytes,
    float* output, int cols, int outputs, bool active, const float* activation_gate, const float* activation_up,
    bool force_gu_int8, bool force_down_int8, bool force_gu_i32, bool force_gu_i64) {
    auto& p = *impl_;
    active = active || p.active_tiles;
    const bool fused = activation_gate || activation_up;
    int backend = 0;
#ifdef STRATA_BLACKWELL_MMQ
    backend = blackwell_mmq_backend(type,up ? 2 : 1);
#endif
    if (!backend && !native_code_sum_grouped_supported(type, cols, outputs))
        throw std::invalid_argument("unsupported private DP4A grouped MMQ device/shape");
    const char* panel=std::getenv("STRATA_BENCH_GROUP_PANEL_TOKENS");
    const bool gu_domain=type==12 && backend==6 && up && !fused && panel && !std::strcmp(panel,"64") && cols==2048 && outputs==512 &&
        source_rows>=1 && source_rows<=4096 && entries==source_rows*8;
    const bool gu_int8=gu_domain && (force_gu_int8 || (p.gu_int8 && source_rows>=1024));
    // Explicit old GU entry remains I16. Ordinary PP requires BOTH startup flags.
#ifdef STRATA_BLACKWELL_MMQ
    const bool gu_i32=gu_int8 && (force_gu_i32 || (!force_gu_int8 && p.gu_i32));
    // Only the measured 4096-row AUTO domain benefits from the wider I64 tile.
    const bool gu_i64=gu_i32 && (force_gu_i64 || (!force_gu_int8 && p.gu_i64 && source_rows==4096));
#endif
    const bool down_domain=(type==12 || type==13) && backend==6 && !up && !fused && !p.down_fused &&
        cols==512 && outputs==2048 && source_rows==entries && entries>=1 && entries<=32768;
    const bool down_int8=down_domain && (force_down_int8 || (p.down_int8 && entries%8==0 && entries/8>=1024));
    if(force_down_int8 && !down_domain) throw std::invalid_argument("explicit Down INT8 domain required");
    if((force_gu_int8 || force_gu_i32 || force_gu_i64) && !gu_domain) throw std::invalid_argument("explicit GU INT8 domain required");
    if (fused && (!activation_gate || !activation_up || q8 || up || backend!=6 ||
        (type!=12 && type!=13) || cols!=512 || outputs!=2048 || source_rows<1 || source_rows>65536))
        throw std::invalid_argument("unsupported fused Down provider input");
    const int tile_rows = native_mmq_group_rows(type, backend, entries);
    const auto plan = native_mmq_plan(gate, up, source_ids, entries, source_rows, active, tile_rows);
    if((gu_int8 || down_int8) && plan.max_rows>8192) throw std::invalid_argument("GU INT8 expert row bound");
    const size_t matrices_bytes = plan.matrices.size() * sizeof(NativeMmqMatrix);
    const size_t map_bytes = size_t(entries) * sizeof(int32_t);
    const size_t tile_bytes = plan.tiles.size() * sizeof(NativeMmqTile);
    const size_t control_bytes = matrices_bytes + 2 * map_bytes + tile_bytes;
    bool q5 = false;
#ifdef STRATA_BLACKWELL_MMQ
    q5 = type==13 && !up && backend==6;
#endif
    size_t required = q8_bytes(entries, cols);
#ifdef STRATA_BLACKWELL_MMQ
    if (backend) required = blackwell_input_bytes(type,cols,source_rows);
#endif
    if ((type != 12 && type != 14 && !q5) || !controls || !packed || (!fused && !q8) || !output || control_bytes > controls_bytes ||
        required > packed_bytes || cols < 512 || cols % 512 || outputs < 128 || outputs % 128)
        throw std::invalid_argument("unreserved bounded grouped MMQ workspace");
#ifdef STRATA_BLACKWELL_MMQ
    if(down_int8) validate_blackwell_down_int8_views(type,q8,controls,control_bytes,packed,packed_bytes,
        output,entries,source_rows,p.stream);
    if(gu_int8) validate_blackwell_gu_int8_views(q8,controls,control_bytes,packed,packed_bytes,
        output,entries,source_rows,p.stream); // Before control upload or any kernel.
#endif
    auto& lease = p.leases[p.next++ % p.leases.size()];
    if (lease.used) {
        const auto ready = cudaEventQuery(lease.done);
        if (ready == cudaErrorNotReady) throw std::runtime_error("grouped MMQ lease requires caller drain");
        ck(ready);
    }
    std::memcpy(lease.bytes, plan.matrices.data(), matrices_bytes);
    std::memcpy(lease.bytes + matrices_bytes, plan.input_ids.data(), map_bytes);
    std::memcpy(lease.bytes + matrices_bytes + map_bytes, plan.output_ids.data(), map_bytes);
    if (tile_bytes) std::memcpy(lease.bytes + matrices_bytes + 2 * map_bytes, plan.tiles.data(), tile_bytes);
    auto* device = static_cast<uint8_t*>(controls);
    try {
        ck(cudaMemcpyAsync(device, lease.bytes, control_bytes, cudaMemcpyHostToDevice, p.stream));
        const NativeMmqEncodedInput* encoded = nullptr;
#ifdef STRATA_BLACKWELL_MMQ
        NativeMmqEncodedInput view;
        if (backend) {
            auto* input_ids=reinterpret_cast<int32_t*>(device+matrices_bytes);
            view = down_int8 ? encode_blackwell_down_int8_input(type,q8,packed,packed_bytes,source_rows,input_ids,p.stream) :
                gu_int8 ? encode_blackwell_gu_int8_input(q8,packed,packed_bytes,source_rows,input_ids,p.stream) :
                fused ? encode_blackwell_swiglu_input(activation_gate,activation_up,packed,packed_bytes,
                type,cols,source_rows,input_ids,p.stream) :
                repack_blackwell_input(q8,packed,type,cols,source_rows,input_ids,backend,p.stream);
            view.tile_rows = tile_rows;
            encoded = &view;
        } else
#endif
            repack_native_q8_grouped(q8,packed,type,cols,source_rows,entries,
                reinterpret_cast<int32_t*>(device+matrices_bytes),p.stream);
#ifdef STRATA_BLACKWELL_MMQ
        if(down_int8) run_blackwell_down_int8(type,reinterpret_cast<NativeMmqMatrix*>(device),int(plan.matrices.size()),
            reinterpret_cast<int32_t*>(device+matrices_bytes+map_bytes),output,entries,plan.max_rows,p.stream,
            active ? reinterpret_cast<NativeMmqTile*>(device+matrices_bytes+2*map_bytes) : nullptr,
            int(plan.tiles.size()),*encoded);
        else if(gu_i64) run_blackwell_gu_int8_i64(reinterpret_cast<NativeMmqMatrix*>(device),int(plan.matrices.size()),
            reinterpret_cast<int32_t*>(device+matrices_bytes+map_bytes),output,entries,plan.max_rows,p.stream,
            active ? reinterpret_cast<NativeMmqTile*>(device+matrices_bytes+2*map_bytes) : nullptr,
            int(plan.tiles.size()),*encoded);
        else if(gu_i32) run_blackwell_gu_int8_i32(reinterpret_cast<NativeMmqMatrix*>(device),int(plan.matrices.size()),
            reinterpret_cast<int32_t*>(device+matrices_bytes+map_bytes),output,entries,plan.max_rows,p.stream,
            active ? reinterpret_cast<NativeMmqTile*>(device+matrices_bytes+2*map_bytes) : nullptr,
            int(plan.tiles.size()),*encoded);
        else if(gu_int8) run_blackwell_gu_int8(reinterpret_cast<NativeMmqMatrix*>(device),int(plan.matrices.size()),
            reinterpret_cast<int32_t*>(device+matrices_bytes+map_bytes),output,entries,plan.max_rows,p.stream,
            active ? reinterpret_cast<NativeMmqTile*>(device+matrices_bytes+2*map_bytes) : nullptr,
            int(plan.tiles.size()),*encoded);
        else
#endif
        native_grouped_mmq(type, reinterpret_cast<NativeMmqMatrix*>(device), int(plan.matrices.size()), up ? 2 : 1,
            reinterpret_cast<int32_t*>(device + matrices_bytes + map_bytes), packed, output,
            entries, plan.max_rows, cols, outputs, p.stream,
            active ? reinterpret_cast<NativeMmqTile*>(device + matrices_bytes + 2 * map_bytes) : nullptr, int(plan.tiles.size()), encoded);
        ck(cudaEventRecord(lease.done, p.stream)); lease.used = true;
    } catch (...) {
        const auto e = cudaEventRecord(lease.done, p.stream); lease.used = e == cudaSuccess;
        if (e != cudaSuccess) cudaStreamSynchronize(p.stream);
        throw;
    }
    return {control_bytes, required, int(plan.matrices.size()), backend ? 2 : 3,
        active ? plan.active_count : ((plan.max_rows + tile_rows - 1) / tile_rows) * int(plan.matrices.size()), plan.active_count, plan.max_rows, ((plan.max_rows + tile_rows - 1) / tile_rows) * int(plan.matrices.size()), entries}; // host plan cost
}
#include "native_grouped_gu_pipeline.hpp"
} // namespace strata::prefill::mmq
