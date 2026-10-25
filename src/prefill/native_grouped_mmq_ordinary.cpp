#include "native_grouped_mmq_state.hpp"
namespace strata::prefill::mmq {
bool NativeGroupedMmqProvider::try_run_down_f32(int type,const unsigned long long* down,const int32_t* source_ids,
    int entries,int source_rows,const float* input,void* controls,size_t controls_bytes,
    void* packed,size_t packed_bytes,float* output,NativeGroupedMmqWork& work) {
    // Only the actual PP workset; unsupported/default-off means false before lease or device writes.
    if(!down_direct_packed_enabled() || (type!=12&&type!=13) || entries<8192 || entries>16384 || entries%8 ||
        source_rows!=entries || !down || !source_ids || !input || !controls || !packed || !output ||
        !ordinary_enabled() || (type==13&&!q5_down_enabled()) ||
        !native_ordinary_grouped_supported(type,512,2048,1,false) || !native_down_direct_packed_resources().eligible)
        return false;
    for(int i=0;i<entries;++i)if(source_ids[i]!=i || !down[i])return false;
    const auto plan=native_mmq_plan(down,nullptr,source_ids,entries,source_rows,false);
    if(plan.input_ids!=plan.output_ids||!plan.tiles.empty())return false; // Identity source rows induce the same exact P.
    const size_t needed=q8_bytes(entries,512),cb=plan.matrices.size()*sizeof(NativeMmqMatrix)+size_t(entries)*8+plan.tiles.size()*sizeof(NativeMmqTile);
    if(needed>packed_bytes||cb>controls_bytes)return false;
    struct Span{uintptr_t first,last;};
    auto span=[](const void* ptr,size_t bytes){const auto a=reinterpret_cast<uintptr_t>(ptr);
        if(!a||a%4||bytes>std::numeric_limits<uintptr_t>::max()-a)return Span{0,0};return Span{a,a+bytes};};
    const std::array<Span,4> spans={span(input,size_t(entries)*512*4),span(controls,cb),span(packed,needed),span(output,size_t(entries)*2048*4)};
    for(size_t i=0;i<spans.size();++i){if(!spans[i].first)return false;
        for(size_t j=0;j<i;++j){
            // Same-stream producer consumes all F32 before MMQ can overwrite its output view.
            if(i==3&&j==0)continue; // Only input/output phase reuse; controls/packed stay disjoint.
            if(spans[i].first<spans[j].last&&spans[j].first<spans[i].last)return false;}}
    work=submit(plan,type,1,entries,source_rows,input,controls,controls_bytes,packed,packed_bytes,output,512,2048,false,true,true);
    return true;
}
NativeGroupedMmqWork NativeGroupedMmqProvider::submit(const NativeMmqPlan& plan,int type,int parts,int entries,int source_rows,
    const void* q8,void* controls,size_t controls_bytes,void* packed,size_t packed_bytes,float* output,
    int cols,int outputs,bool active,bool ordinary,bool direct) {
    auto& p=*impl_;
    const size_t matrices_bytes = plan.matrices.size() * sizeof(NativeMmqMatrix);
    const size_t map_bytes = size_t(entries) * sizeof(int32_t);
    const size_t tile_bytes = plan.tiles.size() * sizeof(NativeMmqTile);
    const size_t control_bytes = matrices_bytes + 2 * map_bytes + tile_bytes;
    const size_t required = q8_bytes(entries, cols);
    if ((type != 12 && type != 14 && !(type == 13 && ordinary)) || !controls || !packed || !q8 || !output || control_bytes > controls_bytes ||
        required > packed_bytes || cols < 512 || cols % 512 || outputs < 128 || outputs % 128)
        throw std::invalid_argument("unreserved bounded grouped MMQ workspace");
    bool gu_warp=p.gu_warp&&!direct&&ordinary&&
        native_gu_warp_repack_shape(type,cols,outputs,parts,entries,source_rows,active)&&
        native_gu_warp_repack_supported();
    if(gu_warp){
        std::array<uint64_t,256> seen{}; // <=16384 entries, no new allocation/control region.
        for(int compact=0;compact<entries;++compact){const int e=plan.output_ids[size_t(compact)];
            if(e<0||e>=entries||plan.input_ids[size_t(compact)]!=e/8||
               (seen[size_t(e)/64]&(uint64_t(1)<<(e%64)))){gu_warp=false;break;}
            seen[size_t(e)/64]|=uint64_t(1)<<(e%64);
        }
    } // Any shape/resource/P mismatch keeps the old map+packer before lease or write.
    auto& lease = p.leases[p.next++ % p.leases.size()];
    if (lease.used) {
        const auto ready = cudaEventQuery(lease.done);
        if (ready == cudaErrorNotReady) throw std::runtime_error("grouped MMQ lease requires caller drain");
        ck(ready);
    }
    std::memcpy(lease.bytes, plan.matrices.data(), matrices_bytes);
    if(gu_warp){auto* inverse=reinterpret_cast<int32_t*>(lease.bytes+matrices_bytes);
        for(int compact=0;compact<entries;++compact)inverse[plan.output_ids[size_t(compact)]]=compact;
    }else std::memcpy(lease.bytes + matrices_bytes, plan.input_ids.data(), map_bytes);
    std::memcpy(lease.bytes + matrices_bytes + map_bytes, plan.output_ids.data(), map_bytes);
    if (tile_bytes) std::memcpy(lease.bytes + matrices_bytes + 2 * map_bytes, plan.tiles.data(), tile_bytes);
    auto* device = static_cast<uint8_t*>(controls);
    try {
        ck(cudaMemcpyAsync(device, lease.bytes, control_bytes, cudaMemcpyHostToDevice, p.stream));
        if(ordinary) {
            if(direct) native_down_direct_packed(static_cast<const float*>(q8),packed,entries,
                reinterpret_cast<int32_t*>(device+matrices_bytes),p.stream);
            else if(gu_warp) native_gu_warp_repack(q8,packed,source_rows,
                reinterpret_cast<int32_t*>(device+matrices_bytes),p.stream);
            else ordinary_grouped_repack(q8,packed,type,cols,source_rows,entries,
                reinterpret_cast<int32_t*>(device+matrices_bytes),p.stream);
            ordinary_grouped_tile(type,reinterpret_cast<NativeMmqMatrix*>(device),int(plan.matrices.size()),parts,
                reinterpret_cast<int32_t*>(device+matrices_bytes+map_bytes),packed,output,
                entries,plan.max_rows,cols,outputs,p.stream,
                active?reinterpret_cast<NativeMmqTile*>(device+matrices_bytes+2*map_bytes):nullptr,int(plan.tiles.size()));
        } else {
            repack_native_q8_grouped(q8, packed, type, cols, source_rows, entries,
                reinterpret_cast<int32_t*>(device + matrices_bytes), p.stream);
            native_grouped_mmq(type, reinterpret_cast<NativeMmqMatrix*>(device), int(plan.matrices.size()), parts,
                reinterpret_cast<int32_t*>(device + matrices_bytes + map_bytes), packed, output,
                entries, plan.max_rows, cols, outputs, p.stream,
                active ? reinterpret_cast<NativeMmqTile*>(device + matrices_bytes + 2 * map_bytes) : nullptr, int(plan.tiles.size()));
        }
        ck(cudaEventRecord(lease.done, p.stream)); lease.used = true;
    } catch (...) {
        const auto e = cudaEventRecord(lease.done, p.stream); lease.used = e == cudaSuccess;
        if (e != cudaSuccess) cudaStreamSynchronize(p.stream);
        throw;
    }
    const int rectangular = ((plan.max_rows + 63) / 64) * int(plan.matrices.size());
    return {control_bytes, required, int(plan.matrices.size()), 3,
        active ? plan.active_count : rectangular, plan.active_count,
        plan.max_rows, rectangular, entries}; // clear padding, repack, tiled product; pure host costs
}
} // namespace strata::prefill::mmq
