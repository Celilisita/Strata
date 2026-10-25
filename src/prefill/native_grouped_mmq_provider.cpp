#include "native_grouped_mmq_state.hpp"
namespace strata::prefill::mmq {
bool native_q5_down_ordinary_eligible() {
    return q5_down_enabled() && ordinary_enabled() &&
        native_ordinary_grouped_supported(13,512,2048,1,false);
}
NativeGroupedMmqProvider::NativeGroupedMmqProvider(void* s) : impl_(std::make_unique<Impl>(s)) {}
NativeGroupedMmqProvider::~NativeGroupedMmqProvider() = default;
NativeGroupedMmqWork NativeGroupedMmqProvider::run(int type, const unsigned long long* gate, const unsigned long long* up,
    const int32_t* source_ids, int entries, int source_rows, const void* q8,
    void* controls, size_t controls_bytes, void* packed, size_t packed_bytes,
    float* output, int cols, int outputs, bool active) {
    const bool ordinary=ordinary_enabled() && (type!=13 || q5_down_enabled()) && entries>=64 && entries<=65536 &&
        native_ordinary_grouped_supported(type,cols,outputs,up?2:1,active);
    if (ordinary) {
        if(impl_->gu_warp && native_gu_warp_repack_shape(type,cols,outputs,up?2:1,entries,source_rows,active))
            (void)native_gu_warp_repack_supported();
        const auto plan=native_mmq_plan(gate,up,source_ids,entries,source_rows,active);
        return submit(plan,type,up?2:1,entries,source_rows,q8,controls,controls_bytes,packed,packed_bytes,output,cols,outputs,active,true,false);
    }
    return run_input(type,gate,up,source_ids,entries,source_rows,q8,controls,controls_bytes,
        packed,packed_bytes,output,cols,outputs,active,nullptr,nullptr);
}
NativeGroupedMmqWork NativeGroupedMmqProvider::run_swiglu_down(int type, const unsigned long long* weights,
    const int32_t* source_ids, int entries, int source_rows, const float* gate, const float* up,
    void* controls, size_t controls_bytes, void* packed, size_t packed_bytes,
    float* output, int cols, int outputs, bool active) {
    return run_input(type,weights,nullptr,source_ids,entries,source_rows,nullptr,controls,controls_bytes,
        packed,packed_bytes,output,cols,outputs,active,gate,up);
}
NativeGroupedMmqWork NativeGroupedMmqProvider::run_gu_int8(const unsigned long long* gate,
    const unsigned long long* up,const int32_t* ids,int entries,int source_rows,const void* q8,
    void* controls,size_t control_bytes,void* packed,size_t packed_bytes,float* output,bool active) {
    return run_input(12,gate,up,ids,entries,source_rows,q8,controls,control_bytes,
        packed,packed_bytes,output,2048,512,active,nullptr,nullptr,true);
}
NativeGroupedMmqWork NativeGroupedMmqProvider::run_gu_int8_i32(const unsigned long long* gate,
    const unsigned long long* up,const int32_t* ids,int entries,int source_rows,const void* q8,
    void* controls,size_t control_bytes,void* packed,size_t packed_bytes,float* output,bool active) {
    return run_input(12,gate,up,ids,entries,source_rows,q8,controls,control_bytes,
        packed,packed_bytes,output,2048,512,active,nullptr,nullptr,true,false,true);
}
NativeGroupedMmqWork NativeGroupedMmqProvider::run_gu_int8_i64(const unsigned long long* gate,
    const unsigned long long* up,const int32_t* ids,int entries,int source_rows,const void* q8,
    void* controls,size_t control_bytes,void* packed,size_t packed_bytes,float* output,bool active) {
    return run_input(12,gate,up,ids,entries,source_rows,q8,controls,control_bytes,
        packed,packed_bytes,output,2048,512,active,nullptr,nullptr,true,false,true,true);
}
NativeGroupedMmqWork NativeGroupedMmqProvider::run_down_int8(int type,const unsigned long long* weights,
    const int32_t* ids,int entries,const void* q8,void* controls,size_t control_bytes,
    void* packed,size_t packed_bytes,float* output,bool active) {
    return run_input(type,weights,nullptr,ids,entries,entries,q8,controls,control_bytes,
        packed,packed_bytes,output,512,2048,active,nullptr,nullptr,false,true);
}
} // namespace strata::prefill::mmq
